import os
import torch
import openai
import asyncio
import itertools
from logging import getLogger
from abc import ABC, abstractmethod
from dataclasses import asdict

import backoff
import copy
from typing import List, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams, RequestOutput
from openai import AzureOpenAI
from openai.types.completion_choice import CompletionChoice

try:
    import anthropic
    import anthropic.types.message as anthropic_message
except ImportError:
    pass

from .api import SampleParams
from .tokenizer_utils import TiktokenHuggingFaceTokenizer
from .config_store import model_configs, cs
from .registry_utils import Registry


LOGGER = getLogger(__name__)


def parse_api_output(response: openai.types.Completion) -> List[Dict[str, Any]]:
    parsed_responses = [
        {"text": response.message.content, "decode_id": response.index}
        for response in response.choices
    ]
    return parsed_responses


def parse_anthropic_output(
    responses,
) -> List[Dict[str, Any]]:
    parsed_responses = [
        {"text": response.content[0].text, "decode_id": rid}
        for response, rid in responses
    ]
    return parsed_responses


def parse_vllm_output(responses: List[RequestOutput]) -> List[Dict[str, Any]]:
    def extract_single(output: RequestOutput):
        res = [
            {
                "text": out.text,
                # "token": out.token_ids,
                "decode_id": out.index,
                "len": len(out.token_ids),
            }
            for out in output.outputs
        ]
        return res

    parsed_response = [
        extract_single(response) for response in responses if response is not None
    ]

    return parsed_response


class LanguageModel(ABC):
    API_URL = None
    api_key = None
    model = None
    sample_params = None
    tokenizer = None

    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def batch_generate(self):
        pass


class VLLMLanguageModel(LanguageModel):
    # This is a synchronize VLLM engine that runs on locally.
    def __init__(self, model, sample_params, tokenizer=None):
        self.model_name = model
        self.model = None  #! We delay the model initilziation
        self.sample_params = asdict(sample_params)
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        else:
            self.tokenizer = tokenizer

    def prepare_sampling_params(
        self,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        stop: str = None,
        n: int = None,
    ) -> SamplingParams:
        final_sample_params = copy.deepcopy(self.sample_params)
        final_sample_params.update(
            {
                k: v
                for k, v in locals().items()
                if v is not None and k in self.sample_params
            }
        )

        def convert_naming(dicts: Dict[str, Any]) -> Dict[str, Any]:
            name_mapping = {}
            result = {}

            for k, v in dicts.items():
                if k in name_mapping:
                    result[name_mapping[k]] = v
                else:
                    result[k] = v
            return result

        final_sample_params = SamplingParams(**convert_naming(final_sample_params))
        return final_sample_params

    def generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        stop: str = None,
        n: int = None,
    ):
        if self.model is None:
            ALL_GPU = os.environ.get("ALL_GPU", False)
            if ALL_GPU:
                gpus = torch.cuda.device_count()
            else:
                gpus = 1
            hostname = os.uname()[1]
            disable_custom_all_reduce = "bluevela" in hostname
            self.model = LLM(
                self.model_name,
                enable_prefix_caching=True,
                enable_chunked_prefill=True,
                swap_space=8,
                tensor_parallel_size=gpus,
                disable_custom_all_reduce=disable_custom_all_reduce,
            )

        sample_params = self.prepare_sampling_params(
            max_tokens,
            temperature,
            top_p,
            frequency_penalty,
            presence_penalty,
            stop,
            n,
        )
        outputs = self.model.generate(prompt, sample_params)
        outputs: List[RequestOutput]
        outputs = parse_vllm_output(outputs)
        return outputs

    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        stop: str = None,
        n: int = None,
    ):
        return self.generate(
            prompts,
            max_tokens,
            temperature,
            top_p,
            frequency_penalty,
            presence_penalty,
            stop,
            n,
        )


class APILanguageModel(LanguageModel):
    API_URL = "https://api.openai.com/v1/"

    def __init__(
        self,
        API_URL=None,
        api_key=None,
        model=None,
        sample_params=None,
        user_stop_token=None,
        tokenizer=None,
    ):
        self.API_URL = API_URL
        self.api_key = api_key
        self.model_name = model
        self.sample_params = asdict(sample_params) if sample_params is not None else {}
        self.tokenizer = tokenizer
        self.user_stop_token = user_stop_token

        self.client: openai.Client = openai.Client(
            api_key=self.api_key,
            base_url=self.API_URL,
        )

    def __repr__(self):
        return f"APILanguageModel({self.model_name}, port={self.API_URL})"

    @backoff.on_exception(backoff.expo, openai.OpenAIError, max_time=60, max_tries=3)
    def generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        stop: str = None,
        n: int = None,
    ):
        final_sample_params = copy.deepcopy(self.sample_params)
        final_sample_params.update(
            {
                k: v
                for k, v in locals().items()
                if v is not None and k in self.sample_params
            }
        )

        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                **final_sample_params,
            )
            return parse_api_output(response)
        except openai.OpenAIError as e:
            LOGGER.error(f"OpenAIError: {e}")
            raise
        except Exception as e:
            LOGGER.error(f"Unexpected error: {e}")
            raise

    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        stop: str = None,
        n: int = None,
        return_text: bool = True,
    ):
        if not isinstance(prompts[0], list):
            prompts = [prompts]
        results = [
            self.generate(
                prompt,
                max_tokens,
                temperature,
                top_p,
                frequency_penalty,
                presence_penalty,
                stop,
                n,
            )
            for prompt in prompts
        ]

        return results


class GPTLanguageModel(APILanguageModel):
    @backoff.on_exception(backoff.expo, openai.OpenAIError, max_time=10, max_tries=3)
    def generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        stop: str = None,
        n: int = None,
    ):
        # This function calls the API for one input prompt
        final_sample_params = copy.deepcopy(self.sample_params)
        final_sample_params.update(
            {
                k: v
                for k, v in locals().items()
                if v is not None and k in self.sample_params
            }
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                **final_sample_params,
            )
        except openai.OpenAIError as e:
            LOGGER.error(f"OpenAIError: {e}")
            raise
        except Exception as e:
            LOGGER.error(f"Unexpected error: {e}")
            raise

        outputs = parse_api_output(response)
        return outputs


class AzureLanguageModel(GPTLanguageModel):
    def __init__(
        self,
        API_URL=None,
        api_key=None,
        model=None,
        sample_params=None,
        user_stop_token=None,
        tokenizer=None,
    ):
        self.API_URL = API_URL
        self.api_key = api_key
        self.model_name = model
        self.sample_params = asdict(sample_params) if sample_params is not None else {}
        self.tokenizer = tokenizer
        self.user_stop_token = user_stop_token

        self.client: openai.Client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.API_URL,
            api_version="2024-08-01-preview",  # TODO (jiabao): we may put it in config
        )


def convert_param2hf(openai_param: Dict[str, Any]) -> Dict[str, Any]:
    mapping = {
        "max_tokens": "max_new_tokens",
        "top_p": "top_p",
        "stop": "stop_string",
        "n": "num_return_sequences",
    }
    return {mapping[k]: openai_param.get(k, None) for k in mapping.keys()}


class HFLanguageModel(LanguageModel):
    def __init__(self, model, sample_params, tokenizer=None):
        super().__init__()
        self.model_name = model
        self.model = AutoModelForCausalLM.from_pretrained(model).cuda().eval()
        self.sample_params = asdict(sample_params)

    def generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        stop: str = None,
        n: int = None,
    ):
        final_sample_params = copy.deepcopy(self.sample_params)
        final_sample_params.update(
            {
                k: v
                for k, v in locals().items()
                if v is not None and k in self.sample_params
            }
        )
        final_sample_params = convert_param2hf(final_sample_params)

        try:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            response = self.model.generate(
                input_ids=input_ids,
                **final_sample_params,
            )
            responses = [
                self.tokenizer.decode(response[i], skip_special_tokens=True).split(
                    prompt
                )[-1]
                for i in range(response.shape[0])
            ]
            # TODO: 01/15: should wrap it as a completionchoice object
            responses = [
                CompletionChoice(
                    finish_reason="stop",
                    index=i,
                    logprobs=None,
                    text=responses[i],
                )
                for i in range(response.shape[0])
            ]
        except openai.OpenAIError as e:
            LOGGER.error(f"OpenAIError: {e}")
            raise
        except Exception as e:
            LOGGER.error(f"Unexpected error: {e}")
            raise
        return responses


class AnthropicLanguageModel(APILanguageModel):
    def __init__(
        self,
        API_URL=None,
        api_key=None,
        model=None,
        sample_params=None,
        user_stop_token=None,
        tokenizer=None,
    ):
        self.API_URL = API_URL
        self.api_key = api_key
        self.model_name = model
        self.sample_params = asdict(sample_params) if sample_params is not None else {}
        self.tokenizer = tokenizer
        self.user_stop_token = user_stop_token
        self.client = anthropic.Anthropic(
            api_key=self.api_key,
        )

    def generate(
        self,
        prompt,
        max_tokens=None,
        temperature=None,
        top_p=None,
        frequency_penalty=None,
        presence_penalty=None,
        stop=None,
        n=None,
    ):
        final_sample_params = copy.deepcopy(self.sample_params)
        final_sample_params.update(
            {
                k: v
                for k, v in locals().items()
                if v is not None and k in self.sample_params
            }
        )
        final_sample_params = convert_param2hf(final_sample_params)

        assert final_sample_params.get("n", 1) == 1, "Anthropic only supports n=1"

        try:
            response = self.client.messages.create(
                model=self.model_name,
                messages=prompt,
                **final_sample_params,
            )
        except anthropic.APIError as e:
            LOGGER.error(f"AntrhopicError: {e}")
            raise
        except Exception as e:
            LOGGER.error(f"Unexpected error: {e}")
            raise

        outputs = parse_anthropic_output(response)
        return outputs


def build_language_model(
    # configs: RunConfig,
    model_name: str,
    raw_model_name: str,
    tokenizer=None,
    PORT: int = None,
    API_KEY: str = None,
    API_URL: str = None,
) -> LanguageModel:
    # This function creates a language model object based on the model name
    # xxx-vllm: the model is initialied as an APILanguageModel, but the API_URL is set to the local server
    # gpt-models: the model is initialized as an APIlanguageModel, but the API_URL is set to the OpenAI API
    # xxx-hf: the model is initailized as an HFLanguageModel
    if tokenizer is None:
        if "gpt" in raw_model_name:
            tokenizer = TiktokenHuggingFaceTokenizer("gpt-4o")
        elif "claude" in raw_model_name:
            # TODO (jiabao): need to implement
            tokenizer = TiktokenHuggingFaceTokenizer("gpt-4o")
        elif "@grok" in model_name:
            tokenizer = TiktokenHuggingFaceTokenizer("gpt-4")
        else:
            tokenizer = AutoTokenizer.from_pretrained(raw_model_name)
    if "@localvllm" in model_name:
        return VLLMLanguageModel(
            model=raw_model_name,
            sample_params=SampleParams(
                max_tokens=100,
                temperature=1.0,
                top_p=1.0,
            ),
            tokenizer=tokenizer,
        )
    elif "@vllm" in model_name:
        return APILanguageModel(
            API_URL=f"http://localhost:{PORT}/v1/",
            api_key="empty",
            model=model_name.split("@vllm")[0],
            sample_params=SampleParams(
                max_tokens=100,
                temperature=1.0,
                top_p=1.0,
            ),
            tokenizer=tokenizer,
        )
    elif "claude" in model_name:
        return AnthropicLanguageModel(
            API_URL="",
            api_key=API_KEY,
            model=model_name,
            sample_params=SampleParams(
                max_tokens=100,
                temperature=1.0,
                top_p=1.0,
            ),
            tokenizer=tokenizer,
        )
    elif "@azure" in model_name:
        return AzureLanguageModel(
            API_URL=API_URL,
            api_key=API_KEY,
            model=model_name.split("@azure")[0],
            sample_params=SampleParams(
                max_tokens=100,
                temperature=1.0,
                top_p=1.0,
            ),
            tokenizer=tokenizer,
        )
    elif "@grok" in model_name:
        return GPTLanguageModel(
            API_URL=API_URL,
            api_key=API_KEY,
            model=model_name.split("@grok")[0],
            sample_params=SampleParams(
                max_tokens=100,
                temperature=1.0,
                top_p=1.0,
            ),
            tokenizer=tokenizer,
        )
    elif "deepseek" in model_name:
        return GPTLanguageModel(
            API_URL=API_URL,
            api_key=API_KEY,
            model=model_name,
            sample_params=SampleParams(
                max_tokens=100,
                temperature=1.0,
                top_p=1.0,
            ),
            tokenizer=tokenizer,
        )
    elif "gpt" in model_name:
        return GPTLanguageModel(
            API_URL="https://api.openai.com/v1/",
            api_key=API_KEY,
            model=model_name,
            sample_params=SampleParams(
                max_tokens=100,
                temperature=1.0,
                top_p=1.0,
            ),
            tokenizer=tokenizer,
        )
    elif "@hf" in model_name:
        return HFLanguageModel(
            model=model_name.split("@hf")[0],
            sample_params=SampleParams(
                max_tokens=100,
                temperature=1.0,
                top_p=1.0,
            ),
            tokenizer=tokenizer,
        )


ModelRegistry = Registry()
for model_name in model_configs:
    model_config = cs.repo[model_name + ".yaml"].node
    model = build_language_model(**model_config)
    ModelRegistry.register(model_name, model)
