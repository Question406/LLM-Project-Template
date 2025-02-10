import openai
import itertools
from abc import ABC, abstractmethod
from dataclasses import asdict

import backoff
import copy
import asyncio
from logging import getLogger
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai.types.completion_choice import CompletionChoice
from vllm import LLM, SamplingParams, RequestOutput

from .api import SampleParams
from .tokenizer_utils import TiktokenHuggingFaceTokenizer
from .config_store import model_configs, cs
from .registry_utils import Registry


LOGGER = getLogger(__name__)


def parse_api_output(response: openai.types.Completion):
    return [choice for choice in response.choices]


class LanguageModel(ABC):
    API_URL = None
    api_key = None
    model = None
    sample_params = None
    tokenizer = None

    @abstractmethod
    def generate(self):
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
            self.model = LLM(self.model_name)
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
        output_texts = [[x.text for x in output.outputs] for output in outputs]
        output_texts = list(itertools.chain.from_iterable(output_texts))
        return output_texts

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
        API_URL,
        api_key,
        model,
        sample_params,
        user_stop_token=None,
        tokenizer=None,
    ):
        self.API_URL = API_URL
        self.api_key = api_key
        self.model_name = model
        self.sample_params = asdict(sample_params)
        self.tokenizer = tokenizer
        self.user_stop_token = user_stop_token

        self.client: openai.AsyncClient = openai.AsyncClient(
            api_key=self.api_key,
            base_url=self.API_URL,
        )

    def __repr__(self):
        return f"APILanguageModel({self.model_name}, port={self.API_URL})"

    @backoff.on_exception(backoff.expo, openai.OpenAIError, max_time=60, max_tries=3)
    async def generate(
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
            response = await self.client.completions.create(
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

    async def batch_generate(
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
        tasks = [
            asyncio.create_task(
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
            )
            for prompt in prompts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions gracefully
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                LOGGER.error(f"Error during batch generation: {result}")
                final_results.append(None)  # Indicate a failure
            else:
                final_results.append(result)

        if not return_text:
            return final_results
        else:
            res = [[choice.text for choice in res] for res in final_results]
            res = list(itertools.chain.from_iterable(res))
            return res


# TODO (jiabao): this has not been tested
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

        return response.choices


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


def build_language_model(
    # configs: RunConfig,
    model_name: str,
    raw_model_name: str,
    tokenizer=None,
    PORT: int = None,
    API_KEY: str = None,
) -> LanguageModel:
    # This function creates a language model object based on the model name
    # xxx-vllm: the model is initialied as an APILanguageModel, but the API_URL is set to the local server
    # gpt-models: the model is initialized as an APIlanguageModel, but the API_URL is set to the OpenAI API
    # xxx-hf: the model is initailized as an HFLanguageModel
    if tokenizer is None:
        if "gpt" in raw_model_name:
            tokenizer = TiktokenHuggingFaceTokenizer("gpt-4o")
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
