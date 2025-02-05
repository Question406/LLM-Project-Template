import openai
from abc import ABC, abstractmethod
from dataclasses import asdict

import backoff
import copy
from logging import getLogger
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai.types.completion_choice import CompletionChoice

from .api import SampleParams
from .tokenizer_utils import TiktokenHuggingFaceTokenizer
from .config_store import model_configs, cs
from .registry_utils import Registry

logger = getLogger(__name__)


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

        self.client: openai.Client = openai.Client(
            api_key=self.api_key,
            base_url=self.API_URL,
        )

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
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                **final_sample_params,
            )
        except openai.OpenAIError as e:
            logger.error(f"OpenAIError: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

        return parse_api_output(response)

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
        # TODO: 01/13 check this implementation, may be incorrect
        # This function calls the API multiple times for a list of input prompts
        final_sample_params = copy.deepcopy(self.sample_params)
        final_sample_params.update(
            {
                k: v
                for k, v in locals().items()
                if v is not None and k in self.sample_params
            }
        )

        try:
            responses = self.client.completions.create(
                model=self.model_name,
                prompt=prompts,
                **final_sample_params,
            )
        except openai.OpenAIError as e:
            logger.error(f"OpenAIError: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise


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
            logger.error(f"OpenAIError: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
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
            logger.error(f"OpenAIError: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
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
    if "@vllm" in model_name:
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
