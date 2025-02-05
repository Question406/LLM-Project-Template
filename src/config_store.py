from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import List, Dict, Optional


@dataclass
class BuildParams:
    depth: int = 3
    width: int = 3
    action_list: List[str] = field(default_factory=list)
    action_prob: List[float] = field(default_factory=list)


@dataclass
class RunConfig:
    prompt_file: str
    dataset_name: str = "openai/gsm8k"
    llm_name: str = "llama"  # The name is set in the registry below
    build_params: BuildParams = BuildParams()
    user_stop_token: str = "<|eot_id|>"
    PORT: int = 10000
    action_name_mapping: Dict[str, str] = field(default_factory=dict)


cs = ConfigStore.instance()
cs.store(name="main_config", node=RunConfig)


@dataclass
class ModelRegistryParams:
    model_name: str
    raw_model_name: str
    PORT: int = None
    API_KEY: Optional[str] = field(default=None)


# List to keep track of all config objects
model_configs = []


# Function to register a model config and add it to the list
def register_model_registry(
    name: str, model_name: str, raw_model_name: str, port: int, API_KEY: str = None
):
    config = ModelRegistryParams(
        model_name=model_name, raw_model_name=raw_model_name, PORT=port, API_KEY=API_KEY
    )
    cs.store(name=name, node=config)
    model_configs.append(name)


# Register all model configurations
register_model_registry(
    name="llama_registry",
    model_name="Llama-3.2-3B-Instruct@vllm",
    raw_model_name="meta-llama/Llama-3.2-3B-Instruct",
    port=10000,
)

register_model_registry(
    name="mistral_registry",
    model_name="Mistral-7B-Instruct-v0.3@vllm",
    raw_model_name="mistralai/Mistral-7B-Instruct-v0.3",
    port=20000,
)

register_model_registry(
    name="deepseek_registry",
    model_name="deepseek-math-7b-instruct@vllm",
    raw_model_name="deepseek-ai/deepseek-math-7b-instruct",
    port=30000,
)

register_model_registry(
    name="qwen_registry",
    model_name="Qwen2.5-Math-7B-Instruct@vllm",
    raw_model_name="Qwen/Qwen2.5-Math-7B",
    port=40000,
)

register_model_registry(
    name="gpt4omini_registry",
    model_name="gpt-4o-mini",
    raw_model_name="gpt-4o-mini",
    port=-1,
    API_KEY="None",
)
