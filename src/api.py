from dataclasses import dataclass, field
from typing import List


@dataclass
class SampleParams:
    max_tokens: int = 100
    temperature: float = 0.5
    top_p: float = 1.0
    frequency_penalty: float = None
    presence_penalty: float = None
    stop: List[str] = field(default_factory=list)
    n: int = 1
