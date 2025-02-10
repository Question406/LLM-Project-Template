# This file contains interface between p
from collections import defaultdict
from typing import Callable
import numpy as np
from transformers import AutoTokenizer
from typing import List
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from .data_utils import DataDict


def pass_at_k(uids: List[str], scores: List[float], k: int = 1) -> float:
    datadict = DataDict.from_dict(
        {
            "uid": uids,
            "score": scores,
        }
    )
    grouped = datadict.groupby("uid")
    result = defaultdict(list)
    for key, sample in grouped.items():
        sample_score = sample["score"]
        result[f"uid_{key}"] = np.mean(
            [max(sample_score[i : i + k]) for i in range(0, len(sample_score), k)]
        )
    return np.mean(list(result.values())).item()
