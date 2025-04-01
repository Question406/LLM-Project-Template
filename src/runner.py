# This file defines a runner class that runs evaluation in batches, and evaluates, and with a monitor class that supports experiment restoration.

import os
import re
import time
import json
import torch
import shutil
import logging
import inspect
import structlog
import concurrent
import numpy as np
from collections import Counter
from tqdm import tqdm
from typing import Dict, Any, Callable, List
from pathlib import Path
from datasets import Dataset
from contextlib import contextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import reduce

from src.eval_utils import pass_at_k
from src.config_store import RunConfig
from src.data_utils import batchify_sampler, DataDict
from src.api import SampleParams
from src.registry_utils import Registry

LOGGER = structlog.getLogger(__name__)


class SequentialRunner:
    def __init__(
        self,
        configs: RunConfig,
        rundir: str,
        dataset: Dataset,
        batch_size: int,
        model: AutoModelForCausalLM,  # We assume that the model has a function generate for generating texts
        tokenizer: AutoTokenizer,
        eval_fn: Callable,
        sampling_params: SampleParams,
        **kwargs,
    ):
        self.monitor = Monitor(rundir)
        self.batch_results = []
        self.metrics = {}
        self.configs = configs

        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.sampling_params = sampling_params
        self.eval_fn = eval_fn

        # For logging
        self.cur_batch_result = None

    def log(self):
        pass

    @torch.no_grad()
    def consume_batch(
        self, batch: DataDict, test_key: str = "reward_model"
    ) -> Dict[str, Any]:
        pass

    def run(self, resume=False, test_key="reward_model"):
        pass

    def stop(self):
        self.metrics = self.monitor.stop()
        LOGGER.info("Stopping the runner and saving the state")
        LOGGER.info("Final state", metrics=self.metrics, state=self.monitor.state)
        # with open("metrics.json", "w") as f:
        # json.dump(self.metrics, f, indent=2)
        self.monitor.dump_results()


class Monitor:
    def __init__(self, rundir: str, resume: bool = True):
        self.rundir = Path(rundir)
        if not self.rundir.exists():
            self.rundir.mkdir(parents=True)
        self.cache_dir = self.rundir / "cached_res"
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)

        self.consumed_samples = 0
        self.total_time = 0  # Track total execution time
        self.num_batches = 0  # Track number of batches processed
        self.all_results = []
        self.full_res = None

        if resume:
            self.restore()

    @property
    def state(self):
        return {
            "consumed_samples": self.consumed_samples,
            "average_time": self.average_time,
            "num_batches": self.num_batches,
            "total_time": self.total_time,
        }

    @property
    def average_time(self):
        """Returns the average batch processing time."""
        return self.total_time / self.num_batches if self.num_batches > 0 else 0

    @contextmanager
    def track_batch_time(self):
        """Context function to track batch execution time."""
        start_time = time.time()
        yield  # Execution of the wrapped function happens here
        end_time = time.time()

        batch_time = end_time - start_time
        self.total_time += batch_time
        self.num_batches += 1

    def update(self, batch_res: Dict[str, Any]):
        self.consumed_samples += batch_res["batch_num"]
        self.all_results.append(batch_res["batch_res"])
        if self.cache_dir.exists():
            with open(str(self.cache_dir / f"batch-{self.num_batches}.json"), "w") as f:
                if isinstance(batch_res["batch_res"], DataDict):
                    json.dump(batch_res["batch_res"].to_dict(), f, indent=2)
                else:
                    json.dump(batch_res, f, indent=2)

    def save(self):
        with open(str(self.rundir / "states.json"), "w") as f:
            json.dump(self.state, f, indent=2)

    def restore(self):
        state_path = self.rundir / "states.json"
        if state_path.exists():
            states = json.load(open(str(state_path)))

            for k, v in states.items():
                assert hasattr(self, k), (
                    f"State key {k} not found in the Monitor object"
                )
                try:
                    setattr(self, k, v)
                except Exception as e:
                    LOGGER.error(f"Error restoring state {k}: {e}")

            if self.cache_dir.exists():
                self.all_results = [
                    DataDict.from_dict(json.load(open(str(f))))
                    for f in self.cache_dir.iterdir()
                ]

            LOGGER.info(f"Resuming from previous state, {states['consumed_samples']}")

    def reset(self):
        """Reset internal counters and remove any saved state."""
        self.consumed_samples = 0
        self.total_time = 0
        self.num_batches = 0
        state_path = self.rundir / "states.json"
        if state_path.exists():
            state_path.unlink()

    def stop(self):
        # merge pass@k results
        metric = {}
        all_res = reduce(lambda x, y: x.union(y), self.all_results)
        if "score" in all_res and "uid" in all_res:
            uids = all_res["uid"]
            accs = all_res["score"]
            self.full_res = all_res
            for k in [1, 2, 4, 8]:
                passk = pass_at_k(uids, accs, k)
                metric[k] = passk
        return metric

    def dump_results(self):
        all_results = reduce(lambda x, y: x.union(y), self.all_results)
        # assert self.full_res is not None
        # all_results = self.full_res
        dataset = Dataset.from_dict(all_results.to_dict())
        dataset.save_to_disk(str(self.rundir / "finalresult"))
        shutil.rmtree(str(self.cache_dir))


RUNNER_REGISTRY = Registry("RunnerCLS")
RUNNER_REGISTRY.register("sequential", SequentialRunner)
