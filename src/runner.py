# This file defines a runner class that runs evaluation in batches, and evaluates, and with a monitor class that supports experiment restoration.

import os
import time
import json
import shutil
import logging
import inspect
import structlog
from tqdm import tqdm
from typing import Dict, Any, Callable
from pathlib import Path
from datasets import Dataset
from contextlib import contextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer
import concurrent
from functools import reduce

from earlyexit.eval_utils import pass_at_k
from earlyexit.config_store import RunConfig
from earlyexit.data_utils import batchify_sampler, DataDict
from earlyexit.api import SampleParams

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

    def consume_batch(self, batch: DataDict):
        # This function is the worker function that consumes a batch of data
        return {
            "batch_num": len(batch),
        }

    def run(self, resume=False):
        all_indexes = batchify_sampler(
            len(self.dataset), self.batch_size, shuffle=False
        )
        if not resume:
            # If not resuming, clear any previous state.
            self.monitor.reset()
            start_batch_index = 0
            LOGGER.info("Evaluation from start")
        else:
            # When resuming, skip over the batches already processed.
            start_batch_index = self.monitor.num_batches
            LOGGER.info("Resuming from batch", start_batch_index)

        for idx, indexes in enumerate(
            tqdm(
                all_indexes[start_batch_index:],
                desc="Running batch",
                initial=start_batch_index,
            )
        ):
            batch = DataDict.from_list_of_dicts([self.dataset[i] for i in indexes])
            with self.monitor.track_batch_time():
                batch_res = self.consume_batch(batch)
            self.monitor.update(batch_res)
            self.monitor.save()

    def stop(self):
        self.metrics = self.monitor.stop()
        LOGGER.info("Stopping the runner and saving the state")
        LOGGER.info("Final state", metrics=self.metrics, state=self.monitor.state)
        with open("metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)
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
        all_res = reduce(lambda x, y: x.union(y), self.all_results)
        uids = all_res["uid"]
        accs = all_res["score"]
        metric = {}
        for k in [1, 2, 4, 8]:
            passk = pass_at_k(uids, accs, k)
            metric[k] = passk

        return metric

    def dump_results(self):
        all_results = reduce(lambda x, y: x.union(y), self.all_results)
        dataset = Dataset.from_dict(all_results.to_dict())
        dataset.save_to_disk(str(self.rundir / "finalresult"))
        shutil.rmtree(str(self.cache_dir))
