# This file defines a runner class that runs evaluation in batches, and evaluates, and with a monitor class that supports experiment restoration.

import os
import time
import json
import logging
import inspect
from tqdm import tqdm
from typing import Dict, Any
from pathlib import Path
from datasets import Dataset
from contextlib import contextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config_store import RunConfig
from src.data_utils import batchify_sampler, DataDict

LOGGER = logging.getLogger(__name__)


class SequentialRunner:
    def __init__(
        self,
        configs: RunConfig,
        rundir: str,
        dataset: Dataset,
        batch_size: int,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
    ):
        self.monitor = Monitor(rundir)
        self.batch_results = {}
        self.metrics = {}

        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

        # For logging
        self.cur_batch_result = None

    def log(self):
        pass

    def consume_batch(self, batch):
        # This function is the worker function that consumes a batch of data

        return {
            "batch_num": len(batch),
        }

    def run(self):
        all_indexes = batchify_sampler(
            len(self.dataset), self.batch_size, shuffle=False
        )

        for idx, indexes in enumerate(tqdm(all_indexes, desc="Running batch")):
            batch = DataDict.from_list_of_dicts([self.dataset[i] for i in indexes])
            with self.monitor.track_batch_time():
                batch_res = self.consume_batch(batch)
            self.monitor.update(batch_res)

    def stop(self):
        LOGGER.info("Stopping the runner and saving the state")
        LOGGER.info("Final state", metrics=self.metrics, state=self.monitor.state)
        pass


class Monitor:
    def __init__(self, rundir: str, resume: bool = True):
        self.rundir = Path(rundir)
        if not self.rundir.exists():
            self.rundir.mkdir(parents=True)
        self.consumed_samples = 0
        self.total_time = 0  # Track total execution time
        self.num_batches = 0  # Track number of batches processed

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

            LOGGER.info(f"Resuming from previous state, {states['consumed_samples']}")
