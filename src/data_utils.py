# This file defines GeneridDict, which is similar to TensorDict but supports non-tensor data, and each batch is a list of items.

import os
import numpy as np
from datasets import (
    Dataset,
    load_dataset,
    load_from_disk,
)
from typing import Dict, Union, List


def build_dataset(dataset_name):
    if os.path.exists(dataset_name):
        if os.path.isdir(dataset_name):
            rawdata = load_from_disk(dataset_name)["train"]
        else:
            if dataset_name.endswith(".json"):
                rawdata = load_dataset("json", data_files=[dataset_name])["train"]
            elif dataset_name.endswith(".parquet"):
                rawdata = load_dataset("parquet", data_files=[dataset_name])["train"]
            else:
                raise ValueError("Unknown dataset format")
    else:
        if "gsm8k" in dataset_name:
            rawdata = load_dataset(dataset_name, "main")["train"]
        else:
            rawdata = load_dataset(dataset_name)["train"]
    return rawdata


class DataDict:
    def __init__(self, data=None):
        """
        A simple implementation of a TensorDict-like container that supports non-tensor data.
        Instead of batching numerically, it stores values in lists.

        Args:
            data (dict, optional): Dictionary containing key-value pairs.
        """
        self.data = data if data is not None else {}

        if self.data:
            for key, value in self.data.items():
                if not isinstance(value, list):
                    self.data[key] = [value]
            lengths = [len(v) for v in self.data.values()]
            if lengths and not all(l == lengths[0] for l in lengths):
                raise ValueError(
                    "All list values in DataDict must have the same length"
                )
            self.length = lengths[0]

    @property
    def batch_size(self):
        """Returns the batch size by checking the length of the first list item."""
        return self.length

    def __getitem__(self, key):
        """If key is an int, return a single indexed item from all keys.
        If key is a string, return the corresponding list."""
        if isinstance(key, int):
            return DataDict.from_dict(
                {k: v[key] for k, v in self.data.items()}
            )  # Extract row-like structure
        if isinstance(key, slice):
            return DataDict({k: v[key] for k, v in self.data.items()})
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        """Deletes a key from the dictionary."""
        if key in self.data:
            del self.data[key]

    def __contains__(self, key):
        """Checks if a key exists."""
        return key in self.data

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def clone(self):
        """Creates a deep copy of the dictionary."""
        return DataDict(
            {k: v.copy() if isinstance(v, list) else v for k, v in self.data.items()}
        )

    def repeat(self, n):
        return DataDict(
            {k: v * n if isinstance(v, list) else v for k, v in self.data.items()}
        )

    def repeat_interleave(self, n):
        # This function repeats the dict like (1, 2, 3) -> (1, 1, 2, 2, 3, 3)
        return DataDict(
            {
                k: [x for x in v for _ in range(n)] if isinstance(v, list) else v
                for k, v in self.data.items()
            }
        )

    def to_list(self):
        """Converts each stored value into a list format, batching by grouping elements."""
        return [{k: v[i] for k, v in self.data.items()} for i in range(self.batch_size)]

    def select(self, keys):
        """Selects a subset of keys from the dictionary."""
        return DataDict({k: self.data[k] for k in keys if k in self.data})

    def update(self, other_dict):
        """Updates the dictionary with another DataDict or standard dict."""
        self.data.update(
            other_dict if isinstance(other_dict, dict) else other_dict.data
        )

    def pop(self, key):
        """Removes a key and returns its value."""
        return self.data.pop(key, None)

    @classmethod
    def from_dict(cls, data):
        """Creates a DataDict from a standard dictionary."""
        return cls(data)

    @classmethod
    def from_list_of_dicts(cls, list_of_dicts):
        """
        Converts a list of dictionaries into a DataDict.

        Example:
        input: [
            {"a": 1, "b": "x"},
            {"a": 2, "b": "y"},
            {"a": 3, "b": "z"}
        ]
        output: DataDict({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        Args:
            list_of_dicts (list): List of dictionaries where each dictionary represents a row.

        Returns:
            DataDict: The batched representation of the data.
        """
        if not list_of_dicts:
            return cls({})

        keys = list_of_dicts[0].keys()
        batched_data = {key: [d.get(key, None) for d in list_of_dicts] for key in keys}
        return cls(batched_data)

    def union_with_dict(self, other_dict):
        """
        Merges another dictionary into the existing DataDict.
        If a key exists in both, it extends the list values.
        """
        for k, v in other_dict.items():
            if k in self.data:
                if isinstance(self.data[k], list) and isinstance(v, list):
                    self.data[k].extend(v)  # Merge lists
                else:
                    self.data[k] = [self.data[k], v]  # Convert to list if necessary
            else:
                self.data[k] = v if isinstance(v, list) else [v]

        self.length = self.length + len(v) if isinstance(v, list) else self.length + 1

    def as_list_dict(self):
        """Ensures all values in the dictionary are lists."""
        return {k: v if isinstance(v, list) else [v] for k, v in self.data.items()}

    def __repr__(self):
        return f"DataDict(batch_size={self.batch_size}, data={list(self.data.keys())})"

    def __len__(self):
        return self.batch_size

    def to_dict(self):
        return self.data

    def groupby(self, keys) -> Dict[str, "DataDict"]:
        """
        Groups the DataDict based on one or multiple keys.

        Args:
            keys (str or list of str): The key(s) used for grouping.

        Returns:
            dict: A dictionary where keys are unique values (or tuples for multiple keys)
                  from the specified grouping keys, and values are DataDict instances
                  containing the grouped data.
        """
        if isinstance(keys, str):
            keys = [keys]  # Convert single key to list for uniform processing

        def clean_group_key(group_key):
            if len(group_key) == 1:
                return group_key[0]
            return group_key

        grouped_data = {}
        key_values_list = [self.data.get(k, []) for k in keys]

        if any(not values for values in key_values_list):  # If any key does not exist
            return {}

        for idx in range(self.batch_size):
            group_key = tuple(
                values[idx] for values in key_values_list
            )  # Create a unique tuple for multiple keys

            if group_key not in grouped_data:
                grouped_data[group_key] = {k: [] for k in self.data.keys()}

            for k, v in self.data.items():
                grouped_data[group_key][k].append(v[idx])

        return {
            clean_group_key(group_key): DataDict(group_values)
            for group_key, group_values in grouped_data.items()
        }


def batchify_sampler(total_num, batch_size, shuffle=False) -> List[List[int]]:
    if shuffle:
        random_index = np.random.permutation(total_num).tolist()
    else:
        random_index = list(range(total_num))

    all_indexes = []
    for i in range(0, total_num, batch_size):
        indexes = [random_index[j] for j in range(i, min(i + batch_size, total_num))]
        all_indexes.append(indexes)
    return all_indexes
