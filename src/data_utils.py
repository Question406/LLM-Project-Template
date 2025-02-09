# This file defines several tool classes and functions.


import os
import numpy as np


class DataDict:
    def __init__(self, data=None):
        """
        A simple implementation of a TensorDict-like container that supports non-tensor data.
        Instead of batching numerically, it stores values in lists.

        Args:
            data (dict, optional): Dictionary containing key-value pairs.
        """
        self.data = data if data is not None else {}

    @property
    def batch_size(self):
        """Returns the batch size by checking the length of the first list item."""
        return len(next(iter(self.data.values()), []))

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

    def as_list_dict(self):
        """Ensures all values in the dictionary are lists."""
        return {k: v if isinstance(v, list) else [v] for k, v in self.data.items()}

    def __repr__(self):
        return f"DataDict(batch_size={self.batch_size}, data={self.data})"

    def __len__(self):
        return self.batch_size

    def to_dict(self):
        return self.data
