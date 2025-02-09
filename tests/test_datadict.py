import unittest

from src.data_utils import DataDict


class TestDataDict(unittest.TestCase):
    def setUp(self):
        """Set up test data for DataDict"""
        self.data = DataDict.from_list_of_dicts(
            [
                {"category": "A", "type": "fruit", "value": 10, "name": "apple"},
                {"category": "B", "type": "fruit", "value": 20, "name": "banana"},
                {"category": "A", "type": "vegetable", "value": 30, "name": "avocado"},
                {"category": "B", "type": "fruit", "value": 40, "name": "blueberry"},
                {"category": "C", "type": "fruit", "value": 50, "name": "cherry"},
                {"category": "A", "type": "fruit", "value": 60, "name": "apricot"},
            ]
        )

    def test_groupby_single_key(self):
        """Test grouping by a single key"""
        grouped = self.data.groupby("category")

        self.assertIn("A", grouped)
        self.assertIn("B", grouped)
        self.assertIn("C", grouped)

        self.assertEqual(
            grouped["A"].to_dict(),
            {
                "category": ["A", "A", "A"],
                "type": ["fruit", "vegetable", "fruit"],
                "value": [10, 30, 60],
                "name": ["apple", "avocado", "apricot"],
            },
        )

    def test_groupby_multiple_keys(self):
        """Test grouping by multiple keys"""
        grouped = self.data.groupby(["category", "type"])

        self.assertIn(("A", "fruit"), grouped)
        self.assertIn(("A", "vegetable"), grouped)
        self.assertIn(("B", "fruit"), grouped)
        self.assertIn(("C", "fruit"), grouped)

        self.assertEqual(
            grouped[("A", "fruit")].to_dict(),
            {
                "category": ["A", "A"],
                "type": ["fruit", "fruit"],
                "value": [10, 60],
                "name": ["apple", "apricot"],
            },
        )
        self.assertEqual(
            grouped[("B", "fruit")].to_dict(),
            {
                "category": ["B", "B"],
                "type": ["fruit", "fruit"],
                "value": [20, 40],
                "name": ["banana", "blueberry"],
            },
        )

    def test_groupby_invalid_key(self):
        """Test grouping by a non-existent key"""
        grouped = self.data.groupby("non_existent_key")
        self.assertEqual(grouped, {})  # Should return an empty dictionary

    def test_groupby_partial_missing_key(self):
        """Test grouping by a mix of valid and invalid keys"""
        grouped = self.data.groupby(["category", "non_existent_key"])
        self.assertEqual(grouped, {})  # Should return an empty dictionary


if __name__ == "__main__":
    unittest.main()
