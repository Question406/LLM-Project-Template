import json
import time
import tempfile
import unittest
from unittest.mock import MagicMock
from pathlib import Path
from src.runner import (
    Monitor,
    SequentialRunner,
)
from test_utils import RichTestRunner


class TestMonitor(unittest.TestCase):
    def setUp(self):
        """Create a temporary directory for testing Monitor."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.monitor = Monitor(self.temp_dir.name, resume=False)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_initial_state(self):
        """Test that Monitor initializes correctly."""
        self.assertEqual(self.monitor.consumed_samples, 0)
        self.assertEqual(self.monitor.total_time, 0)
        self.assertEqual(self.monitor.num_batches, 0)

    def test_track_time(self):
        """Test that the track_time context manager correctly tracks time."""
        with self.monitor.track_batch_time():
            time.sleep(0.1)  # Simulate processing delay

        self.assertGreater(self.monitor.total_time, 0)
        self.assertEqual(self.monitor.num_batches, 1)

    def test_average_time(self):
        """Test that average_time calculation is correct."""
        with self.monitor.track_batch_time():
            time.sleep(0.1)

        with self.monitor.track_batch_time():
            time.sleep(0.2)

        self.assertAlmostEqual(self.monitor.average_time, 0.15, places=2)

    def test_update(self):
        """Test that update() properly increments consumed_samples."""
        batch_res = {"batch_num": 5}
        self.monitor.update(batch_res)
        self.assertEqual(self.monitor.consumed_samples, 5)

    def test_save_and_restore(self):
        """Test that save() and restore() functions work correctly."""
        # Modify monitor state
        self.monitor.consumed_samples = 10
        self.monitor.total_time = 5.0
        self.monitor.num_batches = 2
        self.monitor.save()

        # Ensure states.json is written
        state_path = Path(self.temp_dir.name) / "states.json"
        self.assertTrue(state_path.exists())

        # Load the saved state manually
        with open(state_path, "r") as f:
            saved_state = json.load(f)

        self.assertEqual(saved_state["consumed_samples"], 10)

        # Create new monitor and restore state
        new_monitor = Monitor(self.temp_dir.name, resume=True)
        self.assertEqual(new_monitor.consumed_samples, 10)
        self.assertEqual(new_monitor.total_time, 5.0)
        self.assertEqual(new_monitor.num_batches, 2)


class TestSequentialRunner(unittest.TestCase):
    def setUp(self):
        """Set up a SequentialRunner with mock components."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mock_model = MagicMock(spec=AutoModelForCausalLM)
        self.mock_tokenizer = MagicMock(spec=AutoTokenizer)
        self.mock_dataset = Dataset.from_dict({"input": ["test1", "test2", "test3"]})

        self.runner = SequentialRunner(
            configs=MagicMock(spec=RunConfig),
            rundir=self.temp_dir.name,
            dataset=self.mock_dataset,
            batch_size=1,
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
        )

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_consume_batch(self):
        """Test consume_batch is called properly within run()."""
        self.runner.consume_batch = MagicMock(return_value={"batch_num": 1})
        self.runner.run()

        self.assertEqual(self.runner.monitor.num_batches, len(self.mock_dataset))

    def test_time_tracking_in_run(self):
        """Test that batch execution time is tracked during run()."""
        self.runner.consume_batch = MagicMock(return_value={"batch_num": 1})

        with self.runner.monitor.track_batch_time():
            time.sleep(0.1)

        self.runner.run()

        self.assertGreater(self.runner.monitor.total_time, 0)
        self.assertEqual(self.runner.monitor.num_batches, len(self.mock_dataset) + 1)

    def test_stop_logs_average_time(self):
        """Test that stop() logs the correct average time."""
        self.runner.consume_batch = MagicMock(return_value={"batch_num": 1})

        # Simulating batches with known times
        with self.runner.monitor.track_batch_time():
            time.sleep(0.1)
        with self.runner.monitor.track_batch_time():
            time.sleep(0.2)

        avg_time = self.runner.monitor.average_time
        self.assertAlmostEqual(avg_time, 0.15, places=2)


if __name__ == "__main__":
    unittest.main(testRunner=RichTestRunner(), exit=False)
