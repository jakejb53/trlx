"""Fractional holdout uses actual rows and never changes their order."""

import unittest
from unittest.mock import patch

from datasets import Dataset
from trlx import TrlxError, config, data_load


class FractionalSplit(unittest.TestCase):
    # In-memory datasets exercise the same slicing path as local and Hub sources.
    def split(self, count, fraction):
        whole = Dataset.from_dict({"text": [str(index) for index in range(count)]})
        spec = config.DatasetSpec(True, config.DatasetRef("memory.jsonl", True, None), fraction, None)
        with patch.object(data_load, "load_ref", return_value=whole):
            return data_load.load(spec, "language modeling or prompt-completion")

    # The same config must adapt when the input grows between runs.
    def test_fraction_tracks_current_size(self):
        for count, fraction, expected_eval in ((100, 0.1, 10), (23, 0.1, 3), (2, 0.1, 1), (100, 0.07, 7)):
            with self.subTest(count=count, fraction=fraction):
                train, evaluation = self.split(count, fraction)
                self.assertEqual(evaluation.num_rows, expected_eval)
                self.assertEqual(train.num_rows + evaluation.num_rows, count)
                self.assertEqual(list(train["text"]) + list(evaluation["text"]), [str(i) for i in range(count)])

    # The fixture migration must retain exactly the existing 60/20 partition.
    def test_quarter_preserves_fixture_partition(self):
        train, evaluation = self.split(80, 0.25)
        self.assertEqual((train.num_rows, evaluation.num_rows), (60, 20))

    # Neither empty side is silently tolerated or fixed by altering the fraction.
    def test_empty_side_is_reported(self):
        for count, fraction in ((0, 0.1), (1, 0.1), (2, 0.9)):
            with self.subTest(count=count, fraction=fraction), self.assertRaisesRegex(TrlxError, "both must be nonempty"):
                self.split(count, fraction)


if __name__ == "__main__":
    unittest.main()
