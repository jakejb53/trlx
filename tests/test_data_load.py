"""Fractional holdout uses actual rows and never changes their order."""

import random
import unittest
from unittest.mock import patch

from datasets import Dataset
from dataset.progress import Progress
from trlx import TrlxError, config, data_load, feedback


class FractionalSplit(unittest.TestCase):
    # Dataset conversion failures retain the operator's source path and a concrete remedy.
    def test_incompatible_column_values_name_dataset(self):
        ref = config.DatasetRef("mixed.jsonl", True, None)
        with patch("trlx.data_load.read_rows", return_value=[{"text": "hello"}, {"text": [1]}]):
            with self.assertRaisesRegex(TrlxError, "mixed.jsonl.*consistent value types"):
                data_load.load_ref(ref)

    # In-memory datasets exercise the same slicing path as local and Hub sources.
    def split(self, count, fraction, *, shuffle=False, seed=None):
        whole = Dataset.from_dict({"text": [str(index) for index in range(count)]})
        spec = config.DatasetSpec(True, config.DatasetRef("memory.jsonl", True, None), fraction, None,
                                  shuffle_eval_data=shuffle)
        with patch.object(data_load, "load_ref", return_value=whole):
            return data_load.load(spec, "language modeling or prompt-completion", seed=seed)

    # Random membership keeps exact rounding, source order, and disjoint full coverage.
    def test_random_split_preserves_counts_order_and_coverage(self):
        before = random.getstate()
        for count, fraction, expected in ((100, .07, 7), (23, .1, 3), (2, .1, 1)):
            with self.subTest(count=count):
                train, evaluation = self.split(count, fraction, shuffle=True, seed=42)
                left, right = list(map(int, train["text"])), list(map(int, evaluation["text"]))
                self.assertEqual(len(right), expected)
                self.assertEqual(left, sorted(left))
                self.assertEqual(right, sorted(right))
                self.assertFalse(set(left) & set(right))
                self.assertEqual(sorted(left + right), list(range(count)))
        self.assertEqual(random.getstate(), before)

    # The same source and seed reproduce membership independently of process RNG state.
    def test_random_split_repeats_by_seed_and_samples_beyond_tail(self):
        first_pair = self.split(100, .2, shuffle=True, seed=0)
        repeated_pair = self.split(100, .2, shuffle=True, seed=0)
        self.assertEqual([part._fingerprint for part in first_pair],
                         [part._fingerprint for part in repeated_pair])
        first = list(first_pair[1]["text"])
        repeated = list(repeated_pair[1]["text"])
        changed = list(self.split(100, .2, shuffle=True, seed=1)[1]["text"])
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, changed)
        self.assertNotEqual(first, [str(index) for index in range(80, 100)])

    # Random selection must never silently become unseeded or tolerate an empty side.
    def test_random_split_rejects_missing_seed_and_empty_partition(self):
        with self.assertRaisesRegex(TrlxError, "integer split seed"):
            self.split(10, .1, shuffle=True)
        with self.assertRaisesRegex(TrlxError, "both must be nonempty"):
            self.split(1, .1, shuffle=True, seed=0)

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


class UsefulFeedback(unittest.TestCase):
    # Exercise the typed event and supervisor policy together with in-memory datasets.
    def setUp(self):
        self.events = []
        self.progress = Progress("trlx sft", events=self.events.append)

    # A useful source note must be explicitly public and survive terminal presentation.
    def assert_visible_note(self, text):
        matching = [event for event in self.events if event["kind"] == "note" and text in event["message"]]
        self.assertEqual(len(matching), 1)
        self.assertTrue(matching[0]["visible"])
        lines = []
        view = feedback.View(lines.append)
        for event in self.events:
            view.consume(dict(event, source="rank 0"))
        self.assertTrue(any(text in line for line in lines))

    # Hub preparation must report its measured result even when the stage has no row total.
    def test_hub_loaded_row_count_stays_inline(self):
        ref = config.DatasetRef("owner/corpus", False, "train")
        whole = Dataset.from_dict({"text": ["a", "b", "c"]})
        with patch.object(data_load, "_load_hub", return_value=whole):
            self.assertIs(data_load.load_ref(ref, progress=self.progress), whole)
        self.assert_visible_note("loaded 3 rows")

    # Fractional splitting publishes actual train/evaluation sizes, not the configured fraction.
    def test_split_row_counts_stay_inline(self):
        whole = Dataset.from_dict({"text": [str(index) for index in range(10)]})
        spec = config.DatasetSpec(True, config.DatasetRef("memory.jsonl", True, None), 0.2, None)
        with patch.object(data_load, "load_ref", return_value=whole):
            train, evaluation = data_load.load(spec, "language modeling or prompt-completion", progress=self.progress)
        self.assertEqual((train.num_rows, evaluation.num_rows), (8, 2))
        self.assert_visible_note("8 training rows; 2 evaluation rows")

    # Replay changes the effective train size, so its measured composition remains inline.
    def test_replay_composition_stays_inline(self):
        train = Dataset.from_dict({"text": ["a", "b", "c", "d"]})
        replay = Dataset.from_dict({"text": ["e", "f"]})
        spec = config.ReplaySpec(config.DatasetRef("replay.jsonl", True, None), 0.2, 0.0)
        with patch.object(data_load, "load_ref", return_value=replay):
            mixed = data_load.mix_replay(train, spec, False, progress=self.progress)
        self.assertEqual(mixed.num_rows, 5)
        self.assert_visible_note("4 training rows + 1 replay rows = 5 rows")


if __name__ == "__main__":
    unittest.main()
