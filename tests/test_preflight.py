"""Resume snapshot comparison (SPEC 2.6): the one preflight check that would
fail silently if wrong, by letting a changed config resume a checkpoint.

`compare_snapshot` is a pure function over parsed TOML documents, so no
model, run directory, or GPU is involved.
"""

import unittest

from trlx.preflight import compare_snapshot

# A run config as the operator wrote it, and the snapshot trlx made of it:
# run_name prepended, [launch] appended.
CURRENT = {
    "output_dir": "runs/a",
    "learning_rate": 1e-4,
    "resume_from_checkpoint": "runs/a/checkpoint-20",
    "model": {"path": "m", "dtype": "bfloat16"},
    "ranges": {"loss": [0, 5]},
}
SNAPSHOT = {
    "run_name": "a",
    "output_dir": "runs/a",
    "learning_rate": 1e-4,
    "model": {"path": "m", "dtype": "bfloat16"},
    "ranges": {"loss": [0, 5]},
    "launch": {"strategy": "ddp", "gpus": ["0", "1"]},
}


class CompareSnapshot(unittest.TestCase):
    def test_same_run_resumes(self):
        self.assertEqual(compare_snapshot(CURRENT, SNAPSHOT, "ddp"), [])

    def test_resume_key_and_launch_are_set_aside(self):
        # The snapshot was made without resume_from_checkpoint; the current
        # file must carry it. Neither that nor [launch].gpus is a difference.
        current = dict(CURRENT, resume_from_checkpoint="runs/a/checkpoint-40")
        snapshot = dict(SNAPSHOT, launch={"strategy": "ddp", "gpus": ["1"]})
        self.assertEqual(compare_snapshot(current, snapshot, "ddp"), [])

    def test_no_strategy_skips_sharding(self):
        # `trlx check` chooses no strategy and passes None.
        self.assertEqual(compare_snapshot(CURRENT, SNAPSHOT, None), [])

    def test_changed_top_level_value(self):
        current = dict(CURRENT, learning_rate=2e-4)
        diffs = compare_snapshot(current, SNAPSHOT, "ddp")
        self.assertEqual(len(diffs), 1)
        self.assertIn("learning_rate", diffs[0])

    def test_changed_nested_value(self):
        current = dict(CURRENT, model={"path": "m", "dtype": "float16"})
        diffs = compare_snapshot(current, SNAPSHOT, "ddp")
        self.assertEqual(len(diffs), 1)
        self.assertIn("model.dtype", diffs[0])

    def test_added_and_removed_keys(self):
        current = dict(CURRENT, warmup_steps=10)
        del current["learning_rate"]
        diffs = compare_snapshot(current, SNAPSHOT, "ddp")
        self.assertEqual(len(diffs), 2)
        self.assertTrue(any("warmup_steps" in d for d in diffs))
        self.assertTrue(any("learning_rate" in d for d in diffs))

    def test_sharding_change_is_a_difference(self):
        diffs = compare_snapshot(CURRENT, SNAPSHOT, "fsdp")
        self.assertEqual(len(diffs), 1)
        self.assertIn("strategy", diffs[0])

    def test_single_to_ddp_is_not_a_difference(self):
        # Both unsharded: a one-GPU run resumes on two.
        snapshot = dict(SNAPSHOT, launch={"strategy": "single", "gpus": ["0"]})
        self.assertEqual(compare_snapshot(CURRENT, snapshot, "ddp"), [])

    def test_operator_run_name_is_compared(self):
        # Only a prepended run_name is trlx's; one the operator wrote counts.
        current = dict(CURRENT, run_name="b")
        diffs = compare_snapshot(current, SNAPSHOT, "ddp")
        self.assertEqual(len(diffs), 1)
        self.assertIn("run_name", diffs[0])


if __name__ == "__main__":
    unittest.main()
