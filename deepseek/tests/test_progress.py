"""Live table rows, validation lines, and the progress file."""

import json
import os
import tempfile
import unittest

from deepseek.qat.progress import COLUMNS, Progress, StepTable


def _flips(dense_changed, expert_changed, layers_changed):
    f = {"dense_values_changed": dense_changed, "dense_value_elements": 5_681_184_768, "dense_class_elements": 5_681_184_768,
         "expert_values_changed": expert_changed, "expert_value_elements": 83_751_862_272, "expert_class_elements": 83_751_862_272}
    for layer in range(43):
        f[f"expert_values_changed_layer{layer}"] = 7 if layer in layers_changed else 0
        f[f"expert_value_elements_layer{layer}"] = 1
    return f


class Table(unittest.TestCase):
    def test_header_then_rows_with_epoch_mean_and_held_out_deltas(self):
        t = StepTable(total_steps=100, steps_per_epoch=17, expert_layers=43)
        t.note_validation(0.5592, baseline=True)
        out = t.row(0, 0.7606, _flips(0, 0, set()), {"dense": 2.6, "expert": 0.9}, 56.5)
        header, row = out.split("\n")
        self.assertEqual(header, t.header())
        self.assertEqual(len(header), sum(w for _, w in COLUMNS) + 2 * (len(COLUMNS) - 1))
        cells = row.split()
        self.assertEqual(cells[:4], ["0/100", "0.06", "0.7606", "0.7606"])
        # No Δ before a second row, no held-out before a validation, flips at zero, norms, time.
        self.assertEqual(cells[4:], ["0.0000", "0.000000", "0/43", "2.600", "0.900", "56.5"])  # drift blank without a sample
        t.note_validation(0.5535)
        row2 = t.row(1, 0.9633, _flips(36_784_598, 1_004_912, {8, 18}), {"dense": 1.5, "expert": 0.7}, 29.0)
        self.assertNotIn("\n", row2)
        cells = row2.split()
        self.assertEqual(cells[:5], ["1/100", "0.12", "0.9633", "0.8619", "+0.1013"])
        # held-out, then Δbase only (no previous validation yet, so Δprev is blank).
        self.assertEqual(cells[5:7], ["0.5535", "-0.0057"])
        self.assertEqual(cells[7:], ["0.6475", "0.001200", "2/43", "1.500", "0.700", "29.0"])
        drift = {"dense": (3.1e-4, 0.42), "expert": (1.2e-3, 0.87)}
        cells = t.row(2, 0.6, _flips(0, 0, set()), {"dense": 1.0, "expert": 0.5}, 28.0, drift).split()
        self.assertEqual(cells[7:14], ["0.0000", "0.000000", "0/43", "3.1e-04", "0.42", "1.2e-03", "0.87"])
        t.note_validation(0.4951)
        row3 = t.row(3, 0.5, _flips(0, 0, set()), None, None)
        cells = row3.split()
        self.assertEqual(cells[5:8], ["0.4951", "-0.0584", "-0.0641"])
        self.assertEqual(cells[8:11], ["0.0000", "0.000000", "0/43"])
        self.assertEqual(len(cells), 11)  # norms and time blank

    def test_epoch_mean_resets_at_epoch_boundary_and_header_repeats(self):
        t = StepTable(total_steps=40, steps_per_epoch=2, expert_layers=0)
        t.row(0, 1.0, None, None, None)
        t.row(1, 3.0, None, None, None)
        cells = t.row(2, 5.0, None, None, None).split()
        self.assertEqual(cells[3], "5.0000")  # new epoch: mean restarts
        self.assertEqual(cells[4], "+3.0000")  # against the previous row's mean 2.0
        self.assertEqual(cells[1], "1.50")
        headers = sum(1 for i in range(3, 45) if "\n" in t.row(i, 1.0, None, None, None))
        self.assertEqual(headers, 2)

    def test_validation_line(self):
        t = StepTable(100, 17, 43)
        t.note_validation(0.5592, baseline=True)
        t.note_validation(0.5535)
        self.assertEqual(t.validation_line(4), "eval step 4: held-out 0.5535 (Δbase -0.0057)")
        t.note_validation(0.4951)
        self.assertEqual(t.validation_line(9), "eval step 9: held-out 0.4951 (Δprev -0.0584, Δbase -0.0641)")


class ProgressFile(unittest.TestCase):
    def test_one_json_record_per_line_on_rank_zero_only(self):
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as root:
            path = os.path.join(root, "p-progress.jsonl")
            Progress(path, rank=1).record("step", {"step": 0})
            self.assertFalse(os.path.exists(path))
            pr = Progress(path, rank=0)
            pr.record("step", {"step": 0, "loss": 0.5})
            pr.record("validation", {"step": 9, "val_loss": 0.4})
            with open(path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            self.assertEqual(rows, [{"kind": "step", "step": 0, "loss": 0.5}, {"kind": "validation", "step": 9, "val_loss": 0.4}])
            Progress(path, rank=0)  # a new run truncates
            self.assertEqual(os.path.getsize(path), 0)
            Progress(None, rank=0).record("step", {})  # no path: nothing written, no error


if __name__ == "__main__":
    unittest.main()
