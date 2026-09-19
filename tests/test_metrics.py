"""metrics.jsonl reader and callback, [ranges] cell evaluation, line renderer.

The reader targets the silent-failure case from PLAN Phase 4: a job still
writing leaves a partial last line, which must be skipped, while corruption
anywhere else must be an error naming the line.
"""

import json
import pathlib
import tempfile
import types
import unittest

from trlx import TrlxError, metrics, ranges, render_lines


# A complete record with the given step and log dict.
def rec(step, log, eval=False, max_steps=100):
    return {"step": step, "max_steps": max_steps, "epoch": step / 50, "num_train_epochs": 2.0,
            "eval": eval, "time": 0.0, "log": log}


class MetricsCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        self.path = self.dir / metrics.FILENAME

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, text):
        self.path.write_text(text)
        return self.path


class Reader(MetricsCase):
    def test_partial_last_line_skipped(self):
        full = json.dumps(rec(1, {"loss": 1.0}))
        partial = json.dumps(rec(2, {"loss": 0.9}))[:-5]
        records = metrics.read(self.write(full + "\n" + partial))
        self.assertEqual([r["step"] for r in records], [1])

    def test_complete_last_line_without_newline_kept(self):
        full = json.dumps(rec(1, {"loss": 1.0}))
        records = metrics.read(self.write(full + "\n" + json.dumps(rec(2, {"loss": 0.9}))))
        self.assertEqual([r["step"] for r in records], [1, 2])

    def test_corrupt_middle_line_is_error(self):
        text = json.dumps(rec(1, {"loss": 1.0})) + "\n{bad\n" + json.dumps(rec(3, {"loss": 0.8})) + "\n"
        with self.assertRaises(TrlxError) as ctx:
            metrics.read(self.write(text))
        self.assertIn("line 2", str(ctx.exception))

    def test_missing_key_is_error(self):
        r = rec(1, {"loss": 1.0})
        del r["max_steps"]
        with self.assertRaises(TrlxError) as ctx:
            metrics.read(self.write(json.dumps(r) + "\n"))
        self.assertIn("missing max_steps", str(ctx.exception))

    def test_missing_file_is_error(self):
        with self.assertRaises(TrlxError):
            metrics.read(self.dir / "nope.jsonl")

    def test_blank_lines_ignored(self):
        records = metrics.read(self.write("\n" + json.dumps(rec(1, {"loss": 1.0})) + "\n\n"))
        self.assertEqual(len(records), 1)


class Record(MetricsCase):
    def state(self, step=5, epoch=0.1):
        return types.SimpleNamespace(global_step=step, max_steps=100, epoch=epoch, num_train_epochs=2)

    def test_train_record(self):
        r = metrics.record(self.state(), {"loss": 1.5, "epoch": 0.1}, 123.0)
        self.assertEqual(r["step"], 5)
        self.assertEqual(r["max_steps"], 100)
        self.assertFalse(r["eval"])
        # The trainer's dict is stored verbatim, epoch key included.
        self.assertEqual(r["log"], {"loss": 1.5, "epoch": 0.1})

    def test_eval_record_flagged(self):
        r = metrics.record(self.state(), {"eval_loss": 1.2, "epoch": 0.1}, 0.0)
        self.assertTrue(r["eval"])

    def test_epoch_none_before_training(self):
        self.assertEqual(metrics.record(self.state(epoch=None), {"loss": 1.0}, 0.0)["epoch"], 0.0)

    def test_callback_writes_readable_lines(self):
        cb = metrics.callback_class()(self.dir)
        cb.on_log(None, self.state(step=1), None, logs={"loss": 2.0})
        cb.on_log(None, self.state(step=2), None, logs={"eval_loss": 1.0})
        cb.on_train_end(None, self.state(step=2), None)
        records = metrics.read(self.path)
        self.assertEqual([(r["step"], r["eval"]) for r in records], [(1, False), (2, True)])


class Evaluate(unittest.TestCase):
    RANGES = {"loss": (0.0, 2.0), "eval_loss": (0.0, 2.0)}

    def records(self):
        return [
            rec(1, {"loss": 1.0}),
            rec(2, {"eval_loss": 1.5}, eval=True),
            rec(3, {"loss": 2.5}),
            rec(4, {"eval_loss": 1.2}, eval=True),
        ]

    def test_change_skips_rows_lacking_the_metric(self):
        rows = ranges.evaluate(self.records(), self.RANGES)
        loss = [r.cells["loss"] for r in rows]
        self.assertEqual([c.value for c in loss], [1.0, None, 2.5, None])
        # Row 3's change is against row 1, the previous row that logged loss.
        self.assertEqual([c.change for c in loss], [None, None, 1.5, None])
        eval_loss = [r.cells["eval_loss"] for r in rows]
        self.assertAlmostEqual(eval_loss[3].change, -0.3)

    def test_out_of_range_marked(self):
        rows = ranges.evaluate(self.records(), self.RANGES)
        self.assertEqual([r.cells["loss"].out_of_range for r in rows], [False, False, True, False])
        self.assertTrue(rows[1].eval)

    def test_metric_not_in_ranges_is_not_a_column(self):
        rows = ranges.evaluate([rec(1, {"loss": 1.0, "grad_norm": 3.0})], {"loss": (0.0, 2.0)})
        self.assertEqual(list(rows[0].cells), ["loss"])

    def test_parse_rejects_bad_shapes(self):
        with self.assertRaises(TrlxError):
            ranges.parse("x.toml", {})
        with self.assertRaises(TrlxError):
            ranges.parse("x.toml", {"loss": [1, 0]})
        self.assertEqual(ranges.parse("x.toml", {"loss": [0, 5]}), {"loss": (0.0, 5.0)})


class Lines(unittest.TestCase):
    def test_marks_and_blanks(self):
        table = {"loss": (0.0, 2.0), "eval_loss": (0.0, 2.0)}
        rows = ranges.evaluate([rec(1, {"loss": 1.0}), rec(2, {"eval_loss": 3.0}, eval=True),
                                rec(3, {"loss": 0.5})], table)
        text = render_lines.render(rows, table)
        lines = text.split("\n")
        self.assertEqual(len(lines), 4)
        self.assertIn("loss", lines[0])
        self.assertIn("1/100", lines[1])
        self.assertIn("eval", lines[2])
        self.assertIn("3.000!", lines[2])
        self.assertIn("-0.500", lines[3])
        # A metric absent from a record renders blank, never "None".
        self.assertNotIn("None", text)


if __name__ == "__main__":
    unittest.main()
