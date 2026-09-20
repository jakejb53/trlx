"""One case per repair class, one ambiguous error left alone, and the file
round trip through heal_file. Fixtures live under tests/ per session rules."""

import json
import pathlib
import tempfile
import unittest

from dataset.heal import heal_file, heal_json, heal_jsonl
from dataset.io import DatasetError

ROOT = pathlib.Path(__file__).resolve().parent


class HealJsonl(unittest.TestCase):
    # Helper: heal one line, assert it parses to `expected` and the repair
    # description contains `name`.
    def _one(self, line, expected, name):
        out, repairs, errors = heal_jsonl(line + "\n")
        self.assertEqual(errors, [])
        self.assertEqual([json.loads(l) for l in out.splitlines()], expected)
        self.assertTrue(any(name in r for r in repairs), repairs)
        return repairs

    def test_valid_untouched(self):
        out, repairs, errors = heal_jsonl('{"a": 1}\n')
        self.assertEqual((out, repairs, errors), ('{"a": 1}\n', [], []))

    def test_trailing_comma(self):
        self._one('{"a": 1,}', [{"a": 1}], "trailing commas")
        self._one('{"a": [1, 2,]}', [{"a": [1, 2]}], "trailing commas")

    def test_unclosed(self):
        self._one('{"a": [1, 2', [{"a": [1, 2]}], "unclosed")
        # Last value is a string: the closing quote must not read as unterminated.
        self._one('{"a": "b"', [{"a": "b"}], "unclosed")
        # Trailing comma then cut off: both repairs compose.
        self._one('{"a": 1,', [{"a": 1}], "unclosed")

    def test_single_quotes(self):
        self._one("{'a': 'it\\'s \"x\"'}", [{"a": 'it\'s "x"'}], "single quotes")

    def test_unquoted_keys(self):
        self._one('{a: 1, b_2: {c: "x"}}', [{"a": 1, "b_2": {"c": "x"}}], "unquoted keys")

    def test_python_literals(self):
        self._one('{"a": True, "b": None, "c": "True"}', [{"a": True, "b": None, "c": "True"}], "Python literals")

    def test_literal_suffix_of_identifier_untouched(self):
        self._one('{isNone: 1, allowFalse: True}', [{"isNone": 1, "allowFalse": True}], "unquoted keys")

    def test_concatenated(self):
        self._one('{"a": 1}{"b": 2} {"c": 3}', [{"a": 1}, {"b": 2}, {"c": 3}], "concatenated")

    def test_truncated_last_line_dropped(self):
        out, repairs, errors = heal_jsonl('{"a": 1}\n{"b": "hel')
        self.assertEqual(out, '{"a": 1}\n')
        self.assertEqual(errors, [])
        self.assertIn("truncated", repairs[0])

    def test_truncated_middle_line_is_error(self):
        out, repairs, errors = heal_jsonl('{"a": "hel\n{"b": 2}\n')
        self.assertEqual(out, '{"a": "hel\n{"b": 2}\n')
        self.assertEqual(repairs, [])
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0].startswith("line 1 column"), errors)

    def test_ambiguous_left_alone(self):
        out, repairs, errors = heal_jsonl('{"a": }\n{"b": 2}\n')
        self.assertEqual(out, '{"a": }\n{"b": 2}\n')
        self.assertEqual(repairs, [])
        self.assertEqual(len(errors), 1)

    def test_strings_not_touched(self):
        # Content inside strings must survive every pass untouched.
        line = '{"a": "x, }", "b": "True", "c": "k: v", "d": "don\'t"}'
        out, repairs, errors = heal_jsonl(line + "\n")
        self.assertEqual(out, line + "\n")
        self.assertEqual(repairs, [])


class HealJson(unittest.TestCase):
    def test_repairs(self):
        out, repairs, errors = heal_json("[{a: 1,}, {'b': False}")
        self.assertEqual(json.loads(out), [{"a": 1}, {"b": False}])
        self.assertEqual(errors, [])
        # single quotes, Python literal, unquoted key, trailing comma, unclosed ]
        self.assertEqual(len(repairs), 5, repairs)

    def test_concatenated_wrapped(self):
        out, repairs, errors = heal_json('{"a": 1}\n{"b": 2}\n')
        self.assertEqual(json.loads(out), [{"a": 1}, {"b": 2}])
        self.assertTrue(any("array" in r for r in repairs))

    def test_error_reported(self):
        out, repairs, errors = heal_json('{"a": }')
        self.assertEqual(out, '{"a": }')
        self.assertEqual(len(errors), 1)


class HealFile(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            src = pathlib.Path(d) / "in.jsonl"
            dst = pathlib.Path(d) / "out.jsonl"
            src.write_text('{"a": 1,}\n{"b": 2}\n')
            repairs, errors = heal_file(str(src), str(dst))
            self.assertEqual(len(repairs), 1)
            self.assertEqual(errors, [])
            self.assertEqual(dst.read_text(), '{"a": 1}\n{"b": 2}\n')

    # Healing follows the named symlink but reads all input before modifying its target.
    def test_in_place_symlink_repair_preserves_link(self):
        for no_staging in (False, True):
            with self.subTest(no_staging=no_staging), tempfile.TemporaryDirectory(dir=ROOT) as d:
                src = pathlib.Path(d) / "in.jsonl"
                link = pathlib.Path(d) / "link.jsonl"
                src.write_text('{"a": 1,}\n', encoding="utf-8")
                link.symlink_to(src)
                with self.assertRaisesRegex(DatasetError, "--force"):
                    heal_file(link, link)
                repairs, errors = heal_file(link, link, force=True, no_staging=no_staging)
                self.assertTrue(link.is_symlink())
                self.assertEqual(src.read_text(), '{"a": 1}\n')
                self.assertEqual((len(repairs), errors), (1, []))

    # Decode failures never damage a pre-existing forced destination.
    def test_invalid_utf8_preserves_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            src, dst = pathlib.Path(d) / "in.jsonl", pathlib.Path(d) / "out.jsonl"
            src.write_bytes(b"\xff")
            dst.write_text("original", encoding="utf-8")
            with self.assertRaisesRegex(DatasetError, "UTF-8"):
                heal_file(src, dst, force=True)
            self.assertEqual(dst.read_text(), "original")
