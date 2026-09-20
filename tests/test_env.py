"""The .env loader (SPEC 1).

Each test writes a file in a temporary directory and restores os.environ, so
nothing here depends on the caller's environment or leaks into it.
"""

import os
import pathlib
import tempfile
import unittest

from dataset.env import load
from dataset.io import DatasetError


class EnvLoadTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).resolve().parent)
        self.dir = pathlib.Path(self._tmp.name)
        self._environ = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._environ)
        self._tmp.cleanup()

    def write(self, text):
        path = self.dir / ".env"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_missing_file_is_not_an_error(self):
        self.assertEqual(load(str(self.dir / "absent")), [])

    def test_assigns_values_and_skips_comments_and_blanks(self):
        path = self.write("# a comment\n\nTRLX_TEST_A=one\nTRLX_TEST_B = two \n")
        self.assertEqual(sorted(load(path)), ["TRLX_TEST_A", "TRLX_TEST_B"])
        self.assertEqual(os.environ["TRLX_TEST_A"], "one")
        self.assertEqual(os.environ["TRLX_TEST_B"], "two")

    def test_matching_quotes_are_stripped(self):
        path = self.write("TRLX_TEST_Q='one two'\nTRLX_TEST_D=\"three\"\n")
        load(path)
        self.assertEqual(os.environ["TRLX_TEST_Q"], "one two")
        self.assertEqual(os.environ["TRLX_TEST_D"], "three")

    # A value containing '=' is common in keys and tokens; only the first
    # separator splits the line.
    def test_value_may_contain_equals(self):
        path = self.write("TRLX_TEST_E=a=b=c\n")
        load(path)
        self.assertEqual(os.environ["TRLX_TEST_E"], "a=b=c")

    def test_existing_environment_wins(self):
        os.environ["TRLX_TEST_SET"] = "from environment"
        path = self.write("TRLX_TEST_SET=from file\n")
        self.assertEqual(load(path), [])
        self.assertEqual(os.environ["TRLX_TEST_SET"], "from environment")

    def test_line_without_separator_is_fatal_with_its_number(self):
        path = self.write("TRLX_TEST_A=one\nnot a pair\n")
        with self.assertRaisesRegex(DatasetError, "line 2: expected KEY=value"):
            load(path)

    # Credential errors identify a line without disclosing its contents.
    def test_nul_reports_line_without_secret(self):
        path = self.write("TRLX_TEST_SECRET=private-value\0suffix\n")
        with self.assertRaisesRegex(DatasetError, "line 1.*NUL") as result:
            load(path)
        self.assertNotIn("private-value", str(result.exception))

    # Decode errors also omit the raw environment bytes.
    def test_invalid_utf8(self):
        path = self.dir / ".env"
        path.write_bytes(b"TRLX_TEST_SECRET=private\xff")
        with self.assertRaisesRegex(DatasetError, "UTF-8") as result:
            load(path)
        self.assertNotIn("private", str(result.exception))


if __name__ == "__main__":
    unittest.main()
