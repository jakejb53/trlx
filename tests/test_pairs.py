"""Alignment by user turns, system exclusion, unmatched reporting, and the
single-input distillation split."""

import unittest

from dataset.io import DatasetError
from dataset.pairs import align, single


# Builds a messages row from alternating user/assistant turn contents.
def conv(*turns, system=None):
    messages = [{"role": "system", "content": system}] if system else []
    roles = ["user", "assistant"]
    messages += [{"role": roles[i % 2], "content": t} for i, t in enumerate(turns)]
    return {"messages": messages}


class Align(unittest.TestCase):
    def test_pairs_by_user_turns(self):
        chosen = [conv("q1", "good1"), conv("q2", "a", "q3", "good3")]
        rejected = [conv("q2", "a", "q3", "bad3"), conv("q1", "bad1")]
        pairs, unmatched = align(chosen, rejected)
        self.assertEqual(unmatched, [])
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["prompt"], [{"role": "user", "content": "q1"}])
        self.assertEqual(pairs[0]["chosen"], [{"role": "assistant", "content": "good1"}])
        self.assertEqual(pairs[0]["rejected"], [{"role": "assistant", "content": "bad1"}])
        self.assertEqual(len(pairs[1]["prompt"]), 3)

    def test_system_excluded_from_key_but_kept_in_prompt(self):
        chosen = [conv("q", "good", system="S1")]
        rejected = [conv("q", "bad", system="S2")]
        pairs, unmatched = align(chosen, rejected)
        self.assertEqual(unmatched, [])
        self.assertEqual(pairs[0]["prompt"][0], {"role": "system", "content": "S1"})

    def test_unmatched_reported_and_dropped(self):
        chosen = [conv("q1", "g"), conv("only-chosen", "g")]
        rejected = [conv("q1", "b"), conv("only-rejected", "b")]
        pairs, unmatched = align(chosen, rejected)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(unmatched, ["chosen row 1", "rejected row 1"])

    def test_duplicate_keys_pair_in_order(self):
        chosen = [conv("q", "g1"), conv("q", "g2")]
        rejected = [conv("q", "b1"), conv("q", "b2")]
        pairs, unmatched = align(chosen, rejected)
        self.assertEqual([p["rejected"][0]["content"] for p in pairs], ["b1", "b2"])

    def test_missing_final_assistant_is_error(self):
        with self.assertRaises(DatasetError):
            align([{"messages": [{"role": "user", "content": "q"}]}], [])


class Single(unittest.TestCase):
    def test_prompt_completion_split(self):
        rows = single([conv("q", "a1", "q2", "a2")])
        self.assertEqual(len(rows[0]["prompt"]), 3)
        self.assertEqual(rows[0]["completion"], [{"role": "assistant", "content": "a2"}])
