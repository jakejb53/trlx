"""CPU tests for the routing-score math: slot and gate-weight accumulation honours
the token mask, scores normalize as ESFT's do, coverage counts experts in
descending order, and selection follows ESFT's threshold rule, on a concentrated
and a uniform routing."""

import unittest

import torch

from deepseek.qat import route_scores as RS


class Accumulate(unittest.TestCase):
    def test_mask_and_sums(self):
        counts = torch.zeros(8, dtype=torch.int64)
        gate_sum = torch.zeros(8, dtype=torch.float64)
        indices = torch.tensor([[0, 1], [2, 3], [0, 7]])
        weights = torch.tensor([[0.6, 0.4], [0.5, 0.5], [0.9, 0.1]])
        mask = torch.tensor([True, False, True])
        RS.accumulate(counts, gate_sum, indices, weights, mask)
        self.assertEqual(counts.tolist(), [2, 1, 0, 0, 0, 0, 0, 1])
        self.assertAlmostEqual(gate_sum[0].item(), 1.5)
        self.assertAlmostEqual(gate_sum[7].item(), 0.1)
        # No mask counts every token; a wrong-length mask is an error.
        RS.accumulate(counts, gate_sum, indices, weights, None)
        self.assertEqual(int(counts.sum()), 4 + 6)  # 2 valid tokens x 2 slots, then 3 tokens x 2 slots
        with self.assertRaises(ValueError):
            RS.accumulate(counts, gate_sum, indices, weights, torch.tensor([True]))

    def test_scores_normalization(self):
        counts = torch.tensor([3, 1, 0, 0], dtype=torch.int64)  # 4 slots at topk 2 = 2 tokens
        gate_sum = torch.tensor([1.2, 0.8, 0.0, 0.0], dtype=torch.float64)
        token_scores, gate_scores = RS.scores(counts, gate_sum, topk=2)
        self.assertAlmostEqual(sum(token_scores), 1.0)
        self.assertEqual(token_scores, [0.75, 0.25, 0.0, 0.0])
        self.assertEqual(gate_scores, [0.6, 0.4, 0.0, 0.0])
        with self.assertRaises(ValueError):
            RS.scores(torch.zeros(4, dtype=torch.int64), gate_sum, 2)


class CoverageAndSelection(unittest.TestCase):
    def test_concentrated_routing(self):
        score = [0.5, 0.3, 0.1, 0.05, 0.05]
        self.assertEqual(RS.coverage(score), {"0.2": 1, "0.5": 1, "0.8": 2})
        # ESFT's rule includes the expert that crosses the threshold.
        self.assertEqual(RS.select(score, 0.2), [0])
        self.assertEqual(RS.select(score, 0.6), [0, 1])
        self.assertEqual(RS.select(score, 1.0), [0, 1, 2, 3, 4])

    def test_uniform_routing(self):
        # Exact binary fractions, so the cumulative sums hit the thresholds without rounding.
        score = [1 / 8] * 8
        self.assertEqual(RS.coverage(score), {"0.2": 2, "0.5": 4, "0.8": 7})
        self.assertEqual(len(RS.select(score, 0.2)), 2)

    def test_select_rejects_bad_top_p(self):
        with self.assertRaises(ValueError):
            RS.select([0.5, 0.5], 0.0)

    def test_summary_fields(self):
        counts = torch.tensor([6, 2, 0, 0], dtype=torch.int64)
        gate_sum = torch.tensor([3.0, 1.0, 0.0, 0.0], dtype=torch.float64)
        s = RS.summarize_layer(counts, gate_sum, topk=2, top_p=0.5, score_function="token")
        self.assertEqual(s["tokens"], 4)
        self.assertEqual(s["selected"], [0])
        self.assertEqual(s["experts_with_zero_slots"], 2)
        self.assertEqual(s["max_token_share"], 0.75)


if __name__ == "__main__":
    unittest.main()
