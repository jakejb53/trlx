"""End-to-end advisory decisions for measured learning trajectories, without training."""

import copy
import unittest
from types import SimpleNamespace

from trlx import assessment, review


# Explicit resolved settings keep recommendations independent of operational defaults.
def arguments(**changes):
    values = dict(learning_rate=.0002, max_steps=-1, num_train_epochs=8, lr_scheduler_type="linear",
                  eval_strategy="steps", eval_steps=5, logging_strategy="steps", logging_steps=1,
                  bf16=True, fp16=False, max_completion_length=128, num_generations=4, temperature=1.,
                  get_warmup_steps=lambda steps: 0)
    return SimpleNamespace(**(values | changes))


# Build the same immutable nested metric envelopes the real callback writes.
def losses(values, steps=None):
    steps = steps if steps is not None else range(len(values))
    return [{"step": step, "eval": True, "log": {"eval_loss": value}} for step, value in zip(steps, values)]


# Quality comparisons require matching conditions and complete rounds.
def quality(step, score, *, series="same", status="complete", phase="scheduled"):
    return {"step": step, "eval": True, "quality": {"series": series, "status": status, "phase": phase},
            "log": {"quality/accuracy": score}}


class AssessmentScenarios(unittest.TestCase):
    # Render the actual report so tests catch advice lost or strengthened by presentation.
    def report(self, records, method="sft", completed=None, planned=None, args=None, checkpoints=None):
        args = args or arguments()
        completed = max(r["step"] for r in records) if completed is None else completed
        planned = completed if planned is None else planned
        original = copy.deepcopy(records)
        report = assessment.run_assessment(method, records, args, completed_steps=completed,
                    planned_steps=planned, checkpoint_steps=checkpoints)
        rendered = review.render_run_assessment(report, args, final=True, width=160)
        self.assertEqual(records, original)
        return report, " ".join(rendered.split())

    # The user's run has a short last interval; rates, not raw interval deltas, establish slowing gains.
    def test_real_diminishing_run_does_not_prescribe_another_epoch(self):
        steps = [0, 5, 10, 15, 20, 25, 30, 32]
        records = losses([2.655, 1.211, 1.050, .970, .939, .9291242361, .9269463420, .9259175658], steps)
        records += [{"step": step, "eval": False, "log": {"learning_rate": value}}
                    for step, value in ((20, .00008125), (25, .00005), (30, .00001875), (32, .00000625))]
        records.sort(key=lambda r: r["step"])
        report, rendered = self.report(records, checkpoints=[28, 32])
        self.assertIn("diminishing", report["final_decision"])
        self.assertIn("steps 20–32", rendered)
        self.assertIn("learning rate fell", rendered)
        self.assertIn("Checkpoint-32 is available", rendered)
        self.assertNotIn("--num-train-epochs 9", rendered)
        self.assertNotIn("Continue", report["decision"])

    # A tiny monotone decline is measurable but not evidence for a worthwhile duration increase.
    def test_tiny_decline_is_not_hidden_or_prescribed_as_more_training(self):
        records = losses([1., .99999, .99998, .99997])
        epoch, rendered = self.report(records)
        steps, rendered_steps = self.report(records, args=arguments(max_steps=3))
        self.assertEqual(epoch["final_decision"], steps["final_decision"])
        self.assertIn("0.99997", rendered)
        self.assertIn("does not establish a useful extension", rendered)
        self.assertNotIn("--max-steps 6", rendered_steps)

    # Early gains cannot conceal repeated late deterioration or invent an unsaved best checkpoint.
    def test_late_reversal_and_checkpoint_availability(self):
        report, rendered = self.report(losses([3., 1.2, 1., .8, .81, .83, .86]), checkpoints=[2, 6])
        self.assertIn("rising", report["final_decision"])
        self.assertIn("No verified saved checkpoint", rendered)
        self.assertIn("available checkpoints: 0.86 at step 6", rendered)
        self.assertNotIn("--max-steps 3", rendered)

    # Different train/evaluation behavior is evidence for inspection, not a causal LR diagnosis.
    def test_rising_training_loss_does_not_override_improving_evaluation(self):
        records = losses([2., 1.8, 1.6, 1.4])
        records += [{"step": i, "eval": False, "log": {"loss": value}}
                    for i, value in enumerate([1., 1.1, 1.2, 1.3])]
        records.sort(key=lambda r: r["step"])
        report, rendered = self.report(records)
        self.assertNotIn("--learning-rate 0.0001", rendered)
        self.assertFalse(any(i["code"] == "rising_training_loss" for i in report["issues"]))
        self.assertIn("do not establish instability", rendered)

    # A display-range warning cannot replace the result or lose the warning's own measurement.
    def test_warnings_keep_outcome_and_support(self):
        args = arguments()
        report = assessment.run_assessment("sft", losses([2., 1.8, 1.6, 1.4]), args,
                   completed_steps=3, planned_steps=3, ranges={"eval_loss": [0, 1]})
        rendered = " ".join(review.render_run_assessment(report, args, final=True, width=160).split())
        self.assertIn(report["final_decision"], rendered)
        self.assertIn("Baseline → latest", rendered)
        self.assertIn("eval_loss at step 3: 1.4", rendered)
        self.assertIn("Configured range: [0, 1]", rendered)

    # Complete task regressions cannot be overridden by improved loss or training reward.
    def test_conflicting_quality_is_not_overruled_by_loss(self):
        records = losses([2., 1.8, 1.6, 1.4]) + [quality(0, .8, phase="baseline"), quality(3, .7)]
        report, rendered = self.report(records)
        self.assertIn("conflict", report["final_decision"])
        self.assertIn("quality/accuracy", rendered)
        self.assertIn("0.8 → 0.7", rendered)

    # Positive quality evidence affects advice without treating it as a universal model score.
    def test_quality_improvement_and_failed_latest_round(self):
        records = [quality(0, .5, phase="baseline"), quality(2, .7), quality(3, .8)]
        report, rendered = self.report(records, method="grpo")
        self.assertIn("Independent task scores improved", report["final_decision"])
        self.assertIn("0.5 → 0.8", rendered)
        report, rendered = self.report(records + [quality(4, .99, status="failed")], method="grpo")
        self.assertNotIn("Independent task scores improved", report["final_decision"])
        self.assertIn("incomplete or failed", rendered)
        self.assertNotIn("0.99", rendered)

    # Changing the scorer/dataset series cannot turn unmatched scores into a baseline comparison.
    def test_mismatched_quality_and_stale_endpoint(self):
        records = losses([2., 1.8, 1.6, 1.4]) + [quality(0, .5, phase="baseline"), quality(3, .8, series="new")]
        report, rendered = self.report(records, completed=4, planned=4)
        self.assertIn("step 4 was not evaluated", report["final_decision"])
        self.assertIn("no completed comparison", rendered)

    # Successful task comparisons cannot hide absent loss evaluation or a worse baseline loss.
    def test_positive_quality_retains_loss_conflicts(self):
        records = losses([1., 2., 1.8, 1.6]) + [quality(0, .5, phase="baseline"), quality(3, .8)]
        report, rendered = self.report(records)
        self.assertIn("Compare both objectives", report["final_decision"])
        records.append(quality(4, .9))
        report, rendered = self.report(records, completed=4, planned=4)
        self.assertIn("step 4 was not evaluated", report["final_decision"])

    # Failed or baseline-only independent checks are not a completed quality comparison.
    def test_unavailable_quality_does_not_invent_measurements(self):
        for item in (quality(3, .8, status="failed"), quality(0, .8, phase="baseline")):
            report, rendered = self.report([item], completed=3, planned=3)
            self.assertIn("No completed held-out comparison", report["final_decision"])
            self.assertNotIn("Use the independent quality measurements", rendered)

    # A cyclic schedule cannot be described as constant just because endpoints match.
    def test_learning_rate_context_uses_interior_observations(self):
        records = losses([2., 1.8, 1.6, 1.4])
        records += [{"step": i, "eval": False, "log": {"learning_rate": value}}
                    for i, value in enumerate([.001, .01, .01, .001])]
        records.sort(key=lambda r: r["step"])
        report, rendered = self.report(records)
        self.assertIn("learning rate varied", rendered)
        self.assertNotIn("learning rate stayed", rendered)

    # Preference loss cannot override accuracy regression; policy methods do not optimize monotone loss.
    def test_method_specific_objectives(self):
        for method in ("dpo", "kto", "reward", "grpo", "rloo", "distillation"):
            records = losses([2., 1.8, 1.6, 1.4])
            records += [{"step": i, "eval": False, "log": {"loss": float(i), "reward": float(i)}} for i in range(4)]
            records.sort(key=lambda r: r["step"])
            report, rendered = self.report(records, method=method)
            self.assertNotIn("--learning-rate 0.0001", rendered)
            self.assertNotIn("--num-train-epochs 9", rendered)
        records = losses([2., 1.8, 1.6, 1.4])
        for row, accuracy in zip(records, [.8, .7, .6, .5]):
            row["log"]["eval_rewards/accuracies"] = accuracy
        report, rendered = self.report(records, method="dpo")
        self.assertIn("conflict", report["final_decision"])
        self.assertIn("eval_rewards/accuracies", rendered)

    # Numerical issues and insufficient post-warmup data must not produce confident tuning advice.
    def test_warmup_nonfinite_and_short_run(self):
        report, rendered = self.report(losses([2., 1.8, 1.6]), args=arguments(get_warmup_steps=lambda n: n))
        self.assertIn("Warmup is still active", rendered)
        self.assertNotIn("--learning-rate", rendered)
        report, rendered = self.report(losses([2., float("nan"), 1.6]), completed=2, planned=5)
        self.assertIn("Training ended at 2/5", report["final_decision"])
        self.assertIn("non-finite", rendered)
        self.assertNotIn("Continue", rendered)

    # Zero variance is not zero reward; cutoff advice requires inspecting the actual answers.
    def test_policy_diagnostics_keep_their_meaning(self):
        records = [{"step": i, "eval": False, "log": {"reward": 10., "reward_std": 0.,
                    "completions/clipped_ratio": .1}} for i in range(3)]
        report, rendered = self.report(records, method="grpo")
        self.assertIn("no within-group variation", rendered)
        self.assertNotIn("reward is zero", rendered)
        self.assertIn("if useful answers are cut short", rendered)
        self.assertNotIn("Increase --max-completion-length", rendered)
