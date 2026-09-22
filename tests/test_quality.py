"""Independent evaluation evidence and state isolation with CPU-only local models."""

import copy
import json
import math
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from transformers import ProcessorMixin

from dataset.io import DatasetError
from trlx import TrlxError, quality, quality_scorers
from tests.test_quality_scorers import prompt_templates


# No dataset backend or model artifact is needed to exercise the full row iterator.
class Rows(list):
    @property
    # The evaluator's dataset contract uses num_rows rather than len().
    def num_rows(self):
        return len(self)


# Keep settings explicit in fixtures so tests never depend on runtime configuration defaults.
def settings(preset="qa", **changes):
    values = dict(quality_preset=preset, quality_dataset="independent-fixture",
                  quality_max_length=1024, quality_max_new_tokens=128,
                  quality_batch_size=2, judge=None, prompts=prompt_templates())
    values.update(changes)
    return SimpleNamespace(**values)


# A reversible character tokenizer makes every eligible token and generated prompt inspectable.
class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "<eos>"
    chat_template = None

    # Full character identity lets series tests detect vocabulary changes without model files.
    def get_vocab(self):
        return {"<pad>": 0, "<eos>": 1, **{chr(index): index + 2 for index in range(128)}}

    # Explicit attention masks mirror the small subset used by independent model scoring.
    def __call__(self, text, *, add_special_tokens=True, return_tensors=None):
        # EOS is one special token, so duplicated EOS remains visible in the model inputs.
        parts = text.split(self.eos_token) if self.eos_token is not None else [text]
        ids = []
        for index, part in enumerate(parts):
            if index:
                ids.append(self.eos_token_id)
            ids.extend(ord(char) + 2 for char in part)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}
        return {"input_ids": ids}

    # Preserve message roles so tests detect accidental chat flattening or reference leakage.
    def apply_chat_template(self, messages, *, tokenize=False, return_dict=False,
                            add_generation_prompt=False, chat_template=None):
        text = "\n".join(f"<{item['role']}>{item['content']}" for item in messages)
        text = text + ("\n<assistant>" if add_generation_prompt else "")
        if not tokenize:
            return text
        encoded = self(text, add_special_tokens=False)
        return encoded if return_dict else encoded["input_ids"]

    # Special-token removal matches generation suffix decoding, not full prompt decoding.
    def decode(self, ids, *, skip_special_tokens=True):
        return "".join(chr(int(token) - 2) for token in ids if int(token) > 1)


# Exercise TRL's real ProcessorMixin branch, including structured text and batch unwrapping.
class Processor(ProcessorMixin):
    # No pretrained components are needed; the processor owns its template independently.
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.chat_template = "processor-template"
        self.calls = []

    # Real processors return a batch dimension even for a single text example.
    def __call__(self, text):
        self.calls.append(text)
        return {"input_ids": [self.tokenizer(text)["input_ids"]]}

    # TRL must normalize strings into structured text blocks before calling a processor.
    def apply_chat_template(self, messages, *, tokenize=False, return_dict=False,
                            add_generation_prompt=False, chat_template=None):
        if not all(isinstance(message["content"], list) for message in messages):
            raise AssertionError("TRL did not normalize processor message content")
        text = "[" + (chat_template or self.chat_template) + "]"
        for message in messages:
            text += f"<{message['role']}>" + "".join(block["text"] for block in message["content"])
        if add_generation_prompt:
            text += "<assistant>"
        if not tokenize:
            return text
        encoded = self(text)
        return encoded if return_dict else encoded["input_ids"]


# The evaluator copies and restores this object; tests can check identity as well as values.
class Generation:
    # Explicit deterministic generation fields are part of comparable-series identity.
    def __init__(self):
        self.eos_token_id = 1
        self.temperature = 1.0

    # Return independent data so series hashing cannot mutate the model configuration.
    def to_dict(self):
        return dict(vars(self))


# Fixed logits and scripted generation reveal scheduling and accounting without learned behavior.
class Model(torch.nn.Module):
    # The parameter gives the runner an authoritative device and a gradient to preserve.
    def __init__(self, outputs=(), *, reward=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.dropout = torch.nn.Dropout(0.5)
        self.generation_config = Generation()
        self.outputs = list(outputs)
        self.reward = reward
        self.forwards = []
        self.generations = []

    # Nonuniform fixed logits make unequal token-count averaging errors observable.
    def forward(self, input_ids, attention_mask=None):
        self.forwards.append(input_ids.detach().clone())
        if self.reward:
            logits = input_ids.float().sum(dim=1, keepdim=True) / 100
        else:
            prior = torch.arange(256, device=input_ids.device, dtype=torch.float32) / 50
            logits = prior.expand(input_ids.shape[0], input_ids.shape[1], -1) + self.weight
        return SimpleNamespace(logits=logits)

    # Return exact policy suffixes plus EOS; padding makes batches independently inspectable.
    def generate(self, input_ids, attention_mask, **kwargs):
        self.generations.append({"inputs": input_ids.detach().clone(), "mask": attention_mask.detach().clone(),
                                 "kwargs": copy.deepcopy(kwargs)})
        suffixes = []
        for _ in range(input_ids.shape[0]):
            text = self.outputs.pop(0)
            suffix = [ord(char) + 2 for char in text]
            if len(suffix) < kwargs["max_new_tokens"]:
                suffix.append(1)
            suffixes.append(suffix[:kwargs["max_new_tokens"]])
        width = max(map(len, suffixes))
        padded = [suffix + [0] * (width - len(suffix)) for suffix in suffixes]
        return torch.cat([input_ids, torch.tensor(padded, device=input_ids.device)], dim=1)


# Mimic PEFT attribute delegation without depending on adapters or model artifacts.
class DelegatingModel(torch.nn.Module):
    # Nested owners share one config while the outer wrapper owns no local shadow.
    def __init__(self):
        super().__init__()
        self.base_model = Model()
        self.other_owner = Model()
        self.other_owner.generation_config = self.base_model.generation_config

    # Ordinary module resolution remains authoritative except for the delegated config.
    def __getattr__(self, name):
        if name == "generation_config":
            return super().__getattr__("base_model").generation_config
        return super().__getattr__(name)


class ObservationalTest(unittest.TestCase):
    # Both normal and exceptional exits restore every observed training-state boundary.
    def test_rng_modes_generation_and_gradients_restored(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                model = Model()
                model.train()
                model.dropout.eval()
                model.weight.grad = torch.tensor(7.0)
                original_generation = model.generation_config
                python_state = random.getstate()
                numpy_state = np.random.get_state()
                torch_state = torch.get_rng_state().clone()
                try:
                    with quality.observational(model):
                        self.assertFalse(torch.is_grad_enabled())
                        self.assertFalse(any(module.training for module in model.modules()))
                        self.assertIsNot(model.generation_config, original_generation)
                        model.generation_config.temperature = 42
                        random.random()
                        np.random.random(10)
                        torch.rand(10)
                        if fail:
                            raise ValueError("simulated scorer failure")
                except ValueError as exc:
                    self.assertTrue(fail)
                    self.assertEqual(str(exc), "simulated scorer failure")
                self.assertEqual(random.getstate(), python_state)
                after_numpy = np.random.get_state()
                self.assertEqual(after_numpy[0], numpy_state[0])
                np.testing.assert_array_equal(after_numpy[1], numpy_state[1])
                self.assertEqual(after_numpy[2:], numpy_state[2:])
                self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
                self.assertTrue(model.training)
                self.assertFalse(model.dropout.training)
                self.assertIs(model.generation_config, original_generation)
                self.assertEqual(model.generation_config.temperature, 1.0)
                self.assertEqual(model.weight.grad.item(), 7.0)
                self.assertEqual(model.weight.item(), 0.0)


    # Config aliases and wrapper-local absence survive rebinding on both success and failure.
    def test_delegated_generation_owners_aliases_and_shadow_cleanup(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                model = DelegatingModel()
                original = model.generation_config
                self.assertNotIn("generation_config", vars(model))
                try:
                    with quality.observational(model):
                        self.assertIsNot(model.generation_config, original)
                        self.assertIs(model.base_model.generation_config, model.other_owner.generation_config)
                        model.base_model.generation_config.temperature = 20
                        model.generation_config = Generation()
                        model.other_owner.generation_config = Generation()
                        model.base_model.dropout.generation_config = Generation()
                        if fail:
                            raise ValueError("generation failed after rebinding")
                except ValueError:
                    self.assertTrue(fail)
                self.assertNotIn("generation_config", vars(model))
                self.assertNotIn("generation_config", vars(model.base_model.dropout))
                self.assertIs(model.generation_config, original)
                self.assertIs(model.base_model.generation_config, original)
                self.assertIs(model.other_owner.generation_config, original)
                self.assertEqual(original.temperature, 1.0)


class DatasetValidationTest(unittest.TestCase):
    # Validation visits the final row rather than blessing only an early sample.
    def test_entire_dataset_checked_before_evaluation(self):
        rows = Rows([{"prompt": "Question", "answer": "answer"}] * 20 + [{"prompt": "Malformed"}])
        with patch.object(quality.data_load, "load_ref", return_value=rows) as load:
            with self.assertRaisesRegex(TrlxError, "row 21"):
                quality.load_data(settings())
        load.assert_called_once_with("independent-fixture", progress=None)

    # Empty datasets are errors, while valid row objects retain their authoritative identity.
    def test_empty_and_valid_datasets(self):
        with patch.object(quality.data_load, "load_ref", return_value=Rows()):
            with self.assertRaisesRegex(TrlxError, "empty"):
                quality.load_data(settings())
        rows = Rows([{"prompt": "Question", "answer": "answer"}])
        with patch.object(quality.data_load, "load_ref", return_value=rows):
            self.assertIs(quality.load_data(settings()), rows)


class InputInspectionTest(unittest.TestCase):
    # Full scans include late outliers and budget the actual generation constraints, without sampling.
    def test_full_generation_lengths_limits_and_position_budget(self):
        rows = Rows([{"prompt": "Q", "answer": "A"}] * 20 + [{"prompt": "Q" * 100, "answer": "A"}])
        configured = settings(quality_max_length=150, quality_max_new_tokens=16)
        result = quality.inspect_inputs(configured, Tokenizer(), rows, {"max_position_embeddings": 200})
        lengths = [len(quality_scorers.generation_prompt("qa", row, configured.prompts)) for row in rows]
        self.assertEqual(result["input_lengths"], [{"row": index, "tokens": [length]}
                                                  for index, length in enumerate(lengths, 1)])
        self.assertEqual(result["over_limit_rows"], [21])
        self.assertEqual(result["input_errors"], [])
        self.assertEqual(result["largest_sequence_budget"], lengths[-1] + 16)
        self.assertEqual({item["code"] for item in result["findings"]},
                         {"quality_input_limits", "quality_context_budget"})
        self.assertEqual(result["findings"][0]["evidence"]["over_limit_rows"], [21])

    # LM corpus size can exceed the input window; the position budget is the window, not the corpus.
    def test_lm_window_budget_and_short_row_error_evidence(self):
        rows = Rows([{"text": "abc"}, {"text": "abcdefghijk"}, {"text": "a"}])
        result = quality.inspect_inputs(settings("language_modeling", quality_max_length=4),
                                        Tokenizer(), rows, {"max_position_embeddings": 4})
        self.assertEqual(result["input_lengths"], [{"row": 1, "tokens": [3]}, {"row": 2, "tokens": [11]}])
        self.assertEqual(result["largest_sequence_budget"], 4)
        self.assertEqual(result["over_limit_rows"], [])
        self.assertEqual(len(result["input_errors"]), 1)
        error = result["input_errors"][0]
        self.assertEqual(error["row"], 3)
        self.assertEqual(error["type"], "TrlxError")
        self.assertIn("fewer than two tokens", error["error"])
        self.assertEqual([finding["code"] for finding in result["findings"]], ["quality_input_limits"])
        json.dumps(result, allow_nan=False)

    # Preference lengths include EOS for each branch, and either branch can exceed the configured limit.
    def test_preference_branch_lengths_include_single_eos(self):
        rows = Rows([{"chosen": "B", "rejected": "A<eos>"}, {"chosen": "abcdef", "rejected": "x"}])
        result = quality.inspect_inputs(settings("preference", quality_max_length=5), Tokenizer(), rows, {})
        self.assertEqual(result["input_lengths"], [{"row": 1, "tokens": [2, 2]}, {"row": 2, "tokens": [7, 2]}])
        self.assertEqual(result["over_limit_rows"], [2])
        self.assertEqual(result["largest_sequence_budget"], 7)

    # Failed encodings carry row-numbered errors and cannot fabricate a zero-token observation.
    def test_all_failed_inputs_leave_budget_unknown(self):
        result = quality.inspect_inputs(settings(), Tokenizer(), Rows([{"prompt": "Q"}]), {})
        self.assertEqual(result["input_lengths"], [])
        self.assertEqual(result["over_limit_rows"], [])
        self.assertIsNone(result["largest_sequence_budget"])
        self.assertEqual(result["input_errors"][0]["row"], 1)
        self.assertIn("answer", result["input_errors"][0]["error"])

    # Processors own chat serialization; the same real TRL normalization serves inspection and inference.
    def test_processor_template_used_and_original_not_mutated(self):
        processor, model = Processor(), Model(["Answer"])
        row = {"prompt": "Question", "answer": "Answer"}
        rows = Rows([row])
        configured = settings()
        result = quality.inspect_inputs(configured, processor, rows, {})
        quality.evaluate(model, processor, rows, configured)
        inputs = model.generations[0]["inputs"][0]
        rendered = processor.tokenizer.decode(inputs)
        self.assertTrue(rendered.startswith("[processor-template]<user>Question"))
        self.assertTrue(rendered.endswith("<assistant>"))
        self.assertIn("concise answer", rendered)
        self.assertEqual(result["input_lengths"], [{"row": 1, "tokens": [len(inputs)]}])
        self.assertEqual(processor.calls, [])
        self.assertIsNone(processor.tokenizer.chat_template)
        self.assertEqual(rows[0], row)

    # Processor batches must unwrap for raw corpus text as well as conversational prompts.
    def test_processor_raw_lm_batch_dimension_unwrapped(self):
        processor, model = Processor(), Model()
        rows = Rows([{"text": "abc"}])
        inspected = quality.inspect_inputs(settings("language_modeling"), processor, rows, {})
        quality.evaluate(model, processor, rows, settings("language_modeling"))
        self.assertEqual(inspected["input_lengths"], [{"row": 1, "tokens": [3]}])
        self.assertEqual(model.forwards[0][0].tolist(), [ord(char) + 2 for char in "abc"])
        self.assertEqual(processor.calls, [])


class LanguageModelingTest(unittest.TestCase):
    # One-token overlap scores every target exactly once, including a short final window.
    def test_window_coverage_and_token_weighted_loss(self):
        tokenizer, model = Tokenizer(), Model()
        text = "abcdefgh"
        result = quality.evaluate(model, tokenizer, Rows([{"text": text}]),
                                  settings("language_modeling", quality_max_length=4))[0]
        windows = [call[0].tolist() for call in model.forwards]
        ids = tokenizer(text)["input_ids"]
        self.assertEqual(windows, [ids[0:4], ids[3:7], ids[6:8]])
        self.assertEqual([token for window in windows for token in window[1:]], ids[1:])
        log_probabilities = torch.log_softmax(torch.arange(256).float() / 50, dim=0)
        expected_sum = -float(log_probabilities[torch.tensor(ids[1:])].sum())
        self.assertEqual(result["details"]["tokens"], 7)
        self.assertEqual(result["details"]["windows"], 3)
        self.assertAlmostEqual(result["details"]["nll_sum"], expected_sum, places=5)
        self.assertAlmostEqual(result["metrics"]["loss"], expected_sum / 7, places=5)
        self.assertAlmostEqual(result["metrics"]["perplexity"], math.exp(expected_sum / 7), places=3)

    # Corpus loss weights targets, not examples, and perplexity is derived from corpus loss.
    def test_aggregate_uses_target_counts(self):
        results = [{"details": {"tokens": 2, "nll_sum": 2}},
                   {"details": {"tokens": 10, "nll_sum": 30}}]
        metrics = quality.aggregate("language_modeling", results)
        self.assertEqual(metrics["quality/tokens"], 12)
        self.assertAlmostEqual(metrics["quality/loss"], 32 / 12)
        self.assertAlmostEqual(metrics["quality/perplexity"], math.exp(32 / 12))
        self.assertEqual(metrics["quality/rows"], 2)

    # Raw corpus text must not inherit a chat template's artificial context tokens.
    def test_raw_text_not_chat_wrapped(self):
        tokenizer, model = Tokenizer(), Model()
        tokenizer.chat_template = "fixture-template"
        quality.evaluate(model, tokenizer, Rows([{"text": "abc"}]), settings("language_modeling"))
        self.assertEqual(model.forwards[0][0].tolist(), tokenizer("abc")["input_ids"])

    # A row without a predictable target is not silently dropped from the denominator.
    def test_too_short_row_rejected(self):
        with self.assertRaisesRegex(TrlxError, "fewer than two tokens"):
            quality.evaluate(Model(), Tokenizer(), Rows([{"text": "a"}]), settings("language_modeling"))


class PreferenceTest(unittest.TestCase):
    # Ties remain ties instead of being counted as successes or discarded.
    def test_rankings_ties_and_signed_margins(self):
        rows = Rows([{"chosen": "B", "rejected": "A"},
                     {"chosen": "A", "rejected": "A"},
                     {"prompt": "Q", "chosen": "A", "rejected": "B"}])
        results = quality.evaluate(Model(reward=True), Tokenizer(), rows, settings("preference"))
        self.assertEqual([row["score"] for row in results], [1.0, 0.0, 0.0])
        self.assertEqual([row["metrics"]["tie_rate"] for row in results], [0.0, 1.0, 0.0])
        self.assertGreater(results[0]["metrics"]["margin"], 0)
        self.assertLess(results[2]["metrics"]["margin"], 0)
        aggregate = quality.aggregate("preference", results)
        self.assertAlmostEqual(aggregate["quality/accuracy"], 1 / 3)
        self.assertAlmostEqual(aggregate["quality/tie_rate"], 1 / 3)

    # Plain reward inputs acquire exactly one EOS, including already terminated references.
    def test_preference_eos_added_once(self):
        tokenizer, model = Tokenizer(), Model(reward=True)
        rows = Rows([{"chosen": "B", "rejected": "A<eos>"}])
        quality.evaluate(model, tokenizer, rows, settings("preference"))
        self.assertEqual([call[0].tolist() for call in model.forwards], [[ord("B") + 2, 1], [ord("A") + 2, 1]])
        for call in model.forwards:
            self.assertEqual(call[0].tolist().count(tokenizer.eos_token_id), 1)

    # Truncated references and nonscalar reward heads would change the benchmark's meaning.
    def test_invalid_pair_model_or_length_fails(self):
        row = Rows([{"chosen": "long answer", "rejected": "bad"}])
        with self.assertRaisesRegex(TrlxError, "no input was silently truncated"):
            quality.evaluate(Model(reward=True), Tokenizer(), row, settings("preference", quality_max_length=3))
        with self.assertRaisesRegex(TrlxError, "scalar reward"):
            quality.evaluate(Model(), Tokenizer(), row, settings("preference"))
        model = Model(reward=True)
        with patch.object(model, "forward", return_value=SimpleNamespace(logits=torch.tensor([[float("nan")]]))):
            with self.assertRaisesRegex(TrlxError, "non-finite"):
                quality.evaluate(model, Tokenizer(), row, settings("preference"))


class GenerationTest(unittest.TestCase):
    # The actual model sees task constraints and all choices, never the held-out answers.
    def test_builtin_generation_prompts_and_batch_size(self):
        cases = [
            ("qa", {"prompt": "Question?", "answer": "hidden_answer"}, "response", "concise answer"),
            ("classification", {"prompt": "Class?", "label": "yes", "labels": ["yes", "no"]}, "yes", "Permitted labels"),
            ("multiple_choice", {"prompt": "Choice?", "answer": "A", "choices": {"A": "Paris", "B": "Rome"}}, "A", "A: Paris"),
            ("json", {"prompt": "Object?", "reference": {"name": "hidden_answer"}, "required_fields": ["name"]},
             '{"name":"response"}', 'top-level keys: ["name"]'),
        ]
        for preset, row, output, constraint in cases:
            with self.subTest(preset=preset):
                model, tokenizer = Model([output] * 3), Tokenizer()
                rows = Rows([row, copy.deepcopy(row), copy.deepcopy(row)])
                with patch.object(quality_scorers, "generation_prompt", wraps=quality_scorers.generation_prompt) as prepare:
                    results = quality.evaluate(model, tokenizer, rows, settings(preset, quality_batch_size=2))
                self.assertEqual(prepare.call_count, 3)
                self.assertEqual([call["inputs"].shape[0] for call in model.generations], [2, 1])
                for call in model.generations:
                    for inputs in call["inputs"]:
                        prompt = tokenizer.decode(inputs)
                        self.assertIn(constraint, prompt)
                        self.assertNotIn("hidden_answer", prompt)
                    self.assertFalse(call["kwargs"]["do_sample"])
                    self.assertFalse(call["kwargs"]["use_cache"])
                    for flag in ("return_dict_in_generate", "output_scores", "output_logits",
                                 "output_attentions", "output_hidden_states"):
                        self.assertIs(call["kwargs"][flag], False)
                self.assertEqual([result["output"] for result in results], [output] * 3)
                self.assertEqual([result["row"] for result in results], [1, 2, 3])
                self.assertEqual([result["metrics"]["completion_cutoff"] for result in results], [0.0] * 3)

    # Left padding uses masks; a generated suffix never accidentally includes prompt tokens.
    def test_padding_and_completion_cutoff(self):
        model, tokenizer = Model(["abcd", "ok"]), Tokenizer()
        rows = Rows([{"prompt": "Short", "answer": "x"}, {"prompt": "A longer prompt", "answer": "x"}])
        results = quality.evaluate(model, tokenizer, rows, settings(quality_max_new_tokens=4))
        masks = model.generations[0]["mask"]
        self.assertEqual(masks[0, 0].item(), 0)
        self.assertEqual(masks[1, 0].item(), 1)
        self.assertEqual(results[0]["output"], "abcd")
        self.assertEqual(results[1]["output"], "ok")
        self.assertEqual(results[0]["metrics"]["completion_cutoff"], 1.0)
        self.assertEqual(results[1]["metrics"]["completion_cutoff"], 0.0)

    # Excess input and missing padding semantics produce errors instead of hidden truncation.
    def test_generation_input_errors(self):
        rows = Rows([{"prompt": "Question", "answer": "x"}])
        with self.assertRaisesRegex(TrlxError, "silently truncated"):
            quality.evaluate(Model(["x"]), Tokenizer(), rows, settings(quality_max_length=3))
        tokenizer = Tokenizer()
        tokenizer.pad_token_id = tokenizer.eos_token_id = None
        with self.assertRaisesRegex(TrlxError, "pad or EOS"):
            quality.evaluate(Model(["x"]), tokenizer, rows, settings())

    # Per-class support makes class imbalance visible beside total accuracy and invalid counts.
    def test_class_aggregate_denominators(self):
        labels = ["yes", "no/other"]
        rows = Rows([{"prompt": "Q", "label": label, "labels": labels} for label in ["yes", " YES ", "no/other"]])
        results = quality.evaluate(Model(["yes", "invalid", "no/other"]), Tokenizer(), rows,
                                   settings("classification"))
        metrics = quality.aggregate("classification", results)
        self.assertEqual(metrics["quality/rows"], 3)
        self.assertEqual(metrics["quality/scored_rows"], 3)
        self.assertEqual(metrics["quality/metric_rows/accuracy"], 3)
        self.assertAlmostEqual(metrics["quality/accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["quality/invalid"], 1 / 3)
        self.assertEqual(metrics["quality/class/yes/rows"], 2)
        self.assertEqual(metrics["quality/class/yes/accuracy"], 0.5)
        self.assertEqual(metrics["quality/class/no%2Fother/rows"], 1)
        self.assertEqual(metrics["quality/class/no%2Fother/accuracy"], 1.0)

    # Optional criteria expose their actual denominator rather than implying every row carried it.
    def test_optional_metric_denominators(self):
        results = [
            {"score": 1.0, "metrics": {"json_valid": 1.0, "reference_values_correct": 1.0}},
            {"score": 0.0, "metrics": {"json_valid": 0.0}},
            {"score": None, "metrics": {"json_valid": 1.0}},
        ]
        metrics = quality.aggregate("json", results)
        self.assertEqual(metrics["quality/rows"], 3)
        self.assertEqual(metrics["quality/scored_rows"], 2)
        self.assertEqual(metrics["quality/score"], 0.5)
        self.assertEqual(metrics["quality/metric_rows/json_valid"], 3)
        self.assertEqual(metrics["quality/metric_rows/reference_values_correct"], 1)
        self.assertAlmostEqual(metrics["quality/json_valid"], 2 / 3)
        self.assertEqual(metrics["quality/reference_values_correct"], 1.0)

    # Aggregating invalid evidence cannot silently make a finite-looking recommendation.
    def test_empty_or_nonfinite_aggregate_fails(self):
        with self.assertRaisesRegex(TrlxError, "no scored rows"):
            quality.aggregate("qa", [])
        with self.assertRaisesRegex(TrlxError, "non-finite or nonnumeric"):
            quality.aggregate("qa", [{"score": 0, "metrics": {"accuracy": float("nan")}}])


class JudgeTest(unittest.TestCase):
    # Endpoint construction receives an explicit secret value without persisting it in evidence.
    def test_judge_endpoint_setup_and_missing_secret(self):
        spec = dict(url="http://judge.invalid/v1", model="judge", api_key="TRLX_TEST_JUDGE_KEY",
                    timeout=10, retries=0, max_tokens=200)
        configured = settings("writing", judge=spec)
        with patch.dict(os.environ, {"TRLX_TEST_JUDGE_KEY": "secret-value"}), patch.object(quality, "Endpoint") as endpoint:
            self.assertIs(quality.judge_endpoint(configured), endpoint.return_value)
            endpoint.assert_called_once_with(spec["url"], "judge", "secret-value", 10, 0)
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(TrlxError, "TRLX_TEST_JUDGE_KEY.*not set"):
            quality.judge_endpoint(configured)

    # A malformed or failed judge response is a failed assessment, never a substituted zero.
    def test_judge_failures_are_contextual_and_restore_model(self):
        rows = Rows([{"prompt": "Write a poem"}])
        for failure in ("Score: 1", DatasetError("endpoint unavailable")):
            with self.subTest(failure=failure):
                endpoint = Mock()
                if isinstance(failure, Exception):
                    endpoint.complete.side_effect = failure
                else:
                    endpoint.complete.return_value = failure
                model = Model(["Poem"])
                model.train()
                generation = model.generation_config
                with patch.object(quality, "judge_endpoint", return_value=endpoint):
                    with self.assertRaisesRegex(TrlxError, "quality rows 1-1"):
                        quality.evaluate(model, Tokenizer(), rows, settings("writing", judge={"max_tokens": 100}))
                self.assertTrue(model.training)
                self.assertIs(model.generation_config, generation)

    # Valid judgments retain both structured evidence and the original reply for inspection.
    def test_judge_evidence_retained(self):
        reply = json.dumps({"score": 0.5, "criteria": dict.fromkeys(quality_scorers.JUDGE_CRITERIA, 0.5),
                            "rationale": "The poem is coherent but incomplete."})
        endpoint = Mock()
        endpoint.complete.return_value = reply
        with patch.object(quality, "judge_endpoint", return_value=endpoint):
            result = quality.evaluate(Model(["Poem"]), Tokenizer(), Rows([{"prompt": "Write a poem"}]),
                                      settings("writing", judge={"max_tokens": 100}))[0]
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["details"]["judge_reply"], reply)
        self.assertTrue(result["details"]["model_judgment"])


class SeriesTest(unittest.TestCase):
    # Only comparable inputs share a baseline; changing scorer semantics invalidates it too.
    def test_series_identity_changes_with_evaluation_inputs(self):
        configured, tokenizer, model = settings(), Tokenizer(), Model()
        baseline = quality.series_id(configured, "data-one", tokenizer, model)
        self.assertEqual(baseline, quality.series_id(configured, "data-one", tokenizer, model))
        self.assertNotEqual(baseline, quality.series_id(configured, "data-two", tokenizer, model))
        for field, value in (("quality_preset", "json"), ("quality_max_length", 2048),
                             ("quality_max_new_tokens", 10), ("quality_batch_size", 3)):
            changed = copy.deepcopy(configured)
            setattr(changed, field, value)
            with self.subTest(field=field):
                self.assertNotEqual(baseline, quality.series_id(changed, "data-one", tokenizer, model))
        tokenizer.chat_template = "a new chat template"
        self.assertNotEqual(baseline, quality.series_id(configured, "data-one", tokenizer, model))
        tokenizer.chat_template = None
        tokenizer.eos_token = "<different-eos>"
        self.assertNotEqual(baseline, quality.series_id(configured, "data-one", tokenizer, model))
        tokenizer.eos_token = "<eos>"
        with patch.object(quality, "SCORER_VERSION", quality.SCORER_VERSION + 1):
            self.assertNotEqual(baseline, quality.series_id(configured, "data-one", tokenizer, model))
        model.generation_config.temperature = 0.5
        self.assertNotEqual(baseline, quality.series_id(configured, "data-one", tokenizer, model))

    # Judge text changes invalidate a baseline even though policy encodings are identical.
    def test_judge_prompt_contents_change_series_identity(self):
        configured, tokenizer, model = settings("writing"), Tokenizer(), Model()
        rows = Rows([{"prompt": "Write about trees"}])
        baseline = quality.series_id(configured, "data", tokenizer, model, dataset=rows)
        configured.prompts["quality_writing_judge"] = "Use a different rubric."
        self.assertNotEqual(baseline, quality.series_id(configured, "data", tokenizer, model, dataset=rows))

    # A processor-owned template defines comparable inputs even when its tokenizer has no template.
    def test_processor_template_changes_series_identity(self):
        configured, processor, model = settings(), Processor(), Model()
        baseline = quality.series_id(configured, "data", processor, model)
        processor.chat_template = "changed-processor-template"
        self.assertIsNone(processor.tokenizer.chat_template)
        self.assertNotEqual(baseline, quality.series_id(configured, "data", processor, model))

    # Service and judge identity matter, while rotated API secrets must never enter baseline identity.
    def test_judge_identity_excludes_secrets(self):
        configured = settings("writing", judge={"url": "http://judge.invalid/v1", "model": "one",
                                               "api_key": "SECRET_A", "max_tokens": 100})
        tokenizer, model = Tokenizer(), Model()
        baseline = quality.series_id(configured, "data", tokenizer, model)
        configured.judge["api_key"] = "SECRET_B"
        self.assertEqual(baseline, quality.series_id(configured, "data", tokenizer, model))
        configured.judge["model"] = "two"
        self.assertNotEqual(baseline, quality.series_id(configured, "data", tokenizer, model))
        configured.judge["model"] = "one"
        configured.judge["max_tokens"] = 200
        self.assertNotEqual(baseline, quality.series_id(configured, "data", tokenizer, model))


class FastTokenizerSeriesTest(unittest.TestCase):
    # Construct a real Rust-backed tokenizer entirely in memory, with an inspectable tiny vocabulary.
    def tokenizer(self):
        from tokenizers import Tokenizer as BackendTokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        backend = BackendTokenizer(models.WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "hello": 3, "HELLO": 4, "world": 5, "##s": 6},
            unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
                                      eos_token="[EOS]", clean_up_tokenization_spaces=False)

    # Equal vocabularies do not establish equal normalization or decoding behavior.
    def test_normalizer_and_decoder_semantics_change_series(self):
        from tokenizers import decoders, normalizers

        configured, model = settings("language_modeling"), Model()
        baseline_tokenizer = self.tokenizer()
        baseline = quality.series_id(configured, "data", baseline_tokenizer, model)
        changed = self.tokenizer()
        changed.backend_tokenizer.normalizer = normalizers.Lowercase()
        self.assertEqual(changed.get_vocab(), baseline_tokenizer.get_vocab())
        self.assertNotEqual(changed("HELLO")["input_ids"], baseline_tokenizer("HELLO")["input_ids"])
        self.assertNotEqual(baseline, quality.series_id(configured, "data", changed, model))
        first, second = self.tokenizer(), self.tokenizer()
        first.backend_tokenizer.decoder = decoders.WordPiece(prefix="##", cleanup=False)
        second.backend_tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")
        self.assertEqual(first.get_vocab(), second.get_vocab())
        self.assertNotEqual(first.decode([3, 6]), second.decode([3, 6]))
        self.assertNotEqual(quality.series_id(configured, "data", first, model),
                            quality.series_id(configured, "data", second, model))

    # Added-token matching flags can change boundaries while token strings and IDs remain equal.
    def test_added_token_flags_change_series(self):
        from tokenizers import AddedToken

        configured, model = settings("language_modeling"), Model()
        for flag in ("single_word", "lstrip", "rstrip", "normalized"):
            with self.subTest(flag=flag):
                first, second = self.tokenizer(), self.tokenizer()
                first.add_tokens([AddedToken("<extra>", **{flag: False})])
                second.add_tokens([AddedToken("<extra>", **{flag: True})])
                self.assertEqual(first.get_vocab(), second.get_vocab())
                self.assertNotEqual(quality.series_id(configured, "data", first, model),
                                    quality.series_id(configured, "data", second, model))

    # Previous padded/truncated calls do not define the unpadded evaluation codec.
    def test_backend_padding_and_truncation_history_ignored(self):
        configured, tokenizer, model = settings("language_modeling"), self.tokenizer(), Model()
        baseline = quality.series_id(configured, "data", tokenizer, model)
        tokenizer.backend_tokenizer.enable_padding(length=16, pad_id=1, pad_token="[PAD]")
        tokenizer.backend_tokenizer.enable_truncation(max_length=16)
        self.assertIsNotNone(json.loads(tokenizer.backend_tokenizer.to_str())["padding"])
        self.assertEqual(baseline, quality.series_id(configured, "data", tokenizer, model))

    # Decoder cleanup and chat templates belong to the wrapper and are independent of vocabulary.
    def test_current_wrapper_cleanup_and_template_change_series(self):
        configured, tokenizer, model = settings("language_modeling"), self.tokenizer(), Model()
        baseline = quality.series_id(configured, "data", tokenizer, model)
        tokenizer.clean_up_tokenization_spaces = True
        self.assertNotEqual(baseline, quality.series_id(configured, "data", tokenizer, model))
        tokenizer.clean_up_tokenization_spaces = False
        self.assertEqual(baseline, quality.series_id(configured, "data", tokenizer, model))
        tokenizer.chat_template = "{{ messages[0]['content'] }}"
        self.assertNotEqual(baseline, quality.series_id(configured, "data", tokenizer, model))

    # Exact encoded examples contribute identity even if an upstream fingerprint was incorrectly reused.
    def test_exact_dataset_encodings_participate_in_series(self):
        configured, tokenizer, model = settings("language_modeling"), self.tokenizer(), Model()
        first = Rows([{"text": "hello world"}])
        second = Rows([{"text": "world hello"}])
        first_id = quality.series_id(configured, "same-fingerprint", tokenizer, model, dataset=first)
        second_id = quality.series_id(configured, "same-fingerprint", tokenizer, model, dataset=second)
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(first_id, quality.series_id(configured, "same-fingerprint", tokenizer, model, dataset=copy.deepcopy(first)))

    # Arbitrary Python codecs must establish full identity rather than silently using only their vocabulary.
    def test_unpickleable_python_codec_is_rejected(self):
        tokenizer = Tokenizer()
        # A local callable intentionally has no portable pickle identity.
        tokenizer.custom_codec = lambda text: text
        with self.assertRaisesRegex(TrlxError, "cannot establish independent quality tokenizer identity"):
            quality.series_id(settings(), "data", tokenizer, Model())


class PublicationTest(unittest.TestCase):
    # Evidence rounds append completely and retain individual inputs and outputs, in repo-local scratch.
    def test_rounds_publish_as_complete_jsonl_records(self):
        with tempfile.TemporaryDirectory(prefix=".test-quality-", dir=Path(__file__).resolve().parents[1]) as folder:
            sample = [{"row": 1, "input": {"prompt": "Q", "answer": "A"}, "output": "A", "score": 1.0}]
            quality.publish(folder, {"phase": "baseline", "step": 0}, sample)
            quality.publish(folder, {"phase": "completion", "step": 10}, sample)
            path = Path(folder) / quality.FILENAME
            lines = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["quality"]["phase"] for row in lines], ["baseline", "completion"])
            self.assertEqual(lines[0]["results"], sample)
            self.assertEqual(list(Path(folder).iterdir()), [path])

    # A staged writer failure preserves the prior round byte-for-byte and cleans staging debris.
    def test_staged_failure_preserves_previous_evidence(self):
        with tempfile.TemporaryDirectory(prefix=".test-quality-", dir=Path(__file__).resolve().parents[1]) as folder:
            quality.publish(folder, {"phase": "baseline"}, [{"score": 1.0}])
            path = Path(folder) / quality.FILENAME
            previous = path.read_bytes()
            with patch("dataset.io._publish_prepared", side_effect=OSError("simulated publication failure")):
                with self.assertRaisesRegex(TrlxError, "cannot publish independent quality evidence"):
                    quality.publish(folder, {"phase": "completion"}, [{"score": 0.0}])
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(folder).iterdir()), [path])

    # Nonfinite evidence is rejected before publication rather than producing nonstandard JSON.
    def test_nonfinite_evidence_rejected(self):
        with tempfile.TemporaryDirectory(prefix=".test-quality-", dir=Path(__file__).resolve().parents[1]) as folder:
            with self.assertRaisesRegex(TrlxError, "cannot publish independent quality evidence"):
                quality.publish(folder, {"phase": "baseline"}, [{"score": float("nan")}])
            self.assertFalse((Path(folder) / quality.FILENAME).exists())


if __name__ == "__main__":
    unittest.main()
