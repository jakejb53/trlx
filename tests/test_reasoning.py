"""Reasoning mapping and supervision against real synthetic-template preparation."""

import copy
import string
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast
from trl import SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from trlx import TrlxError, reasoning


# Arbitrary native fields prevent tests from depending on a model-family heuristic.
def tokenizer(field="deliberation", *, ignore=False, masked=False):
    tokens = ["<pad>", "<unk>", "<eos>", "<user>", "<assistant>", "<end>"]
    tokens += list(string.ascii_letters + string.digits + " .,!?\n_")
    backend = Tokenizer(BPE({token: index for index, token in enumerate(tokens)}, [], unk_token="<unk>"))
    result = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>",
                                    eos_token="<eos>", additional_special_tokens=["<user>", "<assistant>", "<end>"])
    trace = "" if ignore else "{% if not hide_reasoning %}{{ message['" + field + "'] }}{% endif %}"
    answer = "{{ message['content'] + '<end>' }}"
    body = trace + "{% generation %}" + answer + "{% endgeneration %}" if masked else (
        "{% generation %}" + trace + answer + "{% endgeneration %}")
    result.chat_template = (
        "{% for message in messages %}{{ '<' + message['role'] + '>' }}"
        "{% if message['role'] == 'assistant' %}" + body +
        "{% else %}{{ message['content'] + '<end>' }}{% endif %}{% endfor %}"
    )
    result.response_template = {"fields": {"content": {"content": "text"}, field: {"content": "text"}}}
    return result


# Explicit settings avoid constructing a model or consulting hardware-dependent defaults.
def config(**overrides):
    settings = dict(eos_token=None, chat_template_path=None, assistant_only_loss=True, completion_only_loss=None,
                    dataset_text_field="text", dataset_kwargs=None, max_length=128, truncation_mode="keep_start",
                    packing=False, eval_packing=None, packing_strategy="bfd", shuffle_dataset=False, seed=1,
                    padding_free=False, dataset_num_proc=None, use_liger_kernel=False,
                    per_device_train_batch_size=2, chat_template_kwargs=None)
    settings.update(overrides)
    return SimpleNamespace(method=SimpleNamespace(name="sft"), dataset=SimpleNamespace(include_reasoning=True),
                           args=SimpleNamespace(**settings))


# Distinct trace letters allow checking actual loss-bearing tokens independently of rendering.
def row(trace="XYZ"):
    return {"messages": [{"role": "user", "content": "question"},
                         {"role": "assistant", "content": "answer"}], "reasoning": trace}


# Explicit native delimiters exercise reasoning-only labeling without model-family assumptions.
def bounded_tokenizer():
    processor = tokenizer()
    processor.add_special_tokens({"additional_special_tokens": ["<trace>", "</trace>", "<system>"]})
    processor.chat_template = processor.chat_template.replace(
        "{{ message['deliberation'] }}", "{{ '<trace>' + message['deliberation'] + '</trace>' }}")
    processor.response_template["fields"]["deliberation"].update(open="<trace>", close_pattern="</trace>")
    return processor


class ReasoningTests(unittest.TestCase):
    # Disabled mapping must not inspect either rows or tokenizer metadata.
    def test_disabled_returns_original_datasets(self):
        cfg = config()
        cfg.dataset.include_reasoning = False
        train, evaluation = Dataset.from_list([{"text": "plain"}]), Dataset.from_list([{"text": "eval"}])
        result = reasoning.prepare(cfg, object(), train, evaluation)
        self.assertIs(result[0], train)
        self.assertIs(result[1], evaluation)

    # Native-field mapping applies equally to primary/replay/eval without mutating its inputs.
    def test_native_fields_all_splits_and_nonmutation(self):
        for field in ("deliberation", "analysis_text"):
            with self.subTest(field=field):
                processor = tokenizer(field)
                original_template = processor.chat_template
                original_schema = copy.deepcopy(processor.response_template)
                train = Dataset.from_list([dict(row("XYZ"), replay=False), dict(row("UVW"), replay=True)])
                evaluation = Dataset.from_list([row("RST")])
                original = train.to_list()
                mapped, mapped_eval = reasoning.prepare(config(eos_token="<end>"), processor, train, evaluation)
                self.assertEqual([item["messages"][-1][field] for item in mapped], ["XYZ", "UVW"])
                self.assertEqual(mapped_eval[0]["messages"][-1][field], "RST")
                self.assertEqual(mapped["replay"], [False, True])
                self.assertEqual(train.to_list(), original)
                self.assertEqual(processor.chat_template, original_template)
                self.assertEqual(processor.response_template, original_schema)
                self.assertEqual(processor.eos_token, "<eos>")

    # Compare labels from installed TRL and its collator, including packed mixed-source rows.
    def test_actual_trainer_supervises_reasoning(self):
        for packing in (False, True):
            for assistant_only in (False, True):
                with self.subTest(packing=packing, assistant_only=assistant_only):
                    cfg = config(packing=packing, assistant_only_loss=assistant_only, shuffle_dataset=packing)
                    processor = tokenizer()
                    mapped, _ = reasoning.prepare(cfg, processor, Dataset.from_list([row("XYZ"), row("UVW")]), None)
                    context = SimpleNamespace(_tokenizer=processor, chat_template=None, completion_only_loss=False)
                    prepared = SFTTrainer._prepare_dataset(context, mapped, processor, cfg.args, packing, None, "train")
                    batch = DataCollatorForLanguageModeling(processor.pad_token_id, padding_free=packing)(list(prepared))
                    active = [value for value in batch["labels"][:, 1:].reshape(-1).tolist() if value != -100]
                    for letter in "XYZUVW":
                        self.assertEqual(active.count(processor.convert_tokens_to_ids(letter)), 1)
                    if assistant_only:
                        self.assertNotIn(processor.convert_tokens_to_ids("q"), active)

    # Separate traces must have a valid and unambiguous raw assistant destination.
    def test_invalid_rows_are_rejected(self):
        cases = []
        for value in (None, "", "   "):
            cases.append((dict(row(), reasoning=value), "nonempty string"))
        missing = row()
        del missing["reasoning"]
        cases.append((missing, "nonempty string"))
        multiple = row()
        multiple["messages"] += [{"role": "user", "content": "again"}, {"role": "assistant", "content": "reply"}]
        cases.append((multiple, "exactly one assistant"))
        cases.append((dict(row(), input_ids=[1, 2]), "prepared token columns"))
        for key in ("labels", "assistant_masks", "completion_mask"):
            cases.append((dict(row(), **{key: [0, 0]}), "prepared token columns"))
        conflict = row()
        conflict["messages"][-1]["deliberation"] = "different"
        cases.append((conflict, "conflicts"))
        for value, message in cases:
            with self.subTest(message=message, value=value):
                with self.assertRaisesRegex(TrlxError, message):
                    reasoning.prepare(config(), tokenizer(), Dataset.from_list([value]), None)

    # Identical native traces are idempotent, not conflicting duplicate annotations.
    def test_existing_identical_native_trace_is_accepted(self):
        source = row()
        source["messages"][-1]["deliberation"] = source["reasoning"]
        mapped, _ = reasoning.prepare(config(), tokenizer(), Dataset.from_list([source]), None)
        self.assertEqual(mapped[0]["messages"][-1]["deliberation"], "XYZ")

    # Missing or ambiguous metadata must never trigger guessed field names.
    def test_unresolvable_metadata(self):
        processor = tokenizer()
        processor.response_template["fields"]["another"] = {"content": "text"}
        with self.assertRaisesRegex(TrlxError, "exactly one"):
            reasoning.prepare(config(), processor, Dataset.from_list([row()]), None)
        processor.response_template = None
        with patch("trlx.reasoning.add_response_schema", side_effect=ValueError("unrecognized template")):
            with self.assertRaisesRegex(TrlxError, "cannot resolve.*unrecognized"):
                reasoning.prepare(config(), processor, Dataset.from_list([row()]), None)

    # Rendered presence and loss-bearing presence are distinct requirements.
    def test_ignored_and_unsupervised_reasoning(self):
        for processor, expected in ((tokenizer(ignore=True), "does not render"),
                                    (tokenizer(masked=True), "loss mask excludes")):
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(TrlxError, expected):
                    reasoning.prepare(config(), processor, Dataset.from_list([row()]), None)

    # Per-row template arguments are honored when validating both data splits.
    def test_row_template_kwargs_cannot_hide_reasoning(self):
        evaluation = Dataset.from_list([dict(row(), chat_template_kwargs={"hide_reasoning": True})])
        with self.assertRaisesRegex(TrlxError, "eval row 1.*does not render"):
            reasoning.prepare(config(), tokenizer(), Dataset.from_list([row()]), evaluation)

    # Either truncation direction must retain every trace token, not merely one token.
    def test_truncation_rejects_partial_trace(self):
        for mode in ("keep_start", "keep_end"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(TrlxError, "truncation or packing excludes"):
                    reasoning.prepare(config(max_length=12, truncation_mode=mode), tokenizer(),
                                      Dataset.from_list([row("XYZ" * 10)]), None)

    # BFD's oversized-row truncation must not silently discard reasoning from a replay row.
    def test_packing_rejects_oversized_reasoning(self):
        with self.assertRaisesRegex(TrlxError, "train row 2.*excludes reasoning"):
            reasoning.prepare(config(packing=True, max_length=24), tokenizer(),
                              Dataset.from_list([row("XYZ"), row("UVW" * 20)]), None)

    # Override resolution uses the override processor while preserving the caller's configuration.
    def test_override_processor_is_used(self):
        cfg = config(chat_template_path="synthetic-override", eos_token="<pad>")
        processor, override = tokenizer(ignore=True), tokenizer("native_trace")
        original = copy.deepcopy(cfg)
        with patch("trlx.model.assessment_processor", return_value=override) as resolve:
            mapped, _ = reasoning.prepare(cfg, processor, Dataset.from_list([row()]), None)
        resolve.assert_called_once_with(cfg, progress=None)
        self.assertEqual(mapped[0]["messages"][-1]["native_trace"], "XYZ")
        self.assertEqual(override.eos_token, "<eos>")
        self.assertEqual(cfg, original)

    # A tokenizer override supplies both the native field mapping and EOS after the earlier EOS setting.
    def test_template_projection_replaces_base_metadata(self):
        from trlx import model

        cfg = config(chat_template_path="synthetic-override", eos_token="<pad>")
        cfg.model = object()
        base, source = tokenizer("old_trace"), tokenizer("new_trace")
        source.eos_token = "<end>"
        with patch.object(model, "load_processor", return_value=base), \
             patch("transformers.AutoTokenizer.from_pretrained", return_value=source):
            projected = model.assessment_processor(cfg)
        self.assertEqual(projected.response_template, source.response_template)
        self.assertEqual(projected.eos_token, "<end>")

    # Library-provided metadata is accepted only when its native field actually survives the training template.
    def test_library_metadata_and_training_template_resolution(self):
        original, training = tokenizer(), tokenizer()
        original.response_template = None
        original.chat_template = original.chat_template.replace("{% generation %}", "").replace("{% endgeneration %}", "")

        # Synthetic template lookup has no family or delimiter assumptions.
        def attach(processor):
            processor.response_template = training.response_template

        with patch.object(reasoning, "add_response_schema", side_effect=attach), \
             patch.object(reasoning, "get_training_chat_template", return_value=training.chat_template):
            mapped, _ = reasoning.prepare(config(), original, Dataset.from_list([row()]), None)
        self.assertEqual(mapped[0]["messages"][-1]["deliberation"], "XYZ")
        self.assertIsNone(original.response_template)

    # Real TRL packing and collation must retain exactly the selected native reasoning span.
    def test_reasoning_only_actual_trainer_all_splits(self):
        from trlx import data_profile

        for packing in (False, True):
            for assistant_only in (False, True):
                with self.subTest(packing=packing, assistant_only=assistant_only):
                    cfg = config(packing=packing, assistant_only_loss=assistant_only)
                    cfg.dataset.reasoning_only_loss = True
                    processor = bounded_tokenizer()
                    primary = row("XYZ")
                    primary["messages"].insert(0, {"role": "system", "content": "instructions"})
                    replay = dict(row("UVW"), replay=True)
                    train = Dataset.from_list([dict(primary, replay=False), replay])
                    evaluation = Dataset.from_list([row("RST")])
                    original = train.to_list()
                    mapped, mapped_eval = reasoning.prepare(cfg, processor, train, evaluation)
                    self.assertEqual(train.to_list(), original)
                    self.assertEqual(mapped["replay"], [False, True])
                    for dataset, name, traces in ((mapped, "train", ("XYZ", "UVW")),
                                                   (mapped_eval, "eval", ("RST",))):
                        for item in dataset:
                            projected, mismatch = data_profile._sft_row(item, processor, cfg.args, None, False)
                            self.assertFalse(mismatch)
                            self.assertEqual(projected["labels"], item["labels"])
                        context = SimpleNamespace(_tokenizer=processor, chat_template=None, completion_only_loss=False)
                        prepared = SFTTrainer._prepare_dataset(context, dataset, processor, cfg.args,
                                                              packing, None, name)
                        batch = DataCollatorForLanguageModeling(
                            processor.pad_token_id, padding_free=packing)(list(prepared))
                        active = [token for token in batch["labels"][:, 1:].reshape(-1).tolist() if token != -100]
                        expected = [token for trace in traces for token in processor.encode(
                            "<trace>" + trace + "</trace>", add_special_tokens=False)]
                        self.assertCountEqual(active, expected)

    # Empty final answers remain masked and do not force the model to terminate its answer.
    def test_reasoning_only_empty_answer_and_regex_open(self):
        cfg = config()
        cfg.dataset.reasoning_only_loss = True
        processor = bounded_tokenizer()
        processor.response_template["fields"]["deliberation"] = {
            "content": "text", "open_pattern": "<trace>", "close": "</trace>"}
        source = row()
        source["messages"][-1]["content"] = ""
        mapped, _ = reasoning.prepare(cfg, processor, Dataset.from_list([source]), None)
        active = [token for token in mapped[0]["labels"] if token != -100]
        self.assertEqual(active, processor.encode("<trace>XYZ</trace>", add_special_tokens=False))
        self.assertNotIn(processor.convert_tokens_to_ids("<end>"), active)

    # Metadata must identify adjacent, rendered boundaries rather than guessing delimiters.
    def test_reasoning_only_invalid_boundaries(self):
        cfg = config()
        cfg.dataset.reasoning_only_loss = True
        for change, message in (("missing", "explicit opening and closing"),
                                ("regex", "invalid reasoning boundary"),
                                ("closing", "missing or ambiguous"),
                                ("ambiguous", "missing or ambiguous"),
                                ("ignored", "does not render")):
            with self.subTest(change=change):
                processor = bounded_tokenizer()
                definition = processor.response_template["fields"]["deliberation"]
                if change == "missing":
                    del definition["close_pattern"]
                elif change == "regex":
                    definition["close_pattern"] = "["
                elif change == "closing":
                    processor.chat_template = processor.chat_template.replace(" + '</trace>'", "")
                elif change == "ambiguous":
                    definition.pop("open")
                    definition["open_pattern"] = "<trace>|(?=X)"
                else:
                    processor = tokenizer(ignore=True)
                with self.assertRaisesRegex(TrlxError, message):
                    reasoning.prepare(cfg, processor, Dataset.from_list([row()]), None)

    # A parser lookahead does not identify native opening tokens to supervise.
    def test_reasoning_only_rejects_zero_width_opening(self):
        cfg = config()
        cfg.dataset.reasoning_only_loss = True
        processor = bounded_tokenizer()
        definition = processor.response_template["fields"]["deliberation"]
        definition.pop("open")
        definition["open_pattern"] = "(?=XYZ)"
        with self.assertRaisesRegex(TrlxError, "nonempty native opening"):
            reasoning.prepare(cfg, processor, Dataset.from_list([row()]), None)

    # A complete reasoning body is insufficient if truncation loses its learned closing delimiter.
    def test_reasoning_only_truncated_closing_boundary(self):
        cfg = config()
        cfg.dataset.reasoning_only_loss = True
        processor = bounded_tokenizer()
        mapped, _ = reasoning.prepare(cfg, processor, Dataset.from_list([row()]), None)
        cfg.args.max_length = mapped[0]["input_ids"].index(processor.convert_tokens_to_ids("</trace>"))
        with self.assertRaisesRegex(TrlxError, "truncation or packing excludes"):
            reasoning.prepare(cfg, processor, Dataset.from_list([row()]), None)

    # An indivisible token spanning the closer and answer cannot receive a reasoning-only label.
    def test_reasoning_only_token_crossing_answer_boundary(self):
        cfg = config(assistant_only_loss=False)
        cfg.dataset.reasoning_only_loss = True
        processor = bounded_tokenizer()
        processor.chat_template = processor.chat_template.replace("</trace>", "END")
        processor.response_template["fields"]["deliberation"]["close_pattern"] = "END"
        processor.add_tokens(["ENDa"])
        with self.assertRaisesRegex(TrlxError, "token crosses"):
            reasoning.prepare(cfg, processor, Dataset.from_list([row()]), None)
