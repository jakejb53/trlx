"""DeepSeek-V4 Python encoding behind the shared Transformers tokenizer contract."""

import copy
import string
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast
from trl.chat_template_utils import is_chat_template_stop_token_trained
from trl.data_utils import _tokenize, apply_chat_template

from trlx import chat_encoding, data_profile, deepseek_v4_encoding, model, reasoning


# A character tokenizer plus native control tokens makes offsets and masks observable without a model download.
def tokenizer():
    controls = [
        deepseek_v4_encoding.bos_token,
        deepseek_v4_encoding.eos_token,
        deepseek_v4_encoding.USER_SP_TOKEN,
        deepseek_v4_encoding.ASSISTANT_SP_TOKEN,
        deepseek_v4_encoding.thinking_start_token,
        deepseek_v4_encoding.thinking_end_token,
    ]
    tokens = ["<pad>", "<unk>", *controls, *string.printable]
    vocab = {token: index for index, token in enumerate(dict.fromkeys(tokens))}
    backend = Tokenizer(BPE(vocab, [], unk_token="<unk>"))
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token=deepseek_v4_encoding.bos_token,
        eos_token=deepseek_v4_encoding.eos_token,
        additional_special_tokens=controls[2:],
    )


# Architecture metadata is the sole automatic-selection input.
def adapted():
    config = SimpleNamespace(architectures=[chat_encoding.DEEPSEEK_V4_ARCHITECTURE])
    return chat_encoding.adapt_processor(tokenizer(), config)


# Rows use the project's separate reasoning column until reasoning.prepare maps it natively.
def row(trace="trace"):
    return {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        "reasoning": trace,
    }


# Explicit arguments keep the reasoning projection independent from trainer and hardware defaults.
def reasoning_config(*, reasoning_only=False):
    args = SimpleNamespace(
        eos_token=None,
        chat_template_path=None,
        assistant_only_loss=True,
        completion_only_loss=None,
        dataset_text_field="text",
        dataset_kwargs=None,
        max_length=256,
        packing=False,
        eval_packing=None,
        packing_strategy="bfd",
        shuffle_dataset=False,
        seed=1,
        padding_free=False,
        dataset_num_proc=None,
        truncation_mode="keep_start",
        tokenizer_kwargs=None,
        processor_kwargs=None,
        chat_template_kwargs=None,
        use_liger_kernel=False,
        per_device_train_batch_size=1,
    )
    dataset = SimpleNamespace(include_reasoning=True, reasoning_only_loss=reasoning_only)
    return SimpleNamespace(method=SimpleNamespace(name="sft"), args=args, dataset=dataset)


class DeepSeekV4EncodingTests(unittest.TestCase):
    # The official low-effort format contains no inferred reasoning-effort prefix.
    def test_exact_default_render_and_explicit_effort(self):
        processor = adapted()
        messages = copy.deepcopy(row()["messages"])
        messages[-1]["reasoning"] = "trace"
        expected = (
            deepseek_v4_encoding.bos_token
            + deepseek_v4_encoding.USER_SP_TOKEN + "question"
            + deepseek_v4_encoding.ASSISTANT_SP_TOKEN + deepseek_v4_encoding.thinking_start_token
            + "trace" + deepseek_v4_encoding.thinking_end_token + "answer"
            + deepseek_v4_encoding.eos_token
        )
        self.assertEqual(processor.apply_chat_template(messages, tokenize=False), expected)
        high = processor.apply_chat_template(messages, tokenize=False, reasoning_effort="high")
        self.assertTrue(high.startswith(deepseek_v4_encoding.bos_token +
                                        deepseek_v4_encoding.REASONING_EFFORT_PROMPTS["high"]))

    # Transformers callers receive the same shapes for single, batch, tensor, and continuation requests.
    def test_transformers_apply_contract(self):
        processor = adapted()
        complete = copy.deepcopy(row()["messages"])
        complete[-1]["reasoning"] = "trace"
        text = processor.apply_chat_template(complete, tokenize=False)
        ids = processor.apply_chat_template(complete, return_dict=False)
        encoded = processor.apply_chat_template(complete, return_dict=True)
        batch = processor.apply_chat_template([complete, complete], return_dict=True, padding=True)
        tensors = processor.apply_chat_template([complete, complete], return_dict=True,
                                                padding=True, return_tensors="pt")
        continued = processor.apply_chat_template(complete, tokenize=False, continue_final_message=True)
        self.assertEqual(ids, encoded["input_ids"])
        self.assertEqual(len(batch["input_ids"]), 2)
        self.assertEqual(tuple(tensors["input_ids"].shape), (2, len(ids)))
        self.assertTrue(text.endswith(deepseek_v4_encoding.eos_token))
        self.assertFalse(continued.endswith(deepseek_v4_encoding.eos_token))

    # TRL's language-model, preference, unpaired-preference, and prompt-only shapes share exact prefixes.
    def test_trainer_dataset_shapes(self):
        processor = adapted()
        prompt = [{"role": "user", "content": "question"}]
        chosen = [{"role": "assistant", "reasoning": "trace", "content": "chosen"}]
        rejected = [{"role": "assistant", "reasoning": "other", "content": "rejected"}]
        preference = apply_chat_template(
            {"prompt": prompt, "chosen": chosen, "rejected": rejected}, processor)
        self.assertEqual(preference["prompt"] + preference["chosen"],
                         processor.apply_chat_template(prompt + chosen, tokenize=False))
        self.assertEqual(preference["prompt"] + preference["rejected"],
                         processor.apply_chat_template(prompt + rejected, tokenize=False))
        prompt_ids = _tokenize(processor, prompt, add_generation_prompt=True)["input_ids"]
        for completion in (chosen, rejected):
            full_ids = _tokenize(processor, prompt + completion)["input_ids"]
            self.assertEqual(full_ids[:len(prompt_ids)], prompt_ids)
        batch = processor.apply_chat_template([prompt, prompt], add_generation_prompt=True,
                                              tokenize=True, return_dict=True)
        self.assertEqual(len(batch["input_ids"]), 2)

    # Assistant masks include both native reasoning boundaries and the final stop token.
    def test_assistant_masks_and_trl_tokenization(self):
        processor = adapted()
        messages = copy.deepcopy(row()["messages"])
        messages[-1]["reasoning"] = "trace"
        encoded = processor.apply_chat_template(messages, return_dict=True, return_assistant_tokens_mask=True)
        self.assertEqual(_tokenize(processor, messages)["input_ids"], encoded["input_ids"])
        active = [token for token, flag in zip(encoded["input_ids"], encoded["assistant_masks"], strict=True)
                  if flag]
        self.assertEqual(active[0], processor.convert_tokens_to_ids(deepseek_v4_encoding.thinking_start_token))
        self.assertEqual(active[-1], processor.convert_tokens_to_ids(deepseek_v4_encoding.eos_token))
        with chat_encoding.trainer_capabilities(processor):
            self.assertTrue(is_chat_template_stop_token_trained(processor))
        self.assertIsNone(processor.chat_template)

    # The response template separates native reasoning and parses repeated DSML function calls.
    def test_response_parsing(self):
        processor = adapted()
        prefix = (deepseek_v4_encoding.bos_token + deepseek_v4_encoding.USER_SP_TOKEN + "question" +
                  deepseek_v4_encoding.ASSISTANT_SP_TOKEN + deepseek_v4_encoding.thinking_start_token)
        response = (
            "trace" + deepseek_v4_encoding.thinking_end_token + "answer\n\n"
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"lookup\">\n"
            "<｜DSML｜parameter name=\"count\" string=\"false\">2</｜DSML｜parameter>\n"
            "</｜DSML｜invoke>\n</｜DSML｜tool_calls>" + deepseek_v4_encoding.eos_token
        )
        parsed = processor.parse_response(response, prefix=prefix)
        self.assertEqual(parsed["reasoning"], "trace")
        self.assertEqual(parsed["content"], "answer")
        self.assertEqual(parsed["tool_calls"][0]["function"], {"name": "lookup", "arguments": {"count": 2}})

    # Request tools attach to an existing system message without creating a second system prompt.
    def test_tools_and_explicit_jinja_override(self):
        processor = adapted()
        messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "question"}]
        tools = [{"type": "function", "function": {
            "name": "lookup", "description": "look up", "parameters": {"type": "object", "properties": {}}
        }}]
        rendered = processor.apply_chat_template(messages, tools=tools, tokenize=False)
        self.assertEqual(rendered.count("## Tools"), 1)
        self.assertEqual(processor.apply_chat_template(
            messages, tokenize=False, chat_template="{{ messages[1]['content'] }}"), "question")

    # Deep copies retain the adapter; explicit-template ownership restores the tokenizer's original state.
    def test_selection_and_copy(self):
        original = tokenizer()
        self.assertIs(chat_encoding.adapt_processor(original, SimpleNamespace(architectures=["Other"])), original)
        processor = adapted()
        clone = copy.deepcopy(processor)
        prompt = [{"role": "user", "content": "question"}]
        self.assertEqual(clone.apply_chat_template(prompt, tokenize=False),
                         processor.apply_chat_template(prompt, tokenize=False))
        self.assertTrue(chat_encoding.is_non_jinja(clone))
        self.assertIs(chat_encoding.remove_adapter(clone), clone)
        self.assertFalse(chat_encoding.is_non_jinja(clone))
        self.assertIsNone(clone.response_template)

    # Both ordinary inclusion and reasoning-only loss survive the shared prepared-dataset path.
    def test_reasoning_preparation_modes(self):
        for only in (False, True):
            with self.subTest(reasoning_only=only):
                cfg, processor = reasoning_config(reasoning_only=only), adapted()
                mapped, _ = reasoning.prepare(cfg, processor, Dataset.from_list([row()]), None)
                self.assertIn("input_ids", mapped.column_names)
                self.assertTrue(any(label != -100 for label in mapped[0]["labels"]))
                self.assertEqual(data_profile.scan(cfg, processor, mapped, None)["train"]["errors"], [])

    # Model loading applies the registry result before returning the shared processor.
    def test_model_loader_applies_architecture_adapter(self):
        spec = SimpleNamespace(path="model", trust_remote_code=None)
        config = SimpleNamespace(architectures=[chat_encoding.DEEPSEEK_V4_ARCHITECTURE])
        processor = tokenizer()
        with patch.object(model.AutoConfig, "from_pretrained", return_value=config), \
             patch.object(model.AutoProcessor, "from_pretrained", return_value=processor):
            loaded = model.load_processor(spec)
        self.assertIs(loaded, processor)
        self.assertTrue(chat_encoding.is_non_jinja(loaded))


if __name__ == "__main__":
    unittest.main()
