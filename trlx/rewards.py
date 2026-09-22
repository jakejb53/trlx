"""Built-in reward functions (SPEC 2.8) and the resolver for [rewards] entries.

Every reward has TRL's signature: `reward(completions, **kwargs) -> list of
float`, one score per completion. `completions` are strings (prompt-only
datasets) or conversational lists; `kwargs` carries `prompts` and every other
dataset column as lists aligned with `completions`. A built-in is a factory
`(where, args) -> reward`; `where` labels the config entry in errors and
`args` is the [rewards] table. Argument names are checked here so a typo
fails at startup, not on the first training step.

The function name is what TRL logs the per-reward mean under
(`rewards/<name>/mean`), so each factory names its closure after the entry.
"""

import difflib
import importlib
import importlib.util
import json
import logging
import os
import pathlib
import re
import string

from dataset.endpoint import DatasetError, Endpoint
from dataset.prompts import load as load_prompt
from trlx import TrlxError

_log = logging.getLogger("trlx.rewards")

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_PUNCTUATION = str.maketrans("", "", string.punctuation)


# Assistant text of one completion. A conversational completion is the list
# of messages the policy produced; its last assistant turn is the answer.
def completion_text(completion):
    if isinstance(completion, str):
        return completion
    for message in reversed(completion):
        if message.get("role") == "assistant":
            return message.get("content", "")
    return ""


# Text of one prompt, for judges. Conversational prompts are flattened to
# "role: content" lines so a rubric sees the whole exchange.
def prompt_text(prompt):
    if isinstance(prompt, str):
        return prompt
    return "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in prompt)


# Lowercase, punctuation removed, whitespace collapsed: the comparison form
# for reference_match so "42." and "42" agree.
def normalise(text):
    return " ".join(text.lower().translate(_PUNCTUATION).split())


# Argument check shared by the factories. `spec` maps each accepted name to
# True when required. Values are validated by the factory itself.
def _args(where, args, spec):
    if not isinstance(args, dict):
        raise TrlxError(f"{where}: args must be a table")
    unknown = sorted(set(args) - set(spec))
    if unknown:
        raise TrlxError(f"{where}: unknown args {', '.join(unknown)}; accepted: {', '.join(spec)}")
    missing = sorted(k for k, required in spec.items() if required and k not in args)
    if missing:
        raise TrlxError(f"{where}: missing args {', '.join(missing)}")


def _column(where, kwargs, column):
    if column not in kwargs:
        raise TrlxError(f"{where}: dataset has no column '{column}'; columns seen: {', '.join(sorted(kwargs))}")
    return kwargs[column]


# equals, contains, or fuzzy match against a dataset column after
# normalisation. fuzzy uses difflib's ratio against `threshold`.
def reference_match(where, args):
    _args(where, args, {"column": True, "mode": True, "threshold": False})
    column, mode = args["column"], args["mode"]
    if not isinstance(column, str):
        raise TrlxError(f"{where}: column must be a dataset column name")
    if mode not in ("equals", "contains", "fuzzy"):
        raise TrlxError(f"{where}: mode must be equals, contains, or fuzzy, got {mode!r}")
    threshold = args.get("threshold")
    if mode == "fuzzy" and not isinstance(threshold, (int, float)):
        raise TrlxError(f"{where}: fuzzy mode requires a numeric threshold in [0, 1]")

    def reward(completions, **kwargs):
        references = _column(where, kwargs, column)
        scores = []
        for completion, reference in zip(completions, references):
            got, want = normalise(completion_text(completion)), normalise(str(reference))
            if mode == "equals":
                hit = got == want
            elif mode == "contains":
                hit = want in got
            else:
                hit = difflib.SequenceMatcher(None, got, want).ratio() >= threshold
            scores.append(1.0 if hit else 0.0)
        return scores

    reward.__name__ = "reference_match"
    return reward


# Pattern match on the completion. With `group` and `column`, that capture
# group must equal the column value (after normalisation) for a score of 1.
def regex(where, args):
    _args(where, args, {"pattern": True, "group": False, "column": False})
    try:
        pattern = re.compile(args["pattern"], re.DOTALL)
    except (re.error, TypeError) as e:
        raise TrlxError(f"{where}: invalid pattern: {e}")
    group, column = args.get("group"), args.get("column")
    if (group is None) != (column is None):
        raise TrlxError(f"{where}: group and column go together")
    if group is not None and (type(group) is not int or group < 0):
        raise TrlxError(f"{where}: group must be a nonnegative integer")
    if column is not None and not isinstance(column, str):
        raise TrlxError(f"{where}: column must be a dataset column name")
    if group is not None and group > pattern.groups:
        raise TrlxError(f"{where}: pattern has {pattern.groups} groups, group {group} does not exist")

    def reward(completions, **kwargs):
        references = _column(where, kwargs, column) if column is not None else [None] * len(completions)
        scores = []
        for completion, reference in zip(completions, references):
            m = pattern.search(completion_text(completion))
            if m is None:
                scores.append(0.0)
            elif group is None:
                scores.append(1.0)
            else:
                scores.append(1.0 if normalise(m.group(group) or "") == normalise(str(reference)) else 0.0)
        return scores

    reward.__name__ = "regex"
    return reward


# Fraction of required phrases present minus fraction of forbidden phrases
# present, case-insensitive; in [-1, 1]. An empty list contributes 0.
def phrases(where, args):
    _args(where, args, {"required": False, "forbidden": False})
    for key in ("required", "forbidden"):
        values = args.get(key, [])
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise TrlxError(f"{where}: {key} must be a list of strings")
    required = [p.lower() for p in args.get("required", [])]
    forbidden = [p.lower() for p in args.get("forbidden", [])]
    if not required and not forbidden:
        raise TrlxError(f"{where}: give required or forbidden phrases")

    def reward(completions, **kwargs):
        scores = []
        for completion in completions:
            text = completion_text(completion).lower()
            good = sum(p in text for p in required) / len(required) if required else 0.0
            bad = sum(p in text for p in forbidden) / len(forbidden) if forbidden else 0.0
            scores.append(good - bad)
        return scores

    reward.__name__ = "phrases"
    return reward


# 1 when the whole stripped completion parses as JSON and, with `keys`, is
# an object carrying every listed key. No extraction from surrounding prose:
# a model that wraps JSON in text can be scored with `regex` instead.
def json_valid(where, args):
    _args(where, args, {"keys": False})
    keys = args.get("keys", [])

    def reward(completions, **kwargs):
        scores = []
        for completion in completions:
            try:
                value = json.loads(completion_text(completion).strip())
            except ValueError:
                scores.append(0.0)
                continue
            if keys and not (isinstance(value, dict) and all(k in value for k in keys)):
                scores.append(0.0)
            else:
                scores.append(1.0)
        return scores

    reward.__name__ = "json_valid"
    return reward


# 1 inside [low, high] in tokens or words; outside, a linear falloff that
# reaches 0 one window width away. `tokens` needs `tokenizer`, a path.
def length_window(where, args):
    _args(where, args, {"unit": True, "low": True, "high": True, "tokenizer": False})
    unit, low, high = args["unit"], args["low"], args["high"]
    if unit not in ("tokens", "words"):
        raise TrlxError(f"{where}: unit must be tokens or words, got {unit!r}")
    if not (isinstance(low, int) and isinstance(high, int) and 0 <= low < high):
        raise TrlxError(f"{where}: need integers 0 <= low < high, got low={low!r} high={high!r}")
    if (unit == "tokens") != ("tokenizer" in args):
        raise TrlxError(f"{where}: tokenizer is required with unit = tokens and not accepted otherwise")
    width = high - low
    if unit == "tokens":
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(args["tokenizer"])
        except OSError as e:
            raise TrlxError(f"{where}: cannot load tokenizer '{args['tokenizer']}': {e}")
        count = lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])  # noqa: E731
    else:
        count = lambda text: len(text.split())  # noqa: E731

    def reward(completions, **kwargs):
        scores = []
        for completion in completions:
            n = count(completion_text(completion))
            distance = low - n if n < low else n - high if n > high else 0
            scores.append(max(0.0, 1.0 - distance / width))
        return scores

    reward.__name__ = "length_window"
    return reward


# Scores from an OpenAI-compatible endpoint. The rubric is the system
# message; the user message carries the prompt and the completion. The first
# number in the reply is the score. A reply with no number scores 0 and is
# logged with its text, so a misbehaving judge is visible in log.txt rather
# than fatal mid-run.
def llm_judge(where, args, *, progress=None, rubric_text=None):
    if "rubric" in args:
        raise TrlxError(f"{where}: rubric was removed; save it in a .prompt file and set rubric_file")
    _args(
        where,
        args,
        {"url": True, "model": True, "rubric_file": True, "timeout": True, "retries": True, "concurrency": True,
         "api_key": False, "max_tokens": False},
    )
    # api_key names an environment variable, never holds the key: the run
    # config is snapshotted into every run directory and is the operator's to
    # commit. The variable may come from .env (dataset/env.py) or the real
    # environment.
    api_key = None
    if "api_key" in args and not isinstance(args["api_key"], str):
        raise TrlxError(f"{where}: api_key must name an environment variable")
    if not isinstance(args["rubric_file"], str) or not args["rubric_file"].strip():
        raise TrlxError(f"{where}: rubric_file must name an operator-authored .prompt file")
    # Config preparation supplies the reviewed contents; direct factory callers must load a file too.
    if rubric_text is None:
        try:
            rubric_text = load_prompt(args["rubric_file"], allowed=None)
        except DatasetError as error:
            raise TrlxError(f"{where}: {error}; use llm-judge.prompt.example as authoring guidance") from error
    if args.get("api_key"):
        api_key = os.environ.get(args["api_key"])
        if not api_key:
            raise TrlxError(f"{where}: api_key: environment variable {args['api_key']} is not set")
    try:
        endpoint = Endpoint(args["url"], args["model"], api_key, args["timeout"], args["retries"])
    except DatasetError as e:
        raise TrlxError(f"{where}: {e}")
    rubric, concurrency, max_tokens = rubric_text, args["concurrency"], args.get("max_tokens")
    if not isinstance(concurrency, int) or concurrency < 1:
        raise TrlxError(f"{where}: concurrency must be a positive integer")

    # Keep the worker's reporter for later training batches; resolution finishes
    # before this callable performs any judge requests.
    def reward(completions, prompts=None, **kwargs):
        prompts = prompts if prompts is not None else [""] * len(completions)
        requests = [
            [
                {"role": "system", "content": rubric},
                {"role": "user", "content": f"Prompt:\n{prompt_text(p)}\n\nResponse:\n{completion_text(c)}"},
            ]
            for p, c in zip(prompts, completions)
        ]
        try:
            replies = endpoint.complete_many(requests, concurrency, max_tokens,
                                             progress=progress, label=f"{where} judge requests")
        except DatasetError as e:
            raise TrlxError(f"{where}: {e}")
        scores = []
        for reply in replies:
            m = _NUMBER.search(reply)
            if m is None:
                _log.warning("%s: judge reply has no number, scoring 0: %r", where, reply)
                scores.append(0.0)
            else:
                scores.append(float(m.group()))
        return scores

    reward.__name__ = "llm_judge"
    return reward


BUILTINS = {
    "reference_match": reference_match,
    "regex": regex,
    "phrases": phrases,
    "json_valid": json_valid,
    "length_window": length_window,
    "llm_judge": llm_judge,
}


# Turns config.RewardEntry values into what GRPOTrainer/RLOOTrainer accept:
# callables, or a model path string for TRL to load. Resolution order for a
# bare string: trlx built-in, trl.rewards name, module:function or
# path.py:function, else a model path. trl.rewards factories (get_*) take
# their keyword arguments from the {name, args} form.
def resolve(entries, *, progress=None):
    import trl.rewards

    funcs = []
    for i, entry in enumerate(entries):
        where = f"[rewards].funcs[{i}] '{entry.spec}'"
        name, args = entry.spec, entry.args
        # Only the endpoint-backed factory needs the worker's progress reporter.
        if name == "llm_judge":
            funcs.append(llm_judge(where, args or {}, progress=progress, rubric_text=entry.rubric_text))
        elif name in BUILTINS:
            funcs.append(BUILTINS[name](where, args or {}))
        elif hasattr(trl.rewards, name):
            target = getattr(trl.rewards, name)
            if args is None:
                funcs.append(target)
            else:
                try:
                    funcs.append(target(**args))
                except TypeError as e:
                    raise TrlxError(f"{where}: trl.rewards.{name} rejected args: {e}")
        elif args is not None:
            raise TrlxError(f"{where}: args are only for trlx built-ins or trl.rewards factories")
        elif ":" in name:
            funcs.append(_import_function(where, name))
        else:
            funcs.append(name)
    return funcs


# `module:function` via the import system, `path.py:function` from a file.
def _import_function(where, spec):
    target, _, attr = spec.rpartition(":")
    if target.endswith(".py"):
        path = pathlib.Path(target)
        if not path.is_file():
            raise TrlxError(f"{where}: no such file {target}")
        module_spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(module_spec)
        try:
            module_spec.loader.exec_module(module)
        except Exception as e:  # noqa: BLE001 - operator code; report, do not trace
            raise TrlxError(f"{where}: {target} failed to import: {type(e).__name__}: {e}")
    else:
        try:
            module = importlib.import_module(target)
        except ImportError as e:
            raise TrlxError(f"{where}: cannot import module {target}: {e}")
    func = getattr(module, attr, None)
    if not callable(func):
        raise TrlxError(f"{where}: {target} has no function {attr}")
    return func
