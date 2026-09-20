"""Preference pairs from two messages datasets, or prompt/completion from one.

Alignment key: the tuple of user-turn contents, system messages excluded.
The prompt written out is the chosen side's messages before its final
assistant turn, system message included, in TRL's conversational format.
"""

from dataset.convert import final_assistant_index, messages_to_prompt_completion
from dataset.io import DatasetError
from dataset.progress import stage


# Content may be a string or, for multimodal data, a list; str() gives a
# deterministic key either way.
def _key(messages):
    return tuple(str(m["content"]) for m in messages if m["role"] == "user")


# Validates one row's messages, prefixing any error with the side it came
# from so a message from the shared validator names the right file.
def _validate(row, index, side):
    if "messages" not in row:
        raise DatasetError(f"{side} row {index}: no 'messages' column")
    try:
        return final_assistant_index(row["messages"], index)
    except DatasetError as e:
        raise DatasetError(f"{side} {e}")


# Returns (pairs, unmatched) where unmatched is a list of "side row N"
# strings. Equal keys pair in order of appearance on each side, so duplicate
# keys pair first-with-first.
def align(chosen_rows, rejected_rows, *, progress=None):
    with stage(progress, "aligning preference pairs", total=len(chosen_rows) + len(rejected_rows), unit="rows") as activity:
        rejected_by_key = {}
        for i, row in enumerate(rejected_rows):
            _validate(row, i, "rejected")
            rejected_by_key.setdefault(_key(row["messages"]), []).append((i, row))
            activity.advance()
        pairs, unmatched = [], []
        for i, row in enumerate(chosen_rows):
            last = _validate(row, i, "chosen")
            bucket = rejected_by_key.get(_key(row["messages"]))
            if not bucket:
                unmatched.append(f"chosen row {i}")
                activity.advance()
                continue
            _, other = bucket.pop(0)
            pairs.append(
                {
                    "prompt": row["messages"][:last],
                    "chosen": row["messages"][last:],
                    "rejected": other["messages"][-1:],
                }
            )
            activity.advance()
        for bucket in rejected_by_key.values():
            unmatched.extend(f"rejected row {i}" for i, _ in bucket)
        unmatched.sort(key=lambda s: (s.split()[0], int(s.split()[2])))
        return pairs, unmatched


# One messages dataset to prompt/completion for distillation.
def single(rows, *, progress=None):
    return messages_to_prompt_completion(rows, progress=progress)
