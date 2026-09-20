"""Conversion between TRL dataset formats: messages and prompt/completion.

File-format conversion needs no code here; read_rows and write_rows dispatch
on extension. This module only reshapes rows.

TRL conversational prompt/completion: `prompt` is a list of messages ending
at the last user turn, `completion` is a list holding the final assistant
turn. Standard prompt/completion: both are strings.
"""

from dataset.io import DatasetError
from dataset.progress import stage

TARGETS = ["messages", "prompt-completion"]


# Validates a messages list and returns the index of its final assistant turn.
def final_assistant_index(messages, index, column="messages"):
    if not isinstance(messages, list) or not messages:
        raise DatasetError(f"row {index}: column '{column}' must be a non-empty list of messages")
    for j, m in enumerate(messages):
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise DatasetError(f"row {index}: {column}[{j}] needs 'role' and 'content'")
    if messages[-1]["role"] != "assistant":
        raise DatasetError(f"row {index}: final message in '{column}' is not an assistant turn")
    return len(messages) - 1


# messages -> prompt/completion (conversational). Other columns are kept.
def messages_to_prompt_completion(rows, *, progress=None):
    with stage(progress, "converting messages to prompt/completion", total=len(rows), unit="rows") as activity:
        out = []
        for i, row in enumerate(rows):
            if "messages" not in row:
                raise DatasetError(f"row {i}: no 'messages' column")
            last = final_assistant_index(row["messages"], i)
            new = {k: v for k, v in row.items() if k != "messages"}
            new["prompt"] = row["messages"][:last]
            new["completion"] = row["messages"][last:]
            out.append(new)
            activity.advance()
        return out


# prompt/completion -> messages. Strings become one user and one assistant
# turn; lists are concatenated. Other columns are kept.
def prompt_completion_to_messages(rows, *, progress=None):
    with stage(progress, "converting prompt/completion to messages", total=len(rows), unit="rows") as activity:
        out = []
        for i, row in enumerate(rows):
            for column in ("prompt", "completion"):
                if column not in row:
                    raise DatasetError(f"row {i}: no '{column}' column")
            prompt, completion = row["prompt"], row["completion"]
            if isinstance(prompt, str) and isinstance(completion, str):
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ]
            elif isinstance(prompt, list) and isinstance(completion, list):
                messages = list(prompt) + list(completion)
            else:
                raise DatasetError(
                    f"row {i}: prompt is {type(prompt).__name__} and completion is "
                    f"{type(completion).__name__}; both must be strings or both lists"
                )
            new = {k: v for k, v in row.items() if k not in ("prompt", "completion")}
            new["messages"] = messages
            out.append(new)
            activity.advance()
        return out


# Dispatches on the requested target. Rows already in the target format are an
# error rather than a no-op, so a wrong --to is never silently accepted.
def convert(rows, to, *, progress=None):
    if to == "messages":
        return prompt_completion_to_messages(rows, progress=progress)
    if to == "prompt-completion":
        return messages_to_prompt_completion(rows, progress=progress)
    raise DatasetError(f"--to must be one of {', '.join(TARGETS)}, got '{to}'")
