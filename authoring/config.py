"""Load dataset-authoring.toml for the authoring scripts.

Every authoring script reads its endpoint, model, tokenization and scoring
contracts, and limits from this file so that a different model means editing
the config, not the scripts. Missing keys fail fast with the dotted path.
"""
import pathlib
import tomllib

REPO = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_PATH = REPO / "dataset-authoring.toml"


class Config:
    def __init__(self, data, path):
        self.data = data
        self.path = path

    def get(self, dotted):
        node = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"{self.path}: missing required key {dotted!r}")
            node = node[part]
        return node

    # Convenience accessors for the contracts the scripts share.
    @property
    def tokenization(self):
        return self.get("probes.tokenization")

    @property
    def scoring(self):
        return self.get("probes.scoring")

    @property
    def model(self):
        return self.get("generation.model")

    @property
    def limit(self):
        return int(self.get("max_full_sequence_tokens"))

    @property
    def context_window(self):
        return int(self.get("probes.server_max_model_len"))

    @property
    def allow_answer_only_rows(self):
        value = self.get("allow_answer_only_rows")
        if not isinstance(value, bool):
            raise SystemExit(f"{self.path}: allow_answer_only_rows must be true or false, not {value!r}")
        return value


def load(path=None):
    p = pathlib.Path(path) if path else DEFAULT_PATH
    with open(p, "rb") as f:
        return Config(tomllib.load(f), p)


def read_prompt(path):
    """Read a prompt file as the canonical user text.

    Surrounding newlines are stripped, the same rule read_field applies to
    reasoning and answer: the chat template strips them when rendering, so the
    text generated from, scored, and saved must be the stripped form.
    """
    return pathlib.Path(path).read_text(encoding="utf-8").strip("\n")


def read_field(path):
    """Read an assistant field (reasoning or answer) as the canonical text.

    Surrounding newlines are stripped because the chat template strips them.
    count_score.py and save.py both read fields through this function, so the
    text that is scored and the text that is saved cannot diverge.
    """
    return pathlib.Path(path).read_text(encoding="utf-8").strip("\n")


def read_reasoning(arg, cfg):
    """Read a reasoning argument under the dataset's answer-only policy.

    "-" declares an answer-only row and is the only way to declare one; it is
    accepted only when `allow_answer_only_rows` is true. A reasoning file that
    reads as empty is always an error, because an empty file is more likely a
    mistake than a declaration. save.py and replace_row.py both go through this
    function so the two write paths enforce one policy.
    """
    if arg == "-":
        if not cfg.allow_answer_only_rows:
            raise SystemExit(f"{cfg.path}: allow_answer_only_rows is false; an answer-only row cannot be saved")
        return None
    reasoning = read_field(arg)
    if not reasoning:
        raise SystemExit(f"{arg}: reasoning is empty; pass - to declare an answer-only row")
    return reasoning
