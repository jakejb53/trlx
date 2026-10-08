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


def load(path=None):
    p = pathlib.Path(path) if path else DEFAULT_PATH
    with open(p, "rb") as f:
        return Config(tomllib.load(f), p)
