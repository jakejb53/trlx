"""Enforces the PLAN.md import rule by scanning source, not by importing.

dataset/ imports nothing from trl or trlx. trlx/ may import from dataset only
dataset.io, dataset.endpoint, dataset.env, dataset.progress and dataset.prompts.
"""

import ast
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Modules trlx/ is permitted to import from the dataset package.
ALLOWED_DATASET_IMPORTS = {"dataset.io", "dataset.endpoint", "dataset.env", "dataset.progress", "dataset.prompts"}


# Yields (file, module) for every import statement under a package directory.
# Relative imports are reported as "." so the rule reads them as in-package.
def _imports(package_dir):
    for path in sorted(package_dir.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    yield path, alias.name
            elif isinstance(node, ast.ImportFrom):
                yield path, "." if node.level else node.module


# True when module is name itself or a submodule of it.
def _under(module, name):
    return module == name or module.startswith(name + ".")


class ImportRule(unittest.TestCase):
    def test_dataset_imports_nothing_from_trl_or_trlx(self):
        for path, module in _imports(ROOT / "dataset"):
            for banned in ("trl", "trlx"):
                self.assertFalse(
                    _under(module, banned), f"{path}: imports {module}"
                )

    def test_trlx_imports_only_allowed_dataset_modules(self):
        for path, module in _imports(ROOT / "trlx"):
            if _under(module, "dataset"):
                self.assertIn(
                    module, ALLOWED_DATASET_IMPORTS, f"{path}: imports {module}"
                )
