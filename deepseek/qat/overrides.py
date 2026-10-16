"""Recipe overrides shared by the torchrun entry points.

``--set KEY=VALUE`` and ``--unset KEY`` let one recipe file serve every arm of a
comparison, every reload, and every train-on-export run. Kept apart from
``proxy_step`` so ``proxy`` can use it without importing the training script.
"""


# Apply --set KEY=VALUE (YAML scalar) and --unset KEY to a loaded NeMo config
# before the recipe is built. Returns what was applied for the report. Bad
# syntax or an --unset of a key the recipe does not have is a usage error, not
# a silent no-op.
def apply_overrides(cfg, sets: list[str], unsets: list[str]) -> dict:
    import yaml
    from nemo_automodel.components.config.loader import ConfigNode

    applied: dict = {"set": {}, "unset": []}
    for item in sets:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        cfg.set_by_dotted(key, yaml.safe_load(raw))
        applied["set"][key] = raw
    for key in unsets:
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node.__dict__.get(part)
            if not isinstance(node, ConfigNode):
                raise SystemExit(f"--unset {key}: {part!r} is not a section of the recipe")
        if parts[-1] not in node.__dict__:
            raise SystemExit(f"--unset {key}: key not present in the recipe")
        del node.__dict__[parts[-1]]
        applied["unset"].append(key)
    return applied


# Register --set/--unset on an argparse parser so every entry point spells them the same way.
def add_override_arguments(parser) -> None:
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a dotted recipe key with a YAML scalar before building (repeatable)")
    parser.add_argument("--unset", action="append", default=[], metavar="KEY",
                        help="delete a dotted recipe key before building (repeatable)")
