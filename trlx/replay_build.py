"""`trlx replay-build`: samples a model on prompts into a `messages` dataset
(SPEC 2.1), the input a [replay] block mixes into SFT.

Prompts come from generate.prompts_from: a `prompt` column as is, a
`messages` column cut after its last user turn. Each output row is the
prompt's turns plus one assistant turn holding the sample. Locally the model
is loaded as standalone verify loads a base (its own dtype, device_map
"auto") and decoded greedily by generate.generate; with --endpoint the same
prompts go to any OpenAI-compatible server (SPEC 2.10) through
dataset.endpoint. --max-tokens bounds the sample in both modes; it is
operational config, so it has no default.
"""

import os

from dataset.endpoint import DatasetError, Endpoint
from dataset.io import validate_rows_output, write_rows
from trlx import TrlxError, config as config_mod, generate, model as model_mod

# Flags that only mean something with --endpoint; the first three are
# required there, --api-key is optional.
ENDPOINT_REQUIRED = ("timeout", "retries", "concurrency")
ENDPOINT_FLAGS = ENDPOINT_REQUIRED + ("api_key",)


# Entry point for the subcommand. `args` is the argparse namespace.
def run(args, *, progress=None):
    _check_flags(args)
    try:
        validate_rows_output(args.out, args.force)
    except DatasetError as e:
        raise TrlxError(str(e)) from e
    ref = config_mod.dataset_ref("trlx replay-build", "--prompts", args.prompts)
    prompts = generate.prompts_from(ref, progress=progress)
    if args.endpoint is not None:
        replies = _from_endpoint(args, prompts, progress=progress)
    else:
        replies = _from_local(args, prompts, progress=progress)
    rows = [{"messages": _turns(p) + [{"role": "assistant", "content": r}]} for p, r in zip(prompts, replies)]
    # Prompts and completions are fully materialized before replacing either input.
    inputs = [args.prompts] if ref.is_file else ()
    try:
        write_rows(args.out, rows, inputs, force=args.force, no_staging=args.no_staging, progress=progress)
    except DatasetError as e:
        raise TrlxError(str(e))
    empty = sum(not r for r in replies)
    line = f"wrote {len(rows)} rows to {args.out}"
    if empty:
        # Kept, not dropped: the operator decides what an empty sample means.
        line += f"; {empty} with an empty assistant turn"
    print(line, flush=True)


# The endpoint flags are required with --endpoint and rejected without it,
# so a flag that would be silently ignored is an error instead.
def _check_flags(args):
    if args.max_tokens <= 0:
        raise TrlxError(f"--max-tokens must be positive, got {args.max_tokens}")
    given = [f for f in ENDPOINT_FLAGS if getattr(args, f) is not None]
    if args.endpoint is None:
        if given:
            raise TrlxError(f"--{given[0].replace('_', '-')} requires endpoint generation; "
                            "add --endpoint URL for an OpenAI-compatible server, or remove "
                            "endpoint-only flags for local generation")
        return
    missing = [f for f in ENDPOINT_REQUIRED if getattr(args, f) is None]
    if missing:
        raise TrlxError("--endpoint requires " + ", ".join("--" + field.replace("_", "-") for field in missing)
                        + "; supply a positive --timeout in seconds, nonnegative --retries, "
                        "and positive --concurrency")


# A prompt as chat turns: a string is one user turn, a messages list is
# taken as is.
def _turns(prompt):
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return list(prompt)


# Samples from the local model. dtype "auto" and device_map "auto" as
# verify loads a base outside a run; greedy decoding for reproducible rows.
def _from_local(args, prompts, *, progress=None):
    spec = config_mod.ModelSpec(path=args.model, dtype="auto", trust_remote_code=None, attn_implementation=None)
    model = model_mod.load_model(spec, model_mod.CAUSAL, device_map="auto", progress=progress)
    processor = model_mod.load_processor(spec, progress=progress)
    print(f"loaded {type(model).__name__} from {args.model}; sampling {len(prompts)} prompts", flush=True)
    return generate.generate(model, processor, prompts, max_new_tokens=args.max_tokens, progress=progress)


# Samples from an OpenAI-compatible endpoint. The key is read from the
# environment variable named by --api-key, as dataset chat does, so it never
# appears on a command line.
def _from_endpoint(args, prompts, *, progress=None):
    api_key = None
    if args.api_key:
        api_key = os.environ.get(args.api_key)
        if not api_key:
            raise TrlxError("--api-key names an unset or empty environment variable; "
                            "set that variable to the credential, or pass the name of a populated "
                            "variable; do not pass the credential itself")
    try:
        endpoint = Endpoint(args.endpoint, args.model, api_key, args.timeout, args.retries)
        print(f"sampling {len(prompts)} prompts from {args.model} at {endpoint.display_url}", flush=True)
        return endpoint.complete_many([_turns(p) for p in prompts], args.concurrency, args.max_tokens,
                                      progress=progress, label="replay generation")
    except DatasetError as e:
        raise TrlxError(str(e))
