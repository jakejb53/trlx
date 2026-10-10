"""Run one generation request with the saved session settings.

Usage: python authoring/gen.py --prompt-file FILE --out-dir DIR --label LABEL
                               [--context-file CONTEXT.json] [--config TOML]

Builds the `dataset generate` stdin object from [generation] and
[generation.sampling], runs the CLI from the repository root, and stores the
request, stdout, and stderr separately as DIR/LABEL.request.json, DIR/LABEL.out,
and DIR/LABEL.err. The exit code is appended to the .err file and returned.
"""
import argparse
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", required=True, help="basename for the three output files")
    ap.add_argument("--context-file", help="Context JSON passed to generate --context-file")
    ap.add_argument("--config", help="path to dataset-authoring.toml")
    args = ap.parse_args()

    cfg = config.load(args.config)
    gen = cfg.get("generation")
    request = {
        "endpoint": gen["endpoint"], "model": gen["model"],
        "user": config.read_prompt(args.prompt_file),
        "system": gen.get("system", ""),
        "sampling": cfg.get("generation.sampling"),
        "timeout": gen["timeout"], "retries": gen["retries"],
    }
    if gen.get("api_key"):
        request["api_key"] = gen["api_key"]

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.label}.request.json").write_text(json.dumps(request), encoding="utf-8")
    cmd = [sys.executable, "-m", "dataset.cli", "generate"]
    if args.context_file:
        cmd += ["--context-file", args.context_file]
    with open(out / f"{args.label}.out", "wb") as so, open(out / f"{args.label}.err", "wb") as se:
        proc = subprocess.run(cmd, cwd=config.REPO, input=json.dumps(request).encode("utf-8"),
                              stdout=so, stderr=se)
        se.write(f"exit={proc.returncode}\n".encode("utf-8"))
    print(f"{args.label}: exit={proc.returncode} stdout={ (out / f'{args.label}.out').stat().st_size } bytes")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
