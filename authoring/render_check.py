"""Mandatory rendering and budget check for Context-assisted generation.

Usage: python authoring/render_check.py CONTEXT.json PROMPT.txt
                                        [--system-file FILE] [--reserve N] [--config TOML]

Builds [system?] + Context + final user prompt, posts it to the recorded
tokenization URL with the generation boundary enabled, detokenizes to confirm
that every tool message and the final prompt rendered, and prints the budget
fields the workflow requires. The reserve defaults to generation.sampling
max_tokens. Exit status is nonzero on FAIL.
"""
import argparse
import json
import pathlib
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import config  # noqa: E402


def post(url, body, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def squash(text):
    return " ".join(text.split())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("context")
    ap.add_argument("prompt")
    ap.add_argument("--system-file")
    ap.add_argument("--reserve", type=int, help="reserved generation tokens; default sampling.max_tokens")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = config.load(args.config)
    tok = cfg.tokenization
    reserve = args.reserve or int(cfg.get("generation.sampling.max_tokens"))
    window = cfg.context_window
    system = pathlib.Path(args.system_file).read_text(encoding="utf-8") if args.system_file else ""
    context = json.loads(pathlib.Path(args.context).read_text(encoding="utf-8"))
    prompt = pathlib.Path(args.prompt).read_text(encoding="utf-8")

    messages = ([{"role": "system", "content": system}] if system.strip() else []) + context
    messages.append({"role": "user", "content": prompt})
    # The generation boundary is always enabled for this check, regardless of
    # the counting flags recorded for complete examples.
    resp = post(tok["url"], {"model": cfg.model, "messages": messages,
                             "add_generation_prompt": True,
                             "continue_final_message": bool(tok["continue_final_message"]),
                             "add_special_tokens": bool(tok["add_special_tokens"])})
    count = resp["count"]
    rendered = post(tok["detokenize_url"], {"model": cfg.model, "tokens": resp["tokens"]})["prompt"]
    flat = squash(rendered)

    tool_msgs = [m for m in context if m.get("role") == "tool"]
    missing = [i for i, m in enumerate(tool_msgs)
               if squash(m.get("content") or "")[:80] and squash(m.get("content") or "")[:80] not in flat]
    prompt_ok = squash(prompt)[:80] in flat
    total = count + reserve
    ok = total <= window and not missing and prompt_ok

    print(f"context_messages: {len(context)}")
    print(f"system_included: {bool(system.strip())}")
    print(f"rendered_input_tokens: {count}")
    print(f"reserved_generation_tokens: {reserve}")
    print(f"total_required_tokens: {total}")
    print(f"model_context_tokens: {window}")
    print(f"remaining_headroom: {window - total}")
    print(f"tool_messages: {len(tool_msgs)} rendered: {len(tool_msgs) - len(missing)}"
          + (f" MISSING: {missing}" if missing else ""))
    print(f"final_prompt_rendered: {prompt_ok}")
    print(f"rendered_tail: {rendered[-160:]!r}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
