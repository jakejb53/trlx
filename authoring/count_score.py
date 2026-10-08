"""Render, count, and score one training example under the saved probes.

Usage: python authoring/count_score.py PROMPT.txt REASONING.txt|- ANSWER.txt
                                       [--score] [--config TOML]

Rendering uses the [probes.tokenization] counting flags with the reasoning
mapped into the assistant message's recorded reasoning field; pass "-" for an
answer-only example. --score posts the detokenized rendering to the scoring
URL with the recorded [probes.scoring] settings and reports mean log-probability
over the reasoning span, the answer span, and both.

Span rules: the chat template strips surrounding newlines from both fields, so
spans are located on the stripped text between the recorded reasoning_open and
reasoning_close boundaries and up to assistant_end. Boundary tokens are
excluded. A token with a null log-probability (the first token on this server)
is excluded from every mask. A fixed-context comparison that isolates an
answer-wording change is obtained by passing the untouched reasoning with the
edited answer.
"""
import argparse
import json
import pathlib
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import config  # noqa: E402


def post(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def render(cfg, prompt, reasoning, answer):
    tok = cfg.tokenization
    assistant = {"role": "assistant", "content": answer}
    if reasoning is not None:
        assistant[tok["reasoning_field"]] = reasoning
    messages = [{"role": "user", "content": prompt}, assistant]
    resp = post(tok["url"], {"model": cfg.model, "messages": messages,
                             "add_generation_prompt": bool(tok["add_generation_prompt"]),
                             "continue_final_message": bool(tok["continue_final_message"]),
                             "add_special_tokens": bool(tok["add_special_tokens"])})
    rendered = post(tok["detokenize_url"], {"model": cfg.model, "tokens": resp["tokens"]})["prompt"]
    return resp["count"], rendered


def spans(rendered, reasoning, answer, r_open, r_close, a_end):
    """Character spans [start, end) of the reasoning text and the answer text.

    Both fields are matched after stripping surrounding newlines, which is what
    the template renders. Each block must occur exactly once.
    """
    out = {}
    reasoning = reasoning.strip("\n") if reasoning else reasoning
    answer = answer.strip("\n")
    if reasoning:
        key = r_open + reasoning + r_close
        assert rendered.count(key) == 1, "reasoning block not found exactly once in the rendering"
        s = rendered.index(key) + len(r_open)
        out["reasoning"] = (s, s + len(reasoning))
    if answer:
        key = answer + a_end
        assert rendered.count(key) == 1, "answer block not found exactly once in the rendering"
        s = rendered.rindex(key)
        out["answer"] = (s, s + len(answer))
    return out


def score(cfg, rendered, sp):
    sc = cfg.scoring
    body = {"model": cfg.model, "prompt": rendered, "echo": bool(sc["echo"]),
            "max_tokens": int(sc["max_tokens"]), "logprobs": int(sc["logprobs"]),
            "add_special_tokens": bool(sc["add_special_tokens"])}
    resp = post(sc["url"], body)
    lp = resp["choices"][0]["logprobs"]
    toks, lps, offs = lp["tokens"], lp["token_logprobs"], lp["text_offset"]
    assert len(toks) == len(lps) == len(offs)
    res, masks = {}, {}
    for name, (s, e) in sp.items():
        vals = [l for l, o in zip(lps, offs) if l is not None and s <= o < e]
        res[name] = (sum(vals) / len(vals) if vals else float("nan"), len(vals))
        masks[name] = vals
    if "reasoning" in masks and "answer" in masks:
        both = masks["reasoning"] + masks["answer"]
        res["both"] = (sum(both) / len(both), len(both))
    return res, len(toks)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("reasoning", help="reasoning file, or - for an answer-only example")
    ap.add_argument("answer")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = config.load(args.config)
    tok = cfg.tokenization
    prompt = pathlib.Path(args.prompt).read_text(encoding="utf-8")
    reasoning = None if args.reasoning == "-" else pathlib.Path(args.reasoning).read_text(encoding="utf-8")
    answer = pathlib.Path(args.answer).read_text(encoding="utf-8")

    count, rendered = render(cfg, prompt, reasoning, answer)
    limit = cfg.limit
    print(f"rendered_tokens: {count} (limit {limit}, {'FITS' if count <= limit else 'OVER by ' + str(count - limit)})")
    r_open, r_close, a_end = tok["reasoning_open"], tok["reasoning_close"], tok["assistant_end"]
    print(f"reasoning_rendered: {bool(reasoning) and (r_open + reasoning.strip(chr(10)) + r_close) in rendered}")
    if args.score:
        sp = spans(rendered, reasoning, answer, r_open, r_close, a_end)
        res, n = score(cfg, rendered, sp)
        print(f"scored_prompt_tokens: {n}")
        for k, (m, c) in res.items():
            print(f"mean_logprob[{k}]: {m:.4f} over {c} tokens")


if __name__ == "__main__":
    main()
