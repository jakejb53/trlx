"""Reuse a hash-verified tool result from an existing Context.

Usage: python authoring/reuse.py CONTEXT.json INDEX OUT.txt

Reads the tool message at INDEX (as printed by `context outline`), splits its
provenance header from its body at the first blank line, recomputes the body's
SHA256, and fails unless the header records that hash. The body is written to
OUT.txt without the header, and the header is printed so its SOURCE, LOCATION,
and RETRIEVED values can be passed to ctxbuild.result() for the new Context.

A tool result whose header records no SHA256 is refused: without a recorded
hash there is nothing to verify against, and the provenance gate requires a
previously verified artifact.
"""
import hashlib
import json
import pathlib
import re
import sys


def verified_body(message):
    """Return (header, body) for a tool message whose body matches its recorded SHA256."""
    assert message.get("role") == "tool", f"message role is {message.get('role')!r}, not 'tool'"
    content = message["content"]
    assert isinstance(content, str) and "\n\n" in content, "tool content has no header/body separator"
    header, body = content.split("\n\n", 1)
    assert header.startswith("SOURCE:"), "tool result does not begin with a SOURCE: header"
    m = re.search(r"SHA256[: ]+([0-9a-f]{64})", header)
    assert m, "header records no SHA256; the artifact cannot be verified for reuse"
    recorded = m.group(1)
    digests = {hashlib.sha256(v.encode("utf-8")).hexdigest() for v in (body, body.rstrip("\n") + "\n")}
    assert recorded in digests, f"body SHA256 does not match the recorded {recorded}"
    return header, body


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    context = json.load(open(sys.argv[1], encoding="utf-8"))
    index = int(sys.argv[2])
    header, body = verified_body(context[index])
    pathlib.Path(sys.argv[3]).write_text(body, encoding="utf-8")
    print(header)
    print(f"verified; wrote {sys.argv[3]}: {len(body)} chars")


if __name__ == "__main__":
    main()
