"""Assemble a Context through the repository's builder CLI.

Import from a build script:

    from authoring.ctxbuild import Builder, result, internal
    b = Builder("topic-name", workdir="/path/to/scratch")
    b.create()
    b.user("...")
    b.tool("read_rfc_sections", {"rfc": "6960", "sections": ["2.2"]},
           result(source, location, representation, verified_excerpt_text))
    b.assistant("...")
    b.description("Two or three sentences for contexts/topic-name.txt")
    b.finish()   # outline + validate

Content files are written under workdir, then builder commands run from the
repository root. result() composes a tool result as the provenance header used
in contexts/ (SOURCE, LOCATION, RETRIEVED, REPRESENTATION) plus the verified
excerpt; the only transformation applied to excerpt text is collapsing runs of
three or more newlines to two. The provenance gate in DATASET-AUTHORING.md
still applies: retrieve and inspect before calling tool().
"""
import datetime
import json
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent


def collapse(text):
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def result(source, location, representation, body, extra=None, retrieved=None):
    retrieved = retrieved or datetime.date.today().isoformat()
    lines = [f"SOURCE: {source}", f"LOCATION: {location}", f"RETRIEVED: {retrieved}",
             f"REPRESENTATION: {representation}"]
    if extra:
        lines.append(extra)
    return "\n".join(lines) + "\n\n" + collapse(body)


def internal(body):
    return "REVIEWED INTERNAL ENGINEERING CONTRACT\n\n" + collapse(body)


class Builder:
    def __init__(self, name, workdir, repo=REPO, python=sys.executable):
        self.name = name
        self.repo = pathlib.Path(repo)
        self.python = python
        self.ctx = f"contexts/{name}.json"
        self.work = pathlib.Path(workdir) / name
        self.work.mkdir(parents=True, exist_ok=True)
        self.n = 0

    def _run(self, *args):
        cmd = [self.python, "-m", "dataset.cli", "context", *args]
        proc = subprocess.run(cmd, cwd=self.repo, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"FAILED: {' '.join(cmd)}\n{proc.stderr}")
        return proc

    def _file(self, label, text):
        self.n += 1
        p = self.work / f"{self.n:02d}-{label}.txt"
        p.write_text(text, encoding="utf-8")
        return str(p)

    def create(self):
        self._run("create", self.ctx)

    def user(self, text):
        self._run("add", self.ctx, "--role", "user", "--content-file", self._file("user", text))

    def assistant(self, text):
        self._run("add", self.ctx, "--role", "assistant", "--content-file", self._file("assistant", text))

    def tool(self, name, args, content):
        self.n += 1
        a = self.work / f"{self.n:02d}-args.json"
        a.write_text(json.dumps(args), encoding="utf-8")
        self._run("tool", self.ctx, "--name", name, "--arguments-file", str(a),
                  "--content-file", self._file("tool", content))

    def description(self, text):
        (self.repo / "contexts" / f"{self.name}.txt").write_text(text.strip() + "\n", encoding="utf-8")

    def finish(self):
        print(self._run("outline", self.ctx).stdout)
        val = self._run("validate", self.ctx)
        print(val.stdout or val.stderr)
