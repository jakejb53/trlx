"""Compare two checkpoints shard by shard and tensor by tensor.

Reports every shard whose bytes differ and, within each, the tensors whose
payload ranges differ, plus header differences and non-weight files that
differ. Exits 0 when both trees are byte-identical, 1 otherwise, so a
zero-replacement export can be asserted identical and a trained export can be
asserted to differ only in its reported tensors.

Usage: python -m deepseek.qat.verify_export SOURCE_DIR OTHER_DIR [--expect NAME ...]

With ``--expect``, exit 0 only when the set of differing tensors equals the
given names exactly and nothing else differs. ``@FILE`` anywhere on the command
line expands to that file's lines, one argument per line, so a long replaced
list (``proxy_step --export`` writes one) is passed as ``--expect @FILE``.
"""

import argparse
import hashlib
import os
import sys

from deepseek.qat.export import COPY_BUFFER, read_shard_header


def file_digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(COPY_BUFFER):
            h.update(chunk)
    return h.hexdigest()


# Hash a byte range of an open file.
def _range_digest(f, start: int, length: int) -> str:
    f.seek(start)
    h = hashlib.sha256()
    remaining = length
    while remaining:
        chunk = f.read(min(COPY_BUFFER, remaining))
        if not chunk:
            break
        h.update(chunk)
        remaining -= len(chunk)
    return h.hexdigest()


# Names of tensors whose payload bytes differ between two shards with identical headers.
def differing_tensors(a_path: str, b_path: str) -> tuple[bool, list[str]]:
    a_len, a_hdr = read_shard_header(a_path)
    b_len, b_hdr = read_shard_header(b_path)
    if a_len != b_len or a_hdr != b_hdr:
        return False, []
    names: list[str] = []
    with open(a_path, "rb") as fa, open(b_path, "rb") as fb:
        for name, meta in a_hdr.items():
            start, end = meta["data_offsets"]
            if _range_digest(fa, 8 + a_len + start, end - start) != _range_digest(fb, 8 + b_len + start, end - start):
                names.append(name)
    return True, names


def _walk(root: str) -> list[str]:
    out = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            out.append(os.path.relpath(os.path.join(dirpath, f), root))
    return sorted(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], fromfile_prefix_chars="@")
    parser.add_argument("source")
    parser.add_argument("other")
    parser.add_argument("--expect", nargs="*", default=None, help="tensor names expected to differ, and nothing else")
    args = parser.parse_args(argv)

    a_files = _walk(args.source)
    b_files = _walk(args.other)
    identical = True
    if a_files != b_files:
        identical = False
        print("file lists differ:")
        for f in sorted(set(a_files) ^ set(b_files)):
            print(f"  only in {'source' if f in a_files else 'other'}: {f}")
    changed_tensors: list[str] = []
    # Any difference other than a tensor payload under an identical header:
    # header change, non-weight file change, or file-list change.
    other_differences = a_files != b_files
    for rel in sorted(set(a_files) & set(b_files)):
        a = os.path.join(args.source, rel)
        b = os.path.join(args.other, rel)
        if file_digest(a) == file_digest(b):
            continue
        identical = False
        if rel.endswith(".safetensors"):
            same_header, names = differing_tensors(a, b)
            if not same_header:
                print(f"{rel}: header differs")
                other_differences = True
            else:
                print(f"{rel}: {len(names)} tensors differ")
                for n in names:
                    print(f"  {n}")
                changed_tensors.extend(names)
        else:
            print(f"{rel}: differs")
            other_differences = True
    if args.expect is not None:
        expected = set(args.expect)
        actual = set(changed_tensors)
        if expected == actual and not other_differences:
            print(f"OK: exactly {len(expected)} expected tensors differ")
            return 0
        print(f"MISMATCH: expected {sorted(expected)}, got {sorted(actual)}" + (", plus other differences" if other_differences else ""))
        return 1
    print("identical" if identical else "DIFFERENT")
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
