"""Structural excerpt extraction for Context source material.

Every mode selects content by unique structural identifiers (anchor ids,
heading elements, Markdown ATX headings, PDF outline and page ranges), never by
prose patterns. Each selected id must match exactly one element or the run
fails. Inspect every written excerpt before using it as evidence.

Usage: python authoring/extract.py MODE ARGS...

  rfchtml  FILE OUT SECTION_ID...   RFC Editor htmlized RFCs (pre-based); ids like section-2.2.
                                    Page headers and footers are removed; the text that follows
                                    a removed element is kept.
  v3html   FILE OUT SECTION_ID...   RFC Editor v3 HTML; <section id="section-3.5">. A trailing
                                    "!" selects a section's own text without nested sections.
  htmlid   FILE OUT ID...           generic HTML: the element with that id, or a heading plus its
                                    following siblings up to the next heading of equal or higher level.
  md       FILE OUT HEADING...      Markdown ATX sections by exact heading text, including
                                    subsections; fenced code is skipped when scanning headings.
  pdfinfo  FILE                     print outline entries with page numbers and the page count.
  pdfpages FILE OUT FIRST LAST      1-based inclusive page range text.

Requires the project's "authoring" extra (lxml, pypdf).
"""
import pathlib
import sys

from lxml import etree, html

MARK = "SEC:"
SEP = ""
HEADINGS = ("h1", "h2", "h3", "h4", "h5", "h6")


def _write(out, parts):
    text = "\n\n".join(parts).strip() + "\n"
    pathlib.Path(out).write_text(text, encoding="utf-8")
    print(f"wrote {out}: {len(text)} chars, {len(parts)} part(s)")
    for p in parts:
        lines = [l for l in p.splitlines() if l.strip()]
        print(f"  [{len(p)} chars] first: {lines[0][:90]!r}")
        print(f"              last:  {lines[-1][:90]!r}")


def _remove_keep_tail(el):
    """Remove an element but keep its tail text, which lxml's remove() discards."""
    parent = el.getparent()
    prev = el.getprevious()
    if el.tail:
        if prev is not None:
            prev.tail = (prev.tail or "") + el.tail
        else:
            parent.text = (parent.text or "") + el.tail
    parent.remove(el)


def rfchtml_sections(path):
    """Map section anchor id -> section text for an htmlized RFC."""
    doc = html.parse(str(path))
    body = doc.getroot().body
    for el in body.xpath('//*[@class="grey" or @class="invisible" or @class="noprint"]'):
        _remove_keep_tail(el)
    for a in body.xpath('//a[starts-with(@id,"section-")]'):
        a.text = f"{MARK}{a.get('id')}{SEP}{a.text or ''}"
    text = etree.tostring(body, method="text", encoding="unicode")
    sections = {}
    for chunk in text.split(MARK)[1:]:
        sid, _, rest = chunk.partition(SEP)
        assert sid not in sections, f"duplicate anchor {sid}"
        sections[sid] = rest
    return sections


def rfchtml(path, out, ids):
    sections = rfchtml_sections(path)
    parts = []
    for sid in ids:
        assert sid in sections, f"missing {sid}; have {sorted(sections)[:10]}..."
        parts.append(f"[{sid}]\n" + sections[sid].strip("\n"))
    _write(out, parts)


def v3html(path, out, ids):
    import copy
    doc = html.parse(str(path))
    parts = []
    for sid in ids:
        own_only = sid.endswith("!")
        sid = sid.rstrip("!")
        els = doc.xpath(f'//section[@id="{sid}"]')
        assert len(els) == 1, f"{sid}: {len(els)} matches"
        el = els[0]
        if own_only:
            el = copy.deepcopy(el)
            for child in el.xpath(".//section"):
                child.getparent().remove(child)
        parts.append(f"[{sid}]\n" + " ".join(
            etree.tostring(el, method="text", encoding="unicode").split(" ")).strip())
    _write(out, parts)


def htmlid(path, out, ids):
    doc = html.parse(str(path))
    parts = []
    for sid in ids:
        els = doc.xpath(f'//*[@id="{sid}"]')
        assert len(els) == 1, f"{sid}: {len(els)} matches"
        el = els[0]
        if el.tag in HEADINGS:
            level = int(el.tag[1])
            buf = [etree.tostring(el, method="text", encoding="unicode")]
            for sib in el.itersiblings():
                if sib.tag in HEADINGS and int(sib.tag[1]) <= level:
                    break
                buf.append(etree.tostring(sib, method="text", encoding="unicode"))
            txt = "\n".join(buf)
        else:
            txt = etree.tostring(el, method="text", encoding="unicode")
        parts.append(f"[{sid}]\n" + txt.strip())
    _write(out, parts)


def md_sections(path):
    """Map heading text -> section text (heading through the next heading of equal or higher level)."""
    lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    heads = []
    in_fence = False
    for i, l in enumerate(lines):
        if l.startswith("```"):
            # A one-line fence (open and close on the same line) does not change fence state.
            if l.count("```") < 2:
                in_fence = not in_fence
            continue
        if not in_fence and l.startswith("#"):
            level = len(l) - len(l.lstrip("#"))
            heads.append((i, level, l[level:].strip()))
    sections = {}
    for idx, level, text in heads:
        end = len(lines)
        for j, lv, _ in heads:
            if j > idx and lv <= level:
                end = j
                break
        assert text not in sections, f"duplicate heading {text!r}"
        sections[text] = "\n".join(lines[idx:end]).strip()
    return sections


def md(path, out, headings):
    sections = md_sections(path)
    parts = []
    for h in headings:
        assert h in sections, f"{h!r}: no such heading"
        parts.append(sections[h])
    _write(out, parts)


def _reader(path):
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise SystemExit("PDF modes need pypdf: install the project's 'authoring' extra") from e
    return PdfReader(path)


def pdfinfo(path):
    r = _reader(path)
    print("pages:", len(r.pages))

    def walk(items, depth=0):
        for it in items:
            if isinstance(it, list):
                walk(it, depth + 1)
            else:
                try:
                    pg = r.get_destination_page_number(it) + 1
                except Exception:
                    pg = "?"
                print(f"{'  ' * depth}{it.title!r} -> p{pg}")
    walk(r.outline)


def pdfpages(path, out, first, last):
    r = _reader(path)
    parts = [f"[page {p}]\n" + r.pages[p - 1].extract_text() for p in range(int(first), int(last) + 1)]
    _write(out, parts)


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    mode, args = argv[1], argv[2:]
    if mode == "rfchtml":
        rfchtml(args[0], args[1], args[2:])
    elif mode == "v3html":
        v3html(args[0], args[1], args[2:])
    elif mode == "htmlid":
        htmlid(args[0], args[1], args[2:])
    elif mode == "md":
        md(args[0], args[1], args[2:])
    elif mode == "pdfinfo":
        pdfinfo(args[0])
    elif mode == "pdfpages":
        pdfpages(args[0], args[1], args[2], args[3])
    else:
        raise SystemExit(f"unknown mode {mode}\n{__doc__}")


if __name__ == "__main__":
    main(sys.argv)
