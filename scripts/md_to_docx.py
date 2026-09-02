#!/usr/bin/env python
"""Convert the post to .docx with headings, tables, lists and figures intact.

Written against the document's own structure rather than by round-tripping HTML,
so headings stay headings, tables stay tables, and the figures are placed at
their referenced positions rather than dropped.
"""
import argparse
import io
import os
import re

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor


INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")


def add_runs(par, text):
    """Render bold, italic and code spans as runs."""
    for piece in INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            par.add_run(piece[2:-2]).bold = True
        elif piece.startswith("*") and piece.endswith("*"):
            par.add_run(piece[1:-1]).italic = True
        elif piece.startswith("`") and piece.endswith("`"):
            r = par.add_run(piece[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(9.5)
        else:
            par.add_run(piece)


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    base = os.path.dirname(os.path.abspath(a.src))
    lines = io.open(a.src, encoding="utf-8").read().split("\n")

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Georgia"
    style.font.size = Pt(11)
    style.paragraph_format.space_after = Pt(10)

    n_tab = n_img = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()

        if not s:
            i += 1
            continue

        if s in ("---", "***", "___"):
            i += 1
            continue

        # image
        m = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", s)
        if m:
            alt, rel = m.group(1), m.group(2)
            path = os.path.join(base, rel)
            if os.path.isfile(path):
                doc.add_picture(path, width=Inches(6.2))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                cap = doc.add_paragraph()
                cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = cap.add_run(alt)
                r.italic = True
                r.font.size = Pt(9)
                r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
                n_img += 1
            i += 1
            continue

        # heading
        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            doc.add_heading(m.group(2).replace("**", ""), level=len(m.group(1)))
            i += 1
            continue

        # table: header row, separator, body
        if s.startswith("|") and i + 1 < len(lines) and \
           re.fullmatch(r"\|(\s*:?-+:?\s*\|)+", lines[i + 1].strip()):
            header = cells(line)
            i += 2
            body = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                body.append(cells(lines[i]))
                i += 1
            tab = doc.add_table(rows=1, cols=len(header))
            tab.style = "Light Grid Accent 1"
            for j, h in enumerate(header):
                c = tab.rows[0].cells[j]
                c.text = ""
                add_runs(c.paragraphs[0], h)
                for run in c.paragraphs[0].runs:
                    run.bold = True
            for row in body:
                r = tab.add_row().cells
                for j, val in enumerate(row[:len(header)]):
                    r[j].text = ""
                    add_runs(r[j].paragraphs[0], val)
            for row in tab.rows:
                for c in row.cells:
                    for par in c.paragraphs:
                        par.paragraph_format.space_after = Pt(2)
                        for run in par.runs:
                            run.font.size = Pt(9)
            doc.add_paragraph()
            n_tab += 1
            continue

        # list
        if re.match(r"^[-*+]\s+", s):
            par = doc.add_paragraph(style="List Bullet")
            add_runs(par, re.sub(r"^[-*+]\s+", "", s))
            i += 1
            continue

        # blockquote
        if s.startswith(">"):
            par = doc.add_paragraph(style="Intense Quote")
            add_runs(par, s.lstrip("> ").strip())
            i += 1
            continue

        # subtitle: the italic line right under the title
        if s.startswith("*") and s.endswith("*") and i < 4:
            par = doc.add_paragraph()
            r = par.add_run(s.strip("*"))
            r.italic = True
            r.font.size = Pt(11.5)
            r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
            i += 1
            continue

        par = doc.add_paragraph()
        add_runs(par, s)
        i += 1

    doc.save(a.out)
    print("tables: %d, images: %d" % (n_tab, n_img))
    print("wrote %s (%.2f MB)" % (a.out, os.path.getsize(a.out) / 1e6))


if __name__ == "__main__":
    main()
