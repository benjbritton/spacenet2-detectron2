#!/usr/bin/env python
"""Render the post to a single self-contained HTML file with the figures inside.

The markdown references its figures by relative path, which a markdown renderer
resolves and a word processor does not. Embedding them as data URIs produces one
file that shows the whole post, images included, in any browser and from any
location.
"""
import argparse
import base64
import io
import os
import re

import markdown

CSS = """
body { max-width: 46rem; margin: 3rem auto; padding: 0 1.5rem;
       font: 16px/1.65 Georgia, 'Times New Roman', serif; color: #1a1a1a; }
h1 { font-size: 1.9rem; line-height: 1.25; margin: 0 0 .4rem; }
h2 { font-size: 1.25rem; margin: 2.4rem 0 .7rem; padding-top: .9rem;
     border-top: 1px solid #ddd; }
h1 + p em { color: #555; font-size: 1.02rem; }
img { max-width: 100%; height: auto; display: block; margin: 1.4rem auto;
      border: 1px solid #ddd; }
table { border-collapse: collapse; width: 100%; margin: 1.3rem 0;
        font: 14px/1.45 -apple-system, Segoe UI, Roboto, sans-serif; }
th, td { border: 1px solid #ddd; padding: .4rem .6rem; text-align: left; }
th { background: #f4f4f4; }
td:not(:first-child), th:not(:first-child) { text-align: right;
     font-variant-numeric: tabular-nums; }
code { background: #f4f4f4; padding: .1rem .3rem; font-size: .9em; }
pre { background: #f7f7f7; padding: .9rem; overflow-x: auto; font-size: .85em; }
blockquote { border-left: 3px solid #ccc; margin: 1.2rem 0; padding: .1rem 1rem;
             color: #444; }
hr { border: 0; border-top: 1px solid #e5e5e5; margin: 2rem 0; }
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    base = os.path.dirname(os.path.abspath(a.src))
    t = io.open(a.src, encoding="utf-8").read()

    # embed every referenced image as a data URI
    n = 0
    def embed(m):
        nonlocal n
        alt, rel = m.group(1), m.group(2)
        path = os.path.join(base, rel)
        if not os.path.isfile(path):
            return m.group(0)
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        n += 1
        ext = os.path.splitext(rel)[1].lstrip(".").lower() or "png"
        return "![%s](data:image/%s;base64,%s)" % (alt, ext, b64)

    t = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", embed, t)

    html = markdown.markdown(t, extensions=["tables", "fenced_code", "sane_lists"])
    title = re.search(r"^#\s+(.+)$", io.open(a.src, encoding="utf-8").read(),
                      re.M)
    title = title.group(1) if title else os.path.basename(a.src)

    doc = ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\n"
           "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
           "<title>%s</title>\n<style>%s</style></head><body>\n%s\n"
           "</body></html>\n" % (title, CSS, html))

    io.open(a.out, "w", encoding="utf-8", newline="\n").write(doc)
    print("embedded %d images" % n)
    print("wrote %s (%.1f MB)" % (a.out, os.path.getsize(a.out) / 1e6))


if __name__ == "__main__":
    main()
