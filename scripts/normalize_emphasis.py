#!/usr/bin/env python
"""Normalize ** markers by pairing them, not by pattern-matching neighbours.

Within a line, ** markers alternate: the first opens, the second closes, and so
on. An opening marker wants a space before it and none after; a closing marker
wants none before and a space after unless punctuation follows. A regex that
looks only at the characters beside a marker cannot tell the two apart, which is
how an earlier pass stripped the spaces before opening markers.

Lines with an odd number of markers are reported and left untouched.
"""
import io
import os
import re
import sys
import tempfile

D = "/home/benja/m2/repos/benjbritton_FA26/posts"
P = os.path.join(D, "Chactun_Multiclass_M2_blog_090226v10.md")

lock = os.path.join(D, ".~lock.Chactun_Multiclass_M2_blog_090226v10.md#")
if os.path.exists(lock):
    sys.exit("v10 is open in an editor -- not overwriting")

lines = io.open(P, encoding="utf-8").read().split("\n")
CLOSE_OK = set(".,;:)!?%—’'\"")

fixed = 0
odd = []
for i, line in enumerate(lines):
    if line.strip().startswith("|") or line.strip().startswith("!["):
        continue
    marks = [m.start() for m in re.finditer(r"\*\*", line)]
    if not marks:
        continue
    if len(marks) % 2:
        odd.append(i + 1)
        continue

    out = []
    prev = 0
    for k, pos in enumerate(marks):
        out.append(line[prev:pos])
        out.append("**")
        prev = pos + 2
    out.append(line[prev:])

    # rebuild, trimming inside and padding outside according to role
    parts = []
    for k in range(len(out)):
        seg = out[k]
        if seg == "**":
            parts.append(seg)
            continue
        parts.append(seg)
    # walk segments: text, **, text, **, text ...
    for k in range(1, len(parts), 2):
        opening = ((k - 1) // 2) % 2 == 0
        before = parts[k - 1]
        after = parts[k + 1] if k + 1 < len(parts) else ""
        if opening:
            if after[:1] in (" ", "\t"):
                after = after.lstrip(" \t")
            if before and not before.endswith((" ", "\t", "(", "[", "—")):
                before = before + " "
        else:
            if before.endswith((" ", "\t")):
                before = before.rstrip(" \t")
            if after and after[0] not in CLOSE_OK and after[0] not in (" ", "\t"):
                after = " " + after
        parts[k - 1] = before
        if k + 1 < len(parts):
            parts[k + 1] = after

    new = "".join(parts)
    new = re.sub(r"(\S)[ \t]{2,}(\S)", r"\1 \2", new)
    if new != line:
        fixed += 1
    lines[i] = new

t = "\n".join(lines)
fd, tmp = tempfile.mkstemp(dir=D, suffix=".tmp")
os.close(fd)
io.open(tmp, "w", encoding="utf-8", newline="\n").write(t)
os.replace(tmp, P)

print("lines normalized: %d" % fixed)
print("lines with an odd marker count (left alone): %s" % (odd or "none"))

print()
print("checks:")
for probe in ("recovers **92%", "gain of **+4.16 AP**",
              "Rotating a **hillshade** would", "by **ground extent**",
              "Band **statistics** were", "1. **No source elevation data.**",
              "characteristics"):
    print("  %-46s %s" % (probe[:44], probe in t))
