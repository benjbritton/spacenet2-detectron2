#!/usr/bin/env python
"""Normalize emphasis markers by pairing them, not by pattern-matching neighbours.

Within a line, markers of a given kind alternate: the first opens, the second
closes, and so on. An opening marker wants a space before it and none after; a
closing marker wants none before and a space after unless punctuation follows.
A regex that looks only at the characters beside a marker cannot tell the two
apart, which is how an earlier pass stripped the spaces before opening markers.

Both ** and * are handled. Running this on ** alone left '*Scientific Data*10'
in the post, because a single-asterisk italic has the same failure mode.

Lines with an odd number of markers are reported and left untouched.
"""
import argparse
import io
import os
import re
import sys
import tempfile

CLOSE_OK = set(".,;:)!?%—’'\"")
OPEN_OK = (" ", "\t", "(", "[", "—")


def normalize(line, mark):
    """Pair up markers of one kind on a line and fix the spacing around them."""
    if mark == "*":
        # a lone asterisk, not part of a ** pair
        spots = [m.start() for m in re.finditer(r"(?<!\*)\*(?!\*)", line)]
    else:
        spots = [m.start() for m in re.finditer(r"\*\*", line)]
    if not spots or len(spots) % 2:
        return line, len(spots) % 2 == 1

    parts, prev = [], 0
    for pos in spots:
        parts.append(line[prev:pos])
        parts.append(mark)
        prev = pos + len(mark)
    parts.append(line[prev:])

    for k in range(1, len(parts), 2):
        opening = ((k - 1) // 2) % 2 == 0
        before = parts[k - 1]
        after = parts[k + 1] if k + 1 < len(parts) else ""
        if opening:
            after = after.lstrip(" \t")
            if before and not before.endswith(OPEN_OK):
                before += " "
        else:
            before = before.rstrip(" \t")
            if after and after[0] not in CLOSE_OK and not after[0].isspace():
                after = " " + after
        parts[k - 1] = before
        if k + 1 < len(parts):
            parts[k + 1] = after

    return re.sub(r"(\S)[ \t]{2,}(\S)", r"\1 \2", "".join(parts)), False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="markdown file to normalize")
    a = ap.parse_args()

    src = os.path.abspath(a.src)
    d, base = os.path.dirname(src), os.path.basename(src)

    lock = os.path.join(d, ".~lock.%s#" % base)
    if os.path.exists(lock):
        sys.exit("%s is open in an editor -- not overwriting" % base)

    lines = io.open(src, encoding="utf-8").read().split("\n")
    fixed, odd = 0, []
    for i, line in enumerate(lines):
        if line.strip().startswith(("|", "![")):
            continue
        new = line
        for mark in ("**", "*"):
            new, is_odd = normalize(new, mark)
            if is_odd:
                odd.append((i + 1, mark))
        if new != line:
            fixed += 1
        lines[i] = new

    t = "\n".join(lines)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    os.close(fd)
    io.open(tmp, "w", encoding="utf-8", newline="\n").write(t)
    os.replace(tmp, src)

    print("lines normalized: %d" % fixed)
    print("odd marker counts, left alone: %s"
          % (", ".join("line %d (%s)" % o for o in odd) or "none"))

    # The pass is idempotent, so a second pass that still changes
    # something means a defect this pairing cannot express. Report that
    # rather than scanning for defect shapes, which cannot tell an opening
    # marker from a stray one and so reports every bold run as a fault.
    again = 0
    for line in t.split(chr(10)):
        if line.strip().startswith(("|", "![")):
            continue
        new_line = line
        for mark in ("**", "*"):
            new_line, _ = normalize(new_line, mark)
        if new_line != line:
            again += 1
    print("unsettled lines after a second pass: %d" % again)
    print("wrote %s" % src)


if __name__ == "__main__":
    main()
