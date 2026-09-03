#!/bin/bash
# Audit what a push would make public. Reports only; changes nothing.
#
# Run this before the first push of a repository, and again whenever a lot has
# changed since the last clean run. Every check exists because the thing it
# looks for is invisible in a diff: an address in a file nobody reads, an
# author line on one commit out of sixty, a default emitted by a tool that
# nobody chose.
#
#   scripts/prepush_audit.sh            # offline checks
#   scripts/prepush_audit.sh --links    # also resolve outbound URLs
#
# Exit status is 1 if any check found something, so it can gate a push.
#
# Deliberately not hardcoded here: the addresses to look for. Naming a personal
# address in a tracked file publishes it, which is the thing this script exists
# to prevent. Check 2 reports every address it finds except GitHub noreply, and
# leaves the judgement to the reader.

set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "not in a git repository"; exit 2; }

SELF=scripts/prepush_audit.sh   # its own regexes name what it hunts for
EX=":(exclude)$SELF"

CHECK_LINKS=0
[ "${1:-}" = "--links" ] && CHECK_LINKS=1

fail=0
hdr() { printf "\n=== %s ===\n" "$1"; }
ok()  { echo "  ok"; }
bad() { echo "$1" | sed 's/^/  !! /'; fail=1; }
note(){ echo "$1" | sed 's/^/  .. /'; }

# The commits a push would send. With no upstream, everything is new.
if git rev-parse '@{u}' >/dev/null 2>&1; then
  RANGE='@{u}..HEAD'
  SCOPE="unpushed commits ($(git rev-list --count '@{u}..HEAD'))"
else
  RANGE='HEAD'
  SCOPE="all commits, no upstream configured ($(git rev-list --count HEAD))"
fi
echo "auditing $SCOPE"

hdr "1. commit authorship"
# One personal address on one commit is enough to publish it, and rebasing it
# out afterwards rewrites every hash downstream of it.
a=$(git log "$RANGE" --format='%an <%ae>%n%cn <%ce>' | sort -u)
note "$a"
echo "$a" | grep -qv 'noreply.github.com' && bad "an address above is not a GitHub noreply address"

hdr "2. addresses in tracked content"
r=$(git grep -hoIE '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' -- . "$EX" 2>/dev/null \
    | grep -v 'noreply.github.com' | sort -u)
[ -n "$r" ] && note "$r" || ok

hdr "3. tool attribution in commit messages"
# The work is the author's. A trailer nobody asked for claims otherwise.
r=$(git log "$RANGE" --format='%h %s%n%b' \
    | grep -inE 'co-authored-by|\bclaude\b|\bchatgpt\b|\bcopilot\b|\banthropic\b|\bopenai\b')
[ -n "$r" ] && bad "$r" || ok

hdr "4. tool attribution in tracked files"
r=$(git grep -nIiE 'co-authored-by|\bclaude\b|\bchatgpt\b|\bcopilot\b|\banthropic\b' -- . "$EX" 2>/dev/null)
[ -n "$r" ] && bad "$r" || ok

hdr "5. credentials and key material"
r=$(git grep -nIE '(api[_-]?key|secret|passwd|password|token)[^A-Za-z0-9]{1,4}[A-Za-z0-9/+_-]{16,}' -- . "$EX" 2>/dev/null)
[ -n "$r" ] && bad "$r" || ok
r=$(git ls-files | grep -E '[.](pem|key|p12|pfx|keystore)$|(^|/)[.](netrc|env|npmrc|pypirc)$')
[ -n "$r" ] && bad "$r" || ok
r=$(git grep -lI -e 'BEGIN RSA PRIVATE KEY' -e 'BEGIN OPENSSH PRIVATE KEY' \
       -e 'BEGIN PRIVATE KEY' -e 'aws_secret_access_key' -- . "$EX" 2>/dev/null)
[ -n "$r" ] && bad "$r" || ok

hdr "6. machine-specific absolute paths"
# A path under someone's home directory is a script that runs on one
# machine. In documentation the same path is usually quoted evidence, so
# it is reported for reading rather than treated as a fault.
r=$(git grep -nIE '/(home|Users)/[A-Za-z0-9._-]+/|[A-Z]:.Users.' -- . "$EX" 2>/dev/null)
CODE='^[^:]*[.](sh|py|ya?ml|json|cfg|ini|toml|bash)[^:]*:|^[^:]*Dockerfile[^:]*:'
c=$(printf "%s" "$r" | grep -E "$CODE")
d=$(printf "%s" "$r" | grep -vE "$CODE")
[ -n "$c" ] && bad "$c"
[ -n "$d" ] && note "$d"
[ -z "$r" ] && ok

hdr "7. editor and build litter"
r=$(git ls-files | grep -E '[.]~lock|__pycache__|[.]ipynb_checkpoints|[.]pyc$|[.]DS_Store|[.]swp$|~$')
[ -n "$r" ] && bad "$r" || ok

hdr "8. large files"
r=$(git ls-files -z | xargs -0 -I{} sh -c \
      'test -f "{}" && s=$(stat -c%s "{}" 2>/dev/null) && [ "$s" -gt 5242880 ] &&
       printf "  %5s MB  %s\n" "$(awk "BEGIN{printf \"%.1f\", $s/1048576}")" "{}"' 2>/dev/null)
[ -n "$r" ] && echo "$r" || ok

hdr "9. working tree clean"
r=$(git status --porcelain)
[ -n "$r" ] && note "$r" || ok

hdr "10. tracked tree"
printf "  %s files, %s\n" "$(git ls-files | wc -l)" \
  "$(git ls-files -z | du -ch --files0-from=- 2>/dev/null | tail -1 | cut -f1)"

if [ "$CHECK_LINKS" -eq 1 ]; then
  hdr "11. outbound links resolve"
  # A 200 means the URL answers, not that an anonymous reader can see the
  # content: a single-page app returns its shell either way. Check visibility
  # of anything permission-gated by opening it while logged out.
  urls=$(git grep -hoIE 'https?://[^ )"<>]+' -- . "$EX" 2>/dev/null \
         | sed 's/[.,;:]$//' | grep -viE 'schemas[.]|w3[.]org|purl[.]org' | sort -u)
  for u in $urls; do
    code=$(curl -s -o /dev/null -w '%{http_code}' -L --max-time 25 "$u" 2>/dev/null)
    printf "  %-4s %s\n" "${code:-ERR}" "$u"
    case "${code:-000}" in 2*|3*) ;; *) fail=1 ;; esac
  done
fi

printf "\n"
if [ "$fail" -eq 0 ]; then echo "AUDIT CLEAN"; else echo "AUDIT FOUND ISSUES (marked !!)"; fi
exit "$fail"
