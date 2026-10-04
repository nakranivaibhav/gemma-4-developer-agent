#!/usr/bin/env bash
# jgrep — grep an append-only journal WITHOUT dumping its prose paragraphs into the
# agent's context. journal.md lines are multi-hundred-char events (p90 ~900 chars,
# max ~2,400); a bare `grep -n pat journal.md` matching a dozen lines costs ~7k
# tokens in ONE tool result, which is what drives premature compaction.
# Prints line numbers + a truncated window so a full line can be pulled deliberately
# with `sed -n 'Np' <journal>`.
# Usage: bash tools/jgrep.sh <journal.md> <pattern> [width=200] [maxhits=25]
set -euo pipefail
J="${1:?usage: jgrep <journal.md> <pattern> [width] [maxhits]}"
PAT="${2:?usage: jgrep <journal.md> <pattern> [width] [maxhits]}"
W="${3:-200}"; N="${4:-25}"
# The pattern is EXTENDED regex (grep -E), where alternation is `|` and `\|` is a
# LITERAL backslash-pipe. A BRE-habit `a\|b` therefore matches nothing and returns a
# silent zero, which reads as "absent from the journal" and is how a closed family gets
# re-proposed. Refuse it rather than answering it. (2026-09-20: cost a false
# "zero journal lines" claim about an experiment with 24 lines.)
case "$PAT" in
  *'\|'*) echo "jgrep: REFUSED — pattern contains '\\|' but this is grep -E (ERE): use 'a|b', not 'a\\|b'." >&2
          echo "jgrep: a '\\|' pattern matches a literal backslash-pipe and would return a FALSE ZERO." >&2
          exit 2 ;;
esac
HITS=$(grep -cE "$PAT" "$J" || true)
if [ "$HITS" -eq 0 ]; then
  echo "--- 0 hit(s) for ERE /$PAT/ — absence here is absence of this REGEX in this FILE only." >&2
fi
grep -nE "$PAT" "$J" | head -"$N" | cut -c1-"$W"
echo "--- $HITS hit(s), showing <=$N, truncated to ${W} chars; sed -n 'Np' $J for a full line ---"
