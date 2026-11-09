#!/bin/sh
# Self-check for scripts/hooks/commit-msg: exits non-zero on failure.
set -e
f=$(mktemp)
printf 'Subject line\n\nBody keeps Co-Authored-By: a human <x@y.z>\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nco-authored-by: Claude <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/abc\n\xf0\x9f\xa4\x96 Generated with [Claude Code](https://claude.com/claude-code)\n\n' > "$f"
sh "$(dirname "$0")/commit-msg" "$f"
want=$(printf 'Subject line\n\nBody keeps Co-Authored-By: a human <x@y.z>')
[ "$(cat "$f")" = "$want" ] || { echo "FAIL:"; cat "$f"; rm -f "$f"; exit 1; }
rm -f "$f"
echo "commit-msg hook: ok"
