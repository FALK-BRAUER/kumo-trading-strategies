#!/usr/bin/env bash
# Claude Code hook. Reads the tool event as JSON on stdin. Exit 0 = proceed, exit 2 = block (stderr
# goes back to the model). Never pipe a gate's exit status through anything.
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
EVENT="$(cat)"
field() { printf '%s' "$EVENT" | python3 -c 'import json,sys; e=json.load(sys.stdin); print(e.get("tool_input",{}).get(sys.argv[1],""))' "$1"; }

CMD="$(field command)"
case "$CMD" in git\ commit*|*"&& git commit"*|*"; git commit"*) ;; *) exit 0 ;; esac

# check-public-tree
if ! ( bin/check-public-tree.sh ); then
  echo "commit gate: check-public-tree REFUSED — fix it, do not bypass the hook" >&2
  exit 2
fi

# pytest
if ! ( uv run --no-sync pytest -q -x ); then
  echo "commit gate: pytest REFUSED — fix it, do not bypass the hook" >&2
  exit 2
fi

exit 0
