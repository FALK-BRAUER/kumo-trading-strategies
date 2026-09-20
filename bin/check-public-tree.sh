#!/usr/bin/env bash
# check-public-tree — refuse any tree that holds what a public repository must never hold (#1045).
#
# ONE SCRIPT, TWO TREES. This file is byte-identical in the private platform repo and in the public
# one (the export copies it). In the public tree it runs in CI over everything tracked. In the private
# tree it runs as a pre-commit hook and skips the paths listed in `.public-exclude` — the operating
# record and the private manual, which the export omits. What the export ships is exactly what the
# private check held to the property, so a private commit that passes cannot produce a public export
# that fails.
#
# Enforce the property, do not enumerate the files: every pattern below is a SHAPE (an account id
# shape, a hostname shape, a data-file shape), never a list of paths. A new file that violates the
# property is caught without anyone editing this script. The one list — literal identifiers that have
# no shape, such as a login name — lives OUTSIDE both trees in the file `KUMO_PUBLIC_DENYLIST` names.
#
# Patterns are POSIX ERE (no \b — git grep -E on macOS does not honour it; the first version of this
# script passed a planted account id for exactly that reason). Word edges are spelled out.
#
# Exit non-zero on any hit and print every offending line. Nothing here prints "clean" it did not check.
set -euo pipefail

cd "${1:-$(git rev-parse --show-toplevel)}"

# Pathspecs to scan: everything, minus this script, minus `.public-exclude`'s entries when present.
SPEC=(':!bin/check-public-tree.sh')
if [ -f .public-exclude ]; then
  while IFS= read -r line; do
    case "$line" in ''|'#'*) continue ;; esac
    SPEC+=(":!$line")
  done < .public-exclude
fi

NB='[^A-Za-z0-9_]'   # a non-word character, used as a word edge
fail=0
hit() { echo "check-public-tree: $1" >&2; fail=1; }

# scan PATTERN LABEL — record a hit and print the offending lines.
# THREE OUTCOMES, NOT TWO: git grep exits 0 on a hit, 1 on none, and ≥2 on an ERROR (a fatal pathspec
# from .public-exclude, a bad regex). The first version folded the error into "no hit", so one bad
# exclude line made all five scans silently pass (peer review of #1081).
scan() {
  local out rc
  # `&& rc=0 || rc=$?` rather than `; rc=$?`: under `set -e` a non-zero command substitution aborts the
  # script before rc is read — git grep's "no match" is exit 1, and the first cut died on the first scan.
  out=$(git grep -nE "$1" -- "${SPEC[@]}" 2>&1) && rc=0 || rc=$?
  case $rc in
    0) hit "$2"; echo "$out" >&2 ;;
    1) ;;
    *) hit "scan could not run ($2): $out" ;;
  esac
}
# tracked FILES matching a path regex, honouring the same exclusions
# `2>/dev/null || true` on ls-files: a fatal pathspec must reach the REFUSAL below with its message,
# not abort the script under `set -e` with git's own exit 128 and no line naming the cause.
tracked() { { git ls-files -- "${SPEC[@]}" 2>/dev/null || true; } | grep -E "$1" || true; }

# 1. Broker account / login shapes (IB paper "DU…", IB live "U…", Alpaca account uuid in an env line).
#    The live shape is its own rule: `DUP?` made the D mandatory and a live id U1234567 passed.
scan "(^|$NB)DUP?[0-9]{6,}($NB|$)"                         "IB paper account id shape found:"
scan "(^|$NB)U[0-9]{7,}($NB|$)"                            "IB live account id shape found:"
scan "ACCOUNT_ID=[0-9a-f]{8}-[0-9a-f]{4}-"                  "account uuid assigned in a tracked file:"

# 2. Private network names (tailnet hostnames). `.local` is NOT matched: `*.env.local` is a standard
#    file suffix and the first version of this rule refused every .gitignore in the tree.
scan "(^|$NB)[a-z0-9-]+\.tail[0-9a-f]+\.ts\.net($NB|$)"  "private hostname found:"
#    A path under a home directory names a person's machine; a private address names a LAN.
#    RFC 5737 documentation addresses (192.0.2.x, 198.51.100.x, 203.0.113.x) are what examples use.
scan "/(Users|home)/[a-z][a-z0-9_-]*/"                      "home-directory path found:"
scan "(^|$NB)(192\.168\.[0-9]{1,3}|10\.[0-9]{1,3}\.[0-9]{1,3}|172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3})\.[0-9]{1,3}($NB|$)" "private network address found:"

# 3. Secret-shaped strings (Telegram bot token, Alpaca key ids, private keys)
scan "(^|$NB)[0-9]{8,10}:[A-Za-z0-9_-]{35}($NB|$)"          "telegram bot token shape found:"
scan "(^|$NB)(PK|AK)[A-Z0-9]{16,20}($NB|$)"                 "alpaca key id shape found:"
scan "BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY"                "private key material found:"

# 4. Market data and operating record (redistribution terms do not cover it; the record is private)
# A backtest SET carries its own data (Falk 2026-09-17): `strategies/<name>/backtests/<set>/…` is allowed.
data=$(tracked '\.(csv|parquet|feather|h5)$' | grep -vE '(^|/)tests?/fixtures/|/backtests/[^/]+/' || true)
if [ -n "$data" ]; then hit "market-data file tracked outside a fixtures directory:"; echo "$data" >&2; fi
record=$(tracked '(^|/)(zz_handoffs|FOR_[A-Z]+\.md|RUNBOOK\.md)')
if [ -n "$record" ]; then hit "operating record tracked:"; echo "$record" >&2; fi

# 4b. Symlinks. git grep reads a symlink's TARGET string as nothing — a tracked link to a path under a
#     home directory passes every rule above (peer review of #1083, M2). Refused outright: nothing in
#     these trees needs one, and a link is the one tracked object whose content no scan reads.
links=$(git ls-files -s -- "${SPEC[@]}" 2>/dev/null | awk '$1 == "120000" {print $4}' || true)
if [ -n "$links" ]; then hit "symlink tracked (its target is invisible to every scan above):"; echo "$links" >&2; fi

# 4c. ENCODED CONTENT IS CONTENT. Two production-captured Redis dumps carried the real account ids
#     inside base64 msgpack blobs, and every rule above, trufflehog and gitleaks passed over them
#     (2026-09-14). Every base64 run of 64+ chars in a tracked file is decoded and the bytes are held
#     to the same shapes and the same denylist. Python for the decode; the rules are the ones above.
b64hits=$(git ls-files -z -- "${SPEC[@]}" 2>/dev/null | python3 -c '
import sys, re, base64
deny = []
import os
dl = os.environ.get("KUMO_PUBLIC_DENYLIST")
if dl and os.path.isfile(dl):
    deny = [t.strip().lower().encode() for t in open(dl) if t.strip()]
shapes = [re.compile(p) for p in (
    rb"(?<![A-Za-z0-9_])DUP?[0-9]{6,}(?![A-Za-z0-9_])",
    rb"(?<![A-Za-z0-9_])U[0-9]{7,}(?![A-Za-z0-9_])",
    rb"[a-z0-9-]+\.tail[0-9a-f]+\.ts\.net",
    rb"BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY",
)]
run = re.compile(rb"[A-Za-z0-9+/=]{64,}")
out = []
for path in sys.stdin.buffer.read().split(b"\0"):
    if not path: continue
    try: data = open(path, "rb").read()
    except Exception: continue
    for i, blob in enumerate(run.findall(data)):
        try: raw = base64.b64decode(blob + b"==", validate=False)
        except Exception: continue
        low = raw.lower()
        reasons = [t.decode() for t in deny if t in low]
        reasons += [f"shape {p.pattern.decode()[:24]}" for p in shapes if p.search(raw)]
        if reasons:
            out.append(f"{path.decode()} (base64 run #{i+1}): {", ".join(sorted(set(reasons)))}")
            if len(out) > 40: break
print("\n".join(out))
') || true
if [ -n "$b64hits" ]; then hit "identifier inside BASE64-encoded content (decoded and matched):"; echo "$b64hits" >&2; fi

# 5. Literal identifiers — the project's own list, one per line, kept OUTSIDE both trees.
#    The export sets KUMO_PUBLIC_DENYLIST to a file of literal strings (names, logins, hosts).
#    ABSENT IS NOT CLEAN: when the variable is set and the file is missing, refuse — a check that
#    silently skipped its one non-shape rule would print "0 hits" for a tree it never held to it.
if [ -n "${KUMO_PUBLIC_DENYLIST:-}" ]; then
  if [ ! -f "$KUMO_PUBLIC_DENYLIST" ]; then
    hit "KUMO_PUBLIC_DENYLIST is set to '$KUMO_PUBLIC_DENYLIST' but no such file exists"
  else
    # FILE:LINE ONLY, never the line itself: a refusal in a public CI log that quotes the leaked
    # string is a second channel for it. The shape rules above print their lines because a shape is
    # not the secret; a denylist term IS.
    raw=$(git grep -niFf "$KUMO_PUBLIC_DENYLIST" -- "${SPEC[@]}" 2>&1) && rc=0 || rc=$?
    lines=$(printf '%s\n' "$raw" | cut -d: -f1,2)
    case $rc in
      0) hit "denylist term found at:"; echo "$lines" >&2 ;;
      1) ;;
      *) hit "denylist scan could not run: $lines" ;;
    esac
    # FILENAMES TOO. git grep reads content only; a file NAMED after a login or a host passes it.
    names=$({ git ls-files -- "${SPEC[@]}" 2>/dev/null || true; } | grep -iFf "$KUMO_PUBLIC_DENYLIST" || true)
    if [ -n "$names" ]; then hit "denylist term in a tracked FILENAME:"; echo "$names" >&2; fi
  fi
fi

n=$({ git ls-files -- "${SPEC[@]}" 2>/dev/null || true; } | wc -l | tr -d ' ')
# 0 OF 0 IS NOT CLEAN. An exclusion spelled as a glob (`*.md`) or a tree with nothing tracked leaves
# nothing to scan, and "0 hits" over nothing is a claim about nothing.
if [ "$n" -eq 0 ]; then hit "nothing left to scan after .public-exclude — refusing to call an empty scan clean"; fi
if [ "$fail" -ne 0 ]; then
  echo "check-public-tree: REFUSED" >&2
  exit 1
fi
x=$(( ${#SPEC[@]} - 1 ))
echo "check-public-tree: 10 checks, 0 hits, $n tracked files scanned, $x exclusion(s) from .public-exclude${KUMO_PUBLIC_DENYLIST:+, denylist $(wc -l < "$KUMO_PUBLIC_DENYLIST" | tr -d ' ') term(s)}"
