# hooks/

Claude Code hooks wired from `.claude/settings.json`. Each reads the tool event on stdin and exits 2 to block.

- **Goes in:** `pre-commit-gate.sh` (Bash → git commit only), `post-write.sh` (Write|Edit)
- **Stays out:** anything that pipes a gate's exit status away
