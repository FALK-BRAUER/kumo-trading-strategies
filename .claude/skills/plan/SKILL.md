---
name: plan
description: >
  Produces a structured execution plan before writing any code. Breaks tasks into
  ordered steps with file paths, dependencies, and verification criteria. Use when
  the user asks to build a feature, refactor code, design a system, or when any
  task touches 3+ files. Also triggers on ambiguous or open-ended requests.
allowed-tools: Read, Glob, Grep, Bash(find:*), Bash(git log:*), Bash(git diff:*)
effort: high
---

## Core Workflow

1. **Understand** — Restate the task clearly. Confirm constraints. Identify missing information. If unclear, ask one clarifying question — then plan.
2. **Plan** — Produce a structured execution roadmap. List files to create/modify, dependencies between steps, and expected outcomes. Keep it concise but complete.
3. **Execute** — Follow the plan step by step. Mark each step done as you go.
4. **Verify** — Check the result against the original request. Run tests. Confirm nothing was missed.

## Rules

- Always produce the plan BEFORE starting implementation.
- Plans must be concrete — file paths, function names, specific changes — not vague intentions.
- If the plan changes during execution, update it and note why.
- For large plans, break into phases that can each be verified independently.
- Save the plan to `.claude/plans/{feature-name}.md` if it spans multiple sessions.
