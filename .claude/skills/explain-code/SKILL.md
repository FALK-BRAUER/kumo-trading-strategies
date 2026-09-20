---
name: explain-code
description: >
  Explains code using analogies, ASCII diagrams, step-by-step walkthroughs, and
  gotcha highlights. Use when the user says "how does this work", "explain this",
  "walk me through", or when onboarding onto unfamiliar code or preparing
  explanations for others.
allowed-tools: Read, Glob, Grep
effort: medium
---

## Explanation Format

Always follow this structure:

1. **Analogy** — Compare the code to something from everyday life. Make it concrete and relatable.
2. **Diagram** — Draw an ASCII art diagram showing flow, structure, or relationships. Keep it readable.
3. **Walkthrough** — Step through the code explaining what each part does and why.
4. **Gotcha** — Highlight one common mistake or misconception about this code.

## Rules

- Tailor depth to the audience. Check the user's profile if available.
- Start high-level, then zoom in. Don't lead with implementation details.
- If the code is complex, break the explanation into layers (data flow, control flow, error handling).
- Use the actual variable/function names from the code in your explanation.
