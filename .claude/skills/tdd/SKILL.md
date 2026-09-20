---
name: tdd
description: Test-driven development workflow. Use when implementing new features or fixing bugs where tests should come first.
---

Follow this cycle strictly:
1. **Red** — Write a failing test that describes the desired behavior
2. **Green** — Write the minimum code to make the test pass
3. **Refactor** — Clean up the code while keeping tests green
4. Repeat

Never write implementation code before writing the test. If you catch yourself doing it, stop and write the test first.

## Test Quality Rules
- Test names describe behavior: "should return 404 when user not found"
- One assertion per test when possible
- No test interdependencies — each test stands alone
- Mock external boundaries, not internal modules
