---
name: review-security
description: >
  Runs a security-focused code review against the OWASP Top 10. Checks input
  validation, auth boundaries, secret exposure, injection, XSS, CSRF, dependency
  vulnerabilities, error leakage, rate limiting, and logging hygiene. Use when
  reviewing PRs, before deployment, or when working with authentication, payments,
  or user data.
allowed-tools: Read, Glob, Grep, Bash(npm audit:*), Bash(npx semgrep:*), Bash(npx trufflehog:*), Bash(git diff:*)
effort: high
---

## Security Checklist

Check each item. Report findings as:
- **Pass** — no issues found
- **Warning** — potential risk, should be addressed
- **Critical** — security vulnerability, must fix before merge/deploy

### The Checks

1. **Input validation** — All user inputs validated with Zod or equivalent? Check API routes, form handlers, query params.
2. **Auth boundaries** — Protected routes actually protected? Can unauthenticated users reach them?
3. **Secrets** — Any API keys, passwords, or tokens in code or committed files? Check `.env` isn't committed.
4. **SQL injection** — Parameterized queries / ORM only? No string concatenation in queries?
5. **XSS** — User-generated content sanitized before rendering? Using `dangerouslySetInnerHTML`?
6. **CSRF** — State-changing endpoints protected with CSRF tokens?
7. **Dependencies** — Run `npm audit`. Any known vulnerabilities?
8. **Error handling** — Error messages leak internal details (stack traces, DB structure, file paths)?
9. **Rate limiting** — Public endpoints rate-limited? Login, signup, password reset especially.
10. **Logging** — Sensitive values (passwords, tokens, PII) excluded from logs?

## Output Format

```
## Security Review: {scope}

### Summary
{X} pass | {Y} warnings | {Z} critical

### Findings
1. [CRITICAL] ... — {file:line} — {what's wrong and how to fix it}
2. [WARNING] ... — {file:line} — {risk and recommendation}
```

## Rules

- Don't report style issues — this is security only.
- If you find a critical issue, stop and report it immediately.
- Run `npm audit` and `npx trufflehog filesystem . --no-update --fail` as automated checks.
