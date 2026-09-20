---
name: dependency-audit
description: Check for outdated, vulnerable, or unused dependencies. Use periodically or before releases.
---

## Workflow

### 1. Check for Vulnerabilities
```bash
npm audit
```
Report critical and high severity issues. Suggest fixes.

### 2. Check for Outdated Packages
```bash
npm outdated
```
Categorize: patch updates (safe), minor updates (review changelog), major updates (breaking changes — plan migration).

### 3. Check for Unused Dependencies
Scan import statements across the codebase. Flag packages in package.json that are never imported.

### 4. Check for Duplicate Packages
Review package-lock.json for multiple versions of the same package.

### 5. Report
- Vulnerabilities: list with severity and fix commands
- Outdated: table with current vs latest version
- Unused: list with recommendation to remove
- Duplicates: list with dedup suggestions
