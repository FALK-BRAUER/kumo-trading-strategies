# Contributing

Contributions are welcome. This document is the whole process; there is no other.

## The path a change takes

1. **Ticket.** Every change starts as an issue that states the defect or the goal in one sentence,
   with the measurement that shows it (a failing command, a wrong number, a missing behaviour).
2. **Red test, seen red.** Write the test that fails because of the defect. Run it. Watch it fail.
   A test that has not been seen failing is an assertion about nothing. Quote the failure in the PR.
3. **Scope.** State in the ticket what the change will and will not touch. Scope creep is a second
   ticket, not a bigger PR.
4. **Implement.** Match the surrounding code. Comments say *why*, in English.
5. **Review.** Every PR gets a review by someone who did not write it. Findings are one line each:
   location, problem, fix.
6. **PR through the merge gate.** the `merge-gate` CI check is the required status. It refuses a stale head, a red suite, and a head that moved after
   the suite ran. Nothing merges around it.

## Rules that apply to every change

- Check NautilusTrader first. If Nautilus already has the mechanism, use it; write out-of-tree code
  only when it does not.
- Every bug gets a test that reproduces it before it is fixed.
- No fallbacks that hide an absent value. A fallback is a silent wrong answer.
- No secrets, account ids, hostnames or personal identifiers in the tree. `bin/check-public-tree.sh`
  runs as a pre-commit hook and in CI and refuses them.
- Every non-hidden directory has a `README.md` (3–5 lines: what it holds, what goes in, what does not).

## Licensing of contributions

This project is licensed under the LGPL-3.0-or-later. By submitting a contribution you agree that it
is licensed under the same terms — inbound = outbound. There is no contributor license agreement.

## Reporting a security issue

Open a private security advisory on the repository rather than a public issue.
