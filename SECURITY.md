# Security

## Reporting a vulnerability

Report it privately through GitHub: open the repository's **Security** tab
and choose **Report a vulnerability**. Please do not open a public issue or
pull request for it.

Say what is affected (the client, or which service), the version or commit,
how to reproduce it, and what an attacker gains. We will confirm that we got
the report, keep you posted while we work on a fix, and credit you in the
advisory unless you ask us not to.

## Scope

In scope: the skcall client (`skcall/`), the services under `services/`, the
setup scripts and the example units.

Things worth knowing when you assess a report:

- skcall joins only rooms that are listed in `SKCALL_ALLOWED_ROOMS`, or a
  one-to-one or two-member room with an operator. A way to make it join, or
  act, in any other room is in scope.
- Tools and Hermes escalation run only for operators. A way for a
  non-operator in a call to run them is in scope.
- The services check per-client bearer tokens and never log a token. A way
  around either is in scope.
- The services bind to `127.0.0.1` by default. Exposing them on an untrusted
  network without tokens is a deployment choice, not a vulnerability.

Out of scope: Nextcloud, Talk, the HPB, the TURN relay and the speech or LLM
engines themselves. Report those to their own projects.
