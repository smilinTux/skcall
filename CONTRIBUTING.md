# Contributing

Pull requests are welcome. There is no CLA and no DCO sign-off: by opening a
pull request you agree that your change is released under the repository's
license (AGPL-3.0-or-later).

## Sending a change

1. Fork the repository and branch from `main`.
2. Keep a pull request to one change, with tests for it.
3. Run the tests below; CI runs the same, plus gitleaks.
4. Open the pull request against `main` and say what it changes and why.

Bigger changes (a new backend, a protocol change) are best raised in an
issue first.

## Running the tests

The client, from the repository root:

```bash
uv venv .venv -p 3.12
uv pip install -p .venv/bin/python -e '.[test,voice]' -c sk/constraints.txt
.venv/bin/python -m pytest -q
```

The tests use fake audio, video, speech and lip-sync, and need no network.

Each service has its own venv and a fake engine, so its tests run on CPU
without models:

```bash
for svc in tts stt lipsync; do
  (cd services/$svc &&
   uv venv .venv -p 3.12 &&
   uv pip install -p .venv/bin/python -r requirements-test.txt &&
   .venv/bin/python -m pytest -q)
done
(cd services/common && ../tts/.venv/bin/python -m pytest -q)
```

Secrets scan (the same rules CI uses), on a clean checkout: a local `.venv`
holds third-party files that the default rules flag.

```bash
gitleaks dir . --config .gitleaks.toml
```

## Rules for changes

- **No site values.** No real room tokens, Talk user ids, host names,
  addresses or personal names in code, tests or docs. Use `example.org`,
  `room0001`, `alice` and `bob`, and `192.0.2.x` (TEST-NET-1). Everything a
  site needs comes from the environment.
- **Service interfaces are additive.** `/v1` paths keep working; a breaking
  change gets a `/v2` beside them. The lip-sync wire protocol is fixed: change
  `services/lipsync/README.md` and `skcall/video/lipsync.py` together.
- **Never log a token** or an app password.
