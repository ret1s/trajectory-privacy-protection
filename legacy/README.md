# Legacy / archived prototypes

These files are earlier, superseded prototypes kept for historical reference only.
They are **not maintained** and are not part of the active codebase described in
the root `README.md` / `CLAUDE.md`.

- `simple_app.py` + `templates/simple_index.html` — the very first Flask demo,
  before SocketIO console logging or caching were added. Superseded by
  `web/app.py`.
- `test_app.py` — a throwaway smoke test used to confirm Flask itself was
  working on this machine; not a real test suite.

Because the core algorithm now lives in the top-level `core/` package, running
`simple_app.py` directly from this folder will fail to import it. If you need
to run it, do so from the project root with the root added to `PYTHONPATH`:

```bash
PYTHONPATH=. python legacy/simple_app.py
```
