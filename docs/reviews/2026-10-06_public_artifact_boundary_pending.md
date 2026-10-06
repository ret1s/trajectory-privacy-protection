# Public-artifact boundary: interim snapshot

**PASS for the declared Git-visible NEW/untracked inventory at07:13 UTC,
06/10/2026. Final fresh-generation audit remains pending.** Ignored local
endpoint/private files and existing tracked modifications were excluded.
Nothing in a sealed source or evidence file was edited by this audit.

| Check | Sanitized count / result |
|---|---:|
| Scanned regular Git-visible NEW files |1,361|
| Public file bytes |171,722,873|
| Gzip files / expanded bytes checked |476 /989,262,799|
| Local private work roots / key files inspected |9 /124|
| Distinct actual32byte keys / raw, hex and base64 variants |48 /300|
| Actual `.key`/database files or SQLite magic headers exported |0 / PASS|
| Actual private-state path components exported |0 / PASS|
| Exact secret material matches in public bytes / expanded gzip |0 /0 / PASS|
| Allowed non-secret private-storage path references in provenance/docs |5|
| Decode/nonregular errors / selected files changed during scan |0 /0|
| New visible files created while scan ran |16; not part of this snapshot|
| Files above the configured Git100MiB gate |0 / PASS|

Largest NEW files: development v3 `attack_selection.json` is7.056MiB;
development v2's counterpart is7.046MiB. The next largest file is the static
catalogue control's compressed utility rows at2.906MiB. Full sanitized largest
file inventory and scanner source hash are in
[`2026-10-06_public_artifact_boundary_pending.json`](2026-10-06_public_artifact_boundary_pending.json).

Private key bytes, key digests and private key filenames were never printed
or saved in this note/report. Allowed local path references are reproduction
metadata, not actual key/state files. Matching covers exact raw32byte secrets,
lower/upper hex, standard/URL-safe base64 and unpadded variants, including gzip
expansion. This does not prove absence of encrypted/fragmented/arbitrarily
encoded or unrelated secrets. Rerun on final fresh outputs and completed local
key set before publication/commit.

The independent scanner's focused tests plus repository-layout guard passed
**15 tests**, including deliberately planted raw/encoded secrets in gzip,
database/state files, ignored-file scope, oversized-file and malformed-gzip
failures. Root reported canonical QA for the current iteration: **893 passed,
9 integration skips**, plus pip-check and Git whitespace checks PASS. The
secret-boundary check is separate from scientific validation and does not
certify privacy, result quality, final fresh evidence or publication readiness.
