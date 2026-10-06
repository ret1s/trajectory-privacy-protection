# Public-artifact boundary: completed fresh snapshot

**PASS at07:41:59 UTC,06/10/2026**, after completed fresh evidence, independent
validation, recommended configuration, figures and research-note writes. This
audit covers the declared **Git-visible NEW/untracked file inventory**, excluding
ignored local private files and existing tracked modifications. No sealed source
or evidence was changed. The sanitized audit JSON and this note were generated
after the selected snapshot; they contain no private key material or key digests.

| Check | Sanitized count / result |
|---|---:|
| Selected/scanned regular NEW files |1,813 /1,813|
| Public file bytes |232,318,889|
| Gzip files / expanded bytes checked |682 /1,351,795,788|
| Local private work roots / key files inspected |11 /217|
| Distinct actual32byte keys |141|
| Key material / SHA256-digest encoding patterns |889 /909|
| Combined exact patterns |1,798|
| Actual `.key`/database files, state paths or SQLite headers exported |0 /0 /0|
| Key/digest matches in public bytes / expanded gzip |0 /0|
| Allowed non-secret private-storage path references in provenance/docs |5|
| Decode/nonregular errors / selected files changed during scan |0 /0|
| New visible files created during scan |0|
| Files above the Git100MiB per-file gate |0|

The largest new file is development v3 `attack_selection.json`, **7.056MiB**;
development v2's corresponding file is7.046MiB. The largest fresh base-Q model
is5.648MiB. The full sanitized inventory, gates and scanner source hashes are in
[`2026-10-06_public_artifact_boundary_final.json`](2026-10-06_public_artifact_boundary_final.json).

The additive V2 scanner checked **actual known private-key material and its
SHA256 digests**, each as raw bytes, lower/upper hex, standard/URL-safe base64
and unpadded variants, including exact embedded substrings and decompressed
gzip. No private key bytes, key digests or private key filenames were printed
or saved. Sanitized `/private/tmp` reproduction references are permitted text,
not exported key/state files. This finite exact-pattern scan does not prove
absence of encrypted, fragmented, arbitrary encodings or unrelated secrets;
existing tracked modifications and ignored files remain outside its scope.

Root reported canonical QA for the completed sources: **915 passed,9 integration
skips**, dependency `pip check` PASS and Git whitespace checks PASS. Subsequent
read-only presentation verification passed19 focused checks, parsed64 new Python
files and found0 missing targets across127 local Markdown links. The final PNG
was visually reviewed, including full forest labels, development/fresh separation,
N/A coverage and actual TEST reply cost. The V2 boundary tests cover planted
digest variants, gzip embedding, sanitization and allowed non-secret source hashes.

This boundary result is separate from
[fresh scientific validation](../../artifacts/benchmarks/qplanner_response_depth_generalization_20261006_v1/validation.json).
It does not certify a privacy theorem, result superiority, production security
or publication readiness. The earlier
[interim snapshot](2026-10-06_public_artifact_boundary_pending.md) is preserved.
