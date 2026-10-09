# Public artifact boundary audit — 07/10/2026

**PASS.** One run of the immutable V2 scanner after `FINAL_SCAN_READY` checked 411 Git-new files (64,731,388 bytes), including 146 gzip files expanded to 1,443,334,885 bytes. No key-material/digest matches, actual key/database/private-state files, decoding errors, unstable files or files above the 100 MiB gate were found. No new visible files appeared during the scan.

The retained local reference set contained 217 key files and 141 distinct 32-byte keys in 11 autodiscovered work directories. The scanner tested 1,798 exact raw/hex/base64 key and SHA256-digest variants. No key values, key digests or private key filenames were exported.

The scope is staged additions and untracked files excluding ignored files. Existing tracked modifications, including `artifacts/reports/graduation_thesis.pdf`, are outside this scanner's inventory; their PDF/source review is separate. This JSON receipt and the present Markdown summary were created after the audited inventory was captured.

This is an exact-match artifact boundary check against the discovered reference set, not a general secret audit or privacy proof. Encrypted/fragmented/arbitrary encodings and unrelated secrets remain outside its guarantees.

Machine-readable evidence: [JSON receipt](2026-10-07_thesis_public_boundary.json). Completed UTC: `2026-10-07T03:04:04.292216+00:00`. Receipt SHA256: `d2afa6561f3a838207399e0eaac59fbff727970afc0928971022121380f2381b`.

Immutable scanner source SHA256: `688f1e41374521206e89521f2611fbf48a288816100dd93e5c40436814c9a5d8`. No old scanner, scientific source, historical evidence or previous receipt was modified.
