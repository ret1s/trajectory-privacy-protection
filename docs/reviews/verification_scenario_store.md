# Verification: immutable scenario database and thesis reproducibility

Date: 2026-09-08. Starting main commit:
`630a8d82f7cd21739bb1cabc950b6190e6e035d0`.

## Verdict

**Verified local storage migration and reproducibility improvement.** Both frozen
SUMO+OSM scenario releases are now available in an actual relational SQLite
database with an independent update-history table. The canonical thesis explains
versioning, data boundaries and the difference between generated coverage and
evaluation readiness. No new protection method, new scenario trajectories,
attacker run, improved score, or all-scenario protection claim is introduced.

228 tests passed, including 31 new storage tests. Every migrated point, scenario
record and allowed private-device stream matches the frozen JSON source. Native
SUMO v2 verification was rerun and exactly matches the existing verification
artifact. The 32 frozen scientific source checks and nine historical artifact
hashes listed in the migration receipt remain unchanged.

## Delivered files

- `artifacts/datasets/scenarios.sqlite3`: approximately 26 MiB; committed local
  evaluator registry with v1 and v2, not a public attacker database.
- `data/scenario_store/schema.sql`, `store.py`, `__init__.py`: schema, explicit
  read-only reader, transactional import and validation. No new Python package.
- `data/scenario_store/README.md`: table grain, commands, lifecycle, limitations
  and primary SQLite references.
- `experiments/scenario_db.py`: inspect log/coverage, verify, import and export;
  device/full exports require explicit release ID plus content hash.
- `experiments/verify_scenario_db.py`: independent frozen-source migration audit,
  all device-view comparisons and historical result/source checksum checks.
- `tests/test_scenario_store.py`: rollback, integrity, concurrency and input parity.
- `artifacts/datasets/scenario_store_verification.json`: machine-readable receipt,
  per-case/per-split counts including zeros, source hashes and event history.
- `thesis/dataset_registry.tex`, included as §3.9 in `thesis/main.tex`; related
  conclusion wording, references and TOC layout updated. Existing method/result
  chapters and numerical result tables retained.
- Canonical `artifacts/reports/graduation_thesis.pdf`: 73 pages after layout QA.

## Data contract and lineage

The change unit is a complete immutable release, not an editable FCD row.
`releases` stores parent ID, original file-byte SHA256, semantic content SHA256,
source path, actor, UTC import time, reason and status. A new revision must name
the expected current parent. Reads for experiments pin a named sealed revision;
there is no implicit latest-dataset substitution.

Family membership determines the split; session identities and resource reuse
are checked before insertion. Points use a composite release/session/index key.
Records, slots and allowed observations have release-scoped composite foreign
keys. Thus an old release's session cannot accidentally satisfy a new release's
foreign key, and a record cannot quietly move to a different split. Ordering is
explicit and contiguity is rechecked. Variable labels/plans/provenance remain
JSON metadata; this does not claim every domain constraint is a SQL constraint.

`update_log` gets one event automatically when the release is sealed. The event
contains predecessor, actor, reason, import time, both hashes and row counts.
All inserts, read-back validation, sealing and the event are in one transaction.
Failed operations roll back and return errors, without a misleading committed
update event. Identical retries add no event and do not move the head backwards.
Updates to sealed data and changes/deletions of existing log entries are rejected.

The registry uses SQLite 3.50.4 here, requires >=3.37 and JSON functions, and enables
foreign keys per connection. Default readers are opened `mode=ro`; writers use
`BEGIN IMMEDIATE`. A source-defined schema fingerprint detects missing/altered
guards. Transactions use the default rollback journal; sidecars are ignored by
Git, and writers were closed before checkpointing the binary file.

v1: 3 families, 36 sessions, 21,818 FCD points, 96 records, 1,894 allowed
observation references. v2: 6 families, 132 sessions, 94,334 FCD points,
207 records, 6,572 allowed references. Totals are 116,152 points, 303 records,
468 session-slot views and 8,466 allowed observation references. Reused points
and slots are not new independent trips/users.

Semantic content hashes:

```text
urban-scenarios-v1  6fc977209401f8591360e35bd7ce390bfadd9cd5098d74b47a5c9d3a9cefe358
urban-scenarios-v2  27ea74e3d13deea059df663b5429f4333af97b096888cb8ee5f0669eefbbaff0
```

These are hashes of the project's deterministic JSON serialization (sorted keys,
preserved array order/numeric types, compact separators, UTF-8, no NaN/Infinity).
They are not RFC 8785/JCS claims and are not the original JSON file-byte hashes.
The latter remain unchanged: v1 `826bf236...aabb9`, v2 `e5ee6dad...56ef8`.
An exported file can have different whitespace and byte SHA while representing
the exact same dataset. Do not feed a semantic hash to a byte-hash checker.

Database byte SHA256 at this checkpoint:
`335b57b189bcdab53067b279e65c070f1be667dd3cca984327f2b447279a1301`.
This identifies this binary checkpoint, **not** a reproducible dataset identity:
rebuilding changes import timestamps and may change SQLite page layout.

## Verification performed

1. `venv/bin/python -m pytest -o addopts='' -q`: **228 passed in 16.24s**.
   The 31 storage tests include numeric/time/index errors, NaN and duplicate JSON
   keys, split/identity mismatch, immutable rows/logs, insert/replace rejection,
   foreign-key failures inside a real transaction, stale parents, retries,
   wrong export pins, overwrite refusal and modified-schema detection.
2. Two real competing writer connections synchronized by a barrier: exactly one
   commits; the other rejects the stale parent. No lost update and two total
   log entries including the initial fixture release.
3. Migration audit reconstructs both complete bundles and compares them against
   source JSON. All 468 device views match the frozen `device_view`, including
   S8 relative timing, query categories and the seven-slot S6.C structure.
4. CLI `device` and `export` exercised with pinned v2 hash. Export content digest
   matches v2. The device envelope labels observations private pre-protection.
5. Independent rebuild from v1 then v2 in a new temporary database succeeds and
   yields both exact semantic hashes and the same complete migration comparisons.
6. `venv/bin/python -m experiments.verify_scenario_suite_v2 --raw`: 54 native
   SUMO runs, all 94,334 points, 7,052 external-edge transitions, maximum 1-second
   displacement 9.203158493m. Output equals the frozen native audit exactly.
7. Nine frozen dataset/result/table hashes and 32 scientific source hashes
   checked by `verify_scenario_db`; detailed values are in its JSON receipt.
8. XeLaTeX/latexmk compilation, text extraction and rendered PDF inspection.
   New §3.9 is physical PDF pages31–32 (printed23–24). Conclusion is physical70,
   references71–73. There is one title page. Existing result values are retained.

PDF SHA256:
`98425de0aaf2334b235c91e69dd32dc4b3a391950e5d2a331426b46d39c8317c`.
The old manually forced TOC page break plus the longer chapter caused a nearly
empty TOC page; it was removed and TOC spacing adjusted. The added conclusion
wording was condensed to avoid a one-line trailing page. Body typography stays
at the established thesis format. No overfull boxes, undefined citations/labels,
duplicate destinations or missing-glyph warnings in the final build.

## Scientific implications and what remains unchanged

The catalogue × split coverage view retains all90 v2 cells, including zeros.
S5.C remains **0 train / 0 validation / 1 confirmation**. It is generated, but
cannot support the current learned/validation-selected attack protocol. A fixed
attacker could still be run diagnostically, without pretending the single record
estimates population performance. S6.C still has11 records, not12. The migration
does not fill data gaps or create a new independent confirmation set.

The database is evaluator-only. `device_view` returns allowed true coordinates,
relative time and query category: these still need protection before LSP access.
It is not an authenticated service or a public view merely because it strips IDs.
Do not expose the DB file, full export, log or provenance envelope as an attacker
transcript. S8 preserves the common epoch; S6.C current-trip slot is6, not0.

Frozen experimental runners deliberately still consume their exact JSON inputs
and original byte hashes. New experiments can use the database reader or an
explicitly pinned export. **This work does not claim every runner/web route was
migrated**, nor does it retune/rerun the proposed-method study. Previous privacy,
utility, infeasible configuration, comparator fidelity and Geo-I limitations
remain those documented in `verification_belief_suite.md`.

## Next steps for Claude Code

1. Build additional independent families for sparse S5/S6/S10 cases, using public
   gates decided before protection scores. Native-verify the candidate, import a
   new release with an explicit parent/reason, then pin it in a new experiment.
   Do not mutate v1/v2 or discard difficult/failed cases to raise means.
2. Introduce a separately versioned DB-backed experiment entry point/protocol.
   Record release/content hash, case eligibility, selected slots, code hashes,
   context/prior, seeds and attacker selection. Keep historical entry points
   frozen and keep paired comparisons on identical allowed observations.
3. Build attacks for S4–S10 with their proper labels and temporal permissions.
   The existence of a scenario record is not attack success or protection evidence.
   Avoid treating aggregate SQL row counts as independent sample sizes.
4. If semantic label JSON evolves, add an explicit adapter and tests, including
   future-index validation and scenario-specific gates; do not accept unknown FCD
   fields by silently discarding them. The current importer supports v1/v2 only.
5. At larger scale, add validated SQLite backups or migrate to a server database
   for multi-host writers. Never use Git merge heuristics on binary DB conflicts;
   preserve both versions and reconcile their revision histories explicitly.

## Limits, literature and report QA rationale

The update log is append-only under the supported workflow, not tamper-evident
against an administrator who can rewrite schema, hashes and logs. Actor values
are supplied labels, not authenticated identities. Raw native SUMO cache paths
are provenance references and are not made durable merely by saving their names
in SQLite. Rebuilding caches still requires the pinned OSM/SUMO prerequisites.

Primary sources consulted on September8:
[SQLite foreign keys](https://www.sqlite.org/foreignkeys.html),
[transactions](https://www.sqlite.org/lang_transaction.html),
[appropriate uses](https://www.sqlite.org/whentouse.html).
These support implementation choices, not a new location-privacy research claim.

Data-quality and validation workflows informed the grain/FK/zero-cell checks,
failure injection and strict distinction between structural and scientific QA.
The technical-report workflow was applied to the existing canonical LaTeX/PDF,
as requested, rather than producing a parallel HTML report. Its roles map to the
existing summary, chapters2–4 definitions, §3.9 storage/evidence/limitations and
chapter7 next steps. No new quantitative chart was needed: migration parity is
an exact audit result, not a new comparative protection finding. CLI modules and
tests provide the reusable audit path; no redundant notebook was created.
