# Verification — supervisor brief, 11/09/2026

## Verdict

Ready as a **10-page supervisor presentation brief**, with evidence limitations
visible. Not a completed SOTA benchmark, all-scenario protection proof or
conference-readiness certification. No algorithm, dataset, stored benchmark
result or graduation-thesis source was modified in this task.

## Checks performed

- Read the current thesis/reproduction notes and primary literature. Added
  protection-related work for next-step/future location and destination inference;
  kept the S8 direct-dummy-defense gap explicit. Corrected WPES 2014 first-author
  attribution and mix-zone reference; restored DLS cloaking-region metric.
- Recomputed the fresh-readout verifier's assertions and compared its generated
  receipt with the existing immutable receipt. The first direct invocation failed
  at the final exclusive write; the read-only receipt-comparison rerun passed.
- Independently reconciled 393 records, 264 traces, 172,443 FCD points, 12 route
  families and all 30 subcase counts. Rechecked Recall, privacy table values,
  paired-family differences and the unchanged `chosen=null` decisions.
- `verify_brief.py`: passed; exact four source hashes, 10 PDF pages, 10 scenario
  rows, all comparator names, visible units/caveats and no replacement glyphs.
- Canonical Data app build and offline export passed with runtime
  `b5514c44f112ed9b8fe2e8674369e515b7e712abfc70a896140b96f0ab65cfd2`.
  PDF conversion reported zero uncaught page errors, eight Recall bars and four
  nonzero family-difference bars; two zero groups are explicitly labeled in text.
- Poppler rendered all ten pages. Every page was visually inspected. Fixed
  overflow/orphan pages, narrow metric-table columns, missing direct chart labels
  and misleading default sign colors. Final pages 1/3/4/7/8 were pixel-identical
  to the inspected version; re-inspected changed pages 2/5/6/9/10 after final export.
- Final report hashes are in
  `docs/supervisor_meeting/2026-09-11_brief/verification.json`;
  PDF conversion receipt is alongside the PDF.
- Authored files pass `git diff --cached --check`. The exact offline HTML export
  has three compiler-generated trailing-space warnings; its verified bytes are
  preserved rather than manually editing the packaged runtime.

## Remaining boundaries

- Browser UI inspection was unavailable (`iab` unavailable; browser inventory
  empty). Headless printing and PDF QA passed; interactive editor, hover and mobile
  behavior were **not** tested.
- AnotherMe original metric protocol is only partially verified from accessible
  abstract/code, not a complete full-text reproduction. S8 still lacks a verified
  direct dummy-generation protection comparator in this survey.
- The new quantitative results compare four **internal variants**, S1–S3 only,
  from one urban area. S4–S10 dataset availability is not a protection result.
- No method passed the locked worst-case Recall rule. Lower Hit100 with the
  switching model accompanies lower MAE; report this tradeoff rather than a win.
- No new experiment or full algorithm regression suite was run in this reporting
  task. Source-level scientific audits remain in the prior fresh-run evidence.

## Handoff

- PDF/HTML: `artifacts/reports/supervisor_brief_2026-09-11.*`
- Speaking guide and regeneration: `docs/supervisor_meeting/2026-09-11_brief/README.md`
- Literature qualifications: `source_notes.md` in that same directory.

## Follow-up revision — explicit method-to-scenario mapping

Request: the sources were visible, but it was unclear which protection method
addresses which scenario. Updated only the supervisor brief and its companion
content/checks; the graduation thesis, algorithms and experiment artifacts remain
unchanged.

- Pages 2–3 now name each method, explain its protection mechanism and distinguish
  a shared target from an adjacent task. S3 has four separate method entries.
  S8 remains an explicitly unverified direct-dummy-defense mapping.
- Page 5 labels the principal S1/S3 targets of the selected comparators. AnotherMe
  remains a virtual-trajectory reference, without an unsupported S4–S6 coverage
  claim. The speaking guide and literature notes use the same distinction.
- Compared the new snapshot with commit `93382fb`: all scenario IDs, targets and
  A/B/C conditions, data counts, metric definitions, numerical tables and both
  chart datasets are unchanged. No experiment was run or benchmark claim upgraded.
- Extended `verify_brief.py` with ten method-to-scenario assertions and the key
  caveats. All assertions passed, including four source hashes, PDF/HTML receipts,
  ten PDF pages, page footers and the unchanged quantitative reconciliation.
- Native Data app build/offline export passed. Final PDF conversion recorded
  zero runtime errors. Visually inspected revised pages 2/3/5; final pages 2/3
  are pixel-identical to the inspected draft. Pages 1/4/6/7/8/9/10 are
  pixel-identical to the previously accepted pages at the same raster resolution.
  No overflow or lost scenario rows were found. Interactive/mobile QA remains
  outside this check.
- The previous local server was not responding; printing succeeded through a
  new loopback server on port 4184. This was a tooling recovery, not a report
  or experimental-data modification.
- `git diff --check` passed for this revision, including the exact generated HTML.
  Current output hashes are recorded in `verification.json` and the PDF receipt.
