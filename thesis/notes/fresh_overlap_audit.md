# Pre-confirmation metadata and overlap audit — 2026-09-10

Before any validation/confirmation protection scoring:

- The established SQLite schema accepts development_validation/confirmation, not
  fresh_validation/fresh_confirmation. The publisher uses a declared, lossless
  metadata adapter: preserve the original role in evaluation_role, map only split
  names, add generator hash. No trajectory, observation, label or source is changed.
  The original generated JSON remains immutable; the DB content hash refers to its
  explicit adapted form. Historical DB and schema are not altered.
- Training completed before this integration issue was found. Its outputs are
  retained under cache/fresh_switching_pre_adapter, not final evidence. Rerun under
  the corrected source hash before scoring validation; no reuse of an invalid source
  binding. This change does not modify protection or attacker training algorithms.
- There are no exact routes or first-12 windows shared with the historical training
  or old development sets. There ARE two identical protected-input windows across
  fresh validation and confirmation: S2.A and S2.B, families 304 and 311 (six and nine
  stationary observations respectively). Identities and complete routes differ.
- Primary analysis retains fixed seeds and windows. Also report a prespecified
  descriptive sensitivity excluding confirmation records v2-r311-002 and
  v2-r311-003, without reselecting attackers or methods. State this overlap explicitly;
  do not claim complete window-disjoint confirmation or independent spatial support.
- S3.B has five confirmation families, all other S1–S3 subcases six. Preserve the
  missing case and use actual per-subcase denominators. S4–S10 coverage is sparse in
  some subcases and is not evidence of protection.
