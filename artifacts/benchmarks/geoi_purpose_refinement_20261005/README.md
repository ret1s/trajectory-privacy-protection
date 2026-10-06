# Geo-I public-purpose objective refinement, retained baseline

This folder is a byte-identical promotion of the complete temporary run;
`promotion.json` records all source input hashes. All nine development
alpha/depth candidates and their rejected gates are preserved. Only the
selection-retained alpha0 L20 and raw have heldout runs/scores.

Changing public POI weights does not change Geo-I REM/noisy-test emission,
protected Z, private-read clock or branch ledger. It CAN change Q. Answering
different private purposes after the SAME fetch does not add network or
protection calls. Alpha0 retains the existing Q objective while the client
supports all four local purposes. No new public-weight candidate was promoted.

`protocol.json` declares policies/cohorts/source pins; `selection.json` retains
every selection gate; `readout.json` is the retained test-development readout.
`utility_rows.json.gz` preserves per-category/purpose references and returned
IDs including empty references; development/test Q streams retain evaluator
truth separately from their public event fields. Public experiment seeds are
simulation reproducibility material, not a deployment private RNG policy.

`verification.json` independently recomputes Recall/selection/attack summaries,
replays POI availability, exact directed-road ranking and JSON wire totals, and
checks Z/ledger identity. Run `python -m experiments.verify_geoi_purpose_refinement`.
Learned/Viterbi predictions lack frozen fitted model files: verification of
their aggregate arithmetic does not independently re-fit their estimates.

The graph has constant8m/s, so fastest and nearest rankings coincide. Finite
nearest-L replies can miss optimal detour POIs even for raw GPS. Three reused
test families × two draws are development evidence; zero Hit100 is not a proof
of protection. See [research explanation](../../../docs/research/2026-10-05_geoi_purpose_refinement.md).
