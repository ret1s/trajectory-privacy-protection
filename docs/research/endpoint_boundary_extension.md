# Endpoint extension — 24 September 2026

Implemented candidate: `core.boundary_release.BoundaryProtectedStream` wraps
`SwitchingCoverLaneDummy` through its existing `reset` / `protect_step` API.
This is a causal delayed-publication interface, not a claim of endpoint anonymity.

```python
from core.boundary_release import BoundaryPolicy, BoundaryProtectedStream
# engine is an existing, newly constructed SwitchingCoverLaneDummy instance.
stream = BoundaryProtectedStream(engine, BoundaryPolicy(60, 60), session_start_s=0)
for point in points:  # increasing device clock; no future samples passed in
    public_events = stream.ingest(point.timestamp_s, point.lat, point.lon)
    # Send only these events. Do not serialize engine state or evaluator_summary.
stream.close(actual_end_time)  # cancel queued tail, never flush
```

Lifecycle:
1. Skip early GPS before calling the core: hidden head cannot seed its anchor.
2. Generate protected candidates causally; queue immutable candidates.
3. Release only when age >= delay, at the next sample arrival. Payload timestamp
   is release time relative to session opening. Multiple overdue events can batch.
4. Close cancels the pending tail. No future endpoint/time is needed at ingest.
5. Keep one core ledger per session, including charges for queued/cancelled events.
   The core's existing horizon controls private reads; do not reset it to improve
   utility. Multiple sessions still require a composition/accounting policy.

Threat-model limitations: session opening/closing, arrival cadence and batch size
remain observable. A known delay can reveal approximate sample times. Endpoint
inference from the remaining road path is still possible. No new end-to-end
Geo-I theorem, no suppression-policy DP claim, no padding or live POI cache.
Warmup/delay=0 is the S1–S3 control; a one-event window can be entirely suppressed
by a nonzero gate and must not silently disappear from utility denominators.

## Reproducible checks

- `python3 -m unittest discover -s tests -p test_boundary_stream.py -v`
- With the existing scientific dependencies: `python -m pytest tests/test_boundary_switching.py -q`
- `python3 -m experiments.check_boundary_schedule`

The schedule audit covers all 62 S9/S10 records (101 session views), four fixed
(h, delay) settings: (0,0), (60,0), (0,60), (60,60). It uses a constant placeholder
engine, not BR. Its 404 rows measure bookkeeping on already allowed windows.
Closing at the last allowed sample is a conservative window audit, not the true
session lifecycle. Release fractions are neither Recall@5 nor privacy success.

Next endpoint experiment must restore the dataset SUMO network and original
service oracle, instantiate the actual core, keep targets and full trip lifecycle
strictly evaluator-side, and log retained/dropped queries plus release latency.
Compare core-only, head-only, buffer-only, combined. Select on development;
new independent groups are needed for confirmation. Include road-constrained,
metadata-aware and repeated-session endpoint attackers; keep the six A/B/C cases
separate and report empty outputs and utility on the original query denominator.

## Literature decisions

- S-TT, Brauer et al. 2022: direct offline sensitive-site protection. Study the
  protection-set partition and joint proximity/direction attack criteria. The
  2023 author chapter is a restatement, not an independent new model.
- EPZ / Dhondt et al. CCS 2022: direct endpoint suppression but vulnerable to
  road, boundary-entry and metadata inference. Use as attack/countermeasure basis.
- Data Stream Obfuscator, Dehaene et al. July 2023: zones/timeslots and consistent
  mappings across services; relevant component, endpoint effectiveness unverified.
- PRISM 2025: predictive semantic gating inspiration; not verified S9/S10 results.
- AGeoI + dummy EV querying 2024: utility-aware querying inspiration; not a
  demonstrated endpoint defense.

Full primary URLs, publication dates, source-access limits and metrics are in
`docs/supervisor_meeting/2026-09-26_brief/endpoint_extension.py` and `sources.json`.
Older direct sources are explicit exceptions to the recent-work window, not
silently re-dated. The six principal comparison models have not been expanded.
