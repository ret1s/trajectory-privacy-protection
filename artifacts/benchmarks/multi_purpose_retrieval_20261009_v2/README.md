# Fixed public multi-purpose retrieval test

Verified paired static-service replay on existing Geo-I Q. No candidate passed the predeclared SELECTION gates. The active recommendation remains Geo-I/REM Epoch8/L30; neither the Geo-I backbone nor its sealed results were overwritten.

The four-type L10 variant requests nearest, fastest, a public 1,000 m radius and a fixed PUBLIC destination bank on every Q. It does not transmit the private destination. It reaches 86.34% macro Recall vs 92.69% for L30 and pays 57.52% more reply bytes on the existing 24-family, three-draw TEST cohort. These are reused same-map synthetic groups, not new-data confirmation.

See the [research note](../../../docs/research/2026-10-09_multi_purpose_retrieval.md) for logic, all comparisons, limitations and the retained v1 numerical correction. `protocol.json` freezes source/input/config; `selection_freeze.json` records rejection before TEST replay; `readout.json` retains every method/split; `validation.json` independently recomputes aggregates/cost/CI and reconciles all 27,172 L30 event windows. `template_overlap.json` checks all 90,570 TEST protected queries. Native forward-oracle response checks cover 45 first-event requests, not every native state.

Implementation is optional/experimental in `benchmark/multi_purpose_retrieval.py`; the operational L30 facade and sampler are unchanged. Each template is costed as a separate request per Q with duplicate records retained; batching, response deduplication, HTTP latency and energy are not measured.
