# Public Q-planner development chronology

This log distinguishes exploratory old-data development from the unopened
new same-map synthetic test. It does not claim a blind preregistration of the
development choices.

## First objective: completed, rejected

The first trial used old native families: 12 TRAIN groups with one private draw
each and 6 SELECTION groups with three private draws each. A single-family
smoke preceded the fixed development gate. The complete aggregate was then
evaluated with all planned blocks retained. The serial prefix was preserved
byte-for-byte in the parallel completion; no completed block or RNG draw was
replaced because of its score.

The immutable evidence is in
[`qplanner_development_20261006_v2`](../../artifacts/benchmarks/qplanner_development_20261006_v2/).
Its independent verifier passed all 30 blocks, 90,720 utility windows, the
same private anchor/read tapes, causal cache and wire accounting, and the
retained finite attacker banks. This certifies the replayed arithmetic and
execution, not a privacy theorem or an exhaustive attacker model.

| Arm | Equal-purpose current Recall@5 | Difference from legacy | Paired 95% family-bootstrap interval |
|---|---:|---:|---:|
| Legacy planner | 92.1885% | — | — |
| L20-aligned nearest | 92.1276% | −0.0609 pp | [−2.0945, +1.9699] pp |
| Multi-purpose mean | 91.7777% | −0.4108 pp | [−1.7371, +1.1507] pp |
| Tail weight .25 | 92.6256% | +0.4371 pp | [−1.9601, +2.6604] pp |
| Tail weight .50 | 92.6210% | +0.4325 pp | [−1.9647, +2.6559] pp |

All candidate gains were positive in draw1 and negative in draws2/3. Every
candidate had a worse lower25% family mean than legacy. The tail arms also
increased selected S5/S6 finite-attacker accuracy by 11.1111 pp, exceeding the
fixed development guard. The durable selection record is **NONE**. No fresh
test was generated or scored for these rejected arms, and no success threshold
was relaxed.

## Second objective: completed, rejected

The next exploratory trial addresses a concrete weighting mismatch: global
conditioning on nonempty latent/purpose/category profiles gives purposes and
cases different effective weights when their references are often empty. The
new objective averages valid categories within each public latent/destination
case, conditions separately within each purpose, and equally weights defined
purposes. It also compares the original public motion slack .03 with slack0;
the third arm adds tail weight .25 to the tight version.

This is a deliberate new algorithm version informed by failed development
results. Geo-I, protected filtering, private read schedule, integer budget,
K5, server L20, reference top5 and local utility scoring remain the same.
Private keys may be reused locally solely to pair the old development controls;
key bytes, key digests and SQLite ledgers are excluded from public evidence.

New sources use the `_v2.py` suffix and the new development output is
`qplanner_development_20261006_v3`. The first trial and its source snapshots
are retained. The numeric development and fresh success gates remain unchanged.

All 30 paired development blocks completed. Independent verification passed
the unchanged control streams, utility, wire accounting, private anchor/read
tapes and retained attacker banks. No candidate achieved the fixed minimum
one-percentage-point utility gain; the selection record is again **NONE**.

| Arm | Current Recall@5 difference from legacy | Paired 95% family-bootstrap interval |
|---|---:|---:|
| L20-aligned nearest | −0.0609 pp | [−2.0945, +1.9699] pp |
| Purpose-normalized mean | +0.1175 pp | [−1.7796, +2.0137] pp |
| Purpose-normalized tight | −0.3454 pp | [−3.3054, +2.6256] pp |
| Purpose-normalized tail | −0.3443 pp | [−3.1013, +2.5004] pp |

These are exploratory intervals on reused development families, with all
planned arms retained. None of these planners was adopted or tested on the
fresh cohort.

## Third objective: paid response depth, selected before fresh scoring

The next intervention preserves the exact legacy Q transcript and all Geo-I
inputs. Only the public server response changes: at most 20, 30, 40 or 60 POIs
per category per Q. Local purpose-specific top5 selection uses the larger pool.
The fixed development rule chooses the smallest depth with at least2 pp
current macro gain, no nearest-purpose loss and reply-byte ratio at most2.5.

L30 was selected: 92.1885% → 94.7616%, a gain of2.5731 pp, with exploratory
paired95% interval [1.7928,3.5090] pp and positive gains in all three draws.
Reply bytes increase31.094%; this is a utility/cost configuration, not a new
Q algorithm, free bandwidth or a privacy improvement. L40/L60 remain visible
in the retained development evidence.

The 60-family fresh dataset was built using public route eligibility before
private choice assignment. It is still the same public map/generator, not
real GPS or cross-city confirmation. A candidate must first pass development
selection and be frozen before its first fresh benchmark. Its single primary
contrast must gain at least2 percentage points, have a positive paired95%
family-bootstrap lower bound, and improve in each of three private draws.

The fresh depth protocol and selection were frozen before generation of any
fresh Q stream or depth score. Fresh generation uses only the unchanged
legacy Geo-I planner, with 24 TRAIN groups, 12 SELECTION groups and24 TEST
groups; TEST has three nested private draws per group. Only L20 and selected
L30 will be scored. Independent base-Q verification precedes depth replay.
Fresh execution and both independent audits completed. The frozen single
primary contrast **passes all three gates**, with 24 TEST family clusters:

- Current conditional macro Recall@5: **89.7144% → 92.6880%**.
- Gain: **2.9737 pp**, paired95% whole-family interval **[2.3079,3.6464] pp**.
- Gains in draws1/2/3: **2.8669 / 2.9286 / 3.1255 pp**; draws remain nested.
- Lower25% family utility improves4.4814 pp; this is a descriptive secondary.
- Reply bytes: **494,713,236 → 648,578,577**, **+31.1019%**; the same90,570
  requests and13,678,985 request bytes. These are compact application JSON,
  not measured HTTP/TLS, latency or battery costs.

The base-Q audit replayed all108 blocks and the finite selected attacker
banks. The separately self-pinned depth audit reconstructed108,688 utility
windows and54,344 wire rows, exact L20 replies/costs, N/A arithmetic, nesting,
and the10,000-resample whole-family statistics. The selected L30 configuration
is retained in
[`recommended_configuration.json`](../../artifacts/benchmarks/qplanner_response_depth_generalization_20261006_v1/recommended_configuration.json).
No further depth was tested or substituted after viewing TEST.

This confirms a **paid-bandwidth utility configuration on new same-map
synthetic routes**. It does not establish a new Q algorithm, privacy theorem,
real-data/cross-city result or superiority over competitors at equal cost.
The local utility oracle uses synthetic ground-truth GPS/destination at each
public event;60s is the protected Geo-I supplier schedule, not a validation
of local reranking with only60s physical GPS fixes. The static/bulk application
gate remains unresolved for a publication claim.

A preflight archive-path mismatch required a separate adapter version before
freeze: the old dataset version2 intentionally references the version1 native
archive. The final adapter authenticates that old metadata and permits only
a byte-identical relocation of this archive beside the fresh dataset. Other
public inputs, privacy sources and numeric gates remain exact. Earlier
adapter/source snapshots and development statistics are preserved.
