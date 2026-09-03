# Verification vòng 6 — audit C5 source candidate tại `1293304`

## Hồ sơ review

| Trường | Giá trị |
|---|---|
| Branch được review | `verifier` |
| Parent | `1fef76a082c5e70036e223ebf127e12226595a32` |
| Candidate | `1293304828cacf567a78cf16e4a9b9e4229105e0` (`C5`) |
| Candidate type | source/tests/manuscript update; **không có O5** |
| Ngày review | 2026-08-24 (Asia/Ho_Chi_Minh) |
| Phạm vi | Data identity, paired design, edge metric, formal claims, production API, tests/RNG, attackers, provenance, graph/environment, statistics, literature, thesis/docs/web và PDF |
| Kết luận | **NEEDS REVISION — chấp nhận như C5 source candidate, không chấp nhận như result package** |

Review này độc lập và commit-scoped. Mọi line reference trỏ tới snapshot
`1293304`, không phải một working tree đã sửa sau review. Verifier tách bốn lớp:

1. **Ideal theorem** — statement/proof kernel có literal-correct không?
2. **Executable mechanism** — float, RNG, state và API có thực thi kernel đó không?
3. **Evaluation evidence** — data, metric, attacker và statistics support kết luận nào?
4. **Artifact provenance** — source/data/environment/command có pin đủ để tá lập không?

`PASS` ở một lớp không tự động kéo theo `PASS` ở các lớp khác. Đặc biệt, test
xanh không biến output cũ thành output của source mới.

### Quy ước trạng thái

- **PASS**: root cause, implementation/evidence và wording liên quan đều đạt
  acceptance criteria.
- **NARROWED**: chưa có capability/evidence mạnh hơn, nhưng claim đã được hạ đúng
  phạm vi.
- **PARTIAL**: có sửa thực chất nhưng còn ít nhất một acceptance criterion fail.
- **FAIL/REGRESSED**: có phản ví dụ trực tiếp, artifact không support claim, hoặc
  declared environment không chạy được implementation.
- **PLANNED**: chỉ là future work, không được gọi là contribution đã verify.

## 1. Kết luận điều hành

### 1.1 Release decision

`1293304` **không phải “newest verified result”**. Đây là một source candidate có
nhiều sửa đổi đúng, nhưng commit không thay `outputs/*` hoặc `thesis/main.pdf`.
Ba blob output/PDF hoàn toàn giống parent:

```text
artifact                                  parent blob                               C5 blob
outputs/benchmark_results.json           cfccefc5e439291256fc7891d1c5aa5986e0b917  same
outputs/averaging_multi_results.json      95171d0bf9ff45842dfc78b82fe70c7227e3ca03  same
thesis/main.pdf                           858778f3b0c809c9950b2b95a4473dd7f7f1c293  same
```

Hai JSON vẫn pin:

```text
source_commit = 4aea94b376510d2a66cb18eb19869237c2801be6
```

PDF committed vẫn là O4, 32 trang. Vì C5 thay tập stay-points, pairing của input
jitter, định nghĩa `on_road_rate` và đường HMM, không được tái sử dụng số O4 cho
claim C5. O5 chỉ nên chạy **sau một C5.1 sửa các blocker source/prose bên dưới**.

Decision chính xác:

> **C5 source candidate: conditionally acceptable with required revisions.**
>
> **C5/O5 result package: not present, therefore not verified.**

### 1.2 Những gì đã verify được

- `git diff --check 1fef76a..1293304`: pass.
- `venv/bin/python -m tests.run_all`: **23 passed, 0 failed**.
- Production-coupled suite: **14/14**; rejection-conditioning suite: **5/5**.
- Stay loader mới trả 40 locations, 21 users, **40/40 UID duy nhất**; khoảng
  cách nhỏ nhất giữa hai stay còn giữ của cùng user là `51.102m`, do đó không còn
  cặp retained nào trong threshold dedup 25m.
- Jitter ngoại sinh đã được paired đúng theo `(home, seed, epsilon)`, không còn
  key theo mechanism; 200 draw độc lập khớp byte-for-byte giữa năm mechanisms.
- Point-to-edge implementation hoạt động đúng trong environment hiện tại
  (Shapely 2.1.1): midpoint edge dài có nearest-vertex hàng trăm mét nhưng
  nearest-edge bằng 0m.
- Independent C5 Planar recomputation cho trajectory-weighted `on_road_rate` là
  `0.818253 / 0.875981 / 0.921468` ở epsilon `.01/.02/.05`, xác nhận metric cũ
  `0.459039 / 0.513228 / 0.585988` không còn hợp lệ.
- Định lý fixed-public-support trung tâm đã chuyển sang event inequality và xử lý
  support-zero đúng hơn; `0/0` không còn được dùng như ratio.
- Online HMM code hiện trừ input-dependent `logZ` đúng dấu.
- `begin_run()` đã được đưa lên trước computation và capture thêm start/end time,
  interpreter, cwd, platform, resolved package versions.
- Literature overlap 2026 đã được đưa vào thesis; novelty wording chính đã hạ
  phạm vi.
- Short caption source fix hoạt động: clean C5 build bằng XeLaTeX/BibTeX tạo PDF
  33 trang, không undefined citation/reference; List of Tables không còn overflow
  caption cũ.

Đây đều là tiến bộ thực và nên được giữ lại.

### 1.3 Blocker còn lại

1. **Không có O5.** Output/PDF hiện tại vẫn là O4/C4 và không thể chứng minh C5.
2. **Declared Shapely compatibility sai.** Code point-to-edge dùng Shapely-2 API
   nhưng `requirements.txt` vẫn cho phép 1.8, dẫn tới runtime failure hợp lệ.
3. **Test coverage chưa bảo vệ các root cause quan trọng.** Không có automated
   tests cho stay UID/dedup, paired runner inputs, point-to-edge geometry,
   provenance schema/command hoặc raw-to-aggregate reconciliation.
4. **PR budget validator chưa thật sự normalize/type-safe.** `True` được nhận như
   budget; `Decimal` có thể pass constructor rồi crash khi `perturb`.
5. **SM statement còn lệch production ở cache miss có temporal history.** Class
   doc vẫn gọi mẫu đầu là REM trong khi code có thể chạy T-REM.
6. **HMM test chưa mutation-sensitive.** Mutation sai dấu `-logZ -> +logZ` vẫn có
   thể pass test hiện tại; web gọi causal filter “IDENTICAL” với offline smoother.
7. **Provenance command không replayable và data chưa byte-pinned.** Schema version
   cũng chưa đổi dù semantics/required fields thay đổi.
8. **Thesis/docs/web còn nhiều số và claim O4.** Đặc biệt Ch5 vẫn nói Planar
   41–54% off-road và suy “riêng tư hơn” từ proxy error.
9. **Hai BibTeX author lists sai.** Citation metadata hiện tại không khớp record
   chính thức.
10. **Graph recipe, dependency lock/CI và statistical strength chưa được giải
    quyết.** Các điểm này có disclosure tốt hơn nhưng chưa đạt reproducibility hay
    confirmatory evidence.

## 2. Status matrix R4-001...R4-015

| ID | Mục tiêu | Verified status | Kết luận ngắn |
|---|---|---|---|
| R4-001 | Stay identity/dedup | **PARTIAL** | Source loader đạt 40/40 UID và dedup 25m; O5 chưa có, record chưa byte-traceable, không loader test |
| R4-002 | Paired exogenous jitter | **PARTIAL** | Implementation đúng; thiếu runner-level test/paired fixture và O5 |
| R4-003 | Point-to-edge metric | **PARTIAL** | Đúng trên Shapely 2.1; không unit test, docs/số cũ còn, declared Shapely 1.8 bị crash |
| R4-004 | SM theorem/production alignment | **PARTIAL** | Ch4 kernel tốt hơn; class doc cache-miss vẫn giả REM, tests chưa bảo vệ representative sampling |
| R4-005 | PR budget validation | **PARTIAL** | Reject 0/negative/NaN/inf; vẫn nhận bool và mishandle Decimal; thiếu parameterized tests |
| R4-006 | Mutation-sensitive production tests | **PARTIAL** | REM coefficient và branch tests tốt hơn; SM raw-input, HMM wrong-sign, paired runner và two-step transcript gaps còn |
| R4-007 | HMM parity/proxy claims | **PARTIAL** | Online `-logZ` đúng; test yếu, causal filter không identical offline smoother, privacy inference còn |
| R4-008 | Provenance/reconstruction | **PARTIAL** | Timing/env cải thiện; command sai, data hash/loader config/schema/raw HMM counts thiếu |
| R4-009 | Fixed-public theorem | **PARTIAL** | Main event inequality pass; test doc/witness còn overclaim ratio/tight factor 2 |
| R4-010 | Repo-wide claim consistency | **FAIL/PARTIAL** | Canonical framing tiến bộ; Ch5/docs/web còn stale values và contradictory claims |
| R4-011 | Literature/novelty positioning | **PARTIAL** | Overlap 2026 đã thêm; author metadata sai, road PTPPM bị gọi stateless, comparator table thiếu |
| R4-012 | Deterministic graph recipe | **FAIL/PARTIAL** | Local artifact hash pass; clean rebuild/download/fail-closed manifest vẫn không đúng |
| R4-013 | Locked environment/CI | **PARTIAL** | Project venv canonical suite pass; broad bounds, no lock/CI, alternate documented commands fail |
| R4-014 | Statistical evidence | **NARROWED/PARTIAL** | Exploratory disclosure pass; vẫn 7 moving users/one root/no clustered CI cho success |
| R4-015 | PDF/layout QA | **PARTIAL** | Source short captions + clean 33-page build pass; committed PDF vẫn O4/32 trang, warnings và semantic QA còn |

Không có mục nào đủ điều kiện **end-to-end PASS** khi acceptance criterion bao gồm
artifact O5. Điều này không phủ nhận các source-level pass nêu trong cột kết luận.

## 3. Evidence và findings ưu tiên

### R5-001 — P0 release blocker: commit không chứa O5

**Evidence**

- `git diff --name-status 1fef76a..1293304` chỉ có 19 source/test/doc files;
  không có `outputs/*` hoặc `thesis/main.pdf`.
- Blob IDs của hai JSON và PDF giống hệt parent.
- Hai JSON pin C4 `4aea94b`, không pin C5 `1293304`.
- Committed PDF có 32 trang/833,952 bytes; clean build từ C5 source có 33
  trang/841,913 bytes.
- Existing averaging artifact vẫn phản ánh lỗi cũ: 40 selected nhưng chỉ 38
  `home_id` duy nhất và 1,600 rows chỉ có 1,520 composite keys duy nhất.

**Impact**

Mọi table/result hiện tại chỉ kiểm chứng O4. Các thay đổi C5 làm thay đổi population,
paired input, road metric và HMM estimate; gọi số cũ là kết quả C5 là provenance
error, không phải chỉ là presentation debt.

**Required fix**

Không rerun ngay từ `1293304`. Trước tiên tạo C5.1 sửa R5-002...R5-010; sau đó
run O5 từ clean tree, update thesis từ đúng JSON và build PDF trong một commit O5
riêng.

**Acceptance**

- JSON pin đúng source commit C5.1 và `source_dirty_before_run=false`.
- Output blobs thật sự thay đổi và khớp C5.1 semantics.
- PDF được build từ đúng JSON/O5, không phải manual table edits.

### R5-002 — P1 runtime compatibility: STRtree code chỉ chạy Shapely 2

**Evidence**

`core/road_network.py:124` làm:

```python
idx = int(self._edge_tree.nearest(p))
return float(self._edge_geoms[idx].distance(p))
```

Đây là semantics Shapely 2, nơi STRtree query trả index. Tuy nhiên
`requirements.txt:3` khai báo `shapely>=1.8.0`. Theo
[Shapely 1.8 manual](https://shapely.readthedocs.io/en/maint-1.8/manual.html),
`STRtree.nearest()` trả geometry; theo
[Shapely 2 migration/release notes](https://shapely.readthedocs.io/en/2.0.0/release/2.x.html),
STRtree query operations đổi sang trả indices. `int(LineString)` sẽ fail trong một
environment mà project hiện tuyên bố support.

Environment verifier đang dùng Shapely 2.1.1 nên canonical tests không phát hiện.

**Positive evidence**

- Edge thẳng dài khoảng 1.11km: midpoint nearest vertex `555.975m`, nearest edge
  `0m`, `on_road_rate=1.0`.
- Bent `LineString`: point trên polyline cách nearest vertex `1017.138m`, nearest
  edge `0m`.
- 13,813 real graph vertices: max point-to-edge distance `0.0`.
- 41,040-edge index build khoảng `0.351s`; 858 queries khoảng `0.019s` trong
  verifier environment.

**Required fix**

Chọn một trong hai và ghi rõ support policy:

1. pin/lock `shapely>=2,<3`; hoặc
2. branch theo return type/version và map geometry về index một cách deterministic.

Thêm test cho straight midpoint, edge có OSM geometry, fallback edge không geometry,
và declared minimum supported Shapely version. Sửa module header
`evaluation/metrics.py:11`, hiện vẫn nói nearest road vertex.

**Acceptance**

- Test long-edge midpoint fail với nearest-vertex implementation và pass với
  point-to-edge.
- Fresh locked environment chạy test đó.
- Requirement/API support không mâu thuẫn.

### R5-003 — P1 evaluation artifact: edge metric mới làm số O4 vô hiệu

Independent C5 recomputation, cùng trajectory-weighted estimand:

| epsilon | O4 nearest-vertex rate | C5 point-to-edge rate | C5 off-edge fraction |
|---:|---:|---:|---:|
| 0.01 | 0.459039 | 0.818253 | 0.181747 |
| 0.02 | 0.513228 | 0.875981 | 0.124019 |
| 0.05 | 0.585988 | 0.921468 | 0.078532 |

Pooled-point estimand cho cùng dữ liệu là `0.806527 / 0.870629 / 0.913753`; vì vậy
phải ghi rõ aggregation là equal-weight trajectory mean hay pooled points.

**Residual contradictions**

- `thesis/chapters/ch5_thucnghiem.tex:69-111` vẫn dùng table O4 và narrative
  “41–54% off-road”.
- `docs/attack_scenarios.md:254,282` và `docs/research_notes.md:102-117` vẫn chứa
  số/diễn giải vertex cũ.
- `evaluation/metrics.py:11` vẫn mô tả vertex metric.

**Required fix / acceptance**

- O5 recompute metric từ raw C5.1 releases.
- Table, prose, docs và conclusion được generate/update từ cùng JSON.
- Ghi rõ estimand/weighting; raw rows đủ để verifier reconstruct table cell.
- Không diễn giải “within tolerance of an edge” thành proof chống map matching hay
  proof privacy.

### R5-004 — P1 data identity: source fix pass, traceability/test chưa đủ

**Positive evidence**

`data/geolife.py:166-230` thêm within-user 25m dedup, stable-looking UID
`user/file/seg/stay` và distinct-location cap. Actual loader profile:

```text
selected locations                       40
users                                    21
unique UID                               40
unique full coordinate                   40
unique rounded-6 (user, coordinate)      40
minimum retained same-user separation    51.102m
prospective (mechanism, UID, seed) keys  1600/1600
```

Hai duplicate/collision cases O4 được loại hoặc thay bằng stay source khác.

**Residual risk**

- `run_averaging_multi.py:126-130` raw row chỉ giữ UID/user/seed/results; không có
  full coordinate, source file SHA, segment time bounds hoặc preprocessing version.
- UID là logical locator nhưng chưa byte-pinned; thay raw GeoLife/preprocessing có
  thể giữ cùng string mà đổi sample semantics.
- Dedup 25m chưa có rationale/sensitivity analysis và không có loader fixture test.

**Required fix / acceptance**

- Pin GeoLife archive/per-file hash và toàn bộ selection/preprocessing parameters.
- Raw selected-record table có UID, source byte identity, coordinate/time bounds.
- Fixture test tạo exact duplicate, near duplicate cùng user, near location khác
  user và boundary quanh 25m.
- O5 assert 40/40 UID và 1600/1600 composite keys.

### R5-005 — P1 paired design đúng ở code nhưng chưa được test end-to-end

`experiments/run_averaging_multi.py:115-122` tách đúng hai streams:

```python
mech.rng = rng_from_key("avg-mech", SEED, eps, m, hid, s)
jitter_rng = rng_from_key("avg-jitter", SEED, eps, hid, s)
```

Verifier xác nhận 200 normal draws giống byte-for-byte giữa năm mechanisms cho cùng
home/seed và đảo/chèn mechanism không đổi stream cũ.

Nhưng `tests/` không reference `avg-jitter`, `load_stay_points`, paired inputs hoặc
runner output permutation. `test_rng_seed_order_independent` chỉ test helper key
derivation, không test actual experiment inputs/results.

**Required fix / acceptance**

- Extract runner sampling plan thành testable function hoặc ghi `input_fixture_hash`
  theo `(home_id, seed, epsilon)`.
- Test đảo mechanism order, chèn mechanism mới và assert input trace của mechanisms
  cũ giống tuyệt đối.
- Mechanism RNG phải khác nhau theo mechanism; exogenous jitter phải giống nhau.
- O5 lưu pairing evidence đủ để audit mà không reverse-engineer RNG.

### R5-006 — P1 SM theorem/implementation/test alignment còn một mismatch thật

Ch4 đã cải thiện đáng kể: định nghĩa cell metric, cặp `C,C'`, common public history,
và phân biệt empty history REM với non-empty history T-REM. Đây là phần nên giữ.

Tuy nhiên `core/mechanisms.py:284-290` vẫn nói khi cache của cell rỗng thì transcript
là post-processing của một sample:

```text
r_C ~ REM(rep(C))
```

Production `StayMemoizedREM` dùng T-REM khi temporal history tồn tại dù cache của
cell C chưa có. “Empty cache” không tương đương “empty temporal history”.

Tests hiện verify same-cell reuse, A-B-A cache revisit, reset và corner geometry,
nhưng chưa verify:

- first miss with empty temporal history samples REM at `rep(C)`;
- first miss with common non-empty public history samples T-REM at `rep(C)`;
- raw secret `x` bên trong cell không lọt vào cache-miss sampler;
- both `C` and `C'` absent from the common cache in theorem scope;
- edge-adjacent boundary case (Ch4 prose nói corner/edge nhưng test chỉ corner).

**Required fix / acceptance**

- Đồng bộ class doc với Ch4: `REM` iff `h=empty`, otherwise `T-REM` under the
  same common public history.
- Mock/spy sampler arguments để mutation `rep(C) -> raw x` chắc chắn fail.
- Thêm empty/non-empty history tests và explicit cache-state preconditions.
- Không gọi exact memoization có arbitrary-trajectory Geo-I; revisit pattern vẫn là
  deterministic side channel ngoài narrow theorem.

### R5-007 — P1 PR budget validation vẫn nhận kiểu không mong muốn

`_require_positive_finite()` convert `float(value)` chỉ để check rồi bỏ normalized
value. Constructor tiếp tục lưu/arithmetics trên object gốc.

Probe matrix:

| Input | Hiện tại | Mong muốn |
|---|---|---|
| `0`, negative, `NaN`, `+/-inf` | reject | reject |
| numeric string | constructor/path khác nhau, không contract rõ | reject hoặc normalize nhất quán |
| `False` | reject do 0 | reject vì bool |
| `True` | **accept như 1.0** | reject vì bool không phải budget |
| `Decimal(...)` | có thể construct rồi crash ở `perturb` | reject hoặc normalize sang float |

Existing `test_pr_constructor_rejects_footguns` chỉ test missing budget, positional
budget và mixed `step_cap + component`; không test các invalid numeric/type cases.

**Required fix / acceptance**

- Validator trả normalized float và constructor dùng giá trị trả về, hoặc enforce
  một explicit numeric protocol xuyên suốt.
- Reject `bool` trước `float()`.
- Parameterized tests cho `None`, bool, string, Decimal policy, 0, negative,
  NaN/inf và overflow sum.
- Với mọi accepted value, object phải chạy ít nhất một `perturb` và advertised
  `privacy_cost_per_step_max` phải finite/positive đúng tổng.

### R5-008 — P1 HMM: code sign đúng, test/claim parity chưa đúng

`evaluation/attacks.py:212-215` hiện thực đúng:

```python
loge = -scale * epsilon * distance
loge = loge - logZ[candidate]
```

Independent analytic comparison xác nhận current online update khớp formula.
Nhưng test `test_hmm_online_uses_logZ_like_offline` chỉ assert kết quả “with logZ”
khác “without logZ”; mutation `-logZ -> +logZ` vẫn có thể tạo khác biệt và pass.

Ngoài ra `web/simulator.py:73-74` gọi online causal forward-only attacker
“IDENTICAL” với offline benchmark. Emission normalization giống nhau, nhưng causal
filter không identical với offline forward-backward smoother vì smoother dùng future
observations.

**Required fix / acceptance**

- Tiny analytic fixture tính posterior expected chính xác và assert từng candidate
  probability/estimate; mutation `+logZ` phải fail.
- Dùng wording “same normalized emission model” thay “identical attacker”.
- Tách rõ online filtering metric và offline smoothing metric trong UI/result schema.
- Ch5 không suy “mechanism riêng tư hơn” chỉ vì một proxy attacker có error cao hơn;
  chỉ nói attacker-specific empirical resistance trong exact setup.

### R5-009 — P1 provenance cải thiện nhưng command/data/schema chưa replayable

**Positive evidence**

- `begin_run()` được gọi trước loop trong cả hai official runners.
- `started_at_utc`, `finished_at_utc`, interpreter, cwd, platform, Python và resolved
  package versions được capture prospectively.
- Benchmark aggregate row có thêm HMM numerator/denominator.

**Gaps**

1. `experiments/provenance.py:99` tạo:

   ```python
   [sys.executable, "-m"] + sys.argv
   ```

   Với `python -m experiments.run_benchmark`, `sys.argv[0]` thường là filesystem
   path của module, nên recorded command thành `python -m /path/run_benchmark.py`,
   không replayable. Với `python -c`, probe cũng thành `python -m -c ...`. Dùng
   `sys.orig_argv` hoặc explicit canonical command do runner khai báo.
2. Dataset block tự ghi per-file hashing là future work. Local GeoLife archive SHA-256
   verifier thấy là `1107c5ac064d0a23c8d021a8736a77e53abc75b227062e6260342c6a8d86bdb6`,
   nhưng output không pin nó.
3. Loader bbox/interval/gap/stay/dedup/cap parameters, selected full coordinates và
   source time bounds chưa được pin.
4. Schema vẫn là `msc-experiment-v1` dù required fields/metric semantics đổi.
5. `hmm_covered/hmm_total` nằm ở prospective aggregate row, không ở raw
   per-trajectory rows; raw rows vẫn chưa reconstruct mọi table cell.
6. Không có schema/minimum-field validation test.

**Required fix / acceptance**

- Record replayable argv plus human canonical command; test round-trip parsing.
- Version schema và fail closed trên required fields/type/semantics.
- Pin data bytes + loader configuration + selected-record identity.
- Put HMM numerator/denominator at raw trajectory grain, then derive aggregate only
  from raw.
- Test raw-to-aggregate reconstruction và provenance completeness.

### R5-010 — P1 repo-wide manuscript/docs/web consistency chưa hoàn tất

| File/lines | Residual issue | Required wording/action |
|---|---|---|
| `thesis/chapters/ch5_thucnghiem.tex:51,60,115-116` | higher proxy error được gọi là “more private” | attacker-specific empirical resistance; không privacy ordering |
| `ch5:69-111` | O4 vertex table và 41–54% off-road | replace only from O5 edge metric |
| `ch5:150-151,166` | MLE gọi exact/consistent quá rộng | limit to vertex-secret, no-jitter, iid, full-search assumptions |
| `ch5:63-64` | nói provenance complete | liệt kê exact unresolved data/env limits |
| `ch5:234-238` | Eclipse gọi closest competitor chưa systematic | comparator table hoặc narrow wording |
| `docs/attack_scenarios.md:254,282` | stale off-road values | O5 values + estimand |
| `docs/attack_scenarios.md:286` | epsilon-T composition bị mô tả như đẩy point xa | composition là privacy bound, không utility mechanism |
| `docs/attack_scenarios.md:321` | absolute novelty claim | bounded overlap/difference statement |
| `docs/research_notes.md:88,93` | REM/T-REM nói close attacks | distinguish metric/structure from attack proof |
| `docs/research_notes.md:102-125` | old values called authoritative | update from O5 and cite artifact identity |
| `docs/research_notes.md:135` | absolute novelty | remove |
| `docs/system_model_and_threats.md:51-52` | fixed support được gán finite-radius usefulness delta=1 | state exact support/domain limitation |
| `system_model_and_threats.md:109` | REM/T-REM nói improve epsilon-T ceiling | no; per-step structure does not remove composition |
| `system_model_and_threats.md:127-128` | T-REM conflated predictive skip/reuse | T-REM temporal bias is not private predictive reuse |
| `system_model_and_threats.md:171` | T-REM nói fill trajectory guarantee | narrow empirical temporal prior only |
| `system_model_and_threats.md:260-263` | metrics “prove” attacks; PR called memoized | proxy evidence; PR is probabilistic previous-release reuse |
| `docs/problem_formulation.md:68-70` | SM first release chỉ gọi REM | REM/T-REM conditioned on common public history |
| `problem_formulation.md:77-79` | says memoization decision does not leak | explicitly state secret-dependent revisit channel |
| `web/simulator.py:157-164`, template `:178` | numeric epsilon-T shown cho SM/baseline | render `N/A` or “reference only, not a bound” |

Repo-wide sweep phải là search-driven, không chỉ sửa Ch5. Acceptance là không còn
stale number/absolute claim trong README, thesis, docs, code docstrings hoặc UI.

### R5-011 — P1 citation metadata/related-work classification sai

1. `thesis/refs.bib:73` ghi first author `Furkan Simitçioğlu`. Record chính thức là
   **Muhammed Esad Simitçioğlu** và M. Emre Gürsoy: [ScienceDirect article,
   Computer Networks 284 (2026), 112333](https://www.sciencedirect.com/science/article/pii/S1389128626003452).
2. `thesis/refs.bib:66` có author list sai cho arXiv:2511.21020. Record chính thức
   liệt kê Minghui Min, Jiahui Liu, Mingge Cao, Shiyin Li, Hongliang Zhang, Miao
   Pan và Zhu Han: [official arXiv record](https://arxiv.org/abs/2511.21020).
3. `ch4_phuongphap.tex:283-286` và `ch6_tongket.tex:18-20` gọi road PTPPM
   “stateless”. Bài có fresh perturbed PF draw/no memoization-or-replay defense,
   nhưng posterior/filter state vẫn được propagate. Wording an toàn là mô tả đúng
   dimension cần so, không gắn nhãn stateless tuyệt đối.
4. Comparator-dimension table vẫn thiếu. Cần map SM vs Memoization/Replay và PR vs
   Replication/predictive mechanisms theo state, trigger, output domain, metric,
   guarantee, attacker/threat và evidence.

**Acceptance**

- BibTeX khớp official metadata; clean build bibliography hiển thị đúng.
- Không còn “chưa ai làm/closest/stateless” nếu chưa có systematic evidence.
- Comparator table phân biệt overlap và thesis-specific delta, không dựng strawman.

### R5-012 — P2 fixed-public theorem test comments còn overclaim

Main Ch4 event inequality hiện xử lý support zero đúng. Tuy nhiên
`tests/test_rejection_conditioning.py` header vẫn nói ratio cho mọi event; với
`P_x(S)=P_x'(S)=0`, ratio không được định nghĩa và phải dùng event inequality.

Witness ratio khoảng `4.2` tại epsilon `ln(3)` chỉ chứng minh epsilon-bound không
đủ. Nó không chứng minh coefficient 2 là necessary/tight; observed exponent ở
fixture đó khoảng `1.306 * epsilon`, nhỏ hơn `2 * epsilon`.

**Required fix**

- Rewrite comments/test names theo event inequality.
- Nếu claim tightness, phải có family tiến tới bound 2 hoặc proof lower bound;
  nếu không, chỉ gọi `2epsilon` là safe upper bound và witness là counterexample
  cho `epsilon`.

### R5-013 — P2 graph artifact local pass, clean rebuild vẫn không fail closed

Local graph/manifest hiện khớp:

```text
GZ SHA-256       ed0af579...
XML SHA-256      5691186b...
graph SHA-256    9e44ec2a...
nodes/edges/WCC  13,813 / 41,040 / 1
```

Nhưng `data/build_beijing_graph.py` vẫn dùng rolling URL, hiện không thực hiện
download như README/docstring nói, không compare expected canonical source hash và
có thể ghi manifest cho bất kỳ local bytes. `assert_graph_matches_manifest` không
enforce required schema: independent probe nhận cả manifest chỉ có `nodes`, chỉ có
đúng graph hash, thậm chí `{"unrelated":"x"}`.

**Acceptance**

- Immutable source URL/version + expected source SHA checked trước parse.
- Manifest required keys/schema fail closed: source/graph hashes, nodes, edges,
  WCC, extent, profile.
- Negative tests cho missing/wrong keys.
- README recipe được chạy từ clean clone/cache và khớp artifact identity.

### R5-014 — P2 environment và CI chưa reproducible

- Canonical project venv: `23 passed, 0 failed`.
- README-style `python3 -m tests.run_all` fail trong current shell vì thiếu
  `networkx`.
- `venv/bin/python -m pytest -q tests` fail vì venv không có pytest.
- Không có CI workflow.
- `requirements.txt` vẫn là broad lower bounds, không trực tiếp pin toàn bộ
  transitive stack; `requirements-dev.txt` chỉ thêm pytest.
- Shapely 1.8/2 STRtree break là ví dụ cụ thể cho việc broad bounds không bảo đảm
  executable semantics.

**Acceptance**

- Một lock/constraints artifact được generate và documented cho supported Python.
- Fresh install từ artifact đó chạy canonical + production + rejection tests.
- CI chạy cùng command, lưu environment/provenance evidence.
- README không quảng cáo command không có dependency recipe tương ứng.

### R5-015 — P2 statistical evidence vẫn exploratory

- Moving benchmark: 20 trajectories, 858 points, chỉ 7 users với counts
  `3,3,3,3,3,3,2`, một root stream, equal-weight trajectory mean; không multi-seed
  hoặc user-cluster CI.
- Averaging prospective population: 40 locations/21 users; median bootstrap có
  cluster theo user, nhưng success probabilities vẫn pooled/no CI.
- `N_BOOT`/`BOOT_SEED` được ghi config, trong khi calls dùng defaults của
  `bootstrap_ci`; hiện cùng `1000/0` nhưng có thể silently diverge nếu constant đổi.
- Ch5/Ch6 đã ghi exploratory, nên đây là **NARROWED/PARTIAL**, không phải failure
  nếu thesis giữ đúng framing.

**Required fix / acceptance**

- Truyền explicit `N_BOOT`/`BOOT_SEED` vào function và test config/runtime parity.
- Report user counts, weighting unit, root seeds và uncertainty ở mọi headline table.
- Cluster/user bootstrap hoặc sensitivity/multi-root analysis cho success metrics.
- Không gọi result confirmatory/generalizable nếu evidence vẫn ở quy mô trên.

### R5-016 — P2 PDF source fix pass, committed artifact/semantic QA fail

Verifier clean-built từ `git archive 1293304` bằng XeLaTeX/BibTeX:

```text
result                     success
pages                      33
undefined citations/refs   0
List of Tables overflow    fixed by short captions
```

Visual inspection các trang List of Tables, Ch5 tables và bibliography xác nhận
layout source tốt hơn. Compile vẫn báo overfull boxes khoảng `24.8pt`, `17.7pt`,
`8.9pt` và một số minor warnings. Quan trọng hơn, clean C5 source build vẫn chứa
semantic content O4: old table values, old 41–54% narrative và bibliography author
sai. Committed PDF thậm chí chưa phải clean build đó; nó vẫn 32 trang/O4.

**Acceptance**

- Build PDF sau O5 từ clean source tree.
- Render toàn bộ pages, không chỉ compile log; inspect List of Tables, both Ch5
  tables, long equations, references và conclusion.
- Không undefined refs/citations, không material clipping/overflow.
- Semantic spot-check truy ngược mỗi headline number tới O5 JSON/raw rows.

## 4. Raw-to-aggregate và artifact integrity

Committed O4 hiện internally reconcilable:

- benchmark: 360 raw rows; max difference giữa aggregate và rounded raw
  `1.821013e-05` qua 180 metric cells;
- averaging: 70 median/CI cells khớp sau rounding 0.1m, 15 success cells khớp;
  max pre-round discrepancy `0.04995`.

Điều này chỉ chứng minh aggregate được tính nhất quán từ O4 raw rows. Nó **không**
chữa được O4 duplicate identities, mechanism-specific jitter hoặc changed metric.
Internal reconciliation và external validity là hai gate khác nhau.

O5 phải nâng grain/provenance để mọi table cell, kể cả HMM coverage, được derive từ
raw records có primary key rõ ràng.

## 5. Thứ tự sửa đề xuất cho Claude Code

### Phase C5.1 — sửa source trước khi chạy lại experiment

1. Fix Shapely support policy + point-to-edge unit tests.
2. Fix PR normalized validation + invalid-budget matrix tests.
3. Align SM class doc/kernel; add representative/raw-input, common-history và
   cache-state tests.
4. Make HMM test analytic/mutation-sensitive; fix online/offline wording.
5. Add stay/dedup and paired runner-level fixtures.
6. Fix provenance replayable command, schema, required data/loader/raw fields.
7. Make graph manifest schema fail closed; align builder/README.
8. Correct BibTeX metadata and sweep stale/absolute claims repo-wide.
9. Add locked environment/CI or explicitly narrow supported environment.

Không sửa implementation theo verifier rồi chạy O5 trong một dirty tree. Commit
C5.1 trước; record exact hash.

### Phase O5 — regenerate evidence từ clean C5.1

1. Checkout/verify clean C5.1; record `git status --porcelain` empty.
2. Run canonical tests in locked environment.
3. Run benchmark và averaging with provenance captured before work.
4. Assert:

   ```text
   source_commit == C5.1 hash
   source_dirty_before_run == false
   selected UID == 40/40 unique
   raw averaging key == 1600/1600 unique
   paired input trace/hash equal across mechanisms
   point-to-edge semantics recorded
   required provenance/schema fields present
   ```

5. Reconstruct every aggregate/table cell independently from raw rows.
6. Update manuscript tables/narrative from O5 only.
7. Clean-build and render PDF; commit O5 artifacts separately.

### Phase R5 — response file

Response nên map từng `R5-001...R5-016` tới:

- exact source commit/line;
- test name và failure-before/pass-after evidence;
- artifact hash/source provenance;
- claim wording đã thay;
- status `RESOLVED`, `NARROWED` hoặc `PLANNED` trung thực.

Không đánh dấu `RESOLVED` chỉ vì source code đã thay nếu O5/PDF acceptance chưa có.

## 6. Acceptance checklist cho vòng verify tiếp theo

### Gate G0 — formal

- [ ] Ch4 và class docs cùng phát biểu REM/T-REM cache-miss theo common public history.
- [ ] Fixed-support statements dùng event inequality, không ratio `0/0`.
- [ ] Không claim coefficient-2 tightness nếu chỉ có epsilon counterexample.
- [ ] SM/PR claims tách exact memoization, private test và trajectory scope.

### Gate G1 — production/tests

- [ ] Canonical suite xanh trong fresh locked environment.
- [ ] Long-edge and curved-geometry tests fail nearest-vertex mutation.
- [ ] Stay dedup/UID fixture covers exact/near/cross-user/boundary cases.
- [ ] Paired runner trace invariant under mechanism reorder/insertion.
- [ ] SM raw-input mutation and HMM `+logZ` mutation fail.
- [ ] Every accepted PR budget value survives a real perturb call.

### Gate G2 — provenance/data

- [ ] Replayable command/argv recorded correctly.
- [ ] GeoLife bytes, graph bytes, loader parameters and selected identities pinned.
- [ ] Versioned schema with required-field validator/negative tests.
- [ ] 40/40 UIDs and 1600/1600 raw composite keys.
- [ ] Raw trajectory rows reconstruct HMM numerator/denominator and every table cell.

### Gate G3 — evaluation/statistics

- [ ] O5 uses point-to-edge metric and names weighting estimand.
- [ ] HMM filter vs smoother distinguished.
- [ ] Proxy errors not presented as formal privacy ordering.
- [ ] Uncertainty/clustering/seed limitations visible next to result claims.

### Gate G4 — manuscript/literature/PDF

- [ ] No stale O4 values or 41–54% narrative remains.
- [ ] BibTeX author metadata matches official sources.
- [ ] Comparator table covers state/trigger/domain/metric/guarantee/threat.
- [ ] No unsupported “chưa ai làm”, “closest”, “stateless” or “proves attack”.
- [ ] PDF is rebuilt from O5, all pages rendered and visually/semantically checked.

## 7. Commands/evidence log

Core commands used by verifier:

```bash
git status --short --branch
git show --stat --oneline 1293304
git diff --check 1fef76a..1293304
git diff --name-status 1fef76a..1293304
git rev-parse 1fef76a:<artifact>
git rev-parse 1293304:<artifact>
venv/bin/python -m tests.run_all
venv/bin/python tests/test_production_mechanisms.py
venv/bin/python tests/test_rejection_conditioning.py
pdfinfo thesis/main.pdf
```

Ngoài canonical suite, verifier chạy read-only probes cho stay selection/identity,
paired RNG draws, straight/curved edge geometry, real graph vertex coverage,
Planar edge-rate recomputation, invalid budget types, HMM analytic sign, manifest
missing-key behavior, raw-to-aggregate reconciliation và clean XeLaTeX build/render.

### Verification limits

- Verifier không tạo O5 và không sửa candidate implementation.
- Full expensive experiment suite không được gọi là pass vì artifact O5 không tồn tại.
- Web literature checks chỉ dùng official publisher/arXiv records cho metadata và
  overlap classification; không dùng search snippet làm evidence.
- Timing microbenchmarks là diagnostic trong current machine/environment, không là
  performance guarantee.

## 8. Final verdict

`1293304` là **một bước C5 có giá trị**: nó sửa đúng identity selection, pairing,
edge metric implementation, central event inequality, online HMM normalization,
provenance timing và một phần claim positioning. Tuy vậy, package vẫn ở trạng thái:

```text
C5 source correctness       PARTIAL / substantial progress
C5 automated protection    PARTIAL
O5 numerical evidence      ABSENT
O5 manuscript/PDF          ABSENT
submission-ready result    NO
```

Handoff đúng là: **sửa C5.1 theo R5-002...R5-016, commit sạch, rồi tạo O5 và đưa
cả source + raw outputs + tables + rendered PDF qua verify tiếp theo**.
