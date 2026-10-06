# Hướng nâng phương pháp Geo-I thành bài JISA

Đánh giá ngày **06/10/2026**. Chưa có deadline/CFP mới, nên thiết kế dưới đây
tạm nhắm research article thông thường. **Đúng chủ đề, nhưng evidence hiện tại
chưa đủ để chốt manuscript.** Giữ Geo-I, mạng đường, protected belief và luồng
Z → Q → POI → lọc local. Các kết quả trước được giữ nguyên.

## 1. Tạp chí trong ảnh và yêu cầu thực tế

Ảnh là **Journal of Information Security and Applications (JISA)** của Elsevier,
không phải hội nghị. Scope chính thức có privacy và network/mobile security,
nhấn mạnh đóng góp kỹ thuật nguyên gốc gắn với ứng dụng/thực nghiệm.
Đó là lý do đề tài phù hợp; không có ngưỡng Recall/MAE nào bảo đảm được nhận.
[Scope chính thức](https://shop.elsevier.com/journals/journal-of-information-security-and-applications/2214-2126).

Special issue trong ảnh cập nhật năm 2022. Chưa xác nhận đợt nhận bài hiện tại;
ngày cập nhật không phải deadline 2026. Guide for Authors trả 403 trong lần
kiểm tra này, nên format/CFP còn cần xác minh trước khi nộp.
[Audit venue và các bài JISA gần đây](../../research/2026-10-06_jisa_venue_evidence.md).

## 2. Bài nên kể một câu chuyện nghiên cứu rõ

**Tên làm việc:** *Road-Aware Geo-Indistinguishability for Multi-Purpose POI
Queries under a Persistent Privacy Budget*. Tên chưa là tuyên bố đã chứng minh
novelty; chưa dùng “optimal”, “anonymous” hoặc “protects all ten scenarios”.

**Câu hỏi chính:** với cùng cap Geo-I và cùng chi phí truy vấn, làm sao chọn
tập Q di chuyển hợp lý để người dùng nhận kết quả POI tốt cho nhiều mục đích,
kể cả lúc mới bắt đầu và những chuyến khó, mà server không nhận purpose thật?

Ba phần đóng góp cần hoàn thiện:

1. **Thiết kế chọn Q có ràng buộc:** dùng phân bố vị trí đã bảo vệ và mạng
   đường công khai; cải thiện coverage của nhiều purpose và phần utility thấp.
   Phải vượt được planner theo coverage trung bình hiện tại ở trade-off đã định.
2. **Phân tích đúng server view:** kế thừa Geo-I, chứng minh composition qua
   nhiều chuyến và purpose noninterference có điều kiện. Chỉ rõ GPS/Z/local P,
   account/IP/click/timing nằm ở đâu trong threat model.
3. **Đánh giá có thể bác bỏ thiết kế:** real mobility + SUMO, đối chứng cùng
   budget, metrics chung và metrics gốc hợp lệ, attacker mạnh, tail/cold-start
   utility và chi phí. Giữ các trường hợp thua và giới hạn.

Geo-I, reuse, Bayesian recovery, dummies, ledger và cache đều có prior art.
Ghép chúng không tự thành đóng góp mới. Cần đọc kỹ các nguồn gần nhất trong
[audit formal/novelty](../../research/2026-10-06_jisa_formal_audit.md), đặc biệt
predictive Geo-I, road/graph mechanisms, remapping và các phương pháp LBS gần đây.
[Follow-up nguồn chính IWPE’26](../../research/2026-10-06_jisa_novelty_followup.md)
xác nhận một paper rất gần về road awareness/mobility prediction; full paper
còn cần đọc trước khi chốt novelty.

## 3. Evidence hiện tại cho thấy gì?

| Vấn đề | Điểm đã có | Khoảng trống cần khắc phục |
|---|---|---|
| Utility | Cache POI static theo phiên bản đạt Recall 99,34%; cuối phiên 400–600s đạt 99,48% trên native cohort đã xem | Chuyến đầu trung bình 94,73%; session tệ nhất ở cửa sổ 400–600s đạt 86,67%. Đây là temporal tail, khác distributional lower-tail CVaR đề xuất bên dưới. Reset cũng được lợi từ cùng cache, nên đây không là ưu thế riêng Geo-I. Chưa đo live availability. |
| S7 | Bốn purpose lọc locally; với cùng GPS/lịch, request không đổi theo purpose | Intent có thể suy từ tuyến đường/lịch sử. Chưa có đánh giá intent tương quan mobility và full-history attacker. |
| S9/S10 | Không delay; Endpoint20 tăng S10 MAE khoảng 579m so với Slack20 cùng L20 trên development | Recall giảm khoảng 2,30 điểm %. Hit100 đều 0 nên cần Hit500, CDF và auxiliary endpoint attacker; chưa là xác nhận mới. |
| S4–S6 | Persistent cap; có raw control và attacker theo task | Nhóm nhỏ, binary future task và history giới hạn. AUC gần 0,5 của một selector không chứng minh không liên kết được; chưa có real-person/vehicle holdout. |
| Comparators | Bảng cũ cùng task/protocol; bổ sung điều kiện dùng native metrics | Một số phương pháp là adaptations; chưa được gọi là faithful SOTA. EIE cùng estimator trùng MAE, không là một bằng chứng độc lập. |
| Lý thuyết | Có proof sketch cho ideal REM/filter/epoch/postprocessing | Float/PRNG simulator chưa là triển khai pure Geo-I được chứng nhận. Cap số học không đồng nghĩa inference yếu. |

**Gate quan trọng của dịch vụ static:** catalogue hiện chỉ 418 POI và public
planner/context đã biết id/category/coordinates/signatures. Cần đối chứng lấy
toàn catalogue một lần rồi trả lời locally: trong contract static này có thể
không cần truy vấn vị trí lặp lại. Chưa chạy/timing baseline đó nên không ghi
điểm hay chi phí đo giả định. Recall gần 100% từ warm cache không đủ chứng minh
một hệ LBS cần Geo-I tốt hơn phương án local-only.

Trước luận điểm application cần: explicit provider/API contract và bulk-access
assumptions; catalogue lớn/cross-region; trạng thái availability/price/travel
time thực sự thay đổi và cần remote fetch. Với dynamic service, planner chỉ
dùng public proxy/model hoặc replies đã nhận, không được biết counterfactual
current provider records tại mọi Q. Local-only/bulk controls và service-success/
freshness phải được đánh giá cùng chi phí. Đây là mở rộng task/evaluation có
lý do, vẫn giữ Geo-I làm cơ chế bảo vệ GPS.

Nguồn: [report và readout đã pin](../../supervisor_meeting/2026-10-06_brief/README.md),
[audit thực nghiệm](../../research/2026-10-06_jisa_evaluation_audit.md).
Các cohort trên đã tham gia phát triển; gọi đúng là **development**, kể cả tạo
draw mới hoặc đổi tên test. Không dùng chúng làm confirmation chưa từng xem.

### Đối chứng cùng cap vừa hoàn tất

[REM/Planar readout đã kiểm tra](../../../artifacts/benchmarks/jisa_native_anchor_ablation_20261006_v1/derived_readout_v2.json)
và [final-code verification](../../../artifacts/benchmarks/jisa_native_anchor_ablation_20261006_v1/validation_recheck.json).
Sáu nhóm native test đã xem, draw mới; cùng C=0,23/m, N8/H12, K5/L20, clocks,
map, service và matched belief theo primitive. Q planner vẫn dùng signature L10
cho cả hai; live availability chưa được chấm. Attacker học/chọn riêng bằng cùng
bank/rule trước test; không chọn cấu hình defense từ điểm test.

| Metric | REM Geo-I | Planar Geo-I | Chênh REM−Planar, diagnostic CI 95% |
|---|---:|---:|---|
| Recall, chỉ phản hồi hiện tại | 94,69% | 97,00% | −2,306 điểm % [−3,665; −0,842] |
| Recall, cùng cache static | 99,91% | 99,35% | +0,553 điểm % [−0,077; +1,720] |
| Recall chuyến đầu, cùng cache static | 99,23% | 94,75% | +4,480 điểm % [−0,627; +13,925] |
| S5/S6 accuracy sau rẽ, task hai ứng viên | 50% | 50% | 0; CI [−33,3; +33,3] điểm % |
| S6 destination MAE sau rẽ | 614,93m | 501,65m | +113,275m [−253,183; +497,305] |

Raw sau rẽ đạt 100% accuracy; trước rẽ Raw cũng chỉ 50%, là ambiguity của
task. S5/S6 dựa cùng lựa chọn hai ứng viên, không là hai bằng chứng độc lập.
CI paired bootstrap theo sáu family, tính posthoc, chưa điều chỉnh multiplicity
hoặc lặp nhiều private draw; chưa dùng để chốt significance/confirmation.
[Paired diagnostic và nguồn tính](../../../artifacts/benchmarks/jisa_native_anchor_ablation_20261006_v1/paired_differences.json)
lưu toàn bộ family deltas; không là phép chọn defense từ test.

Cả hai có 7.550 Q requests ở test. REM: request 1.140.292B + reply 41.235.772B;
Planar: 1.140.457B + 41.224.091B, là compact JSON estimates, chưa HTTP/TLS.
Cap và số requests bằng nhau; byte totals gần nhau nhưng không chính xác bằng.
Kết quả định vị điểm cần cải tiến: current-reply coverage của REM còn kém,
cache đem lợi ích cho cả hai, và chênh privacy chưa rõ. Không gộp draw này với
con số 99,34%/86,67% của cohort readout trước hoặc chọn riêng run đẹp.

## 4. Cải tiến thuật toán đáng làm trước

**Ưu tiên: chọn Q để giảm những trường hợp utility thấp.** Planner hiện tối ưu
coverage trung bình; vùng phổ biến có thể lấn át vùng ít gặp và purpose khó.
Cache giúp các chuyến sau nhưng không giải quyết đầy đủ chuyến đầu.
Trước khi thêm objective mới, align signature response với L20 thật và
reference top-k=5/multiple purposes, thay vì dùng planner L10 của pilot.
Kiểm tra riêng thay đổi service oracle này để tách lợi ích khỏi CVaR.

Candidate sau đây **chưa được tích hợp hoặc có kết quả benchmark**. Với vị trí
khả dĩ x và purpose công khai g, đặt f(x,g,Q) là tỷ lệ reference POI top-k được
phủ bởi hợp các response top-L từ Q. Các purpose/radius/destination prototypes
và trọng số phải là public/shadow workload đã chốt, không là QuerySpec thật.

Đã có [prototype objective và optimizer riêng](../../../benchmark/risk_aware_cover.py)
với [21 checks](../../../tests/test_risk_aware_cover.py): fractional CVaR, N/A,
trade-off, mean floor, per-track feasibility, deterministic ties và ví dụ
local optimum khác global optimum. API không nhận GPS/purpose/clock/network;
caller vẫn phải chứng minh nguồn public/protected của profiles. Prototype
chưa thay engine Geo-I, chưa tham gia pilot REM/Planar, chưa chứng minh tăng
Recall hoặc privacy trong dataset thật.

Đề xuất tối ưu:

\[
J_\lambda(Q)=(1-\lambda)\mathbb E[f]+\lambda\operatorname{LCVaR}_a(f),
\quad
\operatorname{LCVaR}_a(f)=\max_\eta\{\eta-\mathbb E[(\eta-f)_+]/a\}.
\]

Ở đây 0<a≤1, 0≤λ≤1; expectation lấy theo protected belief nhân public purpose/
prototype weights đã khóa. Reference rỗng giữ N/A/completion riêng, không gán
utility 0 hoặc 1 để thay denominator. LCVaR là utility trung bình của phần xác
suất a có kết quả thấp nhất, gồm
fractional mass khi phân bố rời rạc. Ví dụ hai phương án có utility các nhóm
bằng nhau về xác suất: A=[1;1;1;0,6], B=[0,85;0,85;0,85;0,85]. A có trung
bình 0,9 nhưng nhóm khó chỉ 0,6; B có trung bình 0,85 và nhóm khó cũng 0,85.
Với a=0,25, objective chọn B thay A khi λ>1/6; đây là trade-off phải chọn trước
test, không phải CVaR luôn chọn B.

CVaR là công cụ kế thừa, không phải phát minh của nhóm. Công thức utility ở
đây đổi dấu từ công thức loss của
[Rockafellar–Uryasev](https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf).
Không chuyển nguyên bound greedy/submodularity của objective cũ sang objective
mới. Cần thiết kế optimizer hữu hạn, so exact oracle trên map nhỏ, giữ phương
án mean-coverage làm fallback và kiểm tra coverage floor/slack.
Floor này áp vào objective theo belief/prototypes, chưa bảo đảm Recall tại GPS
thật; calibration/prior mismatch và utility thực tế phải được chấm riêng.

**Ranh giới bảo vệ giữ nguyên:** planner chỉ dùng Z/protected history, belief,
public graph/catalogue/model và public clock. Không đọc GPS/purpose thật để
chọn Q, đổi K/L/lịch, refill budget hay quyết định gửi lại. Local top-k vẫn
dùng GPS/purpose thật trên thiết bị. Học prior/calibration trên cohort riêng;
đánh giá độ lệch prior và purpose ngoài bộ prototypes.

**Phép thử bác bỏ:** nếu candidate không cải thiện cold-start/lower-tail ở
cùng cap/traffic, hoặc tăng attacker success đáng kể, không nhận là đóng góp.
Điều chỉnh chỉ trên train/selection; lưu kết quả thất bại và khóa trước nguồn mới.

## 5. Chương trình thí nghiệm theo thứ tự

| Bước | Công việc và đối chứng | Điều kiện chuyển bước |
|---|---|---|
| 0 — application gate | Static catalogue prefetch/local-only; catalogue-size sweep; API/bulk limits; dynamic provider workload và public-proxy planner | Chứng minh nhu cầu remote service và điểm trade-off có ích so với offline/bulk control. Chưa có kết quả baseline này; không lấy warm-cache Recall làm novelty. |
| 1 — hoàn tất development | Matched REM vs full-plane Planar Laplace: cùng C=0,23/m, N8/H12, K5/L20, graph, service, clocks, planner; cùng cache hiện tại/static, attacker bank học riêng | Final-code verification pass; bảng ở trên cho thấy current-only còn kém và chưa có lợi thế privacy rõ. Chưa là faithful paper reproduction/confirmation. |
| 2 | Mean coverage vs risk-aware Q; public-prior vs protected belief; reuse on/off ở cap bằng nhau; endpoint strength sweep; cùng static cache cho mọi comparator | Chọn trên selection bằng rule công bố trước; báo Pareto privacy/utility/traffic, không đòi thắng mọi metric. |
| 3 | Acquisition/loader mobility thật: GeoLife cho người, Porto cho xe; SUMO kiểm soát fork/stops/POI. Tách source group và thời gian; không tách ticks cùng người/xe vào train/test | License/provenance, gap/time/order/duplicate checks, public map/POI độc lập với holdout GPS, đăng ký cohort và khóa code/config trước confirmation. |
| 4 | Attacker nhìn toàn history; S7 dùng intent–route correlation; S9/S10 dùng start/speed/time/direction/known roads; linkage có cả similarity và inverted-score family chọn trên selection | Raw có tín hiệu; permutation/ambiguity controls đúng; cùng cơ hội thông tin phụ cho mọi phương pháp. Không chọn hướng AUC theo test. |
| 5 | Confirmation mới; đo bytes, rounds, CPU/RAM; cold/warm cache, catalogue version changes và live POI riêng | Uncertainty theo family/person/vehicle độc lập; completion/N/A denominator rõ; không thay seed/family để bỏ run xấu. |

**Budget sweep dự kiến, chưa khóa:** C∈{0,02;0,05;0,10;0,23}/m; tính lại
u=C/[N(2H−1)]. Giữ mọi điểm trên đường trade-off, không chỉ điểm đẹp. Tại
C=0,23 và r=100m, bound transcript là exp(Cr)=exp(23)≈9,7×10⁹: hợp lệ về
accounting nhưng rất lỏng. Báo C·r và attacker thực nghiệm cùng utility, không
gọi giá trị C thấp bằng cách nhìn số thập phân.

**Metrics:** Hit100/Hit500/attacker MAE–CDF; S4 orientation chọn ở selection,
ROC-AUC + balanced accuracy; S5/S6 accuracy/top-k/rank tùy task; S7 purpose và
category accuracy so chance/class prior. Utility gồm Recall@5 theo purpose,
completion, cold-start, lower-tail và min-group; cost gồm bytes/requests/CPU/RAM.
Metrics gốc đi kèm đúng output/trust/task của từng paper. ASR cần real-member
candidate set; entropy cần posterior được định nghĩa; thiếu điều kiện thì N/A.
Không ép ciphertext hoặc synthetic database vào bảng coordinate Q-only.

Không coi số tick/POI là số mẫu độc lập. Cỡ cohort cần được chọn theo precision
của paired confidence interval/power trên development; đây là thiết kế của nhóm,
không phải mức tối thiểu JISA công bố. Hai datasets cũng chưa đủ nếu split rò
hoặc label person/vehicle không có ý nghĩa; taxi ID không xác nhận driver ID.

Utility native hiện giả định local ranking có vị trí hiện tại chính xác từ
evaluator. GPS reads trong ledger chỉ đếm probes của lớp bảo vệ/network,
không là phép đo mọi lần GNSS của thiết bị. Trước claim về ứng dụng/battery,
cần đo riêng cadence/accuracy của local GPS, lỗi định vị và chi phí cảm biến.

## 6. Checkpoint có thể kiểm tra bằng code

[study_design.json](study_design.json) ghi thiết kế **draft**, source/config/
contract/units và các dataset development đã biết. Chưa có cohort confirmation
được đăng ký; candidate risk-aware và full-history attack vẫn là công việc dự kiến.

```bash
python -m experiments.jisa_publication_preflight \
  --protocol docs/publication/jisa_20261006/study_design.json
python -m experiments.jisa_publication_preflight \
  --protocol docs/publication/jisa_20261006/study_design.json --check-confirmation
```

Lệnh đầu audit integrity; lệnh sau phải chưa cho qua khi chưa khóa code/config
và đăng ký actual fresh data. Không đổi `inspection` bằng suy đoán. Hashes không
chứng minh DP, novelty, fidelity thực chất hoặc việc con người chưa xem data.
[Audit draft hiện hành](../../../artifacts/publication/jisa_readiness_20261006_v3/preflight.json)
pass integrity và giữ ba điều kiện confirmation còn thiếu. Các draft trước
review metric/source được giữ trong `revisions/`, receipts cũ không sửa.
Khi sửa implementation, tạo revision protocol mới và kiểm tra lại pins; không
chỉnh receipt cũ để trông như đã pass từ trước.

**Sửa thuật ngữ quan trọng:** H12 là tham số quy đổi với U=2H−1=23 đơn vị;
first release chi và reserve 1; sau đó reuse chi 1, refresh chi 2 nhưng trước
GPS phải reserve đủ nhánh tốn 2. Nếu chạy đến lúc filter dừng, số lần đọc là
12–22 tùy nhánh; phiên ngắn có thể ít hơn. Không là hard limit 12. Report hiện hành đã
sửa prose và giữ bản trước audit. [Giải thích và enumeration](../../research/2026-10-06_h_accounting_correction.md).

## 7. Bố cục manuscript sau khi đủ evidence

1. Introduction: nhiều-purpose LBS, joint transcript và utility ở nhóm khó;
   khoảng trống cụ thể so prior art, ba contributions đã xác nhận.
2. Related work: Geo-I/road/predictive/dummy/query privacy, contract–metrics–trust;
   bảng scenario chỉ là taxonomy hỗ trợ, không là luận điểm chính.
3. System/threat model: server view, attacker knowledge, adjacency/D∞, units,
   epoch/device scope, static/live catalogue và logical timing assumptions.
4. Method: framed layered architecture, algorithm/filter/ledger, Q objective,
   public prototypes, local ranking/cache và computational complexity.
5. Analysis: inherited ideal Geo-I + composition/purpose statement, utility
   result đúng objective; numerical/sampler implementation scope riêng.
6. Evaluation: frozen split/config/banks, source-backed baselines, matched
   ablations, budget curves, real mobility/SUMO, uncertainty và deployment cost.
7. Limitations/conclusion: identity channels, intent correlations, dynamic POI,
   source/time mismatch và những trade-off còn chưa giải quyết.

Trước nộp: đọc lại current Guide for Authors, lập data/code availability theo
license, mô tả AI hỗ trợ và kiểm tra toàn bộ citations/tables. Các bước này
được đặt sau khi phương pháp và evidence ổn định; không dùng viết đẹp để bù
cho novelty hoặc confirmation còn thiếu.
