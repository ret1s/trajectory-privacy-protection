"""Author the next meeting report from pinned, separate experiment readouts.

This does not rerun attacks, change a dataset or overwrite a previous report.
Rates are stored as ratios in evidence and displayed as percentages here.
"""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SOURCES = {}


def read(path):
    raw = (ROOT / path).read_bytes()
    SOURCES[path] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def pct(value):
    return f'{value * 100:.2f}'.replace('.', ',') + '%'


def num(value, digits=0):
    return f'{value:,.{digits}f}'.replace(',', '|').replace('.', ',').replace('|', '.')


def p(text): return {'type': 'p', 'text': text}
def key(text): return {'type': 'key', 'text': text}
def bullets(*items): return {'type': 'list', 'items': list(items)}
def table(headers, rows, widths=None):
    result = {'type': 'table', 'headers': headers, 'rows': rows}
    if widths: result['widths'] = widths
    return result
def page(heading, *blocks): return {'heading': heading, 'blocks': list(blocks)}


def main():
    evidence = json.loads((OUT / 'evidence.json').read_text())
    paper = evidence['paper_v2']
    purpose = read('artifacts/benchmarks/geoi_purpose_refinement_20261005/readout.json')
    read('artifacts/benchmarks/query_purpose_20261005/snapshot/readout.json')
    cost = read('artifacts/benchmarks/native_cost_diagnostic_20261005/readout.json')
    identity = read('artifacts/benchmarks/session_budget_20261005/round3/readout.json')
    native = read('artifacts/benchmarks/future_native_20261005_v1/results.json')
    depth = read('artifacts/benchmarks/future_native_depth_20261005_v1/results.json')
    ttl = read('artifacts/benchmarks/native_static_cache_20261006_v1/results.json')
    static = read('artifacts/benchmarks/native_versioned_static_20261006_v1/results.json')
    robust = read('artifacts/benchmarks/endpoint_robust_selection_20261006/readout.json')
    endpoint = read('artifacts/benchmarks/endpoint_generalization_20261005/readout.json')
    read('artifacts/benchmarks/endpoint_generalization_20261005/protocol.json')
    read('artifacts/benchmarks/future_native_20261005_v1/protocol.json')

    work_rows = []
    for work in evidence['related_work']:
        work_rows.append([
            f"[{work['short_title']} ({work['year']})]({work['url']})",
            work['output_contract'],
            ', '.join(work['fully']) or '—',
            ', '.join(work['partially']) or '—',
            'Privacy: ' + '; '.join(work['native_privacy_metrics']) + '. Utility: ' +
            '; '.join(work['utility_metrics']) + '. Cost: ' + '; '.join(work['cost_metrics']) + '.'
        ])
    # Use short labels in the presentation; full definitions remain inspectable.
    work_rows = [
        [work_rows[0][0], 'Một vị trí nhiễu/mốc', work_rows[0][2], work_rows[0][3], 'EIE ↑; chênh chi phí tới đích ↓; chưa khóa metric vận hành'],
        [work_rows[1][0], 'GPS thật + dummy theo ngữ nghĩa/đường', work_rows[1][2], work_rows[1][3], 'ASR ↑; DER ↑ (dummy hợp lý); thời gian tạo ↓'],
        [work_rows[2][0], 'Tập thật + dummy, xen query giả', work_rows[2][2], work_rows[2][3], 'ASR/số đường khó phân biệt ↑; thời gian/RSS ↓; chưa chấm POI'],
        [work_rows[3][0], 'Beacon xe + tuyến giả, đổi pseudonym', work_rows[3][2], work_rows[3][3], 'Ghép đúng trace ρ ↓; số tuyến giả ψ ↓; chưa chấm POI'],
        [work_rows[4][0], 'Bộ trace tổng hợp offline', work_rows[4][2], work_rows[4][3], 'MI ↓, ε-DP; JSD phân bố ↓; độ phức tạp'],
        [work_rows[5][0], 'AGeoI + dummy; Edge trộn query', work_rows[5][2], work_rows[5][3], '(ε,δ)-AGeoI; CoP ↓, CoP=0 ↑; Wasserstein IBU ↓'],
    ]

    labels = {'unprotected':'GPS thật', 'dls_graph_adaptation':'DLS*',
              'transprotect_adaptation':'TransProtect*',
              'semantic_correlation_local_adaptation':'Semantic*', 'br_private':'**BR-Dummy / Geo-I**'}
    comparison_rows = []
    for method in paper['display_methods']:
        cells = []
        for row in paper['scenario_rows']:
            m = row['methods'][method]
            cell = f"{pct(m['hit100'])} / {num(m['mae_m'])} / {pct(m['recall_at_5'])}"
            cells.append('**'+cell+'**' if method == 'br_private' else cell)
        comparison_rows.append([labels[method]] + cells)
    utility_rows = []
    for row in paper['per_method_utility']:
        if row['method'] not in labels: continue
        u = row['macro_utility']
        utility_rows.append([labels[row['method']], pct(u['recall_at_5']),
                             num(u['payload_bytes_per_event']), num(u['generation_ms_per_event'], 2)])

    purpose_names = {'nearest_distance':'Gần nhất theo quãng đường', 'fastest_travel':'Đến nhanh nhất',
                     'within_radius':'Trong bán kính riêng tư', 'minimum_detour':'Ít vòng đường tới đích riêng tư'}
    purpose_rows = [[name, pct(purpose['test']['alpha000_L20']['utility']['by_purpose'][code])]
                    for code, name in purpose_names.items()]
    cache_rows = []
    for label, all_row, tail_row in [
        ('L10, phản hồi hiện tại', depth['results']['geoi_epoch8']['test']['10']['all_0_600'], depth['results']['geoi_epoch8']['test']['10']['tail_400_600']),
        ('L20, phản hồi hiện tại', depth['results']['geoi_epoch8']['test']['20']['all_0_600'], depth['results']['geoi_epoch8']['test']['20']['tail_400_600']),
        ('L20, cache 60s', ttl['results']['geoi_epoch8']['test']['rolling60']['all_0_600'], ttl['results']['geoi_epoch8']['test']['rolling60']['tail_400_600']),
        ('**L20, cache POI cố định theo phiên bản**', static['results']['geoi_epoch8']['test']['all_0_600'], static['results']['geoi_epoch8']['test']['tail_400_600'])]:
        minimum = tail_row.get('min_session_recall5', tail_row.get('minimum_session', {}).get('recall5'))
        cache_rows.append([label, pct(all_row['family_macro_recall5']), pct(tail_row['family_macro_recall5']), pct(minimum)])
    s4_rows = []
    for method, label, cap in [('raw','GPS thật','—'), ('per_session_reset_H12','Geo-I reset từng chuyến','0,345'),
                               ('global_cap_H12','GeoI-Epoch6-H12','0,23'), ('global_cap_H8','**GeoI-Epoch6-H8**','0,23')]:
        m = identity['results'][method]
        auc = [num(m['S4'][target]['selection_best_auc']['test']['roc_auc'], 3) for target in ['same_person','same_vehicle']]
        s4_rows.append([label, cap, ' / '.join(auc), pct(m['utility']['test']['recall'])])
    future_rows = []
    for method, label in [('raw','GPS thật'), ('geoi_session_reset','Geo-I reset từng chuyến'), ('geoi_epoch8','**GeoI-Epoch8-H12**')]:
        m = native['results'][method]['turn_visible']
        future_rows.append([label, pct(m['S5_next_edge']['test']['exact_candidate_edge_accuracy']),
                            pct(m['S6_history_destination']['test']['destination_hit100']),
                            num(m['S6_history_destination']['test']['destination_mae_m'])])
    end_rows = []
    for method, label in [('scale100_L20','GeoI-Slack20 (đối chứng cùng L)'), ('scale025_L20','**GeoI-Endpoint20**')]:
        r = robust['results'][method]['robust_crossfit']
        end_rows.append([label, num(r['S9']['mae_m']), num(r['S10']['mae_m']),
                         pct(r['S10']['hit100']), pct(r['S10']['hit500'])])
    ci = robust['paired_family_uncertainty']['S10/robust/Endpoint20-plainL20/mae_m']
    s10_gain = f"{num(ci['delta'])} m; khoảng bootstrap 95% [{num(ci['bootstrap95_low'])}; {num(ci['bootstrap95_high'])}] m"

    report = {'title':'Bảo vệ riêng tư quỹ đạo với Geo-I',
              'subtitle':'Kiến trúc, đối chứng và các cải tiến sau buổi gặp 03/10', 'pages':[
        page('1. Related works và cách so sánh công bằng',
             p('Giữ Geo-I làm cơ chế bảo vệ GPS. Vòng này mở rộng dịch vụ POI, quản lý ngân sách qua nhiều chuyến và kiểm tra attacker kỹ hơn.'),
             table(['Paper trong 3 năm','Output','Fully','Partially','Metrics gốc'], work_rows, [1.4,1.45,.6,.85,2.05]),
             p('Cửa sổ online đầu tiên: 06/10/2023–06/10/2026. **Fully:** trực tiếp xét và đánh giá task/threat trong giả định paper. **Partially:** cơ chế liên quan hoặc phạm vi hẹp hơn. Đây là đối chiếu của nhóm; scenario không ghi là chưa có bằng chứng trực tiếp. Geo-I (2013), DLS (2014), RDG (2021) là nền tảng ngoài cửa sổ.'),
             p('S1 vị trí; S2 nơi dừng; S3 đoạn đường; S4 liên kết danh tính người/xe; S5 đường kế tiếp; S6 đích tương lai; S7 ý định query; S8 dữ liệu người đi cùng; S9 điểm đầu bị che; S10 điểm cuối của chuyến đã kết thúc.'),
             bullets('**Output khác nhau:** một vị trí, tập thật + dummy, Q-only hoặc dataset offline; ASR cần thành viên thật nên không áp trực tiếp cho Q-only.',
                     '**Metrics trả lời khác câu hỏi:** entropy/DER đo độ khó phân biệt hoặc tính hợp lý; ε đo giới hạn cơ chế; chúng chưa đo đồng thời khả năng attacker và chất lượng POI.',
                     '**Dùng hai lớp đo:** Hit100/MAE + Recall@5 + chi phí trên cùng protocol; thêm metrics gốc khi đúng output và đủ dữ liệu. N/A phải có lý do.')),
        page('2. Kiến trúc: từ GPS tới câu trả lời POI',
             {'type':'figure','svg':'figures/architecture.svg','pdf':'figures/architecture.pdf',
              'height_cm':18.5, 'alt':'Bốn layer của mô hình Geo-I trên thiết bị, server bên ngoài; luồng Z, Q, POI và P.',
              'caption':'Mũi tên liền: luồng xử lý. Nét đứt: input/phương án không đọc GPS mới. **Z** nội bộ; **Q** gửi server; **P** là POI trả cho người dùng.'},
             p('**Layer 2** cập nhật vùng vị trí có thể xảy ra từ Z, lịch sử đã bảo vệ và khả năng di chuyển trên đường. Nó giúp chọn Q khả thi, phủ POI tốt; phiên đã cấp cap vẫn có thể dự đoán khi không đọc GPS mới. Phiên vượt giới hạn chung không đọc GPS, không gửi Q. Purpose thật chỉ dùng ở layer 4.'),
             p('**S1–S3:** Geo-I và phép thử tái dùng Z có nhiễu. **S9/S10:** nhiễu mạnh hơn ở mọi lần đọc, vì online chưa biết lần nào là cuối. **S7:** request chung, lọc purpose locally. **S4–S6:** ngân sách chung hạn chế tích lũy, chưa tự giấu danh tính. S8 chưa có đánh giá mới.')),
        page('3. Cấu hình và attacker trước khi đọc kết quả',
             table(['Tham số','GeoI-Endpoint20','GeoI-Epoch8-H12'],[
                 ['Mục tiêu','Kiểm tra S9/S10 không delay','S5/S6 và utility qua 8 chuyến'],
                 ['ε thử = ε tạo Z, đơn vị /m','0,0025','0,00125'],
                 ['H: số lần đọc tối đa/chuyến','12','12'],
                 ['B: ngân sách danh nghĩa/chuyến','0,06','0,03'],
                 ['Cap hiệu dụng/chuyến','0,0575','0,02875'],
                 ['Ngân sách qua nhiều chuyến','Cộng theo số chuyến; chưa là cap chung','N=8; tổng C=0,23/m'],
                 ['Lịch, điểm Q và kết quả','GPS 60s; K=5; L=20; k=5','GPS 60s; K=5; L20 chọn bằng utility'],
                 ['Đồng hồ gửi Q công khai','Mỗi 60s trong cohort endpoint','Mỗi 20s + mốc hiệu chuẩn công khai'],
                 ['Quy tắc khác','Ngưỡng 200m; slack 0,03','Cùng backbone; cache local riêng'],
                 ['Warmup / delay','0 / 0','0 / 0'],
             ], [1.15,1.5,1.5]),
             p('Đây là các cấu hình của cùng thuật toán. Với cap chung C, tối đa N chuyến và horizon H công khai: **u=C/[N(2H−1)]**, ε thử=ε tạo=u, B=2Hu. H/N được đặt trước; giữ phần cap trong sổ trước GPS; không cấp lại khi restart. K là số Q, L là số POI/loại mỗi Q, k là số câu trả lời tối đa.'),
             bullets('**S1–S3, bảng đối chứng:** **Shadow KNN (k=3)** cùng centroid, continuity, road filter và decoder đường. S2 thêm running mean/intersection khi output chứa GPS thật.',
                     '**S7:** **kNN, ExtraTrees, logistic** đoán purpose/category từ request, reply, tọa độ, thời gian và kích thước.',
                     '**S4:** **kNN/ExtraTrees** liên kết cặp phiên từ summary/shape; **S5/S6:** **ExtraTrees trên hình học ứng viên**, decoder đường/lịch sử và uniform.',
                     '**S9/S10 mới:** **OLS, kNN, ExtraTrees, Viterbi**, gồm tracking **Hungarian**. Chọn riêng cho MAE/Hit trên validation; khóa trước chấm test.'),
             key('Hit100 ↓: tỷ lệ đoán trong 100m. MAE ↑: sai số attacker trung bình (m). Recall@5 ↑: phần POI chuẩn tìm lại được. Ở S1, 33,33% / 299m nghĩa attacker đúng trong 100m ở 1/3 số mốc, sai trung bình 299m.'),
             p('Lúc chấm test, attacker chỉ thấy dữ liệu công bố và thông tin phụ đã định; GPS/nhãn test thuộc evaluator. GPS/nhãn shadow-train được phép dùng để học attacker. Raw kiểm tra bài toán có tín hiệu. Các họ attacker áp cho mọi phương pháp phù hợp, học/chọn riêng theo output; không chỉ có một Shadow KNN.')),
        page('4. Đối chứng trên cùng protocol và metrics gốc',
             p('Bảng paper-v2 giữ nguyên: 12 chuyến test, cùng trace/dịch vụ/split và họ attacker. **Dấu * là adaptation tại repo**, chưa tái lập đầy đủ paper. Mỗi ô là **Hit100 / MAE(m) / Recall@5**, gộp đều theo chuyến; các phương pháp không cùng bảo đảm ε.'),
             table(['Phương pháp','S1','S2','S3','S9','S10'], comparison_rows, [1.4,1,1,1,1,1]),
             p('BR có Hit100 thấp hơn hoặc bằng ba adaptations ở cả năm scenario. MAE lớn hơn TransProtect ở cả năm; DLS có MAE lớn hơn BR ở S1/S3/S9. Recall S3/S10 của BR dưới 90%. **Chưa thể nói thắng mọi metric.** Raw S9/S10 chỉ thấy cửa sổ sau che đầu/cuối; sai số còn bao gồm độ khó của masking.'),
             table(['Phương pháp','Recall macro ↑','JSON bytes/mốc ↓','Tạo output ms/mốc ↓'], utility_rows, [1.5,1,1.25,1.25]),
             bullets('**EIE point estimate:** có thể tính lại; cùng ước lượng/khoảng cách/phép gộp thì đúng bằng MAE ở trên, không là bằng chứng thứ hai.',
                     '**ASR/entropy/DER:** N/A khi thiếu posterior/prior hoặc quy tắc dummy gốc. Q-only không có vị trí thật làm thành viên để chấm ASR.',
                     '**Δc của TransProtect:** N/A trong archive cũ vì thiếu cost tables/prior đích gốc. Diagnostic riêng trên mạng đường: Raw 0m, Endpoint20 trung bình Q '+num(cost['summaries']['scale025_L20']['mean_m'],2)+'m; đây là méo chi phí ↓, không phải privacy và không là head-to-head paper.'),
             p('Chi phí bảng này gồm request JSON + response ID POI, chưa HTTP/TLS; thời gian trên máy benchmark cũ. Các thử nghiệm sau dùng cohort/protocol khác và được đọc riêng. [Nguồn và định nghĩa](evidence.json).')),
        page('5. S7: nhiều purpose, cùng request ra mạng',
             p('Mọi Q lấy POI của các loại theo L cố định. Thiết bị hợp và bỏ trùng; sau đó GPS thật + purpose riêng tư quyết định câu trả lời. Đổi purpose/category/radius/destination không đổi Q, request hoặc lịch gửi.'),
             table(['Purpose đã hiện thực','Recall@5, cấu hình giữ lại'], purpose_rows, [2,1]),
             p('Cohort development 3 nhóm, 2 RNG/chuyến; giữ α=0,L20 vì các mục tiêu Q mới chưa đạt đồng thời yêu cầu privacy/byte. Mean bốn purpose **94,33%**, không trả POI vi phạm điều kiện. Mạng này có tốc độ đồng nhất nên gần nhất/nhanh nhất trùng ranking; kiểm thử với tốc độ khác xác nhận hai hàm khác nhau.'),
             p('Audit nội dung: payload tường minh bị đoán purpose 100%; request chung 25% cho bốn purpose, 16,67% cho sáu category. Đây là kiểm tra kênh nội dung trực tiếp; tuyến đường đặc trưng, account hoặc click vẫn có thể tiết lộ nhu cầu. Giá/giờ mở cửa/trạng thái realtime cần metadata riêng.'),
             {'type':'h3','text':'Cải thiện utility mà giữ nguyên Q và Geo-I'},
             table(['GeoI-Epoch8-H12, cohort native','Recall toàn cửa sổ','Recall 400–600s','Chuyến thấp nhất, 400–600s'], cache_rows, [2,1,1,1.2]),
             p('Sáu nhóm test, 8 chuyến/nhóm. Cache mới chỉ giữ **metadata POI cố định đã nhận trước đó**, dùng qua các chuyến trong cùng epoch công khai; hết epoch/đổi phiên bản thì vô hiệu. Không đọc dữ liệu tương lai, không tải thêm POI theo nhu cầu. Tối đa 416/418 ID trong test.'),
             p('Ba phương án L20 cùng **7.550 request Q, 41.233.562 reply JSON bytes**; L20 tốn khoảng 1,56× reply L10. Recall là conditional: 1.402/1.510 mốc có reference (92,85%); đoạn cuối 462/528 (87,50%). Chuyến đầu chưa có cache còn yếu. Trạng thái đang mở/còn chỗ phải có dữ liệu mới, nếu thiếu là chưa biết.'),
             p('Cache đã có adapter client tùy chọn và replay causal trên transcript đóng băng; client mặc định vẫn cache epoch60. Đây là xử lý local dùng được cho cả đối chứng: reset từng chuyến cùng cache đạt 99,82%. Không thay bảng privacy S5/S6; cần confirmation độc lập sau vòng development này.')),
        page('6. S4–S6: ngân sách chung và suy tương lai',
             table(['S4, cohort development','Cap /6 chuyến','AUC người / xe','Recall@5 ↑'], s4_rows, [1.7,.75,1.1,.9]),
             p('AUC gần 0,5 nghĩa attacker đã chọn chưa phân biệt ổn định trong mẫu này. H8 chỉ có 3 nhóm test đã dùng trong phát triển; audit bank có raw AUC theo nhóm tới 0,694 và 0,229 (đảo chiều), nên chưa kết luận khó liên kết. H12 chưa tốt hơn reset. **Ở cùng tổng cap, đổi cách ghi sổ không đổi output hoặc điểm attacker.** Sổ giúp thực thi cap qua nhiều phiên, chưa che account/IP/biển số.'),
             {'type':'h3','text':'S5/S6 trên prefix thật của native SUMO'},
             table(['Sau khi bắt đầu rẽ','S5 đúng cạnh ↓','S6 Hit100 ↓','S6 MAE(m) ↑'], future_rows, [1.8,1,1,1]),
             p('24 nhóm, 192 chuyến; 12 train/6 chọn/6 test. Attacker biết hai hướng rẽ/đích có thể xảy ra và sáu lịch sử đã bảo vệ; chỉ thấy prefix tới mốc công khai. Các cạnh test chưa xuất hiện trong train. Sau rẽ Raw 100% xác nhận có tín hiệu; trước rẽ Raw cũng 50%, là mơ hồ có sẵn.'),
             p('Epoch8 giữ tổng cap 0,23/m cho tám chuyến, reset có tổng 1,84/m. Epoch8 chọn uniform trên selection và chưa tốt hơn reset về attacker; hai target cùng dựa trên một lựa chọn hướng rẽ nên không là hai bằng chứng độc lập. Attacker chưa dùng đầy đủ các liên kết giữa tám chuyến.'),
             key('Đóng góp đã có: cap không cấp lại theo chuyến/restart; prefix tương lai được chấm đúng thời điểm; utility phục hồi bằng L/cache local. Chưa chứng minh giấu toàn bộ danh tính hoặc thắng mọi dự báo tương lai.')),
        page('7. S9/S10: nhiễu không delay và kiểm tra attacker',
             p('Không bỏ đoạn đầu và không trì hoãn đoạn cuối. Endpoint20 dùng ε nhỏ hơn trên **mọi lần đọc** vì chưa biết trước đâu là lần cuối. So với GeoI-Slack20: cùng backbone, K=5/L20, mọi mốc được gửi; ε giảm 0,01 → 0,0025/m.'),
             table(['28 nhóm, 112 runs/phương pháp','S9 MAE↑','S10 MAE↑','S10 Hit100↓','S10 Hit500↓'], end_rows, [1.8,.85,.85,.9,.9]),
             p('Chọn attacker bằng cross-fit theo nhóm: lần lượt giữ ngoài 1 trong 8 nhóm train, cộng 2 nhóm selection; ưu tiên điểm ổn định hơn qua nhóm, không chọn bằng test. Attacker GeoI-Slack20 có MAE S10 **758m** thay vì 901m ở bank cũ; Endpoint20 vẫn **1.337m**. Chênh MAE cùng L: **'+s10_gain+'**. Bootstrap theo nhóm, không coi 112 runs là độc lập.'),
             p('Hit100 đều 0% nên không phân biệt hai cấu hình; thêm Hit500 cho thước đo rộng hơn. Quy tắc mới cải thiện MAE nhưng Hit chưa ổn định: trên Slack L10, Hit500 S10 giảm 30,36% → 14,29%. Không gọi attacker mới tốt hơn ở mọi metric.'),
             p('Endpoint20 Recall **'+pct(endpoint['results']['scale025_L20']['recall'])+'** ở cohort này, giảm khoảng 2,30 điểm % so với Slack20; 6/112 runs dưới 90%, thấp nhất 77,61%. Xáo thứ tự Q bỏ nhãn slot nhưng Hungarian vẫn ghép được bằng hình học; không xem shuffle là lợi thế privacy độc lập.'),
             {'type':'h3','text':'Bước tiếp theo để kết luận chắc hơn'},
             bullets('Khóa policy/cache/selector, chạy nhóm và thành phố chưa dùng để chỉnh phương pháp; ưu tiên chuyến đầu và chuyến dài có Recall thấp.',
                     'Mở rộng attacker liên kết toàn bộ lịch sử S4–S6; kiểm tra category/purpose lệch tần suất và suy ý định từ tuyến đường.',
                     'Bổ sung metadata và metrics gốc đúng hợp đồng từng paper; báo cả privacy, utility, chi phí và N/A. Giữ Geo-I làm nền tảng.'),
             p('Các vòng mới là development/diagnostic, chưa là confirmation chưa từng xem. Kernel Geo-I lý tưởng là nền tảng; sampler float hiện tại vẫn xấp xỉ. S8 và các kênh account/IP chưa được giải quyết. [Endpoint evidence](../../../artifacts/benchmarks/endpoint_robust_selection_20261006/readout.json) · [Cache evidence](../../../artifacts/benchmarks/native_versioned_static_20261006_v1/results.json).'))
    ]}

    preparation = {'title':'Lời nói gợi ý cho buổi GVHD tiếp theo',
                   'subtitle':'Khoảng 10–12 phút · đi theo 7 phần của report', 'pages':[
        page('1–3. Vấn đề, phương pháp và cách đo',
             {'type':'h3','text':'Mở đầu — 30 giây'},
             p('“Sau buổi 03/10, em giữ Geo-I làm nền tảng. Em tập trung ba việc: so sánh đúng metrics, mở rộng query purpose và cải thiện bảo vệ khi dữ liệu tích lũy qua nhiều chuyến.”'),
             {'type':'h3','text':'Related works — 1 phút'},
             p('“Các paper công bố output khác nhau nên metrics không chuyển trực tiếp được. Ví dụ ASR cần tìm điểm thật trong một tập, còn output của em chỉ có Q giả. Vì vậy em dùng cùng attacker để đo Hit100/MAE, cùng dịch vụ để đo Recall, rồi bổ sung metrics gốc khi đủ điều kiện. Bảng Fully/Partially là đối chiếu của em theo phạm vi paper.”'),
             {'type':'h3','text':'Kiến trúc — 2 phút, chỉ tay theo mũi tên'},
             bullets('“Layer 1 kiểm tra lịch và ngân sách rồi mới đọc GPS. Geo-I thử có nhiễu xem còn giữ Z cũ được không; cần thì tạo Z mới. Z chỉ ở thiết bị.”',
                     '“Layer 2 dùng Z, lịch sử đã bảo vệ và mạng đường để ước lượng vùng có thể đang ở. Từ đó chọn năm Q khả thi, phủ POI tốt. Q không phải năm điểm quanh GPS thật.”',
                     '“Layer 3 gửi cùng loại request từ năm Q. Server trả top-L POI mỗi loại; em hợp, bỏ trùng và dùng cache còn hiệu lực.”',
                     '“Layer 4 mới dùng GPS thật và nhu cầu thật: gần nhất, nhanh nhất, trong bán kính hay ít đi vòng. Kết quả P được trả local; nhu cầu này không gửi server.”'),
             {'type':'h3','text':'Cấu hình và attacker — 1 phút'},
             p('“Endpoint20 ưu tiên kiểm tra đầu/cuối không delay. Epoch8-H12 có ngân sách chung cho tám chuyến. Chúng dùng cùng Geo-I nhưng khác giới hạn, không so như thể cùng ngân sách. Attacker không chỉ là KNN: mỗi task có bank phù hợp, chọn trên validation rồi khóa.”'),
             key('Nếu hỏi 33,33% / 299m: “Một phần ba số mốc được đoán trong 100m; sai số trung bình là 299m. Hit thấp và MAE cao là tốt cho privacy; Recall cao là tốt cho người dùng.”')),
        page('4–7. Kết quả và điều cần trao đổi',
             {'type':'h3','text':'Đối chứng — 1 phút'},
             p('“Bảng cũ có cùng trace và protocol. Geo-I của em có Hit100 thấp hơn hoặc bằng các adaptations, nhưng DLS có MAE tốt hơn ở vài scenario và Recall S3/S10 của em chưa đạt 90%. EIE cùng point estimate chính là MAE; các metrics gốc thiếu đầu vào em ghi N/A. Hiện em chưa kết luận thắng mọi metric.”'),
             {'type':'h3','text':'S7 và utility — 2 phút'},
             p('“Em đã tách lấy ứng viên khỏi nhu cầu thật, hỗ trợ bốn purpose local. Mean Recall của thử nghiệm này là 94,33%, không trả POI sai điều kiện. Kiểm tra request chung chỉ đoán purpose ở mức 25%, nhưng đây chưa loại được suy ý định qua tuyến đường.”'),
             p('“Ở cohort nhiều chuyến, tăng L10 lên L20 nâng Recall 89,95% lên 94,96%, đổi lại reply tăng 1,56 lần. Cache POI cố định theo phiên bản nâng tiếp lên 99,34% mà không gửi thêm Q. Em chỉ giữ metadata đã nhận trong cùng epoch, không suy trạng thái realtime. Chuyến đầu chưa có cache vẫn có thể yếu.”'),
             {'type':'h3','text':'S4–S6 — 1 phút'},
             p('“Ngân sách chung ngăn reset toàn bộ ε sau mỗi chuyến. H8 có AUC liên kết gần 0,5 nhưng chưa xác nhận trên nhóm mới. Bài toán tương lai dùng prefix thật: Raw đoán đúng 100% sau rẽ; Geo-I còn 41,67–50%. Epoch8 chưa thắng reset về attacker, nhưng dùng tổng cap nhỏ hơn.”'),
             {'type':'h3','text':'S9/S10 — 1 phút'},
             p('“Online chưa biết lần GPS cuối nên em tăng nhiễu ở mọi lần đọc, không delay. Cùng L20, MAE S10 tăng từ 758m lên 1.337m, chênh khoảng 579m. Hit100 bằng 0 cho cả hai nên cần đọc thêm Hit500. Selector mới ổn hơn về MAE nhưng còn lỗi Hit; đây là kết quả development.”'),
             {'type':'h3','text':'Đề xuất với GVHD — 30 giây'},
             p('“Em đề xuất khóa cấu hình hiện tại, kiểm tra trên nhóm/thành phố chưa dùng để chỉnh và mở rộng attacker liên kết lịch sử. Phần ưu tiên là chuyến đầu, chuyến dài và metrics gốc còn N/A. Geo-I tiếp tục là backbone.”')),
        page('Câu trả lời ngắn khi GVHD hỏi',
             table(['Câu hỏi','Cách trả lời'],[
                 ['Layer 2 có đọc GPS thật không?','Để tạo Q, nó chỉ dùng thông tin đã bảo vệ và input công khai. GPS thật chỉ vào cơ chế Geo-I khi được phép và xếp hạng local ở layer 4.'],
                 ['Tại sao cần budget?','Nhiều điểm nhiễu vẫn cho attacker tích lũy thông tin. Giới hạn chung khống chế tổng quan sát; không cấp lại theo chuyến/restart.'],
                 ['B và H từ đâu?','H là số lần đọc tối đa đặt trước. Với N chuyến và tổng cap C, tính u=C/[N(2H−1)], B=2Hu. Chúng là policy, không học từ đáp án test.'],
                 ['Cache có làm lộ thêm GPS/purpose?','Candidate này dùng POI đã nhận, chỉ xử lý local; giữ nguyên Q và traffic. Đổi phiên bản/epoch thì hết hiệu lực; chưa biết realtime thì không tự gán khả dụng.'],
                 ['Vì sao không chỉ nhiễu điểm cuối?','Khi đang chạy chưa biết lúc nào chuyến kết thúc. Nhiễu mọi lần đọc tránh phải dùng tương lai hoặc trì hoãn.'],
                 ['Đã giải quyết S1–S10 chưa?','Chưa. Đã có cơ chế và thực nghiệm cho nhiều task; S8, account/IP, attacker lịch sử đầy đủ và confirmation độc lập còn thiếu.'],
                 ['Kết quả mới có fair với paper không?','Bảng đối chứng cũ cùng protocol là adaptations. Các cải tiến mới có cohort riêng; không ghép thành thứ hạng chung hoặc gọi là tái lập đầy đủ paper.'],
             ], [1.1,2.6]),
             p('Không đọc hết bảng. Chỉ nêu câu hỏi mỗi bảng trả lời, một kết quả chính và một giới hạn. Khi hỏi chi tiết, mở report HTML hoặc evidence thay vì đưa nhãn sample vào bài nói.'))
    ]}
    content = {'schema':'supervisor-next-meeting-content-v1','build_status':'updating',
               'prepared_on':'2026-10-06','meeting_date':'not specified',
               'source_sha256':SOURCES,'report':report,'preparation':preparation}
    (OUT/'content.json').write_text(json.dumps(content,ensure_ascii=False,indent=2)+'\n')
    evidence['next_meeting_iteration_sources'] = SOURCES
    evidence['next_meeting_scope'] = 'Development groups kept separate; no universal dominance or fresh confirmation claim.'
    (OUT/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print('Authored 7 report sections and 3 preparation pages from', len(SOURCES), 'readouts.')


if __name__ == '__main__': main()
