"""Shared concise Geo-I configuration, comparisons and contribution arguments."""
from geoi_evidence import build_evidence, SCENARIOS


def add_geoi_content(ns):
    p, key, sub, sec, page = (ns[k] for k in ('p', 'key', 'sub', 'sec', 'page'))
    evidence = build_evidence()
    pct = lambda x: f'{100*x:.2f}%'.replace('.', ',')
    num = lambda x, d=0: f'{x:,.{d}f}'.replace(',', 'X').replace('.', ',').replace('X', '.')
    tables = []
    def table(headers, rows, widths):
        tables.append({'headers': headers, 'rows': rows})
        ns['table'](headers, rows, widths)

    sub('Định lượng hai cấu hình Geo-I')
    p('**GeoI-Paced:** bản so sánh nội bộ. **GeoI-Slack:** thêm độ linh hoạt khi chọn điểm truy vấn. Hai bản dùng cùng ngân sách riêng tư và cùng dịch vụ. Ngân sách là cận lý thuyết cho cả phiên, không phải tỷ lệ attacker đoán đúng.')
    table(['Tham số', 'GeoI-Paced', 'GeoI-Slack'], [
        ['Geo-I / ngân sách', 'ε_test = ε_release = 0,01 /m; θ = 200 m; cận phiên 0,23 /m', 'Giống GeoI-Paced'],
        ['Lịch đọc / giữ kết quả', 'Dùng GPS mới để tạo truy vấn cách ≥60 giây; giữ phản hồi còn hiệu lực trong khoảng 60 giây', 'Giống GeoI-Paced'],
        ['Dịch vụ', 'K=5 điểm truy vấn; máy chủ trả L=10 POI/loại/điểm; thiết bị chọn 5 POI mỗi loại', 'Giống GeoI-Paced'],
        ['Độ linh hoạt (slack)', 'Chọn điểm theo mức phủ POI hiện tại', 'Cho phép giảm tối đa 0,03 điểm đánh giá độ phủ để đi tới vùng hữu ích hơn; không phải giảm Recall 3%'],
    ], [4.0, 6.45, 6.45])
    key('Đóng góp: phối hợp lịch sử vị trí đã làm nhiễu, lịch đọc và ngân sách, đường xe đi được và chất lượng tìm POI. Geo-I + điểm giả đã có tiền lệ [R12]; lợi ích của cách phối hợp cần kiểm tra bằng benchmark.')

    page(); sec('Benchmark Geo-I: so trực tiếp với đối chứng')
    p('**Benchmark so sánh, phiên bản 2:** K=5, server trả top-5; ba seed, 12 chuyến test. “v2” là phiên bản benchmark, không phải tên attacker. BR tái dùng có cận phiên 0,24 /m; bản trước GeoI-Slack, báo riêng.')
    p('**Cùng tập loại attacker đánh giá mọi phương pháp.** Bộ học như **Shadow kNN** được huấn luyện riêng từ output mỗi phương pháp. Chọn bộ suy luận theo phương pháp/scenario và chỉ số trên tập chọn, rồi giữ cố định khi test.')
    p('**Attacker thấy gì?** Tọa độ công bố và thời gian trong cửa sổ được phép; biết phương pháp và mạng đường. Không biết điểm thật trong tập dummy, neo nội bộ hay đích ẩn. Đối chứng GPS thật công bố tọa độ thật.')
    table(['Scenario / đáp án cần suy', 'Tên các bộ suy luận chính và cách dùng'], [
        ['S1 — tọa độ hiện tại', '**Centroid**: tâm tập điểm; **Prior**: phân bố vị trí; **Road filter**: lọc đường; **Shadow kNN**: học từ chuyến phụ trợ.'],
        ['S2 — tọa độ nơi dừng', '**Running mean / Full mean**: gộp nhiều lần nhìn; **Stationary intersection**: tìm điểm chung khi output có điểm thật + dummy; thêm các bộ đường/chuỗi.'],
        ['S3 — các điểm của đoạn đã đi', '**Continuity**: nối theo chuyển động; **Road filter**; **Full path (Viterbi)**: truy ngược toàn cửa sổ; **Shadow kNN**.'],
        ['S9 — điểm đầu bị che', '**Road filter / Full path** dùng phần tuyến còn thấy; **Shadow kNN** học quan hệ giữa cửa sổ và điểm đầu.'],
        ['S10 — điểm cuối bị che', '**Road filter / Full path** dùng tuyến trước đoạn bị che; **Shadow kNN** học quan hệ với điểm cuối; không đọc suffix ẩn.'],
    ], [4.4, 12.5])
    p('**Cách chọn:** MAE thấp nhất / Hit100 cao nhất trên tập chọn. Tập học, chọn và test dùng xe khác nhau. Tập attacker hữu hạn, chưa bảo đảm tối ưu lý thuyết.')
    sub('So sánh privacy và chất lượng dịch vụ')
    p('Mỗi ô theo thứ tự **Hit100 (%) ↓ / MAE (m) ↑ / Recall@5 (%) ↑**. Hit100: đoán trong 100 m; MAE: sai số trung bình; Recall: tìm đúng POI. Recall là chỉ số dịch vụ, không phải điểm của attacker.')
    table(['Phương pháp', 'S1', 'S2', 'S3', 'S9', 'S10'], [
        [('**'+r['label']+'**') if r['method'] == 'br_private' else r['label']] + [pct(r['scenarios'][s]['hit100'])+' / '+num(r['scenarios'][s]['mae_m'])+' / '+pct(r['scenarios'][s]['recall'])
                        for s in SCENARIOS] for r in evidence['legacy_rows']
    ], [3.0, 2.78, 2.78, 2.78, 2.78, 2.78])
    key('Ví dụ 33,33% / 299 / 95,83%: attacker đoán trong 100 m ở 33,33% trường hợp; sai số trung bình 299 m; dịch vụ tìm đúng 95,83% POI chuẩn. Dấu / tách ba chỉ số, không phải phép chia.')
    p('**Điểm mạnh:** Hit100 S2/S3 thấp hơn ba adapter. **Đánh đổi:** Recall S3/S10 dưới 90%; Geo-I + dummy đường có Hit100 S3 tốt hơn nhưng Recall chỉ 81,96%. DLS có MAE lớn hơn ở S1/S3/S9. Không có phương pháp thắng mọi chỉ số.')
    p('* Ba đối chứng là bản thích nghi, không phải tái lập paper. RDG/Fake-query chưa có so trực tiếp cùng protocol; AnotherMe chỉ hoàn thành 11/12 chuyến S3 nên báo riêng. S9/S10 suy tọa độ đầu/cuối, chưa suy nhãn nhà/nơi làm việc.')

    page(); sub('Bản Geo-I hiện tại: kết quả theo scenario')
    p('Khung nghiên cứu gồm **S1–S10**. Benchmark hiện có kết quả cho **S1, S2, S3, S9, S10**; S4–S8 chưa được đánh giá. Hai đợt thực nghiệm khác protocol, báo riêng.')
    p('**Bản hiện tại:** 12 nhóm phát triển, 165 bản ghi từ 94 chuyến; K=5, L=10, cận phiên 0,23 /m. POI khả dụng mô phỏng ở mức 80%. Hai cột Hit100 so attacker khi thấy GPS thật và khi thấy dữ liệu Geo-I; MAE/Recall là của GeoI-Slack.')
    p('**Ngân hàng attacker ở phép thử này:** **kNN**, **ExtraTrees** (cây hồi quy), **ước lượng hình học** và **chiếu lên đường**. Học trên 64 nhóm phụ trợ, chọn trên 16 nhóm khác, giữ cố định khi chấm 12 nhóm phát triển. S2 gộp quan sát khi dừng; S3 dùng cả cửa sổ; S9/S10 suy điểm biên, có xét chuyến lặp. Input chỉ là dữ liệu công bố và thời gian; không nhận GPS/đích thật khi chấm Geo-I. Ngân hàng này khác bảng 3.1.')
    table(['Scenario', 'GPS thật: Hit100 (%)', 'Geo-I: Hit100 (%) ↓', 'Sai số TB (m) ↑', 'POI đúng (%) ↑'], [
        [r['scenario'], pct(r['raw_hit100']), pct(r['hit100']), num(r['mae_m']), pct(r['recall'])]
        for r in evidence['scenario_rows']
    ], [1.6, 3.7, 3.7, 3.6, 3.3])
    table(['Cấu hình', 'POI đúng: Recall@5 ↑', 'Scenario có Recall TB ≥90%', 'Byte gửi + nhận / sự kiện ↓'], [
        [('**'+r['label']+'**') if r['label'] == 'GeoI-Slack' else r['label'], pct(r['recall']), str(r['scenario_gates'])+'/5', num(r['request_bytes']+r['response_bytes'], 1)]
        for r in evidence['service_rows']
    ], [5.0, 3.7, 3.3, 4.9])
    cache, slack = evidence['contrasts']
    ci = lambda r: '['+num(100*r['ci95'][0], 2)+'; '+num(100*r['ci95'][1], 2)+']'
    key('Kết quả chính: Recall 95,44%; cả năm scenario có Recall trung bình ≥90%. S1 thấp nhất ở mức trung bình 91,24%; traffic khoảng 994 byte/sự kiện, so với 214 byte khi gửi GPS thật.')
    p('**Cách gộp:** trung bình đều các điều kiện trong mỗi scenario, rồi trung bình đều năm scenario. Đạt ngưỡng trung bình chưa nghĩa là mọi điều kiện đều đạt: Recall thấp nhất trong S1 vẫn là '+pct(evidence['scenario_rows'][0]['min_recall'])+'.')
    p('**Component:** cache tăng '+num(100*cache['delta'], 2)+' điểm % Recall, không thêm query; khoảng tin cậy 95% '+ci(cache)+'. Slack tăng '+num(100*slack['delta'], 2)+' điểm %, nhưng khoảng '+ci(slack)+' chứa 0: chưa chắc chắn. Lấy mẫu lại 12 nhóm, chưa hiệu chỉnh nhiều so sánh.')
    p('**Giới hạn:** Hit100 thấp chưa nghĩa là an toàn tuyệt đối. Ở S9, Hit100='+pct(evidence['scenario_rows'][3]['hit100'])+' nhưng Hit500 còn '+pct(evidence['scenario_rows'][3]['hit500'])+'. S9/S10 chưa bật BR-Boundary. Đây là dữ liệu phát triển; cần xác nhận trên nhóm mới.')
    p('Byte gồm toàn bộ request + response JSON của 94 chuyến, chưa gồm HTTP/TLS/latency. Privacy dùng lại attacker trên cùng tọa độ công bố; cache không đổi query.')
    evidence['presentation_tables'] = tables
    return evidence
