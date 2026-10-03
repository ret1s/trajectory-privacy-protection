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
    p('**GeoI-Paced:** cấu hình đối chứng nội bộ. **GeoI-Slack:** bản hiện tại, thêm độ linh hoạt khi chọn dummy. Hai bản cùng ngân sách riêng tư và cùng dịch vụ.')
    table(['Tham số / component', 'GeoI-Paced', 'GeoI-Slack'], [
        ['Neo / ngân sách', 'ε_test = ε_release = 0,01 /m; θ = 200 m; cận phiên 0,23 /m', 'Giống GeoI-Paced'],
        ['Đọc GPS / cache', 'Đọc cách ≥60 giây; cache cùng epoch 60 giây', 'Giống GeoI-Paced'],
        ['Dịch vụ', 'K=5 điểm; L=10 POI/loại/điểm; chọn top-5 tại thiết bị', 'Giống GeoI-Paced'],
        ['Slack', 'Không nới điểm mục tiêu của bộ chọn', 'Cho phép giảm tối đa 0,03 điểm mục tiêu để di chuyển hữu ích hơn; không phải giảm Recall 3%'],
    ], [4.0, 6.45, 6.45])
    key('Đóng góp: phối hợp lịch sử Geo-I đã bảo vệ, ngân sách, mạng làn và mục tiêu phủ POI. Geo-I + dummy đã có tiền lệ [R12]; lợi ích của cách phối hợp phải được kiểm tra bằng benchmark.')

    page(); sec('Benchmark Geo-I: so trực tiếp với đối chứng')
    p('**Bản v2:** K=5, ba seed, 12 chuyến test; năm cửa sổ/chuyến có tương quan. BR tái dùng có cận phiên 0,24 /m. Đây là phiên bản trước GeoI-Slack hiện tại; báo riêng, không gộp số.')
    sub('Attacker đoán gần vị trí thật bao nhiêu lần?')
    p('Bảng là **Hit100 (%)**: tỷ lệ đoán cách vị trí thật ≤100 m. **Thấp hơn = tốt hơn cho privacy.** S1: vị trí; S2: nơi dừng; S3: đoạn đã đi; S9/S10: điểm đầu/cuối.')
    table(['Phương pháp', 'S1', 'S2', 'S3', 'S9', 'S10'], [
        [('**'+r['label']+'**') if r['method'] == 'br_private' else r['label']] + [pct(r['scenarios'][s]['hit100'])
                        for s in SCENARIOS] for r in evidence['legacy_rows']
    ], [3.6, 2.66, 2.66, 2.66, 2.66, 2.66])
    sub('Đọc riêng sai số và chất lượng dịch vụ của BR v2')
    ours = next(r for r in evidence['legacy_rows'] if r['method'] == 'br_private')['scenarios']
    table(['Scenario', 'Đoán trong 100 m: Hit100 ↓', 'Sai số trung bình: MAE (m) ↑', 'POI đúng: Recall@5 ↑'], [
        [s, pct(ours[s]['hit100']), num(ours[s]['mae_m']), pct(ours[s]['recall'])] for s in SCENARIOS
    ], [2.0, 4.8, 5.1, 5.0])
    key('Ví dụ S1: 33,33% = khoảng một phần ba trường hợp attacker đoán trong 100 m; 299 m = sai số suy luận trung bình. Hai số là hai chỉ số riêng, không phải phép chia hoặc độ chính xác của model ta.')
    p('**Điểm mạnh:** Hit100 S2/S3 thấp hơn ba adapter. **Đánh đổi:** Recall S3/S10 dưới 90%; Geo-I + dummy đường có Hit100 S3 tốt hơn nhưng Recall chỉ 81,96%. DLS có MAE lớn hơn ở S1/S3/S9. Không có phương pháp thắng mọi chỉ số.')
    p('* Ba đối chứng là bản thích nghi, không phải tái lập paper nguyên bản. RDG/Fake-query chưa có so trực tiếp cùng protocol; AnotherMe chỉ hoàn thành 11/12 ca S3 nên báo riêng. Hit và MAE chọn attacker riêng. Bảng đầy đủ nằm trong method_evidence.json.')

    page(); sub('Bản Geo-I hiện tại: 14 ca và kiểm tra từng component')
    p('**Bản hiện tại:** 12 nhóm phát triển, 165 bản ghi từ 94 chuyến; K=5, L=10, cận phiên 0,23 /m. POI khả dụng mô phỏng ở mức 80%. Hai cột Hit100 so attacker khi thấy GPS thật và khi thấy dữ liệu Geo-I; MAE/Recall là của GeoI-Slack.')
    table(['Ca', 'GPS thật: Hit100 (%)', 'Geo-I: Hit100 (%) ↓', 'Sai số TB (m) ↑', 'POI đúng (%) ↑'], [
        [r['case'], pct(r['raw_hit100']), pct(r['hit100']), num(r['mae_m']), pct(r['recall'])]
        for r in evidence['case_rows']
    ], [1.6, 3.7, 3.7, 3.6, 3.3])
    table(['Cấu hình', 'POI đúng: Recall@5 ↑', 'Ca đạt ≥90%', 'Byte gửi + nhận / sự kiện ↓'], [
        [('**'+r['label']+'**') if r['label'] == 'GeoI-Slack' else r['label'], pct(r['recall']), str(r['gates'])+'/14', num(r['request_bytes']+r['response_bytes'], 1)]
        for r in evidence['service_rows']
    ], [5.0, 3.7, 3.3, 4.9])
    cache, slack = evidence['contrasts']
    ci = lambda r: '['+num(100*r['ci95'][0], 2)+'; '+num(100*r['ci95'][1], 2)+']'
    key('Kết quả chính: Recall 95,44%; 13/14 ca đạt ngưỡng. S1.C còn 80,18%; traffic khoảng 994 byte/sự kiện, so với 214 byte khi gửi GPS thật.')
    p('**Component:** cache tăng '+num(100*cache['delta'], 2)+' điểm % Recall, không thêm query; khoảng tin cậy 95% '+ci(cache)+'. Slack tăng '+num(100*slack['delta'], 2)+' điểm %, nhưng khoảng '+ci(slack)+' chứa 0: chưa chắc chắn. Lấy mẫu lại 12 nhóm, chưa hiệu chỉnh nhiều so sánh.')
    p('**Giới hạn:** Hit100=0 chưa nghĩa là an toàn tuyệt đối. S9.C có Hit100=0 cả với GPS thật; Geo-I vẫn có Hit500=58,33%. S9/S10 ở đây chưa bật BR-Boundary. Đây là dữ liệu phát triển; cần xác nhận trên nhóm mới.')
    p('Recall gộp đều ca trong scenario rồi đều năm scenario. Byte gồm toàn bộ request + response JSON của 94 chuyến, chưa gồm HTTP/TLS/latency. Privacy dùng lại attacker trên cùng tọa độ công bố; cache không đổi query. S10 chỉ A/B, B giữ mã nguồn cũ S10.C.')
    evidence['presentation_tables'] = tables
    return evidence
