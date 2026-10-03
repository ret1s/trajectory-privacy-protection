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

    page(); sub('Định lượng hai cấu hình Geo-I và đóng góp của mô hình')
    p('Hai tên dễ nhớ dưới đây đều thuộc **BR-Dummy trên nền Geo-I**, dùng cùng ngân sách riêng tư. GeoI-Paced là mốc so sánh nội bộ; GeoI-Slack là cấu hình làm việc hiện tại. Chúng khác cách chọn dummy, không phải hai mức ε ưu tiên privacy/utility khác nhau.')
    table(['Tham số / component', 'GeoI-Paced', 'GeoI-Slack'], [
        ['Mã thực nghiệm', 'response_paced / epoch_cache', 'response_paced_slack03 / epoch_cache'],
        ['Neo riêng tư', 'REM; ε_test = ε_release = 0,01 /m; θ = 200 m', 'Cùng tham số neo'],
        ['Ngân sách phiên', '23 đơn vị × 0,01; cận phiên 0,23 /m', 'Cùng cận phiên 0,23 /m'],
        ['Pacing đọc GPS', 'Cách nhau ít nhất 60 giây', 'Cùng nhịp đọc'],
        ['Road / belief / POI', 'Belief từ neo; làn có hướng, luật rẽ; chọn tập phủ POI', 'Cùng nền; thêm bước di chuyển có slack'],
        ['Slack trong bộ chọn', 'Không nới objective', 'Cho phép mất tối đa 0,03 objective để tiến về mục tiêu; không phải giảm Recall 3%'],
        ['Đầu ra / dịch vụ', 'K=5 tọa độ; 6 loại POI; server trả top-10 mỗi loại/điểm', 'Cùng K=5, L=10'],
        ['Cache và kết quả riêng', 'Hợp phản hồi cùng epoch 60 giây; GPS thật xếp hạng top-5 tại thiết bị', 'Cùng cache; không thay query'],
    ], [4.0, 6.45, 6.45])
    p('Lần đầu tạo neo chi 1 đơn vị; trước bước sau dự trữ 2 đơn vị. Test tái dùng chi 1, test + tạo neo mới chi 2. Giữa hai lần đọc hoặc khi hết ngân sách, mô hình chỉ dùng trạng thái đã bảo vệ. Nhịp đọc GPS, nhịp query dịch vụ và epoch hiệu lực phản hồi là ba khái niệm riêng; pacing không che giờ bắt đầu/kết thúc chuyến.')
    p('Trong phân tích lý tưởng với đồng hồ cố định, cận tỷ số xác suất của transcript là exp(0,23 × D∞), với D∞ là độ lệch GPS lớn nhất giữa hai chuỗi. Đây là cận theo khoảng cách, không phải xác suất “an toàn 23%”. Hậu xử lý từ neo giữ cận này; mã số thực chưa có chứng nhận pure-DP chính xác.')
    sub('Đóng góp cần trình bày như thế nào?')
    p('**Đề xuất:** từ lịch sử Geo-I đã bảo vệ, sinh tập truy vấn đi được trên mạng làn và phủ POI hữu ích trong ngân sách hữu hạn. Giá trị nằm ở sự phối hợp cơ chế theo nhiệm vụ dịch vụ: giảm suy luận gần, giữ khả năng tìm đúng địa điểm và kiểm soát tải truy vấn. Geo-I, tái dùng neo và dummy đã có tiền lệ; paper EV [R12] đã kết hợp AGeoI + dummy, nên riêng tổ hợp hai từ này chưa là điểm mới.')
    p('**So với đối chứng:** DLS chú trọng độ khó phân biệt giữa điểm thật/dummy; các hướng TransProtect/Semantic chú trọng tính hợp lý của vị trí/chuỗi. Ta bổ sung việc chọn cả tập truy vấn theo hợp POI từ belief đã bảo vệ, thay vì dùng độ giống quỹ đạo làm đại diện cho utility. Road-aware tạo tính khả thi; privacy vẫn phải kiểm tra bằng attacker. Với RDG và Fake-query, đây mới là khác biệt thiết kế: chưa có benchmark Geo-I cùng protocol để chứng minh ưu thế số điểm.')
    key('Lập luận đóng góp phải có cả cơ chế và bằng chứng: so trực tiếp bản Geo-I v2 với adapter để thấy đánh đổi; dùng phép bỏ/thêm component trên bản mới để kiểm tra lợi ích dịch vụ. Không gộp hai phiên bản thành một kết quả.')

    page(); sec('Benchmark Geo-I: so trực tiếp với đối chứng')
    p('**Phép thử v2 đã được kiểm chứng:** K=5, ba seed 81/82/83, 12 chuyến test riêng biệt; mỗi chuyến có năm cửa sổ S1/S2/S3/S9/S10, nên 60 cửa sổ không phải 60 mẫu độc lập. Bản BR tái dùng dùng nền Geo-I với cận phiên 0,24 /m trên đồ thị nút giao. Đây là phiên bản trước mạng làn/pacing/slack/cache hiện tại, không có phân ca A/B/C.')
    p('* DLS là bản thích nghi lên đồ thị; TransProtect và Semantic là adapter rút gọn trong benchmark v2, không tái lập Transformer/LSTM của paper. Không thay chúng bằng adapter mới rồi giữ số cũ. RDG/Fake-query chưa có kết quả so trực tiếp với Geo-I trong phép thử này. AnotherMe chỉ áp dụng S3: 11/12 thành công, không cùng mẫu số nên báo riêng, không đưa vào bảng chung.')
    table(['Phương pháp', 'S1', 'S2', 'S3', 'S9', 'S10'], [
        [('**'+r['label']+'**') if r['method'] == 'br_private' else r['label']] + [pct(r['scenarios'][s]['hit100'])+' / '+num(r['scenarios'][s]['mae_m'])
                        for s in SCENARIOS] for r in evidence['legacy_rows']
    ], [3.6, 2.66, 2.66, 2.66, 2.66, 2.66])
    p('Mỗi ô: **Hit100 (%) ↓ / MAE (m) ↑**. MAE và Hit chọn attacker riêng từ tập chọn đối thủ; không phải kết quả của một đối thủ duy nhất tối ưu cả hai.')
    table(['Recall@5 ↑', 'S1', 'S2', 'S3', 'S9', 'S10'], [
        [('**'+r['label']+'**') if r['method'] == 'br_private' else r['label']] + [pct(r['scenarios'][s]['recall']) for s in SCENARIOS]
        for r in evidence['legacy_rows']
    ], [3.6, 2.66, 2.66, 2.66, 2.66, 2.66])
    p('**Bằng chứng hỗ trợ:** ở S2, BR tái dùng có Hit100 **22,22%**, thấp hơn DLS/Semantic (100%) và TransProtect (66,67%), với Recall 97,01%. Ở S3, Hit100 **7,44%**, thấp hơn ba adapter (31,21–69,28%) và Geo-I + dummy (52,65%). Kết quả hỗ trợ thiết kế phối hợp trạng thái đã bảo vệ theo thời gian; chưa cô lập riêng tác động tái dùng vì tham số và cách sinh dummy cũng khác.')
    p('**Đánh đổi phải nói rõ:** BR không thắng mọi metric. Geo-I + dummy đường có Hit100 S3 thấp hơn (3,57%) nhưng Recall thấp hơn (81,96% so với 87,67%); cả hai chưa đạt 90%. DLS có MAE lớn hơn ở S1/S3/S9 dù Hit100 có thể cao hơn. BR cũng chưa đạt Recall 90% ở S10 (88,33%); S10 Hit100=0 của nhiều phương pháp nên không đủ phân biệt. S9/S10 chỉ suy tọa độ đầu/cuối SUMO, chưa chứng minh che nhà hay nơi làm việc.')
    key('Kết luận v2: BR trên nền Geo-I giảm tỷ lệ suy đúng gần ở một số nhiệm vụ quan trọng, có đánh đổi utility. Đây là bằng chứng đóng góp của bản thích nghi trong benchmark, chưa chứng minh vượt các paper nguyên bản hoặc ưu thế ở cùng ngân sách.')

    page(); sub('Bản Geo-I hiện tại: 14 ca và kiểm tra từng component')
    p('**GeoI-Slack:** 12 nhóm phát triển, 165 bản ghi từ 94 chuyến nguồn; S10 chỉ A/B. K=5, L=10, cận phiên 0,23 /m. Recall dưới đây dùng POI khả dụng mô phỏng ở p=0,8, ba thế giới khả dụng và hai lần chạy cơ chế. Privacy kế thừa từ đúng tọa độ công bố đã chấm bằng tập attacker hữu hạn: cache và trạng thái server độc lập với GPS không đổi query. Đây là dữ liệu phát triển, không phải xác nhận trên nhóm mới.')
    table(['Ca', 'GPS thật: Hit100', 'GeoI-Slack: Hit100 ↓', 'MAE (m) ↑', 'Recall@5 ↑', 'Nhóm'], [
        [r['case'], pct(r['raw_hit100']), pct(r['hit100']), num(r['mae_m']), pct(r['recall']), str(r['families'])]
        for r in evidence['case_rows']
    ], [1.6, 3.0, 3.5, 3.0, 3.2, 1.6])
    table(['Cấu hình', 'Recall@5 ↑', 'Ca ≥90%', 'Byte / sự kiện ↓'], [
        [('**'+r['label']+'**') if r['label'] == 'GeoI-Slack' else r['label'], pct(r['recall']), str(r['gates'])+'/14', num(r['request_bytes']+r['response_bytes'], 1)]
        for r in evidence['service_rows']
    ], [5.0, 3.7, 3.3, 4.9])
    cache, slack = evidence['contrasts']
    ci = lambda r: '['+num(100*r['ci95'][0], 2)+'; '+num(100*r['ci95'][1], 2)+']'
    p('**Đóng góp có thể định lượng:** cache tăng Recall '+num(100*cache['delta'], 2)+' điểm %, CI 95% '+ci(cache)+', không thêm query/byte trong schema này. Thêm slack tăng trung bình '+num(100*slack['delta'], 2)+' điểm %, CI '+ci(slack)+' chứa 0: hướng cải thiện có triển vọng, chưa đủ kết luận chắc chắn. CI lấy mẫu lại ghép cặp 12 nhóm 10.000 lần, chưa hiệu chỉnh nhiều so sánh.')
    p('**Giới hạn còn thấy:** S1.C chỉ đạt 80,18% Recall, nên cấu hình đạt 13/14 ca ≥90%, không phải đạt mọi ca. S9.C có Hit100=0 cả với GPS thật, nên riêng chỉ số này chưa chứng minh cải thiện; Geo-I vẫn có Hit500='+pct(next(r['hit500'] for r in evidence['case_rows'] if r['case'] == 'S9.C'))+': điểm đầu vẫn có thể bị khoanh vùng. Những số S9/S10 là của lõi Geo-I qua cửa sổ quan sát; không gán cho BR-Boundary, vì wrapper cắt đầu/buffer đuôi chưa được benchmark cùng dịch vụ này.')
    p('Recall gộp đều ca trong scenario rồi đều năm scenario. Byte gộp theo sự kiện, tính toàn bộ traffic của 94 chuyến được giữ, gồm yêu cầu + phản hồi JSON; chưa gồm HTTP/TLS hoặc latency. S10.B ở đây giữ nguồn cũ S10.C. Các số tổng/CI được tính lại đúng 14 ca, không chạy lại mô hình và không trộn với kết quả v2.')
    evidence['presentation_tables'] = tables
    return evidence
