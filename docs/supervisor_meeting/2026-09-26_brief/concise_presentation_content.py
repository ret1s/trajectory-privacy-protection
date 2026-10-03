"""Concise presentation with scenario-level benchmarks and no sample appendix."""
from related_work_coverage import SCENARIOS, ROWS, coverage_evidence
from geoi_content import add_geoi_content
import json


def build_concise_content(ns):
    ns['blocks'] = []
    p, key, sec, sub, page, table, graphic = (
        ns[k] for k in ('p', 'key', 'sec', 'sub', 'page', 'table', 'graphic'))
    out = ns['OUT']
    sec('Related works: scenario và metrics trong một bảng')
    p('Khảo sát trong **21/09/2023–21/09/2026**; ↑/↓ là hướng tốt hơn. Bảng gộp phạm vi scenario và bộ đo của từng paper.')
    p(SCENARIOS)
    p('**Fully:** xử lý trực tiếp nhiệm vụ; **Partially:** xử lý một phần, trong giả định paper. Đây là đối chiếu của ta, không phải chứng nhận bảo vệ hoàn toàn. **CX:** chưa đủ nguồn; scenario không liệt kê là chưa thấy xử lý trực tiếp (R7/R9: CX; R8/R10: không áp dụng).')
    metric_rows = [
        ['EIE ↑: sai số suy luận', 'Δc ↓: chênh chi phí hành trình; byte/độ trễ dịch vụ: CX'],
        ['ASR ↑: đạt ẩn danh; DER ↑: dummy/ngữ nghĩa', 'DER chưa đo top-5 POI; thời gian sinh ↓'],
        ['Số đường khó phân biệt ↑; ASR ↑', 'POI: CX; thời gian sinh, RSS ↓'],
        ['Sai số ↑; tấn công đúng ↓', 'Dịch chuyển đầu ra ↓; chi phí LBS: CX (preprint)'],
        ['Tỷ lệ khớp quỹ đạo ρ ↓', 'VANET; số quỹ đạo giả ψ ↓ (chi phí gián tiếp)'],
        ['MI ↓; bảo đảm ε-DP', 'JSD phân bố ↓; độ phức tạp (offline)'],
        ['Phân tích an toàn; metric tấn công: CX', 'Truy hồi: CX; tính toán (mới đủ tóm tắt)'],
        ['Tái dựng/liên kết: nhóm phép đo', 'Phân bố, hình học, tác vụ (khảo sát)'],
        ['LPI: công thức CX', 'TDD, DC: định nghĩa CX; thời gian xử lý/phản hồi'],
        ['Tỷ lệ dùng tính năng', 'Nhận thức/sử dụng (nghiên cứu hành vi)'],
        ['S-TT / k-site-unidentifiability', 'Che điểm nhạy cảm (tổng quan S-TT 2022)'],
        ['AGeoI + dummy', 'Truy vấn trạm sạc; phép thử điểm đầu/cuối: CX'],
    ]
    merged_rows = [row[:3] + metrics for row, metrics in zip(ROWS, metric_rows)]
    table(['Paper', 'Fully', 'Partially / CX', 'Privacy / chẩn đoán', 'Utility / chi phí'],
          merged_rows, [3.2, 1.55, 2.2, 4.35, 5.6])
    p('Mốc nền: DLS 2014 [B1], RDG 2021 [B2]. AnotherMe [B3] online 11/09/2023, ngoài cửa sổ dù số tạp chí ghi 2024.')

    p('**Vì sao cần bộ metrics chung?** Điểm số gốc chưa so trực tiếp được vì:')
    ns['bullets']([
        '**Đầu ra khác nhau:** vị trí thay thế, quỹ đạo giả, tập dummy hay query chèn. Độ lệch của một điểm công bố chưa cho biết attacker suy được gì từ toàn bộ output.',
        '**Nhiệm vụ và attacker khác nhau:** đo một lần gửi khác đo cả chuỗi hay điểm biên; quyền quan sát, dữ liệu và ngưỡng đánh giá cũng khác.',
        '**Utility khác tác vụ:** giữ phân bố/hình học hoặc dummy hợp ngữ nghĩa chưa đồng nghĩa tìm đúng top-5 POI.',
        '**Chi phí khác phạm vi:** thời gian sinh hay số dummy chưa phản ánh đủ dữ liệu gửi và phản hồi của dịch vụ.',
    ])
    key('Ta dùng Hit100 (%) ↓ và MAE (m) ↑ để chấm cùng đáp án từ toàn bộ output attacker được phép thấy; Recall@5 (%) ↑ để đo cùng dịch vụ POI; byte/sự kiện ↓ để tính request + response. So sánh cùng scenario, dữ liệu và quyền quan sát; kiểm tra Recall mỗi ca ≥90%.')

    page(); sec('Kiến trúc mô hình BR-Dummy: các bước xử lý')
    p('Mục tiêu: tìm địa điểm quan tâm (**POI**, như trạm sạc hoặc bệnh viện) mà không gửi GPS thật. Đọc sơ đồ theo **1 → 2 → 3a/3b → 4 → máy chủ → 5**; khung xanh là mô hình.')
    graphic(r'\includegraphics[width=\linewidth,height=0.79\textheight,keepaspectratio]{figures/architecture_report_flow.pdf}',
            '<img src="figures/architecture_report_flow.svg" alt="Các bước xử lý: giới hạn và lịch đọc GPS, Geo-I, ước lượng vị trí và kiểm tra đường, chọn điểm truy vấn, máy chủ và chọn kết quả trên thiết bị">',
            'Chưa đến lịch hoặc hết ngân sách thì bỏ bước 2, dùng lịch sử đã bảo vệ. Hai bước 3a/3b cùng giúp chọn điểm truy vấn. Bước 5 nhận kết quả từ máy chủ. S9/S10 là phần mở rộng tùy chọn để che đầu/cuối chuyến.')
    page(); sub('Mỗi bước làm gì và vì sao cần?')
    p('**Vị trí đã bảo vệ** là tọa độ tham chiếu đã làm nhiễu bằng Geo-I, chỉ dùng trong thiết bị. Từ đó mô hình chọn **điểm truy vấn giả (dummy)** để gửi thay GPS thật.')
    table(['Bước / scenario', 'Cách hoạt động và mục đích'], [
        ['1. Giới hạn + lịch đọc GPS — S2/S3', 'Kiểm tra lịch đọc và ngân sách còn lại. Không đọc GPS mới để chọn truy vấn khi chưa đến lịch hoặc không đủ ngân sách.'],
        ['2. Geo-I: làm nhiễu GPS — S1/S2', 'Phép kiểm tra có nhiễu quyết định giữ vị trí tham chiếu cũ hoặc tạo vị trí mới. Tái dùng giảm các mẫu nhiễu mới khi quan sát lặp.'],
        ['3a. Ước lượng vùng vị trí (belief)', 'Dùng lịch sử vị trí đã bảo vệ để ước lượng người dùng có thể ở đâu; không coi một tọa độ nhiễu là vị trí chính xác.'],
        ['3b. Kiểm tra đường đi — S3', 'Từ điểm giả trước đó, hướng làn, luật rẽ và thời gian, xác định các điểm tiếp theo xe có thể tới.'],
        ['4. Chọn điểm truy vấn — dịch vụ', 'Chọn 5 điểm có thể trả về các POI bổ sung nhau. Ưu tiên điểm giúp tiếp tục di chuyển tới vùng có kết quả hữu ích.'],
        ['5. Hợp kết quả + chọn top-5', 'Giữ phản hồi còn hiệu lực trong 60 giây; hợp các POI nhận được. GPS thật chỉ xếp hạng trên thiết bị để chọn 5 POI mỗi loại.'],
        ['Che đầu chuyến — S9', 'Khi bật: không đưa GPS của h giây đầu vào các bước tạo điểm truy vấn.'],
        ['Che cuối chuyến — S10', 'Khi bật: giữ tạm kết quả trước khi gửi. Khi chuyến kết thúc, hủy phần chưa gửi để không công bố đoạn cuối.']
    ], [5.0, 11.9])
    p('Các bước 3–4 chỉ dùng vị trí đã bảo vệ, bản đồ và POI công khai. Có 5 điểm giả không tự bảo đảm ẩn danh. Hai cơ chế che đầu/cuối đã kiểm tra tích hợp, chưa benchmark kết hợp với dịch vụ.')

    evidence = add_geoi_content(ns)

    literature = coverage_evidence(ns['refs'])
    literature['presentation_headers'] = ['Paper', 'Fully', 'Partially / CX', 'Privacy / chẩn đoán', 'Utility / chi phí']
    literature['presentation_rows'] = merged_rows
    (out/'related_work_coverage.json').write_text(json.dumps(literature, ensure_ascii=False, indent=2)+'\n')

    page(); sec('Nguồn và khả năng truy nguyên')
    p('So trực tiếp: artifacts/benchmarks/paper_benchmark/results.json. Bản hiện tại: iteration18_expanded_attacks.json, iteration28_live_service.json và iteration28_verification.json trong artifacts/benchmarks/research_loop/. geoi_evidence.py kiểm tra hash và gộp số đo theo scenario; method_evidence.json lưu nguồn, phép gộp và bảng đã in. preparation_evidence.json truy nguyên guide.')
    p('Bản trình bày chỉ dùng mức scenario S1–S10; benchmark hiện đo S1, S2, S3, S9, S10. Giữ nguyên dữ liệu và thực nghiệm gốc, không chạy thêm mô hình. Cấu hình và benchmark dùng chung giữa report/guide; cách dựng ở README.md.')
    for rid, citation, url, doi, verify, role in ns['refs']:
        ns['blocks'].append(('ref', (rid, citation, url, verify)))
    import hashlib
    paths = ['core/mechanisms.py', 'core/boundary_release.py',
             'benchmark/engines/budgeted.py', 'benchmark/engines/switching_cover.py',
             'benchmark/engines/paced_slack.py', 'benchmark/engines/filtered_cover.py',
             'benchmark/engines/contextual_lane.py', 'benchmark/engines/slack_progress.py',
             'benchmark/anchor_belief.py', 'benchmark/switching_belief.py']
    paths += [str((out/n).relative_to(ns['ROOT'])) for n in
              ('geoi_evidence.py', 'geoi_content.py', 'related_work_coverage.py', 'related_work_coverage.json', 'plot_report_architecture.py', 'figures/architecture_report_flow.pdf', 'figures/architecture_report_flow.svg')]
    evidence['sources'].update({n: hashlib.sha256((ns['ROOT']/n).read_bytes()).hexdigest() for n in paths})
    evidence['model_architecture'] = {
        'name': 'BR-Dummy: Geo-I anchor + protected-history belief + directed-road/POI-aware dummy selection',
        'working_name': 'GeoI-Slack', 'internal_control_name': 'GeoI-Paced',
        'boundary_wrapper': 'BR-Boundary v1; integration evidence, not full service benchmark'}
    evidence['model_architecture']['figure'] = 'figures/architecture_report_flow.pdf'
    evidence['model_architecture']['execution_order'] = 'budget/pacing -> anchor if permitted -> belief + reachable road domain -> POI selector -> server response -> cache/local ranking; optional S9 before core and S10 after selector'
    evidence['report_case_labels'] = ns['report_label_manifest']()
    (out/'method_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n')
    return ns['blocks']
