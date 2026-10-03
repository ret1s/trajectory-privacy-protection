"""Concise presentation with shared literature coverage and priority-case appendix."""
from related_work_coverage import LEGEND, SCENARIOS, ROWS, NOTE, coverage_evidence
from geoi_content import add_geoi_content
import json
from report_case_labels import display_case


def build_concise_content(ns):
    ns['blocks'] = []
    p, key, sec, sub, page, table, graphic = (
        ns[k] for k in ('p', 'key', 'sec', 'sub', 'page', 'table', 'graphic'))
    out = ns['OUT']
    sec('Related works: scenario và metrics trong một bảng')
    p('Khảo sát trong **21/09/2023–21/09/2026**; ↑/↓ là hướng tốt hơn. Bảng gộp phạm vi scenario và bộ đo của từng paper.')
    p(SCENARIOS)
    p('**Fully:** xử lý trực tiếp nhiệm vụ; **Partially:** xử lý một phần, trong giả định paper. Đây là đối chiếu của ta, không phải đã vượt mọi ca A/B/C. **CX:** chưa đủ nguồn; scenario không liệt kê là chưa thấy xử lý trực tiếp (R7/R9: CX; R8/R10: không áp dụng).')
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

    page(); sec('Kiến trúc mô hình BR-Dummy: Geo-I, mạng đường và các cơ chế')
    p('**Geo-I tạo neo riêng tư từ GPS.** Lịch sử neo + mạng đường + POI công khai tạo tập dummy. Server thấy query; thiết bị giữ GPS để xếp hạng kết quả cuối.')
    graphic(r'\includegraphics[width=\linewidth]{figures/architecture_model.pdf}',
            '<img src="figures/architecture_model.svg" alt="Kiến trúc Geo-I: input, các layer trong khung mô hình và output công khai/riêng">',
            'Khung xanh là mô hình. Output công khai: K điểm dummy và thời điểm phát. Output riêng: top-5 POI tại thiết bị.')
    table(['Component', 'Vai trò trọng tâm'], [
        ['Geo-I / REM — S1', 'Tạo neo bằng cơ chế nhiễu xác suất; bảo vệ tọa độ từng lần đọc.'],
        ['Tái dùng + ngân sách + pacing — S2/S3', 'Giảm nhiều mẫu độc lập khi dừng; giới hạn rò rỉ qua chuỗi; giãn đọc GPS.'],
        ['Belief + road-aware — S3', 'Ước lượng vị trí từ neo đã bảo vệ; dummy đi được theo hướng làn và luật rẽ.'],
        ['POI-aware + cache — dịch vụ', 'Chọn tập dummy phủ POI bổ sung nhau; hợp phản hồi còn hiệu lực, chọn top-5 cục bộ.'],
        ['Boundary — S9/S10', 'S9 bỏ đầu trước lõi; S10 buffer sau lõi, hủy phần chưa phát khi chuyến kết thúc.']
    ], [5.0, 11.9])
    p('Bộ chọn dummy chỉ dùng neo đã bảo vệ và dữ liệu công khai. K dummy không tự tạo k-anonymity. Boundary đã kiểm tra tích hợp, chưa benchmark kết hợp với dịch vụ.')

    evidence = add_geoi_content(ns)

    literature = coverage_evidence(ns['refs'])
    literature['presentation_headers'] = ['Paper', 'Fully', 'Partially / CX', 'Privacy / chẩn đoán', 'Utility / chi phí']
    literature['presentation_rows'] = merged_rows
    (out/'related_work_coverage.json').write_text(json.dumps(literature, ensure_ascii=False, indent=2)+'\n')

    guide = json.loads((out/'scenario_guide.json').read_text())
    readings = json.loads((out/'case_readings.json').read_text())
    questions = {
        'S1': 'Đối thủ cần suy ra tọa độ tại một lần gửi. A/B thay ràng buộc mạng đường; C thêm ngữ cảnh POI hiếm.',
        'S2': 'Đối thủ cần suy ra tọa độ nơi dừng. A/B thay thời lượng và nhịp quan sát; C kiểm tra rời đi rồi quay lại.',
        'S3': 'Đối thủ cần tái dựng các vị trí của đoạn đã đi. A là chuỗi đều; B ít nhánh; C quan sát thưa.',
        'S9': 'Đối thủ cần suy điểm đầu bị giấu. A suy ngược một chuyến; B phân biệt hai nguồn nhập tuyến; C kết hợp chuyến lặp.',
        'S10': 'Đối thủ cần suy điểm cuối của chuyến đã hoàn tất. A dùng một chuyến; B kết hợp nhiều chuyến cùng đích. Đây khác dự đoán đích tương lai ở S6.'}
    panel_count = 0
    for i, sc in enumerate(questions):
        page()
        if i == 0:
            sec('Phụ lục: đọc chi tiết 14 sample A/B/C')
            p('A/B/C là các điều kiện cùng nhiệm vụ, không phải thứ tự độ khó. Mỗi ca chọn một bản ghi thật từ bộ minh họa 12 nhóm/264 chuyến; số bản ghi không phải số chuyến độc lập. **S10 chỉ có A/B** trong tài liệu; B ánh xạ mã nguồn cũ S10.C để giữ truy nguyên, không có ca C độc lập trong phạm vi hiện tại.')
            p('Chấm màu = mẫu trước bảo vệ; nét đứt = đường thật để chấm; sao đỏ = đáp án; trục thời gian tách các mẫu chồng nhau. Đối thủ chỉ nhận dữ liệu đã bảo vệ và thông tin phụ trợ được phép, không nhận các tọa độ/nhãn thật này. Bản đồ minh họa dùng nền SUMO với cùng mã kiểm tra tệp OSM; thiếu mạng gốc để xác minh trùng toàn bộ hình học. © OpenStreetMap contributors.')
        sub(guide[sc]['title'])
        p(questions[sc])
        for c, desc in zip(ns['suffixes'](sc), guide[sc]['cases']):
            case = sc+'.'+c
            r = ns['FIRST'][case]
            counts = '**Bản ghi minh họa '+r['record_id']+'.** '+'; '.join(
                sid+': '+str(len(ix))+' mẫu, chỉ số GPS FCD['+str(ix[0])+'…'+str(ix[-1])+']'
                for sid, ix in zip(r['session_ids'], r['observed_indices']))
            ns['blocks'].append(('case_panel', (case, desc, readings[display_case(case)], counts)))
            panel_count += 1
    assert panel_count == 14
    p('S9/S10 chấm tọa độ đầu/cuối; chưa gán nhãn nhà/nơi làm việc. Xem bản đồ phóng to của 29 ca của khung đầy đủ ở sample_maps.html; tọa độ và bản ghi ở data_samples.json, số đếm ở scenario_inventory.csv.')

    page(); sec('Nguồn và khả năng truy nguyên')
    p('Kết quả so trực tiếp: artifacts/benchmarks/paper_benchmark/results.json và docs/reviews/verification_paper_benchmark_v2.md. Bản mới: iteration18_expanded_attacks.json, iteration28_live_service.json và iteration28_verification.json trong artifacts/benchmarks/research_loop/. geoi_evidence.py kiểm tra hash và gộp lại đúng 14 ca; method_evidence.json lưu nguồn, số đo và bảng đã in. preparation_evidence.json truy nguyên guide. Không chạy thêm mô hình; hai phiên bản được báo riêng.')
    p('Ngày 03/10/2026 đưa Geo-I về phương pháp chính; giữ nguyên thực nghiệm gốc và báo riêng hai phiên bản. S10.B là nhãn cho mã nguồn S10.C cũ. Cấu hình, benchmark và phụ lục được dùng chung giữa report/guide; cách dựng và tệp tác giả ở README.md.')
    for rid, citation, url, doi, verify, role in ns['refs']:
        ns['blocks'].append(('ref', (rid, citation, url, verify)))
    import hashlib
    paths = ['core/mechanisms.py', 'core/boundary_release.py',
             'benchmark/engines/budgeted.py', 'benchmark/engines/switching_cover.py',
             'benchmark/engines/paced_slack.py', 'benchmark/engines/filtered_cover.py',
             'benchmark/engines/contextual_lane.py', 'benchmark/engines/slack_progress.py',
             'benchmark/anchor_belief.py', 'benchmark/switching_belief.py']
    paths += [str((out/n).relative_to(ns['ROOT'])) for n in
              ('geoi_evidence.py', 'geoi_content.py', 'related_work_coverage.py', 'related_work_coverage.json', 'plot_model_architecture.py', 'figures/architecture_model.pdf', 'figures/architecture_model.svg')]
    evidence['sources'].update({n: hashlib.sha256((ns['ROOT']/n).read_bytes()).hexdigest() for n in paths})
    evidence['model_architecture'] = {
        'name': 'BR-Dummy: Geo-I anchor + protected-history belief + directed-road/POI-aware dummy selection',
        'working_name': 'GeoI-Slack', 'internal_control_name': 'GeoI-Paced',
        'boundary_wrapper': 'BR-Boundary v1; integration evidence, not full service benchmark'}
    evidence['report_case_labels'] = ns['report_label_manifest']()
    (out/'method_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n')
    return ns['blocks']
