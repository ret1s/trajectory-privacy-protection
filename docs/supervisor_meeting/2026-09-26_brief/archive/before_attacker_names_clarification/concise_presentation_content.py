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
    sec('Related works: mức bao phủ S1–S10')
    p('Khảo sát giữ cửa sổ **21/09/2023–21/09/2026**. Bảng đối chiếu từng paper với mười nhiệm vụ của ta; mức bao phủ phải đọc cùng giả định và giới hạn ở cột cuối.')
    p(SCENARIOS)
    p(LEGEND)
    table(['Paper', 'Fully', 'Partially / CX', 'Căn cứ và giới hạn'], ROWS, [3.5, 1.8, 2.7, 8.9])
    p(NOTE)
    page(); sub('Metrics gốc và điều rút ra từ related works')
    p('↑/↓ là hướng tốt hơn; CX là chưa đủ nguồn xác minh. Điểm số gốc không so trực tiếp vì khác output, dataset và mục tiêu.')
    table(['Paper / hướng nghiên cứu', 'Privacy / chẩn đoán', 'Utility', 'Chi phí / phạm vi'], [
        ['TransProtect 2024 [R1]', 'EIE ↑: sai số suy luận', 'Δc ↓: chênh chi phí hành trình', 'Byte + độ trễ toàn dịch vụ: CX'],
        ['Semantic correlation 2026 [R2]', 'ASR ↑: đạt ẩn danh; DER ↑: dummy/ngữ nghĩa', 'Chưa đo top-5 POI bằng DER', 'Thời gian sinh ↓'],
        ['Fake queries 2026 [R3]', 'Số đường khó phân biệt ↑; ASR ↑', 'Chất lượng POI: CX', 'Thời gian sinh, RSS ↓'],
        ['Road-aware PTPPM 2025 [R4]', 'Sai số ↑; tấn công đúng ↓', 'Dịch chuyển đầu ra ↓', 'Chi phí LBS đầu-cuối: CX; preprint'],
        ['CPCROK 2025 [R5]', 'Tỷ lệ khớp quỹ đạo ρ ↓', 'Bối cảnh VANET', 'Số quỹ đạo giả ψ ↓; ước lượng gián tiếp chi phí'],
        ['DP-FETC 2025 [R6]', 'MI ↓; bảo đảm ε-DP', 'JSD phân bố ↓', 'Độ phức tạp; xuất bản offline'],
        ['Improved PIR 2025 [R7]', 'Phân tích an toàn; metric tấn công: CX', 'Truy hồi cụ thể: CX', 'Chi phí tính toán; mới đủ tóm tắt'],
        ['SoK 2024 [R8]', 'Tấn công tái dựng/liên kết', 'Phân bố, hình học, tác vụ', 'Khảo sát, không phải model'],
        ['PRISM 2025 [R9]', 'LPI; công thức CX', 'TDD, DC; định nghĩa CX', 'Thời gian xử lý/phản hồi'],
        ['Options to Action 2026 [R10]', 'Tỷ lệ dùng tính năng', 'Nhận thức và sử dụng', 'Nghiên cứu hành vi'],
        ['Forsch et al. 2023 [R11]', 'Trình bày S-TT / k-site-unidentifiability', 'Che điểm nhạy cảm', 'Chương tổng quan; kế thừa S-TT 2022'],
        ['EV querying 2024 [R12]', 'AGeoI + dummy', 'Dịch vụ truy vấn trạm sạc', 'Chưa xác minh phép thử điểm đầu/cuối']
    ], [4.0, 4.5, 4.0, 4.4])
    p('Mốc nền: DLS 2014 [B1], RDG 2021 [B2]. AnotherMe [B3] online 11/09/2023, ngoài cửa sổ dù số tạp chí ghi 2024.')

    sub('Cách đọc bộ đo chung')
    p('Một paper trả vị trí thay thế, paper khác trả tập dummy hoặc query chèn. Vì vậy, ta chấm cùng đáp án từ toàn bộ dữ liệu attacker được phép thấy; đo cùng dịch vụ và tính đủ traffic.')
    table(['Chỉ số', 'Con số có nghĩa gì?', 'Hướng tốt hơn'], [
        ['Hit100 (%)', 'Tỷ lệ attacker đoán cách vị trí thật ≤100 m. 33,33% nghĩa là khoảng một phần ba trường hợp.', 'Thấp hơn: khó đoán gần hơn'],
        ['MAE (m)', 'Sai số suy luận trung bình. 299 m nghĩa là dự đoán lệch vị trí thật trung bình 299 m.', 'Cao hơn: attacker sai xa hơn'],
        ['Recall@5 (%)', 'Tỷ lệ tìm lại đúng POI chuẩn. Đúng 4/5 địa điểm = 80%. Ngưỡng mỗi ca: 90%.', 'Cao hơn: dịch vụ tốt hơn'],
        ['Byte/sự kiện', 'Dữ liệu yêu cầu + phản hồi cho mỗi lần dùng dịch vụ.', 'Thấp hơn: ít traffic hơn']
    ], [3.2, 9.1, 4.6])
    key('Hit100 là tỷ lệ đoán đúng của attacker, không phải độ chính xác của model ta. Đọc theo đơn vị và chú giải của bảng; xét privacy, utility và chi phí cùng nhau.')

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

    (out/'related_work_coverage.json').write_text(json.dumps(coverage_evidence(ns['refs']), ensure_ascii=False, indent=2)+'\n')

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
