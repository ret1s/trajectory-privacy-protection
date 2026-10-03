"""Five-page presentation with a shared, source-backed priority-case appendix."""
import json
from report_case_labels import display_case


def build_concise_content(ns):
    ns['blocks'] = []
    p, key, sec, sub, page, table, graphic = (
        ns[k] for k in ('p', 'key', 'sec', 'sub', 'page', 'table', 'graphic'))
    scope, out = ns['SCOPE'], ns['OUT']
    pct = lambda v: '—' if v is None else f'{100*v:.2f}%'.replace('.', ',')
    num = lambda v, d=0: '—' if v is None else f'{v:,.{d}f}'.replace(',', 'X').replace('.', ',').replace('X', '.')
    online = ['dls', 'rdg', 'transprotect_markov', 'semantic_poi', 'fake_queries']
    labels = {'raw': 'Vị trí thật', 'dls': 'DLS', 'rdg': 'RDG',
              'transprotect_markov': 'TransProtect*', 'semantic_poi': 'Semantic*',
              'fake_queries': 'Fake-query*', 'ours30': 'Đề xuất 30', 'ours67': 'Đề xuất 67',
              'calendar30': '30 + lịch', 'calendar67': '67 + lịch'}
    h = {(r['method'], r['case_id']): r for r in scope['historical']['summaries'] if r['split'] == 'new_groups'}
    ha = {r['method']: r for r in scope['historical']['aggregates'] if r['split'] == 'new_groups'}
    ea = {r['method']: r for r in scope['endpoint']['aggregates']}
    es = {(r['method'], r['case_id']): r for r in scope['endpoint']['summaries']}
    byte = lambda a: a['request_bytes_per_service_event'] + a['response_bytes_per_service_event']
    hit = lambda m, sc: sum(h[m, sc+'.'+c]['hit100'] for c in 'ABC') / 3

    sec('Related works ba năm gần đây và bảng metrics gốc')
    p('Khảo sát đã cập nhật sang các nguồn trong **21/09/2023–21/09/2026** (mốc khóa khảo sát), tập trung vào bảo vệ vị trí/quỹ đạo và cách đánh giá. Bảng dưới rút gọn các metric đã kiểm tra trong hồ sơ nguồn; không xếp hạng bằng số liệu lấy từ các dataset khác nhau. CX = chưa xác minh đủ định nghĩa.')
    table(['Paper / hướng nghiên cứu', 'Privacy / chẩn đoán', 'Utility', 'Chi phí / phạm vi'], [
        ['TransProtect 2024 [R1]', 'EIE ↑: sai số suy luận', 'Δc ↓: chênh chi phí hành trình', 'Byte + latency đầu-cuối: CX'],
        ['Semantic correlation 2026 [R2]', 'ASR ↑: đạt ẩn danh; DER ↑: dummy/ngữ nghĩa', 'Chưa đo top-5 POI bằng DER', 'Thời gian sinh ↓'],
        ['Fake queries 2026 [R3]', 'Số đường khó phân biệt ↑; ASR ↑', 'Chất lượng POI: CX', 'Thời gian sinh, RSS ↓'],
        ['Road-aware PTPPM 2025 [R4]', 'Sai số ↑; tấn công đúng ↓', 'Dịch chuyển đầu ra ↓', 'Chi phí LBS đầu-cuối: CX; preprint'],
        ['CPCROK 2025 [R5]', 'Tỷ lệ khớp quỹ đạo ρ ↓', 'Bối cảnh VANET', 'Số quỹ đạo giả ψ ↓; proxy chi phí'],
        ['DP-FETC 2025 [R6]', 'MI ↓; bảo đảm ε-DP', 'JSD phân bố ↓', 'Độ phức tạp; xuất bản offline'],
        ['Improved PIR 2025 [R7]', 'Phân tích an toàn; metric tấn công: CX', 'Truy hồi cụ thể: CX', 'Chi phí tính toán; mới đủ tóm tắt'],
        ['SoK 2024 [R8]', 'Tấn công tái dựng/liên kết', 'Phân bố, hình học, tác vụ', 'Khảo sát, không phải model'],
        ['PRISM 2025 [R9]', 'LPI; công thức CX', 'TDD, DC; định nghĩa CX', 'Thời gian xử lý/phản hồi'],
        ['Options to Action 2026 [R10]', 'Tỷ lệ dùng tính năng', 'Nhận thức và sử dụng', 'Nghiên cứu hành vi'],
        ['Forsch et al. 2023 [R11]', 'Trình bày S-TT / k-site-unidentifiability', 'Che điểm nhạy cảm', 'Chương tổng quan; kế thừa S-TT 2022'],
        ['EV querying 2024 [R12]', 'AGeoI + dummy', 'Dịch vụ truy vấn trạm sạc', 'Chưa xác minh benchmark endpoint']
    ], [4.0, 4.5, 4.0, 4.4])
    p('**Mốc nền giữ riêng:** DLS 2014 [B1], RDG 2021 [B2]; AnotherMe [B3] có số tạp chí 2024 nhưng online 11/09/2023, ngoài cửa sổ. S-TT 2022 [B4], tấn công EPZ 2022 [B5] và Data Stream Obfuscator 07/2023 [B6] bổ sung góc nhìn endpoint. Không gọi các nguồn này là paper mới trong ba năm.')
    key('Kết quả khảo sát: metric gốc chấm các mục tiêu khác nhau; cần một bộ đo chung theo điều đối thủ suy ra, dịch vụ người dùng nhận và chi phí thực tế.')

    page(); sub('Vì sao chọn bộ metrics đề xuất?')
    table(['Vấn đề từ related works', 'Metric đề xuất', 'Ý nghĩa trong bài toán của ta'], [
        ['Entropy, DER, MI hoặc nhận dạng thật/ảo không cùng đáp án; ASR có nghĩa khác nhau giữa paper.',
         'Hit50/100/200 ↓; MAE, median, p90 ↑',
         'Chấm tọa độ thật mà attacker suy ra, dù đầu ra là dummy hay vị trí thay thế. Hit100 là tỷ lệ đoán cách đáp án ≤100 m.'],
        ['MAE lớn vẫn có thể che một số lần suy đúng rất gần.',
         'Đọc Hit cùng phân bố sai số',
         'MAE cho độ lệch trung bình; Hit cho tỷ lệ lộ chính xác. S1: một điểm; S2: nơi dừng; S3: chuỗi điểm; S9/S10: endpoint.'],
        ['JSD, Δc và DER chưa trả lời người dùng có nhận đúng POI đang khả dụng.',
         'Recall@5 ↑; trả đủ ↑; ΔD đường ↓',
         'So top-5 cục bộ với top-5 chuẩn dùng GPS thật và cùng trạng thái server. Ngưỡng Recall từng ca ≥90%.'],
        ['Số dummy/thời gian sinh bỏ sót phản hồi, query chèn và tải khi không có chuyến.',
         'Byte yêu cầu + phản hồi ↓; latency p50/p95 ↓',
         'Đếm toàn bộ traffic trên mỗi sự kiện dịch vụ thật. Latency đầu-cuối chưa đo; chưa có điểm tổng Q thực nghiệm.']
    ], [5.6, 4.2, 7.1])
    p('Recall@5 = số POI đúng có trong kết quả / số POI của tập chuẩn (tối đa 5). Yêu cầu có đáp án nhưng cơ chế không phục vụ nhận Recall=0; yêu cầu không có POI chuẩn để null. “Trả đủ” đo số lượng; ΔD đo phần tăng khoảng cách đường. Hai chỉ số này bổ sung Recall khi phân tích utility chi tiết.')
    p('Để so được: giữ cùng mục tiêu, quyền quan sát, dịch vụ top-10/top-5 và quy tắc chọn attacker. Mỗi kiểu transcript có decoder phù hợp, được fit/chọn trên các nhóm riêng rồi khóa. Gộp seed/world trong record và nhóm, trung bình đều ca trong scenario rồi đều năm scenario; S10 chia hai ca A/B. Cùng dịch vụ vẫn phải báo riêng ngân sách truyền.')
    key('Bộ đo đề xuất vận hành hóa ba câu hỏi: attacker suy đúng tới đâu; người dùng nhận được gì; phải trả bao nhiêu. Các metric là chỉ số đã có, được chọn và ghép cho nhiệm vụ này.')

    page(); sec('Kiến trúc hiện tại và vai trò đối với S1/S2/S3/S9/S10')
    p('Dịch vụ có **419 POI, sáu loại**, trạng thái khả dụng thay đổi theo epoch 60 giây. Server trả tối đa top-10 cho một truy vấn loại–tọa độ; thiết bị chọn top-5 gần GPS thật theo mạng đường có hướng.')
    graphic(r'\includegraphics[width=\linewidth]{figures/architecture_current.pdf}',
            '<img src="figures/architecture_current.svg" alt="Kế hoạch công khai bảo vệ S1–S3; lịch cố định bổ sung bảo vệ S9–S10; GPS chỉ vào xếp hạng cục bộ">',
            'Luồng hiện tại: chuẩn bị từ dữ liệu công khai → tải theo lịch → nhận trạng thái → chọn POI tại thiết bị.')
    table(['Component', 'Đóng góp vào bảo vệ / dịch vụ'], [
        ['(1) Tập truy vấn công khai', 'Chọn tham lam theo POI chưa phủ, khóa 30/67 query tại 19/52 tọa độ. Không phụ thuộc GPS: hạn chế dấu hiệu về vị trí S1, nơi dừng S2 và đường đi S3; cũng không bám endpoint.'],
        ['(2) Lịch tải cố định', 'Mỗi 60 s trong [0, 3600), kể cả trước/sau chuyến và không dùng. Giờ đi/dừng không điều khiển giờ gửi: bổ sung cho S2 và đặc biệt S9/S10.'],
        ['(3) Cache phản hồi hợp lệ', 'Hợp kết quả cùng epoch; đọc cache không phát thêm query. Duy trì dịch vụ mà không lộ giờ sử dụng qua bản tin mới.'],
        ['(4) Xếp hạng bằng GPS trên máy', 'GPS và lựa chọn top-5 ở thiết bị. Giữ utility theo vị trí thật; không đưa thêm vị trí hoặc lựa chọn vào transcript.']
    ], [4.3, 12.6])
    p('Với vùng, kế hoạch, khoảng đăng ký và trạng thái server đã cố định, thay hành trình hoặc giờ đọc cache không đổi transcript. Điều kiện là không đổi vùng/kế hoạch theo GPS và không bật/tắt lịch theo chuyến. IP, account, click và lỗi mạng chưa nằm trong bảo đảm. Bản hiện tại dùng kế hoạch công khai và lịch; các lớp Geo-I/buffer biên lịch sử không tạo ra số liệu đang trình bày.')

    page(); sec('Benchmark: privacy tốt hơn với chi phí truyền tăng')
    p('Các đối chứng chạy cùng dịch vụ và được chấm trên cùng mục tiêu. **Năm adapter trực tuyến:** DLS, RDG, TransProtect*, Semantic*, Fake-query*. * TransProtect dùng Markov thay Transformer; Semantic dùng predictor thực nghiệm thay LSTM; Fake-query dùng giả định lịch chèn công khai. AnotherMe là tham chiếu offline, báo riêng phía dưới.')
    sub('S1–S3: bốn nhóm tuyến, client gửi theo sự kiện')
    p('54 record từ 30 chuyến trong bốn nhóm 901–904; attacker fit/chọn trên nhóm khác. Bảng là **Hit100 ↓**, trung bình đều A/B/C; utility ở p=0,8 (80% khả dụng), trung bình đều năm scenario. Đây chưa phải kết quả của client theo lịch trên cohort 32 nhóm.')
    methods = ['raw'] + online + ['ours30', 'ours67']
    table(['Phương pháp', 'S1', 'S2', 'S3', 'Recall@5 ↑', 'Byte / sự kiện ↓'], [
        [labels[m]] + [pct(hit(m, sc)) for sc in ('S1', 'S2', 'S3')]
        + [pct(ha[m]['recall_0.8']), num(byte(ha[m]), 1)] for m in methods
    ], [3.6, 2.1, 2.1, 2.1, 3.0, 3.7])
    for m in ('ours30', 'ours67'):
        assert all(hit(m, sc) == 0 for sc in ('S1', 'S2', 'S3'))
    p('**Cách đọc:** vị trí thật và các adapter vẫn cho attacker suy đúng gần ở một phần ca; hai kế hoạch công khai đạt Hit100=0 trong bank đã thử. Điều này phù hợp với việc tọa độ công bố không bám GPS, nơi dừng hay tuyến thật. Không diễn giải 0% thành an toàn trước mọi attacker.')
    p('**Utility và giá phải trả:** bản 30 giảm traffic so với 67 nhưng độ phủ POI thấp hơn. Bản 67 đạt Recall cao hơn nhờ thêm query; không coi privacy tốt hơn là thống trị cả ba trục. Byte là request + response JSON, giữ cả query phụ; chưa gồm HTTP/TLS. Bản 67 còn tám POI ngoài độ phủ tĩnh, nên Recall=100% trên mẫu chưa là chứng nhận toàn bản đồ.')

    page(); sub('S9–S10: 32 nhóm tuyến, client theo lịch công khai')
    p('Cohort có 704 chuyến; utility dùng 435 record từ 246 chuyến, privacy endpoint dùng 150 record. Attacker khóa trước cohort này, bổ sung suy luận xuôi/ngược trên mạng đường và kết hợp các chuyến lặp. S9 gộp A/B/C, S10 gộp A/B; mỗi ô privacy là **Hit100 ↓ / MAE (m) ↑**.')
    methods = ['raw'] + online + ['calendar30', 'calendar67']
    table(['Phương pháp', 'S9: Hit / MAE', 'S10: Hit / MAE', 'Recall@5 ↑', 'Byte / sự kiện ↓'], [
        [labels[m], pct(ea[m]['S9']['hit100'])+' / '+num(ea[m]['S9']['mae_m']),
         pct(ea[m]['S10']['hit100'])+' / '+num(ea[m]['S10']['mae_m']),
         pct(ea[m]['recall_0.8']), num(byte(ea[m]), 1)] for m in methods
    ], [3.3, 3.8, 3.8, 2.8, 3.0])
    p('**Privacy:** lịch công khai đạt Hit100=0 so với S9 **19,76–40,81%** và S10 **10,75–26,11%** của năm adapter. Raw control suy được endpoint ở cả S10.A ('+pct(es['raw', 'S10.A']['hit100'])+') và S10.B ('+pct(es['raw', 'S10.C']['hit100'])+'), nên phép thử có sức phân biệt. CI 95% của chênh lệch Hit100 nằm dưới 0 so với từng adapter ở cả S9/S10 (3.000 bootstrap theo nhóm; chưa hiệu chỉnh nhiều so sánh).')
    p('**Utility:** tại p=0,8, bản 30 đạt '+pct(ea['calendar30']['recall_0.8'])+', '+str(ea['calendar30']['gates_0.8'])+'/14 ca đạt ≥90%; bản 67 đạt '+pct(ea['calendar67']['recall_0.8'])+', 14/14 ca. Stress p=0,95: bản 30 đạt '+pct(ea['calendar30']['recall_0.95'])+', '+str(ea['calendar30']['gates_0.95'])+'/14 ca; bản 67 đạt 100%, 14/14 ca ở các mức đã thử.')
    p('**Chi phí:** khoảng '+num(byte(ea['calendar30']))+' / '+num(byte(ea['calendar67']))+' byte mỗi sự kiện thật. So cùng kế hoạch chỉ refresh khi có hoạt động, lịch công khai tốn '+num(scope['ablation']['calendar_to_active_epoch_byte_ratio']['calendar67'], 2)+' lần byte: đây là giá của việc che giờ hoạt động, không phải tăng độ phủ POI.')
    am = ea['anotherme_offline']
    p('**AnotherMe:** paper mô tả hệ thống online, nhưng adapter VTGA hiện có đọc cả chuyến. Trên 32 nhóm, privacy của tập con thành công có Hit100=0, Recall toàn phạm vi chỉ '+pct(am['recall_0.8'])+'. Lỗi/thiếu đầu ra giữ utility bằng 0, privacy để thiếu; không xếp các số này như cùng mẫu số với năm adapter online.')
    key('Kết luận hiện tại: thiết kế phối hợp kế hoạch công khai, lịch tải và xếp hạng cục bộ có lợi thế privacy trong năm scenario đã thử, đổi lại traffic cao hơn. Chưa chứng minh vượt các paper nguyên bản hoặc tốt hơn ở cùng ngân sách; dữ liệu vẫn cùng thành phố/bộ sinh SUMO.')

    guide = json.loads((out/'scenario_guide.json').read_text())
    readings = json.loads((out/'case_readings.json').read_text())
    questions = {
        'S1': 'Attacker cần suy ra tọa độ tại một lần gửi. A/B thay ràng buộc mạng đường; C thêm ngữ cảnh POI hiếm.',
        'S2': 'Attacker cần suy ra tọa độ nơi dừng. A/B thay thời lượng và nhịp quan sát; C kiểm tra rời đi rồi quay lại.',
        'S3': 'Attacker cần tái dựng các vị trí của đoạn đã đi. A là chuỗi đều; B ít nhánh; C quan sát thưa.',
        'S9': 'Attacker cần suy điểm đầu bị giấu. A suy ngược một chuyến; B phân biệt hai nguồn nhập tuyến; C kết hợp chuyến lặp.',
        'S10': 'Attacker cần suy endpoint của chuyến đã hoàn tất. A dùng một chuyến; B kết hợp nhiều chuyến cùng đích. Đây khác dự đoán đích tương lai ở S6.'}
    panel_count = 0
    for i, sc in enumerate(questions):
        page()
        if i == 0:
            sec('Phụ lục: đọc chi tiết 14 sample A/B/C')
            p('A/B/C là các điều kiện cùng nhiệm vụ, không phải thứ tự độ khó. Mỗi ca chọn một record thật từ bộ minh họa 12 nhóm/264 chuyến; số record không phải số chuyến độc lập. **S10 chỉ có A/B** trong tài liệu; B ánh xạ mã nguồn cũ S10.C để giữ truy nguyên, không có ca C độc lập trong phạm vi hiện tại.')
            p('Chấm màu = mẫu trước bảo vệ; nét đứt = đường thật để chấm; sao đỏ = đáp án; trục thời gian tách các mẫu chồng nhau. Attacker chỉ nhận transcript sau bảo vệ và phụ trợ được phép, không nhận các tọa độ/nhãn thật này. Map minh họa dùng nền SUMO cùng checksum OSM; thiếu mạng gốc để xác minh trùng toàn bộ hình học. © OpenStreetMap contributors.')
        sub(guide[sc]['title'])
        p(questions[sc])
        for c, desc in zip(ns['suffixes'](sc), guide[sc]['cases']):
            case = sc+'.'+c
            r = ns['FIRST'][case]
            counts = '**Mẫu '+r['record_id']+'.** '+'; '.join(
                sid+': '+str(len(ix))+' mẫu, FCD['+str(ix[0])+'…'+str(ix[-1])+']'
                for sid, ix in zip(r['session_ids'], r['observed_indices']))
            ns['blocks'].append(('case_panel', (case, desc, readings[display_case(case)], counts)))
            panel_count += 1
    assert panel_count == 14
    p('S9/S10 chấm tọa độ đầu/cuối; chưa gán nhãn nhà/nơi làm việc. Phóng to 29 ca của khung đầy đủ ở sample_maps.html; tọa độ và record ở data_samples.json, số đếm ở scenario_inventory.csv.')

    page(); sec('Nguồn và khả năng truy nguyên')
    p('Bảng rút gọn đọc artifacts/benchmarks/active_scope_ac_v2/readout.json đã xác minh, không chạy thêm model. method_evidence.json giữ hash benchmark; preparation_evidence.json giữ bảng guide. Hồ sơ metrics gốc nằm trong bản đầy đủ lưu ở archive/before_concise_2026-10-03/ và sources.json. Module concise_presentation_content.py tạo nội dung rút gọn; scenario_appendix.tex dùng chung cho hai tài liệu.')
    p('Ngày 03/10/2026 chỉ rút gọn nội dung và vẽ lại sơ đồ; mốc khảo sát và số liệu thực nghiệm được giữ riêng. Tra data_samples.json để phân biệt nhãn trình bày với mã thực nghiệm, đặc biệt S10.B (mã nguồn S10.C cũ).')
    for rid, citation, url, doi, verify, role in ns['refs']:
        ns['blocks'].append(('ref', (rid, citation, url, verify)))
    return ns['blocks']
