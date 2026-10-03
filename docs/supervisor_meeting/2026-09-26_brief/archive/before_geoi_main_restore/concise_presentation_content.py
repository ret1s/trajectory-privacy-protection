"""Concise presentation with shared literature coverage and priority-case appendix."""
from related_work_coverage import LEGEND, SCENARIOS, ROWS, NOTE, coverage_evidence
from public_cover_content import PUBLIC_LABELS, build_public_cover
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
    labels.update(PUBLIC_LABELS)
    h = {(r['method'], r['case_id']): r for r in scope['historical']['summaries'] if r['split'] == 'new_groups'}
    ha = {r['method']: r for r in scope['historical']['aggregates'] if r['split'] == 'new_groups'}
    ea = {r['method']: r for r in scope['endpoint']['aggregates']}
    es = {(r['method'], r['case_id']): r for r in scope['endpoint']['summaries']}
    byte = lambda a: a['request_bytes_per_service_event'] + a['response_bytes_per_service_event']
    hit = lambda m, sc: sum(h[m, sc+'.'+c]['hit100'] for c in 'ABC') / 3

    sec('Related works: mức bao phủ S1–S10')
    p('Khảo sát giữ cửa sổ **21/09/2023–21/09/2026**. Bảng đối chiếu từng paper với mười nhiệm vụ của ta; mức bao phủ phải đọc cùng giả định và giới hạn ở cột cuối.')
    p(SCENARIOS)
    p(LEGEND)
    table(['Paper', 'Fully', 'Partially / CX', 'Căn cứ và giới hạn'], ROWS, [3.5, 1.8, 2.7, 8.9])
    p(NOTE)
    page(); sub('Metrics gốc và điều rút ra từ related works')
    p('Bảng dưới ghi các chỉ số đã xác minh trong nguồn. ↑/↓ là hướng tốt hơn theo mục tiêu của paper; CX nghĩa là chưa xác minh đủ định nghĩa. Các paper dùng dataset và nhiệm vụ khác nhau, nên không so trực tiếp điểm số gốc của chúng.')
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
    p('**Mốc nền giữ riêng:** DLS 2014 [B1], RDG 2021 [B2]; AnotherMe [B3] có số tạp chí 2024 nhưng online 11/09/2023, ngoài cửa sổ. S-TT 2022 [B4], tấn công EPZ 2022 [B5] và Data Stream Obfuscator 07/2023 [B6] bổ sung góc nhìn điểm đầu/cuối. Không gọi các nguồn này là paper mới trong ba năm.')
    key('Các paper đo những khía cạnh khác nhau. Để so trên bài toán của ta, cần cùng đo khả năng suy luận của đối thủ, chất lượng dịch vụ và chi phí.')

    page(); sub('Vì sao chọn bộ metrics đề xuất?')
    table(['Vấn đề từ related works', 'Metric đề xuất', 'Ý nghĩa trong bài toán của ta'], [
        ['Độ bất định, tương đồng dummy và lượng thông tin rò rỉ đo các khía cạnh khác nhau. ASR cũng không có cùng định nghĩa giữa các paper.',
         'Hit50/100/200 ↓; MAE, median, p90 ↑',
         'Đối thủ (attacker) phải đoán tọa độ thật từ dữ liệu công bố. Hit100 là tỷ lệ đoán cách đáp án ≤100 m, dùng được cho cả tập dummy và vị trí thay thế.'],
        ['MAE lớn vẫn có thể che một số lần suy đúng rất gần.',
         'Đọc Hit cùng phân bố sai số',
         'MAE là sai số trung bình; median là trung vị; p90 là ngưỡng chứa 90% sai số mẫu. S1 chấm một điểm, S2 nơi dừng, S3 chuỗi điểm, S9/S10 điểm đầu/cuối.'],
        ['Bảo toàn phân bố hoặc giảm độ lệch chưa cho biết người dùng có nhận đúng địa điểm cần tìm.',
         'Recall@5 ↑; trả đủ ↑; ΔD đường ↓',
         'So 5 địa điểm được chọn tại thiết bị với 5 địa điểm chuẩn theo GPS thật, cùng trạng thái máy chủ. Mỗi ca cần Recall ≥90%.'],
        ['Chỉ đếm dummy hoặc thời gian sinh sẽ bỏ sót phản hồi, truy vấn chèn và tải ngoài chuyến.',
         'Byte yêu cầu + phản hồi ↓; latency p50/p95 ↓',
         'Đếm toàn bộ dữ liệu gửi/nhận cho mỗi lần dùng dịch vụ thật. Độ trễ toàn dịch vụ chưa đo; chưa có điểm tổng Q thực nghiệm.']
    ], [5.6, 4.2, 7.1])
    p('POI là địa điểm như nhà thuốc hoặc trạm sạc. Recall@5 = số POI chuẩn tìm được / số POI chuẩn (tối đa 5). Ví dụ, tìm đúng 4/5 địa điểm được 80%. Có đáp án mà không phục vụ thì tính 0; không có đáp án chuẩn thì để thiếu. “Trả đủ” đo số lượng; ΔD đo khoảng cách đường tăng thêm.')
    p('Để so công bằng, giữ cùng mục tiêu, quyền quan sát và dịch vụ: máy chủ trả top-10, thiết bị chọn top-5. Với mỗi loại dữ liệu công bố (transcript), chọn bộ suy luận phù hợp trên nhóm phát triển rồi giữ cố định khi kiểm tra. Gộp các lần chạy trong bản ghi và nhóm tuyến; cho các ca trọng số bằng nhau trong scenario, rồi cho năm scenario trọng số bằng nhau. S10 chỉ có A/B. Chi phí truyền vẫn phải báo riêng.')
    key('Bộ đo trả lời ba câu hỏi: đối thủ đoán đúng tới đâu, người dùng nhận đúng kết quả đến đâu và cần bao nhiêu chi phí. Đóng góp là cách chọn và áp dụng chung các chỉ số đã có cho nhiệm vụ này.')

    page(); sec('Kiến trúc mô hình BR-Dummy: Geo-I, mạng đường và các cơ chế')
    p('**Nền tảng là Geo-I:** từ GPS thật, mô hình lấy ngẫu nhiên một vị trí đại diện đã được bảo vệ, gọi là **neo**. Lịch sử neo, mạng đường và POI công khai được dùng để sinh K vị trí/quỹ đạo giả. Lớp xử lý biên (boundary) bổ sung bảo vệ điểm đầu/cuối.')
    graphic(r'\includegraphics[width=\linewidth]{figures/architecture_model.pdf}',
            '<img src="figures/architecture_model.svg" alt="Kiến trúc BR-Dummy: neo Geo-I, ngân sách, belief, ràng buộc mạng làn, chọn dummy theo POI và mở rộng boundary S9/S10">',
            'Đọc từ trái sang phải: input → layer bảo vệ Geo-I → layer ngữ cảnh → layer sinh dummy → output. Khung xanh bao toàn bộ mô hình của ta, gồm ba layer lõi và layer 4 boundary S9/S10 ở hàng dưới; input và bên nhận ở ngoài khung. Switching, pacing và slack tùy biến thể.')
    table(['Component', 'Cơ chế và vai trò đối với scenario'], [
        ['Geo-I / neo REM', 'Lấy neo bằng cơ chế mũ REM: vị trí càng xa GPS theo khoảng cách Euclid, xác suất được chọn càng thấp. Tập ứng viên đường được xác định công khai; không cắt ứng viên theo bán kính riêng quanh GPS. Đây là nền bảo vệ S1.'],
        ['Tái dùng neo + ngân sách', 'Kiểm tra khoảng cách có nhiễu để giữ neo cũ hoặc tạo neo mới; cả hai bước đều tiêu ngân sách riêng tư. Tái dùng neo giúp giảm số mẫu độc lập khi dừng (S2). Ngân sách giới hạn rò rỉ tích lũy qua chuỗi (S3); pacing giãn các lần đọc GPS.'],
        ['Belief từ lịch sử neo', 'Belief là phân bố ước lượng xe có thể đang ở đâu, được cập nhật từ neo đã bảo vệ. Biến thể switching phân biệt trạng thái dừng/đi. Khối này hỗ trợ chọn dummy cho S2/S3, không dùng tốc độ thật hay đích tương lai.'],
        ['Road-aware + POI-aware', 'Road-aware giới hạn dummy ở nơi có thể tới theo hướng làn và luật rẽ. POI-aware chọn K điểm để phủ nhiều POI bổ sung nhau. Exchange đổi ứng viên; slack cho phép độ lệch khi chọn, tùy biến thể. Hỗ trợ chuỗi hợp lý (S3) và dịch vụ; sức chống suy luận vẫn cần đo.'],
        ['S9: bỏ đầu trước lõi', 'Không đưa GPS của h giây đầu vào lõi. Vì vậy, những điểm đầu bị che không đi vào trạng thái neo rồi ảnh hưởng các bản tin sau. Đối thủ vẫn có thể suy ngược từ tuyến còn thấy.'],
        ['S10: buffer sau lõi', 'Giữ mỗi bản tin đã bảo vệ trong bộ đệm Δ giây rồi mới phát. Khi chuyến kết thúc, hủy bản tin chưa phát. Cách này che phần cuối mà không cần biết trước đích, nhưng làm tăng độ trễ và chưa loại hết suy luận từ phần còn thấy.']
    ], [4.1, 12.8])
    p('Belief, mạng đường và bộ chọn POI chỉ xử lý neo đã bảo vệ cùng dữ liệu công khai. Theo tính chất hậu xử lý, chúng giữ bảo đảm của neo khi tính đúng ngân sách qua các lần truy cập (composition). K dummy không tự tạo k-anonymity. GPS thật chỉ thêm ở bước chọn top-5 tại thiết bị sau phản hồi, không đưa vào bộ chọn dummy. Pacing chưa che giờ chuyến. Boundary đã kiểm tra mã/tích hợp; chưa có benchmark toàn mô hình với dịch vụ.')

    page(); build_public_cover(ns)

    page(); sec('Benchmark: phân biệt lõi Geo-I và nhánh truy vấn công khai')
    key('Phạm vi bảng dưới: CoverLite-30 / CoverPlus-67 thuộc nhánh PublicCover, không dùng neo Geo-I. Đây là kết quả của nhánh mở rộng dịch vụ; không gán Hit100=0 cho kiến trúc Geo-I/BR-Dummy hoặc BR-Boundary ở mục 2.')
    p('Các đối chứng chạy cùng dịch vụ và được chấm trên cùng mục tiêu. **Năm bản triển khai điều chỉnh từ paper (adapter), chạy trực tuyến:** DLS, RDG, TransProtect*, Semantic*, Fake-query*. * TransProtect dùng Markov thay Transformer; Semantic dùng bộ dự báo thực nghiệm thay LSTM; Fake-query dùng giả định lịch chèn công khai. AnotherMe là tham chiếu offline, báo riêng phía dưới.')
    sub('Nhánh công khai: S1–S3 trên bốn nhóm tuyến')
    p('54 bản ghi từ 30 chuyến trong bốn nhóm 901–904; đối thủ được huấn luyện/chọn trên nhóm khác. Bảng là **Hit100 ↓**, trung bình đều A/B/C; chất lượng dịch vụ tại p=0,8 (80% POI khả dụng), trung bình đều năm scenario. Đây chưa phải kết quả của cơ chế gửi theo lịch trên bộ 32 nhóm.')
    methods = ['raw'] + online + ['ours30', 'ours67']
    table(['Phương pháp', 'S1', 'S2', 'S3', 'Recall@5 ↑', 'Byte / sự kiện ↓'], [
        [labels[m]] + [pct(hit(m, sc)) for sc in ('S1', 'S2', 'S3')]
        + [pct(ha[m]['recall_0.8']), num(byte(ha[m]), 1)] for m in methods
    ], [3.6, 2.1, 2.1, 2.1, 3.0, 3.7])
    for m in ('ours30', 'ours67'):
        assert all(hit(m, sc) == 0 for sc in ('S1', 'S2', 'S3'))
    p('**Cách đọc:** vị trí thật và các adapter vẫn cho đối thủ suy đúng gần ở một phần ca; hai kế hoạch công khai đạt Hit100=0 trong tập đối thủ đã thử. Điều này phù hợp với việc tọa độ công bố không bám GPS, nơi dừng hay tuyến thật. Không diễn giải 0% thành an toàn trước mọi đối thủ.')
    p('**Chất lượng dịch vụ và giá phải trả:** bản 30 giảm traffic so với 67 nhưng độ phủ POI thấp hơn. Bản 67 đạt Recall cao hơn nhờ thêm query; cần đọc riêng cả ba mặt: riêng tư, dịch vụ và chi phí. Byte là request + response JSON, giữ cả query phụ; chưa gồm HTTP/TLS. Bản 67 còn tám POI ngoài độ phủ tĩnh, nên Recall=100% trên mẫu chưa là chứng nhận toàn bản đồ.')

    page(); sub('Nhánh công khai: S9–S10 trên 32 nhóm tuyến, gửi theo lịch công khai')
    p('Bộ 32 nhóm có 704 chuyến. Chất lượng dịch vụ được chấm trên 435 bản ghi từ 246 chuyến; riêng tư điểm đầu/cuối trên 150 bản ghi. Đối thủ được chọn cố định trước khi kiểm tra bộ này, có thể suy xuôi/ngược theo mạng đường và kết hợp chuyến lặp. S9 gộp A/B/C, S10 gộp A/B; mỗi ô privacy là **Hit100 ↓ / MAE (m) ↑**.')
    methods = ['raw'] + online + ['calendar30', 'calendar67']
    table(['Phương pháp', 'S9: Hit / MAE', 'S10: Hit / MAE', 'Recall@5 ↑', 'Byte / sự kiện ↓'], [
        [labels[m], pct(ea[m]['S9']['hit100'])+' / '+num(ea[m]['S9']['mae_m']),
         pct(ea[m]['S10']['hit100'])+' / '+num(ea[m]['S10']['mae_m']),
         pct(ea[m]['recall_0.8']), num(byte(ea[m]), 1)] for m in methods
    ], [3.3, 3.8, 3.8, 2.8, 3.0])
    p('**Privacy:** lịch công khai đạt Hit100=0 so với S9 **19,76–40,81%** và S10 **10,75–26,11%** của năm adapter. Đối chứng dùng GPS thật vẫn cho phép suy điểm cuối ở cả S10.A ('+pct(es['raw', 'S10.A']['hit100'])+') và S10.B ('+pct(es['raw', 'S10.C']['hit100'])+'), nên phép thử vẫn phát hiện được rò rỉ khi có. Khoảng tin cậy (CI) 95% của chênh lệch Hit100 giữa đề xuất và từng adapter nằm dưới 0 ở S9/S10, hỗ trợ kết luận đề xuất có tỷ lệ đoán đúng thấp hơn. Tính bằng 3.000 lần lấy mẫu lại nhóm tuyến (bootstrap); chưa hiệu chỉnh nhiều so sánh.')
    p('**Chất lượng dịch vụ:** tại p=0,8, bản 30 đạt '+pct(ea['calendar30']['recall_0.8'])+', '+str(ea['calendar30']['gates_0.8'])+'/14 ca đạt ≥90%; bản 67 đạt '+pct(ea['calendar67']['recall_0.8'])+', 14/14 ca. Khi tăng tỷ lệ POI khả dụng lên p=0,95: bản 30 đạt '+pct(ea['calendar30']['recall_0.95'])+', '+str(ea['calendar30']['gates_0.95'])+'/14 ca; bản 67 đạt 100%, 14/14 ca ở các mức đã thử.')
    p('**Chi phí:** khoảng '+num(byte(ea['calendar30']))+' / '+num(byte(ea['calendar67']))+' byte mỗi sự kiện thật. So với cùng kế hoạch chỉ gửi lại truy vấn khi có hoạt động, lịch công khai tốn '+num(scope['ablation']['calendar_to_active_epoch_byte_ratio']['calendar67'], 2)+' lần byte: đây là giá của việc che giờ hoạt động, không phải tăng độ phủ POI.')
    am = ea['anotherme_offline']
    p('**AnotherMe:** paper mô tả hệ thống online, nhưng adapter VTGA hiện có đọc cả chuyến. Trên 32 nhóm, riêng tư chỉ chấm trên tập con sinh đầu ra thành công có Hit100=0, Recall toàn phạm vi chỉ '+pct(am['recall_0.8'])+'. Ca lỗi/thiếu đầu ra vẫn tính chất lượng dịch vụ bằng 0, chỉ số riêng tư để thiếu; không so trực tiếp các số này như có cùng mẫu số với năm adapter online.')
    key('Kết luận cho nhánh công khai: thiết kế phối hợp kế hoạch công khai, lịch tải và xếp hạng cục bộ có tỷ lệ suy đúng thấp hơn trong năm scenario đã thử, đổi lại traffic cao hơn. Chưa chứng minh vượt các paper nguyên bản hoặc tốt hơn ở cùng ngân sách; dữ liệu vẫn cùng thành phố/bộ sinh SUMO.')

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
    p('Bảng rút gọn đọc artifacts/benchmarks/active_scope_ac_v2/readout.json đã xác minh, không chạy thêm mô hình. method_evidence.json giữ hash benchmark; preparation_evidence.json giữ bảng guide. Hồ sơ metrics gốc nằm trong bản đầy đủ lưu ở archive/before_concise_2026-10-03/ và sources.json. public_cover_evidence.json truy nguyên cấu hình và tên CoverLite-30/CoverPlus-67. Module concise_presentation_content.py tạo nội dung rút gọn; plot_model_architecture.py vẽ mô hình từ các component trong mã. scenario_appendix.tex dùng chung cho hai tài liệu.')
    p('Ngày 03/10/2026 cập nhật cách trình bày, đối chiếu bao phủ scenario và làm rõ kiến trúc; không thay số liệu thực nghiệm. Tra data_samples.json để phân biệt nhãn trình bày với mã thực nghiệm, đặc biệt S10.B (mã nguồn S10.C cũ).')
    for rid, citation, url, doi, verify, role in ns['refs']:
        ns['blocks'].append(('ref', (rid, citation, url, verify)))
    import hashlib
    paths = ['core/mechanisms.py', 'core/boundary_release.py',
             'benchmark/engines/budgeted.py', 'benchmark/engines/switching_cover.py',
             'benchmark/engines/paced_slack.py', 'benchmark/engines/filtered_cover.py',
             'benchmark/engines/contextual_lane.py', 'benchmark/engines/slack_progress.py',
             'benchmark/anchor_belief.py', 'benchmark/switching_belief.py']
    paths += [str((out/n).relative_to(ns['ROOT'])) for n in
              ('public_cover_content.py', 'public_cover_evidence.json', 'plot_publiccover_architecture.py', 'figures/architecture_publiccover.pdf', 'figures/architecture_publiccover.svg', 'related_work_coverage.py', 'related_work_coverage.json', 'plot_model_architecture.py', 'figures/architecture_model.pdf', 'figures/architecture_model.svg')]
    evidence = json.loads((out/'method_evidence.json').read_text())
    evidence['sources'].update({n: hashlib.sha256((ns['ROOT']/n).read_bytes()).hexdigest() for n in paths})
    evidence['model_architecture'] = {
        'name': 'BR-Dummy: Geo-I anchor + protected-history belief + road/POI-aware dummy selection',
        'boundary_wrapper': 'BR-Boundary v1; integration evidence, not full service benchmark',
        'public_30_67_tables_are_not_GeoI_model_results': True,
        'public_cover_names': PUBLIC_LABELS}
    (out/'method_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n')
    return ns['blocks']
