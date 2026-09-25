"""Final public category-query method, results and bounded interpretation."""
from pathlib import Path
import hashlib
import json


def add_method_content(ns):
    p, key, sec, sub, page, table, eq, bullets = (ns[x] for x in
        ['p', 'key', 'sec', 'sub', 'page', 'table', 'eq', 'bullets'])
    root, out = ns['ROOT'], ns['OUT']
    base = root/'artifacts/benchmarks/research_loop'
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    def load(name): return json.loads((base/name).read_text())
    def pct(v): return f'{100*v:.2f}%'.replace('.', ',')
    def num(v, digits=1): return f'{v:.{digits}f}'.replace('.', ',')
    old = load('iteration28_live_service.json')
    oldread = load('iteration28_readout.json')
    attack = load('iteration18_expanded_attacks.json')
    dev = load('iteration29_category_cover.json')
    new = load('iteration30_category_confirmation.json')
    readout = load('iteration30_readout.json')
    verified = load('iteration30_verification.json')
    plans = load('iteration29_plans.json')
    assert verified['status'] == 'passed'
    for doc in (dev, new, readout, verified):
        for path, expected in doc['source_sha256'].items():
            assert sha(root/path) == expected, path
    assert oldread['source_sha256'] == sha(base/'iteration28_live_service.json')
    assert old['provenance']['dataset_sha256'] == attack['dataset_sha256'] == dev['dataset_sha256']
    assert old['provenance']['screen_sha256'] == attack['screen_sha256']
    for path, expected in old['provenance']['source_sha256'].items():
        assert sha(root/path) == expected, path
    assert dev['plans_sha256'] == sha(base/'iteration29_plans.json')
    assert plans['protocol_sha256'] == sha(base/'iteration29_protocol.json')
    assert plans['planner_sha256'] == sha(root/'evaluation/category_cover.py')
    for item in old['shards']:
        assert sha(root/item['file']) == item['sha256']
    cases = [f'S{s}.{c}' for s in (1, 2, 3, 9, 10) for c in 'ABC']
    method = 'response_paced_slack03'
    b30, b67 = 'public_category_budget30', 'public_category_full_cover'
    os = {(s['method'], s['mode'], s['case_id']): s for s in old['summaries'] if s['probability'] == .8}
    scores = {(s['method'], s['mode']): s for s in oldread['table']}
    ds = {(s['method'], s['case_id']): s for s in dev['summaries'] if s['probability'] == .8}
    nscores = {(s['method'], s['case_id']): s for s in new['summaries'] if s['probability'] == .8}
    totals = {(s['split'], s['method'], s['probability']): s for s in readout['table']}
    costs = {(s['method'], s['mode']): s for s in readout['new_family_costs']}
    labels = {b30: 'Công khai: 30 query', b67: 'Công khai: 67 query',
              'fixed_K5': 'Fixed K5', 'fixed_K12': 'Fixed K12', 'bulk_current': 'Bulk trạng thái'}
    sources = ['artifacts/benchmarks/research_loop/'+x for x in (
        'iteration18_expanded_attacks.json', 'iteration28_protocol.json', 'iteration28_live_service.json',
        'iteration28_readout.json', 'iteration28_verification.json', 'iteration29_protocol.json',
        'iteration29_plans.json', 'iteration29_category_cover.json', 'iteration30_protocol.json',
        'iteration30_category_confirmation.json', 'iteration30_readout.json', 'iteration30_verification.json')]
    sources += ['artifacts/datasets/research_loop_expanded_v1/dataset.json',
        'artifacts/datasets/research_loop_confirmation_v1/dataset.json',
        'benchmark/engines/paced_slack.py', 'evaluation/live_poi.py',
        'evaluation/category_cover.py', 'benchmark/category_client.py',
        'experiments/research_loop_category_cover.py', 'experiments/research_loop_category_confirmation.py',
        'experiments/research_loop_category_readout.py', 'experiments/verify_research_loop_category.py',
        str(Path(__file__).relative_to(root)), str((out/'plot_current_architecture.py').relative_to(root)),
        str((out/'figures/architecture_current.pdf').relative_to(root)),
        str((out/'figures/architecture_current.svg').relative_to(root))]
    evidence = {'status': 'public_category_method_with_separate_comparator_and_dataset_scopes', 'checked_on': '2026-09-25',
        'sources': {x: sha(root/x) for x in sources},
        'final_configurations': {'30_queries': b30, '67_queries': b67},
        'geo_i_comparator': method+'/epoch_cache',
        'privacy_evidence': 'conditional GPS-independent transcript argument and information-flow checks; no new timing/identity attack bank',
        'same_K5_interface': False, 'same_bytes': False,
        'expanded_source_sessions': dev['physical_source_sessions'], 'expanded_records': dev['target_records'],
        'new_family_count': new['family_count'], 'new_completed_trips': new['new_SUMO_completed_sessions'],
        'new_source_sessions': new['source_sessions'], 'new_records': new['target_records'],
        'new_family_check_after_plan_freeze': True, 'independent_city_confirmation': False,
        'six_paper_superiority_established': False, 'global_full_cover_certificate': False,
        'new_family_30_query_p95_gates': totals['new_family_check', b30, .95]['case_gates_passed']}
    (out/'method_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n')

    # Proposed method: final architecture and the two operating configurations.
    page(); sec('Phương pháp truy vấn công khai theo loại POI')
    p('**Mục tiêu:** tìm top-5 POI đang khả dụng gần người dùng theo đường có hướng. Thiết bị có mạng đường và danh mục 419 POI; máy chủ giữ trạng thái khả dụng. Cơ chế chọn trước tập truy vấn từ dữ liệu công khai; GPS thật chỉ dùng để xếp hạng kết quả tại thiết bị.')
    ns['graphic'](r'\includegraphics[width=\linewidth]{figures/architecture_current.pdf}',
        '<img src="figures/architecture_current.svg" alt="Kiến trúc truy vấn công khai theo loại POI; GPS chỉ vào khối xếp hạng tại thiết bị" style="width:100%;height:auto">',
        'Hình. Kế hoạch truy vấn dùng chung cho mọi người trong vùng đã xác định. Máy chủ trả các POI khả dụng; thiết bị hợp phản hồi hợp lệ và dùng GPS thật để chọn kết quả, không gửi lựa chọn đó ra ngoài.')
    sub('Chọn tập truy vấn theo độ phủ danh mục')
    p('Với mỗi loại, dựng top-10 tĩnh tại các tọa độ ứng viên trên mạng đường. Mỗi bước chọn truy vấn bổ sung tỷ lệ POI chưa phủ lớn nhất của một loại; tọa độ phải qua đúng quy tắc server ánh xạ lên đường. Đây là tối ưu tham lam từ danh mục công khai, không phải deep learning hay một cơ chế nhiễu epsilon.')
    eq(r'(c^*,v^*)=\arg\max_{c,v}\frac{|\mathcal P_{10,c}(v)\setminus U_c|}{|\mathcal P_c|},\qquad U_{c^*}\leftarrow U_{c^*}\cup\mathcal P_{10,c^*}(v^*).',
       'Chọn loại c và tọa độ v bổ sung tỷ lệ POI chưa phủ lớn nhất; U là hợp POI đã phủ theo từng loại.')
    p('Tử số đếm POI mới do truy vấn bổ sung; mẫu số là tổng POI của loại đang xét. U là tập POI đã phủ của từng loại, được cập nhật sau mỗi lựa chọn.')
    table(['Cấu hình','Quy tắc dừng và phân bổ','Giao diện mỗi lần tải'],[
        ['30 truy vấn','Dừng ở 30: cà phê 11; phòng khám 1; trạm xăng 1; bệnh viện 1; nhà thuốc 2; nhà hàng 14.','30 truy vấn loại–tọa độ, tại 19 tọa độ khác nhau.'],
        ['67 truy vấn: phủ rộng','Tiếp tục đến khi không tăng độ phủ tĩnh; tổng 67 truy vấn.','67 truy vấn loại–tọa độ, tại 52 tọa độ khác nhau.']],[3.4,7.0,6.5])
    p('Mỗi tọa độ được giữ cố định; các loại không buộc dùng chung một tập vị trí. Thiết bị hỏi theo kế hoạch đã khóa, không chọn query từ GPS, tuyến tương lai hoặc trạng thái khả dụng thật. So sánh phải đếm riêng truy vấn loại, tọa độ và byte.')

    page(); sub('Lập luận privacy cho S1/S2/S3/S9/S10')
    p('Gọi X là hành trình; C gồm vùng dịch vụ, danh mục, kế hoạch và đồng hồ công khai; W là trạng thái server đã biết; T là transcript yêu cầu/phản hồi. Do bộ tạo yêu cầu không nhận GPS:')
    eq(r'T=f(C,W)\quad\Longrightarrow\quad I(X;T\mid C,W)=0.',
       'Với vùng, kế hoạch, đồng hồ và trạng thái server cố định, thay GPS không đổi transcript.')
    table(['Mục tiêu cần bảo vệ','Vì sao tọa độ công bố không bổ sung thông tin'],[
        ['S1: vị trí hiện tại','Ở vị trí nào trong cùng vùng, client cũng gửi cùng kế hoạch.'],
        ['S2: nơi dừng','Tọa độ không tập trung quanh nơi người dùng dừng; lịch gửi vẫn là điều kiện của lập luận.'],
        ['S3: đoạn đường đã đi','Chuỗi tọa độ cố định không bám đường đi thật.'],
        ['S9/S10: điểm đầu/cuối','Tọa độ đầu/cuối truy vấn không được chọn từ origin/destination; liên kết nhiều phiên cùng plan không thêm kênh tọa độ phụ thuộc GPS.']],[4.5,12.4])
    key('Bảo vệ phần thông tin bổ sung từ tọa độ, không phải bảo đảm đối thủ không thể đoán. Không gán Hit=0 hoặc MAE vô hạn từ công thức trên.')
    p('Prior, vùng dịch vụ, thời gian mở/đóng ứng dụng, account, IP hoặc click vẫn có thể tiết lộ thông tin. Tự đổi vùng, loại hoặc kế hoạch theo GPS tạo kênh mới ngoài lập luận. Chưa có bộ attacker mới đánh giá đầy đủ timing/identity; số MAE/Hit của đối chứng Geo-I không được chuyển sang phương pháp này.')
    sub('Lập luận utility và hiệu lực phản hồi')
    p('Một POI thuộc top-10 tĩnh vẫn thuộc top-10 sống nếu chính nó khả dụng: loại các POI không khả dụng không làm nó tụt hạng. Nếu hợp truy vấn phủ mọi POI có thể tới được, hợp phản hồi chứa đủ POI để phục hồi top-5 thật. Điều kiện gồm cùng khoảng cách đường, quy tắc phá hòa và phản hồi hợp lệ.')
    p('Trên mạng đang dùng, cấu hình 30 còn **160 POI**, cấu hình 67 còn **8 POI** ngoài hợp phủ tĩnh. Tọa độ ứng viên thuộc thành phần liên thông mạnh lớn nhất, còn tập đích xét toàn danh mục đường. Vì vậy, 100% trên các ca đã thử chưa phải chứng nhận toàn bản đồ.')
    p('Server giữ trạng thái trong epoch 60 giây. Hợp các phản hồi còn hiệu lực rồi xếp hạng cục bộ không làm giảm Recall; sang epoch mới phải lấy trạng thái mới. Vì kế hoạch cố định, có thể chỉ tải ở sự kiện đầu epoch. Quy tắc refresh dùng đồng hồ công khai, không dùng cache hit riêng tư; các đối chứng fixed K5/K12 được hưởng cùng tối ưu.')

    # Evaluation design and aggregate results.
    page(); sec('Kết quả: tập phát triển và kiểm tra sau khóa kế hoạch')
    p('Cùng mạng OSM 20/09: 102.404 trạng thái làn, 419 POI, sáu loại. Server trả tối đa 10 POI khả dụng cho mỗi loại–tọa độ; client chọn top-5 theo đường. Availability mô phỏng độc lập với GPS, giữ 60 giây. Mức chính p=0,8, kiểm tra thêm 0,5 và 0,95; mọi phương án cùng world trong một phép so.')
    table(['Bộ / bước','Quy mô dùng chấm năm scenario','Phạm vi'],[
        ['Tập phát triển','12 nhóm; 102 chuyến nguồn, 173 record; ba world seed, hai seed bảo vệ cho đối chứng Geo-I.','So hai cấu hình công khai, Geo-I + planner, fixed và bulk trên cùng tập.'],
        ['Nhóm kiểm tra bổ sung','Khóa cả hai kế hoạch trước khi sinh 4 nhóm 901–904; 88/88 chuyến đến nơi; dùng 32 chuyến, 56 record.','Ba world seed mới. Cùng thành phố/generator; Geo-I + planner chưa chạy trên nhóm này.']],[3.6,7.2,6.1])
    p('Recall gộp loại có đáp án → sự kiện → record → nhóm → ca, rồi trung bình đều 15 ca. Không có tham chiếu thì giữ null, không gán 1. Nhóm mới: S2 của family-904 không có POI tham chiếu; S9.B/S10.B chỉ có hai nhóm đủ đặc tả. Không coi seed hoặc các điểm FCD là người dùng độc lập.')
    sub('So sánh tại p=0,8')
    overview = []
    a = scores[method, 'epoch_cache']
    overview.append(['Geo-I + planner', pct(a['case_mean_recall']), str(a['case_gates_passed'])+'/15', 'Chưa chạy', 'Chưa chạy'])
    for m in (b30, b67, 'fixed_K5', 'fixed_K12', 'bulk_current'):
        d = totals.get(('expanded_development', m, .8))
        if d is None:
            o = scores[m, 'fresh' if m == 'bulk_current' else 'epoch_cache']
            d = {'case_mean_recall': o['case_mean_recall'], 'case_gates_passed': o['case_gates_passed']}
        n = totals['new_family_check', m, .8]
        overview.append([labels[m], pct(d['case_mean_recall']), str(d['case_gates_passed'])+'/15',
                         pct(n['case_mean_recall']), str(n['case_gates_passed'])+'/15'])
    table(['Phương án','Recall phát triển','Ca đạt','Recall nhóm mới','Ca đạt'], overview,[5.3,3.2,1.8,3.2,1.8])
    key('S1.C: cấu hình 30 đạt 95,57% so với Geo-I + planner 80,18% trên cùng tập phát triển; trên nhóm mới đạt 96,11%. 15/15 là ngưỡng utility theo ca, không phải 15/15 ca đã bảo vệ mọi kênh riêng tư.')
    p('Cả hai cấu hình được khóa trước nhóm kiểm tra bổ sung. Recall 100% là kết quả trên mẫu đã chấm; tám POI chưa phủ tĩnh vẫn nằm trong danh mục. Không suy thành bảo đảm toàn thành phố hoặc mọi trạng thái server.')

    # Full case table and paired uncertainty.
    page(); sub('Đủ 15 ca A/B/C ở p=0,8')
    table(['Ca','Geo-I: phát triển','30 query: phát triển','30 query: nhóm mới','Nhóm mới: có đáp án / có record¹'],[
        [c, pct(os[method, 'epoch_cache', c]['recall']), pct(ds[b30, c]['recall']), pct(nscores[b30, c]['recall']),
         str(sum(v is not None for v in nscores[b30, c]['family_recall'].values()))+'/'+str(nscores[b30, c]['family_count'])]
        for c in cases],[1.6,3.3,3.4,3.4,4.3])
    p('¹ Cột cuối đếm nhóm có tham chiếu / nhóm có record trên tập mới. S2 là 3/4, S9.B/S10.B là 2/2. Mỗi nhóm chỉ có một record cho từng ca này. Cấu hình 67 đạt 100% ở cả 15 dòng trên hai bộ. Cột nhóm mới kiểm tra riêng, không trừ trực tiếp khỏi cột Geo-I.')
    sub('Chênh lệch trên cùng 12 nhóm phát triển')
    contrasts = readout['expanded_comparisons']
    table(['Cấu hình 30 so với','Δ Recall trung bình (đpt)','Khoảng bootstrap 95%','Δ S1.C (đpt)'],[
        [('Geo-I + planner' if r['control'].startswith(method) else labels[r['control'].split('/')[0]]),
         num(100*r['case_mean_delta'], 2), ' đến '.join(num(100*v, 2) for v in r['family_bootstrap_ci95']),
         num(100*r['S1_C_delta'], 2)] for r in contrasts],[4.5,4.1,4.7,3.6])
    acontrast = next(r for r in contrasts if r['control'].startswith(method))
    p('So với Geo-I + planner: trung bình chỉ +0,07 điểm phần trăm và khoảng chứa 0, chưa có lợi thế ổn định về trung bình. Riêng S1.C tăng '+num(100*acontrast['S1_C_delta'], 2)+' điểm phần trăm, khoảng ['+'; '.join(num(100*v, 2) for v in acontrast['S1_C_family_bootstrap_ci95'])+']. S2 và một số ca S10 giảm utility; lợi ích không đồng đều.')
    p('Bootstrap ghép cặp 10.000 lần theo nhóm, gộp world/seed trong nhóm; mang tính thăm dò, chưa hiệu chỉnh nhiều phép so sánh. Chênh lệch phản ánh cả cách chọn truy vấn và giao diện chi phí; không quy toàn bộ cho một thành phần đơn lẻ.')

    # Retain the failing stress and exact accounting together.
    page(); sub('Độ nhạy và chi phí trên nhóm mới')
    table(['Phương án','Khả dụng 50%','Khả dụng 80%','Khả dụng 95%'],[
        [labels[m]]+[pct(totals['new_family_check', m, q]['case_mean_recall'])+'; '+str(totals['new_family_check', m, q]['case_gates_passed'])+'/15'
                    for q in (.5, .8, .95)] for m in (b30, b67, 'fixed_K5', 'fixed_K12', 'bulk_current')],[5.2,3.9,3.9,3.9])
    p('Mỗi ô: Recall trung bình và số ca ≥90%, gộp ba world mới. **Cấu hình 30 chỉ đạt 11/15 ở p=0,95:** thiếu S2.A/B/C và S10.B. Trên tập phát triển, cấu hình 30 qua 15/15 ở cả ba p; sự khác biệt cho thấy cần kiểm tra nhóm mới. Giữ cả hai cấu hình đã khóa, không chọn lại plan từ các ca thất bại.')
    sub('Đếm truy vấn, tọa độ và byte riêng')
    costrows = []
    for m in (b30, b67, 'fixed_K5', 'fixed_K12', 'bulk_current'):
        f, e = costs[m, 'every_event'], costs[m, 'epoch_refresh']
        costrows.append([labels[m], str(round(f['category_queries_per_event'])),
            str(round(f['distinct_coordinates_per_event'])),
            num(f['native_request_bytes_per_event']+f['native_response_bytes_per_event']),
            num(e['native_request_bytes_per_event']+e['native_response_bytes_per_event'])])
    table(['Phương án','Query loại / lần tải','Tọa độ / lần tải','Byte / event: tải mỗi lần','Byte / event: refresh epoch'], costrows,[4.5,2.7,2.7,3.5,3.5])
    p('Byte gồm thân request + response theo schema từng phương án ở p=0,8; chưa gồm HTTP/TLS, RTT hoặc chi phí server. Tọa độ dùng chung được serialize gọn hơn. Cột cuối đếm đúng lần tải ở event đầu mỗi epoch; các kế hoạch công khai và fixed K5/K12 cùng hưởng tối ưu, trung bình 0,289 refresh/event. Đây không phải số kết nối HTTP.')
    p('Cấu hình 30 có cùng 30 truy vấn loại với Geo-I/fixed K5 nhưng dùng 19 thay vì 5 tọa độ và nhiều byte hơn. Cấu hình 67 không rẻ hơn fixed K12 chỉ vì 67 < 72. Nếu cho tải bulk, bitmap 419 trạng thái + epoch vẫn đạt 100% với chi phí thấp hơn; truy vấn theo điểm không thắng bulk trong workload này.')
    p('Chưa có latency đầu-cuối hoặc phép đo điện thoại. Vì giao diện và chi phí khác nhau, chưa tính Q thực nghiệm mới hoặc gắn nhãn một cấu hình thắng cả privacy, utility và performance.')

    page(); sec('Đóng góp và phạm vi của kết quả')
    p('Giá trị cần đánh giá là thiết kế dịch vụ: phân bổ truy vấn theo loại POI công khai, tách GPS khỏi network policy và dùng phản hồi có hiệu lực để xếp hạng cục bộ. Geo-I + dummy đã có tiền lệ [R12]; tham lam, set cover và cache cũng không được nhận là nguyên lý mới.')
    table(['Luận điểm','Bằng chứng','Phạm vi kết luận'],[
        ['Giữ dịch vụ ở ngân sách truy vấn loại hạn chế','Cấu hình 30 đạt 95,67% trên tập phát triển và 94,61% trên nhóm mới; đều 15/15 ca tại p=0,8.','S1.C tốt hơn Geo-I + planner; trung bình chênh lệch nhỏ và khoảng bootstrap chứa 0. Nhiều tọa độ/byte hơn K5.'],
        ['Tách kênh tọa độ khỏi GPS','Kế hoạch công khai; transcript bất biến với GPS khi giữ vùng, clock và trạng thái server.','Áp dụng cho mục tiêu tọa độ của năm scenario, chưa bao gồm timing/identity hoặc chọn vùng theo GPS.'],
        ['Có cấu hình ưu tiên độ phủ','Cấu hình 67 đạt 100% trên các ca/mức availability đã chấm, kể cả nhóm sau khóa kế hoạch.','Tốn 67 query loại và 52 tọa độ; tám POI chưa phủ tĩnh, chưa bảo đảm toàn bản đồ.'],
        ['Có kiểm tra ngoài tập phát triển ban đầu','Bốn nhóm, 88/88 chuyến hoàn tất; dùng 56 record từ 32 chuyến, ba world mới.','Cùng thành phố và generator; S2 có 3 nhóm có tham chiếu, S9.B/S10.B có 2 nhóm. Chưa xác nhận tổng quát hóa rộng.']],[4.0,6.6,6.3])
    sub('Điều kiện sử dụng và lựa chọn cấu hình')
    bullets(['**API theo điểm, hỏi riêng từng loại:** chọn cấu hình 30 hoặc 67 theo ngân sách byte và mức độ phủ cần thiết. Kế hoạch cố định cho vùng đã công khai; không thích nghi từ GPS hoặc cache hit riêng tư.',
        '**Phản hồi có phiên bản/expiry:** chỉ dùng dữ liệu còn hiệu lực và refresh theo lịch công khai. Server đổi trạng thái liên tục cần cơ chế xác nhận tương ứng.',
        '**Ràng buộc đúng năm tọa độ di động:** giao diện phương pháp này khác; không gắn nhãn K=5 hoặc lấy kết quả thay cho benchmark đó.',
        '**Có API bulk:** so với bulk trước khi chọn; nó vẫn tốt hơn về utility và byte trong workload 419 POI hiện tại.'])
    key('Có bằng chứng duy trì utility và lập luận hạn chế thông tin tọa độ cho năm scenario trong các điều kiện đã nêu. Chưa chứng minh bảo vệ mọi metadata, vượt sáu phương pháp từ paper hoặc sẵn sàng triển khai thực.')
    p('Để mở rộng kết luận: chạy các adapter paper cùng dịch vụ/chi phí và đối thủ phù hợp; kiểm tra timing, vùng và click; đo latency trên thiết bị; đánh giá thành phố hoặc quần thể độc lập sau khi khóa yêu cầu. Giữ các ca thất bại ở stress 95% thay vì chỉ báo mức danh định.')
    p('Bằng chứng có thể kiểm tra lại: hai kế hoạch dựng từ dữ liệu công khai, 8.532 đối chiếu Dijkstra xuôi, 315 dòng tổng hợp và 14 kiểm tra luồng dữ liệu. Nguồn, cấu hình và hash nằm trong method_evidence.json; báo cáo không coi việc vượt kiểm tra code là bằng chứng hiệu quả trước mọi đối thủ.')
