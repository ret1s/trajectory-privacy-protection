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
    cases = list(ns['PRIORITY_CASES'])
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
        'privacy_evidence': 'conditional GPS-independent transcript argument, information-flow checks and common finite coordinate/relative-time attack bank; not identity protection',
        'same_K5_interface': False, 'same_bytes': False,
        'expanded_source_sessions': dev['physical_source_sessions'], 'expanded_records': dev['target_records'],
        'new_family_count': new['family_count'], 'new_completed_trips': new['new_SUMO_completed_sessions'],
        'new_source_sessions': new['source_sessions'], 'new_records': new['target_records'],
        'new_family_check_after_plan_freeze': True, 'independent_city_confirmation': False,
        'six_paper_superiority_established': False, 'global_full_cover_certificate': False,
        'historical_iteration30_new_family_30_query_p95_gates': totals['new_family_check', b30, .95]['case_gates_passed']}
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

    page(); sub('Lập luận privacy cho S1/S2/S3/S9 và S10.A/C')
    p('Gọi X là hành trình, A là giờ sử dụng riêng tư; C gồm vùng, danh mục, kế hoạch và khoảng đăng ký công khai; W là trạng thái server; T là transcript yêu cầu/phản hồi. Client gửi mỗi 60 giây trong trọn khoảng [0, 3600), kể cả trước/sau chuyến hoặc không dùng dịch vụ; đọc cache không kích hoạt bản tin. Với C được chốt trước hoạt động:')
    eq(r'T=f(C,W)\quad\Longrightarrow\quad I((X,A);T\mid C,W)=0.',
       'Với vùng, kế hoạch, khoảng đăng ký và trạng thái server cố định, thay hành trình hoặc giờ đọc cache không đổi transcript.')
    table(['Mục tiêu cần bảo vệ','Vì sao tọa độ công bố không bổ sung thông tin'],[
        ['S1: vị trí hiện tại','Ở vị trí nào trong cùng vùng, client cũng gửi cùng kế hoạch.'],
        ['S2: nơi dừng','Tọa độ không tập trung quanh nơi dừng; lịch vẫn chạy khi không có hoạt động.'],
        ['S3: đoạn đường đã đi','Chuỗi tọa độ cố định không bám đường đi thật.'],
        ['S9; S10.A/C: đầu/cuối','Tọa độ không chọn từ endpoint; giờ bắt đầu/kết thúc chuyến không điều khiển lịch gửi trong khoảng đăng ký.']],[4.5,12.4])
    key('Bảo vệ payload và giờ gửi trong khoảng công khai; prior vẫn có thể giúp đối thủ đoán. Không gán Hit=0 hoặc MAE vô hạn từ công thức trên.')
    p('Điều kiện: đăng ký khoảng cố định trước chuyến, không hủy sớm hay đổi vùng/kế hoạch theo GPS. Account, IP, click và lỗi mạng nằm ngoài bảo đảm. Sáu kiểm tra trên server mô phỏng đối chiếu lịch đọc sớm/muộn/không đọc cho cùng yêu cầu và phản hồi. Đây là kiểm tra cơ chế, không chứng minh mọi kênh mạng đã an toàn.')
    sub('Lập luận utility và hiệu lực phản hồi')
    p('Một POI thuộc top-10 tĩnh vẫn thuộc top-10 sống nếu chính nó khả dụng: loại các POI không khả dụng không làm nó tụt hạng. Nếu hợp truy vấn phủ mọi POI có thể tới được, hợp phản hồi chứa đủ POI để phục hồi top-5 thật. Điều kiện gồm cùng khoảng cách đường, quy tắc phá hòa và phản hồi hợp lệ.')
    p('Trên mạng đang dùng, cấu hình 30 còn **160 POI**, cấu hình 67 còn **8 POI** ngoài hợp phủ tĩnh. Tọa độ ứng viên thuộc thành phần liên thông mạnh lớn nhất, còn tập đích xét toàn danh mục đường. Vì vậy, 100% trên các ca đã thử chưa phải chứng nhận toàn bản đồ.')
    p('Server giữ trạng thái trong epoch 60 giây; client chỉ dùng phản hồi còn hiệu lực. Bảng mục 9 là kiểm tra trên bốn nhóm, gửi mỗi sự kiện. Mục 10 dùng 32 nhóm mới và client theo lịch công khai; tính cả bản tin ngoài thời gian hoạt động. Ablation so với cùng kế hoạch chỉ refresh trong epoch có hoạt động để đo đúng giá của việc che giờ bắt đầu/kết thúc.')

    from paper_comparison_content import add_comparison_content
    add_comparison_content(ns)
