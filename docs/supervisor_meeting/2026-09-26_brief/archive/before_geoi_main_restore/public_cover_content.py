"""Quantitative description and presentation names for frozen public plans."""
from collections import Counter
import hashlib
import json

PUBLIC_LABELS = {'ours30': 'CoverLite-30', 'ours67': 'CoverPlus-67',
                 'calendar30': 'CoverLite-30 + lịch', 'calendar67': 'CoverPlus-67 + lịch'}
PLAN_PATH = 'artifacts/benchmarks/research_loop/iteration29_plans.json'
CALENDAR_PATH = 'artifacts/benchmarks/endpoint_calendar_v1/protocol.json'


def build_public_cover(ns):
    p, key, sub, table, graphic = (ns[k] for k in ('p', 'key', 'sub', 'table', 'graphic'))
    root, out, scope = ns['ROOT'], ns['OUT'], ns['SCOPE']
    source = json.loads((root/PLAN_PATH).read_text())
    calendar = json.loads((root/CALENDAR_PATH).read_text())['calendar']
    plans = [source['plans'][name] for name in ('public_category_budget30', 'public_category_full_cover')]
    a = {r['method']: r for r in scope['endpoint']['aggregates']}
    methods = ['calendar30', 'calendar67']
    ticks = int((calendar['end_s']-calendar['start_s'])/calendar['epoch_s'])
    total = sum(plans[0]['target_counts'])
    assert total == sum(plans[1]['target_counts']) == 419
    assert [q['category_queries'] for q in plans] == [30, 67]
    assert plans[0]['queries'] == plans[1]['queries'][:30]
    assert [q['distinct_coordinates'] for q in plans] == [19, 52]
    assert calendar['epoch_s'] == 60 and ticks == 60
    covered = [total-sum(map(len, q['uncovered_ids'])) for q in plans]
    percent = lambda v: f'{100*v:.2f}%'.replace('.', ',')
    number = lambda v: f'{v:,.1f}'.replace(',', 'X').replace('.', ',').replace('X', '.')
    rows = [
        ['Cách chọn kế hoạch', 'Greedy, dừng ở ngân sách 30', 'Greedy đến khi không tăng độ phủ; được 67'],
        ['Truy vấn logic / lần gửi', '30 cặp loại POI–tọa độ', '67 cặp loại POI–tọa độ'],
        ['Tọa độ khác nhau', str(plans[0]['distinct_coordinates']), str(plans[1]['distinct_coordinates'])],
        ['POI tĩnh nằm trong hợp phản hồi', f'{covered[0]}/{total}; thiếu {total-covered[0]}', f'{covered[1]}/{total}; thiếu {total-covered[1]}'],
        ['Lượt POI trả về tối đa / lần', '≤300 (30 × L=10)', '≤670 (67 × L=10)'],
        ['Lịch endpoint: 60 lần gửi / giờ', '1.800 truy vấn logic / giờ', '4.020 truy vấn logic / giờ'],
        ['Recall@5 tại p=0,8; ca đạt ≥90%', *[percent(a[m]['recall_0.8'])+'; '+str(a[m]['gates_0.8'])+'/14' for m in methods]],
        ['Recall@5 tại p=0,95; ca đạt ≥90%', *[percent(a[m]['recall_0.95'])+'; '+str(a[m]['gates_0.95'])+'/14' for m in methods]],
        ['Hit100 S9 / S10; lịch, 32 nhóm', *[percent(a[m]['S9']['hit100'])+' / '+percent(a[m]['S10']['hit100']) for m in methods]],
        ['Byte gửi + nhận / sự kiện dịch vụ', *[number(a[m]['request_bytes_per_service_event']+a[m]['response_bytes_per_service_event']) for m in methods]],
    ]
    sub('Nhánh PublicCover: CoverLite-30 và CoverPlus-67')
    p('**Hai cấu hình của cùng nhánh bảo vệ dịch vụ:** CoverLite-30 giảm số truy vấn; CoverPlus-67 tăng độ phủ POI. Kế hoạch được lập từ bản đồ/danh mục công khai trước khi đọc chuyến đánh giá. Nhánh này không dùng neo Geo-I; số 30/67 đếm cặp loại POI–tọa độ, không phải K dummy hay ngân sách ε. Greedy chọn từng cặp có tỷ lệ POI mới lớn nhất trong loại; 67 là số thu được trên danh mục hiện tại.')
    graphic(r'\includegraphics[width=\linewidth]{figures/architecture_publiccover.pdf}',
            '<img src="figures/architecture_publiccover.svg" alt="PublicCover: đầu vào công khai, kế hoạch phủ POI cố định, lịch gửi và cache, xếp hạng cục bộ bằng GPS; đầu ra công khai và kết quả riêng tại thiết bị">',
            'Khung xanh là PublicCover tại thiết bị. Layer 1 lập kế hoạch cố định; layer 2 gửi theo lịch và lưu phản hồi; layer 3 dùng GPS để chọn top-5 cục bộ. GPS và lúc dùng dịch vụ không điều khiển kế hoạch hoặc lịch đã đăng ký.')
    table(['Thông số / kết quả', 'CoverLite-30', 'CoverPlus-67'], rows, [5.7, 5.6, 5.6])
    p('**Đọc số liệu:** độ phủ tĩnh chấm 419 POI mục tiêu; Recall chấm top-5 của mẫu chuyến. Lượt phản hồi có thể trùng POI. Cả hai hỏi đủ sáu loại: cafe, clinic, fuel, hospital, pharmacy, restaurant; máy chủ trả tối đa L=10 mỗi cặp, thiết bị chọn k=5. Các hàng Recall/Hit/byte dùng kết quả 32 nhóm với lịch công khai, p là tỷ lệ POI khả dụng; byte gồm traffic ngoài chuyến, chưa gồm HTTP/TLS. Lịch là [0, 3.600) giây, tick mỗi 60 giây; số truy vấn/giờ là phép đếm từ cấu hình, không phải số gói HTTP.')
    key('Lite và Plus chủ yếu là đánh đổi độ phủ dịch vụ–chi phí. Hai cấu hình cùng có Hit100=0 trong phép thử S9/S10; chưa chứng minh Plus riêng tư hơn Lite. Che giờ chuyến đến từ lịch cố định; bảo vệ có điều kiện vùng/lịch đăng ký trước, không bao gồm IP, account hoặc click. Plus vẫn thiếu 8 POI tĩnh dù Recall trên mẫu đạt 100%.')
    evidence = {'family': 'PublicCover', 'names_are_presentation_aliases': True,
                'labels': PUBLIC_LABELS, 'rows': rows, 'calendar': calendar,
                'static_target_count': total, 'covered_counts': covered,
                'queries_by_category': [dict(Counter(q['category'] for q in plan['queries'])) for plan in plans],
                'benchmark_scope': '32 groups, calendar; frozen active_scope_ac_v2 readout',
                'not_GeoI_or_dummy_K': True, 'new_experiment': False,
                'sources': {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
                            (PLAN_PATH, CALENDAR_PATH, 'artifacts/benchmarks/active_scope_ac_v2/readout.json')}}
    (out/'public_cover_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n')
