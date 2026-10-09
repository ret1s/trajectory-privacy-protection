"""Compact scientific tables from retained, source-pinned benchmark evidence.

This module only formats supplied readouts. It never changes a mechanism,
selects an attacker, evaluates a trajectory, or pools experimental cohorts.
Benchmark-table rates are fractions; slide-content sensitivity rates are
already percentages. Byte costs use decimal MB and application JSON only.
"""
from focus_visuals import table


def _n(value, digits=2):
    return 'N/A' if value is None else f'{value:.{digits}f}'.replace('.', ',')


def _p(value, digits=2):
    return 'N/A' if value is None else _n(100 * value, digits)


def _th(h):
    return {key: h[key] for key in ('text', 'line', 'ink', 'gray', 'teal')}


def _note(h, y, value, *, x=46, color=None, bold=False, size=20):
    return h['text'](x, y, value, size, color=color or h['gray'],
                     weight=700 if bold else 400, lh=1.18)


def location_results(data, **h):
    """Retain all five methods and three metrics for each historical scenario."""
    methods = ('br_private', 'dls_graph_adaptation',
               'semantic_correlation_local_adaptation',
               'transprotect_adaptation', 'unprotected')
    indexed = {(r['scenario'], r['method']): r
               for r in data['S1_S3_historical']['rows']}
    rows = []
    for scenario in ('S1', 'S2', 'S3'):
        for label, field, fmt in (('Hit100 (%) ↓', 'hit100', _p),
                                  ('MAE (m) ↑', 'mae_m', _n),
                                  ('Recall@5 (%) ↑', 'recall5', _p)):
            rows.append([f'{scenario} · {label}'] +
                        [fmt(indexed[scenario, method][field]) for method in methods])
    b = _note(h, 137, 'K = 5; phương pháp đối chứng là các bản triển khai thích nghi.',
              color=h['ink'], bold=True, size=22)
    t, _ = table(['Thước đo', 'Đề xuất\n(bản trước)', 'DLS\n(thích nghi)',
                  'Semantic\n(thích nghi)', 'TransProtect\n(thích nghi)', 'GPS chưa\nbảo vệ'],
                 rows, [250, 187, 181, 196, 204, 170], x=46, y=184,
                 row_h=32, size=20, header_size=20, **_th(h))
    b += t
    b += _note(h, 553, 'Hit thấp / MAE cao: khó suy vị trí hơn; Recall cao: trả lời POI tốt hơn.',
               color=h['ink'], bold=True)
    b += _note(h, 586, 'Mỗi hàng gộp 12 ô (mẫu, lượt nhiễu); Hit và MAE có thể chọn bộ suy luận khác nhau.')
    b += _note(h, 619, 'Không phải kết quả Epoch8/L30 hiện tại hoặc tái lập nguyên bản toàn bộ paper.',
               color=h['orange'], bold=True)
    return b


def linkage_companion_results(data, **h):
    """Separate linkage and companion cohorts; retain both ordinary and BA scores."""
    text = h['text']
    indexed = {(r['method'], r['target']): r
               for r in data['S4_historical_linkage']['rows']}
    left, colors = [], []
    for target, name in (('same_person', 'người'), ('same_vehicle', 'xe')):
        for method, label in (('raw', 'GPS thật'),
                              ('geoi_slack_reconstructed', 'Geo-I')):
            r = indexed[method, target]
            left.append([f'{label} · {name}', _p(r['accuracy']),
                         _p(r['balanced_accuracy']), _n(r['roc_auc'], 3)])
            colors.append(h['gray'] if method == 'raw' else h['teal'])
    b = text(46, 137, 'S4 · Liên kết người / phương tiện', 23,
             color=h['teal'], weight=700)
    t, _ = table(['Quan sát / đích', 'Đúng ↓\n(%)', 'BA ↓\n(%)', 'AUC'],
                 left, [253, 105, 105, 105], x=46, y=185,
                 row_h=38, size=20, header_size=20, colors=colors, **_th(h))
    b += t
    views = {'target_only': 'Chỉ mục tiêu',
             'joint_protected_partner': 'Đồng hành bảo vệ',
             'joint_public_raw_partner': 'Đồng hành GPS thật',
             'joint_unrelated_protected_partner': 'Không liên quan',
             'raw_target_positive_control': 'GPS mục tiêu thật'}
    right = [[views[r['view']], _n(r['mae_m'], 1), _p(r['hit100'])]
             for r in data['S8_historical_companion']['rows']]
    b += text(666, 137, 'S8 · Suy vị trí từ người đồng hành', 23,
              color=h['teal'], weight=700)
    t, _ = table(['Nguồn quan sát', 'MAE ↑\n(m)', 'Hit100 ↓\n(%)'],
                 right, [301, 137, 130], x=666, y=185,
                 row_h=38, size=20, header_size=20, **_th(h))
    b += t
    max_auc = max(indexed['geoi_slack_reconstructed', 'same_person']['family_auc'].values())
    b += _note(h, 445, ['3 nhóm; 45 cặp mỗi tác vụ.',
                        'BA cân bằng hai lớp; BA < 50% không',
                        'tự chứng minh bảo vệ mạnh hơn.'])
    b += _note(h, 531, 'AUC người ở một nhóm đã bảo vệ: ' + _n(max_auc, 3) + '.',
               color=h['orange'], bold=True)
    b += _note(h, 564, 'Bộ suy luận: kNN / cây trên hình dạng Q.')
    b += _note(h, 478, ['3 nhóm; 24 mốc mục tiêu duy nhất;',
                        '41 quan sát cặp, trung bình theo nhóm.',
                        'GPS đồng hành thật làm giảm MAE.'], x=666)
    b += _note(h, 564, ['Bộ suy luận có thể bỏ qua người đồng hành;',
                        'các hàng bằng nhau không chứng minh riêng tư nhóm.'], x=666)
    b += _note(h, 629, 'Hai phép thử lịch sử riêng; chưa xác nhận S4/S8 cho Epoch8/L30; chưa ẩn tài khoản/IP.',
               color=h['orange'], bold=True)
    return b


def future_results(data, **h):
    """Retain the Planar negative comparison and the finite binary threat scope."""
    section = data['S5_S6_matched_native_pilot']
    indexed = {r['method']: r for r in section['rows']}
    methods = (('raw', 'GPS chưa bảo vệ'), ('rem_epoch8', 'Geo-I / REM, L20'),
               ('planar_epoch8', 'Planar, L20'))
    rows, colors = [], []
    for method, label in methods:
        r = indexed[method]
        c = r['cost']
        rows.append([label, _p(r['current_recall5']),
                     _p(r['S5_exact_candidate_edge_accuracy']),
                     _p(r['S6_destination_hit100']),
                     _n(r['S6_destination_mae_m'], 1),
                     _n((c['request_bytes_total'] + c['reply_bytes_total']) / 1e6)])
        colors.append(h['teal'] if method == 'rem_epoch8' else h['ink'])
    b = _note(h, 137, '6 nhóm kiểm tra, 12 phiên truy vấn; REM / Planar cùng K = 5 và cap epoch 0,23/m.',
              color=h['ink'], bold=True, size=22)
    t, _ = table(['Phương pháp', 'Recall@5 ↑\n(%)', 'S5 đúng cạnh ↓\n(%)',
                  'S6 Hit100 ↓\n(%)', 'S6 MAE ↑\n(m)', 'JSON tổng\n(MB)'],
                 rows, [284, 177, 203, 180, 161, 183], x=46, y=185,
                 row_h=43, size=21, header_size=20, colors=colors, **_th(h))
    b += t
    b += _note(h, 425, 'Planar có Recall cao hơn REM trong phép thử này; suy luận đúng cùng 50%.',
               color=h['orange'], bold=True, size=22)
    b += _note(h, 466, 'S5 và S6 cùng quyết định nhị phân giữa hai cạnh / đích công khai; không phải hai tác vụ độc lập.')
    b += _note(h, 504, 'Bộ suy luận đã chọn: GPS thật → Candidate Trees; REM → Motion; Planar → Curve Mean.')
    b += _note(h, 542, 'Recall gần nhất theo loại POI; 1.402 / 1.510 mốc có tham chiếu. GPS thật gửi một tọa độ.')
    b += _note(h, 580, 'Chi phí JSON là byte ứng dụng; chưa đo độ trễ mạng. Bộ nhớ đệm tĩnh là phép thử khác.')
    b += _note(h, 620, 'Pilot đã khảo sát, cấu hình L20 riêng; không phải xác nhận bốn mục đích của L30 hiện tại.',
               color=h['gray'], bold=True)
    return b


def delay_results(audit, **h):
    """All five four-family historical arms, including lost releases and delay."""
    study = audit['historical_matched_study']
    indexed = {r['method']: r for r in study['rows']}
    methods = (('raw', 'GPS thật'), ('scale100_L10', 'Geo-I L10'),
               ('scale100_L20', 'Geo-I L20'), ('delay60_L10', 'Delay 60 s, L10'),
               ('scale025_L20', 'Endpoint20, L20'))
    rows, colors = [], []
    for method, label in methods:
        r = indexed[method]
        rows.append([label, _n(r['S9']['mae_m'], 1), _n(r['S10']['mae_m'], 1),
                     _p(r['recall']), f"{r['released_events']}/{r['input_events']}",
                     _n(r['mean_publication_delay_s'], 0), _n(r['bytes_per_input'], 1)])
        colors.append(h['orange'] if method == 'delay60_L10' else
                      h['teal'] if method == 'scale025_L20' else h['ink'])
    b = _note(h, 137, '4 nhóm; 16 lượt chạy mỗi phương pháp; 170 mốc đầu vào.',
              color=h['ink'], bold=True, size=22)
    t, _ = table(['Phương pháp', 'S9 MAE ↑\n(m)', 'S10 MAE ↑\n(m)',
                  'Recall@5 ↑\n(%)', 'Mốc gửi /\nđầu vào', 'Delay\n(s)', 'JSON byte\n/mốc'],
                 rows, [248, 161, 175, 174, 152, 116, 162], x=46, y=185,
                 row_h=40, size=21, header_size=20, colors=colors, **_th(h))
    b += t
    b += _note(h, 487, 'Delay làm mất câu trả lời tức thời: chỉ gửi 122 / 170 mốc; Recall còn 78,08%.',
               color=h['orange'], bold=True, size=22)
    b += _note(h, 525, 'Hit100 S9 / S10: GPS thật 100%; bốn cấu hình bảo vệ đều 0% trước bank hữu hạn đã chọn.')
    b += _note(h, 563, 'Delay L10: cap 0,23/m. Endpoint20 L20: cap 0,0575/m; khác cả ngân sách và độ sâu phản hồi.')
    b += _note(h, 601, 'Không phải so sánh chỉ đổi delay; cohort này riêng với phép thử 28 nhóm tiếp theo.',
               color=h['gray'], bold=True)
    b += _note(h, 636, 'Gửi ngay không đồng nghĩa độ trễ mạng bằng 0; chưa có đo độ trễ mạng thực tế.')
    return b


def endpoint_results(data, **h):
    """Eight full-bank rows; preserve service/cost tradeoff and exploratory CIs."""
    section = data['S9_S10_full_bank_endpoint']
    indexed = {(r['scenario'], r['method']): r for r in section['rows']}
    methods = (('raw', 'GPS thật'), ('scale100_L10', 'Geo-I L10'),
               ('scale100_L20', 'Geo-I L20'), ('scale025_L20', 'Endpoint20 L20'))
    rows, colors = [], []
    for scenario in ('S9', 'S10'):
        for method, label in methods:
            r = indexed[scenario, method]
            rows.append([scenario, label, _n(r['mae_m'], 1), _p(r['hit100']),
                         _p(r['hit500']), _p(r['recall5_same_frozen_service']),
                         _n(r['family_mean_total_JSON_bytes_per_input_event'], 1)])
            colors.append(h['teal'] if method == 'scale025_L20' else h['ink'])
    b = _note(h, 137, '28 nhóm; 112 quan sát mỗi tác vụ; cấu hình lịch sử riêng, không phải Epoch8/L30.',
              color=h['ink'], bold=True, size=22)
    t, _ = table(['Tác vụ', 'Phương pháp', 'MAE ↑\n(m)', 'Hit100 ↓\n(%)',
                  'Hit500 ↓\n(%)', 'Recall cả chuyến ↑\n(%)', 'JSON byte\n/mốc'],
                 rows, [95, 255, 157, 143, 143, 213, 182], x=46, y=184,
                 row_h=32, size=20, header_size=20, colors=colors, **_th(h))
    b += t
    contrasts = {(r['scenario'], r['metric']): r
                 for r in section['Endpoint20_minus_plain_L20_uncertainty']}
    mae = contrasts['S10', 'mae_m']
    hit = contrasts['S10', 'hit500']
    service = section['utility_cost_delta']['scale025_L20-scale100_L20/recall']
    b += _note(h, 530, 'Endpoint20 − Geo-I L20: ΔMAE S10 +' + _n(mae['delta'], 1)
               + ' m; CI 95% [' + _n(mae['CI95'][0], 1) + '; '
               + _n(mae['CI95'][1], 1) + '] (thăm dò).', color=h['teal'], bold=True)
    b += _note(h, 562, 'S10 Hit100 cùng 0%; ΔHit500 ' + _p(hit['delta'])
               + ' điểm %; CI [' + _p(hit['CI95'][0]) + '; ' + _p(hit['CI95'][1])
               + '] chạm 0.', color=h['orange'], bold=True)
    b += _note(h, 594, 'Recall cả chuyến giảm ' + _p(-service['delta'])
               + ' điểm %; CI ΔRecall [' + _p(service['bootstrap95_low']) + '; '
               + _p(service['bootstrap95_high']) + ']. Tất cả gửi ngay; thời điểm mở / đóng còn lộ.')
    b += _note(h, 626, 'Bank hữu hạn, chọn riêng theo thước đo. Endpoint20: u = 0,0025/m, bằng 25% đối chứng L20 lịch sử.')
    return b


def utility_results(data, **h):
    """Current fresh four-purpose utility, same Q, plus measured traffic cost."""
    section = data['current_fresh_four_purpose_utility']
    rows, colors = [], []
    for r in section['rows']:
        lo, hi = r['paired95_family_bootstrap_gain_pp']
        rows.append(['Trung bình 4 mục đích' if r['primary'] else r['label_vi'],
                     _p(r['L20_recall']), _p(r['L30_recall']), '+' + _n(r['gain_pp']),
                     '[' + _n(lo) + '; ' + _n(hi) + ']'])
        colors.append(h['teal'] if r['primary'] else h['ink'])
    b = _note(h, 137, '24 nhóm mới × 3 lượt nhiễu × 8 chuyến; cùng K = 5, Z, Q, lịch gửi và ngân sách.',
              color=h['ink'], bold=True, size=22)
    t, _ = table(['Mục đích', 'L20 Recall ↑\n(%)', 'L30 Recall ↑\n(%)',
                  'Chênh lệch\n(điểm %)', 'CI 95% chênh lệch\n(điểm %)'],
                 rows, [355, 197, 197, 177, 262], x=46, y=185,
                 row_h=36, size=21, header_size=20, colors=colors, **_th(h))
    b += t
    cost = section['cost']
    primary = next(r for r in section['rows'] if r['primary'])
    draw_gains = ' / '.join('+' + _p(primary['within_draw_gains'][str(i)]) for i in (1, 2, 3))
    b += _note(h, 466, 'Ba lượt nhiễu: ' + draw_gains + ' điểm %; macro là chỉ tiêu chính, CI từng mục đích là phụ.',
               color=h['teal'], bold=True)
    request_count = f"{cost['L30']['requests']:,}".replace(',', '.')
    b += _note(h, 504, request_count + ' truy vấn; request JSON giữ ' + _n(cost['L30']['request_bytes']/1e6)
               + ' MB. Phản hồi: ' + _n(cost['L20']['reply_bytes']/1e6) + ' → '
               + _n(cost['L30']['reply_bytes']/1e6) + ' MB (+' + _n(cost['reply_growth_percent']) + '%).',
               color=h['orange'], bold=True)
    radius = next(r for r in section['rows'] if r['purpose'] == 'within_radius')['coverage']
    b += _note(h, 542, 'Bán kính: ' + f"{radius['defined_windows']:,}/{radius['total_windows']:,}".replace(',', '.')
               + ' mốc có tham chiếu; tham chiếu rỗng là N/A, có tham chiếu nhưng không trả lời nhận 0.')
    b += _note(h, 580, 'L là POI tối đa mỗi loại / mỗi Q; local top-k = 5. L30 trả nhiều byte hơn, không đổi cơ chế Geo-I.')
    b += _note(h, 620, 'Mô phỏng cùng bản đồ, POI tĩnh; GPS từng mốc và đích biết tại thiết bị. Chưa so ưu thế ở cùng chi phí.')
    return b


def sensitivity_results(content, **h):
    """Two compact diagnostic tables; content rates are already percentages."""
    sensor, dynamic = content['sensor'], content['dynamic']
    text = h['text']
    b = text(46, 137, 'GPS cục bộ: lấy mẫu 60 s', 23, color=h['teal'], weight=700)
    rows = [[str(sigma), _n(sensor['hold20'][i]), _n(sensor['hold30'][i]),
             _n(sensor['velocity30'][i])] for i, sigma in enumerate((0, 5, 15))]
    t, _ = table(['σ (m)', 'Giữ GPS\nL20 (%)', 'Giữ GPS\nL30 (%)', 'Ngoại suy\nL30 (%)'],
                 rows, [91, 151, 151, 175], x=46, y=185, row_h=39,
                 size=21, header_size=20, **_th(h))
    b += t
    b += _note(h, 396, 'GPS chính xác từng mốc, L30: ' + _n(sensor['oracle30']) + '%.',
               color=h['ink'], bold=True)
    b += _note(h, 434, ['Ngoại suy hai fix giảm sai số vị trí,',
                        'nhưng Recall thấp hơn giữ fix ở cả ba σ.'],
               color=h['orange'], bold=True)
    b += _note(h, 495, ['Nhiễu Gaussian giả lập; không phải đo GPS thực.',
                        'Đây là clock xếp hạng local riêng;',
                        'không đổi GPS đã sinh Geo-I, Q hay ngân sách.'])
    b += _note(h, 583, ['Đích riêng được giả định đã biết; ngoại suy',
                        'chỉ dùng hai fix quá khứ, chặn tốc độ 8 m/s.'])
    b += text(666, 137, 'POI có trạng thái thay đổi', 23, color=h['teal'], weight=700)
    rows = [['L20 · hiện tại', _n(dynamic['l20_current'])],
            ['L20 · có cache', _n(dynamic['l20_cached'])],
            ['L30 · hiện tại', _n(dynamic['l30_current'])],
            ['L30 · có cache', _n(dynamic['l30_cached'])],
            ['Bulk · toàn danh mục', _n(dynamic['bulk'])]]
    t, _ = table(['Dịch vụ', 'Recall@5 ↑ (%)'], rows, [357, 211],
                 x=666, y=185, row_h=34, size=21, header_size=20, **_th(h))
    b += t
    b += _note(h, 428, 'JSON tổng: L30 ' + _n(dynamic['l30_total_JSON_bytes']/1e6)
               + ' MB;', x=666, color=h['ink'], bold=True)
    b += _note(h, 460, 'bulk ' + _n(dynamic['bulk_total_JSON_bytes']/1e6) + ' MB.',
               x=666, color=h['ink'], bold=True)
    b += _note(h, 500, ['Bulk mạnh hơn khi API cho phép lấy cả danh mục;',
                        'kết quả không chứng minh luôn cần truy vấn Q.'],
               x=666, color=h['orange'], bold=True)
    b += _note(h, 561, ['Trạng thái đổi theo epoch 60 s; cache không',
                        'tạo truy vấn bổ sung theo mục đích riêng.'], x=666)
    b += _note(h, 640, 'Hai chẩn đoán trên 72 khối đã khảo sát (24 nhóm × 3 lượt nhiễu), mỗi khối 8 chuyến; dữ liệu mô phỏng.',
               color=h['gray'])
    return b
