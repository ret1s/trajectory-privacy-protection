"""Render scoped benchmark tables from retained, source-pinned readouts.

Rates in the input JSON are fractions, distances are meters, and costs are
application JSON bytes. Rendering rounds for display; it never evaluates a
mechanism, fits an attacker, or combines different experimental cohorts.
"""
from focus_visuals import table, topic


def number(value, digits=2):
    return 'N/A' if value is None else f'{value:.{digits}f}'.replace('.', ',')


def percent(value, digits=2):
    return 'N/A' if value is None else number(100 * value, digits)


def _table_helpers(h):
    return {key: h[key] for key in ('text', 'line', 'ink', 'gray', 'teal')}


def historical_location_results(data, **h):
    """S1--S3, five historical methods; each metric remains a separate row."""
    text = h['text']
    records = data['S1_S3_historical']['rows']
    methods = ['br_private', 'dls_graph_adaptation',
               'semantic_correlation_local_adaptation',
               'transprotect_adaptation', 'unprotected']
    indexed = {(row['scenario'], row['method']): row for row in records}
    rows = []
    for scenario in ('S1', 'S2', 'S3'):
        for label, field, transform in [('Hit100 (%) ↓', 'hit100', percent),
                                        ('MAE (m) ↑', 'mae_m', number),
                                        ('Recall@5 (%) ↑', 'recall5', percent)]:
            rows.append([f'{scenario} · {label}'] + [
                transform(indexed[scenario, method][field]) for method in methods])
    b = text(56, 140, 'K = 5; so sánh kiểm soát các bản triển khai thích nghi của phương pháp đối chứng.',
             26, weight=700)
    headers = ['Thước đo', 'Đề xuất\n(bản trước)', 'DLS\n(thích nghi)',
               'Semantic\n(thích nghi)', 'TransProtect\n(thích nghi)', 'GPS chưa\nbảo vệ']
    t, _ = table(headers, rows, [245, 190, 167, 194, 194, 178],
                 y=186, row_h=37, size=23, header_size=22,
                 **_table_helpers(h))
    b += t
    b += text(56, 615, '12 ô mẫu × lượt nhiễu mỗi hàng; Hit và MAE có thể dùng bộ suy luận khác nhau.',
              24, color=h['gray'])
    b += text(56, 650, 'Đây là cấu hình trước; chưa xác nhận S1–S3 cho Epoch8/L30 hoặc tái lập nguyên bản paper.',
              24, color=h['orange'], weight=700)
    return b


def linkage_companion_results(data, **h):
    """Two separately scoped historical studies, with no pooled identity claim."""
    text = h['text']
    left = []
    colors = []
    indexed = {(row['method'], row['target']): row
               for row in data['S4_historical_linkage']['rows']}
    for label, target in [('người', 'same_person'), ('xe', 'same_vehicle')]:
        for method, name in [('raw', 'GPS thật'), ('geoi_slack_reconstructed', 'Geo-I')]:
            row = indexed[method, target]
            left.append([f'{name} · {label}', percent(row['balanced_accuracy']),
                         number(row['roc_auc'], 3)])
            colors.append(h['gray'] if method == 'raw' else h['teal'])
    b = text(56, 145, 'S4 · Liên kết phiên của người / xe', 29, color=h['teal'], weight=700)
    t, _ = table(['Dữ liệu / đích', 'BA ↓\n(%)', 'AUC'], left,
                 [315, 117, 112], y=192, row_h=55, size=25,
                 header_size=23, colors=colors, **_table_helpers(h))
    b += t
    views = {'target_only': 'Chỉ mục tiêu',
             'joint_protected_partner': 'Đồng hành đã bảo vệ',
             'joint_public_raw_partner': 'Đồng hành GPS thật',
             'joint_unrelated_protected_partner': 'Người không liên quan',
             'raw_target_positive_control': 'GPS mục tiêu thật'}
    right = []
    right_colors = []
    for row in data['S8_historical_companion']['rows']:
        right.append([views[row['view']], number(row['mae_m'], 1), percent(row['hit100'])])
        right_colors.append(h['blue'] if row['view'] == 'joint_public_raw_partner'
                            else h['gray'] if row['view'] == 'raw_target_positive_control'
                            else h['ink'])
    b += text(674, 145, 'S8 · Suy vị trí từ người đồng hành', 29, color=h['teal'], weight=700)
    t, _ = table(['Nguồn quan sát', 'MAE ↑\n(m)', 'Hit100 ↓\n(%)'], right,
                 [318, 126, 106], x=674, y=192, row_h=54, size=23,
                 header_size=22, colors=right_colors, **_table_helpers(h))
    b += t
    b += text(56, 523, ['3 nhóm kiểm tra; 45 cặp mỗi tác vụ.',
                       'BA: cân bằng hai lớp; AUC người',
                       'đạt 0,861 ở một nhóm đã bảo vệ.'],
              24, color=h['gray'], lh=1.25)
    b += text(674, 548, ['3 nhóm; 24 mốc mục tiêu duy nhất.',
                        '41 quan sát cặp; lấy trung bình theo nhóm.'],
              24, color=h['gray'], lh=1.25)
    b += text(56, 643, 'Hai phép thử thăm dò riêng; chưa xác nhận S4/S8 cho Epoch8/L30, không phải ẩn tài khoản/IP.',
              24, color=h['orange'], weight=700)
    return b


def endpoint_final_results(data, **h):
    """Matched L20 endpoint contrast; service metrics refer to whole trips."""
    text = h['text']
    section = data['S9_S10_full_bank_endpoint']
    indexed = {(row['scenario'], row['method']): row for row in section['rows']}
    rows = []
    colors = []
    for scenario, label in [('S9', 'S9 · điểm đầu'), ('S10', 'S10 · điểm cuối')]:
        for method, name in [('scale100_L20', 'Geo-I đối chứng, L20'),
                             ('scale025_L20', 'Endpoint20, L20')]:
            row = indexed[scenario, method]
            rows.append([label, name, number(row['mae_m'], 1),
                         percent(row['hit100']), percent(row['hit500'])])
            colors.append(h['teal'] if method == 'scale025_L20' else h['gray'])
    b = text(56, 141, '28 nhóm đã khảo sát; 112 quan sát mỗi kịch bản; cấu hình L20 riêng với L30 hiện tại.',
             25, weight=700)
    t, _ = table(['Kịch bản', 'Cấu hình', 'MAE ↑\n(m)', 'Hit100 ↓\n(%)', 'Hit500 ↓\n(%)'],
                 rows, [205, 345, 208, 205, 205], y=191, row_h=62,
                 size=25, header_size=24, colors=colors, **_table_helpers(h))
    b += t
    control = indexed['S10', 'scale100_L20']
    selected = indexed['S10', 'scale025_L20']
    b += text(56, 550, f"Recall cả chuyến: {percent(control['recall5_same_frozen_service'])}% → "
              f"{percent(selected['recall5_same_frozen_service'])}%; gửi ngay, không trì hoãn.",
              27, weight=700)
    b += text(56, 593, 'JSON / mốc: ' + number(control['family_mean_total_JSON_bytes_per_input_event'])
              + ' → ' + number(selected['family_mean_total_JSON_bytes_per_input_event'])
              + ' byte. Không phải độ trễ mạng đo được.', 25, color=h['gray'])
    contrast = next(row for row in section['Endpoint20_minus_plain_L20_uncertainty']
                    if row['scenario'] == 'S10' and row['metric'] == 'mae_m')
    hit = next(row for row in section['Endpoint20_minus_plain_L20_uncertainty']
               if row['scenario'] == 'S10' and row['metric'] == 'hit500')
    b += text(56, 634, 'ΔMAE S10: +' + number(contrast['delta'], 1)
              + ' m, CI [' + number(contrast['CI95'][0], 1) + '; '
              + number(contrast['CI95'][1], 1) + ']. CI ΔHit500 ['
              + number(100*hit['CI95'][0]) + '; ' + number(100*hit['CI95'][1])
              + '] điểm %, chạm 0.', 25, color=h['orange'], weight=700)
    return b


def fresh_utility_results(data, **h):
    """Fresh current-only primary contrast, all purposes plus traffic tradeoff."""
    text = h['text']
    section = data['current_fresh_four_purpose_utility']
    rows = []
    colors = []
    for row in section['rows']:
        is_primary = row['primary']
        label = 'Trung bình 4 mục đích' if is_primary else row['label_vi']
        lo, hi = row['paired95_family_bootstrap_gain_pp']
        rows.append([label, percent(row['L20_recall']), percent(row['L30_recall']),
                     '+' + number(row['gain_pp']), '[' + number(lo) + '; ' + number(hi) + ']'])
        colors.append(h['teal'] if is_primary else h['ink'])
    b = text(56, 141, '24 nhóm mới × 3 lượt nhiễu × 8 chuyến; mô phỏng cùng bản đồ, trên cùng năm Q.',
             26, weight=700)
    t, _ = table(['Mục đích', 'L20\nRecall@5 (%)', 'L30\nRecall@5 (%)',
                  'Chênh lệch\n(điểm %)', 'CI 95% chênh lệch\n(điểm %)'], rows,
                 [354, 180, 180, 180, 274], y=193, row_h=53,
                 size=25, header_size=23, colors=colors, **_table_helpers(h))
    b += t
    cost = section['cost']
    b += text(56, 552, f"Số truy vấn: {cost['L30']['requests']:,}".replace(',', '.')
              + '; giữ nguyên Q, Z, lịch gửi và ngân sách.', 27, weight=700)
    b += text(56, 595, 'Phản hồi JSON: ' + number(cost['L20']['reply_bytes']/1e6)
              + ' → ' + number(cost['L30']['reply_bytes']/1e6) + ' MB (+'
              + number(cost['reply_growth_percent']) + '%).', 27, color=h['orange'], weight=700)
    b += text(56, 638, 'Macro là chỉ tiêu chính; CI từng mục đích mang tính thăm dò. Chưa so ưu thế ở cùng chi phí.',
              24, color=h['gray'])
    return b
