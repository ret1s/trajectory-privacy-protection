"""Scientific multi-step figures from the frozen TRAIN demonstration only.

The caller owns slide framing/export. These functions render source coordinates
and saved component triggers; no privacy mechanism or attacker is executed.
"""
import math

from focus_visuals import table
from sample_visuals import GPS_COLOR, map_svg


def _rows(data):
    selection = data['selection']
    if (selection['family_id'], selection['split'], selection['draw'], selection['main_slot']) != (
            'freshqp-001', 'train', 1, 0):
        raise ValueError('This figure requires the fixed TRAIN illustration')
    rows = data['main_sample']['events']
    if [r['t_s'] for r in rows] != [0, 20, 60, 120, 180, 240, 300, 600]:
        raise ValueError('Every declared illustrative time must be retained')
    if data['main_sample']['allocation']['max_units'] != 23:
        raise ValueError('Session allowance changed')
    if [r['protection']['spent_after_units'] for r in rows] != [1, 1, 3, 5, 7, 9, 11, 21]:
        raise ValueError('Saved cumulative budget changed')
    return rows


def _num(value, digits=0):
    if not math.isfinite(value):
        raise ValueError('Finite source number required')
    return f'{value:.{digits}f}'.replace('.', ',')


def moving_timeline(data, *, text, line, ink, teal, blue, gray, orange, **helpers):
    """All eight displayed events of one trip; costs include unshown reads."""
    events = _rows(data)
    rows, colors = [], []
    for event in events:
        protection, belief = event['protection'], event['belief']
        utility = event['retrieval_and_local_ranking']
        rows.append((
            _num(event['t_s']),
            'Có' if protection['GPS_read'] else 'Không',
            'Tạo mới' if protection['branch'] == 'fresh' else 'Giữ nguyên',
            'Cập nhật từ Z' if belief['observed_emission_this_event'] else 'Chỉ dự đoán',
            f"+{protection['cost_units']} / {protection['spent_after_units']}",
            str(utility['merged_unique_count']),
        ))
        colors.append(blue if protection['GPS_read'] else gray)
    body = text(56, 149, 'Một chuyến: GPS thay đổi theo hành trình; Q công bố mỗi 20 s', 28, weight=700)
    rendered, bottom = table(
        ['t (s)', 'GPS cho\nGeo-I', 'Z tham chiếu', 'Phân bố b', '+ chi /\nđã dùng', 'POI hợp nhất'],
        rows, [105, 155, 210, 255, 220, 223], text=text, line=line,
        ink=ink, gray=gray, teal=teal, y=194, row_h=37,
        size=25, header_size=24, colors=colors,
    )
    body += rendered
    if bottom > 565:
        raise ValueError('Timeline exceeds its reserved area')
    body += text(56, 590, '300 → 600 s còn đọc 360/420/480/540 s; tổng 21/23 đã bao gồm các lần này.',
                 25, color=orange, weight=700)
    body += text(56, 635, 'Mỗi mốc: 5 Q → phản hồi L30 → bỏ trùng → top-5 theo GPS và mục đích tại thiết bị.',
                 25, color=teal)
    return body


def moving_maps(data, *, text, line, dot, ink, teal, blue, gray, orange, **helpers):
    """Same metric viewport at fixed start/120s/end; markers retain source XY."""
    events = _rows(data)
    times = (0, 120, 600)
    # All three map calls see exactly the same important coordinates. Thus the
    # existing map renderer computes one shared origin, scale and viewport.
    common = []
    for event in events:
        common += [event['gps_xy'], event['Z_xy'], *event['Q_xy'], *event['belief_top_xy']]
    roads = data['map_main']['road_segments_xy']
    body = text(56, 146, 'Cùng bản đồ và tỷ lệ: GPS, Z và năm Q qua các mốc chuyển động', 27, weight=700)
    body += line(57, 182, 69, 194, GPS_COLOR, 3) + line(57, 194, 69, 182, GPS_COLOR, 3)
    body += text(83, 196, 'GPS local / phân tích', 23, color=GPS_COLOR)
    body += text(422, 196, '◆ Z nội bộ', 23, color=blue)
    body += dot(677, 188, 7, teal) + text(693, 196, 'Q gửi máy chủ', 23, color=teal)
    body += text(991, 196, 'b: top 12 trọng số', 22, color=orange)
    for frame, t in enumerate(times):
        event = next(row for row in events if row['t_s'] == t)
        x = 56 + frame * 397
        body += text(x, 239, f't = {t} s', 28, weight=700, color=blue)
        marks = [{'xy':event['gps_xy'], 'color':GPS_COLOR, 'shape':'cross',
                  'label':'GPS', 'label_dx':-15, 'label_dy':22, 'anchor':'end'},
                 {'xy':event['Z_xy'], 'color':blue, 'shape':'diamond',
                  'label':'Z', 'label_dx':-14, 'anchor':'end'}]
        if t == 600:
            marks[1].update(label_dx=17, label_dy=5, anchor='start')
        for j, p in enumerate(event['Q_xy']):
            mark = {'xy':p, 'color':teal, 'label':f'{j+1}', 'radius':7}
            if t == 600 and j == 2:
                mark.update(label_dx=22, label_dy=-4)
            marks.append(mark)
        cells = [{'xy':r['xy_m'], 'weight':r['weight']} for r in event['belief']['top_weights']]
        body += map_svg(f'multistep-{frame}', (x, 256, 374, 279), roads, common, marks,
                        text=text, line=line, dot=dot, belief=cells)
        body += text(x, 568, f"Đã dùng {event['protection']['spent_after_units']}/23 đơn vị", 24, color=blue)
        body += text(x, 594, f"Top 12: {_num(100*event['belief']['top_weights_mass'],2)}% khối lượng b", 22, color=orange)
    body += text(56, 621, '1–5 là nhãn theo dõi nội bộ của Q, không phải ID ổn định gửi máy chủ.',
                 22, color=gray)
    body += text(56, 646, 'Top 12 chỉ là một phần của b; b không phải posterior đã hiệu chuẩn.',
                 22, color=gray)
    return body


def actual_noisy_reuse(data, *, text, line, box, arrow, dot, math_text,
                       ink, teal, blue, gray, orange, **helpers):
    """First actual saved reuse in another session, never spliced into main."""
    _rows(data)
    inset = data['test_reuse_inset']
    if inset is None or inset['slot'] != 5 or [r['t_s'] for r in inset['events']] != [0, 60]:
        raise ValueError('The mechanically first noisy-reuse inset is absent')
    before, after = inset['events']
    p = after['protection']
    if not p['GPS_read'] or p['branch'] != 'reuse' or p['cost_units'] != 1 or p['spent_after_units'] != 2:
        raise ValueError('Actual reuse branch/cost changed')
    if before['Z_xy'] != after['Z_xy'] or p['Laplace_noise_recorded']:
        raise ValueError('Reuse must retain Z and must not invent a noise draw')
    body = text(56, 148, 'Cùng family, chuyến 6 tại 60 s: đã đọc GPS nhưng giữ Z cũ', 28, weight=700)
    rows = [(str(int(row['t_s'])), 'Có', 'Tạo mới' if row['protection']['branch']=='fresh' else 'Giữ nguyên',
             f"+{row['protection']['cost_units']} / {row['protection']['spent_after_units']}") for row in inset['events']]
    rendered, bottom = table(['t (s)', 'GPS cho Geo-I', 'Z', '+ chi / đã dùng'], rows,
                            [85, 200, 170, 150], text=text, line=line, ink=ink,
                            gray=gray, teal=teal, y=217, row_h=60, size=25, header_size=23)
    body += rendered
    body += text(710, 210, 'Những thành phần được kích hoạt', 27, color=teal, weight=700)
    body += line(710, 230, 1224, 230, ink, 1.3)
    body += text(710, 271, '1. Đọc GPS sau kiểm tra ngân sách.', 25)
    body += text(710, 318, '2. Thử có nhiễu đạt → giữ Z.', 25, color=blue)
    body += text(710, 365, '3. b cập nhật từ quan sát đã bảo vệ.', 25, color=orange)
    body += text(710, 412, '4. Chọn Q, lấy POI, trả top-5 tại thiết bị.', 25, color=teal)
    body += text(56, 406, f"Khoảng cách GPS tới Z cũ: {_num(p['distance_GPS_to_previous_Z_m'])} m.", 26, color=blue)
    body += math_text(56, 455, ['d', ('E','sub'), '(x', ('60','sub'), ', Z', ('0','sub'), ') + η', ('60','sub'), ' ≤ 200 m'], 29)
    body += text(56, 501, 'Nhiễu có thể âm; không so khoảng cách thật trực tiếp với 200 m.', 24, color=gray)
    body += text(56, 548, f"Giới hạn suy ra của η khoảng {_num(p['inferred_noise_condition']['bound_m'])} m; không lưu η cụ thể.", 25, color=gray)
    body += text(56, 599, 'Đọc rồi giữ Z: chi 1 đơn vị. Không đọc theo lịch: chi 0; b chỉ dự đoán.', 25, color=teal, weight=700)
    body += text(56, 641, 'Ô phụ từ phiên khác; không nối vào phiên đầu. Không lấy mẫu lại Z/Q hoặc nhiễu.', 23, color=gray)
    return body
