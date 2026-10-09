"""Compact source-backed algorithm and sample figures for the supervisor deck.

Only rendering is performed. Protected outputs, costs, branches, POI replies
and local rankings are supplied by frozen sample JSON. The public caller owns
slide titles, footers and export; the earlier figure modules remain untouched.
"""
import math

from sample_visuals import GPS_COLOR, map_svg


def _number(value, digits=0):
    if not math.isfinite(float(value)):
        raise ValueError('Finite source number required')
    return f'{float(value):.{digits}f}'.replace('.', ',')


def _table(headers, rows, widths, *, text, line, ink, x=46, y=160,
           size=21, header_size=22, row_padding=14, colors=None):
    positions = [x]
    for width in widths[:-1]:
        positions.append(positions[-1] + width)
    result = line(x, y-26, x+sum(widths), y-26, ink, 1.2)
    header_height = max(len(str(value).split('\n')) for value in headers)*header_size*1.13
    for pos, value in zip(positions, headers):
        result += text(pos+5, y, value, header_size, weight=700, lh=1.13)
    cursor = y + header_height + 9
    result += line(x, cursor-11, x+sum(widths), cursor-11, ink, 1.1)
    for index, row in enumerate(rows):
        color = colors[index] if colors else ink
        height = max(len(str(value).split('\n')) for value in row)*size*1.13 + row_padding
        for j, (pos, value) in enumerate(zip(positions, row)):
            result += text(pos+5, cursor+size, value, size,
                           color=color, weight=700 if j==0 else 400, lh=1.13)
        cursor += height
    result += line(x, cursor+5, x+sum(widths), cursor+5, ink, 1.2)
    return result, cursor+5


def _events(data):
    selection = data['selection']
    if (selection['family_id'], selection['split'], selection['draw'], selection['main_slot']) != (
            'freshqp-001', 'train', 1, 0):
        raise ValueError('The fixed TRAIN demonstration is required')
    rows = data['main_sample']['events']
    if [r['t_s'] for r in rows] != [0, 20, 60, 120, 180, 240, 300, 600]:
        raise ValueError('All eight illustrative times must remain present')
    if [r['protection']['spent_after_units'] for r in rows] != [1, 1, 3, 5, 7, 9, 11, 21]:
        raise ValueError('The frozen cumulative costs differ')
    if data['main_sample']['allocation']['max_units'] != 23:
        raise ValueError('The source session cap differs')
    return rows


def architecture(content, *, text, line, dot, arrow, box, math_text,
                 ink, teal, blue, gray, orange, **helpers):
    """Nine ordered components; GPS is supplied after the public read gate."""
    result = box(244, 139, 762, 411, stroke=teal)
    result += text(262, 162, 'PHƯƠNG PHÁP TẠI THIẾT BỊ', 22, color=teal, weight=700)
    result += text(46, 148, 'ĐẦU VÀO', 20, color=gray, weight=700)
    result += text(46, 178, 'GPS cho Geo-I\nđọc sau kiểm tra', 21, color=blue, lh=1.15)
    # The private coordinate enters component 3, never admission/read gates.
    result += arrow([(208, 190), (232, 190), (232, 128), (897, 128), (897, 176)], blue)
    result += text(262, 203, 'Bảo vệ GPS', 21, color=blue, weight=700)
    result += box(405, 176, 175, 63, '1. Nhận phiên\nslot / cap epoch', stroke=blue, size=20)
    result += box(608, 176, 175, 63, '2. Lịch ≥60 s\ncap dự toán', stroke=blue, size=20)
    result += box(811, 176, 175, 63, '3. Thử / REM\nGiữ hoặc tạo Z', stroke=blue, size=20)
    result += arrow([(580, 207), (608, 207)], blue)
    result += arrow([(783, 207), (811, 207)], blue)
    result += text(262, 303, 'Ước lượng\nvà chọn Q', 21, color=orange, weight=700, lh=1.13)
    result += box(405, 282, 210, 64, '4. b từ Z\nvà lịch sử bảo vệ', stroke=orange, size=20)
    result += box(653, 282, 333, 64, '5. Chọn 5 Q theo đường\nphủ POI / tiến độ / slack', stroke=orange, size=20)
    result += arrow([(897, 239), (897, 255), (392, 255), (392, 314), (405, 314)], blue)
    result += arrow([(615, 314), (653, 314)], orange)
    result += text(46, 309, 'Bản đồ + POI\nprior, lịch, tham số\ncông khai', 20, color=gray, lh=1.16)
    result += arrow([(208, 323), (229, 323), (229, 266), (379, 266), (379, 330), (405, 330)], gray, '4 4')
    result += text(262, 400, 'Truy hồi\nPOI chung', 21, color=teal, weight=700, lh=1.13)
    result += box(405, 378, 581, 59, '6. Chỉ gửi Q; mọi loại POI, L = 30 cố định', stroke=teal, size=21)
    result += arrow([(820, 346), (820, 361), (695, 361), (695, 378)], teal)
    result += text(1034, 359, 'MÁY CHỦ', 21, color=orange, weight=700)
    result += box(1034, 378, 200, 59, '7. Top-L POI\nmỗi loại / Q', stroke=orange, size=20)
    result += arrow([(986, 407), (1034, 407)], teal)
    result += text(262, 495, 'Kết quả\ntại thiết bị', 21, color=teal, weight=700, lh=1.13)
    result += box(405, 472, 210, 64, '8. Hợp POI\nbỏ trùng / hợp lệ', stroke=teal, size=20)
    result += box(653, 472, 333, 64, '9. Xếp hạng local\nGPS + nhu cầu ψ → top-5', stroke=teal, size=20)
    result += arrow([(1034, 428), (1018, 428), (1018, 450), (392, 450), (392, 504), (405, 504)], teal)
    result += arrow([(615, 504), (653, 504)], teal)
    result += text(46, 477, 'GPS local + ψ\nloại POI, bán kính,\nđích riêng', 20, color=blue, lh=1.16)
    result += arrow([(208, 493), (229, 493), (229, 562), (801, 562), (801, 536)], blue)
    result += arrow([(920, 536), (920, 565)], teal)
    result += text(920, 585, '≤5 POI phù hợp tại thiết bị', 21, color=teal, weight=700, anchor='middle')
    result += text(46, 608, 'u = 0,00125/m; θ = 200 m', 20, color=blue)
    result += text(405, 608, 'K = 5; planner L10; server L30; local k = 5', 20, color=teal)
    result += text(46, 637, 'Epoch8 / H12: Cepoch = 0,23/m; Cphiên = 23u = 0,02875/m. Lần đầu 1u; không đọc 0u; giữ 1u; thử + tạo mới 2u.', 19, color=gray)
    return result


def changes(content, *, text, line, ink, teal, blue, gray, orange, **helpers):
    """Nine concrete retained/updated facts; boundary arm stays separate."""
    rows = [
        ('REM / noisy reuse', 'Đã có tạo / giữ Z có nhiễu', 'Giữ nguyên nguyên lý; u: 0,01 → 0,00125/m'),
        ('Lịch đọc GPS', 'Ít nhất 60 s; có ngân sách', 'Giữ ≥60 s; dự toán trước khi gọi GPS'),
        ('Phạm vi ngân sách', 'Cận 0,23/m cho từng phiên', 'Cap chung 8 phiên: 0,23/m; mỗi phiên 0,02875/m'),
        ('Ước lượng b', 'Từ lịch sử đã bảo vệ', 'Giữ; không đọc → dự đoán, có đọc → cập nhật'),
        ('Mạng đường', 'Mạng nhỏ dựng lại để minh họa', 'Đúng mạng SUMO nguồn, gồm đoạn nội bộ nút giao'),
        ('Bộ chọn 5 Q', 'Đường khả thi, phủ POI, tiến độ', 'Giữ slack 0,03 và chữ ký planner L10'),
        ('Phản hồi POI', 'Mọi loại POI, L10 / loại / Q', 'L30 / loại / Q; đã so L20→L30 trên cùng Q'),
        ('Gộp / dữ liệu hợp lệ', 'Bỏ trùng; đã có status epoch60', 'Static cache theo version tùy chọn; status phải mới'),
        ('Xếp hạng local', 'Top-5 gần nhất mỗi loại', '4 mục đích; GPS / bán kính / đích không gửi'),
    ]
    result, bottom = _table(['Thành phần', 'Trước', 'Hiện tại'], rows, [224, 396, 568],
                            text=text, line=line, ink=ink, y=158, size=20, row_padding=16)
    if bottom > 558:
        raise ValueError('Changes table exceeds its reserved area')
    result += text(46, 577, 'Đầu / cuối: bỏ đầu + delay trước đây; Epoch8/L30 gửi ngay. Endpoint20 là nhánh L20 riêng.', 21, color=orange)
    result += text(46, 609, 'Cache không phải thành phần hoàn toàn mới; đánh giá GPS thưa / status động kiểm tra giả định sử dụng.', 20, color=gray)
    result += text(46, 640, 'Z giữ nội bộ; năm Q không phải năm mẫu REM độc lập. Tăng L trả thêm dữ liệu, không tăng mức bảo vệ tọa độ.', 20, color=gray)
    return result


def timeline(data, *, text, line, ink, teal, blue, gray, orange, **helpers):
    events = _events(data)
    result = text(46, 142, 'Cùng chuyến TRAIN: các lần đọc, giữ / tạo Z, b và POI thực tế', 23, weight=700)
    rows, colors = [], []
    for event in events:
        p = event['protection']
        rows.append((_number(event['t_s']), 'Có' if p['GPS_read'] else 'Không',
                     'Tạo mới' if p['branch']=='fresh' else 'Giữ',
                     'Từ quan sát Z' if p['GPS_read'] else 'Chỉ dự đoán',
                     f"+{p['cost_units']} / {p['spent_after_units']}",
                     str(event['retrieval_and_local_ranking']['merged_unique_count'])))
        colors.append(blue if p['GPS_read'] else gray)
    rendered, bottom = _table(['t (s)', 'GPS\nGeo-I', 'Z', 'b', '+chi / tổng', 'POI hợp'], rows,
                              [80, 115, 140, 185, 135, 210], text=text, line=line,
                              ink=ink, y=187, size=21, row_padding=16, colors=colors)
    result += rendered
    if bottom > 572:
        raise ValueError('Timeline table exceeds reserved height')
    x = 951
    result += text(x, 173, 'Diễn giải các nhánh', 22, color=teal, weight=700)
    result += line(x, 185, 1234, 185, ink, 1.1)
    result += text(x, 216, '0 s: REM lần đầu\nchi 1u; Z mới.', 21, color=blue, lh=1.18)
    moved = events[1]['movement_since_previous_display']
    if not (150 < moved['Q_straight_line_m_by_track'][0] < 160 and moved['Q_straight_line_m_by_track'][1:]==[0,0,0,0]):
        raise ValueError('The illustrated t20 track movement changed')
    result += text(x, 286, '20 s: không đọc mới\nGPS đổi ~118 m; Q1 ~155 m\nQ2–Q5 đứng yên.', 20, color=gray, lh=1.18)
    result += text(x, 380, '60 s: thử rồi tạo Z\nchi 2u; tổng 3u.', 21, color=blue, lh=1.18)
    result += text(x, 452, '600 s: GPS đã dừng\n21u = 0,02625/m\nCòn 2u; chưa hết cap.', 20, color=blue, lh=1.18)
    result += text(x, 549, 'u = 0,00125/m\nCap = 23u = 0,02875/m', 20, color=teal, lh=1.18)
    result += text(46, 592, '300→600 s còn các lần đọc 360/420/480/540 s; tổng 21 đã tính đủ, không bỏ chi phí giữa hai hình.', 20, color=orange)
    result += text(46, 619, 'Mọi event: 5 Q → L30 → hợp POI → top-5 local. Mẫu current-only / static, không có status hay cache.', 20, color=teal)
    result += text(46, 645, 'Bán kính tại 240 s có tham chiếu rỗng: N/A, không gán 100%. 600 s vẫn không phải nhánh hết ngân sách.', 19, color=gray)
    return result


def maps(data, *, text, line, dot, ink, teal, blue, gray, orange, **helpers):
    events = _events(data)
    result = text(46, 141, 'Cùng vùng nhìn và tỷ lệ mét: GPS phân tích, Z nội bộ và Q công bố', 23, weight=700)
    result += text(46, 174, '× GPS local', 20, color=GPS_COLOR)
    result += text(296, 174, '◆ Z giữ trên thiết bị', 20, color=blue)
    result += dot(584, 166, 6, teal)+text(600, 174, 'Q gửi máy chủ', 20, color=teal)
    result += text(897, 174, 'Cam: 12 trọng số b cao nhất', 20, color=orange)
    important = [p for event in events for p in [event['gps_xy'], event['Z_xy'], *event['Q_xy'], *event['belief_top_xy']]]
    def label_text(x, y, value, size=23, **kw):
        return text(x, y, value, size*.82, **kw)
    for i, t in enumerate((0,120,600)):
        row = next(e for e in events if e['t_s']==t)
        x = 46 + i*405
        result += text(x, 207, f't = {t} s', 23, color=blue, weight=700)
        marks = [{'xy':row['gps_xy'], 'color':GPS_COLOR, 'shape':'cross', 'label':'GPS',
                  'label_dx':-14, 'label_dy':20, 'anchor':'end'},
                 {'xy':row['Z_xy'], 'color':blue, 'shape':'diamond', 'label':'Z',
                  'label_dx':-12, 'anchor':'end'}]
        if t==600:
            marks[1].update(label_dx=15, label_dy=4, anchor='start')
        for j, p in enumerate(row['Q_xy']):
            mark = {'xy':p,'color':teal,'radius':7,'label':str(j+1)}
            if t==600 and j==2:
                mark.update(label_dx=19,label_dy=-5)
            marks.append(mark)
        belief = [{'xy':p['xy_m'],'weight':p['weight']} for p in row['belief']['top_weights']]
        result += map_svg(f'compact-moving-{i}', (x,223,378,324), data['map_main']['road_segments_xy'],
                          important, marks, text=label_text,line=line,dot=dot,belief=belief)
        result += text(x, 575, f"Chi {row['protection']['spent_after_units']}/23; GPS {'đã dừng' if t==600 else 'thay đổi theo hành trình'}", 20, color=blue)
        result += text(x, 605, f"Top 12 = {_number(100*row['belief']['top_weights_mass'],2)}% khối lượng b", 20, color=orange)
    result += text(46, 633, '1–5 là nhãn theo dõi nội bộ; không gửi ID track ổn định. Từng Q có thể đứng yên.', 19, color=gray)
    result += text(46, 657, 'b chưa được hiệu chuẩn thành posterior / vùng tin cậy; màu cam không phải toàn bộ phân bố.', 18, color=gray)
    return result


def reuse(data, *, text, line, math_text, ink, teal, blue, gray, orange, **helpers):
    events = _events(data)
    inset = data['test_reuse_inset']
    if inset is None or inset['slot']!=5 or [e['t_s'] for e in inset['events']]!=[0,60]:
        raise ValueError('The mechanically first actual reuse inset is required')
    before, after = inset['events']; p = after['protection']
    if not p['GPS_read'] or p['branch']!='reuse' or p['cost_units']!=1 or before['Z_xy']!=after['Z_xy']:
        raise ValueError('The saved reuse branch differs')
    if p['Laplace_noise_recorded'] or p['Laplace_noise_m'] is not None:
        raise ValueError('No numerical noise draw may be supplied to this figure')
    result = text(46, 142, 'Chuyến 6, 60 s: đã đọc GPS, phép thử có nhiễu vẫn giữ nguyên Z', 23, weight=700)
    rows = [(str(int(e['t_s'])), 'Có', 'Tạo mới' if e['protection']['branch']=='fresh' else 'Giữ nguyên',
             f"+{e['protection']['cost_units']} / {e['protection']['spent_after_units']}") for e in inset['events']]
    rendered, _ = _table(['t (s)', 'Đọc GPS?', 'Z', '+chi / tổng'], rows,[80,170,190,160],
                          text=text,line=line,ink=ink,y=190,size=21,row_padding=24)
    result += rendered
    result += text(710, 185, 'Phép thử và ngân sách', 23, color=blue, weight=700)
    result += text(710, 225, f"dE(GPS, Z cũ) = {_number(p['distance_GPS_to_previous_Z_m'])} m", 21, color=blue)
    result += math_text(710, 271, ['d',('E','sub'),'(x',('60','sub'),', Z',('0','sub'),') + η',('60','sub'),' ≤ 200 m'], 25)
    result += text(710, 309, 'η có thể âm; đây không phải ngưỡng\ncứng áp vào khoảng cách thật.', 21, color=gray, lh=1.15)
    result += text(46, 359, f"Giới hạn η suy ra khoảng {_number(p['inferred_noise_condition']['bound_m'])} m; giá trị nhiễu thực tế không được lưu.", 21, color=gray)
    result += text(46, 393, 'Trước khi đọc phải dự toán 2u; nhánh giữ thực tế chi 1u. b vẫn cập nhật từ quan sát đã bảo vệ.', 20, color=teal)
    comparison = [
        ('Phiên 1 · 20 s','Không','Giữ Z','0u','Chỉ dự đoán'),
        ('Phiên 6 · 60 s','Có','Giữ Z','1u','Dự đoán + cập nhật'),
        ('Phiên 1 · 60 s','Có','Tạo Z mới','2u','Dự đoán + cập nhật'),
    ]
    rendered,bottom = _table(['Trạng thái thực tế','Đọc GPS?','Kết quả Z','Chi phí','b'], comparison,
                             [280,165,215,165,363],text=text,line=line,ink=ink,y=444,
                             size=21,row_padding=16,colors=[gray,teal,blue])
    result += rendered
    if bottom>616:
        raise ValueError('Branch comparison exceeds reserved height')
    result += text(46, 640, 'So ba nhánh từ hai phiên, không ghép thành một timeline; sau mỗi nhánh vẫn chọn Q, lấy POI và trả top-5 local.', 19, color=gray)
    return result


def utility(data, protection, *, text, line, dot, ink, teal, blue, gray, orange, **helpers):
    category = data['illustration_category']
    output = {purpose:row[category] for purpose,row in data['local_results'].items()}
    projection = protection['metadata']['projection']
    def xy(record):
        return [record['lon']*projection['m_per_deg_lon'],record['lat']*projection['m_per_deg_lat']]
    gps = xy(data['inputs']['GPS'])
    nearest = output['nearest_distance']['answer']
    points = [xy(p) for p in nearest]
    result = text(46, 142, 'Cùng Q tại 60 s: 420 bản ghi → 252 POI; bốn nhu cầu chỉ đổi xếp hạng local', 23, weight=700)
    marks = [{'xy':pos,'color':blue,'shape':'square','label':p['display_alias']} for pos,p in zip(points,nearest)]
    for mark in marks:
        if mark['label']=='P13':mark.update(label_dx=-16,label_dy=0,anchor='end')
        if mark['label']=='P224':mark.update(label_dx=16,label_dy=-12,anchor='start')
    marks.append({'xy':gps,'color':GPS_COLOR,'shape':'cross','label':'GPS'})
    def label_text(x,y,value,size=23,**kw):
        return text(x,y,value,size*.82,**kw)
    result += map_svg('compact-utility',(46,177,505,317),protection['map_main']['road_segments_xy'],
                      [gps,*points],marks,text=label_text,line=line,dot=dot)
    result += text(46, 522, 'GPS local và năm POI café gần nhất', 21, color=blue)
    result += text(46, 552, 'P là POI thật trong danh mục; Q là tọa độ truy vấn.', 20, weight=700)
    result += text(46, 583, 'L30 là tối đa / loại / Q; mẫu trả 84 bản ghi / Q.', 19, color=gray)
    rows=[]
    for purpose,label in [('nearest_distance','Gần nhất\n(khoảng cách m)'),
                          ('fastest_travel','Nhanh nhất\n(free-flow s)'),
                          ('within_radius','Trong bán kính\n≤1.000 m'),
                          ('minimum_detour','Ít đi vòng\n(độ dài thêm m)')]:
        answer = output[purpose]['answer']
        cells = [label]+[f"{p['display_alias']}\n{_number(p['score'],1 if purpose=='fastest_travel' else 0)} {p['score_unit']}" for p in answer]
        cells += ['—']*(6-len(cells))
        rows.append(cells)
    rendered,bottom = _table(['Nhu cầu riêng','POI 1','POI 2','POI 3','POI 4','POI 5'],rows,
                            [168,94,94,94,94,94],text=text,line=line,ink=ink,
                            x=596,y=197,size=19,header_size=20,row_padding=29,
                            colors=[blue,blue,teal,ink])
    result += rendered
    if bottom>561:
        raise ValueError('Four-purpose POI table exceeds reserved height')
    result += text(596, 563, 'Bán kính trả 1 POI; các ô — không là POI giả.', 19, color=teal)
    result += text(596, 590, 'Đổi ψ không tạo thêm Q hay request; nearest / fastest có thể trùng.', 18, color=gray)
    result += text(46, 619, 'Khoảng cách theo đường có hướng, không phải đường thẳng trên hình; fastest là thời gian free-flow, không đo traffic.', 19, color=gray)
    result += text(46, 645, 'Đích detour là đáp án local trong evaluator; triển khai cần đích người dùng đã biết. Một mẫu TRAIN không thay benchmark.', 19, color=gray)
    return result
