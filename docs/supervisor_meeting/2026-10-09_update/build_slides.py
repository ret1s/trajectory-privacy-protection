"""Build a self-contained Vietnamese HTML slide deck from retained evidence.

The SVG objects are semantic algorithm diagrams and numeric charts, not
decorative illustrations. Chrome's print engine exports the reviewed PDF.
No benchmark is sampled, rescored or overwritten by this builder.
"""
from pathlib import Path
import hashlib
import html
import json
import re
import sample_visuals

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INK, TEAL, BLUE, GRAY = '#20232a', '#285b70', '#3f5995', '#737b85'
PALE, LINE, ORANGE = '#fafafa', '#d7d9de', '#8a6139'


def text(x, y, value, size=27, color=INK, weight=400, anchor='start', lh=1.35, italic=False):
    lines = value.split('\n') if isinstance(value, str) else value
    spans = ''.join(f'<tspan x="{x}" dy="{0 if i == 0 else size * lh}">{html.escape(str(line))}</tspan>'
                    for i, line in enumerate(lines))
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" font-style="{"italic" if italic else "normal"}" text-anchor="{anchor}">{spans}</text>')


def math_text(x, y, runs, size=30, anchor='start'):
    """Typeset notation with real subscripts/superscripts in a math serif face."""
    spans=[]
    for run in runs:
        value, position = run if isinstance(run, tuple) else (run, 'normal')
        script = position in ('sub', 'super')
        tokens=re.split(r'(exp|Lap|prev|session|[A-Za-zℓηθΔσ])',value)
        formatted=''.join(f'<tspan font-style="{"italic" if token and len(token)==1 and (token.isalpha() or token in "ℓηθΔσ") and token!="m" else "normal"}">{html.escape(token)}</tspan>' for token in tokens if token)
        spans.append(f'<tspan baseline-shift="{position if script else "baseline"}" '
                     f'font-size="{size*.72 if script else size}">{formatted}</tspan>')
    return (f'<text x="{x}" y="{y}" fill="{INK}" text-anchor="{anchor}" '
            f'xml:space="preserve" style="white-space:pre" '
            f'font-family="STIX Two Math, STIXGeneral, Times New Roman, serif">'+''.join(spans)+'</text>')


def line(x1, y1, x2, y2, color=LINE, width=2, dash=''):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>'


def arrow(points, color=TEAL, dash=''):
    pts = ' '.join(f'{x},{y}' for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3" stroke-dasharray="{dash}" marker-end="url(#arr-{color[1:]})"/>'


def box(x, y, w, h, label='', fill='white', stroke=LINE, size=25, bold=False):
    shape = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="white" stroke="{stroke}" stroke-width="1.35"/>'
    if label:
        labels = label.split('\n')
        first_y = y + h / 2 - (len(labels)-1) * size * .65 + size * .32
        shape += text(x+w/2, first_y, labels, size, weight=700 if bold else 400, anchor='middle', lh=1.3)
    return shape


def dot(x, y, radius=7, fill=TEAL):
    return f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}"/>'


def base(title, body, footer, number, total):
    markers=''.join(f'<marker id="arr-{c[1:]}" markerWidth="9" markerHeight="9" refX="8" refY="3.5" orient="auto"><path d="M0,0 L8,3.5 L0,7" fill="{c}"/></marker>' for c in [TEAL, BLUE, GRAY, ORANGE])
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 720" role="img" aria-label="{html.escape(title)}">'
            f'<defs>{markers}</defs><rect width="1280" height="720" fill="white"/>'
            + ('' if number == 1 else text(56,77,title,43,color=TEAL,weight=400) + line(56,101,1224,101,TEAL,1.4))
            + body + line(56,662,1224,662,LINE,1)
            + text(56,687,footer,18,color=GRAY,lh=1.05)
            + text(1224,687,f'{number} / {total}',18,color=GRAY,anchor='end') + '</svg>')


def cover():
    b=text(640,222,'Bảo vệ riêng tư vị trí',58,color=TEAL,anchor='middle')
    b+=text(640,300,'bằng Geo-I / REM',58,color=TEAL,anchor='middle')
    b+=line(275,340,1005,340,TEAL,1.5)
    b+=text(640,399,'Truy hồi POI theo mạng đường và đa mục đích',32,anchor='middle')
    b+=text(640,484,'Phương pháp, ví dụ minh họa và đánh giá thực nghiệm',29,anchor='middle')
    b+=math_text(640,563,['K = 5', '     L = 30', '     k = 5'],29,anchor='middle')
    b+=text(640,607,'K: số truy vấn     L: độ sâu phản hồi     k: số POI trả tại thiết bị',23,color=GRAY,anchor='middle')
    return b


def architecture():
    b=box(252,144,706,480,fill='white',stroke=TEAL)
    b+=text(280,177,'Phương pháp tại thiết bị',23,color=TEAL,weight=700)
    b+=text(57,182,'ĐẦU VÀO',23,color=GRAY,weight=700)
    b+=text(57,237,['GPS cho Geo-I','đọc sau kiểm tra'],25)
    b+=arrow([(58,300),(225,300),(225,275),(280,275)],BLUE)
    b+=text(280,221,'Tầng 1. Bảo vệ GPS',24,color=BLUE,weight=700)
    b+=box(280,241,205,78,'1. Kiểm tra lịch\n2. Geo-I / REM',fill='#edf3fb',stroke=BLUE,size=24)
    b+=arrow([(485,280),(544,280)],BLUE)
    b+=box(550,241,116,78,'Z nội bộ',fill='#edf3fb',stroke=BLUE,size=25)
    b+=text(717,271,['Điểm tham chiếu','giữ tại thiết bị'],23,color=GRAY)
    b+=text(57,348,['Bản đồ, POI','phân bố ban đầu,','lịch công khai'],23)
    b+=arrow([(57,432),(225,432),(225,426),(280,426)],GRAY,'5 5')
    b+=text(280,358,'Tầng 2. Ước lượng và chọn truy vấn',24,color=BLUE,weight=700)
    b+=box(280,375,205,69,'3. Phân bố b',fill='#fcf3ea',stroke=ORANGE)
    b+=box(550,375,372,69,'4. Chọn 5 Q đi được theo đường',fill='#fcf3ea',stroke=ORANGE,size=24)
    b+=arrow([(605,319),(605,336),(268,336),(268,395),(280,395)],BLUE)
    b+=arrow([(485,409),(544,409)],ORANGE)
    b+=arrow([(922,409),(993,409),(993,483),(1040,483)],TEAL)
    b+=text(1022,405,'MÁY CHỦ',23,color=ORANGE,weight=700)
    b+=box(1040,427,185,119,'5. Mọi loại POI\nL = 30 / loại / Q\nTrả POI ứng viên',fill='#fcf3ea',stroke=ORANGE,size=23)
    b+=text(280,482,'Tầng 3. Truy hồi mọi loại POI, L = 30',24,color=BLUE,weight=700)
    b+=text(280,539,'Tầng 4. Gộp và xếp hạng tại thiết bị',24,color=BLUE,weight=700)
    b+=box(280,554,205,55,'6. Gộp POI hợp lệ',fill=PALE,stroke=TEAL,size=24)
    b+=box(550,554,372,55,'7. Xếp hạng theo GPS và mục đích',fill=PALE,stroke=TEAL,size=24)
    b+=arrow([(1040,528),(980,528),(980,513),(267,513),(267,581),(280,581)],TEAL)
    b+=arrow([(485,581),(544,581)],TEAL)
    b+=text(57,533,['GPS cục bộ','mục đích riêng ψ'],25)
    b+=arrow([(57,577),(225,577),(225,633),(640,633),(640,609)],BLUE)
    b+=arrow([(830,609),(830,631)],TEAL)
    b+=text(736,650,'Kết quả trên thiết bị: tối đa 5 POI',25,color=TEAL,weight=700,anchor='middle')
    return b


def private_read():
    b=text(56,158,'Thuật toán 1. GPS trong phiên đã nhận',28,weight=700)
    b+=line(56,178,590,178,INK,1.4)
    b+=text(56,218,'1. Kiểm tra lịch công khai và ngân sách.',27)
    b+=text(85,255,'Không đủ: tiếp tục từ lịch sử đã bảo vệ.',25,color=GRAY)
    b+=text(56,304,'2. Đọc GPS khi được phép.',27)
    b+=text(56,353,'3. Lần đầu: lấy mẫu Z bằng REM.',27)
    b+=text(56,402,'4. Đã có Z: kiểm tra tái sử dụng có nhiễu.',27)
    b+=math_text(320,456,['d',('E','sub'),'(x',('t','sub'),', Z',('prev','sub'),') + η',('t','sub'),' ≤ θ'],31,anchor='middle')
    b+=math_text(320,500,['η',('t','sub'),' ∼ Lap(1/u)'],28,anchor='middle')
    b+=text(85,546,'Đạt: giữ Z. Không đạt: lấy mẫu REM mới.',25)
    b+=line(56,575,590,575,INK,1.4)
    b+=text(56,614,'Ít nhất 60 s giữa hai lần đọc GPS bảo vệ.',26,color=TEAL)
    b+=text(650,158,'Cơ chế REM trên miền đường cố định',29,weight=700)
    b+=math_text(650,248,['R',('u','sub'),'(v | x) ='],28)
    b+=math_text(1020,220,['exp[−u d',('E','sub'),'(x, g(v))/2]'],27,anchor='middle')
    b+=line(817,233,1224,233,INK,1.4)
    b+=math_text(1020,273,['∑',('w ∈ V','sub'),' exp[−u d',('E','sub'),'(x, g(w))/2]'],25,anchor='middle')
    b+=text(650,329,'V: miền đường công khai, cố định.',26)
    b+=text(650,366,'Khoảng cách Euclid dùng trong REM.',26)
    b+=math_text(650,417,['u = 0,00125 m',('−1','super')],28)
    b+=math_text(990,417,['θ = 200 m'],28)
    b+=text(650,472,'Chi phí ngân sách theo nhánh',27,weight=700)
    b+=line(650,489,1224,489,INK,1.4)
    b+=text(650,522,'Lần đầu: u',25)
    b+=text(840,522,'Giữ Z: u',25)
    b+=text(1000,522,'Thử + tạo mới: 2u',25)
    b+=line(650,541,1224,541,INK,1.4)
    b+=math_text(650,584,['U = 2H − 1 = 23    (H = 12)'],28)
    b+=math_text(650,629,['C',('session','sub'),' = Uu = 0,02875 m',('−1','super')],27)
    return b


def belief_and_q():
    b=text(56,147,'Ước lượng từ thông tin đã bảo vệ, chọn truy vấn để phủ POI',29,weight=700)
    b+=box(56,198,245,110,'Z và lịch sử\nđã bảo vệ',fill='#edf3fb',stroke=BLUE,size=28)
    b+=arrow([(301,251),(365,251)],BLUE)
    b+=box(372,198,326,110,'Phân bố ước lượng b\ntrên lưới đường công khai',fill='#fcf3ea',stroke=ORANGE,size=26)
    b+=arrow([(698,251),(762,251)],ORANGE)
    b+=box(768,198,456,110,'Chọn K = 5 truy vấn Q\nPhủ POI, khả thi theo đường',fill=PALE,stroke=TEAL,size=27)
    b+=text(56,341,'Có lần đọc GPS bảo vệ:',24,color=GRAY)
    b+=math_text(385,341,['b',('t','sub'),'(ℓ) ∝ (b',('t−1','sub'),' T',('Δt','sub'),')(ℓ) · ℒ(Z',('t','sub'),' | ℓ, Z',('prev','sub'),')'],29)
    b+=text(56,379,'Giữa hai lần đọc GPS:',24,color=GRAY)
    b+=math_text(385,379,['b',('t','sub'),'(ℓ) = (b',('t−1','sub'),' T',('Δt','sub'),')(ℓ)'],29)
    # A directed public-road schematic; no raw GPS or benchmark sample.
    edges=[((385,410),(595,410)),((595,410),(805,410)),((805,410),(1015,410)),
           ((385,548),(595,548)),((595,548),(805,548)),((805,548),(1015,548)),
           ((385,410),(385,548)),((595,548),(595,410)),((805,410),(805,548)),((1015,548),(1015,410))]
    for a,c in edges:b+=arrow([a,c],GRAY)
    for x,y in [(385,410),(595,410),(805,410),(1015,410),(385,548),(595,548),(805,548),(1015,548)]:
        b+=dot(x,y,13,'white')+f'<circle cx="{x}" cy="{y}" r="13" fill="white" stroke="{GRAY}" stroke-width="2"/>'
    qs=[(430,410),(755,410),(1015,502),(665,548),(385,480)]
    for i,(x,y) in enumerate(qs,1):
        b+=dot(x,y,19,TEAL)+text(x,y+7,str(i),22,color='white',weight=700,anchor='middle')
    b+=text(56,436,['b phục vụ lựa chọn','truy vấn có ích từ','nhiều vị trí có thể','phù hợp với lịch sử.'],26)
    b+=text(385,610,'Mạng đường có hướng và năm Q. Các Q có thể trùng nhau.',23,color=GRAY)
    return b


def local_service():
    b=text(56,142,'Truy hồi chung tại máy chủ, xếp hạng theo mục đích tại thiết bị',28,weight=700)
    b+=box(56,194,289,137,'5 tọa độ truy vấn Q\nMọi loại POI công khai\nL = 30 mỗi loại / Q',fill=PALE,stroke=TEAL,size=26)
    b+=arrow([(345,261),(408,261)],TEAL)
    b+=box(415,194,335,137,'Máy chủ\nTop-L gần nhất mỗi loại POI',fill='#fcf3ea',stroke=ORANGE,size=27)
    b+=arrow([(750,261),(810,261)],TEAL)
    b+=box(817,194,407,137,'Thiết bị\nGộp POI, loại trùng ID',fill=PALE,stroke=TEAL,size=27)
    b+=text(56,395,['GPS cục bộ','Mục đích riêng ψ'],28,color=BLUE,weight=700,lh=1.2)
    b+=text(56,479,['loại POI, mục đích,','bán kính và đích riêng'],25)
    b+=arrow([(315,441),(369,441)],BLUE)
    rows=[('Gần nhất','Khoảng cách đường'),('Nhanh nhất','Thời gian khi đường thông thoáng'),('Trong bán kính','Khoảng cách đường ≤ r'),('Ít vòng tới đích','Khoảng cách đi vòng qua POI')]
    for i,(name,score) in enumerate(rows):
        y=394+i*57
        b+=text(392,y,name,26,weight=700)+text(710,y,score,25)
        if i<3:b+=line(392,y+17,1114,y+17)
    b+=arrow([(1020,331),(1020,355),(656,355),(656,367)],TEAL)
    b+=text(56,627,'Tối đa 5 POI. Mục đích ψ giữ nguyên Q và lịch gửi, không gây truy vấn bổ sung.',27,color=TEAL,weight=700)
    return b


def scenario_scope():
    rows=[
        ('S1–S3','Vị trí, nơi dừng, đoạn đường','Geo-I/REM + cap; Q theo đường','Cận lý tưởng; benchmark lịch sử'),
        ('S4','Liên kết danh tính người / xe','Ngân sách chung qua phiên','Chưa bảo vệ account/IP; task tổng hợp'),
        ('S5–S6','Đường / đích tương lai','Lịch sử đã bảo vệ; không dùng tương lai','Mới thử bài toán hai lựa chọn'),
        ('S7','Nội dung / mục đích query','Request mọi category; xếp hạng local','Đúng khi cùng vị trí và lịch công khai'),
        ('S8','Suy luận từ người đồng hành','Đã bổ sung phép thử lịch sử','Chưa xác nhận cho Epoch8 / L30'),
        ('S9–S10','Điểm đầu / cuối hành trình','Nhiễu khi đọc; gửi ngay, không delay','Endpoint20 riêng; giờ mở/đóng còn lộ'),
    ]
    xs=[56,167,478,855]; widths=[99,295,361,368]
    headers=['Scenario','Cần bảo vệ','Thành phần / cơ chế','Phạm vi bằng chứng']
    b=''.join(text(x,165,h,24,weight=700) for x,h in zip(xs,headers))+line(56,185,1224,185,INK)
    for i,row in enumerate(rows):
        y=228+i*64
        # Explicit line breaks keep the scientific distinctions legible.
        wrap={
          0:[row[0],'Vị trí, nơi dừng,\nđoạn đường','Geo-I/REM + ngân sách;\nQ theo đường','Cận tọa độ lý tưởng;\nbenchmark lịch sử'],
          1:[row[0],'Liên kết phiên:\nngười / xe','Ngân sách chung\nqua phiên','Chưa ẩn account/IP;\nphép thử mô phỏng'],
          2:[row[0],'Cạnh kế tiếp /\nđích tương lai','Lịch sử đã bảo vệ;\nkhông dùng tương lai','Mới thử bài toán\nhai lựa chọn'],
          3:[row[0],'Nội dung /\nmục đích query','Mọi loại POI;\nxếp hạng tại thiết bị','Cùng trạng thái đã bảo vệ\nvà lịch công khai'],
          4:[row[0],'Suy luận từ\nngười đồng hành','Đánh giá suy luận\ntừ đồng hành','Thăm dò ở cấu hình riêng;\nchưa xác nhận cho L30'],
          5:[row[0],'Điểm đầu / cuối\nhành trình','Nhiễu khi đọc;\ntruy vấn gửi ngay','Endpoint20 riêng;\ngiờ mở/đóng còn lộ'],
        }[i]
        for j,(x,v) in enumerate(zip(xs,wrap)):b+=text(x,y,v,23,color=TEAL if j==0 else INK,weight=700 if j==0 else 400,lh=1.2)
        b+=line(56,y+38,1224,y+38)
    b+=text(56,637,'Bằng chứng phụ thuộc bài toán và cấu hình; chưa xác nhận bảo vệ đầy đủ S1–S10.',26,weight=700)
    return b


def pct(x, digits=2):
    return f'{x:.{digits}f}'.replace('.',',')


def fresh_chart(d):
    b=text(56,154,f"{pct(d['before'])}% → {pct(d['after'])}%",47,color=TEAL,weight=700)
    b+=text(617,151,f"+{pct(d['gain_pp'])} điểm phần trăm",31,weight=700)
    b+=text(617,195,f"CI95% [{pct(d['ci_pp'][0])}; {pct(d['ci_pp'][1])}]",24,color=GRAY)
    x0,x1=260,726
    for tick in [75,80,85,90,95,100]:
        x=x0+(tick-75)/25*(x1-x0)
        b+=line(x,255,x,504)+text(x,545,str(tick),22,color=GRAY,anchor='middle')
    for i,row in enumerate(d['purposes']):
        y=288+i*64
        b+=text(56,y+7,row['label'],24)
        a=x0+(row['before']-75)/25*(x1-x0);c=x0+(row['after']-75)/25*(x1-x0)
        b+=line(a,y,c,y,GRAY,4)+dot(a,y,8,GRAY)+dot(c,y,9,TEAL)
        b+=text(c+17,y+7,'+'+pct(row['gain_pp']),23,color=TEAL,weight=700)
    b+=text(482,586,'Recall@5 (%)',25,anchor='middle')
    b+=dot(274,225,7,GRAY)+text(290,232,'Geo-I/REM, L20',22,color=GRAY)
    b+=dot(515,225,7,TEAL)+text(531,232,'Geo-I/REM, L30',22,color=TEAL)
    b+=text(814,277,'Phản hồi JSON',29,weight=700)
    for i,(label,val,color) in enumerate([('L20',d['reply_before']/1e6,GRAY),('L30',d['reply_after']/1e6,TEAL)]):
        y=321+i*80
        b+=text(814,y+25,label,24,color=color)
        w=270*val/700
        b+=f'<rect x="884" y="{y}" width="{w}" height="32" fill="{color}"/>'
        b+=text(884,y+58,f'{pct(val)} MB',24,color=color)
    b+=text(814,516,'+'+pct(d['reply_growth_pct'])+'%',40,color=ORANGE,weight=700)
    b+=text(814,558,['Số truy vấn không đổi:',f"{d['requests']:,}".replace(',','.')+' truy vấn'],24)
    b+=text(56,630,'Giữ nguyên Q, Z và ngân sách; tăng chất lượng bằng phản hồi sâu hơn.',27,weight=700)
    return b


def matched_baseline(d):
    b=text(56,143,'Cùng ngân sách, K = 5, L = 20 và mạng đường; thay cơ chế REM / Planar',28,weight=700)
    xs=[56,440,690,930,1154]
    headers=['Phương pháp','Recall@5 ↑\ngần nhất','S5 · Accuracy ↓\ncạnh kế tiếp','S6 · Hit100 ↓\nđích tương lai','S6 · MAE ↑\n(m)']
    for j,(x,v) in enumerate(zip(xs,headers)):
        b+=text(x,215,v,24,weight=700,anchor='start' if j==0 else 'middle',lh=1.25)
    b+=line(56,262,1224,262,INK)
    names={'raw':'GPS chưa bảo vệ','rem_epoch8':'Geo-I / REM','planar_epoch8':'Geo-I / Planar'}
    for i,row in enumerate(d['rows']):
        y=314+i*77
        col=TEAL if row['method']=='rem_epoch8' else (BLUE if row['method']=='planar_epoch8' else GRAY)
        values=[names[row['method']],pct(100*row['current_recall5'])+'%',pct(100*row['S5_exact_candidate_edge_accuracy'],0)+'%',pct(100*row['S6_destination_hit100'],0)+'%',pct(row['S6_destination_mae_m'],0)]
        for j,(x,v) in enumerate(zip(xs,values)):
            b+=text(x,y,v,29,color=col,weight=700 if j==0 else 400,anchor='start' if j==0 else 'middle')
        b+=line(56,y+24,1224,y+24)
    b+=text(56,529,'Planar có Recall cao hơn; REM có MAE đích lớn hơn trong phép thử này.',27,weight=700)
    b+=text(56,573,'S5/S6: cùng quyết định giữa hai lựa chọn công khai, độ chính xác 50%.',25,color=ORANGE)
    b+=text(56,622,'Bộ suy luận chọn trên tập xác thực: Motion / CurveMean / CandidateTrees.',25,color=TEAL,weight=700)
    return b


def sensor_chart(d):
    b=text(56,147,'GPS tại thiết bị cập nhật mỗi 60 s; so sánh L20 và L30',30,weight=700)
    x0,x1,y0,y1=124,750,248,546
    lo,hi=82,94
    for tick in [82,85,88,91,94]:
        y=y1-(tick-lo)/(hi-lo)*(y1-y0)
        b+=line(x0,y,x1,y)+text(x0-15,y+7,str(tick),22,color=GRAY,anchor='end')
    for sigma in [0,5,15]:
        x=x0+sigma/15*(x1-x0)
        b+=text(x,y1+40,str(sigma),23,anchor='middle')
    b+=text(437,622,'σ mỗi trục (m)',25,anchor='middle')
    for key,label,color,dash in [('hold20','L20: giữ vị trí gần nhất',GRAY,''),('hold30','L30: giữ vị trí gần nhất',TEAL,''),('velocity30','L30: ngoại suy vị trí',BLUE,'7 5')]:
        points=[(x0+sigma/15*(x1-x0),y1-(v-lo)/(hi-lo)*(y1-y0)) for sigma,v in zip([0,5,15],d[key])]
        ps=' '.join(f'{x},{y}' for x,y in points)
        b+=f'<polyline points="{ps}" fill="none" stroke="{color}" stroke-width="3" stroke-dasharray="{dash}"/>'
        for x,y in points:b+=dot(x,y,7,color)
        legend_y={'hold20':183,'hold30':211,'velocity30':239}[key]
        b+=line(865,legend_y-7,900,legend_y-7,color,3,dash)+text(913,legend_y,label,24,color=color)
    b+=text(849,330,[f"σ = 0: giữ vị trí, L30 {pct(d['hold30'][0])}%",f"GPS chuẩn tại mỗi mốc: {pct(d['oracle30'])}%"],24)
    b+=text(849,438,['Ngoại suy giảm lỗi vị trí,','nhưng Recall thấp hơn giữ vị trí','ở cả ba mức nhiễu'],24,color=ORANGE,weight=700)
    b+=text(849,568,[f"L30 − L20, GPS thưa:",f"+{pct(d['gain_range_pp'][0])} đến +{pct(d['gain_range_pp'][1])} pp"],24,color=TEAL,weight=700)
    b+=text(124,211,'Recall@5 (%)',24,color=GRAY)
    return b


def dynamic_chart(d):
    b=text(56,147,'Trạng thái POI thay đổi mỗi 60 s; chỉ dùng thông tin còn hiệu lực',29,weight=700)
    x0,x1=330,1010
    for tick in [85,90,95,100]:
        x=x0+(tick-85)/15*(x1-x0)
        b+=line(x,210,x,510)+text(x,546,str(tick),23,color=GRAY,anchor='middle')
    rows=[('L20: phản hồi hiện tại',d['l20_current'],GRAY),('L20: gộp trong 60 s',d['l20_cached'],GRAY),('L30: phản hồi hiện tại',d['l30_current'],TEAL),('L30: gộp trong 60 s',d['l30_cached'],TEAL),('Toàn danh mục POI',d['bulk'],BLUE)]
    for i,(label,val,color) in enumerate(rows):
        y=234+i*64;x=x0+(val-85)/15*(x1-x0)
        b+=text(56,y+7,label,24,color=color)+dot(x,y,10,color)+text(x+20,y+7,pct(val)+'%',25,color=color,weight=700)
    b+=text(670,588,'Recall@5 (%)',25,anchor='middle')
    b+=text(56,634,'Tải toàn danh mục: Recall 100%, chi phí thấp hơn L30 trong bài toán này.',27,color=ORANGE,weight=700)
    return b


def endpoint_chart(d):
    b=text(56,146,'Endpoint20: tăng nhiễu ở mọi lần đọc được phép, gửi truy vấn ngay',29,weight=700)
    b+=text(56,193,'Đánh giá cấu hình L20 riêng; chưa suy kết quả cho mô hình L30',26,color=ORANGE,weight=700)
    maxval=max(d['s9_before'],d['s9_after'],d['s10_before'],d['s10_after'])*1.13
    x0,x1=308,840
    for val in [0,500,1000,1500]:
        if val>maxval:continue
        x=x0+val/maxval*(x1-x0)
        b+=line(x,270,x,540)+text(x,579,str(val),23,color=GRAY,anchor='middle')
    for i,(label,before,after) in enumerate([('S9 · điểm đầu',d['s9_before'],d['s9_after']),('S10 · điểm cuối',d['s10_before'],d['s10_after'])]):
        y=319+i*126;b+=text(56,y+17,label,28,weight=700)
        for dy,val,col in [(0,before,GRAY),(43,after,TEAL)]:
            w=val/maxval*(x1-x0)
            b+=f'<rect x="{x0}" y="{y+dy}" width="{w}" height="29" fill="{col}"/>'+text(x0+w+17,y+dy+23,pct(val,0)+' m',25,color=col)
    b+=text(955,308,['MAE: sai số vị trí','của bộ suy luận'],25)
    b+=text(955,404,[f"S10 tăng khoảng",pct(d['s10_after']-d['s10_before'],0)+' m'],28,color=TEAL,weight=700)
    b+=text(955,513,['S10 · Hit500 giảm',pct(d['s10_hit500_before'])+'% → '+pct(d['s10_hit500_after'])+'%'],25,color=TEAL)
    b+=line(310,240,346,240,GRAY,8)+text(360,247,'GeoI-Slack, L20',24,color=GRAY)
    b+=line(680,240,716,240,TEAL,8)+text(730,247,'Endpoint20',24,color=TEAL)
    b+=text(56,626,'S10: Hit100 = 0% ở cả hai; CI chênh Hit500 chạm 0. Giờ mở/đóng vẫn quan sát được.',24)
    return b


def main():
    content=json.loads((HERE/'slide_content.json').read_text())
    protection=json.loads((HERE/'sample_walkthrough.json').read_text())
    utility=json.loads((HERE/'utility_sample.json').read_text())
    endpoint=json.loads((HERE/'endpoint_sample.json').read_text())
    helpers=dict(text=text,line=line,dot=dot,arrow=arrow,box=box,ink=INK,teal=TEAL,blue=BLUE,gray=GRAY,orange=ORANGE)
    functions=[cover,architecture,private_read,belief_and_q,local_service,
               lambda:sample_visuals.protection_slide(protection,**helpers),
               lambda:sample_visuals.utility_slide(utility,protection,**helpers),
               lambda:sample_visuals.endpoint_slide(endpoint,**helpers),scenario_scope]
    functions += [lambda:fresh_chart(content['fresh']),lambda:matched_baseline(content['matched_baseline']),lambda:sensor_chart(content['sensor']),lambda:dynamic_chart(content['dynamic']),lambda:endpoint_chart(content['endpoint'])]
    rendered=[];notes=[]
    for i,(item,func) in enumerate(zip(content['slides'],functions),1):
        svg=base(item['title'],func(),item['footer'],i,len(functions)).replace('\u2013','-').replace('\u2014','-')
        # The projected deck contains only the slide canvas. Presenter prose
        # lives in presentation_script.md; source notes remain a reference.
        rendered.append(f'<section class="slide" id="slide-{i}" aria-label="Slide {i}">{svg}</section>')
        notes.append(f"SLIDE {i}: {item['title']}\n\n{item['notes']}\n\nNguồn:\n"+'\n'.join(item['sources'])+'\n')
    page='''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Geo-I / REM - Phương pháp và đánh giá thực nghiệm</title><style>
    *{box-sizing:border-box}body{margin:0;background:#e7e7e9;font-family:"STIX Two Text","Times New Roman",serif}.slide{display:none;height:100vh;align-items:center;justify-content:center;position:relative}.slide.active{display:flex}.slide svg{display:block;width:min(100vw,177.77778vh);height:auto;max-height:100vh;font-family:"STIX Two Text","Times New Roman",serif}@page{size:13.333333in 7.5in;margin:0}@media print{html,body{height:auto;overflow:visible;background:white}.slide,.slide.active{display:block;width:1280px;height:720px;overflow:hidden;break-after:page;page-break-after:always}.slide:last-of-type{break-after:auto;page-break-after:auto}.slide svg{width:1280px;height:720px;max-height:none}}
    </style><body>'''+''.join(rendered)+'''
    <script>(()=>{const slides=[...document.querySelectorAll('.slide')];let index=0;const paint=()=>{slides.forEach((s,i)=>s.classList.toggle('active',i===index));history.replaceState(null,'','#'+(index+1));document.title=(index+1)+' / '+slides.length+' · Geo-I / REM'};const hash=Number(location.hash.slice(1));if(hash>=1&&hash<=slides.length)index=hash-1;paint();addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' ','ArrowLeft','PageUp','Home','End'].includes(e.key))e.preventDefault();if(['ArrowRight','PageDown',' '].includes(e.key))index=Math.min(index+1,slides.length-1);if(['ArrowLeft','PageUp'].includes(e.key))index=Math.max(index-1,0);if(e.key==='Home')index=0;if(e.key==='End')index=slides.length-1;if(e.key.toLowerCase()==='f'){if(document.fullscreenElement)document.exitFullscreen();else document.documentElement.requestFullscreen().catch(()=>{})}paint()});})();</script></body></html>'''
    (HERE/'slides.html').write_text(page)
    (HERE/'speaker_notes.txt').write_text('\n\n'.join(notes))
    sources={}
    for item in content['slides']:
        for name in item['sources']:
            p=ROOT/name
            if not p.is_file():raise FileNotFoundError(name)
            sources[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    for data, key in [(protection,'source_pins_sha256'),(utility,'source_sha256'),(endpoint,'source_sha256')]:
        for name,expected in data[key].items():
            p=ROOT/name
            actual=hashlib.sha256(p.read_bytes()).hexdigest()
            if expected!=actual:raise ValueError('Sample source changed: '+name)
            sources[name]=expected
    (HERE/'source_pins.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n')
    print(f'Built {len(rendered)} HTML slides; {len(sources)} source pins. No scores regenerated.')


if __name__=='__main__':main()
