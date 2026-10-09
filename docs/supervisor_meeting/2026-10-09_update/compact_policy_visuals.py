"""Compact scientific layouts using the existing, frozen model evidence.

Only presentation text and layout are authored here. No sampler, attacker,
budget setting, query output, or benchmark is changed.
"""
from focus_visuals import table


def _table(headers, rows, widths, h, **kw):
    return table(headers, rows, widths, text=h['text'], line=h['line'],
                 ink=h['ink'], gray=h['gray'], teal=h['teal'], x=46,
                 size=21, header_size=20, **kw)[0]


def cover(content, **h):
    t, ln, box, arrow = h['text'], h['line'], h['box'], h['arrow']
    b=t(46,155,'Bảo vệ riêng tư vị trí bằng Geo-I / REM',42,color=h['teal'])
    b+=t(46,203,'Truy hồi POI theo mạng đường và mục đích riêng tại thiết bị',25)
    b+=ln(46,224,1234,224,h['teal'],1.3)
    b+=t(46,263,'Bài toán',23,weight=700)
    b+=t(46,297,'Giảm thông tin vị trí lộ qua chuỗi truy vấn, đồng thời giữ chất lượng tìm địa điểm.',22)
    b+=t(46,329,'GPS và mục đích dùng tại thiết bị. Máy chủ nhận tập tọa độ Q và yêu cầu POI chung.',22)
    labels=['GPS → Z nội bộ','Ước lượng b\nChọn K = 5 Q','Máy chủ\nL = 30 / loại / Q','Thiết bị\nGộp, trả tối đa 5 POI']
    xs=[46,350,654,958]
    for i,(x,label) in enumerate(zip(xs,labels)):
        b+=box(x,365,276,73,label,stroke=h['blue'] if i<2 else h['teal'],size=22)
        if i<3:b+=arrow([(x+276,401),(xs[i+1]-6,401)],h['teal'])
    b+=t(46,482,'Nội dung nghiên cứu',23,weight=700)
    b+=t(46,515,['S9–S10: chính sách đầu/cuối và ví dụ delay lịch sử.',
                     'S1–S8: cơ chế bảo vệ và phạm vi đã được đánh giá.'],22,lh=1.45)
    b+=t(650,515,['Benchmark: privacy, chất lượng POI và chi phí.',
                      'Kiến trúc hiện tại và thành phần kích hoạt theo thời gian.'],22,lh=1.45)
    b+=t(46,605,'K: số tọa độ truy vấn. L: số POI mỗi loại tại mỗi Q. k: số POI trả tại thiết bị.',21,color=h['gray'])
    b+=t(46,639,'Dữ liệu SUMO tổng hợp. Endpoint20 L20 và mô hình Epoch8/L30 có bằng chứng riêng.',20,color=h['gray'])
    return b


def endpoint_policy(audit, **h):
    t=h['text']
    rows=[
        ['Đầu phiên','Bỏ 60 s đầu trước khi gọi Geo-I.','Warmup = 0 s; gửi ngay.'],
        ['Trong phiên','Giữ Q 60 s rồi mới công bố.','Delay = 0 s; tăng nhiễu trên mọi lần đọc.'],
        ['Đóng phiên','Hủy Q chưa gửi khi quan sát tín hiệu đóng.','Không cần biết trước điểm cuối; giờ đóng vẫn lộ.'],
    ]
    b=_table(['Chính sách','Delay / holdback lịch sử','Endpoint20 đã đo sau đó'],rows,[162,510,516],h,
             y=144,row_h=42)
    b+=t(46,334,'Endpoint20: u = 0,0025/m, bằng 25% đối chứng L20 lịch sử (u = 0,01/m).',21,color=h['blue'])
    b+=t(46,365,'Ví dụ u701_00: thời điểm tạo Q và thời điểm công bố là hai mốc khác nhau',23,weight=700)
    saved=audit['historical_walkthrough']
    sample=saved['samples']['boundary']
    assert saved['session_id']=='u701_00' and saved['close_s']==384
    indexed={float(row['t']):row for row in sample['trigger_rows']}
    events=[]
    for time in [0,60,80,120,360,380,384]:
        r=indexed[time]
        if r['head_skipped']:
            gps='Bỏ đầu, chưa gọi Geo-I';queue='Chưa sinh Q'
        else:
            gps=('Đọc GPS, '+('tạo Z mới' if r['branch']=='fresh' else 'giữ Z cũ')) if r['private_read'] else 'Không đọc GPS mới'
            queue=f'Giữ Q({time})' if time!=384 else 'Thêm Q(384), đóng phiên'
        emitted=r['released_source_times']
        output=f'Q({int(emitted[0])}) gửi lúc {time} s' if emitted else ('Hủy Q chưa gửi' if time==384 else 'Chưa gửi')
        events.append(['0 / 20 / 40' if time==0 else str(time),gps,queue,output])
    b+=_table(['t (s)','GPS cho Geo-I','Hàng đợi tại thiết bị','Công bố'],events,[125,342,334,387],h,
              y=403,row_h=26)
    b+=t(46,641,'Đóng: hủy Q(340/360/380/384). Mẫu: 14/21 mốc gửi, 6 lần đọc, Recall 71,43%.',20,color=h['gray'])
    return b


def location_policy(content, **h):
    rows=[
        ['S1 · Định vị','REM lấy mẫu Z trên miền đường công khai cố định.\nChọn Q từ trạng thái đã bảo vệ; giữ Z nội bộ.',
         'Cận tọa độ Geo-I của sampler lý tưởng.\nMáy chủ nhận Q, không nhận GPS hay Z.'],
        ['S2 · Nơi dừng','Kiểm tra tái sử dụng có nhiễu; giữ hoặc tạo Z.\nMỗi quyết định đọc đều tính chi phí.',
         'Hạn chế thông tin tích lũy khi gửi lặp.\nGiữ Z vẫn chi ngân sách; không đọc mới thì chi 0.'],
        ['S3 · Quỹ đạo','b dùng lịch sử đã bảo vệ; chọn Q khả thi\ntrên mạng đường có hướng.',
         'Chuyển động Q hỗ trợ chất lượng POI.\nTính khả thi đường không tự bảo đảm privacy.'],
        ['S4 · Liên kết','Sổ bền vững và cap chung qua 8 phiên\nđược khai báo liên kết trong một epoch.',
         'Đổi chuyến không tự cấp lại ngân sách.\nChưa ẩn account, IP, thiết bị hay danh tính thật.'],
        ['S5–S6 · Tương lai','Chọn Q bằng tiền tố và lịch sử đã bảo vệ.\nKhông dùng phần tương lai để bảo vệ hiện tại.',
         'Native pilot: cạnh / đích giữa 2 lựa chọn.\nS5/S6 phụ thuộc cùng quyết định nhánh.'],
    ]
    b=_table(['Scenario','Thành phần xử lý','Ý nghĩa và phạm vi'],rows,[177,505,506],h,y=153,row_h=65)
    b+=h['math_text'](46,558,['Pr[M(x) ∈ A] ≤ exp(ε d',('E','sub'),'(x,x′)) Pr[M(x′) ∈ A]'],24)
    b+=h['text'](46,591,'Hiện tại: u = 0,00125/m, θ = 200 m, U = 23 đơn vị/phiên; cap epoch = 0,23/m.',21,color=h['blue'])
    b+=h['text'](46,626,'Ngân sách cộng theo các lần đọc và phiên liên kết. Benchmark S1–S6 có phạm vi riêng.',21,color=h['gray'])
    return b


def content_policy(content, **h):
    t,ln,box,arrow=h['text'],h['line'],h['box'],h['arrow']
    b=t(46,137,'S7 · Mục đích riêng, phản hồi chung',23,color=h['teal'],weight=700)
    for x,w,label in [(46,221,'5 Q\nMọi loại POI, L cố định'),(302,221,'Máy chủ\nTop-L / loại / Q'),(558,226,'Thiết bị\nGộp, loại trùng ID')]:
        b+=box(x,163,w,69,label,stroke=h['teal'],size=20)
    b+=arrow([(267,197),(296,197)],h['teal'])+arrow([(523,197),(552,197)],h['teal'])
    headers=['Xếp hạng tại thiết bị','Tiêu chí']
    rows=[['Gần nhất','Khoảng cách đường'],['Nhanh nhất','Thời gian đường thông thoáng'],['Trong bán kính','Khoảng cách đường ≤ r'],['Ít đi vòng tới đích','Độ dài đi vòng qua POI tới đích']]
    b+=_table(headers,rows,[270,468],h,y=276,row_h=34)
    b+=t(46,484,['GPS, loại POI, bán kính r và đích riêng chỉ dùng khi xếp hạng.',
                     'Cùng trạng thái đã bảo vệ và lịch công khai:',
                     'đổi mục đích giữ nguyên Q, payload và số request.'],21,lh=1.35)
    b+=t(46,590,['Phạm vi: loại bỏ kênh nội dung gửi trực tiếp.',
                     'Tương quan mục đích–tuyến đường, click và account còn là giới hạn.'],20,color=h['gray'],lh=1.4)
    b+=ln(817,127,817,642,h['gray'],1)
    b+=t(845,137,'S8 · Người đồng hành',23,color=h['teal'],weight=700)
    b+=t(845,181,['Attacker kết hợp luồng mục tiêu',
                      'với quan sát từ người đồng hành.'],21,lh=1.4)
    b+=t(845,268,'Diagnostic đã có',22,weight=700)
    b+=t(845,307,['Target-only, joint và người không liên quan.',
                      '3 nhóm; 24 mốc mục tiêu duy nhất.',
                      'Chấm vị trí của cùng người mục tiêu.'],20,lh=1.7)
    b+=t(845,451,'Phạm vi bằng chứng',22,weight=700)
    b+=t(845,490,['Cấu hình lịch sử, bank hữu hạn.',
                      'Chưa xác nhận cơ chế bảo vệ nhóm',
                      'cho Epoch8 / L30 hiện tại.'],21,color=h['orange'],lh=1.6)
    return b


def protocol(content, **h):
    t,ln=h['text'],h['line']
    b=t(46,140,'Quan sát và huấn luyện attacker',23,color=h['teal'],weight=700)
    b+=t(46,178,['Nhận Q, thời điểm gửi và metadata được phép.',
                     'GPS thật, Z và nhánh ngân sách chỉ thuộc evaluator.',
                     'Shadow transcript riêng cho từng phương pháp.',
                     'Chọn decoder trên tập chọn rồi khóa khi chấm.'],21,lh=1.6)
    b+=t(46,344,'Bank theo tác vụ',23,color=h['teal'],weight=700)
    b+=t(46,382,'kNN / Extra Trees / OLS / Motion / Viterbi',22,weight=700)
    b+=t(46,417,['Hit và MAE có thể chọn decoder khác nhau.',
                     'S4 liên kết người: kNN15.',
                     'S10 MAE: centroid_ols3_120s.'],21,lh=1.55)
    b+=ln(640,125,640,524,h['gray'],1)
    b+=t(675,140,'Thước đo và hướng đọc',23,color=h['teal'],weight=700)
    metrics=[['Hit100 / Hit500 ↓','Tỷ lệ đoán đúng trong 100 / 500 m.'],['MAE ↑','Sai số định vị trung bình, đơn vị m.'],['Recall@5 ↑','Chất lượng top-5 POI tại thiết bị.'],['S4: BA / AUC','Khả năng liên kết người hoặc xe.'],['S5: Accuracy ↓','Tỷ lệ dự đoán đúng cạnh kế tiếp.'],['Chi phí','Số request, JSON byte, delay công bố.']]
    for i,(label,value) in enumerate(metrics):
        y=182+i*52
        b+=t(675,y,label,21,weight=700)+t(675,y+25,value,20,color=h['gray'])
    b+=ln(46,542,1234,542,h['ink'],1.2)
    b+=t(46,574,'Nguyên tắc so sánh',23,weight=700)
    b+=t(46,608,'Tách cohort và cấu hình. GPS thật là đối chứng dương; adaptation khác tái lập paper nguyên bản.',21)
    b+=t(46,640,'Số liệu hiện có là mô phỏng. JSON ở lớp ứng dụng; độ trễ mạng và năng lượng chưa đo.',20,color=h['gray'])
    return b
