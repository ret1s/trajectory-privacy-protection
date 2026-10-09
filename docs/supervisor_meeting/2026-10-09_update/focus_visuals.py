"""Scientific slide layouts for scenario mechanisms and frozen evidence.

All numeric values and movement coordinates are passed from source-pinned JSON.
This module renders evidence; it never runs a mechanism or attacker.
"""
import html


def table(headers, rows, widths, *, text, line, ink, gray, teal,
          x=56, y=185, row_h=67, size=25, header_size=23, colors=None):
    positions=[x]
    for width in widths[:-1]:
        positions.append(positions[-1]+width)
    header_lines=max(len(str(value).split('\n')) for value in headers)
    body_start=y+(header_lines-1)*header_size*1.2+31
    b=line(x,y-27,x+sum(widths),y-27,ink,1.6)
    for j,(pos,value) in enumerate(zip(positions,headers)):
        b+=text(pos+8,y,value,header_size,weight=700,lh=1.2)
    b+=line(x,body_start-13,x+sum(widths),body_start-13,ink,1.2)
    for i,row in enumerate(rows):
        color=colors[i] if colors else ink
        for j,(pos,value) in enumerate(zip(positions,row)):
            b+=text(pos+8,body_start+22+i*row_h,value,size,color=color,
                    weight=700 if j==0 else 400,lh=1.15)
    bottom=body_start+len(rows)*row_h+5
    b+=line(x,bottom,x+sum(widths),bottom,ink,1.6)
    return b,bottom


def topic(x,y,title,lines, *, text,line,ink,teal,width=544,size=28):
    b=text(x,y,title,29,color=teal,weight=700)
    b+=line(x,y+18,x+width,y+18,ink,1.2)
    yy=y+64
    for line_value in lines:
        b+=text(x,yy,line_value,size,lh=1.24)
        yy+=(len(line_value.split('\n'))*size*1.24)+22
    return b


def endpoint_mechanisms(*,text,line,box,arrow,ink,teal,blue,gray,orange,**kw):
    b=topic(56,151,'Delay / holdback: cấu hình trước',[
        'Đầu phiên: chưa tạo/gửi truy vấn\ntrong cửa sổ công khai 60 s.',
        'Sau đó: giữ Q trong hàng đợi 60 s\nrồi mới công bố.',
        'Đóng phiên: hủy Q còn trong hàng đợi.\nGiảm quan sát trực tiếp phần cuối.',
    ],text=text,line=line,ink=ink,teal=teal,size=27)
    b+=topic(674,151,'Endpoint20: cấu hình đã đo sau đó',[
        'Warmup = delay = 0 s.\nGửi ngay tại các mốc đầu và cuối.',
        'Ngân sách mỗi lần đọc bằng 25%\nđối chứng L20 lịch sử, tăng mức nhiễu.',
        'Áp dụng cho mọi lần đọc được phép.\nKhông cần biết trước thời điểm kết thúc.',
    ],text=text,line=line,ink=ink,teal=teal,width=550,size=27)
    b+=text(56,622,'Delay làm giảm dịch vụ tức thời; nhiễu bảo vệ tọa độ, còn thời điểm mở/đóng vẫn lộ.',27,color=orange,weight=700)
    return b


def location_mechanisms(*,text,line,math_text,ink,teal,blue,gray,orange,**kw):
    rows=[
        ('S1 · Một lần gửi','REM tạo Z trên miền đường công khai;\nmáy chủ chỉ nhận tập Q từ trạng thái đã bảo vệ.'),
        ('S2 · Nơi dừng','Phép thử tái sử dụng có nhiễu và ngân sách hữu hạn\nhạn chế công bố lặp lại từ GPS tại nơi dừng.'),
        ('S3 · Đường đã đi','b chỉ dùng lịch sử đã bảo vệ; năm Q chuyển động\nkhả thi trên đường có hướng, không đọc tương lai.'),
    ]
    b,bottom=table(['Kịch bản','Cơ chế'],rows,[315,853],text=text,line=line,ink=ink,gray=gray,teal=teal,
                   y=170,row_h=110,size=27,header_size=25)
    b+=math_text(56,580,['Pr[M(x) ∈ A] ≤ exp(ε d',('E','sub'),'(x,x′)) Pr[M(x′) ∈ A]'],30)
    b+=text(56,628,'Geo-I: cận cho cơ chế lý tưởng. Q đi được theo đường không tự tạo bảo đảm riêng tư.',25,color=gray)
    return b


def linkage_future(*,text,line,box,arrow,ink,teal,blue,gray,orange,**kw):
    b=topic(56,152,'S4 · Liên kết người / phương tiện',[
        'Một sổ ngân sách dùng chung qua\ntám phiên công khai trong epoch.',
        'Cộng chi phí các phiên được liên kết;\nkhông tự cấp lại ngân sách khi đổi chuyến.',
        'Chưa ẩn tài khoản, IP, thiết bị\nhoặc danh tính ngoài tọa độ.',
    ],text=text,line=line,ink=ink,teal=teal,size=27)
    b+=topic(674,152,'S5–S6 · Đường và đích tương lai',[
        'Bộ chọn Q dùng tiền tố và lịch sử\nđã bảo vệ; không dùng đoạn tương lai.',
        'Phép thử native: cạnh tiếp theo và đích\nđược chọn từ hai ứng viên công khai.',
        'Đã đo tín hiệu suy luận; chưa bao phủ\ndự đoán mọi cạnh / đích thực tế.',
    ],text=text,line=line,ink=ink,teal=teal,width=550,size=27)
    b+=text(56,625,'Ngân sách liên phiên xử lý tích lũy vị trí; các mục tiêu danh tính vẫn chỉ được bảo vệ một phần.',26,color=orange)
    return b


def purpose_mechanism(*,text,line,box,arrow,math_text,ink,teal,blue,gray,orange,**kw):
    b=box(56,155,286,104,'K = 5 tọa độ Q\nMọi loại POI, L cố định',stroke=teal,size=27)
    b+=arrow([(342,207),(402,207)],teal)
    b+=box(410,155,333,104,'Máy chủ\nTop-L mỗi loại POI / Q',stroke=gray,size=27)
    b+=arrow([(743,207),(802,207)],teal)
    b+=box(810,155,414,104,'Thiết bị\nGộp và loại trùng POI',stroke=teal,size=27)
    b+=text(56,325,'Mục đích ψ ở thiết bị',29,color=blue,weight=700)
    b+=text(56,371,['GPS, loại POI, bán kính,','đích riêng'],27)
    b+=arrow([(334,373),(389,373)],blue)
    rows=[('Gần nhất','Khoảng cách đường'),('Nhanh nhất','Thời gian đường thông thoáng'),('Trong bán kính','Chỉ giữ POI cách đường ≤ r'),('Ít đi vòng','Độ dài đi vòng qua POI tới đích')]
    for i,(label,score) in enumerate(rows):
        y=326+59*i
        b+=text(410,y,label,27,weight=700)+text(714,y,score,26)
        if i<3:b+=line(410,y+18,1224,y+18)
    b+=arrow([(1017,259),(1017,284),(583,284),(583,303)],teal)
    b+=text(56,588,'Cùng trạng thái đã bảo vệ và lịch công khai: đổi ψ giữ nguyên Q và payload.',27,weight=700)
    b+=text(56,632,'Bảo vệ kênh nội dung trực tiếp; tương quan ý định–tuyến đường và click còn là giới hạn.',25,color=gray)
    return b


def companion_mechanism(*,text,line,box,arrow,ink,teal,blue,gray,orange,**kw):
    b=text(56,155,'S8: quan sát người đồng hành có thể bổ sung thông tin về vị trí mục tiêu',28,weight=700)
    b+=box(56,211,328,113,'Người mục tiêu\nGPS → đầu ra đã bảo vệ',stroke=blue,size=28)
    b+=box(56,397,328,113,'Người đồng hành\nCùng không gian / thời điểm',stroke=gray,size=27)
    b+=arrow([(384,267),(476,267)],blue)
    b+=arrow([(384,453),(443,453),(443,300),(476,300)],gray)
    b+=box(486,225,286,97,'Bộ suy luận joint\nHai chuỗi quan sát',stroke=teal,size=27)
    b+=arrow([(772,272),(836,272)],teal)
    b+=box(844,225,380,97,'Ước lượng vị trí\ncủa người mục tiêu',stroke=teal,size=27)
    b+=text(486,397,['Đã có phép thử lịch sử: target-only,','joint và người không liên quan.'],28)
    b+=text(486,487,['Chưa có cơ chế nhóm được xác nhận','cho mô hình Epoch8 / L30 hiện tại.'],28,color=orange,weight=700)
    b+=text(56,628,'Ba nhóm kiểm tra, 24 mốc mục tiêu duy nhất; không kết luận đã giải quyết đầy đủ S8.',26,color=gray)
    return b


def evaluation_protocol(*,text,line,ink,teal,gray,blue,orange,**kw):
    b=topic(56,151,'Đối thủ và quy tắc đánh giá',[
        'Nhìn Q, thời điểm và siêu dữ liệu\nđược phép; không nhận GPS thật hay Z.',
        'Fit trên shadow transcript của từng\nphương pháp; chọn attacker trên tập chọn.',
        'Các cột Hit / MAE có thể chọn\nnhững bộ suy luận khác nhau.',
    ],text=text,line=line,ink=ink,teal=teal,size=27)
    b= b+topic(674,151,'Các thước đo cần đọc cùng nhau',[
        'Hit100 / Hit500 ↓: tỷ lệ suy đúng\ntrong bán kính 100 / 500 m.',
        'MAE ↑: sai số vị trí trung bình (m).\nRecall@5 ↑: chất lượng câu trả lời POI.',
        'S4: BA / AUC. S5: đúng cạnh.\nChi phí: số truy vấn, JSON byte, độ trễ.',
    ],text=text,line=line,ink=ink,teal=teal,width=550,size=27)
    b+=text(56,554,'Bank theo phép thử: kNN / Extra Trees / OLS / Motion / Viterbi',25,weight=700)
    b+=text(56,593,'GPS thật là đối chứng dương; các cấu hình dùng shadow transcript riêng.',25,color=gray)
    b+=text(56,632,'Tách rõ dữ liệu và cấu hình; không gộp điểm số thành một mô hình thắng mọi kịch bản.',25,color=orange)
    return b


def changes_table(*,text,line,ink,teal,gray,blue,orange,**kw):
    rows=[
        ('Nền tảng','Geo-I/REM, tái sử dụng Z','Giữ nguyên nguyên lý'),
        ('Ngân sách','Hữu hạn trong từng phiên','Sổ bền vững, cap chung 8 phiên'),
        ('Q theo đường','Ước lượng b, độ phủ, slack','Giữ; mạng native và kiểm tra nguồn'),
        ('Truy hồi POI','L10; xếp hạng gần nhất','L30; 4 mục đích tại thiết bị'),
        ('S9 / S10','Warmup + delay + hủy hàng đợi','Gửi ngay; nhánh Endpoint20 (L20)\nđược đo riêng'),
        ('Đánh giá','Ví dụ trên mạng minh họa','Tuyến mới, bank tấn công,\nGPS thưa / POI đổi trạng thái'),
    ]
    b,bottom=table(['Thành phần','Trước','Hiện tại'],rows,[250,407,511],text=text,line=line,ink=ink,gray=gray,teal=teal,
                   y=166,row_h=65,size=25,header_size=25)
    b+=text(56,626,'Bổ sung ngân sách liên phiên và xử lý mục đích; Geo-I vẫn là nền tảng.',26,color=teal)
    return b
