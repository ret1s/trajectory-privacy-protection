"""Eight-slide mechanism-focused summary; all readouts remain frozen."""
from focus_visuals import table
from sample_visuals import GPS_COLOR, map_svg


def n(v, digits=2):
    return f'{v:.{digits}f}'.replace('.', ',')


def tab(headers, rows, widths, h, *, x=46, y=164, row_h=58, size=21):
    return table(headers, rows, widths, x=x, y=y, row_h=row_h,
                 size=size, header_size=21,
                 **{key:h[key] for key in ['text','line','ink','gray','teal']})[0]


def overview(content, **h):
    t=h['text']
    b=t(46,145,'Hoàn thiện phương pháp Geo-I / REM',40,color=h['teal'])
    b+=t(46,193,'Các cơ chế bổ sung để bảo vệ nhiều scenario và giữ chất lượng dịch vụ',24)
    rows=[
        ['Ngân sách liên phiên','Từ cap từng chuyến sang cap chung 8 phiên.','Hạn chế tích lũy tọa độ; hỗ trợ S4.'],
        ['Mục đích truy vấn','4 cách xếp hạng tại thiết bị; request chung.','Bảo vệ kênh nội dung trực tiếp S7.'],
        ['Ứng viên POI','Phản hồi L30; vẫn 5 Q và top-5 tại thiết bị.','Phục hồi utility khi vị trí đã làm nhiễu.'],
        ['Điểm đầu / cuối','Đối chiếu delay cũ với Endpoint20 gửi ngay.','Thử nghiệm bảo vệ S9/S10 riêng.'],
    ]
    b+=tab(['Thay đổi','Cơ chế','Mục tiêu'],rows,[230,525,433],h,y=256,row_h=66)
    b+=t(46,601,'Giữ Geo-I/REM, phép thử tái sử dụng Z, ước lượng b và chọn Q theo mạng đường.',22,color=h['teal'],weight=700)
    b+=t(46,639,'S1–S6 còn phạm vi có điều kiện; S8 mới có diagnostic, chưa xác nhận bảo vệ nhóm.',20,color=h['gray'])
    return b


def architecture_comparison(content, **h):
    """Aligned historical/current flows; the external service is outside both device frames."""
    t=h['text'];b=''
    versions=[
        (46,'Bản trước (26/09–03/10)',[
            '1. GPS: cap phiên + lịch + Geo-I/REM\nCphiên = 0,23/m; thử giữ / tạo Z',
            '2. Ước lượng b → chọn 5 Q theo đường\nLịch sử bảo vệ; phủ POI / tiến độ / slack',
            '3. Chỉ gửi Q; yêu cầu mọi loại POI\nL10 / loại / Q; warmup / delay tùy chọn',
            '4. Hợp / bỏ trùng POI → top-5 local\nXếp hạng gần nhất theo GPS',
        ]),
        (680,'Hiện tại (Epoch8/L30)',[
            '1. GPS: cap 8 phiên + lịch + Geo-I/REM\nCepoch = 0,23/m; Cphiên = 0,02875/m',
            '2. Ước lượng b → chọn 5 Q theo đường\nLịch sử bảo vệ; phủ POI / tiến độ / slack',
            '3. Chỉ gửi Q; yêu cầu mọi loại POI\nL30 / loại / Q; gửi ngay',
            '4. Hợp / bỏ trùng POI → top-5 local\n4 mục đích theo GPS + nhu cầu riêng ψ',
        ]),
    ]
    for index,(x,title,labels) in enumerate(versions):
        b+=t(x,128,title,25,color=h['teal'],weight=700)
        b+=t(x,158,'Đầu vào: GPS + bản đồ / POI / lịch công khai',20,color=h['blue'])
        b+=h['box'](x,180,554,344,stroke=h['teal'])
        b+=t(x+16,201,'PHƯƠNG PHÁP TẠI THIẾT BỊ',18,color=h['teal'],weight=700)
        b+=h['arrow']([(x+480,164),(x+480,218)],h['blue'])
        for layer,label in enumerate(labels):
            y=218+layer*76
            color=h['teal'] if index==1 and layer!=1 else h['gray']
            b+=h['box'](x+18,y,518,60,label,stroke=color,size=20,bold=index==1 and layer!=1)
            if layer<2:
                b+=h['arrow']([(x+277,y+60),(x+277,y+76)],h['gray'])
        # Query leaves the frame, replies enter local aggregation; no direct GPS to server.
        b+=h['arrow']([(x+18,400),(x+8,400),(x+8,583),(x+18,583)],h['blue'])
        b+=h['box'](x+18,554,252,59,'MÁY CHỦ (ngoài mô hình)\nTrả top-L POI / loại / Q',stroke=h['blue'],size=19)
        b+=h['arrow']([(x+270,583),(x+278,583),(x+278,537),(x+544,537),(x+544,476),(x+536,476)],h['blue'])
        b+=h['arrow']([(x+405,506),(x+405,554)],h['teal'])
        b+=h['box'](x+302,554,234,59,'ĐẦU RA RIÊNG\n≤5 POI cho người dùng',stroke=h['teal'],size=20)
    b+=t(46,638,'Tô xanh ở bản hiện tại: phần thay đổi. Giữ Geo-I/REM, ước lượng b và bộ chọn Q theo mạng đường.',20,color=h['teal'])
    b+=t(46,659,'GPS cho Geo-I đọc sau kiểm tra; Z và nhu cầu ψ giữ tại thiết bị. Endpoint20 là nhánh L20 riêng (slide 8).',18,color=h['gray'])
    return b


def budget(data, **h):
    t=h['text']
    b=t(46,139,'Cơ chế mới: sổ ngân sách bền vững qua nhiều phiên',24,color=h['teal'],weight=700)
    rows=[['Cận mỗi phiên','0,23/m','0,02875/m'],
          ['Tổng cận cho 8 phiên','1,84/m nếu cấp lại từng phiên','0,23/m dùng chung'],
          ['Một đơn vị u','0,01/m','0,00125/m']]
    b+=tab(['Phân bổ','Trước','Epoch8 hiện tại'],rows,[330,429,429],h,y=190,row_h=42)
    b+=t(46,391,'Trước khi đọc GPS',23,weight=700)
    labels=[(46,288,'Còn slot phiên?'),(367,346,'Đủ cap dự toán và cách ≥60 s?'),(746,488,'Đọc GPS → giữ / tạo Z bằng Geo-I')]
    for i,(x,w,label) in enumerate(labels):
        b+=h['box'](x,418,w,65,label,stroke=h['blue'],size=21)
        if i<2:b+=h['arrow']([(x+w,450),(labels[i+1][0]-6,450)],h['blue'])
    b+=t(46,530,'Chi phí thực tế: lần đầu 1u; đọc rồi giữ Z 1u; thử rồi tạo mới 2u; không đọc 0u.',22)
    b+=t(46,568,'Không đủ cap đọc: phiên đã nhận vẫn chọn Q từ lịch sử đã bảo vệ. Hết slot: không đọc, không gửi Q.',20,color=h['gray'])
    b+=t(46,608,'Hỗ trợ S2/S3 khi quan sát lặp và S4 khi gom nhiều chuyến; đổi phiên không tự cấp lại cap.',21,color=h['teal'],weight=700)
    b+=t(46,643,'Cap tọa độ không tự ẩn tài khoản, IP hoặc toàn bộ đặc trưng liên kết người/xe.',20,color=h['gray'])
    return b


def service(data, **h):
    t=h['text']
    labels=[(46,300,'5 Q; mọi loại POI\nL30 / loại / Q'),(383,336,'Máy chủ\nTrả POI ứng viên'),(756,478,'Thiết bị\nGộp, lọc và xếp hạng theo GPS + ψ')]
    b=''
    for i,(x,w,label) in enumerate(labels):
        b+=h['box'](x,133,w,74,label,stroke=h['teal'],size=22)
        if i<2:b+=h['arrow']([(x+w,170),(labels[i+1][0]-6,170)],h['teal'])
    rows=[['Gần nhất','Khoảng cách đường'],['Nhanh nhất','Thời gian đường thông thoáng'],
          ['Trong bán kính','Khoảng cách đường ≤ r'],['Ít đi vòng tới đích','Độ dài đi vòng qua POI tới đích']]
    b+=tab(['Mục đích riêng','Xử lý tại thiết bị'],rows,[245,466],h,y=269,row_h=43)
    category=data['illustration_category']
    result={purpose:r[category]['answer'] for purpose,r in data['local_results'].items()}
    b+=t(805,253,'Ví dụ cùng Q tại 60 s',23,color=h['blue'],weight=700)
    b+=t(805,289,f"{data['merged']['raw_record_count']} bản ghi → {data['merged']['unique_count']} POI duy nhất",21)
    b+=t(805,340,'Gần nhất: '+', '.join(r['display_alias'] for r in result['nearest_distance']),20,color=h['blue'])
    b+=t(805,390,'Bán kính 1.000 m: chỉ '+result['within_radius'][0]['display_alias'],21,color=h['teal'])
    b+=t(805,438,'Ít đi vòng: '+', '.join(r['display_alias'] for r in result['minimum_detour']),20)
    b+=t(46,525,'Mới: tổng quát hóa 4 mục đích và tăng độ sâu phản hồi. Yêu cầu mọi loại POI đã có từ trước.',21,weight=700)
    b+=t(46,563,'S7: cùng trạng thái đã bảo vệ và lịch công khai, đổi ψ không đổi Q, payload hay số request.',21,color=h['teal'])
    b+=t(46,602,'GPS, loại POI, bán kính và đích dùng local. Lấy thêm ứng viên giúp utility, nhưng tăng byte phản hồi.',21)
    b+=t(46,639,'Chưa loại bỏ suy luận ý định từ tuyến đường / click. Fastest chưa đo ùn tắc; detour cần đích đã biết.',20,color=h['gray'])
    return b


def endpoints(audit, **h):
    t=h['text']
    rows=[['Đầu phiên','Bỏ 60 s đầu trước Geo-I.','Warmup = 0 s, gửi ngay.'],
          ['Trong phiên','Giữ Q 60 s trước công bố.','Nhiễu mạnh hơn ở mọi lần đọc được phép.'],
          ['Khi đóng','Hủy Q còn trong hàng đợi.','Không cần biết trước lần đọc nào là cuối.']]
    b=tab(['Chính sách','Delay lịch sử','Endpoint20 đã đo riêng'],rows,[180,465,543],h,y=157,row_h=50)
    b+=t(46,377,'Endpoint20: u = 0,0025/m, bằng 25% đối chứng L20 lịch sử (u = 0,01/m).',22,color=h['blue'])
    b+=t(46,421,'Delay giảm dữ liệu trực tiếp ở hai biên, nhưng chậm / mất câu trả lời. Nhiễu bảo vệ tọa độ khi gửi ngay.',21)
    sample=audit['historical_walkthrough']['samples']['boundary']
    assert sample['cancelled_source_times_s']==[340.0,360.0,380.0,384.0]
    b+=t(46,471,'Ví dụ delay đã lưu',23,weight=700)
    b+=t(46,510,'Q tạo tại 60 s → gửi tại 120 s. Q tạo tại 320 s → gửi tại 380 s.',22)
    b+=t(46,549,'Đóng tại 384 s: hủy Q(340), Q(360), Q(380), Q(384); không gửi bù phần cuối.',21,color=h['teal'])
    b+=t(46,594,'Epoch8/L30 hiện tại cũng gửi ngay. Benchmark Endpoint20 thuộc nhánh L20 riêng.',21,weight=700)
    b+=t(46,634,'S9/S10 mới được xử lý một phần: thời điểm mở/đóng vẫn lộ; chưa xác nhận endpoint riêng cho L30.',20,color=h['gray'])
    return b


def walkthrough(data, **h):
    t=h['text']
    events=data['main_sample']['events']
    rows=[]
    for time in [0,20,60,120,600]:
        r=next(e for e in events if e['t_s']==time);p=r['protection']
        rows.append([str(time),'Có' if p['GPS_read'] else 'Không',
                     'Tạo Z' if p['branch']=='fresh' else 'Giữ Z',
                     'Cập nhật' if p['GPS_read'] else 'Dự đoán',str(p['spent_after_units'])+'/23'])
    b=t(46,138,'Một chuyến TRAIN đã lưu: mỗi mốc gửi Q đều lấy POI và xếp hạng local',23,weight=700)
    b+=tab(['t (s)','GPS\nGeo-I','Z','b','Đã chi'],rows,[75,110,110,150,130],h,y=189,row_h=41)
    r=next(e for e in events if e['t_s']==120)
    points=[r['gps_xy'],r['Z_xy'],*r['Q_xy'],*r['belief_top_xy']]
    marks=[{'xy':r['gps_xy'],'shape':'cross','color':GPS_COLOR,'label':'GPS'},
           {'xy':r['Z_xy'],'shape':'diamond','color':h['blue'],'label':'Z'}]
    marks += [{'xy':pos,'color':h['teal'],'label':str(i+1),'radius':7} for i,pos in enumerate(r['Q_xy'])]
    def maptext(x,y,value,size=23,**kw):return t(x,y,value,size*.82,**kw)
    b+=t(682,182,'t = 120 s: GPS, Z và năm Q',22,color=h['blue'],weight=700)
    b+=map_svg('summary-trip',(682,209,552,287),data['map_main']['road_segments_xy'],points,marks,
               text=maptext,line=h['line'],dot=h['dot'],belief=[{'xy':c['xy_m'],'weight':c['weight']} for c in r['belief']['top_weights']])
    b+=t(46,500,'20 s: không đọc GPS mới, b chỉ dự đoán; Q vẫn có thể đổi theo đường. Không chi thêm ngân sách.',20)
    reuse=data['test_reuse_inset']['events'][-1]['protection']
    b+=t(46,540,f"Nhánh khác, phiên 6 tại 60 s: đọc rồi giữ Z, chi {reuse['cost_units']}u; b vẫn cập nhật từ quan sát giữ.",21,color=h['teal'])
    b+=t(46,580,'Đọc rồi giữ Z khác không đọc mới. Z giữ nội bộ; chỉ Q gửi server; GPS local chọn POI sau phản hồi.',21)
    b+=t(46,620,'Tổng 21/23 tại 600 s đã tính cả các lần đọc 360/420/480/540 s; mẫu chưa hết ngân sách.',20,color=h['gray'])
    b+=t(46,650,'Màu cam: một phần trọng số b. Nhãn Q là ký hiệu nội bộ; ô giữ Z thuộc phiên khác, không ghép vào timeline.',18,color=h['gray'])
    return b


def evidence(data, **h):
    t=h['text'];utility=data['current_fresh_four_purpose_utility']
    primary=next(r for r in utility['rows'] if r['primary']);cost=utility['cost']
    b=t(46,140,'Utility hiện tại: L20 → L30 trên cùng Q',23,color=h['teal'],weight=700)
    b+=t(46,175,'24 nhóm mới × 3 lượt nhiễu × 8 chuyến',21)
    b+=tab(['Cấu hình','Recall trung bình\n4 mục đích (%)','Phản hồi JSON\n(MB)'],
           [['L20',n(100*primary['L20_recall']),n(cost['L20']['reply_bytes']/1e6)],
            ['L30',n(100*primary['L30_recall']),n(cost['L30']['reply_bytes']/1e6)]],
           [130,245,225],h,y=224,row_h=42)
    lo,hi=primary['paired95_family_bootstrap_gain_pp']
    b+=t(46,404,f"+{n(primary['gain_pp'])} điểm %; CI 95% [{n(lo)}; {n(hi)}].",21,color=h['teal'],weight=700)
    b+=t(46,442,f"Byte phản hồi +{n(cost['reply_growth_percent'])}%; giữ nguyên Q, Z, lịch và cap.",20)
    ep=data['S9_S10_full_bank_endpoint'];index={(r['scenario'],r['method']):r for r in ep['rows']}
    b+=t(690,140,'Endpoint20: phép thử L20 riêng',23,color=h['teal'],weight=700)
    b+=t(690,175,'28 nhóm đã khảo sát; 112 quan sát / tác vụ',21)
    rows=[]
    for scenario in ['S9','S10']:
        rows.append([scenario,n(index[scenario,'scale100_L20']['mae_m'],0),n(index[scenario,'scale025_L20']['mae_m'],0)])
    b+=tab(['MAE ↑ (m)','Geo-I L20','Endpoint20'],rows,[150,180,214],h,x=690,y=224,row_h=42)
    before=index['S10','scale100_L20'];after=index['S10','scale025_L20']
    b+=t(690,404,f"Recall: {n(100*before['recall5_same_frozen_service'])}% → {n(100*after['recall5_same_frozen_service'])}%.",21)
    b+=t(690,442,'S10 Hit100 cùng 0%; CI ΔHit500 chạm 0.',20,color=h['gray'])
    b+=h['line'](46,482,1234,482,h['ink'],1.2)
    b+=t(46,519,'S5/S6 pilot: GPS thật đúng 100%; REM và Planar cùng 50% trong bài toán hai lựa chọn.',21)
    b+=t(46,553,'Planar có Recall 97,00%, REM 94,69%: chưa có ưu thế đồng loạt trên mọi đối chứng.',20,color=h['gray'])
    b+=t(46,598,'Kết luận đo được: L30 tăng chất lượng với chi phí byte cao hơn; Endpoint20 tăng sai số suy biên trong bank đã thử.',20,weight=700)
    b+=t(46,636,'Các phép thử dùng cấu hình / cohort riêng; dữ liệu SUMO. Không ghép thành một mô hình thắng toàn bộ S1–S10.',20,color=h['gray'])
    return b


def scope(content, **h):
    rows=[['S1–S3','Geo-I/REM, noisy reuse, b/Q theo đường.','Bảo đảm lý tưởng có điều kiện; đánh giá lịch sử.'],
          ['S4','Cap chung hạn chế tích lũy qua phiên.','Chưa ẩn account/IP và toàn bộ danh tính.'],
          ['S5–S6','Đánh giá cạnh / đích tương lai trên prefix.','Pilot hai lựa chọn; cần mở rộng nhiệm vụ.'],
          ['S7','Request bất biến khi đổi mục đích local.','Đúng cho cùng trạng thái và lịch công khai.'],
          ['S8','Đã có diagnostic người đồng hành.','Chưa xác nhận cơ chế bảo vệ nhóm hiện tại.'],
          ['S9–S10','Delay lịch sử; Endpoint20 được đo riêng.','Giờ mở/đóng còn lộ; cần xác nhận L30.']]
    b=tab(['Scenario','Đã có','Phạm vi còn thiếu'],rows,[150,507,531],h,y=154,row_h=51,size=20)
    b+=h['text'](46,536,'Đóng góp hiện tại: ngân sách liên phiên, nhiều mục đích local và phục hồi utility bằng phản hồi sâu hơn.',21,color=h['teal'],weight=700)
    b+=h['text'](46,580,'Tiếp theo: chấm privacy trên cùng Epoch8/L30; ưu tiên S8 và liên kết danh tính; kiểm tra chi phí triển khai.',21)
    b+=h['text'](46,622,'Chứng minh hợp thành ngân sách và hậu xử lý thuộc mô hình lý tưởng; chưa chứng nhận sampler số thực.',20,color=h['gray'])
    return b
