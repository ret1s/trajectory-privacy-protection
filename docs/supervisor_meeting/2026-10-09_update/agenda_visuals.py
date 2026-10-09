"""Eight-page scientific presentation in the supervisor's discussion order.

Render retained evidence only; no model sampling or score calculation.
"""
from summary_visuals import tab, n
from sample_visuals import GPS_COLOR, map_svg


def query_content(content, **h):
    t=h['text']
    b=t(46,67,'S7: bảo vệ nội dung và mục đích truy vấn',32,color=h['teal'])
    b+=h['line'](46,89,1234,89,h['teal'],1.1)
    b+=t(46,135,'Tách nhu cầu riêng của người dùng khỏi yêu cầu gửi tới máy chủ',24,weight=700)
    b+=h['box'](46,166,510,282,stroke=h['teal'])
    b+=t(64,195,'THIẾT BỊ: PHẦN RIÊNG',20,color=h['teal'],weight=700)
    b+=h['box'](64,221,474,57,'Lịch sử đã bảo vệ → bộ chọn Q',stroke=h['blue'],size=22)
    b+=t(64,310,['GPS local + nhu cầu ψ:','loại POI, tiêu chí, bán kính, đích'],22,lh=1.3)
    b+=h['box'](64,364,474,68,'Gộp POI, bỏ trùng ID\nLọc và sắp xếp theo GPS + ψ',stroke=h['teal'],size=22)
    b+=h['arrow']([(305,342),(305,364)],h['teal'])
    b+=h['box'](680,166,554,282,stroke=h['blue'])
    b+=t(698,195,'MÁY CHỦ: YÊU CẦU CHUNG',20,color=h['blue'],weight=700)
    b+=t(698,240,['5 Q × L30, yêu cầu mọi loại POI','≤150 bản ghi / loại, trước bỏ trùng','Không có ψ, GPS thật hoặc Z'],23,lh=1.5)
    b+=h['arrow']([(538,249),(680,249)],h['blue'])
    b+=h['arrow']([(680,398),(538,398)],h['blue'])
    b+=t(618,229,'Q',20,color=h['blue'],anchor='middle')
    b+=t(618,426,'POI',20,color=h['blue'],anchor='middle')
    b+=t(46,490,'Đã có từ trước: truy hồi mọi loại POI và xử lý phản hồi tại thiết bị.',22)
    b+=t(46,528,'Mới: 4 mục đích local; kiểm tra Q, payload, lịch và số request không đổi khi đổi ψ.',22,color=h['teal'],weight=700)
    b+=t(46,577,'Điều kiện: cùng trạng thái đã bảo vệ và lịch công khai ⇒ transcript mạng giống nhau.',22)
    b+=t(46,624,'Còn khó: ý định tương quan với tuyến đường; click / tương tác có thể tiết lộ nhu cầu.',21,color=h['gray'])
    return b


def candidates(data, **h):
    t=h['text'];category=data['illustration_category']
    answers={p:r[category]['answer'] for p,r in data['local_results'].items()}
    b=t(46,136,'Độ gần dùng để thu ứng viên; tiêu chí của người dùng dùng để chọn câu trả lời',23,weight=700)
    b+=h['box'](46,164,346,78,'5 Q × L30 / loại POI\n≤150 bản ghi / loại, trước bỏ trùng',stroke=h['blue'],size=21)
    b+=h['arrow']([(392,203),(433,203)],h['teal'])
    b+=h['box'](439,164,365,78,'Gộp phản hồi, bỏ trùng ID\nTập ứng viên A trên thiết bị',stroke=h['teal'],size=22)
    b+=h['arrow']([(804,203),(845,203)],h['teal'])
    b+=h['box'](851,164,383,78,'GPS + ψ → lọc và sắp xếp\nDanh sách phù hợp trên thiết bị',stroke=h['teal'],size=22)
    rows=[['Gần nhất','Khoảng cách theo đường'],['Nhanh nhất','Thời gian đường thông thoáng'],
          ['Trong bán kính','Khoảng cách đường ≤ r'],['Ít đi vòng','Độ dài đi vòng qua POI tới đích']]
    b+=tab(['Nhu cầu riêng','Tiêu chí local'],rows,[250,485],h,y=306,row_h=41)
    b+=t(831,292,'t = 60 s (trích danh sách)',23,color=h['blue'],weight=700)
    b+=t(831,330,f"{data['merged']['raw_record_count']} bản ghi → {data['merged']['unique_count']} POI duy nhất",21)
    b+=t(831,369,'Đầu danh sách gần nhất:',20,color=h['blue'])
    b+=t(831,395,', '.join(r['display_alias'] for r in answers['nearest_distance'])+', …',19,color=h['blue'])
    b+=t(831,435,'Bán kính 1 km: '+answers['within_radius'][0]['display_alias'],21,color=h['teal'])
    b+=t(831,474,'Đầu danh sách ít đi vòng:',20)
    b+=t(831,500,', '.join(r['display_alias'] for r in answers['minimum_detour'])+', …',19)
    b+=t(46,552,'L30 tăng độ phủ khi Q đã bị làm nhiễu; không khẳng định mọi mục đích luôn chọn POI gần nhất.',21,weight=700)
    b+=t(46,594,'Lọc và sắp xếp trong A theo nhu cầu local. Đủ đáp án chỉ khi A chứa các POI cần thiết.',21)
    b+=t(46,637,'Đánh đổi: byte phản hồi cao hơn. Fastest chưa có ùn tắc; detour cần đích riêng đã biết local.',20,color=h['gray'])
    return b


def identity_future(content, **h):
    t=h['text']
    rows=[
        ['S4','Liên kết các phiên\ncùng người / cùng xe','Sổ ngân sách chung; giảm thông tin\ntọa độ tích lũy qua nhiều chuyến.','Không ẩn account/IP; hình dạng\nvà routine vẫn nhận dạng được.'],
        ['S5','Suy cạnh đường\nkế tiếp','Chỉ dùng prefix đã bảo vệ để chọn Q;\nGeo-I + giới hạn lần đọc GPS.','Hình học và chuyển động còn\ngợi ý hướng rẽ; pilot 2 ứng viên.'],
        ['S6','Suy đích chưa tới\ntừ lịch sử + prefix','Cùng cơ chế tọa độ và ngân sách;\nkhông đưa tương lai thật vào Q.','Routine nhiều chuyến có thể\nlộ đích; chưa thử open-world.'],
    ]
    b=t(46,137,'Từ bảo vệ từng tọa độ đến quan sát liên phiên và dự đoán tương lai',23,weight=700)
    b+=tab(['Case','Attacker muốn biết','Cơ chế đang có','Khó khăn còn lại'],rows,[70,243,467,408],h,y=199,row_h=91,size=20)
    b+=t(46,542,'Nghiên cứu đe dọa: tính nhận dạng quỹ đạo (de Montjoye, 2013); dự báo hành trình (Ziebart, 2008).',20,color=h['teal'])
    b+=t(46,580,'S8: suy vị trí qua người đồng hành. Đã có diagnostic; chưa có cơ chế nhóm được xác nhận.',21)
    b+=t(46,619,'Phạm vi: bảo vệ tọa độ và nội dung truy vấn; S8 là hướng mở về bảo vệ cấp nhóm.',21,weight=700)
    b+=t(46,651,'Tương quan nhiều người: interdependent location privacy (Olteanu et al., 2017).',18,color=h['gray'])
    return b


def budget(data, **h):
    t=h['text']
    b=t(46,134,'Phiên = một chuyến; epoch = nhóm tối đa 8 phiên dùng chung một sổ ngân sách.',23,weight=700)
    rows=[['Cap mỗi phiên','0,23/m','0,02875/m'],['Tổng cận của 8 phiên','1,84/m nếu cấp lại từng chuyến','0,23/m dùng chung']]
    b+=tab(['Phân bổ','Bản trước','Epoch8 hiện tại'],rows,[330,429,429],h,y=197,row_h=40)
    b+=h['math_text'](46,350,['C',('epoch','sub'),' = 0,23/m;    C',('phiên','sub'),' = C',('epoch','sub'),'/8 = 0,02875/m'],25)
    b+=h['math_text'](46,388,['H = 12;    U = 2H − 1 = 23;    u = C',('phiên','sub'),'/U = 0,00125/m'],25)
    labels=[(46,256,'Còn slot phiên?'),(335,415,'Còn cap dự toán; cách ≥60 s?'),(783,451,'Đọc GPS → thử giữ / tạo Z')]
    for i,(x,w,label) in enumerate(labels):
        b+=h['box'](x,418,w,58,label,stroke=h['blue'],size=21)
        if i<2:b+=h['arrow']([(x+w,447),(labels[i+1][0]-6,447)],h['blue'])
    b+=t(46,519,'Chi phí: tạo Z lần đầu 1u; đọc rồi giữ Z 1u; thử + tạo mới 2u; không đọc 0u.',22)
    b+=t(46,556,'Cận hợp thành: tổng chi phí tọa độ qua các phiên ≤ Cepoch; đổi phiên không tự cấp lại cap.',21,color=h['teal'],weight=700)
    b+=t(46,593,'Không đủ cap đọc: dùng lịch sử đã bảo vệ. Hết 8 slot: không nhận phiên mới, không đọc / gửi Q.',20)
    b+=t(46,634,'Bổ sung cho Geo-I: kiểm soát quan sát lặp S2/S3 và liên phiên S4; chưa tạo tính không liên kết danh tính.',20,color=h['gray'])
    return b


def full_sample(data, utility, **h):
    t=h['text'];events=data['main_sample']['events'];rows=[]
    for time in [0,20,60,120,600]:
        r=next(e for e in events if e['t_s']==time);p=r['protection']
        rows.append([str(time),'Có' if p['GPS_read'] else 'Không','Tạo Z' if p['branch']=='fresh' else 'Giữ Z',
                     'Cập nhật' if p['GPS_read'] else 'Dự đoán',f"{p['cost_units']}u",str(p['spent_after_units'])+'/23'])
    b=t(46,136,'Mẫu đã lưu: freshqp-001, TRAIN, draw 1, phiên 1; Q được gửi mỗi mốc 20 s',22,weight=700)
    b+=tab(['t (s)','GPS\nGeo-I','Z','b','Chi\nmốc này','Tổng'],rows,[67,90,102,125,97,114],h,y=190,row_h=39,size=20)
    r=next(e for e in events if e['t_s']==60)
    points=[r['gps_xy'],r['Z_xy'],*r['Q_xy'],*r['belief_top_xy']]
    marks=[{'xy':r['gps_xy'],'shape':'cross','color':GPS_COLOR,'label':'GPS'},
           {'xy':r['Z_xy'],'shape':'diamond','color':h['blue'],'label':'Z'}]
    marks += [{'xy':pos,'color':h['teal'],'label':str(i+1),'radius':7} for i,pos in enumerate(r['Q_xy'])]
    def mt(x,y,value,size=23,**kw):return t(x,y,value,size*.82,**kw)
    b+=t(688,174,'t = 60 s: GPS → Z → b → 5 Q',22,color=h['blue'],weight=700)
    b+=map_svg('agenda-trip',(688,192,546,263),data['map_main']['road_segments_xy'],points,marks,
               text=mt,line=h['line'],dot=h['dot'],belief=[{'xy':c['xy_m'],'weight':c['weight']} for c in r['belief']['top_weights']])
    b+=t(46,475,'Tại 60 s: dự toán 2u → đọc GPS → thử giữ thất bại → REM tạo Z → cập nhật b → chọn Q.',20)
    b+=t(46,513,f"5 Q, L30 mọi loại → {utility['merged']['raw_record_count']} bản ghi → {utility['merged']['unique_count']} POI duy nhất → GPS + ψ xếp hạng local.",21,color=h['teal'],weight=700)
    result={p:r[utility['illustration_category']]['answer'] for p,r in utility['local_results'].items()}
    near=', '.join(p['display_alias'] for p in result['nearest_distance'])
    radius=', '.join(p['display_alias'] for p in result['within_radius'])
    b+=t(46,551,f'Trích danh sách đã sắp xếp: gần nhất [{near}, …]; bán kính 1 km [{radius}].',20)
    reuse=data['test_reuse_inset']['events'][-1]['protection']
    b+=t(46,589,f"Nhánh giữ Z: phiên 6 tại 60 s đọc rồi giữ, chi {reuse['cost_units']}u. Khác mốc 20 s không đọc / chi 0u.",20)
    b+=t(46,626,'600 s: 21/23 gồm mọi lần đọc giữa các mốc; chưa hết cap. Gửi ngay; Z và ψ không ra mạng.',20,color=h['gray'])
    b+=t(46,653,'Màu cam: một phần trọng số b. Nhánh giữ thuộc phiên khác; mẫu minh họa, không là kết quả test.',18,color=h['gray'])
    return b


def utility_results(data, **h):
    t=h['text'];u=data['current_fresh_four_purpose_utility'];cost=u['cost']
    b=t(46,135,'Báo cáo trước: L10, tìm gần nhất, Recall 95,44% (12 nhóm phát triển).',22)
    b+=t(46,170,'Hiện tại: 4 mục đích, cap 8 phiên. Bảng dưới so L20–L30 trên cùng Q ở 24 nhóm mới × 3 lượt.',21,color=h['teal'])
    rows=[[r['label_vi'],n(100*r['L20_recall']),n(100*r['L30_recall']),'+'+n(r['gain_pp'])] for r in u['rows']]
    b+=tab(['Recall@5 ↑ (%)','L20 đối chứng','L30 hiện tại','Δ điểm %'],rows,[480,240,240,228],h,y=226,row_h=39,size=21)
    r=next(r for r in u['rows'] if r['primary']);lo,hi=r['paired95_family_bootstrap_gain_pp']
    b+=t(46,505,f"Macro +{n(r['gain_pp'])} điểm %; CI 95% [{n(lo)}; {n(hi)}]; ba lượt nhiễu đều tăng.",22,color=h['teal'],weight=700)
    b+=t(46,549,f"Phản hồi JSON: {n(cost['L20']['reply_bytes']/1e6)} → {n(cost['L30']['reply_bytes']/1e6)} MB (+{n(cost['reply_growth_percent'])}%).",22)
    b+=t(46,589,'Giữ nguyên Q, Z, lịch, cap và 90.570 request; chỉ đổi độ sâu phản hồi. Chưa đo latency / năng lượng.',20)
    b+=t(46,634,'95,44% trước và 92,69% hiện tại khác mục đích, cap, dữ liệu và cách gộp: không so trực tiếp thành tăng / giảm.',20,color=h['gray'])
    return b


def privacy_results(data, **h):
    t=h['text'];s4=data['S4_historical_linkage'];s56=data['S5_S6_matched_native_pilot'];ep=data['S9_S10_full_bank_endpoint']
    idx={(r['method'],r['target']):r for r in s4['rows']};p={r['method']:r for r in s56['rows']};e={(r['scenario'],r['method']):r for r in ep['rows']}
    rows=[]
    for target,label in [('same_person','S4: cùng người'),('same_vehicle','S4: cùng xe')]:
        a=idx['raw',target];b=idx['geoi_slack_reconstructed',target]
        rows.append([label,'AUC ↓',n(a['roc_auc'],3),n(b['roc_auc'],3),'3 nhóm; Geo-I lịch sử'])
    rows.append(['S5: cạnh kế tiếp','Đúng cạnh ↓',n(100*p['raw']['S5_exact_candidate_edge_accuracy'])+'%',n(100*p['rem_epoch8']['S5_exact_candidate_edge_accuracy'])+'%','Pilot Epoch8 / L20'])
    rows.append(['S6: đích tương lai','Hit100 ↓',n(100*p['raw']['S6_destination_hit100'])+'%',n(100*p['rem_epoch8']['S6_destination_hit100'])+'%','Pilot Epoch8 / L20'])
    for sc,label in [('S9','S9: điểm đầu'),('S10','S10: điểm cuối')]:
        rows.append([label,'MAE ↑ (m)',n(e[sc,'scale100_L20']['mae_m'],0),n(e[sc,'scale025_L20']['mae_m'],0),'28 nhóm; Endpoint20'])
    b=t(46,135,'Trước: S4–S8 chưa có benchmark. Nay: bổ sung S4, S5/S6 và kiểm tra đầu/cuối.',22,weight=700)
    b+=t(46,169,'Đối chứng trong từng hàng: GPS thật (S4–S6); Geo-I L20 lịch sử (S9/S10).',21)
    b+=tab(['Scenario','Metric','Đối chứng','Được bảo vệ','Phạm vi phép thử'],rows,[243,180,180,190,395],h,y=231,row_h=38,size=21)
    b+=t(46,527,'S5/S6: Planar cũng 50%; Recall Planar 97,00%, REM 94,69%. Hai task chung quyết định nhị phân.',20)
    b+=t(46,566,'Endpoint20: nhiễu mạnh hơn, gửi ngay; Recall 98,94% → 96,64%. S10 Hit100 cùng 0%.',21,color=h['teal'])
    b+=t(46,605,'S4: AUC từng nhóm còn tới 0,861; AUC dưới 0,5 không chứng minh đã ẩn danh. S8 còn mở.',20)
    b+=t(46,644,'Không phải một benchmark L30 thống nhất: cần xác nhận privacy cùng Epoch8/L30; S1–S3 giữ ở bản chi tiết.',20,color=h['gray'])
    return b
