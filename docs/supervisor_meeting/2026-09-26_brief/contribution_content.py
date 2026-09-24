"""Measured contribution claims; tables read the executable stress artifact."""
import json

def add(ns):
    page,sub,p,table=(ns[k] for k in ('page','sub','p','table'))
    root=ns['ROOT']; d=json.loads((root/'artifacts/benchmarks/contribution_stress/results.json').read_text())
    ab=d['ablation']; ep=d['endpoints']; pct=lambda x:f'{100*x:.2f}%'
    page();sub('Đóng góp nào đã có bằng chứng: tách từng thành phần')
    p('Kiểm tra bổ sung ngày 24/09/2026 trên toàn bộ 636 đầu ra S1–S3 đã lưu. Giữ đối thủ đã chọn bằng validation, K=5, ngân sách 0,24 và độ sâu phản hồi L=5. Trung bình các lần lặp trong từng nhóm/ca, rồi nhóm trong ca, rồi chín ca với trọng số bằng nhau. Bootstrap ghép cặp 10.000 lần theo sáu nhóm tuyến; không coi 636 dòng là 636 mẫu độc lập.')
    rows=[]
    for c in ab['contrasts']:
        for metric,label,scale in [('mae_m','MAE suy luận (m) ↑',1),('hit100','Hit100 (điểm %) ↓',100),('recall_L5','Recall (điểm %) ↑',100),('step_ms','Thời gian/bước (ms) ↓',1)]:
            v=c['metrics'][metric]; rows.append([c['baseline'],label,f"{scale*v['delta']:+.2f}",f"[{scale*v['ci95'][0]:+.2f}; {scale*v['ci95'][1]:+.2f}]"])
    table(['Bản so sánh','Metric','Switching − bản so sánh','Khoảng 95%'],rows,[3.3,4.3,4.6,4.7])
    p('**Lợi thế được hỗ trợ:** so với dummy hình học, cả sai số suy luận và Recall đều tăng. Recall tăng 7,95 điểm %, khoảng [6,21; 9,71]. Đổi lại tăng khoảng 70,78 ms/bước trên máy chạy lịch sử. Đây là đóng góp của tổ hợp cơ chế; chưa tách riêng tác dụng của mạng đường vì các nhánh cùng dùng mạng làn.')
    p('**Phần switching chưa chứng minh được lợi thế riêng:** so với mean_exchange, các khoảng MAE, Hit và Recall đều chứa 0; Recall gần như bằng nhau. So với mean_greedy, MAE giảm 16,32 m là bất lợi về privacy dù Hit trung bình thấp hơn. Không nên lấy switching làm đóng góp chính ở thời điểm này.')
    p('Đây là phân tích hồi cứu với chỉ sáu cụm, chưa hiệu chỉnh so sánh nhiều metric. Khoảng 95% thể hiện độ bất định trong mẫu hiện có, không chứng minh tổng quát sang thành phố khác. Chưa có cấu hình vượt ngưỡng Recall 90% ở mọi ca validation; không bỏ điều kiện này để chọn một số tổng đẹp.')

    page();sub('Tấn công lại S9/S10: thay đối thủ có làm đổi kết luận?')
    p('Đã chạy mới 792 lượt đánh giá trên 264 transcript lịch sử: 11 cơ chế × 2 scenario × 12 chuyến × 3 mức cắt. Mỗi transcript K=5 có 71 bộ ước lượng: trung bình/trung vị, từng track dummy, và ngoại suy tuyến tính từ 2/3/6 điểm sát biên với 30/60/120 giây. Chỉ nhận transcript công khai; nhãn thật chỉ dùng để chấm và chọn đối thủ ở tập huấn luyện.')
    p('Chia theo ba seed mô phỏng: chọn riêng đối thủ tối ưu MAE/Hit trên hai seed, kiểm tra trên seed còn lại rồi luân phiên. Mỗi fold chỉ bốn chuyến test; chưa xác minh tách biệt địa điểm giữa các seed. Đây là stress test hồi cứu, không phải xác nhận độc lập hay chạy lại defender mới. AnotherMe có 24 dòng not_applicable trong nguồn endpoint nên giữ trạng thái thiếu, không gán điểm 0.')
    picked=[s for s in ep['summaries'] if s['extra_cut_s']==0]
    by={(s['scenario'],s['method']):s for s in picked}
    methods=['unprotected','br_fresh','br_private','dls_graph_adaptation','transprotect_adaptation','semantic_correlation_local_adaptation','geo_i_anchored_dummy','uniform_dummy']
    labels=['Nền đã che biên','BR neo mới','BR tái dùng neo','DLS thích nghi','TransProtect thích nghi','Semantic thích nghi','Geo-I + dummy','Dummy đều']
    rows=[]
    for m,label in zip(methods,labels):
        a,b=by['S9',m],by['S10',m]
        rows.append([label,pct(a['hit100']),f"{a['mae_m']:.0f}",pct(a['recall_all_queries']),pct(b['hit100']),f"{b['mae_m']:.0f}",pct(b['recall_all_queries'])])
    table(['Cơ chế K=5','S9 Hit ↓','MAE m ↑','Recall ↑','S10 Hit ↓','MAE m ↑','Recall ↑'],rows,[4.0,2.1,2.0,2.2,2.1,2.2,2.3])
    p('**S9:** BR neo mới bị đoán đúng 7/12 chuyến (58,33%). BR tái dùng neo đạt 0/12 với Recall 96,60% trong cửa sổ chấm; đây là tín hiệu có lợi trong phiên bản lịch sử này, chưa phải kết quả của lõi switching hay BR-Boundary. Đối thủ chọn cho BR neo mới ở cả ba fold là hồi quy sáu centroid rồi ngoại suy ngược 30 giây. Áp chính đối thủ đó lên BR tái dùng neo vẫn trúng 2/12 (16,67%): 0/12 của bộ chọn không có nghĩa mọi đối thủ đều thất bại. Đây là chẩn đoán hậu kiểm, chưa chứng minh nguyên nhân do tái dùng neo. Các bản thích nghi paper không phải tái lập nguyên bản; ngân sách giữa mọi cơ chế chưa đồng nhất.')
    p('**S10:** ngay nền đã che biên cũng đạt Hit=0/12 với bộ chọn mới. Vì vậy Hit100 đang không phân biệt tốt các cơ chế; 0/12 của BR không chứng minh đã giải quyết S10. Cần bổ sung Hit theo nhiều bán kính, đo mức cải thiện của đối thủ so với prior, tấn công đường/POI và nhiều chuyến cùng đích. MAE lớn có thể do đối thủ yếu hoặc khoảng che vốn khó, không chỉ do cơ chế tốt.')
    p('Recall giữ mẫu số 12 sự kiện gốc của cửa sổ, truy vấn cắt thêm nhận 0; vẫn chưa chứa 60 giây đã giấu ban đầu nên không phải utility toàn chuyến. Toàn bộ 11 cơ chế và ba mức cắt được lưu trong artifacts/benchmarks/contribution_stress/results.json. Bảng trình bày các nhánh chính; không dùng bảng này để tuyên bố thắng toàn bộ sáu paper đối chứng.')

    page();sub('Điều kiện để khẳng định bảo vệ đủ năm scenario')
    table(['Giả thuyết đóng góp','Bằng chứng hiện có','Phép thử còn cần'],[
      ['Chọn tập truy vấn theo utility','S1–S3: Recall và MAE tốt hơn hình học, với chi phí tính toán tăng.','Xác nhận trên nhóm mới; so sánh ở cùng budget, K, độ sâu phản hồi và ngưỡng utility.'],
      ['Tái dùng neo có ngân sách','S9 lịch sử: 0/12 so với 7/12 của bản neo mới dưới đối thủ bổ sung.','Giữ cố định các khối khác và ngân sách tiêu thụ; thử nhiều phiên cùng nguồn/đích. Chưa suy ra riêng quan hệ nhân quả chỉ từ hai nhánh lịch sử.'],
      ['Belief switching','Chưa vượt mean_exchange một cách ổn định.','Giữ làm nhánh ablation; khóa tiêu chí chấp nhận trước khi dùng lại tập mới.'],
      ['Cổng đầu + buffer cuối','Đã chạy code tích hợp; chưa có số privacy/utility của defender kết hợp trên mạng thành phố.','So lõi, chỉ cổng đầu, chỉ buffer và cả hai; đo độ trễ, tỷ lệ phục vụ, Recall toàn phiên và metadata.'],
      ['Bảo vệ S9/S10 A/B/C','Chưa kiểm tra đủ biến thể và nhiều phiên lặp.','Tấn công phụ thuộc đường/POI và gộp nhiều chuyến; khóa site/split trước huấn luyện; không đọc endpoint thật để quyết định gate.']],[3.5,6.1,7.3])
    p('**Không cắt thêm để lấy số đẹp:** trong stress test, S9/BR tái dùng neo cắt thêm 40 giây làm Hit từ 0 lên 16,67%, Recall cửa sổ từ 96,60% xuống 80,37%. S10 vẫn Hit=0 nhưng Recall từ 88,33% xuống 75,07%. Các bộ tấn công hữu hạn có thể phản ứng khác nhau khi cắt; đối thủ tối ưu có transcript dài vốn cũng tự cắt được. Đây không phải bằng chứng cắt làm tăng thông tin.')
    p('Thí nghiệm tiếp theo cần khóa protocol trước: dùng các nhóm mới chưa xem để xác nhận; giữ nhóm tuyến và site nhạy cảm tách giữa train/validation/test; đo cả truy vấn bị bỏ và độ trễ, không chỉ truy vấn còn lại. Cùng một session giữ một sổ ngân sách qua mọi cửa sổ. Báo cáo riêng mỗi scenario và A/B/C, kèm đối thủ được chọn ngoài tập test; điểm tổng chỉ dùng sau khi qua ngưỡng utility/performance đã chốt.')
    p('**Điểm đang chặn lần chạy đầy đủ:** workspace thiếu Beijing.osm.gz và beijing_smoke.net.xml gốc. Thử tải nguồn BBBike chưa thành công. Chưa thay bằng bản đồ khác để ghép số vào benchmark cũ. Khi khôi phục, cần kiểm tra hash nguồn/cấu trúc mạng trước khi sinh lại đầu ra; nếu dùng mạng mới phải mở một phiên benchmark riêng.')
    p('Kết luận hiện tại: đóng góp thực nghiệm rõ nhất là cách chọn dummy theo utility trong S1–S3. Tái dùng neo có tín hiệu tốt ở S9 lịch sử. Chưa đủ bằng chứng khẳng định phiên bản mới bảo vệ S9/S10 hoặc vượt các phương pháp gốc trong papers. Cần hoàn tất vòng thí nghiệm trên để nâng các giả thuyết này thành contributions đã xác nhận.')
