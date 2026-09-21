"""Detailed method/evidence chapters shared by the LaTeX and HTML brief."""
from pathlib import Path
import hashlib,json,math,statistics

def add_method_content(ns):
    p,key,sec,sub,page,table,eq,bullets=(ns[x] for x in ['p','key','sec','sub','page','table','eq','bullets'])
    root,out=ns['ROOT'],ns['OUT']
    def load(path):return json.loads((root/path).read_text())
    def figure(name,caption,width='\\linewidth'):
        ns['graphic'](r'\includegraphics[width='+width+']{figures/'+name+'.pdf}',
            '<img src="figures/'+name+'.svg" alt="'+caption.replace('"','&quot;')+'" style="width:100%;height:auto">',caption)
    paper=load('artifacts/benchmarks/paper_benchmark/results.json')
    fresh=load('artifacts/benchmarks/fresh_switching/confirmation.json')
    readout=load('artifacts/benchmarks/fresh_switching/readout.json')
    select=load('artifacts/benchmarks/fresh_switching/selection.json')
    audit=load('artifacts/benchmarks/report_boundary_audit/results.json')
    def pct(x):return f'{100*x:.2f}%'
    def f1(x):return f'{x:.1f}'
    sources=['artifacts/benchmarks/paper_benchmark/results.json','artifacts/benchmarks/fresh_switching/confirmation.json',
      'artifacts/benchmarks/fresh_switching/readout.json','artifacts/benchmarks/fresh_switching/selection.json',
      'artifacts/benchmarks/report_boundary_audit/results.json','benchmark/engines/budgeted.py',
      'benchmark/engines/contextual_lane.py','benchmark/engines/service_cover.py','benchmark/engines/fair_cover.py',
      'benchmark/engines/switching_cover.py','benchmark/switching_belief.py','core/mechanisms.py']
    evidence={'status':'historical_measured_results_plus_new_exploratory_reanalysis','checked_on':'2026-09-21',
      'sources':{x:hashlib.sha256((root/x).read_bytes()).hexdigest() for x in sources},
      'historical_versions_not_pooled':True,'latest_model_S9_S10_results_available':False,
      'new_audit_rows':len(audit['rows']),'source_rows_reverified':audit['verified_source_rows']}
    (out/'method_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    page();sec('Phương pháp BR-Dummy: kiến trúc và luồng xử lý')
    p('**Ý tưởng trung tâm:** vị trí thật tạo một neo riêng tư; từ lịch sử neo và ngữ cảnh công khai, thiết bị chọn một tập vị trí truy vấn vừa đi được trên đường vừa có khả năng trả đúng POI. Máy chủ chỉ thấy tập công bố. Sau khi nhận phản hồi, thiết bị dùng vị trí thật để gộp và chọn kết quả phục vụ người dùng.')
    figure('architecture','Hình 1. Kiến trúc lõi đã có trong mã nguồn và vị trí dự kiến của cổng bảo vệ biên S9/S10. Neo và nhãn thật không thuộc transcript máy chủ.')
    table(['Khối','Đầu vào → đầu ra','Trạng thái'],[
      ['Neo + ngân sách','GPS hiện tại, neo trước, tham số → neo riêng tư zₜ.','Đã hiện thực; quyết định tái dùng cũng có nhiễu.'],
      ['Belief chuyển động','Neo đã bảo vệ, thời gian, mạng đường → phân bố vị trí/chế độ.','Đã có biến thể hai trạng thái; chưa hiệu chỉnh bằng giao thông thật.'],
      ['Chọn tập truy vấn','Belief, miền đường tới được và danh mục POI → K tọa độ.','Tham lam rồi tối đa 3 lần thay một điểm.'],
      ['Phục vụ và lọc','Phản hồi từ K điểm → loại trùng → top-5 cho nhu cầu thật.','Đã có oracle POI cục bộ và đánh giá chất lượng.'],
      ['Cổng bảo vệ biên','Transcript đã bảo vệ + chính sách biên → lịch công bố có kiểm soát.','Đề xuất tích hợp; mới có phép thử cắt transcript ngoại tuyến.']],[3.6,7.5,5.8])
    p('Phần lõi hiện tại là cơ chế xác suất + lọc Markov ẩn + tối ưu tổ hợp; không phải mạng deep learning đầu-cuối. Ranh giới quan trọng là bộ chọn dummy không đọc GPS thật, vận tốc thật hoặc đích tương lai. Bản đồ và POI đi vào cả mô hình chuyển động lẫn mục tiêu chất lượng.')

    page();sub('Neo riêng tư và sổ ngân sách theo phiên')
    p('Ở lần đầu, thiết bị lấy một neo ngẫu nhiên từ cơ chế mũ trên tập vị trí công khai. Ở lần tiếp theo, nó dùng kiểm tra có nhiễu để quyết định giữ neo cũ hay lấy neo mới. Việc kiểm tra phụ thuộc GPS thật nên cũng phải trả ngân sách; không được kiểm tra khoảng cách thật rồi gọi toàn bộ bước này là hậu xử lý.')
    eq(r'd(x_t,z_{t-1})+\operatorname{Lap}(1/\varepsilon_{\rm test})\le\theta\ \Rightarrow\ z_t=z_{t-1}.',
       'Khoảng cách thật tới neo trước + nhiễu kiểm tra ≤ θ → giữ neo; nếu không, lấy neo mới.')
    eq(r'\Pr[z_t=v\mid x_t]\propto\exp\!\left[-\frac{\varepsilon_{\rm release}}{2}d_E(x_t,v)\right].',
       'Khi lấy neo mới, vị trí gần GPS thật có xác suất cao hơn; mọi vị trí trong support công khai vẫn có xác suất.')
    p('Bản đang thử dùng B=0,24 m⁻¹, H=12 sự kiện, θ=200 m; chia ε_release=ε_test=B/(2H)=0,01 m⁻¹. Lần đầu ghi tối đa 0,01; 11 lần sau ghi 0,02 mỗi lần, tổng 0,23 ≤ B. Đây là sổ theo trường hợp xấu nhất, không hoàn ngân sách theo nhánh bí mật vừa xảy ra.')
    table(['Tình huống','Cách xử lý và ý nghĩa'],[
      ['Người dừng ở một chỗ','Tái dùng neo làm giảm việc phát nhiều vị trí nhiễu độc lập; chỉ là giảm rủi ro trong chân trời hữu hạn, không chống được mọi phép lấy trung bình.'],
      ['Hết H=12 sự kiện','Không đọc tọa độ riêng tư mới để cập nhật neo; tiếp tục từ trạng thái đã bảo vệ. Có thể mất chất lượng nếu chuyến kéo dài.'],
      ['Reset hoặc mở phiên khác','Phát sinh ngân sách khác; không reset miễn phí nhiều lần rồi vẫn gọi tổng chuyến là B=0,24.'],
      ['Bảo đảm lý thuyết','Chỉ trong mô hình giả định và phạm vi composition đã phân tích. Không suy từ B ra xác suất nhận diện 1/K hoặc bảo đảm giấu nhà.']],[4.3,12.6])
    sub('Belief hai trạng thái: biết đang dừng hay đang đi bằng cách nào?')
    p('Bộ lọc theo dõi hai chế độ: dừng và di chuyển. Nó cập nhật từ chuỗi neo đã bảo vệ, không được nhìn vận tốc thật để chọn chế độ. Nhân dừng giữ nhiều khối xác suất tại vị trí cũ; nhân di chuyển cho khối xác suất lan ra trên mạng đường. Khoảng trống quá 120 giây dùng quy tắc hồi về prior công khai đã định.')
    eq(r'\widetilde b_t(i,m)=\sum_{j,n}b_{t-1}(j,n)A_{nm}(\Delta t)T^{(m)}_{\Delta t}(j,i),\quad b_t(i,m)\propto\ell_t(i)\widetilde b_t(i,m).',
       'Dự đoán vị trí/chế độ từ bước trước → cập nhật bằng likelihood của neo mới → chuẩn hóa.')
    p('bₜ là belief nội bộ để phục vụ chọn truy vấn, không được đồng nhất với posterior thật của mọi đối thủ. Các hệ số chuyển động hiện là heuristic cố định. Việc một metric privacy tốt hơn phải được chứng minh bằng thực nghiệm, không suy từ tên “hai trạng thái”.')

    page();sub('Chọn K điểm theo độ phủ POI và đường có thể tới')
    p('Mỗi track dummy có một miền trạng thái tới được từ điểm công bố trước, theo thời gian trôi qua và mạng làn có hướng. Chọn trên miền này giúp chuỗi công bố có chuyển động hợp lệ. Tập trạng thái và ràng buộc giao thông là công khai; không cắt miền quanh vị trí thật bằng một bán kính bí mật.')
    eq(r'\max_{v_j\in\mathcal R_{j,t}}\ \sum_{p\in\cup_j\operatorname{POI}(v_j)}w_t(p),\qquad j=1,\ldots,K.',
       'Chọn một vị trí trong mỗi miền tới được để tổng trọng số của hợp POI thu được lớn nhất.')
    p('wₜ(p) được suy từ belief vị trí và danh mục POI. Một POI đã có trong hợp không được cộng lặp khi thêm dummy khác. Cách này khuyến khích K điểm bổ sung kết quả cho nhau. Đây là mục tiêu proxy của bộ chọn; Recall thật vẫn phải đo tại GPS thật sau khi nhận và lọc phản hồi.')
    bullets(['Khởi tạo bằng tham lam: chọn cặp track–vị trí có ích lợi tăng thêm lớn nhất trong các lựa chọn còn hợp lệ.',
      'Tinh chỉnh tối đa 3 lần: thay một vị trí nếu làm mục tiêu tăng. Không tuyên bố tìm tối ưu toàn cục.',
      'Giữ thứ tự track và ràng buộc đường. K vị trí có thể trùng; K không tự là mức k-anonymity.',
      'Trong vòng hiện tại, K=5 và 6 loại POI tạo 30 truy vấn loại–tọa độ mỗi sự kiện. L=5/10 là số phản hồi mỗi truy vấn; ứng dụng cuối vẫn cần top-5.'])
    sub('Ví dụ đầu vào → neo → transcript thực sự đã sinh')
    row=next(r for r in fresh['rows'] if r['case_id']=='S3.A' and r['method']=='switching_exchange' and r['replicate']==1)
    rec=next(r for r in fresh['records'] if r['record_id']==row['record_id'])
    raw=rec['points'][0];anchor=row['evaluator_anchors'][0];event=row['public']['events'][0]
    p(f'Mẫu {row["record_id"]}, S3.A, nhóm {row["family_id"]}, lần lặp 1. Các tọa độ bên dưới được lấy từ confirmation.json, không tự dựng để minh họa.')
    table(['Phía nhìn thấy','Giá trị ở t=0, theo thứ tự (lat; lon)'],[
      ['Thiết bị / đánh giá: GPS thật',f'({raw["lat"]:.6f}; {raw["lon"]:.6f})'],
      ['Thiết bị / đánh giá: neo',f'({anchor[0]:.6f}; {anchor[1]:.6f})'],
      ['Máy chủ: 5 điểm công bố','; '.join(f'{i+1}: ({c["lat"]:.6f}; {c["lon"]:.6f})' for i,c in enumerate(event['candidates']))],
      ['Phản hồi cafe ở L=10','5 truy vấn; 50 mục POI phản hồi; Recall@5=1,00 sau lọc; ΔD=0 m ở riêng truy vấn này.']],[5.4,11.5])
    p('Một sự kiện đúng đủ POI không chứng minh toàn bộ phương pháp đạt yêu cầu. Biểu đồ ở trang sau cho thấy cả chuỗi vị trí thật, neo và các track công bố; chỉ panel bên phải là thông tin không gian máy chủ nhận trong phép thử này.')

    page();sub('Quan sát kiến trúc qua một chuỗi đã chạy')
    figure('protected_sample','Hình 2. Mẫu S3.A của biến thể switching_exchange. Hai panel dùng cùng hệ trục và cùng tỷ lệ; gốc tọa độ đặt ở điểm thật đầu cửa sổ, chỉ phục vụ hình đánh giá.')
    table(['Nhìn vào hình','Điều được kiểm tra'],[
      ['Neo có thể đứng yên qua nhiều lần','Kết quả của kiểm tra tái dùng riêng tư; không phải giữ nguyên GPS thật.'],
      ['Dummy thay đổi ít ở mẫu này','Năm track có lần lượt 1, 2, 2, 1, 1 vị trí phân biệt. Nhiều sự kiện chồng nhau nên khó thấy đường nối; ràng buộc tới được cho phép đứng yên.'],
      ['Dummy không cần trùng đường thật','Mục tiêu là vừa hợp lệ trên đường vừa giữ dịch vụ, không mô phỏng đúng mọi bước của người dùng.'],
      ['Máy chủ không thấy panel trái','Không đưa neo, truth, trạng thái evaluator hoặc nhãn tuyến vào đặc trưng tấn công.'],
      ['Hai panel khác nhau','Khoảng cách nhìn thấy không phải metric privacy: cần chạy đối thủ rồi so dự đoán với truth.']],[5.0,11.9])
    sub('Các kiểm tra bắt buộc trước khi mở rộng')
    bullets(['Thay phần tương lai mà đầu ra tiền tố không đổi: kiểm tra nhân quả, áp dụng cho cả cơ chế và bộ chọn.',
      'Tái tạo sổ ngân sách; khi hết H, thay GPS tương lai không được làm đầu ra mới thay đổi.',
      'Đối chiếu chuyển trạng thái với mạng làn có hướng, không dùng tính hợp lệ trên đồ thị nút giao thay cho luật rẽ.',
      'Tính lại Recall từ ID POI và oracle; tách byte mã ID, payload và lưu lượng mạng thực.',
      'Nguồn mã: budgeted.py → contextual_lane.py → service_cover.py / fair_cover.py → switching_cover.py.'])

    page();sec('Kết quả thực nghiệm: phiên bản nào đã được đánh giá?')
    table(['Vòng','Dữ liệu / cơ chế','Được kết luận đến đâu'],[
      ['paper-v2, lịch sử','3 seed 81–83; 12 chuyến test; 5 scenario S1/S2/S3/S9/S10. BR trên đồ thị nút giao, K=3/5.','Có số đo đầu/cuối thật; chưa là bộ 30 A/B/C hiện tại, chưa là biến thể làn + bộ lọc mới.'],
      ['fresh-switching, lịch sử mới hơn','6 nhóm validation 301–306, 6 nhóm confirmation 307–312. 53 bản ghi confirmation S1–S3 × 3 lặp × 4 biến thể = 636 lượt.','So thành phần của BR-làn; không phải so đủ sáu paper và chưa đo bản mới trên S9/S10.'],
      ['Boundary audit, bổ sung lần này','Tái phân tích 48 hàng nguồn S9/S10; 3 mức cắt → 144 hàng chẩn đoán.','Kết quả mới từ transcript đã lưu; thăm dò trên dữ liệu đã xem, không có test độc lập mới.']],[3.5,7.0,6.4])
    key('Không ghép các bảng thành một model đã đạt S1–S10. Cần giữ phiên bản cơ chế, dataset, đối thủ và mẫu số cạnh từng kết quả.')
    sub('Kết quả S1–S3 của bốn biến thể nội bộ mới nhất')
    names={'geometric':'Hình học','mean_greedy':'Phủ tham lam','mean_exchange':'Phủ + thay điểm','switching_exchange':'Hai trạng thái + thay điểm'}
    table(['Biến thể','MAE (m) ↑','Hit chọn ↓','Hit envelope ↓','Recall L=5 ↑','Recall L=10 ↑'],
      [[names[m],f1(s['mae_m']),pct(s['hit100']),pct(s['envelope_hit100']),pct(s['utility']['5']['recall']),pct(s['utility']['10']['recall'])] for m,s in readout['methods'].items()], [4.5,2.3,2.4,2.8,2.5,2.5])
    p('Trung bình đều qua chín ca, không gộp tất cả điểm thành một mẫu số. “Hit chọn” dùng đối thủ chọn trên validation; “envelope” là mức cao nhất trong bộ đối thủ đã thử trên confirmation, để chẩn đoán. Đối thủ tối thiểu MAE có thể khác đối thủ tối đa Hit.')
    p('Thêm bộ lọc hai trạng thái vào bộ phủ + thay điểm: Hit chọn giảm 3,73% → 2,92%, nhưng MAE giảm 577,4 → 555,1 m. Nghĩa là ít lần bị định vị trong 100 m hơn, song sai số trung bình của đối thủ cũng nhỏ hơn. Không có cải thiện đồng thời trên mọi metric privacy. Recall gần như giữ nguyên.')
    p('Tăng L từ 5 lên 10 nâng Recall của biến thể mới từ 92,76% lên 96,31%, đồng thời byte danh sách ID tăng khoảng 2.735 → 4.504 mỗi sự kiện. Đây là thêm tài nguyên phản hồi, không phải hiệu quả miễn phí của thuật toán.')

    page();sub('Kết quả theo từng ca A/B/C: biến thể hai trạng thái')
    ss=[s for s in fresh['summaries'] if s['method']=='switching_exchange']
    table(['Ca','Nhóm','MAE m ↑','Hit chọn ↓','Hit env. ↓','Recall L5 ↑','Recall L10 ↑'],
      [[s['case_id'],str(s['families']),f1(s['selected_mae_m']),pct(s['selected_hit100']),pct(s['envelope_hit100']),pct(s['utility']['5']['poi_recall_at_5']),pct(s['utility']['10']['poi_recall_at_5'])] for s in ss], [1.7,1.2,2.6,2.8,2.8,3.0,3.0])
    figure('fresh_cases','Hình 3. Kết quả chín ca của switching_exchange. Đường 90% chỉ giúp đọc bảng confirmation; không dùng để chọn lại cấu hình.',width=r'0.91\linewidth')
    p('Ca khó về dịch vụ là chuyển động và gửi thưa: S3.A/S3.C ở L=5 chỉ đạt 85,56%/87,01%. S3.B có 5 nhóm confirmation, các ca khác có 6; ba lần nhiễu lặp không làm tăng số nhóm độc lập.')

    page();sub('Quy tắc chọn vẫn chưa có cấu hình khả thi')
    table(['L','Biến thể','Recall ca yếu nhất trên validation','Đạt mọi ca ≥90%?'],
      [[l,names[c['method']],pct(c['min_case_recall']),'Có' if c['min_case_recall']>=.9 else 'Không'] for l,g in select['method_selection_by_depth'].items() for c in g['candidates']], [1.0,5.9,6.6,3.4])
    p('Dù confirmation có các con số đẹp, không quay lại chọn cấu hình từ confirmation. Kết luận của vòng đã khóa vẫn là **chưa có cấu hình khả thi trong lưới đã thử**. S3.C trên validation là nút thắt. Vòng phát triển tiếp theo có thể dùng các phát hiện này, nhưng phải xác nhận bằng dữ liệu mới.')
    sub('S1, S2, S3, S9, S10 trong cùng vòng paper-v2')
    names_old={'unprotected':'Mặt nạ / không nhiễu','br_fresh':'BR neo mới','br_private':'BR tái dùng'}
    table(['Case','Phương pháp, K=5','MAE m ↑','Hit100 ↓','Recall@5 ↑'],[
      [c,names_old[m],f1(s['location_mae_m']),pct(s['location_hit_100m']),pct(s['poi_recall_at_k'])]
      for c in ['S1','S2','S3','S9','S10'] for m in names_old
      for s in [next(s for s in paper['summary'] if s['scenario']==c and s['method']==m and s['k']==5)]],[1.2,5.3,3.0,3.1,4.3])
    p('Mỗi dòng có 12 chuyến ở ba seed; các dòng dùng lại các chuyến đó, không phải các tập người độc lập. Baseline S9/S10 đã bị che 60 giây và chỉ lộ cửa sổ 220 giây. Baseline S1–S3 có sai số nhỏ do biểu diễn nút giao khác FCD, không phải tọa độ thật tự có nhiễu.')

    page();sub('S9/S10 hiện đã cho thấy gì, còn thiếu gì?')
    p('**S9:** mặt nạ nền có Hit100=41,67%; BR tái dùng giảm xuống 0/12 ở đối thủ đã chọn, MAE tăng 120,2 → 316,0 m; Recall trong cửa sổ còn lại đạt 96,60%. Đây là dấu hiệu khả quan trong vòng cũ, chưa chứng minh chống được nhiều chuyến lặp hoặc metadata mạnh hơn.')
    p('**S10:** mặt nạ nền đã chỉ còn Hit100=8,33%; BR tái dùng là 0/12 nhưng Recall còn 88,33%. MAE 2.314,6 m có thể đồng thời phản ánh khó khăn do che biên, độ trễ của dummy trên biểu diễn cũ và sức của đối thủ hữu hạn. Không quy toàn bộ chênh lệch cho một cơ chế bảo vệ điểm cuối riêng biệt.')
    p('br_selected trong artifact là cấu hình dự phòng chọn theo quy tắc đã định khi không cấu hình nào qua ngưỡng, không phải một nghiệm đã đạt yêu cầu. Cả sáu lần chọn seed/K của paper-v2 đều không qua yêu cầu Recall tối thiểu trên mọi scenario. Vì vậy bảng trên ưu tiên hai biến thể cố định, không quảng bá tên “selected” như kết quả thắng.')
    sub('Các giới hạn cần giữ cạnh kết quả')
    bullets(['Mới chấm một cửa sổ tối đa 12 sự kiện; chưa bao phủ full transcript dài hoặc đủ A/B/C của S9/S10 hiện tại.',
      'Đối thủ cũ gồm centroid, prior, nối đường, lọc đường, giải mã toàn chuỗi và shadow kNN; chưa tái lập đầy đủ tấn công dựa vào metadata vùng che.',
      'Utility của S9/S10 chỉ trên cửa sổ được giữ. Chưa tính yêu cầu LBS trong 60 giây đã giấu, nên chưa là utility toàn chuyến.',
      'Zero hits trong 12 mẫu không phải xác suất tấn công bằng 0. Ba seed mô phỏng là bằng chứng rất hẹp; không thay bằng 12 mẫu độc lập tuyệt đối.',
      'Chi phí sinh cũ khoảng 6 ms/sự kiện và khoảng 3,2 kB payload/sự kiện của BR là phép đo máy phát triển/giao diện cũ; không so thẳng với byte ID của vòng làn mới.'])
    key('Cần thêm cơ chế điều khiển công bố ở biên, nhưng “che thêm là tốt hơn” phải được kiểm tra cùng utility, đối thủ và thời điểm ra quyết định.')
    sub('Khả năng tái lập ở lần soạn này')
    p('Đã đối chiếu 48 hàng nguồn S9/S10 của hai phương pháp với tọa độ dự đoán đã lưu và phép chiếu gốc; tính lại Recall từ tập ID POI. Máy hiện thiếu cache mạng đường và môi trường gốc để tái chạy đầy đủ SUMO/defender/attacker. Các bảng lịch sử được dẫn bằng hash trong method_evidence.json; phép audit mới không giả làm lượt chạy lại toàn bộ model.')

    page();sec('Mở rộng bảo vệ biên cho S9/S10')
    sub('Vì sao nhiễu từng vị trí chưa đủ?')
    p('Đối thủ có thể kết hợp thời điểm bắt đầu/kết thúc, đường một lối, vị trí xuất hiện đầu tiên/cuối cùng và các lần đi lặp. Khi đó các điểm giữa chuyến trông khó đoán vẫn chưa đảm bảo hai đầu kín. Cần xem lịch công bố và metadata là một phần của cơ chế, không chỉ thay tọa độ trong mỗi bản tin.')
    table(['Thành phần đề xuất','S9: đầu chuyến','S10: cuối chuyến'],[
      ['Cổng theo lịch công khai','Có thể trì hoãn bắt đầu công bố trong khoảng cố định từ khi phiên bắt đầu. Phải kiểm soát việc mốc mở phiên tự lộ điểm đầu.','Không thể biết chắc “60 giây cuối” khi xe còn đi. Không đọc đích/tương lai để quyết định online.'],
      ['Buffer / độ trễ','Không bắt buộc cho mặt nạ đầu đơn giản. Truy vấn đầu dùng cache hoặc chấp nhận không phục vụ.','Có thể giữ bản tin thêm Δ giây rồi mới gửi; khi phiên kết thúc, bỏ phần chưa gửi. Đổi lại mọi phản hồi qua nhánh đó đều trễ.'],
      ['Dịch vụ cục bộ','Cache/POI tải sẵn để phục vụ trong phần không gửi; tải cache cũng có chi phí và có thể lộ ngữ cảnh.','Cache phục vụ tại thời điểm hiện tại; không thể dùng phản hồi LSP chưa gửi như thể đã có.'],
      ['Metadata / lịch kết thúc','Chuẩn hóa đồng hồ và bỏ nhãn evaluator; khai báo rõ lịch/nhịp công bố mà đối thủ vẫn quan sát.','Cần xem xét padding hoặc phiên theo lịch cố định; chỉ xóa tọa độ không xóa được tín hiệu im lặng/kết thúc.'],
      ['Nhiều phiên lặp','S9.C: quản lý ngân sách/liên kết qua nhiều lần khởi hành.','S10.C: tránh kết thúc công bố luôn tạo cùng một dấu vết nhận diện đích.']],[4.0,6.5,6.4])
    p('Buffer áp dụng cho luồng sẽ công bố. Nếu dùng nó cho truy vấn LBS trực tiếp, độ trễ Δ là chi phí thật; nếu chỉ dùng cho chia sẻ lịch sử, nó là bài toán dịch vụ khác. Báo cáo không đánh đồng hai giao diện này.')
    sub('Ranh giới quyền truy cập của extension')
    p('Cắt theo số sự kiện/thời gian công khai trên một transcript đã bảo vệ là hậu xử lý trong mô hình cố định đó. Nhưng quyết định cắt dựa vào “đang gần nhà”, đích thật, thời gian kết thúc riêng tư hoặc thay đổi nhịp theo GPS là thêm đường truy cập bí mật: cần threat model và phân tích ngân sách mới. Chưa có định lý rằng toàn bộ cổng biên đề xuất tự thừa hưởng mọi bảo đảm của neo.')
    p('Trạng thái hiện tại: lõi BR đã hiện thực; cắt thêm cửa sổ đã chạy được dưới dạng audit ngoại tuyến; buffer, cache, padding và liên kết phiên chưa được tích hợp/đo đầu-cuối. Chúng là các nhánh cần làm, không phải thành phần đã chứng minh hiệu quả.')

    page();sub('Phép thử mới: cắt thêm cửa sổ có thực sự giúp?')
    p('Protocol được ghi trước khi tính kết quả: giữ hai phương pháp, K=5, tất cả 12 mẫu mỗi scenario; cắt thêm 0/2/4 sự kiện, tương ứng 0/40/80 giây, ngoài mặt nạ 60 giây cũ. S9 cắt đầu; S10 cắt cuối. Không sinh lại dummy, không huấn luyện và không chọn mức cắt tốt nhất sau khi xem số.')
    p('Ba đối thủ cố định: centroid tại biên, centroid trung bình cửa sổ, và ngoại suy tuyến tính từ ba sự kiện sát biên. Đối thủ biết độ dài mặt nạ theo protocol nhưng không nhận đáp án. Bảng dùng envelope mô tả: MAE thấp nhất hoặc Hit cao nhất của một đối thủ trên cả nhóm, không chọn dự đoán tốt nhất riêng cho từng người.')
    table(['Ca / cơ chế','Cắt thêm','MAE env. m ↑','Hit env. ↓','Recall giữ lại ↑','Recall cửa sổ gốc ↑'],
      [[s['scenario']+' / '+('BR' if s['method']=='br_private' else 'Nền'),str(s['extra_cut_s'])+' s',f1(s['envelope_mae_m']),pct(s['envelope_hit100']),pct(s['utility']['recall_retained']),pct(s['utility']['recall_all_original_queries'])] for s in audit['summaries']], [3.1,1.7,2.8,2.6,3.1,3.6])
    p('“Recall giữ lại” chỉ chấm các truy vấn còn gửi. “Recall cửa sổ gốc” giữ mẫu số ban đầu 12 sự kiện, truy vấn mới bị cắt nhận 0 vì audit không có cache. Mẫu số này **vẫn chưa chứa 60 giây giấu ban đầu**, nên chưa gọi là chất lượng toàn chuyến.')
    p('S9/BR: cắt thêm 40 giây làm Hit envelope tăng 0 → 16,67% trong bộ decoder thử, đồng thời Recall cửa sổ giảm 96,60 → 80,37%. S10/BR: Hit vẫn 0 nhưng Recall giảm 88,33 → 75,07 → 61,48%. Cắt thêm không đem lại bằng chứng cải thiện Hit ở bộ mẫu S10 này.')
    p('Kết quả khác bảng lịch sử do audit dùng bộ đối thủ mới và không chọn trên validation. Không so chéo MAE hai bảng để kết luận model đã tốt/xấu đi. Kết quả audit không được đưa vào Q như một lượt xác nhận độc lập.')

    page();sub('Đọc đánh đổi và xác định phép thử tiếp theo')
    figure('boundary_tradeoff','Hình 4. Phép thử thăm dò trên transcript đã lưu. Mỗi điểm dựa trên 12 chuyến; các mức cắt cùng dùng lại các chuyến đó.')
    p('Cắt dữ liệu không làm tăng lượng thông tin khả dụng cho đối thủ tối ưu: đối thủ có transcript dài cũng có thể tự bỏ điểm. Tuy nhiên ba decoder hữu hạn ở đây không làm mọi phép bỏ điểm có thể có; khi đổi cửa sổ, chúng có thể dự đoán tốt hơn. Vì thế Hit đo được tăng không phải mâu thuẫn định lý hậu xử lý, mà là dấu hiệu phải kiểm tra đối thủ kỹ hơn.')
    table(['Bước tiếp theo','Điều kiện để có kết luận mạnh hơn'],[
      ['Áp dụng lõi mới cho S9/S10 A/B/C','Giữ đầy đủ cửa sổ được đặc tả, nhãn đầu/cuối và lịch truy vấn toàn chuyến; xử lý chân trời H thay vì reset miễn phí.'],
      ['Thử cổng biên nhân quả','S9: lịch trì hoãn khởi đầu; S10: buffer hoặc luồng chia sẻ trễ. Tính cache hit, latency, byte và truy vấn mất.'],
      ['Tăng sức đối thủ','Cho phép tự bỏ phần transcript, suy từ mạng đường, metadata được phép và các phiên lặp. Chọn trên validation riêng.'],
      ['Xác nhận','Tăng nhóm độc lập và khóa protocol trước khi sinh/chấm; giữ S1–S3 để phát hiện extension gây hồi quy.']],[5.0,11.9])
    key('Đến hiện tại: có kết quả nền cho cả năm scenario; có kết quả chi tiết của lõi mới cho S1–S3; còn thiếu xác nhận lõi mới + cơ chế biên trên S9/S10.')

    # Companion sample registry; detailed coordinate tables stay outside the main report.
    D=ns['D'];first=ns['FIRST'];samples=[]
    concise={
      'S1.A':'legal_successors=4','S1.B':'legal_successors=1','S1.C':'pharmacy; distance=12,03 m; category_share=3,11%',
      'S2.A':'stop=[54,83]; duration=29 s','S2.B':'stop=[59,238]; duration=179 s','S2.C':'stops=[214,258] và [1153,1197]; 44 s/lần',
      'S3.A':'target_indices=361,381,…,581; chu kỳ=20 s','S3.B':'single_successor_fraction=0,556','S3.C':'target_indices=[361,421,481,541]; chu kỳ=60 s',
      'S4.A':'same_person=true; same_device=true','S4.B':'same_person=true; same_device=false','S4.C':'same_person=false; same_device=true',
      'S5.A':'next_edge=1055449630; future_index=106','S5.B':'next_edge=177353867#16; future_index=31',
      'S5.C':'next_edges=[1104429375,120491943]; future_indices=[122,122]',
      'S6.A':'target_indices=[693,483]; đích cách 4474,40 m','S6.B':'target_indices=[693,782]; đích cách 46,21 m',
      'S6.C':'target_slot=6; target_index=659; routine; history={routine:5, rare:1}',
      'S7.A':'query=[clinic]; intent=medical_visit; plaintext','S7.B':'query thật=[clinic]; intent=medical_visit; bó cố định 6 loại',
      'S7.C':'[pharmacy,clinic,hospital]; intent=medical_visit; chung query đầu=pharmacy',
      'S8.A':'companions=true; gần 325/466 mẫu; liên tục 324 s','S8.B':'companions=true; gần 590/590 mẫu; liên tục 589 s',
      'S8.C':'companions=false; gần 309/460 mẫu; liên tục 228 s',
      'S9.A':'target_index=0; mask=60 s; legal_access=1','S9.B':'target_indices=[0,0]; hai điểm đầu cách 281,76 m',
      'S9.C':'target_indices=[0,0]; lặp điểm đầu; mask=60 s',
      'S10.A':'target_index=817; mask=60 s; legal_access=1','S10.B':'target_indices=[693,483]; offline_hidden_endpoint',
      'S10.C':'target_indices=[592,581]; lặp điểm cuối; mask=60 s'}
    order=['S1','S2','S3','S9','S10','S5','S6','S8','S4','S7']
    for scenario in order:
        for suffix in 'ABC':
            case=scenario+'.'+suffix;r=first[case]
            slot=-1 if case=='S6.C' else 0
            sid=r['session_ids'][slot];idx=r['observed_indices'][slot][0];pt=D['traces'][sid][idx]
            obs=[]
            for session,indices in zip(r['session_ids'],r['observed_indices']):
                ix=str(indices) if len(indices)<=7 else '['+','.join(map(str,indices[:2]))+',…,'+str(indices[-1])+']'
                obs.append(session+': '+ix)
            samples.append({'case_id':case,'record_id':r['record_id'],'first_shown_point':{'session_id':sid,'fcd_index':idx,**pt},'observation_text':' '.join(obs),'label_text':concise[case]})
    (out/'printed_samples.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2)+'\n')
