"""Current method, source-backed results and conditions for their interpretation."""
from pathlib import Path
import hashlib
import json


def add_method_content(ns):
    p,key,sec,sub,page,table,eq,bullets = (ns[x] for x in ['p','key','sec','sub','page','table','eq','bullets'])
    root,out = ns['ROOT'],ns['OUT']
    base = root/'artifacts/benchmarks/research_loop'
    def load(name): return json.loads((base/name).read_text())
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    def pct(x): return f'{100*x:.2f}%'.replace('.', ',')
    def number(x, digits=1): return f'{x:.{digits}f}'.replace('.', ',')
    live, readout, attacks = (load(x) for x in ('iteration28_live_service.json','iteration28_readout.json','iteration18_expanded_attacks.json'))
    verified = load('iteration28_verification.json')
    assert verified['status'] == 'passed' and verified['sha256'] == sha(base/'iteration28_live_service.json')
    assert readout['source_sha256'] == sha(base/'iteration28_live_service.json')
    assert live['provenance']['screen_sha256'] == attacks['screen_sha256']
    assert live['provenance']['dataset_sha256'] == attacks['dataset_sha256']
    for name, expected in live['provenance']['source_sha256'].items():
        assert sha(root/name) == expected
    for item in live['shards']:
        assert sha(root/item['file']) == item['sha256']
    summary = {(s['method'],s['mode'],s['case_id']):s for s in live['summaries'] if s['probability'] == .8}
    scores = {(s['method'],s['mode']):s for s in readout['table']}
    privacy = {(s['method'],s['case_id']):s for s in attacks['summaries']}
    method = 'response_paced_slack03'
    cases = [f'S{s}.{c}' for s in (1,2,3,9,10) for c in 'ABC']
    sources = ['artifacts/benchmarks/research_loop/'+x for x in (
        'iteration28_protocol.json','iteration28_live_service.json','iteration28_readout.json',
        'iteration28_verification.json','iteration18_expanded_screening.json',
        'iteration18_expanded_attacks.json','iteration24_site_density_components.json')]
    sources += ['artifacts/datasets/research_loop_expanded_v1/dataset.json',
        'benchmark/engines/paced_slack.py','benchmark/engines/filtered_cover.py',
        'benchmark/engines/matched_filter.py','benchmark/engines/slack_progress.py',
        'benchmark/response_aware_belief.py','evaluation/live_poi.py',
        'experiments/research_loop_live_service.py','experiments/research_loop_live_readout.py',
        'experiments/verify_research_loop_live_service.py',str(Path(__file__).relative_to(root)),
        str((out/'plot_live_architecture.py').relative_to(root)),
        str((out/'figures/architecture_live.pdf').relative_to(root)),
        str((out/'figures/architecture_live.svg').relative_to(root))]
    evidence = {'status':'current_live_service_and_inherited_coordinate_attack_evidence',
        'checked_on':'2026-09-24','sources':{x:sha(root/x) for x in sources},
        'working_configuration':readout['working_configuration'],
        'historical_versions_not_pooled':True,'latest_model_S9_S10_results_available':True,
        'coordinate_attacks_reused_because_transcripts_unchanged':True,
        'new_privacy_improvement_claimed':False,'six_paper_superiority_established':False,
        'source_sessions':live['physical_source_sessions'],'records':live['target_records'],
        'case_rows':live['case_rows'],'deterministic_service_session_evaluations':live['deterministic_service_session_evaluations'],
        'independent_confirmation':False}
    (out/'method_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')

    page();sec('Phương pháp hiện tại và dịch vụ mục tiêu')
    p('**Mục tiêu:** truy hồi các POI đang khả dụng gần người dùng, đồng thời hạn chế suy luận vị trí từ chuỗi truy vấn. Thiết bị biết bản đồ và danh mục POI; máy chủ giữ trạng thái khả dụng mới nhất. Ví dụ, ứng dụng cần tìm điểm dịch vụ còn nhận khách gần vị trí hiện tại. Trạng thái trong thực nghiệm là mô phỏng, chưa phải dữ liệu hoạt động thực tế.')
    key('Cấu hình làm việc: paced + slack 0,03 + cache phản hồi; K=5 tọa độ, L=10 POI mỗi loại và mỗi tọa độ; ứng dụng chọn top-5 tại thiết bị.')
    ns['graphic'](r'\includegraphics[width=\linewidth]{figures/architecture_live.pdf}',
        '<img src="figures/architecture_live.svg" alt="Kiến trúc truy vấn POI khả dụng" style="width:100%;height:auto">',
        'Hình. Luồng của cấu hình đã đánh giá. GPS thật dùng ở khối tạo neo và lọc kết quả cục bộ; máy chủ chỉ nhận tọa độ công bố. Cache không thay đổi lịch hay nội dung truy vấn.')
    p('Lõi gồm cơ chế nhiễu xác suất, bộ lọc belief và tối ưu tổ hợp trên mạng đường; không phải deep learning đầu-cuối. Bộ chọn dùng lịch sử neo đã bảo vệ, mạng làn và POI công khai. Nó không đọc GPS thật, vận tốc thật, đích tương lai hoặc toàn bộ vector khả dụng của máy chủ.')
    p('Bản hiện tại tiếp tục công bố theo lịch cố định và được chấm qua cửa sổ của từng ca S1/S2/S3/S9/S10. Cổng cắt/buffer biên của các vòng trước **không nằm trong cấu hình tạo ra kết quả mới**. Các phép thử biên cũ được lưu riêng, tránh gán lợi ích của một cơ chế khác cho model đang báo cáo.')

    page();sub('Neo riêng tư, ngân sách và bộ chọn theo dịch vụ')
    p('Khi được phép đọc GPS, cơ chế kiểm tra có nhiễu để giữ neo cũ hoặc lấy neo mới. Bộ lọc ngân sách dự trữ chi phí xấu nhất của bước kế tiếp trước khi cho đọc. Giãn đọc theo thời gian giúp giữ ngân sách cho chuyến dài; giữa các lần đọc hoặc khi không đủ ngân sách, đầu ra tiếp tục từ trạng thái đã bảo vệ.')
    eq(r'\Pr[z_t=v\mid x_t]\propto\exp\!\left[-\frac{\varepsilon_{\rm release}}{2}d_E(x_t,v)\right],\quad d_E(x_t,z_{t-1})+\operatorname{Lap}(1/\varepsilon_{\rm test})\le\theta\Rightarrow z_t=z_{t-1}.',
       'Neo mới: xác suất giảm theo khoảng cách tới GPS. Tái dùng: quyết định dựa trên khoảng cách đã cộng nhiễu.')
    table(['Tham số / khối','Cấu hình và ý nghĩa'],[
        ['Neo và kiểm tra','ε_release=ε_test=0,01 m⁻¹; θ=200 m. Support vị trí là tập công khai, không cắt riêng quanh GPS thật.'],
        ['Ngân sách phiên','Tối đa 23 đơn vị 0,01: cận phiên 0,23 m⁻¹. Lần đầu cần 1 đơn vị; trước bước sau phải dự trữ 2. Nhánh tái dùng chi 1, nhánh tạo mới chi 2 trong phân tích transcript mở rộng.'],
        ['Giãn đọc GPS','Ít nhất 60 giây giữa các lần đọc riêng tư. Đồng hồ truy vấn dịch vụ vẫn giữ nguyên.'],
        ['Belief và miền khả thi','Belief gần đúng từ neo; mỗi track chỉ đi tới trạng thái làn hợp lệ trong thời gian đã trôi qua.'],
        ['Phủ dịch vụ và slack','Tối ưu hợp top-L theo nhu cầu top-5 từ belief. Slack 0,03 cho phép giảm tối đa 0,03 objective hiện tại để tiến về mục tiêu hữu ích; không phải cận giảm Recall thật.']],[4.0,12.9])
    eq(r'F_t(A)=\sum_{p\in\cup_{v\in A}\mathcal P_L(v)}w_t(p),\quad v_j\in\mathcal R_{j,t};\qquad F_t(A_{\rm move})\ge F_t(A_{\rm base})-0{,}03.',
       'Chọn một điểm trong miền tới được của mỗi track để phủ POI hữu ích; POI trùng chỉ tính một lần. Bước di chuyển được nới objective một lượng nhỏ.')
    p('wₜ được suy từ belief và danh mục tĩnh. Bộ chọn **chưa dự báo hoặc tối ưu trực tiếp trạng thái khả dụng hiện tại**. Máy chủ mới lọc theo trạng thái khi trả lời; thiết bị dùng phản hồi để phục vụ. Tăng Recall trên workload mới vì vậy không tự là một đóng góp tối ưu động.')
    p('Cận tọa độ lý tưởng với đồng hồ cố định chặn tỷ số xác suất bởi exp(0,23·D∞), với D∞ là độ lệch lớn nhất giữa hai chuỗi GPS. Hai phiên có thể liên kết cộng thành tối đa 0,46 m⁻¹. Ở 50 m, cận một phiên khoảng 98.716: đây không phải xác suất an toàn và không đủ để kết luận che được nhà. Chứng minh dựa trên kernel lý tưởng; bộ sinh dấu phẩy động chưa có chứng nhận pure-DP chính xác.')

    page();sub('Cache theo hiệu lực: lợi ích và điều kiện bảo toàn riêng tư')
    p('Máy chủ giữ trạng thái cố định trong từng khoảng 60 giây và gắn epoch cho phản hồi. Thiết bị giữ hợp các POI đã nhận trong epoch hiện tại; sang epoch mới thì xóa. Đây là thời hạn hiệu lực dữ liệu, độc lập về vai trò với khoảng giãn đọc GPS, dù thí nghiệm đang đặt cùng giá trị.')
    p('Ví dụ: lần trước trả {A,B,C}, lần hiện tại trả {C,D,E}. Nếu cùng epoch, thiết bị chọn từ {A,B,C,D,E}; chỉ dùng phản hồi mới sẽ bỏ quên A và B. Sang epoch mới, A và B không được coi là còn khả dụng cho đến khi được xác nhận lại.')
    eq(r'U_t=\bigcup_jR_{t,j},\qquad C_t=\bigcup_{s\le t:\,e(s)=e(t)}U_s,\qquad U_t\subseteq C_t.',
       'Uₜ: hợp phản hồi hiện tại. Cₜ: hợp phản hồi đã nhận trong cùng epoch. Thiết bị xếp hạng Cₜ theo đường từ GPS thật.')
    p('**Utility:** mọi POI trong Cₜ còn hợp lệ và Cₜ chứa Uₜ. Với cùng quy tắc khoảng cách và phá hòa, một POI thuộc top-5 chuẩn đã nằm trong Uₜ không thể bị loại bởi một POI xếp sau nó trong danh sách chuẩn. Vì vậy Recall sau cache không giảm. Lập luận cần trạng thái ổn định trong epoch, cùng bản đồ/loại POI và cùng quy tắc xếp hạng.')
    p('**Privacy:** cache không sửa tọa độ hay thời điểm truy vấn; top-5 cuối cùng chỉ ở thiết bị. Với trạng thái máy chủ độc lập với di chuyển như trong mô phỏng, phản hồi là hàm của truy vấn và dữ liệu máy chủ đã biết. Cache không phát sinh thêm thông tin tọa độ trong transcript này, nhưng cũng không làm tấn công cũ mất hiệu lực. Nếu gửi lượt nhấp, POI đã chọn, cache hit hoặc refresh tùy GPS, cần đánh giá lại kênh đó.')
    table(['Điều kiện sử dụng','Cách xử lý'],[
        ['API trả top-L theo điểm','Áp dụng K tọa độ → phản hồi → cache → xếp hạng cục bộ.'],
        ['Có epoch / thời hạn hiệu lực','Chỉ hợp phản hồi cùng phiên bản; xóa cache hoặc tái xác nhận khi hết hạn.'],
        ['Trạng thái đổi giữa epoch','Không áp dụng bảo đảm Recall đơn điệu nếu thiếu version/expiry bảo đảm tính hợp lệ.'],
        ['API tải toàn bộ trạng thái','So với bulk trước khi dùng dummy: với 419 POI, bulk có lợi thế rõ trong thí nghiệm.']],[4.8,12.1])

    page();sec('Kết quả thực nghiệm của cấu hình hiện tại')
    sub('Thiết lập, đơn vị quan sát và đối chứng')
    p('Mạng OSM khôi phục ngày 20/09 có 102.404 trạng thái làn và 419 POI thuộc sáu loại. Bộ mở rộng gồm 12 nhóm, 264 chuyến và 415 record; năm scenario ưu tiên dùng 102 chuyến nguồn và 173 record. Giữ nguyên tọa độ công bố, lịch toàn phiên và seed bảo vệ khi replay dịch vụ khả dụng. Ảnh minh họa ở mục dữ liệu thuộc bộ 393 record lịch sử, không phải nguồn của các số dưới đây.')
    p('Trạng thái mô phỏng độc lập với GPS, giữ nguyên 60 giây; p=0,8 là mức danh định, thêm p=0,5/0,95. Ba chuỗi trạng thái được chia sẻ giữa mọi phương pháp; mỗi chuyến replay từ thời gian tương đối 0. Seeds và vector trạng thái đầy đủ chỉ thuộc server/evaluator. Các phương án K5/slack được so trên dữ liệu phát triển; chọn cấu hình làm việc theo Recall trung bình dưới cùng cận ngân sách.')
    p('Recall gộp theo loại có đáp án → sự kiện → record → nhóm → ca, rồi trung bình đều 15 ca. Có đáp án nhưng trả rỗng nhận 0; loại không có POI khả dụng được đếm riêng, không gán 1. Bootstrap theo 12 nhóm, không coi điểm FCD, hai seed bảo vệ hoặc ba chuỗi trạng thái là người dùng mới.')
    sub('Utility và chi phí phản hồi ở p=0,8')
    names=[(method,'fresh','Paced + slack'),(method,'epoch_cache','Paced + slack + cache'),
        ('response_paced','epoch_cache','Paced + cache'),('fixed_K5','epoch_cache','Cố định K5 + cache'),
        ('fixed_K12','epoch_cache','Cố định K12 + cache'),('raw_current','fresh','Vị trí thật'),
        ('static_local','fresh','Danh mục tĩnh'),('stale_epoch0','fresh','Trạng thái ban đầu'),('bulk_current','fresh','Bulk trạng thái mới')]
    table(['Cấu hình','K','Recall ↑','Ca ≥90%','Byte phản hồi / sự kiện ↓'],[
        [label,str(int(scores[m,mode]['coordinate_queries_per_event'])),pct(scores[m,mode]['case_mean_recall']),
         str(scores[m,mode]['case_gates_passed'])+'/15',number(scores[m,mode]['response_body_bytes_per_event'])]
        for m,mode,label in names],[5.1,0.9,2.7,2.3,5.9])
    p('K đếm tọa độ gửi, không đếm kết nối. K5 với sáu loại là 30 truy vấn loại–tọa độ mỗi sự kiện. Byte là thân phản hồi theo schema mô phỏng; chưa có HTTP/TLS hoặc độ trễ mạng. K=0 của bulk vẫn có yêu cầu tải trạng thái. Bản đồ/danh mục dùng chung, chưa tính chi phí tải lần đầu.')
    key('Cùng K5/L10, đạt 95,60% so với fixed K5 82,07%: tăng 13,53 điểm phần trăm. Đây là lợi thế utility; fixed K5 không dùng GPS để chọn tọa độ nên không suy ra ưu thế đồng thời về privacy.')

    page();sub('Đủ 15 ca: dịch vụ mới và tấn công trên cùng transcript')
    table(['Ca','Nhóm / record','Recall +cache ↑','Fixed K5 ↑','MAE m ↑','Hit100 ↓','Hit500 ↓'],[
        [case,f"{summary[method,'epoch_cache',case]['family_count']} / {summary[method,'epoch_cache',case]['record_count']}",
         pct(summary[method,'epoch_cache',case]['recall']),pct(summary['fixed_K5','epoch_cache',case]['recall']),
         number(privacy[method,case]['metrics']['mae_m']),pct(privacy[method,case]['metrics']['hit100']),pct(privacy[method,case]['metrics']['hit500'])]
        for case in cases],[1.5,2.2,2.7,2.5,2.5,2.3,2.3])
    p('Recall dùng ba chuỗi availability và hai seed bảo vệ ở phương pháp thích nghi. MAE/Hit lấy từ vòng tấn công trên **chính tọa độ công bố này**, không nhân thêm ba lần vì đổi availability. Từng metric dùng đối thủ đã chọn trên tập phụ trợ; đối thủ tối thiểu MAE có thể khác đối thủ tối đa Hit. S3 tính lỗi các điểm trong chuỗi; cặp phiên tổng hợp đúng mục tiêu record, không suy số hit nguyên từ cột phần trăm.')
    p('Bộ học đối thủ dùng 64 nhóm phụ trợ để fit, 16 nhóm khác để chọn; có k-NN, cây và ước lượng hình học/ngoại suy. Các điểm trên chỉ đo kênh tọa độ với thời gian cửa sổ chuẩn hóa; chưa có đối thủ mới dùng epoch để suy tuổi phiên hoặc kết hợp kích thước phản hồi. Bộ đối thủ hữu hạn và bộ mở rộng đã phục vụ phát triển, chưa phải xác nhận trên dữ liệu độc lập.')
    p('**S1.C chưa đạt:** Recall 80,18%, thấp hơn fixed K5 82,88%. Miền còn đi tới được của track và belief gần đúng có thể giới hạn dịch vụ. Cache không tạo ra POI chưa từng nhận, nên chỉ nâng S1.C từ 79,64% lên 80,18%. Ngưỡng 90% áp dụng cho trung bình từng ca, không bảo đảm từng truy vấn.')
    p('**S9 còn rủi ro ở bán kính lớn:** S9.C có Hit100=0 trong bộ đối thủ đã chọn nhưng Hit500=58,33%, MAE khoảng 514 m. Không diễn giải số 0 là đã giấu được nơi xuất phát. Tấn công cùng site ở biến thể paced không slack còn cho Hit100=3/24; đây là cảnh báo riêng của biến thể ấy, không gán cho slack + cache.')
    p('**S10 phải đọc cùng raw:** S10.A raw có MAE 191 m, Hit500=100%; bản bảo vệ có MAE 1.345 m, Hit500=8,33%: có tín hiệu giảm suy luận. Riêng S10.B, raw đã có MAE 1.367 m và Hit500=12,5%, model là 1.318 m và 9,38%. Tiền tố vốn khó xác định đích; không lấy ca này làm bằng chứng chính cho defender.')

    page();sub('Lợi ích đến từ thành phần nào, có ổn định không?')
    labels={f'{method}/fresh':'Thêm cache vào paced + slack',
            'response_paced/epoch_cache':'Slack 0,03 so với không slack',
            'fixed_K5/epoch_cache':'Thích nghi so với fixed K5',
            'fixed_K12/epoch_cache':'K5 thích nghi so với fixed K12'}
    table(['So sánh','Chênh lệch Recall (đpt)','Khoảng bootstrap 95% (đpt)'],[
        [labels[c['control']],number(100*c['case_mean_delta'],2),
         number(100*c['family_bootstrap_ci95'][0],2)+' đến '+number(100*c['family_bootstrap_ci95'][1],2)]
        for c in readout['comparisons']],[7.0,4.0,5.9])
    p('Khoảng lấy từ 10.000 lần bootstrap ghép cặp theo nhóm, đã trung bình chuỗi trạng thái trong nhóm. Đây là phân tích thăm dò, chưa hiệu chỉnh nhiều phép so sánh. Khoảng của riêng slack chứa 0: chưa đủ nói slack tốt hơn ổn định. Phần cache tăng nhỏ nhưng đúng chiều; không gán toàn bộ chênh lệch 13,53 điểm phần trăm cho cache.')
    table(['Khả dụng','Slack, phản hồi hiện tại','Slack + cache','Fixed K5','Fixed K12'],[
        [pct(probability)]+[pct(sum(s['recall'] for s in live['summaries'] if s['probability']==probability and (s['method'],s['mode'])==(m,mode))/15)
            for m,mode in [(method,'fresh'),(method,'epoch_cache'),('fixed_K5','fresh'),('fixed_K12','fresh')]]
        for probability in (.5,.8,.95)],[2.5,4.5,3.3,3.3,3.3])
    p('Giảm tỷ lệ khả dụng có thể làm một số loại POI thưa và dễ phủ hơn; Recall thay đổi không đơn thuần đo sức mạnh thuật toán. So sánh phải ở cùng p, cùng world và cùng tham chiếu. Cả ba mức đều được giữ, không chọn riêng mức thuận lợi để đại diện triển khai thực.')
    bullets(['**So với bản chưa cache:** cùng tọa độ, thời điểm, byte yêu cầu/phản hồi và cận riêng tư; chỉ đổi hậu xử lý cục bộ. Tăng 0,24 điểm phần trăm ở p=0,8.',
        '**So với fixed K5:** cùng K và L, nhưng số POI trả khác; thân phản hồi tăng khoảng 1,26% (776,6 so với 767,0 byte/sự kiện). Không gọi là cùng byte tuyệt đối.',
        '**So với fixed K12:** utility thấp hơn 2,01 điểm phần trăm, dùng 5 thay vì 12 tọa độ. Cần chọn theo chi phí và yêu cầu riêng tư, không mặc định K5 thắng.',
        '**Performance và Q:** đã đo payload mô phỏng; chưa có latency đầu-cuối/điện thoại và các mốc chuẩn hóa được xác nhận. Chưa tính Q thực nghiệm mới. Nếu yêu cầu mọi ca đạt 90%, cấu hình còn thiếu S1.C.'])

    page();sec('Lập luận đóng góp và điều kiện sử dụng')
    p('Đóng góp hiện có nằm ở thiết kế hệ thống và bằng chứng định lượng: phối hợp ngân sách tọa độ, chuyển động hợp lệ, mục tiêu phủ dịch vụ và lọc cục bộ. Geo-I kết hợp dummy đã có tiền lệ như [R12]; cache và tối ưu tham lam cũng không phải nguyên lý mới. Giá trị cần chứng minh là tác dụng của cách tích hợp trên cùng dịch vụ, nguồn lực và quyền quan sát.')
    table(['Luận điểm','Bằng chứng','Giới hạn'],[
        ['Bộ chọn nhận biết dịch vụ giữ utility ở K nhỏ','95,60% so với fixed K5 82,07%; cùng K5/L10, đủ 15 ca và ba mức availability.','So với control cố định, chưa phải vượt sáu paper. S1.C thấp hơn control; privacy không đồng thời vượt fixed K5.'],
        ['Cache tăng dịch vụ, không đổi transcript','Tăng 0,24 điểm phần trăm; cùng chi phí truyền; Recall không giảm trong cùng epoch.','Cần hiệu lực phản hồi được đảm bảo; không có lợi ích privacy mới hay tuyên bố cache là thuật toán mới.'],
        ['Giảm suy luận endpoint trong một số ca','S10.A có MAE/Hit500 tốt hơn raw; đã chấm cửa sổ S9/S10 với đối thủ phụ trợ.','S9 còn Hit500 cao; S10.B raw yếu; thời gian, danh tính và click chưa được bảo vệ bởi cận tọa độ.']],[4.2,6.5,6.2])
    sub('Khi nào nên dùng?')
    p('Phạm vi phù hợp là dịch vụ **chỉ cung cấp truy vấn theo điểm**, thiết bị có thể gửi nhiều tọa độ và tự xếp hạng, phản hồi có phiên bản hoặc thời hạn hiệu lực rõ ràng. Bản đồ, loại POI, phá hòa và thời điểm truy vấn phải giống giữa đối chứng; không cho model của ta thêm trạng thái thật mà đối chứng không có.')
    p('Nếu có API tải toàn bộ trạng thái, cần so với phương án đó trước. Với 419 POI, bitmap chỉ cần 53 byte cộng 8 byte epoch mỗi refresh; bulk đạt Recall 100% mà không gửi tọa độ. Thân phản hồi trung bình 17,1 byte/sự kiện vì chỉ refresh khi đổi epoch. Đây là API khác top-L theo điểm, có lợi hơn trong phạm vi nhỏ này; không dùng giả định “bulk quá tốn” để loại nó.')
    sub('Phát biểu kết quả và bước kiểm chứng tiếp theo')
    key('Trên workload POI khả dụng mô phỏng với API theo điểm, K5/slack/cache cải thiện utility so với control cố định cùng K; thêm cache giữ nguyên transcript. Chưa xác nhận ưu thế tổng thể so với sáu paper hoặc bảo vệ đầy đủ S9/S10.')
    p('Để mở rộng kết luận, cần chạy adapter paper cùng workload và học đối thủ theo từng đầu ra; tách bản dùng tương lai hoặc khác giao diện. Sau khi chốt cấu hình, đánh giá nhóm/địa điểm độc lập. Availability tương quan với người dùng, trạng thái đổi liên tục và metadata cần kiểm tra riêng trước khi diễn giải như kết quả triển khai.')
    p('Nguồn và hash ở method_evidence.json. Đã chấm 14.688 lượt replay dịch vụ, 24.912 hàng record trên chín cấu hình trạng thái; không phải chuyến hay lần sinh nhiễu mới. Kiểm tra gồm 648 đối chiếu với Dijkstra xuôi, tính lại bảng và kiểm tra cache/cost. Nội dung và kết quả lịch sử được giữ ở archive/ cùng artifact nguồn, không cộng gộp với benchmark mới.')
