"""Final common-service comparison with six named, qualified paper adapters."""
from pathlib import Path
import hashlib
import json
import numpy as np


def add_comparison_content(ns):
    p,key,sec,sub,page,table,bullets=(ns[k] for k in ('p','key','sec','sub','page','table','bullets'))
    root,out=ns['ROOT'],ns['OUT'];base=root/'artifacts/benchmarks/live_paper_comparison_v1'
    def load(name):return json.loads((base/name).read_text())
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    def pct(v):return '—' if v is None else f'{v*100:.2f}%'.replace('.',',')
    def num(v,n=1):return '—' if v is None else f'{v:.{n}f}'.replace('.',',')
    result=load('readout.json');check=load('verification.json');contracts=load('contract_checks.json')
    assert check['status']==contracts['status']=='passed' and check['readout_sha256']==sha(base/'readout.json')
    for path,digest in result['source_sha256'].items():assert sha(root/path)==digest,path
    scope=ns['SCOPE']
    result={**result,**scope['historical']}
    methods=['raw','dls','rdg','transprotect_markov','semantic_poi','fake_queries','anotherme_offline','ours30','ours67']
    labels={'raw':'Vị trí thật','dls':'DLS','rdg':'RDG','transprotect_markov':'TransProtect*','semantic_poi':'Semantic*',
            'fake_queries':'Fake-query*','anotherme_offline':'AnotherMe†','ours30':'Đề xuất 30','ours67':'Đề xuất 67'}
    summaries={(s['split'],s['method'],s['case_id']):s for s in result['summaries']}
    aggregates={(s['split'],s['method']):s for s in result['aggregates']}
    cases=list(ns['PRIORITY_CASES'])
    def completed(split,method):
        rows=[summaries[split,method,c] for c in cases]
        total=2*sum(s['attempted_records'] for s in rows)
        return total-sum(s['failed_privacy_runs'] for s in rows),total
    def scenario_metric(method,scenario,metric):
        values=[summaries['new_groups',method,f'{scenario}.{c}'][metric] for c in ns['suffixes'](scenario)]
        return float(np.mean([v for v in values if v is not None])) if any(v is not None for v in values) else None
    evid=json.loads((out/'method_evidence.json').read_text())
    files=[str((base/f).relative_to(root)) for f in ('protocol.json','attacker_protocol.json','readout.json','verification.json','contract_checks.json')]
    files += ['benchmark/paper_comparators.py','evaluation/live_comparison_attacks.py','evaluation/live_comparison_endpoint_attacks.py',
              'experiments/run_live_paper_comparison.py','experiments/read_live_paper_comparison.py','experiments/verify_live_paper_comparison.py',
              str(Path(__file__).relative_to(root))]
    evid['sources'].update({f:sha(root/f) for f in files})
    evid['direct_comparison']={'named_paper_adaptations':6,'online_paper_adaptations':5,'offline_reference':'AnotherMe VTGA',
        'case_rows':len(result['summaries']),'data_scope':'previously inspected development and four new-group data; not a new confirmation',
        'same_service_and_attacker_selection':True,'same_bytes':False,'faithful_six_paper_superiority':False}
    (out/'method_evidence.json').write_text(json.dumps(evid,ensure_ascii=False,indent=2)+'\n')

    page();sec('Đối chứng trực tiếp trên cùng dịch vụ')
    p('Bảng chính so phương pháp đề xuất với **DLS, RDG, TransProtect, Semantic correlation, Fake-query insertion và AnotherMe**. Các số dưới được chạy lại trong cùng bộ đánh giá. Đây là các bản thích nghi có mô tả mã và giả định; không phải số chép từ paper hoặc chứng nhận tái lập đầy đủ.')
    p('Cùng OSM/SUMO, 419 POI, sáu loại; server trả top-10 khả dụng, thiết bị chọn top-5 theo đường. Trạng thái đổi theo epoch 60 giây; thử p=0,5/0,8/0,95 và ba world seed. Tất cả được hợp phản hồi còn hiệu lực. Bảng này gửi ở mọi sự kiện, chưa che giờ hoạt động; đếm cả query giả và byte JSON thực. Client theo lịch công khai được đo riêng ở mục 10.')
    p('Fit thống kê/attacker trên nhóm 501–502; chọn attacker trên 601–602. Phạm vi 14 ca giữ 165 record từ 94 chuyến của 12 nhóm phát triển và 54 record từ 30 chuyến của bốn nhóm 901–904. Các số được tổng hợp lại từ kết quả cố định; không huấn luyện/chọn attacker mới. Đây là đối chứng phát triển, chưa phải xác nhận độc lập mới.')
    sub('Kết quả chính trên bốn nhóm 901–904, p=0,8')
    rows=[]
    for m in methods:
        a=aggregates['new_groups',m];done,total=completed('new_groups',m)
        cost=a['request_bytes_per_service_event']+a['response_bytes_per_service_event'] if a['request_bytes_per_service_event'] is not None else None
        elapsed=a['generation_ms_per_service_event']
        timing='<0,001' if elapsed is not None and 0<elapsed<.001 else num(elapsed,3)
        rows.append([labels[m],pct(a['recall_0.8']),str(a['gates_0.8'])+'/14',num(cost),timing,f'{done}/{total}'])
    table(['Phương pháp','Recall@5 ↑','Ca ≥90%','Byte / yêu cầu thật ↓','Sinh tọa độ: ms / yêu cầu ↓','Lượt record hợp lệ'],rows,[3.8,2.5,1.8,3.2,3.4,2.6])
    p('Lượt record = record × hai seed; có thể dùng chung chuyến. Chi phí tính trên các chuyến còn thuộc phạm vi, giữ toàn bộ clock và bản tin gốc của mỗi chuyến; chưa gồm HTTP/TLS. Thời gian là tính toán Python, không gồm server/RTT hoặc chuẩn bị model. Chi phí AnotherMe chỉ trên phiên sinh thành công; không xếp hạng với phương pháp trực tuyến.')
    p('**Giữ đúng đầu ra:** DLS/RDG/Semantic gửi tập có vị trí thật; TransProtect/AnotherMe trả vị trí thay thế; Fake-query thêm thời điểm gửi. Đề xuất 30/67 dùng 19/52 tọa độ. Cùng dịch vụ không có nghĩa cùng byte, cùng số tọa độ hoặc cùng bảo đảm privacy.')
    key('Fixed K5/K12 và bulk là phép kiểm tra thiết kế phụ. Chúng không thay thế sáu đối chứng nghiên cứu trong bảng chính.')

    page();sub('Privacy: suy luận đúng trong bán kính 100 m')
    p('Mỗi ô là **Hit100 ↓ / MAE (m) ↑**, trung bình đều các ca: S10 dùng A/C, các scenario khác dùng A/B/C. Gộp seed trong record rồi nhóm. MAE và Hit chọn decoder riêng trên tập chọn attacker. Đối thủ chỉ thấy tọa độ, thời gian tương đối và liên kết được phép; không thấy nhãn thật/giả hoặc đáp án.')
    table(['Phương pháp','S1','S2','S3','S9','S10'],[
        [labels[m]]+[pct(scenario_metric(m,f'S{s}','hit100'))+' / '+num(scenario_metric(m,f'S{s}','mae_m'),0) for s in (1,2,3,9,10)]
        for m in methods],[3.8,2.7,2.7,2.7,2.7,2.7])
    p('Bộ đối thủ gồm thống kê tập điểm, prior, Viterbi, ngoại suy 30/60/120 giây, kNN và Extra Trees; chọn bằng nhóm 601–602. AnotherMe chỉ gộp ca có đầu ra: S1 thiếu C; S2 thiếu cả ba; S9 thiếu B; S10 chỉ có C. Không so các ô này như cùng mẫu số. Chi tiết 14 ca: active_scope_ac_v2/readout.json.')
    raw10=scenario_metric('raw','S10','hit100')
    p('**Phép thử bốn nhóm này:** S10.A/C của vị trí thật đạt Hit100 '+pct(raw10)+', A là 0%, C là 25%. S9 chỉ B có hit. Bằng chứng endpoint ở đây chủ yếu từ S9.B/S10.C. Mục 10 dùng bộ suy luận đường mạnh hơn và 32 nhóm để kiểm tra S9, S10.A/C; không trộn hai bộ số.')
    p('Hit bằng 0 chỉ là không trúng trong các mẫu và bộ attacker đã thử. Vùng dịch vụ, lịch mở/đóng ứng dụng, account, IP và click chưa được bảo vệ toàn diện. Lập luận tọa độ độc lập GPS ở mục 8 vẫn có điều kiện; không thay thế bằng chứng tấn công thực nghiệm.')
    sub('Bản triển khai được so sánh là gì?')
    table(['Đối chứng','Thành phần chạy và giới hạn'],[
        ['DLS / RDG','DLS chọn theo entropy; RDG dùng pool DLS và entropy của trọng số max-product. Thống kê từ background SUMO; trạng thái đường thay ô lưới paper.'],
        ['TransProtect*','Lõi chọn ứng viên + nhiễu của paper; bộ dự báo Markov thay Transformer chưa có checkpoint tác giả. Không diễn giải là đã vượt model deep learning gốc.'],
        ['Semantic*','Bộ chọn theo ngữ nghĩa/chuyển tiếp; nhãn POI OSM và dự báo thực nghiệm thay dữ liệu AMap/LSTM chưa có weights.'],
        ['Fake-query*','Có chèn thật các bản tin toàn giả, lọc chuyển động theo đường và fallback DLS. Phân bố thời gian, lịch sử và ngưỡng là giả định công khai của adapter.'],
        ['AnotherMe†','Chạy VTGA với ánh xạ/lập tuyến nội bộ. Đọc cả chuyến nên chỉ là tham chiếu offline; mẫu quá ngắn không bị kéo giãn để tạo kết quả hợp lệ.']],[3.8,13.1])

    page();sub('Đủ 14 ca hiện tại: utility và tỷ lệ chạy thành công')
    p('Recall@5 ở p=0,8 trên cùng bốn nhóm; **† là tham chiếu offline**. Trong bảng này, nếu record thiếu đầu ra của một phiên, mọi yêu cầu có đáp án trong record nhận Recall=0; yêu cầu không có POI tham chiếu giữ null. Privacy thiếu đầu ra để trống. Mục 10 báo riêng giao thức tính lỗi theo từng yêu cầu/phiên.')
    table(['Ca','DLS','RDG','Trans.*','Semantic*','Fake*','Another†','Đề xuất 30','Đề xuất 67'],[
        [c]+[pct(summaries['new_groups',m,c]['recall_0.8']) for m in methods if m!='raw'] for c in cases],
        [1.4,1.6,1.6,1.9,2.0,1.7,1.9,2.2,2.2])
    sub('Đóng góp được hỗ trợ tới đâu trên năm scenario?')
    online=['dls','rdg','transprotect_markov','semantic_poi','fake_queries']
    endpoints={a['method']:a for a in scope['endpoint']['aggregates']}
    coverage=[]
    for s in (1,2,3,9,10):
        sc=f'S{s}'
        vals=[scenario_metric(m,sc,'hit100') if s<9 else endpoints[m][sc]['hit100'] for m in online]
        coverage.append([sc+(' (A/C)' if s==10 else ''),pct(min(vals))+'–'+pct(max(vals))+' → 0%',
                         '4 nhóm; kế hoạch mỗi sự kiện' if s<9 else '32 nhóm; kế hoạch theo lịch'])
    table(['Scenario','Hit100: năm adapter → đề xuất','Mức bằng chứng'],coverage,[2.6,6.7,7.6])
    p('**Đóng góp hiện có:** phối hợp chọn truy vấn theo độ phủ POI, xếp hạng GPS tại thiết bị và lịch công khai để giảm lộ vị trí/đường/endpoint trong dịch vụ trạng thái động. Đã có lợi thế thực nghiệm ở cả năm scenario; lịch được kiểm tra riêng về giờ hoạt động. S1–S3 cần chạy lại privacy trên cohort lớn; chưa gán số bốn nhóm cho phiên bản theo lịch.')
    p('Đây là đóng góp thiết kế và đánh giá trong phạm vi đã định. Chi phí truyền cao hơn; chưa chứng minh mới so với toàn bộ tài liệu, vượt sáu model gốc hoặc tối ưu ở cùng byte/latency. AnotherMe là tham chiếu offline với mẫu số khác.')

    from endpoint_calendar_content import add_endpoint_content
    add_endpoint_content(ns)
