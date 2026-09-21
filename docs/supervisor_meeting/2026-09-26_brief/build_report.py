#!/usr/bin/env python3
"""Build the 26 Sep brief and traceable samples using only the Python standard library.
Run from any directory; PDF compilation is a separate XeLaTeX/Tectonic step.
"""
from pathlib import Path
from collections import Counter
import csv, hashlib, html, json, math, re
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SRC = ROOT / 'artifacts/datasets/urban_fresh_v2/dataset.json'
D = json.loads(SRC.read_text())
SHA = hashlib.sha256(SRC.read_bytes()).hexdigest()
COUNTS = Counter(r['case_id'] for r in D['records'])
FIRST = {c['case_id']: next(r for r in D['records'] if r['case_id'] == c['case_id']) for c in D['catalogue']}
assert len(D['families']) == 12 and len(D['traces']) == 264 and len(D['records']) == 393
assert len(FIRST) == 30 and sum(len(t) for t in D['traces'].values()) == 172443
sessions = {s['session_id']: s for f in D['families'] for s in f['sessions']}
samples=[]
for case, r in FIRST.items():
    views=[]
    for sid, indices in zip(r['session_ids'], r['observed_indices']):
        selected=list(dict.fromkeys(indices[:2]+indices[-2:]))
        assert all(0 <= i < len(D['traces'][sid]) for i in indices)
        views.append({'session_id':sid,'allowed_point_count':len(indices),
          'first_allowed_index':indices[0],'last_allowed_index':indices[-1],
          'device_evaluator_samples':[{'fcd_index':i,**D['traces'][sid][i]} for i in selected]})
    samples.append({'case_id':case,'case_record_count':COUNTS[case], 'record':r,
      'sampled_device_evaluator_views':views,
      'attacker_view':'Not generated here. Pass only the protected transcript through the scenario access policy; never send this record or its labels to the attacker.'})
(OUT/'data_samples.json').write_text(json.dumps({'source':str(SRC.relative_to(ROOT)), 'source_sha256':SHA,
  'purpose':'30 source-backed examples; raw coordinates and labels are evaluator/device data, not attacker inputs.',
  'examples':samples},ensure_ascii=False,indent=2)+'\n')
with (OUT/'scenario_inventory.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['case_id','records','example_record_id','family_id','session_ids','allowed_point_counts'])
    for s in samples:
        r=s['record'];w.writerow([s['case_id'],s['case_record_count'],r['record_id'],r['family_id'],';'.join(r['session_ids']),';'.join(str(len(i)) for i in r['observed_indices'])])

# The IDs below are used consistently in the survey, metrics table and bibliography.
refs=[
('R1','Yadav et al. (2024). Protecting Vehicle Location Privacy with Contextually-Driven Synthetic Location Generation (TransProtect). SIGSPATIAL.','https://arxiv.org/html/2409.09495v1','10.1145/3678717.3691211','Toàn văn tác giả; §5.1.4, Eq. 13.','recent'),
('R2','Liu, Peng, Zhou (2026). A Dummy-Based Location Privacy Protection Scheme with Semantic Correlation of Moving Paths. JKSUCIS 38:478.','https://link.springer.com/article/10.1007/s44443-026-00899-w','10.1007/s44443-026-00899-w','Toàn văn NXB; §6.3–6.5.','recent'),
('R3','Liu, Hu, Zhou (2026). Location Privacy Protection in Continuous LBSs: Enhancing Anonymity via Fake Queries. JKSUCIS 38:53.','https://link.springer.com/article/10.1007/s44443-025-00438-z','10.1007/s44443-025-00438-z','Toàn văn NXB; §6.3–6.6.','recent'),
('R4','Road Network-Aware Personalized Trajectory Protection with Differential Privacy under Spatiotemporal Correlations (2025), arXiv:2511.21020v1.','https://arxiv.org/html/2511.21020v1','arXiv:2511.21020v1','Bản tiền công bố; §VI, Eq. 26–27.','recent'),
('R5','Wang et al. (2025). CPCROK: A Communication-Efficient and Privacy-Preserving Scheme for Low-Density Vehicular Ad Hoc Networks. Future Internet 17(4):165.','https://uhra.herts.ac.uk/id/eprint/25705/1/futureinternet-17-00165.pdf','10.3390/fi17040165','Toàn văn lưu tại trường tác giả; §5.4.','recent'),
('R6','Yue, Li, Liu, Li (2025). DP-FETC: A differentially private trajectory publishing method based on feature extraction and trajectory correlation. ERA 33(11):6631–6651.','https://www.aimspress.com/aimspress-data/era/2025/11/PDF/era-33-11-293.pdf','10.3934/era.2025293','Toàn văn NXB; §4.4, §5.2, Eq. 19–21.','recent'),
('R7','Gu et al. (2025). Research on Joint Protection of LBS Location and Query Privacy in Internet of Vehicles Based on an Improved PIR Algorithm. Concurrency and Computation: Practice and Experience.','https://onlinelibrary.wiley.com/doi/abs/10.1002/cpe.70447','10.1002/cpe.70447','Chỉ xác minh tóm tắt NXB; chưa đủ định nghĩa các metric.','recent'),
('R8','Buchholz et al. (2024). SoK: Can Trajectory Generation Combine Privacy and Utility? PoPETs 2024(3):75–93.','https://petsymposium.org/popets/2024/popets-2024-0068.pdf','10.56553/popets-2024-0068','Toàn văn; khảo sát và hướng dẫn đánh giá, không phải một cơ chế đối chứng.','recent'),
('R9','Farahnakiyan, Esmaeilyfard, Javidan (2025). A proactive privacy-preserving framework for mobile trajectory sharing (PRISM). JNCA 242:104271.','https://www.sciencedirect.com/science/article/pii/S1084804525001687','10.1016/j.jnca.2025.104271','Tóm tắt/đoạn công khai của NXB; chưa xác minh công thức LPI/TDD/DC.','recent'),
('R10','Ioannou, Aktypi, Athanasopoulos (2026). From Options to Action: Evaluating Adoption of Privacy Features in Fitness-Tracking Platforms. CHI 2026.','https://elathan.github.io/papers/chi26.pdf','10.1145/3772318.3791408','Toàn văn tác giả; bảng 3, §6.4. Đo sử dụng tính năng, không đo sức chống tấn công.','recent'),
('B1','Niu et al. (2014). Achieving k-Anonymity in Privacy-Aware Location-Based Services (DLS). INFOCOM.','https://doi.org/10.1109/INFOCOM.2014.6848002','10.1109/INFOCOM.2014.6848002','Đối chứng nền; định nghĩa entropy theo hồ sơ khảo sát đã có trong repo.','historical_baseline'),
('B2','Shaham et al. (2021). Privacy Preservation in Location-Based Services: A Novel Metric and Attack Model (RDG). IEEE TMC 20(10).','https://doi.org/10.1109/TMC.2020.2993599','10.1109/TMC.2020.2993599','Đối chứng nền; entropy/chuyển tiếp/Viterbi theo hồ sơ đã có trong repo.','historical_baseline'),
('B3','Li et al. (2024 issue; online 11/09/2023). AnotherMe: A Location Privacy Protection System Based on Online Virtual Trajectory Generation. IEEE TDSC 21(4):2552–2567.','https://ieeexplore.ieee.org/document/10246991/','10.1109/TDSC.2023.3314200','Metadata/tóm tắt và mã nguồn tác giả; chưa kiểm chứng đầy đủ mẫu số/đơn vị phép đo.','requested_baseline_exception')]
refmap={r[0]:r for r in refs}
(OUT/'sources.json').write_text(json.dumps({'checked_on':'2026-09-21','recent_window':['2023-09-21','2026-09-21'],
 'scope':'Focused update, not an exhaustive systematic review. Every included source has a metrics-table row.',
 'sources':[dict(zip(['id','citation','url','doi_or_id','verification','role'],r)) for r in refs],
 'additional_primary_source':{'AnotherMe_code':'https://github.com/fang-zhiyou/AnotherMe'},
 'caveats':['R7 and R9: partial full-text access; exact definitions are not invented.',
 'R10 section 6.4 has inconsistent count/percentage (9552/19200 versus 14.52%); do not reuse that numerical result.',
 'DLS/RDG are historical controls, excluded from the recent survey window.',
 'AnotherMe first-online date is ten days before the rolling window; retained as the explicitly requested sixth comparator.']},ensure_ascii=False,indent=2)+'\n')

# Simple shared document model: identical prose and tables for LaTeX and HTML.
blocks=[]
def p(s): blocks.append(('p',s))
def key(s):blocks.append(('key',s))
def sec(s):blocks.append(('sec',s))
def sub(s):blocks.append(('sub',s))
def page():blocks.append(('page',None))
def table(headers,rows,widths=None):blocks.append(('table',(headers,rows,widths)))
def eq(tex,readable):blocks.append(('eq',(tex,readable)))
def bullets(items):blocks.append(('list',items))
def graphic(tex,svg,caption):blocks.append(('graphic',(tex,svg,caption)))
def example(case):return FIRST[case]['record_id']
def case_table(n,descs):
    table(['Ca / n','Điều kiện và điều cần phân biệt','Mẫu thật'],
      [[f'{n}.{c} / {COUNTS[n+"."+c]}',d,example(n+'.'+c)] for c,d in zip('ABC',descs)], [2.0,11.1,3.8])
def sample_point(sid,i):
    x=D['traces'][sid][i]
    return [sid,str(i),f"{x['lon']:.6f}; {x['lat']:.6f}",x['edge_id']]

sec('Khung cập nhật và dữ liệu nền')
p('**Bản chuẩn bị ngày 26/09/2026 — cập nhật theo ghi chú 19/09.** Phạm vi bảo vệ vẫn gồm S1–S10; thứ tự triển khai thay đổi. Ưu tiên chốt dữ liệu, quyền quan sát và giao thức đánh giá trước khi tối ưu phương pháp. Tài liệu được rà soát đến 21/09/2026; các thiết lập mới dưới đây là đề xuất cho vòng tiếp theo, chưa phải kết quả thực nghiệm.')
table(['Nội dung cần cập nhật','Cách xử lý trong bản này'],[
['Thứ tự phát triển','Giai đoạn I: S1, S2, S3, S9, S10 → II: S5, S6, S8 → III: S4, S7.'],
['A/B/C và mẫu dữ liệu','Giải thích đủ 30 ca; mỗi ca có số mẫu và ID nguồn. Ví dụ tọa độ, nhãn và cửa sổ được trích từ dataset.'],
['Related works và metrics','10 công trình trong cửa sổ ba năm; bảng metrics cho toàn bộ 13 nguồn được đưa vào, gồm 3 ngoại lệ đối chứng.'],
['Sáu đối chứng','DLS, RDG, TransProtect, semantic correlation, fake-query insertion và AnotherMe.'],
['Điểm tổng hợp','Ba trục P–U–F, chuẩn hóa cố định, trung bình nhân có trọng số; báo thêm từng thành phần và kiểm tra độ nhạy.']],[3.9,13.0])
sub('Ba cấp dữ liệu: nhóm tuyến → chuyến → bản ghi')
p('**12 nhóm tuyến** là 12 cấu hình tuyến có quan hệ trên cùng một mạng đường. Mỗi nhóm có tuyến gốc và các biến thể: chung đoạn đầu rồi rẽ khác, nhập tuyến từ nơi khác, dừng, quay lại, đi cùng hoặc lặp chuyến. Chúng không phải 12 loại tấn công và cũng không đơn giản là 12 tuyến độc lập.')
p('**Mỗi nhóm có 22 phiên/chuyến hoàn tất**, tổng cộng 264. Một nhóm trải trên chín ngày mô phỏng để tạo các quan hệ cần kiểm tra. **393 bản ghi** là các phép thử được rút từ những chuyến ấy: chọn cửa sổ quan sát, một hoặc nhiều chuyến, thông tin phụ trợ và đáp án. Có 172.443 điểm FCD và đủ 30 ca A/B/C.')
key('Một chuyến có thể phục vụ nhiều phép thử; một phép thử cũng có thể cần nhiều chuyến. 393 bản ghi không phải 393 chuyến độc lập.')
p('Ví dụ: chuyến u301_00 cung cấp một điểm cho S1.A, một đoạn cho S3.A và phần sau khi che đầu chuyến cho S9.A. S4.B lại cần ghép u301_00 với u301_10 để kiểm tra liên kết người khi đổi thiết bị. Có thể tạo thêm bản ghi bằng đặc tả mới, nhưng lấy nhiều cửa sổ từ cùng chuyến không tự tạo thêm bằng chứng độc lập.')
sub('Quyền nhìn thấy dữ liệu')
table(['Phía giữ dữ liệu','Nội dung được sử dụng'],[
['Thiết bị / bộ đánh giá','Tọa độ thật, FCD, nhãn người–xe–thiết bị, đáp án tương lai; dùng để sinh bảo vệ hoặc chấm theo đúng vai trò.'],
['Đối thủ','Chỉ transcript đã bảo vệ và thông tin phụ trợ được cho phép. Không nhận record_id, nhãn thật, kế hoạch tuyến hoặc toàn bộ dataset.']],[3.9,13.0])
p('Các mẫu trong báo cáo là dữ liệu của thiết bị/bộ đánh giá, chưa phải đầu ra bảo vệ. Đồng hồ được chuẩn hóa theo cửa sổ; S8 dùng mốc chung cho cặp xe. Đầy đủ 30 mẫu và dấu vết nguồn nằm trong data_samples.json; số lượng tra trong scenario_inventory.csv.')

page();sec('Kịch bản và ba ca A/B/C')
p('A/B/C thay đổi điều kiện của cùng một nhiệm vụ suy luận, không phải ba mức độ khó tăng dần. Số sau dấu “/” là số bản ghi hiện có. Mã mẫu giúp truy ngược đến đúng cửa sổ và nhãn; ví dụ đời thường chỉ minh họa ý nghĩa, không phải sự kiện có thật trong dữ liệu mô phỏng.')
sub('S1 — suy ra vị trí tại một lần gửi')
case_table('S1',[
'Một bản tin tại cạnh có ≥2 hướng đi tiếp hợp lệ: kiểm tra khi mạng đường còn nhiều khả năng.',
'Một bản tin tại cạnh chỉ có 1 hướng đi tiếp: bản đồ tự loại bớt khả năng.',
'Cách POI ≤150 m; loại POI chiếm ≤5% danh mục công khai. Hiếm theo loại địa điểm, không phải theo tần suất ghé thăm.'])
p('**Ví dụ.** Bạn tìm quán cà phê khi đang gần trường. A: đang ở đường có nhiều nhánh; B: đang trong đường một lối; C: đang gần một loại địa điểm hiếm trong khu vực. Cả ba đều hỏi “bạn đang ở đâu?”, nhưng bằng chứng bản đồ khác nhau.')
sub('S2 — suy ra nơi dừng qua nhiều lần gửi')
case_table('S2',[
'Dừng có chủ đích ≥20 giây; lấy truy vấn mỗi 5 giây.',
'Dừng ≥120 giây; lấy truy vấn mỗi 20 giây.',
'Hai lần dừng trên cùng làn, mỗi lần ≥20 giây; ở giữa thực sự di chuyển.'])
p('**Ví dụ.** Bạn chờ trước cổng trường: A chờ ngắn, B chờ lâu, C rời đi mua nước rồi quay lại. Đối thủ tận dụng tính lặp để suy ra điểm dừng. Mẫu A có 6 lần quan sát trong đoạn dừng 29 giây; mẫu B có 9 lần trong đoạn dừng 179 giây.')
sub('S3 — tái dựng đoạn đường đã đi')
case_table('S3',[
'Đoạn di chuyển ≥60 giây qua ≥3 cạnh; gửi mỗi 20 giây.',
'Ít nhất một nửa số cạnh được lấy mẫu chỉ có một hướng đi tiếp: hành lang ít lựa chọn.',
'Lấy mẫu đoạn di chuyển thưa hơn, mỗi 60 giây.'])
p('**Ví dụ.** Bạn đi từ nhà tới trường. A có chuỗi đủ dày để nối đường; B đi qua hành lang khó rẽ; C chỉ để lộ một điểm mỗi phút. Đối thủ phải tái dựng cùng loại đáp án từ mức quan sát khác nhau. Mẫu S3.A có 12 điểm; S3.C có 4 điểm của cùng đoạn u301_00.')
sub('Một tọa độ thật trong mẫu S1.A')
table(['Phiên','Chỉ số FCD','Kinh độ; vĩ độ','Cạnh đường'],[sample_point('u301_00',333)],[2.2,2.2,5.5,7.0])
p('Bộ đánh giá giữ điểm trên làm đáp án; cơ chế tạo transcript công bố từ đầu vào được phép. Không lấy tọa độ giả tự dựng làm “kết quả bảo vệ đã chạy”.')

page();sub('S9 — suy ra điểm xuất phát bị che')
case_table('S9',[
'Cạnh xuất phát chỉ một lối ra. Che 60 giây đầu, lấy phần còn lại mỗi 20 giây.',
'Hai chuyến khác điểm đầu nhưng nhập một đoạn chung; chỉ cấp từ chỗ nhập tuyến.',
'Nhiều chuyến có cùng điểm xuất phát dự kiến; che 60 giây đầu từng chuyến.'])
p('**Ví dụ.** Ứng dụng không hiện đoạn bạn rời nhà. A: chỉ có một lối ra khu dân cư; B: hai khu nhà có đường nhập chung; C: qua nhiều thứ Bảy, đoạn công bố luôn bắt đầu gần cùng một vùng. Đối thủ suy ngược nơi xuất phát từ những phần còn nhìn thấy.')
sub('S10 — suy ra điểm kết thúc bị che')
case_table('S10',[
'Cạnh kết thúc chỉ một lối vào. Che 60 giây cuối của chuyến đã hoàn tất.',
'Hai chuyến chung tiền tố nhưng tới hai đích khác nhau; giữ đoạn cuối làm đáp án ngoại tuyến.',
'Nhiều chuyến cùng điểm kết thúc dự kiến; che 60 giây cuối từng chuyến.'])
p('**Ví dụ.** Bạn đã đến trường nhưng ứng dụng giấu đoạn vào cổng. A có một cổng vào; B đoạn nhìn thấy có thể dẫn đến trường hoặc bệnh viện; C nhiều chuyến đều biến mất gần cùng một nơi. S10 hỏi nơi đã kết thúc; S6 hỏi nơi sẽ tới khi chuyến vẫn đang diễn ra.')
key('S9/S10 chấm điểm FCD đầu/cuối, chưa khẳng định đó là “nhà” hay “nơi làm việc”. Che 60 giây cũng khác che một vùng tròn trên bản đồ.')
graphic(r'''\begin{tikzpicture}[x=1cm,y=1cm,>=Latex,font=\small]
\node[anchor=east] at (1,1.1) {S9}; \draw[very thick,dashed,gray] (1.2,1.1)--(4,1.1); \draw[very thick,teal,->] (4,1.1)--(14,1.1);
\node[above] at (2.6,1.1) {Che đầu chuyến}; \node[above] at (8.9,1.1) {Phần công bố được phép};
\node[below] at (1.2,1.1) {Đáp án}; \node[below] at (4,1.1) {60 s};
\node[anchor=east] at (1,-.4) {S10}; \draw[very thick,teal] (1.2,-.4)--(11.2,-.4); \draw[very thick,dashed,gray,->] (11.2,-.4)--(14,-.4);
\node[above] at (6,-.4) {Phần công bố được phép}; \node[above] at (12.6,-.4) {Che cuối chuyến};
\node[below] at (11.2,-.4) {$T-60$ s}; \node[below] at (14,-.4) {Đáp án};
\end{tikzpicture}''', '''<svg viewBox="0 0 850 175" role="img" aria-label="S9 che đầu; S10 che cuối chuyến"><g fill="#183a45" font-size="17"><text x="0" y="43">S9</text><text x="0" y="127">S10</text><text x="80" y="21">Che đầu chuyến</text><text x="425" y="21">Phần công bố được phép</text><text x="180" y="106">Phần công bố được phép</text><text x="655" y="106">Che cuối chuyến</text><text x="55" y="72">Đáp án</text><text x="220" y="72">60 s</text><text x="635" y="162">T − 60 s</text><text x="769" y="162">Đáp án</text></g><path d="M55 40H240 M650 125H820" stroke="#777" stroke-width="4" stroke-dasharray="7 5"/><path d="M240 40H820 M55 125H650" stroke="#087f7a" stroke-width="4"/></svg>''','Hình 1. Cửa sổ của ca A/C; sơ đồ thời gian, không phải kết quả bảo vệ. Trong phần được phép, đối thủ vẫn chỉ nhận dữ liệu sau bảo vệ.')
sub('Mẫu dữ liệu: S9.A')
r=FIRST['S9.A'];inds=r['observed_indices'][0]
p(f'Bản ghi {r["record_id"]}, phiên u301_00: đáp án là FCD[0]; cửa sổ được phép chứa {len(inds)} điểm, từ FCD[{inds[0]}] đến FCD[{inds[-1]}]. Bộ phát lại đặt lần gửi đầu về thời gian tương đối 0; không tự tiết lộ thời điểm khởi hành gốc hoặc toàn bộ thời lượng chuyến.')
table(['Vai trò điểm','Chỉ số','Kinh độ; vĩ độ'],[
['Đáp án bị che','0',sample_point('u301_00',0)[2]],['Đầu cửa sổ được phép','60',sample_point('u301_00',60)[2]],['Cuối cửa sổ được phép','580',sample_point('u301_00',580)[2]]],[5.8,2.6,8.5])
p('Dịch vụ bị bỏ lỡ trong khoảng không gửi phải được ghi nhận khi đánh giá chính sách che đầu/cuối. Không chỉ chấm các truy vấn sống sót rồi kết luận utility không bị ảnh hưởng.')

page();sub('S5 — dự đoán cạnh đường kế tiếp')
case_table('S5',[
'Chỉ tiền tố trước một lượt rẽ có ≥2 lựa chọn; nhãn là cạnh kế tiếp thực sự đi vào.',
'Lượt chuyển chỉ có 1 lựa chọn; kiểm tra mức suy luận vốn đã có từ bản đồ.',
'Hai chuyến chung tiền tố tuyến kế hoạch nhưng đi vào hai cạnh khác nhau. Tốc độ và thời điểm không bắt buộc giống hệt.'])
p('**Ví dụ.** Bạn đang tới ngã rẽ gần trường. A có thể rẽ vào trường hoặc đi thẳng; B là đường chỉ đi tiếp một hướng; C hai chuyến đều đi tới ngã rẽ đó nhưng sau đó rẽ khác nhau. Không cần biết tên bạn để đoán bước tiếp theo; độ tin cậy phải được kiểm chứng, không chỉ nhìn đường rồi phỏng đoán.')
sub('S6 — dự đoán đích chưa tới')
case_table('S6',[
'Hai chuyến chung tiền tố nhưng có đích khác nhau; phần sau không được cấp.',
'Hai đích cách nhau ≤500 m: phân biệt đúng đích có thể khó dù sai số tọa độ nhỏ.',
'Sáu chuyến lịch sử: 5 tới đích thường, 1 tới đích hiếm; chỉ tiền tố của chuyến hiện tại. Lịch sử phải được bảo vệ trước khi cấp.'])
p('**Ví dụ.** Bạn thường tới trường lúc 15h thứ Bảy nhưng hôm nay có thể tới bệnh viện. A: đoạn đang đi chưa phân biệt được hai nơi; B: trường và quán cà phê rất gần nhau; C: đối thủ có lịch sử liên quan để tăng xác suất dự đoán. Lịch 15h thứ Bảy chỉ là minh họa; dataset dùng lịch tổng hợp nhiều ngày, không chứa lịch cá nhân này.')
sub('Cùng dữ liệu hình học, ba câu hỏi khác nhau')
table(['Mẫu / phần được phép','Đáp án giữ kín','Ý nghĩa'],[
['S5.C: v2-r304-013; u304_00 và u304_02; chỉ số [0,20,40,60,80,100,119] ở mỗi phiên.','Cạnh 1104429375 và 120491943, tại FCD[122].','Hai nhánh tiếp theo khác nhau.'],
['S6.A: v2-r304-014; cùng cặp và tiền tố.','FCD cuối [693,483]; hai đích cách ≈4.474,4 m.','Dự đoán đích khi còn đang đi.'],
['S10.B: v2-r304-015; cùng cặp và tiền tố.','Cùng hai điểm cuối, chuyến đã hoàn tất.','Suy ngược phần kết thúc bị giấu.']],[6.4,5.8,4.7])
p('S6.B có một mẫu với hai đích chỉ cách 46,2 m. Vì vậy Hit100 có thể coi cả hai dự đoán là gần đúng trong khi nhãn đích khác nhau: cần báo cả sai số tọa độ và độ đúng đích.')
key('S5/S6 phải có đối thủ chỉ dùng bản đồ hoặc tần suất nền. Báo cả khả năng đoán đúng tuyệt đối và phần tăng thêm do transcript công bố.')
p('Mẫu S6.C v2-r302-030 gồm u302_14…u302_20: sáu phiên lịch sử và tiền tố 7 điểm của phiên cuối. Nhãn destination_class=routine và target_index=659 nằm ở bộ đánh giá. Không cấp nhãn này hoặc FCD tương lai cho cơ chế trực tuyến/đối thủ.')

page();sub('S8 — định vị khi có dữ liệu người đi cùng')
case_table('S8',[
'Đi cùng một phần rồi tách; ở gần ≤100 m trong ≥10 giây liên tục.',
'Chung tuyến; ≥80% thời điểm đồng xuất hiện cách ≤100 m, có ≥10 giây liên tục ở gần.',
'Không gán quan hệ đồng hành nhưng tình cờ ở gần; đối chứng âm.'])
p('**Ví dụ.** Bạn và bạn học cùng đi gặp giảng viên. Dù vị trí của bạn được làm mờ, vị trí của người kia có thể giúp định vị bạn. A người kia rẽ trước trường; B đi cùng gần suốt đường; C chỉ là xe lạ chạy gần. Mẫu S8.A có 325/466 thời điểm đồng bộ ở trong 100 m; S8.C vẫn có 309/460. Gần nhau chưa đủ để kết luận quan hệ xã hội.')
p('S8 chấm cùng mục tiêu trong hai chế độ: không có và có dữ liệu đồng hành. Quy định rõ đó là dữ liệu công bố đã bảo vệ hay dữ liệu thật bị lộ. Dùng một mốc thời gian chung; không đưa hai phiên về 0 riêng rẽ làm mất độ lệch khởi hành.')
sub('S4 — liên kết danh tính giữa các phiên')
case_table('S4',[
'Hai phiên không chồng lấn, cùng người và cùng thiết bị.',
'Cùng người nhưng đổi thiết bị: liên kết người đúng, liên kết thiết bị sai.',
'Khác người nhưng dùng chung thiết bị: liên kết người sai, liên kết thiết bị đúng.'])
p('**Ví dụ.** Lịch tới trường lúc 15h thứ Bảy có thể giúp nối các phiên của bạn dù đổi điện thoại. Nhưng người khác mượn điện thoại hoặc xe của bạn không phải chính bạn. Cần chấm riêng **người, phương tiện và thiết bị**; nối hai phiên cùng người chưa đồng nghĩa biết tên thật.')
p('**Trạng thái nhãn.** Bản ghi hiện có same_person và same_device; physical_vehicle_id nằm ở bảng phiên. Chưa có trường same_vehicle trong bản ghi và chưa có thiết kế cân bằng đầy đủ các tổ hợp người–xe–máy. Vòng mới cần suy nhãn xe từ thực thể gốc, thêm cặp âm/dương phù hợp và kiểm tra không chồng lấn trước khi báo metric xe.')
sub('S7 — suy ra nội dung hoặc ý định truy vấn')
case_table('S7',[
'Gửi nguyên văn; nhãn tổng hợp gồm khám bệnh, mua sắm, đi đường dài.',
'Trộn truy vấn thật trong bó cố định 6 loại; chính sách tham chiếu, chưa phải đầu ra BR-Dummy.',
'Cùng truy vấn đầu nhưng chuỗi sau khác nhau; kiểm tra suy luận từ chuỗi.'])
p('**Ví dụ.** Một lần tìm nhà thuốc chưa chắc là đi khám; chuỗi nhà thuốc → phòng khám → bệnh viện cung cấp dấu hiệu rõ hơn. S7.C v2-r301-019 chứa đúng chuỗi này, nhãn medical_visit. Các ý định là mẫu viết tay để kiểm thử, chưa đại diện hành vi con người.')
key('Đủ 30 ca không có nghĩa đủ benchmark. Ca ít mẫu cần mở rộng theo đặc tả đã khóa; S4 cần nhãn xe, S7 cần dữ liệu nội dung phong phú hơn.')

page();sec('Related works gần đây: cơ chế và phạm vi')
p('Khảo sát tập trung vào công trình công bố trong **21/09/2023–21/09/2026**, liên quan tới công bố vị trí/quỹ đạo, nội dung, liên kết hoặc đánh giá bảo vệ. Đây là cập nhật có chọn lọc, chưa phải tổng quan hệ thống bao quát mọi paper. Mỗi nguồn được đưa vào đều có một dòng metrics ở mục sau. Liên hệ với S1–S10 là phân tích của nghiên cứu này, không phải các paper đã đánh giá đúng bộ ca A/B/C của ta.')
table(['Nguồn','Cách tiếp cận trong 1–2 dòng','Liên hệ và giới hạn'],[
['TransProtect (2024) [R1]','Học ngữ cảnh không gian–thời gian để tạo ứng viên; phối hợp cơ chế nhiễu/tối ưu để công bố vị trí thay thế.','Gần S1–S3; đầu ra không phải tập thật + dummy.'],
['Semantic correlation (2026) [R2]','Dự báo ngữ nghĩa bằng LSTM/attention, chọn dummy phù hợp chuyển tiếp đường.','S1–S3; giống ngữ nghĩa không tự bảo vệ ý định S7.'],
['Fake queries (2026) [R3]','Chèn truy vấn toàn giả giữa các lần truy vấn để gây khó nối đường qua thời gian.','S2–S3; cần tính toàn bộ bản tin chèn.'],
['Road-aware PTPPM (2025) [R4]','Cá nhân hóa nhiễu theo nhạy cảm và tương quan, dùng mạng đường để ràng buộc đầu ra.','S1–S3 và phụ thuộc thời gian; preprint, không chứng minh đã giải S5/S6 của ta.'],
['CPCROK (2025) [R5]','Dùng dự báo Kalman/RNN và thông điệp giả hỗ trợ đổi bí danh xe trong mạng thưa.','Liên kết xe S4; bối cảnh beacon VANET khác truy vấn POI.'],
['DP-FETC (2025) [R6]','Sinh quỹ đạo từ đặc trưng thống kê; xử lý điểm đầu/cuối cho quỹ đạo tương quan cao.','Gần tương quan S8 và đầu/cuối S9/S10; xuất bản ngoại tuyến.'],
['Improved PIR (2025) [R7]','Kết hợp truy hồi riêng tư theo từ khóa, BFV và tổ chức chỉ mục không gian.','Vị trí + nội dung S7; giao diện mã hóa khác dummy; mới xác minh tóm tắt.'],
['SoK (2024) [R8]','Hệ thống hóa cách kiểm tra tính hữu dụng, bảo vệ và khả năng triển khai của quỹ đạo sinh.','Cơ sở thiết kế phép đo và đối thủ; không phải đối chứng thứ bảy.'],
['PRISM (2025) [R9]','Dự báo LSTM kết hợp ánh xạ ngữ nghĩa phân cấp để bảo vệ chủ động.','Gợi ý xử lý ngữ cảnh/tương lai; chưa xác minh tác vụ dự đoán đích theo S6.'],
['Options to Action (2026) [R10]','Nghiên cứu việc sử dụng các tính năng riêng tư trên ứng dụng thể thao.','Liên quan che đầu/cuối; tỷ lệ bật tính năng không phải tỷ lệ chống tấn công.']],[3.9,6.4,6.6])
p('**Đối chứng lịch sử được tách riêng.** DLS [B1] và RDG [B2] giữ vai trò mốc nền. AnotherMe [B3] thuộc số tạp chí 2024 nhưng công bố trực tuyến 11/09/2023, sớm hơn cửa sổ 10 ngày; giữ như ngoại lệ theo yêu cầu đối chứng. Không dùng ba ngoại lệ này để chứng minh khảo sát đã cập nhật gần đây.')
p('**Khoảng trống còn lại.** Không ép quan hệ “một scenario = một paper giải quyết trọn vẹn”. Riêng S5/S6 cần kiểm chứng đầu suy luận tương lai đúng cửa sổ; tài liệu dùng dự báo bên trong cơ chế bảo vệ chưa đủ làm bằng chứng chống suy đoán đích. S8/S9/S10 cũng phải phân biệt xuất bản ngoại tuyến với dịch vụ trực tuyến.')

page();sec('Sáu phương pháp đối chứng và điều kiện so sánh')
table(['Phương pháp','Backbone và đầu vào/đầu ra','Vai trò trong benchmark'],[
['DLS [B1]','Thống kê xác suất truy vấn/entropy; nhận vị trí hiện tại và phân bố nền, công bố tập vị trí thật + dummy.','Mốc dummy thống kê; dùng nguyên lý chọn của paper, bổ sung wrapper trạng thái nếu cần và ghi rõ.'],
['RDG [B2]','Thống kê chuyển tiếp, tối ưu lựa chọn dummy qua chuỗi; đầu ra các tập ứng viên theo thời gian.','Mốc có xét liên kết thời gian; Viterbi thuộc phía tấn công/đánh giá.'],
['TransProtect [R1]','Mô hình học sâu tạo ứng viên theo ngữ cảnh, kết hợp nhiễu/tối ưu; công bố một vị trí thay thế.','Đối chứng khác loại đầu ra; không cho đối thủ “chọn phần tử thật” trong một tập không chứa nó.'],
['Semantic correlation [R2]','LSTM/attention + lọc theo đường/ngữ nghĩa; công bố thật cùng K−1 dummy.','Kiểm tra ích lợi của ràng buộc ngữ nghĩa so với thống kê.'],
['Fake-query insertion [R3]','Quy tắc tạo/chèn truy vấn toàn giả dựa trên tính nối tiếp; transcript có thêm thời điểm/bản tin.','Giữ đúng lịch chèn; không cho đối thủ biết nhãn thật/giả nội bộ.'],
['AnotherMe [B3]','Hệ thống sinh quỹ đạo ảo trực tuyến, ánh xạ POI và lập tuyến; phục vụ qua vị trí/quỹ đạo thay thế.','**Đối chứng thứ sáu.** Tái lập nhánh trực tuyến hoặc gắn nhãn adapter ngoại tuyến hiện có.']],[3.7,7.0,6.2])
sub('AnotherMe: sửa đúng vai trò và trạng thái')
p('Paper gốc mô tả hệ thống **online**; không nên gọi bản thân AnotherMe là phương pháp ngoại tuyến. Tuy nhiên adapter đang có trong repo xử lý cả đoạn/chuyến và đã được tách khỏi nhánh trực tuyến. Việc thêm tên vào bảng chưa có nghĩa đã hoàn tất một phép so sánh nhân quả với năm phương pháp còn lại. Mã tác giả: https://github.com/fang-zhiyou/AnotherMe.')
p('Cần đóng gói giao diện theo tiền tố, kiểm tra đầu ra trước thời điểm t không đổi khi sửa phần tương lai, và xác định các ca áp dụng. Nếu chỉ chạy được adapter cả đoạn, báo riêng kết quả tham chiếu ngoại tuyến; không xếp chung điểm tổng với cơ chế chỉ thấy tiền tố. Khả năng áp dụng của phương pháp gốc không suy từ việc adapter hiện tại không hỗ trợ một ca.')
sub('Sáu điều phải đồng nhất trước khi so điểm')
bullets([
'**Đáp án:** cùng vị trí/đoạn đường/nhãn cần suy ra; không đổi tác vụ giữa các phương pháp.',
'**Đầu vào và thời gian:** cùng tiền tố, lịch sử, phụ trợ; không dùng tương lai trong nhánh trực tuyến.',
'**Đầu ra và đối thủ:** đối thủ được thiết kế theo transcript thật sự công bố, nhưng chấm cùng mục tiêu.',
'**Dịch vụ:** cùng tập POI, yêu cầu top-5, lọc trên thiết bị; ghi độ sâu phản hồi L và tất cả truy vấn phát sinh.',
'**Nguồn lực:** cùng phần cứng, ranh giới đo và ngân sách; không đồng nhất K dummy, K ứng viên nội bộ và k=5 POI.',
'**Mức tái lập:** phân biệt mã gốc, tái hiện và adaptation; báo lỗi/không áp dụng, không bỏ khỏi mẫu số để tăng điểm.'])
p('Sáu đối chứng là danh sách nghiên cứu, chưa mặc định cả sáu đều bảo vệ nội dung S7 hoặc danh tính S4. Nhánh không đổi nội dung có thể được đo như đối chứng lộ nội dung, nhưng phải trình bày đúng khả năng của nó.')

page();sec('Bảng metrics gốc của toàn bộ nguồn được chọn')
p('Mũi tên mô tả hướng mong muốn **theo nghĩa metric gốc**. “CX” = chưa xác minh đủ trong nguồn đã truy cập, không có nghĩa paper chắc chắn không đo. Một số nguồn có vai trò khảo sát/hành vi nên không có ba cột tương đương thuật toán. Không lấy số đo trên dataset khác nhau để xếp hạng trực tiếp.')
table(['Nguồn','Privacy / chẩn đoán','Utility','Performance / phạm vi đã kiểm tra'],[
['R1: TransProtect','EIE ↑: sai số suy luận kỳ vọng dưới đối thủ, đơn vị khoảng cách.','Sai lệch chi phí hành trình Δc ↓, so cùng đích.','CX đối với đầy đủ byte + độ trễ đầu-cuối; §5.1.4 và Eq. 13.'],
['R2: Semantic','ASR ↑ theo điều kiện ẩn danh; DER ↑ đánh giá dummy/ngữ nghĩa.','DER không phải chất lượng top-5 POI.','Thời gian sinh ↓. Xem §6.3–6.5; cần khóa cách tính DER khi tái lập.'],
['R3: Fake queries','Số đường khó phân biệt ↑; ASR ↑ dưới các phép loại ứng viên.','CX về chất lượng POI.','Độ trễ/tạo truy vấn và RSS ↓; §6.3–6.6.'],
['R4: PTPPM','Sai số kỳ vọng của đối thủ ↑; xác suất tấn công thành công ↓.','Khoảng dịch chuyển kỳ vọng giữa thật và đầu ra ↓.','CX chi phí dịch vụ đầu-cuối; §VI, Eq. 26–27.'],
['R5: CPCROK','Tỷ lệ khớp quỹ đạo của đối thủ ρ ↓.','Không cùng tác vụ top-5 POI.','Số quỹ đạo giả tối thiểu ψ ↓; §5.4. Đây là proxy chi phí, không phải byte.'],
['R6: DP-FETC','MI ↓ giữa quỹ đạo gốc và sinh; paper trình bày bảo đảm ε-DP.','JSD ↓ cho phân bố vị trí, thời lượng và quãng đường.','Phân tích độ phức tạp; chưa có phép đo LBS đầu-cuối tương đương.'],
['R7: Improved PIR','Phân tích an toàn; metric tấn công thực nghiệm CX.','CX metric truy hồi cụ thể.','Tóm tắt nêu tổng chi phí tính toán; đơn vị/mẫu số CX.'],
['R8: SoK','Khuyến nghị kiểm tra bằng tấn công tái dựng/liên kết, bên cạnh bảo đảm hình thức.','Phân biệt metric phân bố, hình học và tác vụ downstream.','Khả năng triển khai/tính toán trong khung khảo sát; không có “điểm model SoK”.'],
['R9: PRISM','Location Privacy Index (LPI); công thức CX.','Total Distance Difference; Directional Consistency; định nghĩa CX.','Privacy processing time, thời gian phản hồi; thông tin công khai, chưa tái lập.'],
['R10: Adoption','Tỷ lệ dùng tính năng bảo vệ; không đo attacker success.','Nghiên cứu nhận thức/trở ngại sử dụng, không có Recall POI.','Không có chi phí thuật toán so sánh trực tiếp.'],
['B1: DLS','Entropy phân bố ứng viên ↑.','CX trong hồ sơ đã kiểm tra.','CX trong hồ sơ đã kiểm tra.'],
['B2: RDG','Entropy/chuyển tiếp ↑ và tỷ lệ bảo vệ dưới Viterbi ↑.','CX trong hồ sơ đã kiểm tra.','CX trong hồ sơ đã kiểm tra.'],
['B3: AnotherMe','Nhận dạng quỹ đạo thật/ảo; gần mức ngẫu nhiên là mục tiêu khi tập cân bằng.','CX định nghĩa metric dịch vụ chi tiết.','Tóm tắt nêu độ trễ đáp ứng và pin; chưa xác minh đơn vị/mẫu số đầy đủ.']],[2.8,5.1,4.2,4.8])
sub('Ba điểm dễ đọc nhầm')
p('**ASR không thống nhất giữa paper.** Ở R2, thành công là đạt điều kiện không nhận diện vị trí thật với xác suất cao hơn 1/K. R3 kiểm tra ẩn danh trước các cách khai thác khác. Không đổi tên ASR thành Hit100 hoặc dùng ASR làm “tỷ lệ tấn công thành công” mà không nêu tử số.')
p('Riêng DER ở R2 có hai cách diễn đạt: trung bình độ tương đồng và tỷ lệ dummy đạt điều kiện. Cần xác định cách thực thi trước khi tái lập, không coi hai cách là tự động tương đương. [R2]')
p('**Dummy trông thật chưa đủ.** Nhận dạng thật/ảo của AnotherMe kiểm tra tính khó phân biệt; vẫn cần đo khả năng suy ngược vị trí thật. Tương tự, MI thấp hoặc entropy cao không tự xác nhận đối thủ cụ thể thất bại. [B3, R6, R8]')
p('**Chất lượng thống kê khác chất lượng dịch vụ.** JSD bảo toàn phân bố; Δc kiểm tra chi phí đến cùng đích; Recall@5 kiểm tra đúng tập POI người dùng cần. Chúng trả lời ba câu hỏi khác nhau. [R1, R6, R8]')

page();sec('Từ hạn chế metrics gốc đến bộ đo chung')
table(['Hạn chế khi dùng trực tiếp','Quyết định cho benchmark','Lập luận'],[
['Entropy/DER/nhận dạng ảo phụ thuộc biểu diễn nội bộ.','Lấy kết quả tấn công trên đáp án thật làm privacy chính.','Một vị trí thay thế và một tập dummy đều có thể bị suy vị trí; không cần cùng cấu trúc ứng viên.'],
['MAE đơn lẻ có thể bị vài sai số lớn chi phối.','MAE + median + p90; Hit50/100/200.','Vừa thấy độ lớn sai số, phân bố, vừa thấy tỷ lệ vẫn đoán rất gần.'],
['Metric vị trí không chấm được người/đích/ý định.','Giữ đầu đo riêng cho từng loại đáp án.','Không ép F1, mét và entropy thành cùng một đại lượng riêng tư tự nhiên.'],
['Giữ phân bố hoặc điểm thay thế gần thật chưa chứng minh POI đúng.','Recall@5, tỷ lệ trả đủ, ΔD theo mạng đường.','Chấm tác vụ dịch vụ thực sự cung cấp; không chỉ chấm hình học đầu ra.'],
['Số dummy hoặc thời gian sinh chưa đủ chi phí.','Byte/truy vấn thật; latency p50/p95; số bản tin phụ.','Tính cả phản hồi, truy vấn chèn và bước gộp/lọc trên thiết bị.'],
['Một điểm tổng có thể che ca yếu.','Điểm tổng theo phạm vi khóa trước + bảng từng ca + ràng buộc tối thiểu.','Giữ cách lựa chọn gọn mà vẫn truy nguyên được đánh đổi.']],[5.3,5.1,6.5])
p('“Bộ đo đề xuất” ở đây là **sự lựa chọn và cách vận hành hóa cho bài toán**, không tuyên bố phát minh MAE, F1, Recall hoặc trung bình nhân. Đóng góp cần chứng minh là giao thức phù hợp với threat model và phát hiện được đánh đổi mà một metric đơn lẻ bỏ qua.')
sub('Privacy cho tọa độ: đo đối thủ, không đo khoảng cách nhiễu')
eq(r'e_i=d_E(x_i,\widehat x_i),\qquad \mathrm{MAE}=\frac1n\sum_i e_i,\qquad \mathrm{Hit}_r=\frac1n\sum_i\mathbf1[e_i\le r].','eᵢ = dE(xᵢ, x̂ᵢ);  MAE = trung bình eᵢ;  Hitᵣ = tỷ lệ eᵢ ≤ r.')
p('x là vị trí thật; x̂ là ước lượng của đối thủ từ dữ liệu được phép. dE tính bằng mét sau đổi hệ tọa độ phù hợp, không lấy khoảng cách Euclid trực tiếp trên kinh/vĩ độ. MAE/median/p90 càng lớn thường càng khó định vị; Hit càng thấp càng tốt về riêng tư. Nhưng p90 cao chỉ cho biết đuôi sai số lớn, không chứng minh đa số người được bảo vệ tốt.')
p('**Quy ước phân vị:** nearest rank, sắp e tăng dần rồi lấy e tại vị trí ceil(q·n), tính từ 1. Với [20,50,80,150,700] m: median=80 m, p80=150 m, p90=700 m. Quy ước cố định tránh kết quả khác nhau do phần mềm nội suy.')
sub('Privacy cho các kịch bản còn lại')
table(['Phạm vi','Chỉ số chính / bổ sung'],[
['S1–S3, S9–S10','Hit100 ↓; MAE/median/p90 ↑ và Hit50/200 ↓. S2 chấm điểm dừng; S3 chấm các thời điểm mục tiêu; S9/S10 chấm đầu/cuối.'],
['S4','Balanced accuracy ↓, precision/recall/F1 liên kết ↓; chấm người, xe, máy riêng. Mỗi đầu cần cả cặp dương và âm.'],
['S5–S6','Accuracy cạnh/đích ↓; macro-F1 nếu mất cân bằng. S6 báo thêm lỗi tọa độ; tập đích ứng viên cố định từ trước.'],
['S7','Macro-F1 ý định ↓; confusion matrix theo lớp. Tách nguyên văn và suy luận chuỗi.'],
['S8','Hit/MAE trong từng chế độ phụ trợ và chênh lệch bật–tắt; cùng tập mục tiêu, cùng đối thủ hợp lệ.']],[4.0,12.9])
p('Với mọi tác vụ, thêm S(chỉ phụ trợ) và ΔS = S(công bố + phụ trợ) − S(chỉ phụ trợ), trong đó S là độ thành công của đối thủ. S5.B có thể dễ đoán ngay từ bản đồ: ΔS nhỏ không đồng nghĩa mức riêng tư tuyệt đối cao.')

page();sub('Utility: kết quả POI còn phục vụ tốt không?')
p('Mỗi truy vấn thật i có tập tham chiếu R* gồm tối đa 5 POI gần nhất hợp lệ của đúng loại, theo cùng oracle đường có hướng. R̂ là tối đa 5 POI sau nhận phản hồi, gộp, loại trùng và lọc trên thiết bị. L là độ sâu phản hồi của máy chủ, không phải k=5 của ứng dụng.')
eq(r'\mathrm{Recall@5}_i=\frac{|R_i^\star\cap\widehat R_i|}{|R_i^\star|},\qquad C_i=\mathbf1[|\widehat R_i|=|R_i^\star|].','Recall@5ᵢ = số POI đúng được giữ / số POI tham chiếu;  Cᵢ = 1 nếu trả đủ số lượng.')
p('**Ví dụ:** tham chiếu {A,B,C,D,E}, kết quả {A,B,C,F,G}: Recall=3/5=0,6 nhưng C=1 vì vẫn đủ 5. Kết quả {A,B,C,D}: Recall=0,8, C=0. Chỉ có đáp án tham chiếu mới vào mẫu số; có đáp án nhưng trả rỗng nhận Recall=0, C=0. Báo số truy vấn không có đáp án, không âm thầm loại bỏ lỗi dịch vụ.')
eq(r'\Delta D_i=\frac{1}{|\widehat R_i|}\sum_{p\in\widehat R_i}d_G(Q(x_i),Q(p))-\frac{1}{|R_i^\star|}\sum_{p\in R_i^\star}d_G(Q(x_i),Q(p)).','ΔDᵢ = khoảng cách đường trung bình từ vị trí thật tới tập POI trả về − tới tập tham chiếu.')
p('dG là đường ngắn nhất có hướng; Q ánh xạ lên mạng đường. **Cả hai vế đều xuất phát từ vị trí thật**, không so vị trí thật với vị trí giả. Chỉ chấm ΔD khi trả đủ số lượng và đáp án hợp lệ, báo số ca đủ điều kiện. Nếu tham chiếu thật sự là các POI gần nhất cùng tập đủ điều kiện thì ΔD ≥0; giá trị âm đáng kể cần kiểm tra oracle/tập lọc, không tự cắt về 0.')
p('Ví dụ năm POI chuẩn có khoảng cách trung bình 600 m, năm POI sau bảo vệ là 850 m: ΔD=250 m. Khi chỉ trả bốn POI, không gán ΔD=0 để coi là tốt. Với S9/S10, lịch truy vấn thật phải được cố định trước chính sách che: truy vấn bị bỏ mà không được cache phục vụ vẫn tính là không phục vụ.')
sub('Performance: trả giá bao nhiêu để có mức bảo vệ đó?')
eq(r'b=\frac{B_{\rm request}+B_{\rm response}}{N_{\rm real\ queries}},\qquad m=\frac{N_{\rm messages}}{N_{\rm real\ queries}}.','b = tổng byte yêu cầu + phản hồi / số truy vấn thật;  m = số bản tin / số truy vấn thật.')
table(['Đại lượng','Ranh giới phải ghi rõ'],[
['b: byte/truy vấn thật ↓','Payload tuần tự hóa thực tế của cả yêu cầu lẫn phản hồi; gồm truy vấn chèn. Nếu chỉ tính payload, ghi rõ chưa tính TLS/TCP và không gọi là lưu lượng mạng đầy đủ.'],
['Thời gian p50/p95 ↓','Từ lúc ứng dụng có truy vấn thật đến lúc có kết quả sau lọc; bao gồm sinh bảo vệ, dịch vụ và gộp. Khi dùng oracle cục bộ, báo latency cục bộ, không suy ra trải nghiệm mạng thực.'],
['Chi phí chẩn đoán','Số tọa độ/bản tin, số POI phản hồi, thời gian sinh, RAM; tách học và tiền xử lý khỏi trực tuyến. Pin chỉ báo nếu đo trên thiết bị.']],[4.3,12.6])
key('Dữ liệu byte mã POI của vòng cũ không đủ để chấm performance tổng hợp mới. Cần đo lại cùng giao diện và cùng phần cứng.')
sub('Mẫu số và độ độc lập')
p('Chấm điểm từng mục tiêu rồi gộp theo bản ghi/chuyến, nhóm tuyến, ca A/B/C và scenario với trọng số được khóa. Giữ các phiên/cặp có quan hệ trong cùng split. Không cho S7 nhiều bản ghi hoặc chuyến dài áp đảo phần còn lại. Khoảng tin cậy lấy mẫu lại theo nhóm tuyến, không coi từng điểm FCD hoặc seed nhiễu là người độc lập.')

page();sec('Điểm tổng hợp privacy–utility–performance')
p('Điểm tổng hợp phục vụ lựa chọn cấu hình trong **cùng phiên bản benchmark và cùng phạm vi**. Đây là điểm ra quyết định theo nhu cầu, không phải bảo đảm riêng tư mới hoặc xác suất an toàn. Vòng đầu chỉ xếp hạng chung S1, S2, S3, S9, S10; các giai đoạn sau mở phiên bản điểm khác.')
sub('Bước 1: đưa ba trục về cùng chiều tốt và thang 0–1')
eq(r'P=1-\frac{1}{|\mathcal S|}\sum_{s\in\mathcal S}\frac13\sum_{c\in\{A,B,C\}}h_{s,c},\qquad U=\operatorname{MacroMean}(\mathrm{Recall@5}).','P = 1 − Hit100 trung bình đều qua ca rồi qua scenario;  U = Recall@5 gộp theo cùng phân tầng.')
p('h là Hit100 của đối thủ đã chọn trên validation cho từng phương pháp/ca; đóng băng lựa chọn trước test. Báo thêm envelope của các đối thủ đã thử như phân tích độ nhạy. P cao nghĩa tỷ lệ định vị đúng trong 100 m thấp. Không dùng MAE để tự co giãn theo model tốt/xấu nhất của bảng, vì thêm một model sẽ làm điểm các model cũ đổi theo.')
eq(r'f(v;g,b)=\operatorname{clip}\!\left(\frac{b-v}{b-g},0,1\right),\quad F=\sqrt{f(b_{\rm payload};b_g,b_b)\,f(T_{95};t_g,t_b)}.','f = 1 ở mức tốt g, giảm tuyến tính về 0 ở mức giới hạn b;  F = căn bậc hai của điểm byte × điểm latency.')
p('Trong f, g và b là mốc tốt và mốc xấu đã chốt trước, b>g. b_payload là byte/truy vấn thật, trung bình đều qua ca rồi scenario; T95 là giá trị lớn nhất trong các latency p95 từng ca. Như vậy một ca rất chậm không bị trung bình che lấp. Các mốc lấy từ yêu cầu dịch vụ/hardware hoặc pilot phát triển, không lấy từ test. Ở ví dụ dưới: byte tốt 2 kB, giới hạn 10 kB; latency tốt 50 ms, giới hạn 250 ms; 1 kB=1.000 byte. Đây chỉ là mốc minh họa.')
sub('Bước 2: kiểm tra điều kiện tối thiểu rồi xếp điểm')
eq(r'Q=100\,P^{w_P}U^{w_U}F^{w_F},\qquad w_P,w_U,w_F>0,\quad w_P+w_U+w_F=1.','Q = 100 × P^wP × U^wU × F^wF;  trọng số dương, tổng bằng 1.')
p('Đề xuất khởi đầu wP=wU=wF=1/3. Trung bình nhân khiến một trục rất yếu kéo điểm xuống; một trục bằng 0 cho Q=0. Trước khi chọn, kiểm tra Recall@5 từng ca ≥0,90 và giới hạn byte/latency đã định. Cấu hình không đạt được ghi “không khả thi”, dù vẫn có thể hiển thị điểm chẩn đoán. Còn thiếu một ca bắt buộc hoặc thiếu phép đo thì Q=NA; không tự bỏ ca rồi chia lại trọng số.')
sub('Ví dụ số — dữ liệu minh họa, không phải kết quả model')
# B has stronger efficiency, A stronger privacy. All examples are generated, not measured.
demos=[{'configuration':'A','P':.95,'U':.95,'bytes':6800,'latency_p95_ms':130},
       {'configuration':'B','P':.70,'U':.95,'bytes':3600,'latency_p95_ms':90}]
for d in demos:
    d['f_bytes']=(10000-d['bytes'])/8000
    d['f_latency']=(250-d['latency_p95_ms'])/200
    d['F']=math.sqrt(d['f_bytes']*d['f_latency'])
    for label,w in [('equal',[1/3]*3),('privacy',[.6,.2,.2]),('performance',[.2,.2,.6])]:
        d['Q_'+label]=100*math.prod(v**a for v,a in zip([d['P'],d['U'],d['F']],w))
(OUT/'score_example.json').write_text(json.dumps({'status':'illustrative_only_not_experimental_results','normalization':{'bytes_good':2000,'bytes_bad':10000,'latency_good_ms':50,'latency_bad_ms':250},'rows':demos},ensure_ascii=False,indent=2)+'\n')
table(['Cấu hình','P / U','Byte; latency p95','F','Q đều'],[[d['configuration'],f"{d['P']:.2f} / {d['U']:.2f}",f"{d['bytes']:,} B; {d['latency_p95_ms']} ms",f"{d['F']:.3f}",f"{d['Q_equal']:.1f}"] for d in demos],[2.3,3.0,5.2,2.9,3.5])
p('A làm đối thủ khó định vị hơn, nhưng B nhẹ hơn. Điểm tổng không nói A “thua về privacy”; nó nói B phù hợp hơn với bộ trọng số đang dùng. Utility trung bình 0,95 chưa đủ chứng minh mọi ca đạt 0,90; ví dụ giả định đã qua kiểm tra từng ca.')

page();sub('Trọng số thay đổi thì quyết định có đổi không?')
table(['Ưu tiên (wP, wU, wF)','Q của A','Q của B','Thứ tự'],[
['Cân bằng (1/3, 1/3, 1/3)',f"{demos[0]['Q_equal']:.1f}",f"{demos[1]['Q_equal']:.1f}",'B > A'],
['Ưu tiên privacy (0,6; 0,2; 0,2)',f"{demos[0]['Q_privacy']:.1f}",f"{demos[1]['Q_privacy']:.1f}",'A > B'],
['Ưu tiên performance (0,2; 0,2; 0,6)',f"{demos[0]['Q_performance']:.1f}",f"{demos[1]['Q_performance']:.1f}",'B > A']],[7.9,2.6,2.6,3.8])
p('Đây là đổi nhu cầu hợp lệ, không phải thay công thức sau khi thấy model nào thắng. Cần công bố hồ sơ trọng số chính trước test và giữ các hồ sơ còn lại làm phân tích độ nhạy. Thực nghiệm phải báo đủ P, U, F, Q, điều kiện khả thi và kết quả từng ca.')
sub('Mở rộng điểm sang S5/S6/S8 rồi S4/S7')
p('Giai đoạn II bổ sung độ đúng cạnh/đích và chế độ có phụ trợ; giai đoạn III bổ sung liên kết và ý định. Khi đó có thể chọn một đại lượng thành công của đối thủ cho từng đầu để tính 1−S, nhưng **không được coi 1−accuracy, 1−F1 và 1−Hit là cùng mức riêng tư tự nhiên**. Số lớp, tỷ lệ cặp dương, tập ứng viên và mức đoán nền phải được cố định. Chỉ tạo Q đa nhiệm khi đã chốt trọng số tác vụ và phép chuẩn hóa tương ứng; hiện chưa có Q chung S1–S10.')
p('S8 cần báo cả mức rủi ro tuyệt đối khi có phụ trợ và phần tăng thêm. S4 phải có ba đầu người/xe/máy; hiện còn thiếu nhãn/độ cân bằng nên chưa đủ điều kiện tạo điểm. Không gán các phần chưa đo bằng 0 hay dùng “không áp dụng” như điểm hoàn hảo.')
key('Chấp nhận một điểm kết hợp ba trục, nhưng luôn gắn với phạm vi, trọng số và điều kiện áp dụng. Không có một con số duy nhất thay thế toàn bộ phân tích S1–S10.')
sub('Quy trình đánh giá đề xuất cho vòng tiếp theo')
bullets([
'Khóa phiên bản dataset, oracle POI, các cửa sổ, tập nhãn và quyền phụ trợ; ghi hash nguồn.',
'Tách dữ liệu học, chọn cấu hình và xác nhận theo nhóm quan hệ. Dữ liệu từng dùng để chỉnh thiết kế không còn là kiểm thử mới.',
'Học/chọn đối thủ trên phần được phép; chọn cấu hình bảo vệ và trọng số trên phát triển/validation, rồi đóng băng.',
'Chạy toàn bộ cấu hình đã đăng ký, cùng seed nhiễu ghép cặp; giữ lỗi và ca thiếu, ghi nguyên nhân.',
'Báo từng ca, từng nhóm, ba trục và điểm Q; khoảng tin cậy theo nhóm tuyến. Test dùng để kết luận, không để tiếp tục tối ưu.'])
p('Tổng 12 nhóm hiện tại chỉ đủ kiểm tra khả thi ban đầu. Ca S5.C/S10.B có 4 bản ghi, S6.A có 4 và S6.B có 5. Cần tăng nhóm độc lập theo điều kiện cố định, không chọn nhóm mới vì model đạt điểm tốt. Thêm cửa sổ từ cùng chuyến không khắc phục được thiếu độc lập.')

from method_content import add_method_content
add_method_content(globals())

page();sec('Nguồn, khả năng tái lập và các điểm cần chốt')
sub('Tệp kèm báo cáo')
table(['Tệp','Vai trò'],[
['report_explained.tex / .pdf / .html','Cùng nội dung; PDF là bản đọc, LaTeX để chỉnh sửa, HTML để tra nhanh.'],
['data_samples.json','30 bản ghi nguồn cùng một số điểm FCD thật; giữ chính sách quan sát và nhãn đánh giá.'],
['scenario_inventory.csv','Đếm từng A/B/C và mã mẫu; số đếm tạo tự động từ dataset.'],
['sources.json','Danh mục đủ 13 nguồn, URL, phạm vi thời gian và mức xác minh.'],
['score_example.json','Ví dụ số minh họa tính lại được; không chứa kết quả thực nghiệm.'],
['method_evidence.json / printed_samples.json','Hash nguồn thực nghiệm và dữ liệu của 30 mẫu hiển thị trực tiếp.'],
['boundary_audit_protocol.md','Protocol phép thử cắt thêm cửa sổ; kết quả ở artifacts/benchmarks/report_boundary_audit.'],
['build_report.py','Sinh lại LaTeX/HTML và các tệp trên từ dữ liệu gốc; biên dịch PDF bằng Tectonic hoặc XeLaTeX.']],[5.3,11.6])
p('Nguồn dataset: artifacts/datasets/urban_fresh_v2/dataset.json.')
p('SHA-256: '+SHA+'.')
p('Dùng số liệu của tệp này; một số bảng lịch sử trong thesis/scenario_dataset_spec.tex nói về bộ sáu nhóm cũ, không được chép số mẫu sang bản này.')
p('Nguồn tiến độ: thesis/notes/fresh_switching_protocol.md; artifacts/benchmarks/fresh_switching/readout.json và selection.json; docs/reproduction/anotherme.md; benchmark/methods/anotherme.py. Khảo sát mới không thay nhãn historical benchmark trong repo.')
sub('Điểm còn mở được ghi rõ')
bullets([
'Chốt phạm vi I, ngưỡng Recall 0,90, mốc byte/latency và hồ sơ trọng số Q trước vòng xác nhận. Các con số chuẩn hóa trong ví dụ chưa phải yêu cầu dịch vụ được phê duyệt.',
'R7/R9 và một phần AnotherMe chưa truy cập đủ định nghĩa thực nghiệm; bảng để CX, không bù bằng suy đoán. DLS/RDG giữ metadata/metrics từ hồ sơ khảo sát hiện có.',
'Nguồn R10 có bất nhất giữa số đếm và tỷ lệ ở §6.4; chỉ sử dụng ý nghĩa chỉ số adoption, không sao lại tỷ lệ đó làm bằng chứng định lượng.',
'Kiểm thử online AnotherMe là điều kiện trước bảng so sánh chung; không dùng thông tin tương lai để làm đối chứng mạnh giả tạo.',
'Đăng ký tập xác nhận mới sau khi protocol đã khóa. Dữ liệu đã xem để chỉnh thiết kế, kể cả audit cắt biên lần này, tiếp tục thuộc phần phát triển.'])
page();sub('Tài liệu tham khảo')
for rid,citation,url,doi,verify,role in refs:
    blocks.append(('ref',(rid,citation,url,verify)))

# Formatting: citations are hyperlinks; no TeX/HTML unescaped user strings.
def esc(s):
    return ''.join({'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}','≥':r'$\ge$','≤':r'$\le$','→':r'$\to$','ₜ':r'$_t$', '⁻':r'$^{-}$','¹':r'$^1$','↑':r'$\uparrow$','↓':r'$\downarrow$','≈':r'$\approx$','Δ':r'$\Delta$','ε':r'$\varepsilon$','ρ':r'$\rho$','ψ':r'$\psi$','−':'--','×':r'$\times$','·':r'$\cdot$'}.get(c,c) for c in s)
def inline(s,tex=False):
    # Tokens are source links, bold spans, and bare URLs; recursively format bold.
    pattern=r'(\b[a-f0-9]{64}\b|\*\*.*?\*\*|\[(?:R\d+|B\d+)(?:, (?:R\d+|B\d+))*\]|https://[^\s]+)'
    result=[]
    for part in re.split(pattern,s):
        if re.fullmatch(r'[a-f0-9]{64}',part):
            result.append((r'\texttt{' + r'\allowbreak{}'.join(part[i:i+8] for i in range(0,64,8)) + '}') if tex else '<code>'+part+'</code>')
        elif part.startswith('**') and part.endswith('**'):
            inner=inline(part[2:-2],tex);result.append(r'\textbf{'+inner+'}' if tex else '<strong>'+inner+'</strong>')
        elif re.fullmatch(r'\[(?:R\d+|B\d+)(?:, (?:R\d+|B\d+))*\]',part):
            ids=part[1:-1].split(', ')
            if tex:result.append('['+', '.join(r'\hyperlink{ref-'+i+'}{'+i+'}' for i in ids)+']')
            else:result.append('['+', '.join(f'<a href="#ref-{i}">{i}</a>' for i in ids)+']')
        elif part.startswith('https://'):
            url=part.rstrip('.');suffix=part[len(url):]
            result.append((r'\url{'+url+'}'+esc(suffix)) if tex else f'<a href="{html.escape(url)}">{html.escape(url)}</a>{suffix}')
        else: result.append(esc(part) if tex else html.escape(part))
    return ''.join(result)
header=r'''% Generated by build_report.py. Edit the shared content there, then regenerate.
\documentclass[10pt,a4paper]{article}
\usepackage{fontspec}
\setmainfont{texgyretermes}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
\usepackage[margin=1.9cm,headheight=14pt]{geometry}
\usepackage{amsmath,amssymb,booktabs,tabularx,longtable,array,ragged2e,enumitem,microtype,xcolor,tikz,fancyhdr,graphicx}
\usetikzlibrary{arrows.meta,positioning}
\usepackage[hidelinks,unicode]{hyperref}
\definecolor{ink}{HTML}{183A45}\definecolor{teal}{HTML}{087F7A}\definecolor{pale}{HTML}{EAF4F2}\definecolor{gray}{HTML}{56656B}
\newcolumntype{P}[1]{>{\RaggedRight\arraybackslash}p{#1}}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\setlength{\emergencystretch}{2em}
\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.16}
\setlist{nosep,leftmargin=1.4em,topsep=4pt}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small\color{gray}BẢO VỆ RIÊNG TƯ QUỸ ĐẠO}\fancyhead[R]{\small\color{gray}26/09/2026}\fancyfoot[C]{\small\thepage}\renewcommand{\headrulewidth}{0.3pt}
\newcommand{\key}[1]{\par\smallskip\noindent\colorbox{pale}{\parbox{\dimexpr\linewidth-2\fboxsep}{\textbf{#1}}}\par\smallskip}
\hypersetup{pdftitle={Bảo vệ riêng tư quỹ đạo: dữ liệu, đối chứng và bộ đo},pdfauthor={}}
\begin{document}
\begin{center}{\LARGE\bfseries\color{ink}Bảo vệ riêng tư quỹ đạo}\\[5pt]{\large Dữ liệu kịch bản, đối chứng và bộ đo đánh giá}\\[4pt]{\small Bản 26/09/2026 · Nguồn rà soát đến 21/09/2026}\end{center}
'''
tex=[header];ht=['<header><h1>Bảo vệ riêng tư quỹ đạo</h1><p>Dữ liệu kịch bản, đối chứng và bộ đo đánh giá</p><p class="muted">Bản 26/09/2026 · Nguồn rà soát đến 21/09/2026</p><p><a href="report_explained.pdf">PDF</a> · <a href="report_explained.tex">LaTeX</a> · <a href="data_samples.json">30 mẫu dữ liệu</a> · <a href="sources.json">Nguồn</a></p></header>']
section_no=0
for kind,data in blocks:
    if kind in ('p','key'):
        tex.append(('\\key{' if kind=='key' else '')+inline(data,True)+('}' if kind=='key' else '')+'\n\n')
        ht.append(f'<p class="{kind}">{inline(data)}</p>')
    elif kind in ('sec','sub'):
        tex.append(('\\section{' if kind=='sec' else '\\subsection{')+inline(data,True)+'}\n')
        if kind=='sec':section_no+=1
        ht.append(f'<h{2 if kind=="sec" else 3}>'+ (str(section_no)+'. ' if kind=='sec' else '')+inline(data)+f'</h{2 if kind=="sec" else 3}>')
    elif kind=='page':tex.append('\n\\clearpage\n');ht.append('<hr class="pagebreak">')
    elif kind=='list':
        tex.append('\\begin{itemize}\n'+''.join('\\item '+inline(x,True)+'\n' for x in data)+'\\end{itemize}\n')
        ht.append('<ul>'+''.join('<li>'+inline(x)+'</li>' for x in data)+'</ul>')
    elif kind=='eq':
        tex.append('\\[\n'+data[0]+'\n\\]\n');ht.append('<div class="equation">'+html.escape(data[1])+'</div>')
    elif kind=='table':
        headers,rows,widths=data
        # Explicit widths below are relative weights; scale to available text width.
        n=len(headers); widths=widths or [1]*n
        avail=17.2-(n-1)*8/28.45274
        ws=[avail*w/sum(widths) for w in widths]
        spec='@{}'+''.join(f'P{{{w:.3f}cm}}' for w in ws)+'@{}'
        heading=' & '.join(r'\textbf{'+inline(h,True)+'}' for h in headers)+r'\\\midrule'+'\n'
        tex.append('\\begingroup\\small\n\\begin{longtable}{'+spec+'}\n\\toprule\n'+heading+'\\endfirsthead\n\\toprule\n'+heading+'\\endhead\n\\bottomrule\\endfoot\n')
        for i,row in enumerate(rows):tex.append(' & '.join(inline(str(x),True) for x in row)+r'\\'+ ('\n\\addlinespace[3pt]\n' if i<len(rows)-1 else '\n'))
        tex.append('\\end{longtable}\n\\endgroup\n')
        ht.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+inline(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+inline(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>')
    elif kind=='graphic':
        tex.append('\\begin{center}\n'+data[0]+'\n\\end{center}\n'+inline(data[2],True)+'\n')
        ht.append('<figure>'+data[1]+'<figcaption>'+inline(data[2])+'</figcaption></figure>')
    elif kind=='ref':
        rid,citation,url,verify=data
        tex.append(r'\par\hypertarget{ref-'+rid+'}{}'+r'\textbf{['+rid+']} '+esc(citation)+' '+r'\href{'+url+r'}{Nguồn gốc}.'+' '+esc(verify)+'\n')
        ht.append(f'<p class="reference" id="ref-{rid}"><strong>[{rid}]</strong> {html.escape(citation)} <a href="{html.escape(url)}">Nguồn gốc</a>. {html.escape(verify)}</p>')
tex.append('\\end{document}\n')
(OUT/'report_explained.tex').write_text(''.join(tex))
css='''body{margin:0;background:#f4f4f1;color:#172329;font:17px/1.55 Georgia,"Times New Roman",serif}main{max-width:1000px;margin:24px auto;background:white;padding:42px 54px}header{text-align:center;border-bottom:1px solid #b6c4c7;padding-bottom:18px}h1{font-size:31px;margin:0}h2{font-size:25px;margin-top:32px;color:#183a45}h3{font-size:20px;margin-top:24px}p{margin:12px 0}a{color:#176364;text-decoration:underline;text-underline-offset:3px}.muted{color:#56656b;font-size:15px}.key{padding:13px 16px;background:#eaf4f2;border-left:3px solid #087f7a;font-weight:bold}.table-wrap{overflow-x:auto;margin:16px 0}table{width:100%;border-collapse:collapse;font-size:15px;line-height:1.45}th,td{text-align:left;vertical-align:top;padding:10px 9px;border-bottom:1px solid #d4dddd;overflow-wrap:anywhere}th{border-top:2px solid #183a45;border-bottom:1px solid #183a45;background:#fafbfb}tr:last-child td{border-bottom:2px solid #183a45}th:first-child{min-width:105px}.equation{padding:15px;background:#f8f9f8;text-align:center;font-size:18px;overflow-wrap:anywhere}li{margin:7px 0}hr{border:0;border-top:1px solid #ccd6d7;margin:38px 0}svg{width:100%;height:auto}figure{margin:24px 0}figcaption,.reference{font-size:15px}.reference{overflow-wrap:anywhere}@media(max-width:650px){body{font-size:16px}main{margin:0;padding:24px 16px}h1{font-size:27px}h2{font-size:23px}table{min-width:620px}}@media print{body{background:white}main{margin:0;max-width:none;padding:0}.pagebreak{break-before:page;border:0;margin:0}.table-wrap{overflow:visible}tr{break-inside:avoid}a{color:inherit}header a{display:none}}'''
(OUT/'report_explained.html').write_text('<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bảo vệ riêng tư quỹ đạo — 26/09/2026</title><style>'+css+'</style></head><body><main>'+''.join(ht).replace('sample_maps.html', '<a href="sample_maps.html">sample_maps.html</a>').replace('sample_maps.pdf', '<a href="sample_maps.pdf">sample_maps.pdf</a>')+'</main></body></html>\n')
print('Generated LaTeX/HTML, 30 sample records, source registry and numerical example.')
print('Source SHA-256:',SHA)
print('Illustrative scores:',[(d['configuration'],round(d['Q_equal'],2),round(d['Q_privacy'],2)) for d in demos])
