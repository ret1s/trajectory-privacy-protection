"""Build a source-backed presentation brief; never rerun/tune protection methods.

Run from repository root: venv/bin/python docs/supervisor_meeting/2026-09-11_brief/build_report.py
The HTML is packaged by the installed Data Analytics canonical renderer separately.
"""
from collections import Counter
from hashlib import sha256
from pathlib import Path
from datetime import datetime
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--complete', action='store_true', help='Mark the finished report as complete.')
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/reports'
TITLE = 'Bảo vệ tính riêng tư về quỹ đạo cho người dùng dịch vụ dựa trên vị trí'
STAMP = datetime.now().astimezone().isoformat(timespec='seconds')

def read(path):
    return json.loads((ROOT / path).read_text())

def digest(path):
    return sha256((ROOT / path).read_bytes()).hexdigest()

D = 'artifacts/datasets/urban_fresh_v2/dataset.json'
R = 'artifacts/benchmarks/fresh_switching/readout.json'
S = 'artifacts/benchmarks/fresh_switching/selection.json'
data, results, selection = read(D), read(R), read(S)
assert len(data['families']) == 12
assert len(data['traces']) == 264
assert sum(map(len, data['traces'].values())) == 172443
assert len(data['records']) == 393
assert results['confirmation_records'] == 53
assert all(v['chosen'] is None for v in selection['method_selection_by_depth'].values())

REF = {
    'DLS': ('Niu và cs., INFOCOM 2014', 'https://doi.org/10.1109/INFOCOM.2014.6848002'),
    'ASA': ('Sun và cs., FGCS 2017', 'https://doi.org/10.1016/j.future.2016.06.017'),
    'RDG': ('Shaham và cs., TMC 2021', 'https://doi.org/10.1109/TMC.2020.2993599'),
    'TP': ('Yadav và cs., SIGSPATIAL 2024', 'https://arxiv.org/html/2409.09495v1'),
    'AM': ('Li và cs., TDSC 2024; online 2023', 'https://ieeexplore.ieee.org/document/10246991/'),
    'SC': ('Liu, Peng, Zhou, JKSUCIS 2026', 'https://link.springer.com/article/10.1007/s44443-026-00899-w'),
    'FQ': ('Liu, Hu, Zhou, JKSUCIS 2026', 'https://link.springer.com/article/10.1007/s44443-025-00438-z'),
    'MIX': ('Beresford & Stajano, 2004', 'https://www.cl.cam.ac.uk/~fms27/papers/2004-BeresfordSta-mix.pdf'),
    'FUT': ('Theodorakopoulos và cs., WPES 2014', 'https://arxiv.org/abs/1409.1716'),
    'DEST': ('Xue và cs., 2017: DesPre/CkiDel', 'https://journals.sagepub.com/doi/10.1177/1550147716685421'),
    'QUERY': ('Wu và cs., WWW 2021; online 2020', 'https://link.springer.com/article/10.1007/s11280-020-00830-x'),
    'CO': ('Olteanu và cs., TMC 2017', 'https://www.mhumbert.com/publications/tmc16.pdf'),
    'GAME': ('Olteanu và cs., PoPETs 2019', 'https://petsymposium.org/popets/2019/popets-2019-0017.pdf'),
    'END': ('Dhondt và cs., CCS 2022', 'https://people.cs.kuleuven.be/~stijn.volckaert/papers/2022_CCS_Fitness_Tracking.pdf'),
    'Q': ('Shokri và cs., S&P 2011', 'https://doi.org/10.1109/SP.2011.18'),
    'SUMO': ('SUMO: FCD', 'https://sumo.dlr.de/docs/Simulation/Output/FCDOutput.html'),
}

def cite(key):
    name, url = REF[key]
    return f'[{name}]({url})'

manifest = dict(version=1, surface='report', title=TITLE,
    description='Nội dung trao đổi với GVHD, 11/09/2026. Khép các yêu cầu cũ trước khi trình bày kết quả mới.',
    generatedAt=STAMP, blocks=[], charts=[], tables=[], sources=[])
datasets = {}
pages = []

for sid, label, path in [
    ('literature', 'Đối chiếu công trình gốc, phạm vi và chỉ số', 'docs/supervisor_meeting/2026-09-11_brief/source_notes.md'),
    ('dataset', 'Bản ghi urban-fresh-v2, SUMO + OSM', D),
    ('result', 'Kết quả xác nhận fresh_switching', R),
    ('selection', 'Quyết định chọn đã khóa trên validation', S),
    ('method', 'Đặc tả phương pháp và thí nghiệm', 'thesis/fresh_switching_comparison.tex'),
]:
    manifest['sources'].append({'id': sid, 'label': label, 'path': path})

def md(bid, body, source=None, new_page=False):
    block = dict(id=bid, type='markdown', body=body)
    if source:
        block['sourceId'] = source
    manifest['blocks'].append(block)
    if new_page:
        pages.append(bid)

def diagram(bid, items, caption):
    """A concept flow in the canonical HTML block, not a data-chart runtime."""
    cells = []
    colors = ['#E6EFF9', '#FBEAD2', '#E6EFF9', '#F3F4F6', '#E6EFF9']
    for index, (title, text) in enumerate(items):
        if index:
            cells.append('<td style="width:3%;text-align:center;font-size:24px;color:#334155">→</td>')
        cells.append(f'<td style="background:{colors[index % len(colors)]};border:1px solid #94A3B8;border-radius:8px;padding:13px 11px;vertical-align:top;color:#172235"><strong>{title}</strong><br><span style="font-size:14px">{text}</span></td>')
    manifest['blocks'].append(dict(id=bid, type='html', body=(
        '<div style="overflow-x:auto"><table style="width:100%;table-layout:fixed;border-spacing:4px"><tbody><tr>'
        + ''.join(cells) + '</tr></tbody></table></div>'
        + f'<p style="font-size:13px;color:#475569">{caption}</p>')))

def table(bid, title, rows, columns, source):
    datasets[bid] = rows
    manifest['tables'].append(dict(id=bid, title=title, dataset=bid, sourceId=source,
        defaultSort={'field': columns[0][0], 'direction': 'asc'},
        columns=[dict(field=k, label=v, type='text') for k, v in columns]))
    manifest['blocks'].append(dict(id=bid+'-block', type='table', tableId=bid))

md('title', '# '+TITLE)
md('overview', '''## Nội dung trình bày ngày 11/09/2026

**Phần 1 - Hoàn thiện yêu cầu tuần trước:** phân biệt 10 kịch bản; nối từng kịch bản với nghiên cứu bảo vệ; chốt danh sách đối chứng và chỉ số gốc.

**Phần 2 - Bằng chứng mới:** cơ chế sinh điểm giả có trạng thái, quy trình thực nghiệm đã khóa và đánh đổi giữa chất lượng POI với rủi ro suy luận.

**Kết luận chính:** đã có khung dữ liệu và phép thử có thể kiểm tra lại. Chưa chứng minh bảo vệ toàn bộ 10 kịch bản hoặc vượt các phương pháp đối chứng hiện đại.''')
diagram('architecture', [
    ('THIẾT BỊ', 'Vị trí, lịch sử, truy vấn thật; chỉ đọc tới hiện tại.'),
    ('CƠ CHẾ BẢO VỆ', 'Chạy cục bộ; nhận thêm bản đồ, luật đường và POI công khai.'),
    ('LSP / ĐỐI THỦ', 'Chỉ nhận dữ liệu đã bảo vệ; xử lý dịch vụ nhưng có thể lưu và suy luận.'),
    ('THIẾT BỊ', 'Nhận POI; gộp, loại trùng và xếp lại bằng vị trí thật cục bộ.')
], 'Ranh giới tin cậy: vị trí thật và trạng thái bảo vệ ở thiết bị. Mũi tên sang LSP là kênh công bố cần đánh giá; sơ đồ mô tả giao tiếp, không phải bảo đảm đã che mọi trường dữ liệu.')
md('scope', '**Phạm vi:** xe con trên mạng đường đô thị có hướng. Dữ liệu SUMO + OpenStreetMap, không dùng GeoLife. Một chuyến có thể tạo nhiều bài kiểm tra với quyền quan sát khác nhau; bản ghi không đồng nghĩa người dùng độc lập.')

scenario_rows = [
('S1', 'Định vị một lần gửi', 'Vị trí hiện tại; chỉ một bản tin.', 'A: nhiều hướng; B: một hướng; C: gần POI hiếm.', 'DLS chọn dummy theo xác suất truy vấn; đối chứng trực tiếp về suy luận điểm. [DLS]', 'Lõi; đã chấm vòng mới.'),
('S2', 'Suy luận điểm dừng', 'Nơi đang dừng; nhiều bản tin trong cùng điểm dừng.', 'A: dừng ≥20 s / gửi 5 s; B: ≥120 s / gửi 20 s; C: dừng - đi - quay lại.', 'ASA chống thống kê dài hạn/vùng bằng SNAME/MNAME và phân bố dummy. Liên quan, không đúng nguyên phép thử dừng. [ASA]', 'Lõi; đã chấm vòng mới.'),
('S3', 'Tái dựng đường đã đi', 'Chuỗi quá khứ; ghép bản tin theo bản đồ và thời gian.', 'A: chạy ≥60 s, ≥3 cạnh; B: hành lang ít nhánh; C: gửi thưa 60 s.', 'RDG chống Viterbi; TransProtect xét đường; semantic correlation xét ngữ nghĩa; fake queries chèn bản tin. [RDG, TP, SC, FQ]', 'Lõi; đã chấm vòng mới.'),
('S4', 'Liên kết người / thiết bị', 'Hai phiên có cùng người hay thiết bị? Chấm riêng hai nhãn.', 'A: cùng người/thiết bị; B: cùng người đổi máy; C: khác người dùng chung máy.', 'Mix zones đổi bí danh để làm khó liên kết. AnotherMe tạo hồ sơ ảo nhưng chưa có bằng chứng cho đúng hai nhãn ở đây. [MIX, AM]', 'Mở rộng; đã có dữ liệu.'),
('S5', 'Dự đoán cạnh tiếp theo', 'Cạnh đường ngay sau tiền tố; không phải vị trí sau đúng 30 s.', 'A: nhiều lựa chọn rẽ; B: chỉ một lựa chọn; C: chung tiền tố, khác cạnh tiếp.', 'LPPM của Theodorakopoulos và cs. bảo vệ hiện tại - tương lai theo mô hình chuyển động. Khác tác vụ dự đoán cạnh xe đô thị. [FUT]', 'Mở rộng; đã có dữ liệu.'),
('S6', 'Dự đoán đích đến', 'Đích chưa tới; chỉ thấy tiền tố và lịch sử được cấp.', 'A: chung đầu, khác đích; B: hai đích ≤500 m; C: đích quen / hiếm qua nhiều ngày.', 'CkiDel xóa check-in lịch sử để chống DesPre; có đánh giá suy luận đích. Khác dữ liệu xe trực tuyến và không sinh dummy. [DEST]', 'Mở rộng; đã có dữ liệu.'),
('S7', 'Suy luận nhu cầu truy vấn', 'Ý định thật qua nội dung và chuỗi yêu cầu.', 'A: gửi nguyên văn; B: trộn sáu loại; C: cùng truy vấn đầu, khác chuỗi sau.', 'Wu và cs. sinh chuỗi truy vấn giả để che vị trí và thuộc tính truy vấn. Ngữ nghĩa địa điểm đơn thuần chưa che được văn bản. [QUERY]', 'Mở rộng; nhãn tổng hợp.'),
('S8', 'Suy luận qua người đi cùng', 'Vị trí mục tiêu khi có thêm dữ liệu của người đồng hành.', 'A: đi cùng rồi tách; B: cùng tuyến; C: tình cờ gần nhau, không đồng hành.', 'Olteanu và cs. lượng hóa rò rỉ và mô hình hóa quyết định chia sẻ. Chưa xác minh bộ sinh dummy bảo vệ đúng S8. [CO, GAME]', 'Mở rộng; chưa có đối chứng trực tiếp.'),
('S9', 'Suy luận điểm xuất phát', 'Điểm đầu bị che; suy ngược từ đoạn công bố.', 'A: đầu chỉ một lối ra; B: khác đầu rồi nhập tuyến; C: lặp điểm đầu.', 'EPZ che biên; Dhondt và cs. đánh giá cả tấn công và biện pháp giảm rò siêu dữ liệu khoảng cách. Khác xe đô thị. [END]', 'Có phép thử biên cũ; vòng mới chỉ dữ liệu.'),
('S10', 'Suy luận điểm kết thúc', 'Điểm cuối bị che sau khi chuyến đã hoàn tất; khác S6 dự báo.', 'A: cuối chỉ một lối vào; B: chung đầu rồi tách; C: lặp điểm cuối.', 'Cùng họ EPZ và biện pháp xử lý siêu dữ liệu. Che 60 s trong bộ thử không tái lập nguyên EPZ hình học. [END]', 'Có phép thử biên cũ; vòng mới chỉ dữ liệu.'),
]
cols = [('id','Ca'),('target','Thông tin đối thủ muốn biết'),('cases','Các trường hợp dữ liệu A / B / C'),('related','Bảo vệ đã được nghiên cứu và giới hạn'),('status','Bằng chứng của ta')]
for i, group in enumerate((scenario_rows[:5],scenario_rows[5:])):
    md(f'scenarios-{i}', '## 1. Kịch bản, dữ liệu và nghiên cứu bảo vệ' + (' (S1-S5)' if not i else ' (S6-S10)') +
       '\n\n**Đọc theo hàng:** mục tiêu → điều kiện tạo dữ liệu → cách bảo vệ liên quan → mức bằng chứng hiện có. S1-S10 là cách tổ chức của luận văn, không phải một hệ phân loại chuẩn của các paper.', new_page=True)
    rows = [dict(id=x[0],target=x[1]+'. '+x[2],cases=x[3],related=x[4],status=x[5]) for x in group]
    table(f'scenarios-table-{i}', 'Ánh xạ kịch bản và bằng chứng', rows, cols, 'literature')
    keys = ['DLS','ASA','RDG','TP','SC','FQ','MIX','AM','FUT'] if not i else ['DEST','QUERY','CO','GAME','END']
    md(f'scenario-refs-{i}', '**Nguồn:** ' + ' · '.join(f'[{k}] {cite(k)}' for k in keys))
    if i:
        md('coverage-conclusion', '**Không đánh dấu “đã bao phủ” chỉ vì có liên quan.** Muốn kết luận hơn đối chứng phải chạy cùng mục tiêu, quyền quan sát, chất lượng dịch vụ và chi phí. Bỏ trống bằng chứng bảo vệ S8 là một thiếu hụt cần xử lý, không chứng minh lĩnh vực chưa có lời giải.')

md('dataset-page', '## 2. Dataset: tách chuyển động, kịch bản và phép đo\n\n**12 nhóm tuyến mới; 264 chuyến; 172.443 điểm FCD; 393 bản ghi.** Hai tập mới đều có đủ 30 ca con, nhưng không phải ca nào cũng nhiều mẫu.', 'dataset', True)
diagram('data-flow', [
    ('1. THIẾT KẾ', 'OSM có hướng; tuyến chung / rẽ khác; điểm dừng; lịch nhiều ngày.'),
    ('2. SUMO', 'Chạy xe thật trong mô phỏng; ghi tọa độ, tốc độ, làn và thời gian mỗi giây.'),
    ('3. LỚP KỊCH BẢN', 'Chọn cửa sổ; gắn truy vấn, người/thiết bị, quan hệ và đáp án.'),
    ('4. KIỂM TRA / LƯU', 'Chấp nhận hoặc ghi lý do loại; SQLite có phiên bản và nhật ký.')
], 'Một bản ghi có thể tham chiếu nhiều chuyến. Đáp án và tương lai chỉ dành cho bộ đánh giá; không đưa vào quan sát của LSP.')
counts = Counter((r['scenario'],r['split'],r['case_id'][-1]) for r in data['records'])
data_rows=[]
for n in range(1,11):
    case=f'S{n}'
    row={'order':n,'scenario':case}
    for label,split in [('validation','fresh_validation'),('confirmation','fresh_confirmation')]:
        row[label]=' / '.join(str(counts[case,split,c]) for c in 'ABC')
    data_rows.append(row)
table('data-counts','Số bản ghi theo ca con A / B / C',data_rows,
      [('order','STT'),('scenario','Kịch bản'),('validation','Tập chọn: 301-306'),('confirmation','Xác nhận: 307-312')],'dataset')
md('dataset-caveat', '**Điều kiện và giới hạn:** tốc độ cấu hình tối đa 8 m/s; lịch truy vấn khác nhịp FCD. Có 43 cặp nhóm-ca không đạt điều kiện; hai cửa sổ dừng trùng tọa độ giữa hai tập được công bố và kiểm tra độ nhạy. Đổi làn đã được sửa ở cấu hình mô phỏng, không sửa tọa độ. Cùng một đô thị; nhãn ý định/danh tính là tổng hợp. [Nguồn FCD SUMO](https://sumo.dlr.de/docs/Simulation/Output/FCDOutput.html).', 'dataset')

md('comparator-page', '''## 3. Chốt đối chứng cho thiết kế benchmark

**Ba phương pháp hiện đại chính:** TransProtect, semantic correlation và fake-query insertion. **Hai nguồn nền:** DLS/enhanced-DLS và RDG. **AnotherMe giữ nhánh tham chiếu quỹ đạo ảo.** Đây là quyết định lựa chọn cho milestone này, không phải tuyên bố tất cả đã được tái lập và chạy trên tập mới.''', new_page=True)
comps = [
('1','TransProtect · 2024','GCN + Transformer xếp hạng vị trí, rồi công bố một vị trí thay thế; đối chiếu ngữ cảnh đường.','EIE: sai số suy luận kỳ vọng (km) ↑; sai lệch chi phí hành trình kỳ vọng ↓.','Có bản thích nghi; lượt cũ dùng Markov thay Transformer. Chưa có kết quả SOTA đầy đủ. [TP]'),
('2','Semantic correlation · 2026','LSTM + attention hỗ trợ chọn K−1 dummy hợp lý về thời gian/ngữ nghĩa; gửi cùng điểm thật.','ASR ẩn danh thành công ↑; DER hiệu quả dummy ↑; thời gian sinh ↓. Không phải Recall POI.','Có bản thích nghi; chưa tái lập mô hình học và ngữ nghĩa đầy đủ. [SC]'),
('3','Fake-query insertion · 2026','Không học sâu; chèn bản tin chỉ có dummy giữa truy vấn thật để làm khó nối đường.','Số đường nối khó phân biệt ↑; ASR ↑; độ trễ, bộ nhớ ↓.','Được chọn bổ sung; chưa có lượt benchmark. Phải tính mọi truy vấn chèn thêm. [FQ]'),
('4','DLS / enhanced-DLS · 2014','Không học sâu; dummy có xác suất truy vấn gần nhau, thêm phân tán không gian.','Entropy H = −Σp log₂p (bit) ↑; vùng che phủ CR lớn hơn trong bản enhanced.','Đối chứng nền đã có bản thích nghi trên đường. Hai cấu hình, không tính thành hai paper. [DLS]'),
('5','RDG · 2021','Không học sâu; sinh dummy chống nối chuỗi bằng Viterbi.','Entropy, entropy chuyển tiếp ↑; hiệu quả chống Viterbi.','Được chọn làm nền theo chuỗi; chưa có lượt đối chiếu trên tập mới. [RDG]'),
('6','AnotherMe · 2024 (online 2023)','Ánh xạ POI, định tuyến và mô phỏng tốc độ để dựng người dùng/quỹ đạo ảo.','Tỷ lệ nhận ra quỹ đạo giả; thời gian đáp ứng và pin. Mốc ngẫu nhiên phụ thuộc cách dựng phép phân loại.','Có VTGA thích nghi, không toàn hệ thống. Chỉ số này xác nhận từ abstract/mã; toàn bộ protocol gốc còn thiếu. [AM]'),
]
table('comparators','Phương pháp được chọn và chỉ số trong nguồn gốc',
      [dict(order=a,method=b,concept=c,metrics=d,status=e) for a,b,c,d,e in comps],
      [('order','STT'),('method','Phương pháp / năm'),('concept','Nguyên lý và vai trò'),('metrics','Chỉ số gốc'),('status','Tình trạng sử dụng')],'literature')
md('comparator-refs','**Nguồn:** '+' · '.join(f'[{k}] {cite(k)}' for k in ['TP','SC','FQ','DLS','RDG','AM']))
md('comparator-boundary','ASA, mix zones, CkiDel, chuỗi truy vấn giả của Wu, EPZ và mô hình đồng hành là **nghiên cứu liên quan cho từng mục tiêu**, chưa đưa tất cả vào bảng xếp hạng chính. Nhóm được chọn gồm cả sinh tập thật-giả và sinh dữ liệu thay thế; không gọi mọi phương pháp là cùng một giao diện dummy-generation.')

md('metrics-page', '''## 4. Giữ chỉ số gốc; thống nhất nhiệm vụ cần chấm

**Không quy entropy, ASR và sai số vị trí về một “điểm riêng tư”.** Giao thức chung hỏi: đối thủ đoán được gì, ứng dụng còn trả đúng bao nhiêu và tốn thêm bao nhiêu. Đây là lựa chọn có lý do cho bài toán này, không phải chuẩn duy nhất của lĩnh vực.''', new_page=True)
table('metrics','Bộ đo dùng chung và lý do chọn',[
dict(axis='1. Riêng tư vị trí', measure='MAE: sai số vị trí trung bình (m) ↑; Hit100: % lần đoán cách vị trí thật ≤100 m ↓.', why='MAE đo độ lớn sai số; Hit đo số lần vẫn đoán rất gần. Một chỉ số có thể tốt lên trong khi chỉ số kia xấu đi.', scope='S1-S3; S9-S10 chấm điểm biên. S8 đo thêm tác động của dữ liệu đồng hành.'),
dict(axis='2. Mục tiêu khác', measure='Liên kết người/thiết bị: F1 riêng; cạnh tiếp/đích: độ đúng; ý định: macro-F1. Thành công đối thủ càng thấp càng tốt.', why='Định vị sai không chứng minh giấu được danh tính, tương lai hoặc nội dung truy vấn.', scope='Đặc tả cho S4-S8; chưa có kết quả bảo vệ vòng mới.'),
dict(axis='3. Chất lượng POI', measure='Recall@5 = số POI đúng còn lấy được / số POI đúng cần tìm ↑. Kèm tỷ lệ trả đủ và quãng đường đi thêm.', why='Đo trực tiếp khả năng trả lời yêu cầu tìm địa điểm, thay vì chỉ khoảng cách dịch chuyển tọa độ.', scope='Giữ danh mục, loại dịch vụ, khoảng cách đường và quy tắc xếp hạng chung.'),
dict(axis='4. Chi phí / hợp lệ', measure='Số truy vấn và tọa độ; byte phản hồi; thời gian sinh p50/p95; lỗi chuyển đường.', why='Tăng dummy hay phản hồi có thể cải thiện Recall nhưng không miễn phí; nằm đúng đường không tự chứng minh riêng tư.', scope='Vòng mới chỉ có byte danh sách mã POI, chưa phải HTTP hoặc độ trễ điện thoại.')
], [('axis','Trục đánh giá'),('measure','Định nghĩa dễ hiểu'),('why','Lý do cần đo'),('scope','Phạm vi / giới hạn')],'literature')
md('metric-contract', '''**Ba giao diện, một đáp án cần suy luận:** một vị trí thay thế → ước lượng vị trí thật; tập chứa thật → chọn ứng viên hoặc ước lượng vị trí; tập không bắt buộc chứa thật → ước lượng từ toàn bộ bản tin. Mỗi đối thủ dùng đúng quan sát được cấp rồi trả cùng loại đáp án để chấm.

**Ba số khác nhau:** K = số điểm công bố; k = 5 POI ứng dụng cần; L = 5 hoặc 10 POI máy chủ trả cho từng truy vấn. Đổi L không đổi k. Ngưỡng Recall 90% là yêu cầu ta đặt, không phải ngưỡng chuẩn của paper.

**Điểm cần giữ khi tái lập:** ASR của semantic correlation đòi một mô hình xác suất nhận diện, không chỉ đếm đủ K điểm. DER trong bài có cả công thức tương đồng và diễn giải tỷ lệ đạt ngưỡng; phải công bố cách thực thi. EIE gốc và MAE của một ước lượng điểm không mặc nhiên là cùng định nghĩa.''')
md('metrics-refs', '**Cơ sở đánh giá theo đối thủ:** '+cite('Q')+'; **đối chiếu định nghĩa gốc:** '+cite('TP')+' và '+cite('SC')+'.')

md('method-page', '''## 5. Phương pháp đề xuất: giữ riêng tư, duy trì đường giả, phục hồi dịch vụ

**BR-Dummy là cơ chế nền đang phát triển.** Hướng mới không chỉ làm điểm giả trông hợp lý: nó chọn tập truy vấn để kết quả trả về vẫn chứa các POI người dùng cần. Phiên bản chưa bị ràng buộc phải dùng hay không dùng học sâu.''', new_page=True)
diagram('method-flow', [
    ('1. NEO RIÊNG TƯ', 'Từ GPS thật tạo neo nhiễu; thử tái sử dụng có nhiễu; ghi ngân sách phiên.'),
    ('2. ƯỚC LƯỢNG', 'Chỉ dùng lịch sử neo đã bảo vệ để ước lượng vùng vị trí và nhu cầu POI khả dĩ.'),
    ('3. CHỌN DUMMY', 'Giữ khả năng đi tới trên đường; chọn tham lam rồi thay điểm để tăng độ phủ.'),
    ('4. DỊCH VỤ', 'LSP trả POI cho điểm giả; thiết bị gộp và xếp lại tại vị trí thật.')
], 'Đường đi dữ liệu của đề xuất: phần sau neo không đọc lại GPS thật, tốc độ thật hoặc nhãn kịch bản. Ngữ cảnh công khai được cố định trước phiên.')
md('method-note', '''**Biến thể mới:** bộ lọc hai chế độ “đang dừng / đang đi” thay cho mô hình chuyển động một chế độ, nhưng giữ ngân sách và bước chọn điểm. Mục tiêu là ước lượng độ phủ tốt hơn; kết quả chưa cho thấy ưu thế đồng đều.

**Phần kế thừa:** Geo-I, hợp thành ngân sách, phép lọc Markov và tìm kiếm tham lam. Giả thuyết đóng góp là cách kết hợp chọn tập truy vấn có ràng buộc đường với chất lượng dịch vụ và đánh giá đối thủ; chưa tự nhận các thành phần chuẩn là thuật toán mới.

**Phạm vi:** lõi kiểm tra S1-S3. Mở rộng cần cơ chế liên phiên cho S4; đa nhánh tương lai S5-S6; xử lý nội dung S7; phối hợp đồng hành S8; chính sách biên S9-S10. Đây là yêu cầu phát triển, không phải các module đã hoàn tất.

**Giới hạn lý thuyết:** B = 0,24 m⁻¹ và H = 12 lần đọc. Chặn lý tưởng ở khoảng cách quỹ đạo 100 m là e²⁴, còn lỏng; không chuyển thành cam kết xác suất đoán đúng nhỏ.''', 'method')

md('protocol-page', '## 6. Quá trình thực nghiệm: khóa lựa chọn trước khi xác nhận\n\n**Vòng mới so sánh bốn biến thể nội bộ, chưa phải bảng so với SOTA.** Dùng cùng B = 0,24 m⁻¹, K = 5, H = 12, dãy neo ngẫu nhiên và dịch vụ POI; chỉ thay các thành phần cần kiểm tra.', 'method', True)
diagram('protocol-flow', [
    ('HỌC', '2 nhóm lõi + 64 nhóm phụ trợ. Học đối thủ; giữ nguyên dữ liệu học.'),
    ('CHỌN', '6 nhóm mới 301-306; 54 bản ghi S1-S3. Chọn đối thủ và cấu hình.'),
    ('KHÓA', 'Recall từng ca ≥90%; sau đó giảm Hit. Không đạt thì ghi không khả thi.'),
    ('XÁC NHẬN', '6 nhóm mới 307-312; 53 bản ghi. Không chọn lại sau khi xem điểm.')
], 'Huấn luyện cơ chế bảo vệ, nếu có, cũng phải nằm ở phần học. Phép xác nhận này không huấn luyện lại một mạng nơ-ron sinh dummy.')
md('protocol-text', '''**Bốn cấu hình:** (1) dummy hình học; (2) phủ POI tham lam; (3) phủ POI + thay điểm; (4) hai chế độ + thay điểm. Ba lần lặp mỗi bản ghi tạo 648 lượt chọn và 636 lượt xác nhận, không phải 636 người độc lập.

**Đối thủ:** suy luận từ tâm/trung bình, đường đi, cả cửa sổ và các mô hình học từ dữ liệu phụ trợ; 26 quyết định ở S1, 33 ở S2/S3. Chọn đối thủ tối thiểu MAE và tối đa Hit riêng trên tập chọn. Báo thêm Hit lớn nhất trong bộ đối thủ đã thử để thấy rủi ro chọn đối thủ yếu.

**Lấy trung bình:** sự kiện trong một lượt → các lượt trong từng ca → trung bình đều qua 9 ca con S1-S3. S3.B chỉ có 5 nhóm xác nhận, các ca còn lại có 6. Mỗi sự kiện thử sáu loại POI; không dùng nhãn kịch bản để điều khiển bộ bảo vệ.

**Kiểm tra:** dữ liệu gốc SUMO, quyền quan sát, học/chọn/kiểm thử, đường đi dummy, chạy lại tiền tố, tính lại dự đoán và Recall. Báo cáo ngắn dùng các kết quả đã lưu; không tinh chỉnh thêm thuật toán trong lần biên tập này.''', 'method')

names={'geometric':'Hình học','mean_greedy':'Phủ tham lam','mean_exchange':'Phủ + thay điểm','switching_exchange':'Hai chế độ + thay điểm'}
recall_rows=[]
privacy_rows=[]
for order,(method,label) in enumerate(names.items(),1):
    a=results['methods'][method]
    for depth in ('5','10'):
        u=a['utility'][depth]
        recall_rows.append(dict(method=label,depth='L = '+depth,recall=100*u['recall'],
            min_case=100*u['min_case_recall'],mae_m=a['mae_m'],hit100=100*a['hit100'],
            id_bytes=u['id_bytes'],cases=9,records=53,split='confirmation',k=5,K=5))
    privacy_rows.append(dict(order=order,method=label,hit=f'{100*a["hit100"]:.2f}%',
        envelope=f'{100*a["envelope_hit100"]:.2f}%',mae=f'{a["mae_m"]:.1f} m'))
datasets['recall']=recall_rows
md('results-page', '## 7. Kết quả mới: chất lượng tăng, riêng tư có đánh đổi\n\n**Đọc biểu đồ:** tỷ lệ top-5 đúng còn lấy được trên tập xác nhận; cao hơn tốt hơn. Cặp cột thay độ sâu phản hồi L, không thay số đáp án k = 5. Tăng L phải tính thêm chi phí.', 'result', True)
manifest['charts'].append(dict(id='recall-chart',title='Recall@5 của bốn biến thể',
    subtitle='Xác nhận cùng đô thị; 9 ca S1-S3, 53 bản ghi; đơn vị %.',showDescription=True,
    type='bar',dataset='recall',sourceId='result',valueFormat='number',
    encodings=dict(x=dict(field='method',type='nominal',label='Biến thể'),y=dict(field='recall',type='quantitative',label='Recall@5 (%)'),color=dict(field='depth',type='nominal',label='LSP trả'),
        tooltip=[dict(field='id_bytes',type='quantitative',label='Byte mã POI / sự kiện')]),
    palette=dict(kind='categorical'),legend=dict(position='top'),labels=dict(values='always')))
manifest['blocks'].append(dict(id='recall-chart-block',type='chart',chartId='recall-chart'))
table('privacy','Riêng tư trên cùng tập xác nhận',privacy_rows,
    [('order','STT'),('method','Biến thể'),('hit','Hit100 đã chọn ↓'),('envelope','Hit100 cực trị bộ thử ↓'),('mae','MAE đã chọn ↑')],'result')
base=results['methods']['mean_exchange']
new=results['methods']['switching_exchange']
cost=(base['utility']['10']['id_bytes']/base['utility']['5']['id_bytes']-1)*100
interpretation = f'''**Hai chế độ so với phủ + thay điểm:** Recall gần như giữ nguyên; Hit100 giảm {100*base['hit100']:.2f}% → {100*new['hit100']:.2f}%, nhưng MAE giảm {base['mae_m']:.1f} → {new['mae_m']:.1f} m, bất lợi cho riêng tư. Chưa kết luận tốt hơn trên mọi tiêu chí.

**Giá của phản hồi sâu hơn:** L = 5 → 10 tăng byte danh sách mã POI khoảng {cost:.1f}%; không phải toàn bộ HTTP. Cực trị là lớn nhất trong bộ đối thủ hữu hạn, không phải bảo đảm trước mọi đối thủ.'''

md('decision-page','## 8. Điều được kết luận và điều chưa được kết luận\n\n**Chưa có cấu hình vượt quy tắc chọn.** Trên validation ở L = 10, Recall ca yếu nhất của “phủ + thay điểm” là 89,25%, của “hai chế độ” là 89,32%; đều dưới 90%. Kết quả xác nhận đẹp hơn không cho phép chọn lại.', 'selection', True)
md('result-interpretation',interpretation,'result')
family_rows=[]
for family,values in results['paired_family_deltas']['switching_exchange__minus__mean_exchange'].items():
    family_rows.append(dict(family=family.replace('family-','Nhóm '),delta=values['hit100_pp'],
        recall_delta=values['recall10_pp'],envelope_delta=values['envelope_hit100_pp'],
        comparison='Hai chế độ trừ phủ + thay điểm',unit='điểm phần trăm'))
datasets['family']=family_rows
manifest['charts'].append(dict(id='family-chart',title='Thay đổi Hit100 đã chọn theo nhóm tuyến',
    subtitle='Hai chế độ trừ phủ + thay điểm; âm là giảm rủi ro, dương là tăng rủi ro.',showDescription=True,
    type='bar',dataset='family',sourceId='result',valueFormat='number',
    encodings=dict(x=dict(field='family',type='nominal',label='Nhóm xác nhận'),y=dict(field='delta',type='quantitative',label='Chênh lệch (điểm %)')),
    palette=dict(kind='sequential'),legend=dict(position='none'),labels=dict(values='always')))
manifest['blocks'].append(dict(id='family-chart-block',type='chart',chartId='family-chart'))
md('finish', '''**Không đồng đều:** Hit giảm ở 2 nhóm, tăng ở 2 nhóm, bằng nhau ở 2 nhóm. Kiểm tra loại hai cửa sổ dừng trùng vẫn cho đánh đổi Hit/MAE; chưa có kiểm định khẳng định thắng. Chỉ có sáu nhóm cùng đô thị và cửa sổ tối đa 12 sự kiện.

**Nội dung có thể báo cáo:** kịch bản và dữ liệu đã cụ thể hơn; có lựa chọn đối chứng và lý do đo; đã thực nghiệm một giả thuyết thuật toán và ghi nhận cả kết quả không thuận lợi.

**Ba việc cần chốt tiếp với thầy:** (1) thống nhất tác vụ/quan sát của S1-S10 và tăng mẫu các ca ít; (2) hoàn thiện tái lập đối chứng đã chọn, đặc biệt fake queries, RDG và các thành phần học; (3) kiểm tra truy vấn thưa S3.C, rồi mở phép thử riêng cho mục tiêu mở rộng. Không hạ ngưỡng hoặc dùng lại tập xác nhận để tuyên bố cải thiện mới.''','result')

artifact=dict(surface='report',manifest=manifest,
    snapshot=dict(version=1,generatedAt=STAMP,status='ready',datasets=datasets),
    sources=manifest['sources'])
HERE.mkdir(parents=True,exist_ok=True)
OUT.mkdir(parents=True,exist_ok=True)
(HERE/'artifact.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2)+'\n')
(HERE/'presentation_pages.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2)+'\n')
(HERE/'evidence.json').write_text(json.dumps(dict(
    source_sha256={p:digest(p) for p in [D,R,S,'artifacts/benchmarks/fresh_switching/readout_verification.json']},
    totals=dict(families=12,trips=264,fcd=172443,records=393),
    scenario_counts=data_rows,selected_external_methods=['TransProtect','Semantic correlation','Fake-query insertion','DLS/enhanced-DLS','RDG','AnotherMe (trajectory reference)'],
    quantitative_scope='Four internal variants, S1-S3, same-city confirmation. No external SOTA win claim.',
    sources={k:dict(label=v[0],url=v[1]) for k,v in REF.items()},
    word_budget_policy='Short paraphrases; no copied publisher figures or long quotations.'
),ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'artifact':str(HERE/'artifact.json'),'records':393,'blocks':len(manifest['blocks']),'charts':len(manifest['charts']),'pages_planned':1+len(pages)},ensure_ascii=False))

# Canonical Data app snapshot. The source is JSON/Python, never invented SQL.
queries = {}
source_by_id = {s['id']: s for s in manifest['sources']}
for widget in manifest['tables'] + manifest['charts']:
    source = source_by_id[widget['sourceId']]
    query_id = widget['dataset']
    queries[query_id] = dict(rows=datasets[query_id], source=dict(
        label=source['label'], files=[source['path']],
        evidenceFlow=[dict(title='Read reviewed local evidence', detail=source['path']),
                      dict(title='Reproducible transformation', detail='Run docs/supervisor_meeting/2026-09-11_brief/build_report.py; no experiment rerun or model tuning.')],
        metricDefinitions=[dict(label=widget['title'], componentIds=[widget['id']],
            definition=('Recall@5 in percent: case-macro average over 9 S1-S3 cases and 53 confirmation records; k=5, K=5; L is response depth.' if query_id=='recall' else
                        'Paired family mean Hit100 percentage-point difference: switching_exchange minus mean_exchange; negative favors the switching method.' if query_id=='family' else
                        'Reviewed table; units, scope, evidence status and paper-specific definitions are retained in each row.'))]),
        methods=[dict(language='python',code='Run docs/supervisor_meeting/2026-09-11_brief/build_report.py from repository root.')])
(HERE/'data.json').write_text(json.dumps(dict(
    id='report:2a1213ff-1d88-4e76-9c15-89e97de08c48',surface='report',title=TITLE,
    generatedAt=STAMP,status='ready',buildStatus='complete' if args.complete else 'creating',
    report=dict(asOf='2026-09-11'),filters=[],queries=queries
),ensure_ascii=False,indent=2)+'\n')
