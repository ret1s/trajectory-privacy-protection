"""Current-scope research note; retain original experimental readouts."""
import json
from experiments.reaggregate_active_scope import ROOT, OUT
from experiments.research_loop_resources import sha


def main():
    d=json.loads((OUT/'readout.json').read_text());v=json.loads((OUT/'verification.json').read_text())
    assert v['status']=='passed' and v['readout_sha256']==sha(OUT/'readout.json')
    a={r['method']:r for r in d['endpoint']['aggregates']}
    h={(r['method'],r['case_id']):r for r in d['historical']['summaries'] if r['split']=='new_groups'}
    labels={'raw':'Vị trí thật','dls':'DLS','rdg':'RDG','transprotect_markov':'TransProtect (Markov)',
            'semantic_poi':'Semantic (thực nghiệm)','fake_queries':'Fake-query adapter',
            'anotherme_offline':'AnotherMe offline†','calendar30':'Đề xuất 30 + lịch','calendar67':'Đề xuất 67 + lịch'}
    controls=['dls','rdg','transprotect_markov','semantic_poi','fake_queries']
    def pct(x):return 'NA' if x is None else f'{100*x:.2f}%'
    def num(x):return 'NA' if x is None else f'{x:.1f}'
    lines=['# Phạm vi hiện tại: S1/S2/S3/S9 và S10.A/C','',
        '**S10 chỉ gồm A và C.** Ca B đã loại khỏi danh mục mẫu và kết quả chính; hướng đoán đích từ tiền tố thuộc S6.A. Giữ mã A/C để truy nguyên. Kho dữ liệu và benchmark cũ được bảo toàn, không nhận việc thu hẹp phạm vi là cải tiến thuật toán hay một phép xác nhận mới.','',
        'Quyết định phạm vi được thực hiện sau [chẩn đoán B](s10b_diagnostic.md). Không chọn lại model, attacker hoặc seed. Các số dưới được tính lại từ kết quả đã khóa.','',
        '## Phạm vi và cách tổng hợp','',
        '- 29 ca toàn bộ S1–S10; 14 ca trong năm scenario ưu tiên.',
        '- Bộ minh họa: 389/393 record gốc, 12 nhóm, 264 chuyến; 29 mẫu trên map.',
        '- Đối chứng phát triển: 165 record từ 94 chuyến; kiểm tra bốn nhóm: 54 record từ 30 chuyến.',
        '- Cohort 32 nhóm: giữ 1.093/1.118 record; utility dùng 435 record từ 246 chuyến, privacy S9/S10 dùng 150 record.',
        '- Gộp seed/world trong record rồi nhóm; trung bình đều ca trong mỗi scenario và đều năm scenario. S10 có A/C nên chia hai, các scenario khác chia ba. Không lấy trung bình phẳng 14 ca để làm giảm trọng số S10.',
        '- Chi phí chỉ trên chuyến còn thuộc phạm vi; giữ toàn bộ clock, mẫu số sự kiện và traffic gốc của chuyến. Client theo lịch bị tính đủ cả giờ, kể cả trước/sau chuyến. Không tái sinh transcript hoặc xóa traffic để làm chi phí giảm.',
        '- AnotherMe có lỗi và thiếu ca: số privacy chỉ mô tả tập con thành công; không đồng hạng với năm adapter trực tuyến.','',
        '## Bằng chứng cho cả năm scenario','',
        '| Scenario | Hit100 năm adapter trực tuyến | Đề xuất | Phạm vi bằng chứng |',
        '|---|---:|---:|---|']
    for sc in ('S1','S2','S3','S9','S10'):
        vals=[sum(h[m,f'{sc}.{c}']['hit100'] for c in 'ABC')/3 for m in controls] if sc in ('S1','S2','S3') else [a[m][sc]['hit100'] for m in controls]
        scope='4 nhóm, kế hoạch mỗi sự kiện' if sc in ('S1','S2','S3') else '32 nhóm, kế hoạch theo lịch'
        lines.append(f'| {sc+(".A/C" if sc=="S10" else "")} | {pct(min(vals))}–{pct(max(vals))} | 0% | {scope} |')
    lines += ['', '**Có cơ sở trình bày đóng góp thiết kế và thực nghiệm trên cả năm scenario trong phạm vi đã thử.** '
        'Giải pháp phối hợp kế hoạch truy vấn theo độ phủ POI, xếp hạng bằng GPS tại thiết bị và lịch công khai. '
        'GPS và giờ đọc cache không điều khiển payload/lịch gửi khi vùng, kế hoạch, trạng thái server và khoảng đăng ký cố định. '
        'Điều này hỗ trợ lập luận giảm lộ vị trí hiện tại, nơi dừng, đường đi và endpoint trong dịch vụ trạng thái động.', '',
        'Ba lớp bằng chứng cần đọc cùng nhau: (1) lập luận phụ thuộc thông tin có điều kiện; '
        '(2) kết quả chống bộ đối thủ đã thử; (3) utility và chi phí đo trên cùng API. '
        'Chọn truy vấn công khai/cover traffic tự thân chưa được chứng minh là ý tưởng mới. '
        'Số bốn nhóm S1–S3 không được gán thành kết quả của client theo lịch trên 32 nhóm.', '',
        '## S9 và S10.A/C: kết quả cùng cohort','',
        'Mỗi ô privacy: Hit100 ↓ / MAE m ↑. Recall tại availability 80%; byte là request + response JSON mỗi sự kiện dịch vụ, chưa gồm HTTP/TLS.','',
        '| Phương pháp | S9 A/B/C | S10 A/C | Recall@5 | Byte/sự kiện |',
        '|---|---:|---:|---:|---:|']
    for m,label in labels.items():
        x=a[m]
        lines.append(f"| {label} | {pct(x['S9']['hit100'])} / {num(x['S9']['mae_m'])} | {pct(x['S10']['hit100'])} / {num(x['S10']['mae_m'])} | {pct(x['recall_0.8'])} | {num(x['request_bytes_per_service_event']+x['response_bytes_per_service_event'])} |")
    lines += ['', '† AnotherMe offline dùng tập con privacy thành công; utility giữ lỗi. TransProtect dùng Markov thay Transformer, Semantic dùng predictor thực nghiệm thay LSTM. Đây là so sánh adapter, chưa xác lập vượt sáu paper nguyên bản.', '',
        '### Chênh lệch S10.A/C đã tính lại','',
        'Bootstrap ghép theo nhóm, 3.000 lần, khoảng 95% chưa hiệu chỉnh nhiều so sánh. Phân tích theo phạm vi sửa sau chẩn đoán, không phải cohort mới.','',
        '| 67 + lịch so với | ΔHit100 (điểm %) | Khoảng 95% |',
        '|---|---:|---:|']
    for c in d['endpoint']['comparisons']:
        if c['target']=='calendar67' and c['scenario']=='S10' and c['metric']=='hit100':
            lines.append(f"| {labels[c['control']]} | {100*c['delta']:.2f} | [{100*c['ci95'][0]:.2f}, {100*c['ci95'][1]:.2f}] |")
    lines += ['', 'Raw S10.A có Hit100 12,90%, S10.C 21,88%; sau bảo vệ theo lịch đều 0% trong bank đã thử. '
        'S10.A có 31 nhóm, C có 32; không cộng thành 63 nhóm độc lập. '
        'Các khoảng A/C ở đây được tính lại từ metric từng nhóm, không tái sử dụng khoảng A/B/C.', '',
        '### Utility và chi phí','',
        f"- Bản 30: Recall {pct(a['calendar30']['recall_0.8'])}, {a['calendar30']['gates_0.8']}/14 ca ≥90% ở p=0,8. Stress p=0,95: {pct(a['calendar30']['recall_0.95'])}, {a['calendar30']['gates_0.95']}/14 ca đạt ngưỡng.",
        f"- Bản 67: Recall {pct(a['calendar67']['recall_0.8'])}; {a['calendar67']['gates_0.8']}/14 ca đạt ngưỡng, vẫn {pct(a['calendar67']['recall_0.95'])} trong stress p=0,95.",
        f"- Trên chuyến thuộc phạm vi mới, lịch công khai tốn trung bình theo nhóm {d['ablation']['calendar_to_active_epoch_byte_ratio']['calendar67']:.2f} lần byte so với cùng kế hoạch chỉ refresh khi hoạt động; giữ {d['ablation']['equal_utility_event_checks']:,} phép so utility bằng nhau.",
        '- DLS/RDG/Semantic/Fake đạt trần Recall nhờ chứa truy vấn tại vị trí thật trong giao diện đang dùng. Đây không phải bằng chứng privacy. Bản 67 còn tám POI ngoài độ phủ tĩnh, nên Recall 100% trên mẫu chưa là chứng nhận toàn bản đồ.', '',
        '## Các kết luận còn cần kiểm chứng','',
        '- Chạy lại privacy S1/S2/S3 trên 32 nhóm bằng client theo lịch và lựa chọn attacker phù hợp, giữ riêng với kết quả bốn nhóm đã có.',
        '- So sánh ở cùng ngân sách byte/latency; chi phí cao hơn ngăn kết luận thống trị toàn diện.',
        '- Kiểm tra một vùng/bộ sinh khác và củng cố đối chứng gốc trước khi kết luận tổng quát hoặc độ mới phương pháp.',
        '- Bảo đảm hiện tại có điều kiện vùng/lịch công khai cố định; IP, account, click, lỗi mạng và kích hoạt dịch vụ theo chuyến nằm ngoài phạm vi.', '',
        '## Nguồn và tái lập','',
        '[Readout hiện tại](../../artifacts/benchmarks/active_scope_ac_v2/readout.json) · '
        '[Verification](../../artifacts/benchmarks/active_scope_ac_v2/verification.json). '
        f"Đã đối chiếu {v['numerical_checks']} giá trị/lựa chọn mẫu số với dữ liệu đo gốc; kiểm tra phân tầng và chi phí có tests riêng.", '',
        '`python -m experiments.reaggregate_active_scope` → `python -m experiments.verify_active_scope` → '
        '`python -m experiments.summarize_active_scope`. Các lệnh chỉ tạo readout mới; không ghi đè kết quả gốc hoặc train lại model.', '']
    (ROOT/'docs/research/active_scope_results.md').write_text('\n'.join(lines))
    print('Wrote active_scope_results.md',flush=True)


if __name__=='__main__':main()
