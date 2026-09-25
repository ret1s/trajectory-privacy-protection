"""Report the fixed larger-cohort evaluation; never select model parameters."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean

from experiments.research_loop_resources import ROOT, sha

OUT = ROOT / 'artifacts/benchmarks/endpoint_calendar_expanded_v1'
DATA = ROOT / 'artifacts/datasets/endpoint_holdout_expanded_v1'
LABELS = {'raw': 'Vị trí thật', 'dls': 'DLS', 'rdg': 'RDG',
          'transprotect_markov': 'TransProtect (Markov)', 'semantic_poi': 'Semantic (thực nghiệm)',
          'fake_queries': 'Fake-query adapter', 'anotherme_offline': 'AnotherMe offline',
          'ours30': '30 / mỗi sự kiện', 'ours67': '67 / mỗi sự kiện',
          'calendar30': '30 / lịch công khai', 'calendar67': '67 / lịch công khai'}


def main():
    def load(path): return json.loads(path.read_text())
    data = load(DATA / 'dataset.json')
    read = load(OUT / 'readout.json')
    check = load(OUT / 'verification.json')
    ablation = load(OUT / 'calendar_ablation.json')
    assert check['status'] == 'passed' and check['readout_sha256'] == sha(OUT / 'readout.json')
    assert ablation['service_sha256'] == sha(OUT / 'holdout_service.json')
    protocol = load(OUT / 'protocol.json')
    assert sha(OUT / 'selection.json') == protocol['frozen_selection_sha256']
    assert sha(ROOT / 'artifacts/benchmarks/endpoint_calendar_v1/selection.json') == protocol['frozen_selection_sha256']
    # The pilot was already inspected before the extension was declared.
    pilot = load(ROOT / 'artifacts/datasets/endpoint_holdout_v1/dataset.json')
    old_routes = {tuple(s['route_edges']) for f in pilot['families'] for s in f['sessions']}
    overlap = sum(tuple(s['route_edges']) in old_routes for f in data['families'] for s in f['sessions'])
    private_records = [r for r in data['records'] if r['scenario'] in ('S1', 'S2', 'S3', 'S9', 'S10')]
    counts = Counter(r['case_id'] for r in data['records'])
    completed = sum(float(v.get('arrival', -1)) >= 0 for f in data['families'] for v in f['actual_vehicles'].values())
    cost_ratios = {}
    for method in ('calendar30', 'calendar67'):
        ratios = defaultdict(lambda: defaultdict(list))
        for r in ablation['rows']:
            if r['method'] != method: continue
            total = lambda name: sum(r[name][k] for k in ('request_bytes', 'response_bytes'))
            ratios[r['family_id']][r['session_id']].append(total('public_calendar') / total('active_epoch'))
        cost_ratios[method] = mean(mean(mean(v) for v in sessions.values()) for sessions in ratios.values())
    presentation = {
        'families_planned': len(protocol['holdout_seeds']), 'families_completed': len(data['families']),
        'completed_trips': completed, 'all_records': len(data['records']),
        'priority_records': len(private_records),
        'priority_source_sessions': len({s for r in private_records for s in r['session_ids']}),
        'endpoint_records': sum(counts[c] for c in counts if c.startswith(('S9.', 'S10.'))),
        'case_counts': dict(counts), 'construction_failures': data['construction_failures'],
        'record_gate_rejections': dict(Counter(r['case_id'] for r in data['rejections'])),
        'exact_route_overlaps_with_pilot': overlap,
        'calendar_to_active_epoch_byte_ratio': cost_ratios,
        'same_utility_event_checks': ablation['service_events_equal_utility_checked'],
        'selection_reused_unchanged': True,
        'sources': {str(p.relative_to(ROOT)): sha(p) for p in
                    (DATA / 'dataset.json', OUT / 'readout.json', OUT / 'verification.json',
                     OUT / 'calendar_ablation.json', OUT / 'protocol.json', Path(__file__))}}
    (OUT / 'presentation.json').write_text(json.dumps(presentation, ensure_ascii=False, indent=2) + '\n')

    def pct(v): return '—' if v is None else f'{100*v:.2f}%'
    def num(v): return '—' if v is None else f'{v:.1f}'
    a = {r['method']: r for r in read['aggregates']}
    s = {(r['method'], r['case_id']): r for r in read['summaries']}
    lines = [
        '# S9/S10: lịch truy vấn công khai và kiểm tra trên nhiều nhóm tuyến', '',
        'Cập nhật 25/09/2026. Kết quả dưới dùng **cohort mới, tách khỏi pilot**. '
        'Kế hoạch 30/67 query và cách chọn attacker được giữ nguyên trước khi xem cohort này.', '',
        '## Quy mô và cách đọc mẫu', '',
        f"Đã thử đủ {presentation['families_planned']} seed 1201–1232: "
        f"{presentation['families_completed']} nhóm dựng được, {completed} chuyến hoàn tất, "
        f"{len(data['records'])} bản ghi toàn bộ S1–S10. Năm scenario ưu tiên dùng "
        f"{len(private_records)} record từ {presentation['priority_source_sessions']} chuyến; "
        f"riêng S9/S10 có {presentation['endpoint_records']} record. "
        'Vòng này chấm utility trên 15 ca của năm scenario, privacy trên sáu ca S9/S10; '
        'không nhận là đã chạy lại privacy S1/S2/S3 trên cohort mới.', '',
        'Nhóm tuyến là đơn vị gộp/ước lượng bất định. 22 chuyến cùng nhóm, '
        'record A/B/C, hai lần ngẫu nhiên cơ chế và ba trạng thái thế giới không được tính thành '
        'những người dùng độc lập. Đây vẫn là dữ liệu SUMO cùng thành phố và bộ sinh.', '',
        'Một số nhóm không đủ điều kiện dựng mọi record: S9.B có 23 nhóm, '
        'S10.A có 31, S10.B có 25; các ca endpoint còn lại có 32. '
        'Các điều kiện tạo ca kiểm tra dữ liệu chuyển động, trước khi chạy model; '
        'không loại record theo kết quả privacy/utility.', '',
        f"Lỗi dựng nhóm: {json.dumps(data['construction_failures'], ensure_ascii=False)}. "
        'Giữ lỗi, không đổi seed để bù. Chọn đủ 32 seed trước khi xem kết quả; không dừng khi số đẹp.', '',
        'Attacker fit trên 10 nhóm 501–502/701–708, chọn trên 10 nhóm '
        '601–602/709–712/901–904. Pilot có bảy nhóm (1104 lỗi dựng); mở rộng này dùng nguyên '
        'selection đã khóa của pilot, không fit/chọn lại. Không gộp pilot vào kết quả chính.', '',
        f"Trùng nguyên chuỗi cạnh với dữ liệu phát triển: {check['exact_route_overlaps_with_historical_sessions']}; "
        f"với pilot: {overlap}. Kiểm tra này không chứng minh các đường hoàn toàn không giao nhau.", '',
        '## Vì sao nhiều đối chứng đạt Recall 100%?', '',
        'DLS/RDG/Semantic/Fake giữ vị trí thật trong tập truy vấn. Trong giao diện đang thử, '
        'server trả top-10 POI khả dụng theo cùng khoảng cách đường; client hợp phản hồi rồi chọn '
        'top-5 thật. Phản hồi tại vị trí thật đã chứa top-5 cần dùng. Vì vậy Recall đạt trần '
        'là hệ quả của giao diện, không chỉ vì ít sample. Tăng số chuyến không làm mất tính chất này.', '',
        'Recall đo chất lượng dịch vụ, **không đo privacy**. Phân biệt các phương pháp bằng '
        'khả năng suy luận của attacker và chi phí để giữ Recall đó. Không làm yếu API của '
        'đối chứng để tạo chênh lệch. Cấu hình 67 vẫn còn tám POI chưa phủ tĩnh, nên Recall '
        '100% trên mẫu chưa bảo đảm 100% toàn bản đồ.', '',
        '## Thành phần bảo vệ S9/S10', '',
        'Client gửi kế hoạch theo loại POI ở mọi tick 60 giây trong khoảng công khai '
        '[0, 3600), kể cả trước/sau chuyến và khi không có truy vấn riêng tư. Thiết bị chỉ '
        'đọc cache hợp lệ và xếp hạng bằng GPS tại chỗ. Không dùng GPS để kích hoạt/dừng lịch.', '',
        'Đây là mở rộng từ độc lập tọa độ sang độc lập thời điểm hoạt động **trong khoảng '
        'đã đăng ký trước**. Với vùng, kế hoạch, khoảng đăng ký và trạng thái server cố định, '
        'đổi hành trình hoặc giờ đọc cache không đổi bản tin yêu cầu/phản hồi. Sáu kiểm tra '
        'luồng dữ liệu đã so lịch đọc sớm, muộn và không đọc. Bảo đảm không bao gồm IP/account, '
        'click, lỗi mạng, đổi vùng theo GPS hoặc đăng ký/hủy dịch vụ theo giờ đi lại.', '',
        'Giữ bank Viterbi/thống kê/ngoại suy/kNN/Extra Trees; bổ sung suy luận endpoint theo '
        'khoảng cách đường có hướng, vận tốc từ phần được quan sát và nhiều horizon. '
        'S9 suy ngược, S10 suy xuôi; C kết hợp các chuyến liên kết được phép. '
        'Đối thủ không nhận endpoint thật, đoạn bị giấu, thời gian cắt thật hoặc tuyến tương lai. '
        'Chọn decoder riêng cho MAE và từng ngưỡng Hit trên tập chọn, không bằng đáp án holdout. '
        'Vì có thể khác decoder, các cột Hit50/100/200 không phải một CDF chung. '
        'Log-gain, entropy và vùng credible trong artifact chỉ là chẩn đoán phụ; '
        'phải đọc cùng coverage thực tế của posterior, không coi entropy cao là bảo đảm privacy.', '',
        'Phần đóng góp hiện có là thiết kế và kiểm chứng phối hợp kế hoạch phủ POI, lịch công khai '
        'và xếp hạng cục bộ trong dịch vụ cụ thể. Số này chưa xác lập rằng cover traffic '
        'hoặc truy vấn công khai là ý tưởng mới so với toàn bộ nghiên cứu trước.', '',
        '## Kết quả cùng cohort và cùng dịch vụ', '',
        'Mỗi scenario gộp đều A/B/C sau khi gộp theo record và nhóm. '
        'Hit100 thấp và MAE cao tốt hơn cho privacy. Recall gộp 15 ca ở p=0,8. '
        'Byte là yêu cầu + phản hồi JSON trên mỗi sự kiện dịch vụ, chưa gồm HTTP/TLS. '
        'Tính toàn bộ giờ cover traffic cho lịch công khai.', '',
        '| Phương pháp | S9 Hit100 / MAE m | S10 Hit100 / MAE m | Recall@5 | Byte/sự kiện |',
        '|---|---:|---:|---:|---:|']
    for method in LABELS:
        r = a[method]
        lines.append(f"| {LABELS[method]} | {pct(r['S9']['hit100'])} / {num(r['S9']['mae_m'])} | "
                     f"{pct(r['S10']['hit100'])} / {num(r['S10']['mae_m'])} | {pct(r['recall_0.8'])} | "
                     f"{num(r['request_bytes_per_service_event']+r['response_bytes_per_service_event'])} |")
    lines += ['', '### Độ nhạy theo tỷ lệ POI khả dụng', '',
              '| Cấu hình lịch | p=0,5 | p=0,8 | p=0,95 |', '|---|---:|---:|---:|']
    for method in ('calendar30', 'calendar67'):
        cells = [f"{pct(a[method][f'recall_{q}'])}; {a[method][f'gates_{q}']}/15 ca ≥90%" for q in (.5, .8, .95)]
        lines.append('| ' + LABELS[method] + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'AnotherMe là tham chiếu VTGA offline và có lỗi sinh đầu ra; '
              'không so privacy của tập con thành công như cùng mẫu số. TransProtect dùng Markov '
              'thay Transformer; Semantic dùng dự báo thực nghiệm thay LSTM. Đây là '
              '**adapter công khai, không phải tuyên bố vượt sáu paper nguyên bản**.', '',
              '### Đọc từng A/B/C', '',
              '| Ca | Nhóm | Raw Hit100 / Hit200 | 67 lịch Hit100 / Hit200 | Raw MAE m | 67 lịch MAE m |',
              '|---|---:|---:|---:|---:|---:|']
    for case in [f'S{n}.{c}' for n in (9, 10) for c in 'ABC']:
        raw, ours = s['raw', case], s['calendar67', case]
        lines.append(f"| {case} | {raw['valid_families']} | {pct(raw['hit100'])} / {pct(raw['hit200'])} | "
                     f"{pct(ours['hit100'])} / {pct(ours['hit200'])} | {num(raw['mae_m'])} | {num(ours['mae_m'])} |")
    lines += ['', 'S10.B chỉ cho thấy phần trước khi phân nhánh: ngay control vị trí thật có thể '
              'không xác định được đích. Phải đọc năng lực raw trước khi nhận Hit=0 là bằng chứng '
              'bảo vệ. Không gộp mọi endpoint thành một kết luận chắc chắn.', '',
              '### Chênh lệch với năm adapter trực tuyến', '',
              'Bootstrap ghép theo nhóm, 3.000 lần, khoảng 95% chưa hiệu chỉnh nhiều phép so sánh. '
              'Âm tốt cho ΔHit100; dương tốt cho ΔMAE. Đây là khoảng bất định theo nhóm trong '
              'một bộ sinh, không đại diện mọi thành phố.', '',
              '| Ca | 67 lịch so với | ΔHit100 điểm % [CI95] | ΔMAE m [CI95] |',
              '|---|---|---:|---:|']
    for scenario in ('S9', 'S10'):
        for control in ('dls', 'rdg', 'transprotect_markov', 'semantic_poi', 'fake_queries'):
            rr = {r['metric']: r for r in read['comparisons'] if r['target'] == 'calendar67' and r['control'] == control and r['scenario'] == scenario}
            def ci(metric, scale):
                r = rr[metric]
                return f"{scale*r['delta']:.2f} [{scale*r['ci95'][0]:.2f}, {scale*r['ci95'][1]:.2f}]"
            lines.append(f"| {scenario} | {LABELS[control]} | {ci('hit100',100)} | {ci('mae_m',1)} |")
    lines += ['', '### Lập luận từ kết quả', '']
    for scenario in ('S9', 'S10'):
        contrasts = [r for r in read['comparisons'] if r['target'] == 'calendar67' and r['scenario'] == scenario and r['metric'] == 'hit100']
        separated = sum(r['ci95'][1] < 0 for r in contrasts)
        controls = [a[m][scenario]['hit100'] for m in ('dls', 'rdg', 'transprotect_markov', 'semantic_poi', 'fake_queries')]
        lines.append(f"- **{scenario}:** Hit100 của năm adapter nằm trong {pct(min(controls))}–{pct(max(controls))}, "
                     f"bản 67 theo lịch là {pct(a['calendar67'][scenario]['hit100'])}. "
                     f"{separated}/5 khoảng 95% của chênh lệch Hit100 nằm hoàn toàn dưới 0.")
    lines += ['', 'Đây là lợi thế privacy trong bộ đối thủ và dịch vụ đã thử, với chi phí truyền cao hơn. '
              'Số 0% không chứng minh đối thủ bất kỳ đều thất bại; lập luận độc lập payload có điều kiện '
              'được kiểm tra riêng bằng luồng dữ liệu. S10.B control yếu vẫn là giới hạn, '
              'dù trung bình toàn scenario có chênh lệch.', '']
    lines += ['', '## Giá của việc che giờ bắt đầu/kết thúc', '',
              f"So với chính kế hoạch đó chỉ refresh ở epoch có hoạt động, lịch công khai dùng "
              f"trung bình theo nhóm {cost_ratios['calendar30']:.2f} lần byte (30 query) và "
              f"{cost_ratios['calendar67']:.2f} lần (67 query). Đã đối chiếu "
              f"{ablation['service_events_equal_utility_checked']} lượt dùng dịch vụ có utility bằng nhau. "
              'Chi phí tăng là cover traffic ngoài thời gian hoạt động, không phải utility tăng.', '',
              'Cả bản mỗi sự kiện và bản theo lịch đã dùng cùng tọa độ công khai. Vì vậy lợi ích '
              'mới của lịch là xóa phụ thuộc vào giờ hoạt động trong subscription, không nên '
              'gán toàn bộ chênh lệch MAE với DLS cho riêng bộ scheduler. Chưa có so sánh '
              'cùng byte/latency hoặc bằng chứng hơn bulk trên danh mục nhỏ 419 POI.', '',
              '## Kiểm tra và tái chạy', '',
              f"Đã tính lại {check['case_metrics_recomputed']} chỉ số, "
              f"{check['cost_aggregates_recomputed']} aggregate chi phí, kiểm tra "
              f"{check['transcripts_checked']} transcript. Generation failures: "
              f"`{check['generation_failures']}`. Lỗi phiên nhận Recall=0 ở các yêu cầu có đáp án; "
              'record nhiều phiên có thể vẫn có Recall từ phiên khác chạy thành công. '
              'Privacy thiếu đầu ra giữ NA, không gán bảo vệ hoàn hảo.', '',
              'Dữ liệu và số đầy đủ: [protocol](../../artifacts/benchmarks/endpoint_calendar_expanded_v1/protocol.json), '
              '[readout](../../artifacts/benchmarks/endpoint_calendar_expanded_v1/readout.json), '
              '[66 dòng ca](../../artifacts/benchmarks/endpoint_calendar_expanded_v1/case_results.csv), '
              '[verification](../../artifacts/benchmarks/endpoint_calendar_expanded_v1/verification.json), '
              '[ablation](../../artifacts/benchmarks/endpoint_calendar_expanded_v1/calendar_ablation.json).', '',
              'Runner: `python -m experiments.extend_endpoint_cohort STAGE` với '
              '`build → generate → attacks/service/flow → ablation → readout → verify`. '
              'Protocol/selection đã khóa; không dùng `prepare` để ghi đè. '
              'Bảng trình bày: `python -m experiments.summarize_endpoint_evidence`.', '']
    lines += ['Dataset được lưu lossless bằng `dataset.json.gz`; phục hồi bằng '
              '`python -m experiments.endpoint_dataset_archive unpack` trước khi đọc hoặc dựng lại report. '
              '`archive.json` giữ SHA-256 của cả JSON gốc lẫn gzip; không thay nội dung đầu vào đã khóa.', '']
    (ROOT / 'docs/research/endpoint_calendar_results.md').write_text('\n'.join(lines))
    print(json.dumps(presentation, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
