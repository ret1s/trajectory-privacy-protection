"""Reconcile presentation evidence and PDF contents without rerunning experiments."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import math
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
def read(path):
    return json.loads(path.read_text())
def digest(path):
    return sha256(path.read_bytes()).hexdigest()
evidence = read(HERE/'evidence.json')
for relative, expected in evidence['source_sha256'].items():
    assert digest(ROOT/relative) == expected, relative
dataset = read(ROOT/'artifacts/datasets/urban_fresh_v2/dataset.json')
results = read(ROOT/'artifacts/benchmarks/fresh_switching/readout.json')
selection = read(ROOT/'artifacts/benchmarks/fresh_switching/selection.json')
snapshot = read(HERE/'data.json')
assert snapshot['buildStatus'] == 'complete'
scenarios = [row for key in ('scenarios-table-0', 'scenarios-table-1')
             for row in snapshot['queries'][key]['rows']]
assert [row['id'] for row in scenarios] == [f'S{i}' for i in range(1,11)]
mapping = {row['id']: row for row in scenarios}
expected_methods = {
    'S1': ['DLS / enhanced-DLS'], 'S2': ['ASA:'],
    'S3': ['RDG:', 'TransProtect:', 'Semantic correlation:', 'Fake queries:'],
    'S4': ['Mix zones:'], 'S5': ['LPPM hiện tại - tương lai:'],
    'S6': ['CkiDel:'], 'S7': ['Chuỗi truy vấn giả của Wu:'],
    'S8': ['Chưa xác minh cơ chế dummy trực tiếp:'],
    'S9': ['Vùng che biên (EPZ):'], 'S10': ['Vùng che biên (EPZ):'],
}
for scenario, terms in expected_methods.items():
    assert all(term in mapping[scenario]['related'] for term in terms), scenario
assert len(mapping['S3']['related'].splitlines()) == 4
assert 'Liên quan' in mapping['S2']['status'] and 'Liên quan' in mapping['S5']['status']
assert 'ID tài khoản/thiết bị ổn định' in mapping['S4']['status']
assert 'AnotherMe' not in mapping['S4']['related']
assert 'Chưa đủ bằng chứng' in mapping['S8']['status']
assert 'không tái lập nguyên EPZ' in mapping['S10']['status']
comparators = snapshot['queries']['comparators']['rows']
assert [row['concept'].split(':')[0] for row in comparators[:5]] == ['S3','S3','S3','S1','S3']
assert len(dataset['records']) == 393
counts = Counter((r['scenario'],r['split'],r['case_id'][-1]) for r in dataset['records'])
for row in snapshot['queries']['data-counts']['rows']:
    for field, split in [('validation','fresh_validation'),('confirmation','fresh_confirmation')]:
        assert row[field] == ' / '.join(str(counts[row['scenario'],split,c]) for c in 'ABC')
names = {'Hình học':'geometric','Phủ tham lam':'mean_greedy',
         'Phủ + thay điểm':'mean_exchange','Hai chế độ + thay điểm':'switching_exchange'}
for row in snapshot['queries']['recall']['rows']:
    expected = results['methods'][names[row['method']]]['utility'][row['depth'].split()[-1]]['recall']*100
    assert math.isclose(row['recall'],expected,abs_tol=1e-12)
    assert (row['k'],row['K'],row['records'],row['cases']) == (5,5,53,9)
for row in snapshot['queries']['privacy']['rows']:
    expected = results['methods'][names[row['method']]]
    assert row['hit'] == f"{100*expected['hit100']:.2f}%"
    assert row['envelope'] == f"{100*expected['envelope_hit100']:.2f}%"
    assert row['mae'] == f"{expected['mae_m']:.1f} m"
for row in snapshot['queries']['family']['rows']:
    family = row['family'].replace('Nhóm ','family-')
    expected = results['paired_family_deltas']['switching_exchange__minus__mean_exchange'][family]['hit100_pp']
    assert math.isclose(row['delta'],expected,abs_tol=1e-12)
assert all(v['chosen'] is None for v in selection['method_selection_by_depth'].values())
pdf = ROOT/'artifacts/reports/supervisor_brief_2026-09-11.pdf'
html = pdf.with_suffix('.html')
receipt = read(pdf.with_suffix('.render.json'))
assert receipt['pdf_sha256'] == digest(pdf)
assert receipt['runtime_errors'] == []
assert len(receipt['sections']) == 10
assert [c['bars'] for c in receipt['charts']] == [8,4]
assert html.stat().st_size > 100000 and pdf.stat().st_size > 100000
text = subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True)
pages = [p for p in text.split('\f') if p.strip()]
assert len(pages) == 10, len(pages)
assert all(len(p)>600 for p in pages), [(i,len(p)) for i,p in enumerate(pages)]
assert '\ufffd' not in text
for term in ['TransProtect','AnotherMe','RDG','393','264','Recall@5','Hit100','MAE','S10','64.7%','89,32%']:
    assert term in text, term
for term in ['Phương pháp nào bảo vệ mục tiêu nào?', 'CkiDel', 'ASA', 'EPZ',
             'Chưa đủ bằng chứng', 'chưa phải kết quả vượt qua các ca SUMO']:
    assert term in ' '.join(text.split()), term
assert all(f'{i} / 10' in page for i,page in enumerate(pages,1))
verification = dict(verified=True,pages=10,charts=2,scenario_rows=10,
                    scenario_method_mappings_verified=10,
                    source_hashes_verified=len(evidence['source_sha256']),
                    pdf_sha256=digest(pdf),html_sha256=digest(html),
                    new_experiment_runs=0,
                    limitations=['Visual page acceptance is recorded separately by the reviewer.',
                                 'No interactive GUI/mobile check; no claim of SOTA reproduction or all-scenario protection.'])
(HERE/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(verification,ensure_ascii=False))
