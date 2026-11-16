"""Read-only argument audit of sealed thesis evidence; Python standard library.

No defense sampling, attacker fitting, new trajectory, or benchmark rewrite.
The offline-information control is a conditional analytic bound, NOT a new
measured six-arm experiment. Setup and catalogue-acquisition costs are unknown.
"""
import argparse
from collections import Counter
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / 'artifacts/benchmarks/paper_comparison_20261028_v1'
OLD = ROOT / 'artifacts/benchmarks/main_depth_20261014_v1'
OUT = ROOT / 'artifacts/analysis/thesis_arguments_20261116_v1'
PURPOSES = ('fastest_travel', 'minimum_detour', 'nearest_distance', 'within_radius')
CACHES = ('current', 'known', 'session')
METHODS = ('raw', 'geo_i_Z_k1', 'transprotect_markov', 'semantic_poi', 'fake_queries', 'geoi_bq')


def read(path):
    data = Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if str(path).endswith('.gz') else data)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pinned_path(base, name):
    base = Path(base).resolve()
    path = (base / name).resolve()
    require(path.is_relative_to(base), 'Pin escapes root: ' + name)
    return path


def authenticate():
    """Authenticate existing receipts and all their pinned source files."""
    spec = read(ROOT / 'configs/geo_i_bq_v1.json')
    require(spec['schema'] == 'geoi-bq-research-reference-v1', 'Wrong release schema')
    sources = {ROOT / 'configs/geo_i_bq_v1.json', MAIN / 'validation.json',
               MAIN / 'results.json', MAIN / 'protocol.json', MAIN / 'resources.json',
               OLD / 'independent_validation.json', OLD / 'release_readout.json'}
    for name, digest in spec['files_sha256'].items():
        path = pinned_path(ROOT, name)
        require(sha(path) == digest, 'Frozen release changed: ' + name)
        sources.add(path)
    require(len(spec['files_sha256']) == 108, 'Unexpected frozen pin inventory')
    audit = read(MAIN / 'validation.json')
    require(audit['status'] == 'pass', 'Main benchmark has not passed validation')
    require(audit['results_sha256'] == sha(MAIN / 'results.json'), 'Main results changed')
    require(audit['protocol_sha256'] == sha(MAIN / 'protocol.json'), 'Main protocol changed')
    for name, digest in audit['files_sha256'].items():
        path = pinned_path(MAIN, name)
        require(sha(path) == digest, 'Benchmark receipt mismatch: ' + name)
        sources.add(path)
    protocol = read(MAIN / 'protocol.json')
    for name, digest in {**protocol['inputs_sha256'], **protocol['source_sha256']}.items():
        path = pinned_path(ROOT, name)
        require(sha(path) == digest, 'Main source/input changed: ' + name)
        sources.add(path)
    for name, digest in protocol['original_families'].items():
        path = pinned_path(OLD / 'families', name)
        require(sha(path) == digest, 'Frozen original family changed: ' + name)
        sources.add(path)
    old_audit = read(OLD / 'independent_validation.json')
    require(old_audit['status'] == 'pass', 'Original evidence has not passed')
    require(old_audit['results_sha256'] == sha(OLD / 'results.json'), 'Original results changed')
    readout = read(OLD / 'release_readout.json')
    require(readout['results_sha256'] == sha(OLD / 'results.json'), 'Readout belongs to different results')
    return spec, protocol, sources


def mask_indices(encoded, size):
    """Decode numpy.packbits' big-endian byte convention, rejecting padding."""
    require(isinstance(encoded, str), 'Mask is missing or not a hex string')
    try:
        data = bytes.fromhex(encoded)
    except ValueError as exc:
        raise ValueError('Invalid mask encoding') from exc
    require(size > 0 and len(data) == (size + 7) // 8, 'Wrong mask byte count')
    padding = 8 * len(data) - size
    require(not padding or not data[-1] & ((1 << padding) - 1), 'Nonzero mask padding')
    return {i for i in range(size) if data[i // 8] & (1 << (7 - i % 8))}


def fraction_summary(values, catalogue_size=None):
    require(bool(values), 'Empty sample is not a successful result')
    total = sum(values)
    average = Fraction(total, len(values))
    result = {'count': len(values), 'sum': total, 'mean_exact': str(average),
              'mean': float(average), 'min': min(values), 'max': max(values)}
    if catalogue_size is not None:
        result.update(catalogue_fraction_exact=str(average / catalogue_size),
                      catalogue_percent=float(100 * average / catalogue_size))
    return result


def family_values(family, method, cache):
    """Match the frozen protocol: purpose mean over family ticks, then macro.

    A failed arm is a failure, never privacy success. This audit refuses missing
    or failed arms instead of excluding them from an apparently clean cohort.
    Legitimate undefined references remain None and are omitted only as N/A.
    """
    values = {purpose: [] for purpose in PURPOSES}
    for trip in family['trips']:
        require(trip.get('status', {}).get(method) == 'ok', 'Missing/failed arm: ' + method)
        rows = trip.get('utility', {}).get(method)
        require(isinstance(rows, list) and len(rows) == len(trip['truth']), 'Missing utility rows')
        for row in rows:
            for purpose in PURPOSES:
                cell = row[cache][purpose]
                score = cell['recall5']
                if score is None:
                    require(cell['defined_categories'] == 0, 'Defined reference discarded as N/A')
                    continue
                require(cell['defined_categories'] > 0 and math.isfinite(score) and 0 <= score <= 1,
                        'Invalid defined utility')
                values[purpose].append(score)
    out = {p: mean(v) if v else None for p, v in values.items()}
    defined = [v for v in out.values() if v is not None]
    require(bool(defined), 'No defined utility; cannot certify this family')
    out['macro'] = mean(defined)
    return out


def validate_cohort(families):
    require(len(families) == 48, 'Expected the complete 48-family cohort')
    ids = [f['family_id'] for f in families]
    require(len(ids) == len(set(ids)), 'Duplicate family')
    require(Counter(f['split'] for f in families) == {'train': 20, 'selection': 8, 'test': 20},
            'Wrong family split')
    test = [f for f in families if f['split'] == 'test']
    all_draws = set()
    for family in test:
        trips = family['trips']
        require(len(trips) == 6, 'TEST family must retain both trips and all three draws')
        sessions = {}
        for trip in trips:
            key = trip['session_id']
            require(key not in all_draws, 'Duplicate retained draw')
            all_draws.add(key)
            sessions.setdefault(trip['native_session_id'], []).append(trip['draw'])
        require(len(sessions) == 2 and all(sorted(v) == [1, 2, 3] for v in sessions.values()),
                'Incomplete native session/draw inventory')
    require(len(all_draws) == 120, 'Expected all 120 TEST draws')
    return test


def build(output=OUT):
    spec, protocol, sources = authenticate()
    resources = read(MAIN / 'resources.json')
    size = resources['pois']
    require(size == 418, 'Unexpected catalogue size')
    paths = sorted((MAIN / 'families').glob('*.json.gz'))
    families = [read(p) for p in paths]
    sources.update(paths)
    test = validate_cohort(families)
    results = read(MAIN / 'results.json')
    source_utility = {(x['method'], x['cache']): x for x in results['utility']}
    verified_cells = 0
    for method in METHODS:
        coverage = results['coverage'][method]
        require(coverage == {'attempted': 120, 'completed': 120, 'failed': 0}, 'Incomplete coverage')
        for cache in CACHES:
            aggregate = []
            by_id = {r['family_id']: r for r in source_utility[method, cache]['family_metrics']}
            require(set(by_id) == {f['family_id'] for f in test}, 'Family metric coverage mismatch')
            for family in test:
                row = family_values(family, method, cache)
                for metric, value in row.items():
                    expected = by_id[family['family_id']][metric]
                    require(value is not None and expected is not None and abs(value - expected) < 1e-12,
                            'Saved family metric mismatch')
                    verified_cells += 1
                aggregate.append(row)
            for metric in PURPOSES + ('macro',):
                expected = source_utility[method, cache]['metrics'][metric]['mean']
                require(abs(mean(r[metric] for r in aggregate) - expected) < 1e-12,
                        'Saved utility aggregate mismatch')

    first, current, final_current, final_session, reads, spent = [], [], [], [], [], []
    prefix_counts = [[] for _ in range(31)]
    branches = Counter()
    runs = []
    for family in test:
        original = read(OLD / 'families' / (family['family_id'] + '.json.gz'))
        old_trips = {t['session_id']: t for t in original['trips']}
        for trip in family['trips']:
            saved = old_trips[trip['session_id']]
            require(trip['events']['geoi_bq'] == saved['events']['cache_target_g1'], 'Proposed wire changed')
            require([p['t'] for p in trip['truth']] == list(range(0, 601, 20)), 'Unexpected target clock')
            require(len(trip['masks']['geoi_bq']) == 31, 'Incomplete received-mask history')
            previous = set()
            for tick, row in enumerate(trip['masks']['geoi_bq']):
                sets = {k: mask_indices(row[k], size) for k in CACHES}
                require(sets['current'] <= sets['known'] <= sets['session'], 'Invalid cache inclusion')
                require(previous <= sets['session'], 'Session cache is not monotone')
                previous = sets['session']
                n = len(sets['current']); current.append(n)
                prefix_counts[tick].append(len(previous))
                if tick == 0: first.append(n)
                if tick == 30:
                    final_current.append(n); final_session.append(len(previous))
            ledger = saved['GeoI_ledger']
            require(len(ledger) == 31, 'Incomplete Geo-I ledger')
            total = 0; count = 0; last_read = None
            for point, row in zip(trip['truth'], ledger):
                cost = row['cost_units']; total += cost; branches[row['branch']] += 1
                require(cost in (0, 1, 2) and total == row['spent_units'] and total <= 23,
                        'Ledger prefix/cap mismatch')
                if row['private_read']:
                    require(cost > 0 and (last_read is None or point['t'] - last_read >= 60),
                            'Invalid paced private read')
                    count += 1; last_read = point['t']
                else:
                    require(cost == 0, 'Unread GPS has private cost')
            reads.append(count); spent.append(total)
            runs.append({'session_id': trip['session_id'], 'family_id': family['family_id'],
                         'private_reads': count, 'spent_units': total,
                         'first_unique': first[-1], 'final_unique': final_session[-1]})
    cap_units = 23
    maximal_reads = 1 + 600 // 60
    maximal_spend = 1 + 2 * (maximal_reads - 1)
    require(maximal_spend == 21 and max(spent) <= maximal_spend < cap_units,
            'Duration/cap regime calculation changed')
    readout = read(OLD / 'release_readout.json')
    g0 = next(x for x in readout['comparisons'] if x['control'] == 'cache_target_g0')['metrics']['macro']
    old_results = read(OLD / 'results.json')
    old_utility = {(x['method'], x['cache']): x for x in old_results['utility']}
    g0_mean = old_utility['cache_target_g0', 'session']['metrics']['macro']['mean']
    g1_mean = old_utility['cache_target_g1', 'session']['metrics']['macro']['mean']
    require(abs(g1_mean - g0_mean - g0['delta']) < 1e-12, 'Gamma comparison mean mismatch')
    ratio = Fraction('2.437372'); cs = Fraction('0.23')
    payload = {
        'schema': 'thesis-argument-audit-v1', 'date': '2026-11-16',
        'scope': 'Read-only audit of inspected frozen cohort; no new defense or attacker experiment',
        'coverage': {'families': 48, 'test_families': 20, 'native_sessions': 40, 'retained_test_draws': 120,
                     'failed_test_arms': 0, 'verified_family_utility_cells': verified_cells},
        'catalogue_size': size,
        'saturation': {'first_packet': fraction_summary(first, size),
                       'all_current_packets': fraction_summary(current, size),
                       'last_current_packet': fraction_summary(final_current, size),
                       'final_session_cache': fraction_summary(final_session, size),
                       'session_by_tick': [dict(t=i * 20, **fraction_summary(v, size))
                                           for i, v in enumerate(prefix_counts)]},
        'cap': {'private_reads': fraction_summary(reads), 'spent_units': fraction_summary(spent),
                'cap_units': cap_units, 'maximal_reads_in_600s': maximal_reads,
                'maximal_spend_in_600s': maximal_spend, 'runs_reaching_cap': sum(v == cap_units for v in spent),
                'cap_exhausted_regime_exercised': False, 'branches': dict(branches),
                'interpretation': 'No 600-second run can exhaust 23 units at this cadence; longer-duration utility remains untested'},
        'spatial_q_crossover': {'Cs_per_m_exact': str(cs), 'R_infinity_decimal_upper_exact': str(ratio),
                                'log_R_over_Cs_m': math.log(float(ratio)) / float(cs),
                                'status': 'binary64 illustrative derived value; not an exact decimal certificate',
                                'meaning': 'Coarse Cs*D bound beats the all-duration Q ceiling only below this distance; finite-prefix/refined bounds differ'},
        'gamma0_comparison': {'primary_mean': g1_mean, 'public_base_mean': g0_mean,
                              'primary_minus_public_base': g0['delta'], 'official_paired95': g0['paired95'],
                              'source': 'artifacts/benchmarks/main_depth_20261014_v1/release_readout.json',
                              'CI_recomputed': False, 'scope': 'original controlled ablation; no universal dominance claim'},
        'offline_information_control': {
            'status': 'conditional analytical information bound; NOT a measured six-arm benchmark',
            'assumptions': ['Client possesses the same complete valid static POI records and directed ranking graphs as the reference',
                            'The private GPS/request is available locally and both use the same filter/order/tie break',
                            'No received-mask restriction forbids using those already-known valid records'],
            'defined_reference_recall_exact': '1', 'undefined_reference': 'N/A, not a successful service case',
            'derivation': 'For each defined reference T, full valid information U=P implies T subset U and |T intersection U|/|T|=1',
            'during_answer_coordinate_queries_exact': '0',
            'setup_and_catalogue_acquisition_cost': None,
            'setup_cost_scope': 'Not measured; zero online answer queries does not imply zero total acquisition/storage/refresh cost',
            'excluded_claims': ['No measured latency, metadata bandwidth or energy result', 'No claim the real provider permits free full-catalogue access',
                                'No deployment claim for dynamic or previously unknown attributes'],
            'information_boundary': 'Prototype reference/reply share catalogue data; production must separate design proxy, evaluator oracle and deployable client inputs'},
        'runs': runs,
    }
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    sources.update({Path(__file__).resolve(), ROOT / 'core/research_release.py',
                    ROOT / 'benchmark/exact_public_reference.py', ROOT / 'benchmark/public_poi_context.py',
                    ROOT / 'benchmark/query_purpose.py', ROOT / 'benchmark/public_segment_scores.py'})
    audit_protocol = {'schema': 'thesis-argument-audit-protocol-v1', 'inputs_sha256': {
        str(p.relative_to(ROOT)): sha(p) for p in sorted(sources)},
        'frozen_pins_verified': len(spec['files_sha256']), 'sampling_fitting_or_source_mutation': False,
        'aggregation': 'Exact integer/Fraction mask and ledger counts; saved float utility compared at 1e-12; original CI copied with provenance',
        'failure_policy': 'Abort on missing, failed or incomplete retained arm; never silently count as successful privacy or service'}
    write(output / 'protocol.json', audit_protocol)
    write(output / 'results.json', payload)
    (output / 'argument_audit.tex').write_text(render_latex(payload))
    write(output / 'validation.json', {'schema': 'thesis-argument-audit-validation-v1', 'status': 'pass',
          'files_sha256': {p.name: sha(p) for p in (output / 'protocol.json', output / 'results.json', output / 'argument_audit.tex')},
          'source_frozen': True, 'defense_resampled': False, 'attacker_refitted': False})
    return payload


def render_latex(payload):
    """Bind displayed statistics to audited values, not a second numeric copy."""
    first=payload['saturation']['first_packet']
    final=payload['saturation']['final_session_cache']
    cap=payload['cap']
    def number(summary):
        return f"{float(Fraction(summary['mean_exact'])):.3f}".rstrip('0').rstrip('.').replace('.',',')
    values={
        '@DRAWS@':str(payload['coverage']['retained_test_draws']),
        '@FIRST@':number(first), '@FINAL@':number(final),
        '@SIZE@':str(payload['catalogue_size']),
        '@FIRST_RANGE@':f"{first['min']}--{first['max']}",
        '@FINAL_RANGE@':f"{final['min']}--{final['max']}",
        '@READS@':str(cap['maximal_reads_in_600s']),
        '@SPEND@':str(cap['maximal_spend_in_600s']),
        '@CAP@':str(cap['cap_units']),
    }
    tex = r'''% Generated by experiments.audit_thesis_arguments_20261116; do not edit values.
\begin{table}[htbp]\centering\small
\caption{Kiểm tra phạm vi bằng chứng trên toàn bộ @DRAWS@ TEST draws đã đóng băng.}\label{tab:argument-audit}
\begin{tabular}{lrr}\toprule
Đại lượng & Trung bình & Khoảng quan sát\\\midrule
POI unique từ packet đầu & @FIRST@ / @SIZE@ & @FIRST_RANGE@\\
POI unique trong cache cuối phiên & @FINAL@ / @SIZE@ & @FINAL_RANGE@\\
\bottomrule\end{tabular}
\end{table}
\noindent Với clock 0--600 s và private reads cách tối thiểu 60 s, tối đa @READS@ lần đọc
tiêu thụ nhiều nhất $1+2(@READS@-1)=@SPEND@<@CAP@$ đơn vị. Vì vậy benchmark chưa kiểm tra chế độ
hết ngân sách Geo-I. Kết quả trên là audit nguồn đã lưu, không phải benchmark mới.
'''
    for name,value in values.items():tex=tex.replace(name,value)
    return tex


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def check(output=OUT):
    output = Path(output)
    receipt = read(output / 'validation.json')
    require(receipt['status'] == 'pass', 'Audit has not passed')
    for name, digest in receipt['files_sha256'].items():
        require(sha(pinned_path(output, name)) == digest, 'Audit output changed: ' + name)
    for name, digest in read(output / 'protocol.json')['inputs_sha256'].items():
        require(sha(pinned_path(ROOT, name)) == digest, 'Audit input changed: ' + name)
    data = read(output / 'results.json')
    require(data['coverage']['retained_test_draws'] == 120 and len(data['runs']) == 120, 'Incomplete audit')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    data = (check if args.check else build)(args.output)
    print('PASS: 108 frozen pins; 120 TEST draws; saturation/cap/utility audited; no model rerun')
    print('First/final unique POI:', data['saturation']['first_packet']['mean_exact'],
          data['saturation']['final_session_cache']['mean_exact'])


if __name__ == '__main__':
    main()
