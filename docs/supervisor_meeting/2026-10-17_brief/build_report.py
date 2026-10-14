"""Build pinned meeting tables and illustrations from retained evidence only.

No protection sampling, model fitting, or benchmark evaluation is run. A TRAIN
sample is chosen by fixed family/trip order, independent of its privacy score.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import gzip
import hashlib
import json
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
MAIN = 'artifacts/benchmarks/main_depth_20261014_v1'
BARE = 'artifacts/benchmarks/geo_i_anchor_comparator_20261014_v1'
SAMPLE = MAIN+'/families/cache-001.json.gz'
PRIMARY = 'cache_target_g1'
NETWORK = 'artifacts/datasets/future_controlled_20261005_v1/public_native.net.xml.gz'
PURPOSES = ('nearest_distance', 'fastest_travel', 'within_radius', 'minimum_detour')
SOURCES = [MAIN+'/'+n for n in ('results.json', 'release_readout.json', 'protocol.json', 'independent_validation.json')]
SOURCES += [BARE+'/'+n for n in ('results.json', 'independent_validation.json', 'protocol.json')]
SOURCES += [SAMPLE, 'artifacts/benchmarks/public_depth_design_20261014_v1/certificate.json',
            'configs/geo_i_bq_v1.json', 'docs/reviews/2026-10-14_mathematical_completion.json',
            'docs/research/2026-10-14_geoi_bq_mathematical_foundation.md',
            'docs/reviews/2026-10-14_mathematical_adversarial_audit.md',
            'docs/supervisor_meeting/2026-10-09_update/report_explained.tex',
            NETWORK,
            'data/lane_states.py', 'core/road_network.py']


def read(name):
    p = ROOT/name
    data = p.read_bytes()
    return json.loads(gzip.decompress(data) if p.suffix == '.gz' else data)


def num(v, digits=3):
    return f'{v:.{digits}f}'.replace('.', ',')


def row(cells):
    return ' & '.join(cells)+r'\\'


def generated():
    main, bare, readout = read(MAIN+'/results.json'), read(BARE+'/results.json'), read(MAIN+'/release_readout.json')
    family = read(SAMPLE)
    assert family['family_id'] == 'cache-001' and family['split'] == 'train'
    trip = family['trips'][0]
    assert trip['native_session_id'] == 'cache-001_day7' and trip['draw'] == 1
    utility = {r['method']: r['metrics'] for r in main['utility'] if r['cache'] == 'session'}
    labels = [('geo_i_Z_k1', 'Geo-I/REM: gửi Z, K1'), ('baseline', 'Geo-I + Service-cover, K5'),
              ('finite_static_g1', 'Giữ initial Q, K5'), ('cache_explore_g1', 'Khám phá lân cận, K5'),
              ('cache_target_g0', r'Nền công khai $\Gamma=0$, K5'), (PRIMARY, r'\textbf{GeoI-BQ $\Gamma=1$, K5}')]
    tables = {}
    tables['UtilityRows'] = []
    tables['CostRows'] = []
    for method, label in labels:
        values = {p: bare['metrics'][p]['bare_mean'] for p in PURPOSES+('macro',)} if method == 'geo_i_Z_k1' else {p: utility[method][p]['mean'] for p in PURPOSES+('macro',)}
        tables['UtilityRows'].append(row([label]+[num(100*values[p]) for p in ('macro',)+PURPOSES]))
        if method in ('geo_i_Z_k1', 'baseline', 'cache_target_g0', PRIMARY):
            cost = bare['cost'] if method == 'geo_i_Z_k1' else main['cost'][method]
            tables['CostRows'].append(row([label, str(round(cost['requests'])), num(cost['reply_bytes']/1000)]))
    attacks = {(r['method'], r['scenario']): r for r in main['attackers']}
    bare_attack = {r['scenario']: r for r in bare['attackers']}
    tables['AttackRows'] = []
    for s in ('S1', 'S2', 'S3', 'S9', 'S10'):
        entries = [bare_attack[s], attacks['baseline', s], attacks[PRIMARY, s]]
        tables['AttackRows'].append(row([s]+[num(100*r['Hit100'])+r'\% / '+num(r['MAE_m'], 1) for r in entries]))
    tables['TimelineRows'] = []
    sample_rows = []
    for t in (0, 20, 40, 60, 80, 120, 180, 300, 600):
        tick = t//20
        ledger, cert = trip['GeoI_ledger'][tick], trip['Q_certificates'][PRIMARY][tick]
        j = tick//3+1
        reserved = Fraction(j, j+1)
        assert abs(cert['reserved_prefix_epsilon_Q']-float(reserved)) < 1e-12
        branch = {'fresh': 'Tạo Z', 'reuse': 'Giữ Z', 'public_clock_skip': 'Không đọc'}[ledger['branch']]
        q_action = 'Chọn đoạn '+str(j) if tick % 3 == 0 else 'Gửi khung đã chọn'
        cache = len(trip['retrieved_POI_indices'][PRIMARY][tick]['session'])
        tables['TimelineRows'].append(row([str(t), branch, str(ledger['spent_units'])+'/23', q_action,
                                           num(float(reserved)), str(cache)]))
        sample_rows.append(dict(t=t, geoi=ledger, Q_action=q_action, reserved_Q=str(reserved),
                                cache_POIs=cache, Q=trip['events'][PRIMARY][tick]['coordinates'],
                                Z=trip['GeoI_anchors'][tick], GPS_xy=trip['truth'][tick]['xy']))
    deltas = {r['control']: r['metrics']['macro'] for r in readout['comparisons']}
    info = dict(primary_recall=utility[PRIMARY]['macro']['mean'], primary_absolute95=readout['primary_absolute95']['macro'],
                matched_baseline_delta=deltas['baseline'], static_delta=deltas['finite_static_g1'],
                public_base_delta=deltas['cache_target_g0'], utility_gate=main['promotion']['utility_gate'],
                cache=readout['POI_coverage'][PRIMARY], fixed_Q_depth=readout['same_Q_L30_counterfactual']['macro'],
                stale_local_GPS=readout['stale_local_GPS_60s'][PRIMARY]['macro'])
    sample = dict(source=SAMPLE, selection='first family lexicographically, first TRAIN trip; not chosen by outcome',
                  split=family['split'], session=trip['session_id'], window_s=[0, 600], rows=sample_rows)
    content = '\n'.join(r'\newcommand{'+'\\'+name+'}{\n'+'\n'.join(rows)+'\n}' for name, rows in tables.items())+'\n'
    pins = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}
    return content, dict(source_sha256=pins, derived=info, benchmarks_rescored=False,
                         evidence_cutoff='2026-10-14', meeting_date='2026-10-17'), sample, trip


def sample_plot(trip):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import numpy as np
    from data.lane_states import build_lane_states
    with tempfile.TemporaryDirectory(prefix='geoi-meeting-public-map-') as scratch:
        net = Path(scratch)/'public.net.xml'
        net.write_bytes(gzip.decompress((ROOT/NETWORK).read_bytes()))
        rn = build_lane_states(net, spacing_m=40.)
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9,
                         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#277DA8', '#D27732', '#439767', '#8D6CB1', '#C85C78']
    q = np.array([e['coordinates'] for e in trip['events'][PRIMARY][:10]])
    xy = np.array([p['xy'] for p in trip['truth'][:10]])
    z = np.array(trip['GeoI_anchors'][:10])
    qxy = np.array([[rn.point_xy(*p) for p in frame] for frame in q])
    zxy = np.array([rn.point_xy(*p) for p in z])
    origin = qxy.reshape(-1, 2).min(axis=0)-200
    end = qxy.reshape(-1, 2).max(axis=0)+200
    def local(points): return (np.asarray(points)-origin)/1000.
    edges = np.array([[rn.xy[a], rn.xy[b]] for a, b in rn.graph.edges()])
    keep = np.any(np.all((edges >= origin) & (edges <= end), axis=2), axis=1)
    backdrop = local(edges[keep])
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.65), gridspec_kw={'width_ratios': [1, 1.2]})
    for ax in axes:
        ax.add_collection(LineCollection(backdrop, colors='#D8DFE1', linewidths=.3, rasterized=False))
        ax.set_xlim(0, (end-origin)[0]/1000); ax.set_ylim(0, (end-origin)[1]/1000)
        ax.set_aspect('equal'); ax.set_xlabel('Đông (km, gốc minh họa)'); ax.set_ylabel('Bắc (km)')
    gps = local(xy); anchors = local(zxy)
    private = np.vstack((gps, anchors))
    lo, hi = private.min(axis=0)-.18, private.max(axis=0)+.18
    axes[0].set_xlim(lo[0], hi[0]); axes[0].set_ylim(lo[1], hi[1])
    axes[0].plot(gps[:, 0], gps[:, 1], '-x', color='#273A45', lw=1.35, ms=4, label='GPS local')
    axes[0].plot(anchors[:, 0], anchors[:, 1], '--D', color='#8D6CB1', lw=1.1, ms=4, label='Z nội bộ')
    offsets = {0: (5, 8), 3: (5, -12), 6: (-23, 8), 9: (5, 8)}
    for tick in (0, 3, 6, 9):
        axes[0].annotate(str(tick*20)+'s', gps[tick], xytext=offsets[tick], textcoords='offset points', fontsize=8,
                         bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=.5))
    axes[0].set_title('Thông tin riêng: GPS và Z', fontsize=10)
    axes[0].legend(loc='upper right', fontsize=8, frameon=True, framealpha=.95)
    q_local = local(qxy)
    for track, color in enumerate(colors):
        points = q_local[:, track]
        axes[1].plot(points[:, 0], points[:, 1], '-o', ms=2.5, lw=1.15, color=color, label=f'Q{track+1}')
        axes[1].annotate(f'Q{track+1}', points[0], xytext=(3, 4), textcoords='offset points', fontsize=8, color=color)
    axes[1].set_title('Máy chủ thấy: 5 tuyến Q, 0–180s', fontsize=10)
    axes[1].legend(loc='upper right', fontsize=7, framealpha=.95)
    fig.tight_layout(pad=.7)
    fig.savefig(HERE/'report_figures'/'sample_paths.pdf', bbox_inches='tight', metadata={'CreationDate': None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    tables, sources, sample, trip = generated()
    payloads = {'report_tables.tex': tables,
                'report_sources.json': json.dumps(sources, ensure_ascii=False, indent=2)+'\n',
                'sample_walkthrough.json': json.dumps(sample, ensure_ascii=False, indent=2)+'\n'}
    if args.check:
        for name, text in payloads.items():
            if (HERE/name).read_text() != text:
                raise SystemExit('Source or derived report changed: '+name)
        print('PASS: source pins, generated tables and fixed TRAIN sample')
    else:
        (HERE/'report_figures').mkdir(exist_ok=True)
        for name, text in payloads.items(): (HERE/name).write_text(text)
        sample_plot(trip)
        print('Report evidence and vector sample figure built; no benchmark rerun')


if __name__ == '__main__': main()
