"""Thesis-native, same-scale geographic support; not a population-density map."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from data.sumo_demo import _load_sumolib
from experiments.run_service_cover import ROOT, read, sha, write


def plot():
    folder = ROOT / 'artifacts/datasets/urban_shadow_v1'
    aux = read(folder / 'dataset.json')
    verified = read(folder / 'verification.json')
    assert verified['verified'] and verified['dataset_sha256'] == sha(folder / 'dataset.json')
    net = _load_sumolib().net.readNet(str(ROOT / aux['network']['path']), withInternal=True)
    old = read(ROOT / 'artifacts/benchmarks/prior_factors/training.json')
    # The attacker uses a local metric frame, not SUMO's network offset. Project
    # raw GPS for all panels through the SAME native network conversion.
    xy = {'old': np.unique([net.convertLonLat2XY(p['lon'], p['lat'])
        for r in old['records'] for p in r['points']], axis=0)}
    for split, name in (('development_train', 'training'), ('development_validation', 'holdout')):
        points = []
        for r in aux['records']:
            if r['split'] != split:
                continue
            trace = aux['traces'][r['session_ids'][0]]
            points.extend(net.convertLonLat2XY(trace[i]['lon'], trace[i]['lat']) for i in r['observed_indices'][0])
        xy[name] = np.unique(points, axis=0)
    assert [len(xy[k]) for k in xy] == [36, 1345, 322]
    roads = [np.asarray(l.getShape()) / 1000 for e in net.getEdges()
             for l in e.getLanes() if l.allows('passenger')]
    all_road = np.concatenate(roads)
    bounds = np.vstack([all_road.min(axis=0), all_road.max(axis=0)])
    assert all(np.all((v / 1000 >= bounds[0] - .1) & (v / 1000 <= bounds[1] + .1)) for v in xy.values())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11.5})
    fig, axes = plt.subplots(1, 3, figsize=(9, 5.3), sharex=True, sharey=True)
    specs = [('old', 'Học cũ\n2 nhóm, 36 điểm', '#3c4147', 'x'),
             ('training', 'Học bổ sung\n64 nhóm, 1.345 điểm', '#2875b9', 'o'),
             ('holdout', 'Giữ lại\n16 nhóm, 322 điểm', '#c87026', '^')]
    for ax, (key, title, color, marker) in zip(axes, specs):
        ax.add_collection(LineCollection(roads, colors='#b6bbc1', linewidths=.30, alpha=.55))
        coordinates = xy[key] / 1000
        opts = {'facecolors': 'none', 'edgecolors': color} if key == 'holdout' else {'color': color}
        ax.scatter(coordinates[:, 0], coordinates[:, 1], s=7 if key == 'training' else 15,
                   marker=marker, linewidths=.65, zorder=3, **opts)
        ax.set_title(title, fontsize=11.5, pad=9)
        ax.set_xlim(bounds[0, 0] - .1, bounds[1, 0] + .1)
        ax.set_ylim(bounds[0, 1] - .1, bounds[1, 1] + .1)
        ax.set_aspect('equal')
        ax.set_xlabel('X cục bộ (km)')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=10.5)
    axes[0].set_ylabel('Y cục bộ (km)')
    fig.suptitle('Phân bố tọa độ nhãn dùng học và kiểm tra đối thủ', fontsize=14, y=.98)
    fig.text(.5, .914, 'Mỗi dấu là một tọa độ khác nhau; cùng mạng đường SUMO, cùng tỉ lệ trục.',
             ha='center', fontsize=11)
    fig.text(.5, .018, 'Nguồn: SUMO + OpenStreetMap contributors (ODbL). Không biểu diễn mật độ cư dân.',
             ha='center', fontsize=10, color='#3c4147')
    fig.subplots_adjust(top=.76, bottom=.15, left=.075, right=.985, wspace=.16)
    destination = folder / 'spatial_support.png'
    fig.savefig(destination, dpi=300, facecolor='white')
    plt.close(fig)
    write(folder / 'spatial_support_provenance.json', {
        'dataset_sha256': sha(folder / 'dataset.json'), 'network_sha256': aux['network']['sha256'],
        'old_training_sha256': sha(ROOT / 'artifacts/benchmarks/prior_factors/training.json'),
        'source_sha256': sha(__file__), 'image_sha256': sha(destination),
        'unique_xy': {k: len(v) for k, v in xy.items()},
        'axis_bounds_km': bounds.tolist(), 'meaning': 'unique coordinates, not independent people'})


if __name__ == '__main__':
    plot()
