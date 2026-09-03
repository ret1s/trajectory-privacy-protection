"""
Loader and preprocessor for the Microsoft GeoLife GPS trajectory dataset.

GeoLife v1.3 (Zheng et al., Microsoft Research Asia): 182 users, ~17,621
trajectories, mostly in Beijing, 2007-2012. Each trajectory is a .plt file:
6 header lines, then one point per line:

    lat,lon,0,altitude_feet,days_since_1899,date,time

Download (see data/README.md):
    https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip
extracted to data/raw/.
"""
import os
import glob
from datetime import datetime

from haversine import haversine, Unit

GEOLIFE_ROOT = os.path.join(
    os.path.dirname(__file__), "raw", "Geolife Trajectories 1.3", "Data"
)

# Dense central-Beijing box (Haidian/Wudaokou area, where GeoLife coverage is
# thickest). Keeping the box small keeps the OSM road graph manageable.
BEIJING_BBOX = (39.96, 116.29, 40.02, 116.36)  # (min_lat, min_lon, max_lat, max_lon)


def parse_plt(path):
    """Parse one .plt file into a list of (lat, lon, datetime) tuples."""
    points = []
    with open(path) as f:
        for line in list(f)[6:]:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            lat, lon = float(parts[0]), float(parts[1])
            ts = datetime.strptime(parts[5] + " " + parts[6], "%Y-%m-%d %H:%M:%S")
            points.append((lat, lon, ts))
    return points


def iter_user_files(root=GEOLIFE_ROOT, users=None):
    """Yield (user_id, plt_path) for every trajectory file."""
    for user_dir in sorted(glob.glob(os.path.join(root, "*"))):
        user_id = os.path.basename(user_dir)
        if users is not None and user_id not in users:
            continue
        for plt in sorted(glob.glob(os.path.join(user_dir, "Trajectory", "*.plt"))):
            yield user_id, plt


def _in_bbox(lat, lon, bbox):
    return bbox[0] <= lat <= bbox[2] and bbox[1] <= lon <= bbox[3]


def resample(points, interval_s=60):
    """Downsample a point list to at least `interval_s` seconds between points."""
    if not points:
        return []
    out = [points[0]]
    for p in points[1:]:
        if (p[2] - out[-1][2]).total_seconds() >= interval_s:
            out.append(p)
    return out


def _split_on_gaps(points, max_gap_s):
    """Split a point sequence wherever consecutive points are > max_gap_s apart."""
    segments, current = [], []
    for p in points:
        if current and (p[2] - current[-1][2]).total_seconds() > max_gap_s:
            segments.append(current)
            current = []
        current.append(p)
    if current:
        segments.append(current)
    return segments


def load_trajectories(
    bbox=BEIJING_BBOX,
    n_trajectories=20,
    min_points=20,
    max_points=60,
    interval_s=60,
    max_gap_s=300,
    min_span_m=500,
    users=None,
    max_per_user=3,
    root=GEOLIFE_ROOT,
):
    """
    Load real GeoLife trajectories suitable for the privacy benchmark.

    Filters to `bbox`, splits on time gaps, resamples to ~1 point/interval_s,
    keeps trips with min_points..max_points points spanning >= min_span_m, and
    stops after n_trajectories. Returns a list of dicts:
        {"user": str, "file": str, "points": [(lat, lon)], "times": [datetime]}
    """
    result = []
    per_user = {}
    for user_id, plt in iter_user_files(root, users):
        if per_user.get(user_id, 0) >= max_per_user:
            continue
        raw = parse_plt(plt)
        inside = [p for p in raw if _in_bbox(p[0], p[1], bbox)]
        if len(inside) < min_points:
            continue
        for segment in _split_on_gaps(inside, max_gap_s):
            pts = resample(segment, interval_s)
            if len(pts) < min_points:
                continue
            pts = pts[:max_points]
            span = haversine(
                (pts[0][0], pts[0][1]), (pts[-1][0], pts[-1][1]), unit=Unit.METERS
            )
            if span < min_span_m:
                continue
            result.append(
                {
                    "user": user_id,
                    "file": os.path.basename(plt),
                    "points": [(p[0], p[1]) for p in pts],
                    "times": [p[2] for p in pts],
                }
            )
            per_user[user_id] = per_user.get(user_id, 0) + 1
            if len(result) >= n_trajectories:
                return result
            if per_user[user_id] >= max_per_user:
                break
    return result


def detect_stay_points(points, dist_thresh_m=200.0, time_thresh_s=1200.0):
    """Stay-point detection of Li et al. (ACM GIS 2008) / Zheng et al. (WWW 2009):
    a maximal run of consecutive fixes staying within `dist_thresh_m` for at
    least `time_thresh_s` collapses to one stay point at the run's mean
    coordinate. Defaults 200 m / 20 min are the standard GeoLife values.
    Returns a list of (lat, lon) stay-point centres."""
    stays = []
    n = len(points)
    i = 0
    while i < n:
        j = i + 1
        while j < n:
            d = haversine(
                (points[i][0], points[i][1]), (points[j][0], points[j][1]),
                unit=Unit.METERS,
            )
            if d > dist_thresh_m:
                break
            j += 1
        dt = (points[j - 1][2] - points[i][2]).total_seconds()
        if dt >= time_thresh_s and j - i >= 2:
            lat = sum(p[0] for p in points[i:j]) / (j - i)
            lon = sum(p[1] for p in points[i:j]) / (j - i)
            stays.append((lat, lon))
            i = j
        else:
            i += 1
    return stays


def load_stay_points(
    bbox=BEIJING_BBOX,
    n_homes=60,
    dist_thresh_m=200.0,
    time_thresh_s=1200.0,
    interval_s=60,
    max_gap_s=600,
    max_per_user=2,
    dedup_m=25.0,
    users=None,
    root=GEOLIFE_ROOT,
):
    """Extract a population of distinct 'significant locations' (stay-point
    centres) from GeoLife inside `bbox`, for the multi-home averaging study.

    Each returned dict is a UNIQUE, SOURCE-BACKED study unit (verifier R4-001):

        {"user", "home": (lat, lon)  # full precision,
         "file": <plt basename>, "seg": <segment idx>, "stay": <stay idx>,
         "uid": "<user>/<file>/seg<seg>/stay<stay>"}

    Two guards make the population a real primary key: (1) near-identical stays
    of the SAME user (within `dedup_m` metres, e.g. the same place detected
    twice or revisited) collapse to their first occurrence — repeated visits are
    NOT counted as independent locations; (2) the per-user cap counts only
    distinct locations. The caller can assert `uid` uniqueness.
    """
    kept = []
    per_user = {}
    for user_id, plt in iter_user_files(root, users):
        if per_user.get(user_id, 0) >= max_per_user:
            continue
        raw = parse_plt(plt)
        inside = [p for p in raw if _in_bbox(p[0], p[1], bbox)]
        if len(inside) < 5:
            continue
        fname = os.path.basename(plt)
        for seg_i, segment in enumerate(_split_on_gaps(inside, max_gap_s)):
            pts = resample(segment, interval_s)
            for stay_i, (lat, lon) in enumerate(
                detect_stay_points(pts, dist_thresh_m, time_thresh_s)
            ):
                # Drop a stay near an already-kept stay of the same user.
                if any(
                    k["user"] == user_id
                    and haversine((lat, lon), k["home"], unit=Unit.METERS) <= dedup_m
                    for k in kept
                ):
                    continue
                kept.append({
                    "user": user_id, "home": (lat, lon),
                    "file": fname, "seg": seg_i, "stay": stay_i,
                    "uid": f"{user_id}/{fname}/seg{seg_i}/stay{stay_i}",
                })
                per_user[user_id] = per_user.get(user_id, 0) + 1
                if per_user[user_id] >= max_per_user:
                    break
            if per_user.get(user_id, 0) >= max_per_user:
                break
        if len(kept) >= n_homes:
            break
    kept = kept[:n_homes]
    uids = [k["uid"] for k in kept]
    assert len(uids) == len(set(uids)), "stay-point uids must be unique (R4-001)"
    return kept


if __name__ == "__main__":
    trajs = load_trajectories(n_trajectories=5)
    print(f"Loaded {len(trajs)} trajectories")
    for t in trajs:
        print(
            f"  user {t['user']} {t['file']}: {len(t['points'])} pts, "
            f"{t['times'][0]} -> {t['times'][-1]}"
        )
