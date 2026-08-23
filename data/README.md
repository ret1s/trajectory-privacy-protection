# Datasets

Raw data lives in `data/raw/` (gitignored). To reproduce the benchmark:

## GeoLife v1.3 (primary dataset)

```bash
cd data/raw
curl -L -o geolife.zip "https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip"
unzip geolife.zip     # -> "Geolife Trajectories 1.3/"
```

182 users, 17,621 GPS trajectories (mostly Beijing, 2007-2012), 313 MB zip.
Loader: `data/geolife.py` (`load_trajectories()` filters to the dense
central-Beijing bbox, splits on gaps, resamples).

## Beijing road graph

Overpass API thường rate-limit; cách ổn định là build từ extract BBBike:

```bash
cd data/raw
curl -L -o Beijing.osm.gz "https://download.bbbike.org/osm/bbbike/Beijing/Beijing.osm.gz"
```

```python
import gzip, shutil, pickle
import osmnx as ox
from data.geolife import BEIJING_BBOX

with gzip.open('data/raw/Beijing.osm.gz','rb') as fi, open('data/raw/Beijing.osm','wb') as fo:
    shutil.copyfileobj(fi, fo)
G = ox.graph_from_xml('data/raw/Beijing.osm', simplify=True, retain_all=True)
b = BEIJING_BBOX
G = ox.truncate.truncate_graph_bbox(G, bbox=(b[1], b[0], b[3], b[2]))
pickle.dump(G, open('data/raw/beijing_graph.pkl','wb'))
```

Kết quả: ~78k nodes / ~209k edges cho bbox thí nghiệm.

## Porto Taxi (secondary, chưa tích hợp)

UCI, không cần đăng ký, CC-BY 4.0:
https://archive.ics.uci.edu/dataset/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015
