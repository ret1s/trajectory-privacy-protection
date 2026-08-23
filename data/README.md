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

Overpass API thường rate-limit; cách ổn định là build từ extract BBBike. Dùng
script tái lập được (nó tải, verify hash, cắt bbox ĐÚNG thứ tự OSMnx 2.x, giữ
thành phần liên thông lớn nhất, và ghi manifest):

```bash
python -m data.build_beijing_graph
```

Script tương đương các bước sau (thứ tự bbox OSMnx 2.x là
`(left,bottom,right,top) = (min_lon,min_lat,max_lon,max_lat)`, và **bắt buộc** giữ
largest weakly-connected component để đồ thị liên thông):

```python
import gzip, shutil, pickle, networkx as nx
import osmnx as ox
from data.geolife import BEIJING_BBOX
b = BEIJING_BBOX  # (min_lat, min_lon, max_lat, max_lon)
with gzip.open('data/raw/Beijing.osm.gz','rb') as fi, open('data/raw/Beijing.osm','wb') as fo:
    shutil.copyfileobj(fi, fo)
G = ox.graph_from_xml('data/raw/Beijing.osm', simplify=True, retain_all=True)
G = ox.truncate.truncate_graph_bbox(G, bbox=(b[1], b[0], b[3], b[2]))   # 19,634 / 47,745, 4,313 WCC
G = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()  # largest WCC
pickle.dump(G, open('data/raw/beijing_graph.pkl','wb'))                 # 13,813 / 41,040, 1 WCC
```

Kết quả: **13.813 nodes / 41.040 edges**, 1 thành phần liên thông, cho bbox thí
nghiệm (lat 39,96–40,02, lon 116,29–116,36). Manifest (hash, counts, extent,
versions) sinh ở `data/beijing_graph.manifest.json`.

## Porto Taxi (secondary, chưa tích hợp)

UCI, không cần đăng ký, CC-BY 4.0:
https://archive.ics.uci.edu/dataset/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015
