# Pre-scoring simulation correction — 2026-09-10

The first fixed-seed build (`urban_fresh_v1`) failed the existing <=10.1 m
one-second displacement check at u309_03, time 426→427 s: 10.8665139563 m,
internal lane :1360099085_1_0 → external lane 1150482281#1_2. No released
protection results on these families have been generated or inspected.

SUMO's default lane change is instantaneous; its documented simple continuous
model uses `--lanechange.duration`. A native same-seed/day diagnostic using 3 s
reduced this trip's maximum step to 9.204685 m without changing GPS samples by
hand. Use this fixed option for **all 12 families**, not only the failed trip.
Rebuild as urban_fresh_v2 in a different native work directory. Retain v1 as
rejected evidence, do not publish it into the evaluation database. Keep all
original physical gates, seeds, scenario definitions and benchmark parameters.

Reference: https://eclipse.dev/sumo/docs/Simulation/SublaneModel.html

The database shard is a new root, not a child release inside the historical DB.
Its source metadata therefore names `external_ancestor` with the old file and
hash; the store's `base_dataset_sha256` key is reserved for in-shard ancestors.
Publication must verify the external old release through the old read-only DB.
