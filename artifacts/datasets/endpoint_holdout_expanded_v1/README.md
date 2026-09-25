# Expanded endpoint cohort

32 predeclared seeds (1201–1232), 704 completed SUMO trips and 1,118 records
on the existing Beijing OSM network. No construction seed was replaced.
The five priority scenarios use 460 records; S9/S10 use 175. Exact per-case
counts are in `summary.json`; missing records and their construction gates
remain in the dataset's `rejections` array.

The source JSON exceeds 100 MB and is stored losslessly in `dataset.json.gz`.
From the repository root, restore the exact input used by the experiments:

```sh
python -m experiments.endpoint_dataset_archive unpack
```

The helper verifies both SHA-256 hashes from `archive.json` and refuses to
overwrite a different existing JSON. The restored `dataset.json` is ignored
by Git. The frozen protocol is in
`artifacts/benchmarks/endpoint_calendar_expanded_v1/protocol.json`.

This cohort was declared after inspecting the seven-family pilot. Defense
plans and attacker selection are unchanged; the new results are reported
separately from that pilot. New seeds do not imply a new city, real users,
or wholly disjoint road edges.
