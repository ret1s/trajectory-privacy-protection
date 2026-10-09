# Public-context storage audit — 07/10/2026

Read-only inspection of the retained dynamic provider fixture. This adds storage
accounting, without changing a benchmark or running a protection mechanism.

| Structure | Decimal MB | Meaning |
|---|---:|---|
| Compressed L60 NPZ | 1.71 | File bytes on disk |
| L60 int32 signatures | 95.31 | Uncompressed array payload |
| int32 access index | 0.26 | Uncompressed array payload |
| Materialized L10 prefix | 15.89 | Analytic payload for the client planner's depth |
| 256 float64 distance vectors | 135.56 | Analytic payload if the default ranking cache is full |
| 4,096 float64 distance vectors | 2,168.88 | Analytic payload if the offline evaluation cache is full |

L60 is a provider/evaluator fixture. The current client planner uses L10
signatures; the L30 service response is a separate depth at the five submitted
coordinates. The L60 table is not a mandatory global client L30 resource.

These are payload/file sizes, **not measured peak RAM**. They exclude graph
objects, beliefs, Python overhead, temporary buffers and response processing.
Neither cache limit establishes how many entries a process actually held.
Device latency, energy and memory feasibility need separate measurements.

[readout.json](readout.json) pins the inspected archive, configuration, source
code and audit runner. NPY headers authenticate shape, dtype and item size;
cache estimates multiply graph state count by vector item size and entry cap.

```sh
python -m experiments.audit_public_resource_footprint_20261007 --check
```

The check authenticates the existing readout; it never replaces it.
