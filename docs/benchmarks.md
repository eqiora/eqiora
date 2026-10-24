# Benchmarks

Benchmark verification is being rebuilt around a smaller set of independently
derived checks. The broad host-CPU suite is no longer a required development CI
gate, so this page currently lists no verified benchmark results.

For executable problem classes and their current verification status, see the
[capability matrix](capability-matrix.md). The [gallery](verification/gallery/README.md)
contains existing worked examples. The [roadmap](roadmap.md) orders further work.

A benchmark result must state the complete problem, geometry, boundary and initial
data, numerical method, reference quantities, tolerances, and execution environment.
Source and result lineage must identify the actual run; a successful example or
a retained fixture is not by itself benchmark acceptance.

## Runtime investigations

The [2026-10-24 repeated-Run baseline](performance/2026-10-24-repeated-runs.md)
records two installed-API workloads, reproducible inputs and remaining costs.
It is an initial performance investigation, not a verified speedup or scalability claim.
