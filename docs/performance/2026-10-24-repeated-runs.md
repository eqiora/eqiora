# Initial repeated-Run baseline (2026-10-24)

This investigation advances [#727](https://github.com/eqiora/eqiora/issues/727)
and [#802](https://github.com/eqiora/eqiora/issues/802); it completes neither.
The cylinder profiles prioritize assembly/residual evaluation and preparation.
The second workload, scalar decay, exposes a latency regression. No 2x speedup,
non-regression bound, conventional CFD parity, or general scalability is established.

A [later comparison](2026-10-24-ida-runs.md) records the ordinary IDA integration
candidate against a more recent reference, with exact cylinder field equality
and the remaining measured performance gap. It does not replace this historical
a7 comparison.

## Environment and boundary

Both installations ran on `dide1` (`didelx01`), Linux 6.8.0-94/glibc 2.35,
AMD Ryzen Threadripper PRO 5995WX (128 logical CPUs), pinned to CPU 16.
The CPU was not reserved exclusively. CPython 3.13.14, NumPy 2.5.2 and Gmsh 4.15.2
were identical; OMP, OpenBLAS, MKL and Rayon thread limits were one. No GPU was used.

Reference: installed PyPI `eqiora==0.1.0a7`, source tag commit
`e72f744cf6956e0e3c1f03aad65868d726f8ebbf`, wheel SHA256
`e03b66ae634a85093a9037712e9e05d50635db95c90e264666fbfff84e90b734`.
Candidate: `4292283ea779d6ea5626aa7b6e38f024baf882eb`, release/locked wheel built
on dide1 using Rust 1.98.0, maturin 1.15.0 and CMake 3.31.6, wheel SHA256
`007bf23e4da43d279f883b1157a515599eb0b07682cab20958f2df5be4ad60bc`. This includes #1693's
admission correction; the unchanged original tolerance pair had been rejected
before execution on its parent. The correction does not change Newton acceptance.

Timing brackets only the ordinary installed `eqiora.run(...)` call. Import,
compile, resolution, mesh generation, initial binding, result inspection and file
writes are excluded. Each invocation creates a fresh ordinary Run from the same
immutable Plan and State; the caller supplies no retained provider or Result.
Cold means the first Run after untimed preparation, without flushing OS caches.
After two untimed warmups, seven samples were measured. A cylinder sample is one
Run; a decay sample is the arithmetic mean of 100 Run call timings. Reported
statistics are the median, range and median absolute deviation of those seven
samples. The [data](2026-10-24-repeated-run-data.json) retains all seven sample
values, cold times, actual loads and measurement-driver hashes. No confidence
interval or CI wall-clock gate is inferred from this shared-host sample.

## Inputs

The cylinder is the original a7 packaged transient example: channel
2.2 by 0.41 m, cylinder centre (0.2, 0.2) m and radius 0.05 m; density 1,
dynamic viscosity 0.001 and inlet speed 0.3 in the model's SI units.
Gmsh controls are maximum boundary error 0.0001, target size 0.025, minimum
mean ratio 0.00001 and maximum boundary facets 50. Both meshes have 2386 vertices
and 4512 triangular cells. Geometry, topology, vertices, cells and acceptance
metadata have the same canonical projection hash:
`ac046556dc98188923b773e6fa33348f2344c894e74e23282f0978faaf68029a`.

The method is MINI/P1, Backward Euler dt=0.01 s, ten accepted steps and all ten
outputs. Linear controls: SparseLu/Identity/Fast/Faer, rtol=1e-6, atol=1e-9,
maximum iterations 20000. Newton: rtol=1e-9, atol=1e-11, maximum iterations 16,
maximum line-search halvings 12. Scaling: length 0.41 m, velocity 0.3 m/s,
pressure 0.09 Pa. Independently recomputed steady seeds differed slightly, so
both versions bind the **same original a7 steady seed's physical coefficients**
through `State.initial`. The retained initial input is not a transient answer.
Its SHA256 is `e4dfc16fba39177d6f488c953134ed6c2344a6c46bfb83371b3ebcdae384f35e`;
the harness checks the bound physical payload and mesh projection exactly.

The distinct workload solves x' + x = 0, x(0)=1 with Tsitouras45, initial step
0.01 s, rtol=1e-9, atol=1e-11, requested times (0.25, 0.5, 1) s.
Both installations agree with independently known exp(-t) within the predeclared
1e-8 absolute probe bound (observed maximum error about 1.53e-10). This probe is
not evidence for arbitrary ODEs, adaptive sensitivity or restart correctness.

## Results and limitations

| Workload | a7 median (range) | Candidate median (range) |
| --- | --- | --- |
| Cylinder | 9.315869 s (9.127444–9.399056) | 5.525034 s (5.501070–5.636818) |
| Decay | 97.93255 µs (91.96932–100.53455) | 244.61038 µs (239.03007–256.63096) |

**The cylinder numbers are not same-discretization speedup evidence.** #1251
replaced hard-coded SkewStationary transport by authored volume/natural-facet
actions. Full attribution of the resulting trajectory difference remains open.
Across ten outputs, maximum absolute differences are about 0.0360952 m/s in
vertex velocity, 0.00475337 m/s in bubble coefficients, and 0.0535260 Pa in pressure.
Both Runs satisfying their own acceptance does not establish cross-version
numerical equivalence. Integer output steps match, while serialized clock values
differ by up to 1.39e-17 s; these differences were retained, not normalized away.
The second workload is about 2.5 times slower, so #802's non-regression goal is
also not met. Do not weaken numerical controls or restore obsolete mathematics
to manufacture a comparison.

A separate candidate cylinder Profile totals 5.930868 s. Exclusive self-time
sums partition it as follows; these are not fractions of the unprofiled medians.

| Work | Seconds | Fraction |
| --- | ---: | ---: |
| Assembly and residual trials | 2.654198 | 44.75% |
| Discretization preparation | 1.319481 | 22.25% |
| Numeric LU (20 calls) | 1.077892 | 18.17% |
| Unpartitioned step work | 0.637270 | 10.75% |
| Other linear solve work | 0.109532 | 1.85% |
| Other execution | 0.076670 | 1.29% |
| Postprocess | 0.033664 | 0.57% |
| Symbolic LU (one call) | 0.022162 | 0.37% |

Symbolic preparation already occurs once per Run. Another symbolic cache is not
the first optimization. Preparation needs finer attribution before changing
admission or lineage checks. Decay's separate Profile identifies solve and
postprocess as broad owners, but does not isolate the cause of its regression.
No memory/scaling campaign, conventional CFD comparator, GPU or MPI was tested.

## Repeat the measurements

The [harness](../../tools/benchmarks/repeated_execution/run.py) preserves the
measured Run boundary and inputs, with explicit output/installation metadata.
Its authoring adapters distinguish the immutable historical a7 API from the
current candidate API; these are benchmark fixtures, not product compatibility
shims. The historical measurements used a scratch driver; its hashes remain in
the data. The repository harness has not retroactively produced those samples.

Install each wheel in a separate CPython 3.13.14 environment with NumPy 2.5.2 and
Gmsh 4.15.2. Build the exact candidate revision with `maturin build --release
--locked` using the toolchain above. Keep each wheel for its digest. Put the
environment's `bin` directory on PATH so Gmsh can run, and use a home-backed
output directory outside the checkout. On the declared Linux host, for each
installation, replace the capitalized arguments with that installation's paths
and source revision:

```bash
taskset -c 16 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  PYTHON tools/benchmarks/repeated_execution/run.py OUTPUT \
  --edition EDITION --revision REVISION --wheel WHEEL --workload cylinder

taskset -c 16 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  PYTHON tools/benchmarks/repeated_execution/run.py OUTPUT \
  --edition EDITION --revision REVISION --wheel WHEEL --workload decay --calls-per-sample 100
```

`EDITION` is `a7` or `candidate`. For candidate-only separate profiling, add
`--profile`; do not mix its timings into unprofiled comparisons. The harness
records the supplied wheel hash and declared source revision; the operator
must install that wheel and bind the build to that revision. It retains Plan,
State, first Result, per-call times and environment metadata. Repeating on a
different environment establishes a new baseline, not a reproduction of these
particular timing values. Further deterministic lifecycle tests and a stable
regression authority remain work owned by #727.
