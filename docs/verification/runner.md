# Verification runner

`eqiora-verify` builds one deterministic case index from the contracts under
`verify/`. Discovery, contract validation, evidence execution, and rendering are
separate phases; `list` and `check` never run numerical evidence.

The repository is selected at runtime from the current working directory.
Use the global `--root /path/to/repository` option to select another checkout;
relative paths are resolved from the invocation directory. A missing or invalid
root fails rather than falling back to the checkout that built the executable.
This keeps a shared Cargo cache or copied executable from selecting stale evidence.

The manifests and the README next to each case are the source of truth for
current claims and non-claims; project summaries link here instead of copying a
capability list. A case enters the index only through a validated
`verify/<area>/<case>/case.toml`.

## Commands

```bash
# Stable ID order, with no evidence execution.
cargo run -p eqiora-verify -- list

# Manifest-derived capability-to-evidence index (all cases are validated first).
cargo run -p eqiora-verify -- index
cargo run -p eqiora-verify -- index --capability convergence

# Validate manifests, declared artifacts, Cargo targets, and Python gate paths.
cargo run -p eqiora-verify -- check
cargo run -p eqiora-verify -- check --case solid.axial-bar

# Run all or any exact set of cases. Fail-fast is the default.
cargo run -p eqiora-verify -- run
cargo run -p eqiora-verify -- run --jobs 1
cargo run -p eqiora-verify -- run --case numerics.poisson-fem-fvm
cargo run -p eqiora-verify -- run --case numerics.cartesian-poisson-fem-fvm
cargo run -p eqiora-verify -- run --case numerics.cartesian-poisson-3d-fem-fvm
cargo run -p eqiora-verify -- run --case time.general-implicit-dae
cargo run -p eqiora-verify -- run \
  --case numerics.poisson-fem-fvm \
  --case time.general-implicit-dae
cargo run -p eqiora-verify -- run --keep-going

# Validate the complete repository, then execute one environment/runner-kind
# intersection. Either filter may also be supplied alone.
cargo run -p eqiora-verify -- run --environment host-cpu --runner-kind cargo
cargo run -p eqiora-verify -- run --environment host-cpu --runner-kind python-installed-wheel
```

`case.toml` is also the only input to the capability-to-evidence index. For each
declared capability, `index` lists the case ID, manifest, maturity, reference
strategy, and structured evidence target. It never reads prose or a separate
hand-maintained registry. Entries are sorted by capability and case, and a
filtered query still validates the whole repository first. A case can also
declare `conformance_kits`: stable, deduplicated IDs consumed by two or more other
cases. The index shows those kit IDs with every capability entry, and unknown
manifest fields cannot pose as this typed field.

## JSON output

`--format json` is a global option and may come before or after the subcommand.
Standard output then contains exactly one `eqiora.verification-report/v6` object
for `list`, `check`, and `run`, or one `eqiora.capability-evidence-index/v3`
object for `index`. Captured child stdout and stderr go into the corresponding
case fields and are never mixed into the report:

```bash
cargo run -q -p eqiora-verify -- run --format json > report.json
cargo run -q -p eqiora-verify -- index --format json > capability-index.json
```

`--case` can be repeated. The report's `selected_cases` array is sorted and
deduplicated; an empty array means the whole registry, duplicate selectors are
harmless, and every unknown ID is reported, in sorted order, before any evidence
runs. `selected_environment` and `selected_runner_kind` record the optional
filters; when both are given, a target must match both. Each evidence target that
started reports `duration_ms`, measured with a monotonic clock and shared by every
case that uses the target. The field is always present and is `null` when the
target did not start or did not run, like every other optional report field.
Human-readable output shows the same duration in whole milliseconds.

Report versions:

- v6 replaced the single `selected_case` with the `selected_cases` array;
- v5 added `selected_runner_kind` and the closed `cargo` /
  `python-installed-wheel` selector;
- v4 added `duration_ms`;
- v3 added the selected evidence environment and the closed `host-cpu` /
  `physical-mpi-cuda` distinction; host-CPU targets kept their earlier JSON shape.

## Exit status and scheduling

The process succeeds only when repository validation succeeds, the case filter
resolves, and every selected evidence target passes. Environment and runner-kind
filters still validate every manifest, Cargo target, and artifact first. A target
excluded by a filter is reported as `not-selected`, with a message naming the
environment or runner kind it needs, and an empty selection succeeds after
validation. Without filters every executable target is selected, so a missing
physical runtime fails instead of being skipped silently.

`--jobs` defaults to the host's available parallelism. With more than one job,
workers take targets from a shared queue, so a free slot starts the next target
immediately. After the first test failure the runner launches nothing new, lets
the targets already running (at most `jobs - 1`) finish, and reports their
results; it never kills a running child. In fail-fast mode with `--jobs 1`, the
remaining selected cases are reported as `skipped`, as serial runs always were;
a higher `--jobs` value can therefore turn a case that would have been `skipped`
into a real `passed` or `failed`.

Cases with status `proposed` or `specified` are reported as `not-runnable`;
`implemented`, `verified`, and `validated` cases must declare an evidence target
and pass it.

## Evidence targets

A manifest cannot supply a shell fragment, working directory, or free-form
arguments. Its `[evidence]` table takes one of three fixed forms.

A workspace Cargo integration test:

```toml
[evidence]
package = "eqiora-numerics"
test = "poisson_fem_fvm"
table = "expected/convergence.csv" # optional
```

Evidence that can only run on a particular physical topology declares that
environment explicitly:

```toml
[evidence]
package = "eqiora"
test = "fixed_reference_fsi_distributed_cuda_solve_mpi_2d"
features = ["mpi-cuda"]
environment = "physical-mpi-cuda"
```

The default is `host-cpu`. Environments are a closed enum, not arbitrary runner
labels or host capabilities.

A single crate-private library unit test can be selected without exporting a
test hook or duplicating `lib.rs` as a Cargo target:

```toml
[evidence]
runner = "cargo-library-test"
package = "eqiora-numerics"
test = "form_compiler::elasticity::oracle::registered_evidence"
```

The test is one complete lowercase Rust test name with at least two
`::`-separated segments, never a prefix, glob, source path, or argument list. The
runner checks that this exact test exists before running it. Host evidence must be
a non-ignored test and physical MPI/CUDA evidence an ignored one, matching the
manifest's environment.

A repository Python gate run against the installed wheel:

```toml
[evidence]
runner = "python-installed-wheel"
script = "tools/ci/python_torch_gate.py"
```

## How targets run

For Cargo, the runner first reads the workspace package, library, and
integration-test inventory from `cargo metadata --format-version=1 --no-deps`.
After validation, case selection, and filtering, it groups the selected Cargo
targets by target shape, package, and sorted feature set; a Python-only
runner-kind selection builds no Cargo group at all. Each integration-test group is
compiled once with `cargo test --locked -p <package> --no-run --message-format=json`,
one `--test` per target, and the group's `--features` value when it is non-empty.
Features from different manifests are never combined. The runner reads Cargo's
compiler-artifact messages and runs each test binary directly.

Each library-test group compiles once with `--lib --no-run`, keeps only the single
library test executable of the selected package, checks the libtest inventory,
and runs only the declared test with `--exact` (plus `--ignored` for physical
MPI/CUDA evidence). A missing, renamed, substring-only, wrongly ignored, absent, or
ambiguous library target is an error. Internally, library selectors carry a
reserved `lib::` prefix that cannot collide with integration-target names;
manifests, indexes, and reports show only the `cargo-library-test` form.

For Python, the runner invokes `PYTHON` (or `python3`) with the single declared
script, which must be a normalized, repository-relative regular `.py` file; no
arguments are accepted. Both runners work from the repository root and never use
a shell.

All Cargo groups are built before any test runs. A failed group build is reported,
with the group named, against every case in that group. It does not count as a
test failure for fail-fast purposes, so targets from groups that built still run.
A failing test binary is attributed only to the cases that select exactly its
execution key. The key contains every per-target process input: target shape,
package, integration-target or library-test name, normalized feature set, and
environment for Cargo; runner, script, and environment for Python. A case's
`table` and claim metadata are validated but are not part of the key, because
neither changes the process. Cases that share a key reuse its single result,
duration, and output streams while keeping separate case reports. This reuse
applies within one runner invocation only; nothing is cached between runs.

Cargo evidence for `physical-mpi-cuda` is ignored by ordinary Cargo test runs.
When that environment is selected, the runner passes `--ignored` directly to the
test binary, and the test still fails if the required physical topology is not
available. Ordinary compilation and feature coverage thus stay separate from
hardware claims without free-form manifest arguments.

Optional evidence artifacts must be normalized relative paths that resolve to
files inside the repository. Unknown runner identities or evidence keys,
duplicate capabilities, unknown statuses, missing artifacts/scripts,
duplicate case IDs, and paths whose ID does not match
`verify/<area>/<case>` are errors before any evidence starts.

## Determinism

Cases are ordered by their declared ID, independent of directory enumeration.
Target completion order never changes case order, result attribution, captured
streams, or the exit status. The report contains no timestamps; it does contain
each started target's monotonic elapsed duration. Child output and duration are
evidence data and may vary with toolchain and host load; consumers that compare
report bytes should normalize or drop those fields.
