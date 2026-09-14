# Local verification

Repository-owned checks establish local technical evidence. Hosted checks protect the submitted
commit, provider identity, and CI/release trust; they do not replace scientific evidence.

Registered evidence is independently derived and claim-local. Add or change it only when a durable
claim needs a stronger falsifier than an ordinary focused product test. Never tune expectations or
tolerances to observed implementation output.

## Choose the smallest check

Run the cheapest repository-owned check that can expose a plausible defect in the actual delta.
Use the focused command named beside the changed surface when one exists.

```bash
# Iteration loop: changed packages and named cases only, minutes not tens of minutes.
mise run pr -- --case packages.hierarchical-physical-boundary

# Fallback for a localized change with no narrower repository-owned task.
mise run fast

# Add every semantically affected registered case explicitly.
mise run fast -- --case packages.hierarchical-physical-boundary

# High-risk integration or genuinely uncertain reverse-dependency closure.
mise run affected -- --case packages.hierarchical-physical-boundary

# Inspect the deterministic plan without executing it.
mise run plan

# Full current-machine compatibility surface for release or unknown closure.
mise run periodic
```

`mise install` provisions declared tools. `mise run setup` owns locked setup and is a
dependency of executable mise gates. Use mise rather than a hand-assembled equivalent; packaging,
toolchain, interpreter, and CI-contract behavior are part of the repository check.

A focused failure permits a focused correction and rerun. Escalate to `fast`, `affected`,
`periodic`, another semantic case, or hosted waiting only when a named anomaly or risk could
make the wider result change the decision. Do not run a broad gate, clean build, or repeated
matrix for reassurance alone.

Do not reuse an old result merely because a patch ID or rebase is unchanged. Reuse is safe only
when the complete candidate bytes and modes are identical and intervening changes cannot affect
the gate, toolchain, inputs, authority, environment, dependency closure, or claim. Running the
focused check is usually cheaper than proving all of that.

## Planner behavior

The planner includes committed changes since merge base, staged and unstaged changes, and
untracked paths. Unknown paths and malformed manifests fail closed. `mise run plan` prints
selected paths, packages, cases, commands, resources, and limitations without running them.

The planner may overlap independent commands within declared CPU, memory, GPU, and named-lock
budgets. This scheduling is an implementation detail, not evidence. A missing tool, unsupported
environment, failed command, or unavailable resource remains an explicit failure or limitation.

Each nonempty plan of at most 32 commands owns one of two home-backed log slots. Raw logs are
limited to 16 MiB per command, 512 MiB per run, and 1 GiB across both slots. Overflow is an
explicit incomplete failure. Success removes its slot; failure retains the announced path for
separately authorized forensic copying and cleanup.

Persistent build roots live under `~/.cache/eqiora/local-verify/<checkout>/`; per-invocation
temporary directories use `~/.cache/eqiora/local-verify-tmp` and are removed after reporting.
`--scratch-root` relocates both to an equivalent home-backed location. Do not use OS `/tmp`
for large verification artifacts.

A new directory under `verify/` must own one durable, bounded claim and an independently derived
oracle or falsifier. Every semantically affected existing or new case must be passed with `--case`;
a shared executor package does not imply semantic ownership of all its cases.

## Gate tiers

`pr` is the iteration loop while a change is being written: formatting, default-target tests
and Clippy for directly changed packages, and explicitly named cases only. It defers
documentation, release-tree, dependency-layer, facade, CI-contract, and surface checks to
hosted pull-request CI or a `fast`/`affected` run, and states that deferral as a limitation.
It never substitutes for the tier a high-risk delta or release requires.

`fast` selects formatting, direct changed-package tests and Clippy, explicitly named
cases, and relevant lightweight documentation, site-source, dependency, or CI-contract checks. It is
the broad fallback for an ordinary localized edit, not a mandatory first step when a narrower
owned command exists.

Site changes select `python3 tools/site/check_site.py source --root .` in `fast`,
`affected`, and `periodic`. This reuses the source checks without building the site;
rendered output and browser checks remain in hosted Pages CI.

`affected` adds conservative Cargo reverse dependencies, Rustdoc, affected clients, registered
case inventory, and selected Python, Studio, dependency-policy, or isolated-experiment checks.
Use it for a high-risk integration boundary, an anomaly, or uncertainty that focused checks
cannot resolve.

`periodic` runs the full current-machine workspace, registered cases, MSRV, dependency policy,
Python, Studio, and isolated experiments. Use it for a release, an unknown affected closure, or
a concrete process investigation. It is not a calendar task.

Default tiers do not prove optional MPI, CUDA, Diffsol, browser, or other environment-specific
claims. Run the matching case or documented environment command when the delta changes one of
those claims.

The planner applies the hosted Cargo test profile to its commands. A manual `cargo test` or
`eqiora-verify` invocation outside the planner may measure a different optimization profile.
State the exact environment for any timing; an aborted run is not a sample.

## Evidence-package behavior

Treat evidence as authority over its explicit claim, not as immutable repository structure.
Investigate disagreements. Change a claim, oracle, falsifier, expected output, or tolerance only
when an independent derivation and risk-focused rationale justify that semantic change; never
copy observed output into the expectation. Migrate obsolete evidence atomically with the product
invariant it observes.

An oracle or falsifier package first proves one ordinary positive end-to-end path. Negative
probes then reach and name the intended gate. If an earlier unrelated denial makes the
capability unusable, the rejection is vacuous and the case fails.

Resource evidence uses raw input limits and deterministic implementation-independent work
bounds. Live allocation, queue depth, worklist lifetime, or scheduler behavior is not an oracle
unless residency or scheduling is the public claim.

When an oracle becomes more complex than the product seam or needs unrelated OS trust, simplify
the claim or separate authority. Never convert a mismatch or unavailable environment into a
pass.

## Heavy scientific candidates

Define a candidate campaign only for a real scientific or gallery claim whose production-scale
acceptance cannot be established in ordinary pull-request conformance.

Production gallery solves, refinement campaigns, and complete media encodes are outside
ordinary pull-request gates:

1. Pull-request conformance uses bounded data to exercise the ordinary path.
2. An exact-head scientific candidate runs the fixed campaign in its declared trusted
   environment with the claim, oracle, tolerances, stop conditions, and affected cases fixed.
3. Publication consumes digest-verified accepted results; it does not rerun the solve.

A candidate binds the exact Model, Geometry, mesh family, correspondence, Realization, Run,
fields/results, source revision, producer/runtime environment, solver/library identities,
oracle/evidence IDs, output inventory, and content digests. Scientific or solver changes need a
new affected candidate. Renderer or site-shell changes may reuse an unchanged accepted result.

`not-selected`, a cache hit, a stale bundle, or successful bounded conformance is not full
scientific acceptance. If the required environment is unavailable, record the limitation and
narrow the claim or stop.

## Known local limitations

- One Python interpreter is not the supported-version matrix.
- One-host MPI is not physical multi-node evidence.
- A build without a matching GPU and driver is not physical CUDA evidence.
- Studio browser checks require documented browser and native system dependencies. When the
  local Chrome executable is absent, the planner omits the Studio interaction tests, records
  the deferral to the hosted Studio lane as an explicit limitation, and completes; the run
  never reports a local Studio interaction pass. Every other Studio command, and any failure
  with Chrome present, remains a failure.
- Run `tools/ci/python_package_gate.py` for installed Python evidence. A hand-written
  `uv run --with .` may reuse a stale cached wheel.

## Optional implementation-agent provenance

An implementation-agent identifier is optional provenance, not correctness evidence. Only when
a final pull-request body supplies one, run:

```bash
python3 tools/ci/check_implementation_agent.py \
  --base origin/main \
  --pr-body-file /path/to/pr-body.md
```

The checker accepts omission, empty input, or `not-supplied`. Otherwise it resolves the
identifier only against the protected-base registry and verifies score and expiry. See
[RFC 0068](../../rfcs/0068-optional-implementation-agent-attestations.md).

## Hosted integration

Changes to scientific meaning or an oracle, public/versioned compatibility, persisted schemas
or exact artifacts, security/data integrity, release/CI trust, governance, or architecture
ceilings wait for every relevant hosted check before merge.

The exact localized low-risk classes and auditable owner/admin bypass are defined by
[the public verification topology](ci-topology.md). Before such a bypass, the exact head still
runs its focused mise gate, scope/DCO audit, and recorded risk-focused self-review. Record the
actor, reason, head, commands, results, and post-merge signal. A high-risk bypass is limited to
a named self-approval deadlock in the protected check itself; all other relevant contexts still
complete before merge.

Record exact commands and limitations in the pull request or Issue. If a check fails, correct
the same slice and rerun the exposing check. Do not create a standing report or another
acceptance implementation.
