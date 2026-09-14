# Public verification and trust topology

Eqiora's ordinary development authority remains repository-owned verification
run on the exact source under review. GitHub Actions reproduces that contract
for untrusted public pull requests and protects release identity; it does not
replace targeted local or physical-backend verification.

## Pull-request boundary

Every pull request reports three independent required contexts:

1. **CI gate** runs the untrusted head revision with read-only repository
   permission and no project secrets. The repository-owned classifier selects
   the affected Rust, MSRV, dependency, Python, Studio, and isolated-experiment
   jobs. Unknown paths fail closed to the complete surface.
2. **CI definition trust** runs under `pull_request_target` from the exact
   protected base revision. It reads complete changed-file metadata and only
   the bounded inert blobs needed for the exact ratchet class below through the
   authenticated GitHub API; it never checks out, imports, or executes head
   code. Except for that narrow class, a change to workflow files, local
   CI/release tooling, the dependency policy, the layer/facade checks, the
   registered-evidence runner, custom Actions, or CODEOWNERS fails this
   context.
3. **Build and verify static documentation** checks out the ordinary untrusted
   pull-request revision and asks the same repository-owned classifier whether
   the complete produced-site input closure changed. A proven-irrelevant delta
   succeeds immediately with the explicit `unchanged input closure` reason;
   public API, generated-reference, content, asset, build, workflow, unknown,
   ambiguous, file-mode/type-changing, or classification-failure input performs
   the complete existing offline build. The unchanged decision records the
   validated current base revision as the content authority that an unchanged
   artifact would use; it does not pretend that the pull-request head identity
   is itself site content. Protected-main pushes and
   manual runs always build in full, and only a successful authenticated main
   full build may deploy.

The public code workflow starts on `opened`, `reopened`, and `synchronize`;
the documentation and base-owned trust workflows also start on `edited` so
retargeting a base reruns both base-sensitive decisions. Opening a
Draft therefore verifies its exact head immediately, while changing that same
head from Draft to Ready reuses the existing required contexts instead of
starting and cancelling an identical full run. A pushed head still starts a
new run, and the per-pull-request concurrency groups cancel only its stale
predecessor. Pages edits queue behind the running job instead of cancelling it.
Once queued work starts, the existing authenticator may reuse a successful full
run on the same head, but only when the complete site input snapshot is unchanged.
Missing or ambiguous prior success still requires a full build; base-owned trust
checks continue to run on edits. This relies on GitHub's provider-owned event contract delivering
those activities for Draft pull requests and retaining commit-bound check runs
across a readiness-only transition. Repository tests pin the workflow side of
that boundary; live Actions observation owns the provider side.

Changed-file pagination must match the provider-owned pull-request count.
Pull requests beyond the API's complete 3,000-file visibility boundary fail
closed instead of trusting a truncated list.

The protected-base trust classifier may approve a coupled architecture ratchet
when the only changed protected path is `tools/ci/architecture-debt.toml`.
Only canonical integer digits of existing `[[file_lines]]` and
`[[public_surface]]` ceilings may change, and each changed ceiling must strictly
decrease. Entry inventory, ordering, identities, metadata, comments, and all
other bytes remain unchanged. The two kinds of reduction may share one change.

For file-line reductions, protected-base code measures each bound-head source
at exactly the new ceiling. For public surfaces, each reduced crate must have
an authenticated added, modified, or removed Rust source path under that exact
crate directory; a rename alone is insufficient. The classifier does not run
head Rust or infer a public-item count. The ordinary architecture CI measures
the actual public surface and rejects a count above the ceiling (its current
exact-freeze rule also requires ratcheting unused allowance). Acceptance requires
both the protected-base `CI definition trust` check and the architecture check
within the required `CI gate` to pass on the candidate. A source edit alone is
not evidence that the declared public surface was reduced.

This is a successful required trust check, not an owner bypass: risk-focused
review, exact-head local verification, and every relevant hosted context remain
mandatory. Increases, added/deleted/reordered entries, altered metadata, public
ratchets without a matching Rust change, and mixed protected-path changes fail
closed. Limits, globs, and other protected definitions remain outside this exception.

The live main ruleset is active and strict, binds required `CI gate` and
`CI definition trust` contexts to the GitHub Actions provider, rejects deletion
and non-fast-forward updates, requires pull requests, and resolves review
threads. Its administrator enforcement is disabled (`enforce_admins=false`)
and it exposes the owner/admin pull-request bypass actor. Repository Actions
must remain immutable-SHA pinned. These are external live facts, not settings
this repository policy may silently mutate.

Normal integration and every high-risk delta wait for the relevant required
contexts. The trust guard intentionally cannot approve its own replacement, so
with one maintainer a legitimate trust-definition change retains its explicit
owner bootstrap bypass after exact-tree repository verification and recorded
risk-focused self-review. This is auditable bootstrap, not external review.

The same existing owner/admin bypass may be used when a required context cannot
approve a change to its own protected definition, and for localized low-risk
deltas such as a one-line non-protection agent-capacity setting, non-governance
documentation, reproducible mechanical output, a private behavior-preserving
refactor, or a localized correction. Before bypass, the exact head must pass its
mise gate, scope and DCO audit, and recorded risk-focused self-review; every
unaffected required context should be allowed to complete whenever possible.
Record the exact head, actor, reason, commands, and results in the pull request.
Continue already-running CI as a post-merge signal and immediately assess
repair or rollback on failure.

This route does not silently weaken, remove, or reconfigure required contexts
or the ruleset; it uses the live auditable bypass narrowly to resolve a named
self-approval deadlock. High-risk work still waits for every relevant context
that can evaluate it, and the pull request records the actor, exact head,
reason, commands, results, and any post-merge signal. External Code Owner
review may be added for a concrete trust decision, but it is not mandatory.

## Change ownership

[`tools/ci/classify_changes.py`](../../tools/ci/classify_changes.py) is the
shared fail-closed surface map. Documentation-only changes avoid heavy product
jobs; site artifact inputs select the full documentation build; changes to CI
definitions or unknown paths select every surface. The Pages workflow has no
parallel `on.paths` approximation.

Conditional jobs may be skipped only when their surface is irrelevant.
[`tools/ci/check_gate.py`](../../tools/ci/check_gate.py) validates the complete
relevance/result vocabulary and publishes the single `CI gate` context.

## Exact-commit reproduction

The same workflow retains an explicit manual mode for a release candidate or
CI-contract audit:

```bash
gh workflow run ci.yml --ref main -f commit=<full-main-head-sha>
# Verify an unmerged candidate in its own ref's cache scope:
gh workflow run ci.yml --ref <candidate-branch> -f commit=<full-candidate-head-sha>
```

Manual CI and release workflows require the selected full lowercase commit SHA
to equal the dispatch ref's frozen revision before executing selected source.
Use a ref pointing at the candidate (or historical commit), not `--ref main` with
an unrelated commit. This keeps candidate code outside the shared main cache
scope. Production additionally checks the annotated tag before importing release
code; promotion still requires the successful exact-main candidate run.
Manual CI selects the full compatibility matrix. A hosted run is evidence only
for the exact commit it reports. No schedule, activity ledger, or missed-run
monitor exists; add one only if a recurring measured failure requires it.

The separate `Windows compile probe` workflow is a manual portability
measurement outside the `CI gate` aggregation. It provisions the Microsoft MPI
SDK and runs the workspace-wide all-targets, all-features check without changing
pull-request runner consumption. Dispatch it from protected `main` only when its
result will choose a concrete portability slice; it has no schedule or standing
monitor.

## Release separation

Release workflows are small and separate from merge CI:

- candidate construction produces one source distribution, wheels, SBOM, and
  hash manifest from one clean source commit;
- TestPyPI publication uploads those exact bytes through OIDC in the protected
  `testpypi` environment;
- replay downloads the exact version, verifies every hash, and installs
  downloaded wheels while resolving dependencies from production PyPI;
- production promotion first authenticates a completed successful candidate
  workflow dispatched from protected `main` at the exact release commit, so a
  failed replay or unrelated workflow run cannot supply artifacts;
- publication accepts only that same verified artifact set, uses the protected
  `pypi` environment, and grants `id-token: write` only to its publish job.

Long-lived PyPI tokens are not repository secrets. Python production is never rebuilt
after TestPyPI acceptance.

The same [production workflow](../../.github/workflows/python-production-publish.yml)
also publishes Rust releases after the annotated tag, source commit, and accepted Python
candidate have been authenticated. Its filename remains unchanged for the existing PyPI
trusted-publisher identity. [The Rust driver](../../tools/release/rust_publish.py) derives the
facade's complete local dependency closure from Cargo metadata and uses the exact Linux Rust
toolchain in `mise.lock`. A job without a publication credential packages and builds the
default-feature archives. Both PyPI and crates.io publication wait for that archive build
to succeed, so a Rust packaging failure cannot start a partial public release. A fresh Rust
publication runner reconstructs the archives without running build scripts and checks their source identity and hashes against that run's retained artifacts.
Only its final publish step receives `CARGO_REGISTRY_TOKEN` from the protected `crates-io`
environment. This environment admits protected branches, matching the production dispatch
from `main`.

Rust uploads run in dependency order with Cargo's locked dependency resolution. The driver
checks every existing version before uploading, resumes only identical unyanked archives,
honors crates.io's explicit retry time, and checks registry checksums after each upload.
Use a new workspace version for each subsequent release: the initial Rust alpha is an
immutable historical publication. If Rust publication stops partway through, rerun the
failed jobs of the same production run. The two registries do not provide a shared atomic
publication operation. This source-package path covers Linux default-feature builds; it
does not establish optional native-backend or scientific claims.

## Cost and security boundaries

- Rust Clippy, tests, and rustdoc use the local planner's reverse-dependency
  closure for ordinary package-owned Rust source changes. Cargo definitions,
  shared fixtures, unknown paths, file-mode changes, or unavailable commit
  authority retain the whole workspace. Manual runs remain full; formatting,
  architecture, MSRV, and the other lanes keep their existing checks.
  Python adapter Rust changes also include the facade and its reverse dependencies:
  the facade's control-plane test reads adapter source outside Cargo dependencies.
- Pull-request jobs run only on GitHub-hosted ephemeral runners.
- No private developer, GPU, or HPC runner is attached to public Actions.
- Checkout credentials are not persisted.
- Fork pull requests receive no release environment or package credential.
- Stable uses a pinned dependency-only Rust cache. Only successful exact-main
  manual CI saves; PRs and candidate-ref runs restore without saving. The cache
  excludes workspace outputs, binaries, incremental data and reports. Compiler,
  runner/native environment and build-profile inputs isolate fallback keys;
  Cargo manifests, lockfiles and configuration determine dependency keys.
  The action's incremental override is reset for non-test builds; tests retain
  their existing non-incremental profile. Seed once through exact-main manual
  CI, then inspect an actual hosted restore and Cargo timings before claiming
  a speedup. No recurring main build exists solely to populate the cache.
- The quality job retains Cargo's HTML build-timing reports for seven days,
  including per-crate compilation and scheduling. Use reports from the same
  hosted profile to investigate build cost before changing optimization or caches.
- The hosted quality job owns formatting, linting, workspace tests, dependency
  layers, the public facade, and rustdoc. Ordinary CI does not separately rerun
  the complete registered Cargo suite or build the complete Python distribution
  candidate. Installed-wheel product tests remain in the Python version matrix;
  the release-candidate workflow retains the complete artifact-family validation
  before publication. CI definition trust and release trust checks are unchanged.
- Pages builds only the Rust executables and Rustdoc consumed by site generation.
  Its temporary CLI, MCP, and facade-check executables use the unoptimized development
  profile without debug information or incremental output: their consumers inspect
  interfaces, not numerical throughput. Release builds and Rustdoc are unchanged.
  It links to the existing case manifests instead of compiling the evidence index
  tool and regenerating a duplicate case inventory.
  Python API documentation is parsed from shipped type stubs without importing the
  package. Wheel construction and installed-package checks belong to the Python
  version matrix and release workflow, not to the documentation build.
- Main Pages runs queue instead of cancelling an in-progress publication. After the
  preceding run finishes, the next run compares its complete site input closure with
  the latest successfully deployed main commit. Unchanged inputs skip both build and
  deployment. Missing, failed, or unauthenticated publication state falls back to a
  full build; comparison never uses only the immediately preceding push. Manual
  dispatch remains a full build.
- PR metadata edits reuse the newest authenticated run that actually performed
  heavy work. Later lightweight successes do not hide that result, and witnesses
  from different runs are never combined.
  A completed CI run may have failed in another lane: reuse still requires each
  selected lane's whole job and named heavy steps to have completed successfully,
  with unchanged inputs. Pages still requires a successful publishing or build run.
- Host-CPU case manifests disconnected from mandatory CI are marked
  `implemented`, and their capability-matrix verification is absent while the
  evidence suite is being reduced. Existing cases remain explicitly runnable
  with `cargo run -p eqiora-verify -- run --case <id>` or `mise run periodic`.
  A passing product test is not an automatic promotion of registered evidence.
- Hosted test steps use Cargo's ordinary `test`
  profile with debug information disabled, incremental compilation disabled,
  and optimization level 1 because their target trees are disposable. Test
  builds of `eqiora-compiler` use the package override in `Cargo.toml` to avoid
  optimizing parsing; runtime artifact validation, numerical kernels, and release
  profiles retain their existing optimization. Debug
  assertions and overflow checks remain enabled, and no relaxed floating-point
  mode is used. `tools/ci/local_verify.py` applies that profile to every command
  it runs, so a local gate reproduces the hosted one without an operator
  remembering a prefix; `HOSTED_TEST_PROFILE` in that file is the single
  definition and a contract test fails if it stops matching the workflow. A
  `cargo test` or `eqiora-verify` command invoked by hand outside the planner
  still needs the variables set explicitly, and the difference is not cosmetic:
  the registered preconditioner-scaling case takes 1150.8 s at Cargo's default
  `opt-level = 0` and 64.5 s at the hosted `opt-level = 1`.
- Physical GPU/MPI evidence remains an explicit maintainer-run verification
  boundary. Running the unfiltered or exact physical case without its declared
  environment still fails closed.

## References

- [GitHub secure use reference](https://docs.github.com/en/actions/reference/security/secure-use)
- [Events that trigger workflows](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows)
- [Rules available in rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets)
- [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
