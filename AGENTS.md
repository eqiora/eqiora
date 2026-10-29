# Eqiora: instructions for agents

Eqiora lets people write a physics model as textbook equations, choose how to solve it, run it
fast on CPU threads, MPI or GPU, and differentiate the result. Rust core, Python first. This file
is a router: direction, hard rules and what enforces each, the work loop, then what to read for
your task. Humans start at CONTRIBUTING.md.

## Direction

- What to build first, and what is deferred: [roadmap direction](docs/roadmap.md#direction). Rank
  work by the user outcome it unblocks there. Install, first run, standard library, method choice,
  speed and external comparison come before new physics areas.
- Build the general mechanism, not a special case, and ship the API features users expect from
  comparable tools: defaults with explicit overrides, lookup by source name, progress and
  cancellation, export, clear errors. Generalize at the existing owner as soon as a second real
  consumer appears.
- A spatial capability names its CPU-parallel, MPI and GPU path, or records the gap in the
  [capability matrix](docs/capability-matrix.md).
- Examples, gallery, README and Get started import standard packages; tutorials also show the
  hand-written form: [examples/README.md](examples/README.md#writing-examples-and-tutorials).

## Hard rules (what enforces each)

- One model, one path: Python, Rust, CLI, Studio and the language server call the same
  compile → resolve → run core. No client keeps private semantics. (review)
- Units and dimensions are checked, never stripped or guessed. (compiler diagnostics)
- An unsupported combination returns a typed diagnostic. Non-test code has no panic on user
  input, no silent approximation, no TODO or partial path. (focused tests, review; add a lint
  when it recurs)
- Pre-1.0 has one coherent surface: migrate every repository consumer in the same PR and delete
  the old name, alias, shim or parallel path. (review)
- Expected values, tolerances and falsifiers come from analytic, manufactured or published
  references, or from running an established open-source tool. Never copy Eqiora's own output
  into an expectation. (`cargo run -p eqiora-verify -- index`, review)
- A change that adds, removes or narrows a user-visible capability updates its area file under
  `docs/capabilities/` (indexed by the [capability matrix](docs/capability-matrix.md)) in the
  same PR. (`tools/ci/check_docs.py`)
- Dependencies and licences pass `deny.toml`. (cargo-deny)
- CI and trust paths (`.github/`, `tools/ci/`, `crates/eqiora-verify/`, `deny.toml`, `CODEOWNERS`)
  change only with the trust review in [CI topology](docs/development/ci-topology.md).
  (`tools/ci/check_trust_boundary.py`)
- Build and worktree scratch goes under home-backed `TMPDIR`, never OS `/tmp`.
- A visible change (plot, gallery, viewer, docs page) is shown with an image you produced.

## Work loop

1. Look at open PRs first. Merge any ready predecessor; claim new work with a draft PR, one
   topic per PR.
2. Classify the actual delta. High risk is only: governance, review or evidence policy;
   scientific meaning, oracles, expected values or tolerances; public or versioned API and
   migrations; persisted schemas or exact artifacts; security, data integrity, release or CI
   trust; architecture ceilings or debt entries.
3. Low risk: smallest change, narrowest check, one self-review, merge. High risk: one explicit
   risk-focused review and the relevant hosted checks
   ([high-risk work](docs/development/vertical-slice-development.md)).
4. Check with `mise run pr` while iterating, `mise run fast` when nothing narrower covers the
   change, and `mise run affected` for high-risk or uncertain dependency closure. Pass affected
   cases with `--case` ([local verification](docs/development/local-verification.md)).
5. Own the PR through merge. Its description says what changed, what was checked and what was
   not. Delete merged branches.
6. If two focused attempts fail, write down what failed in the issue and change approach.
   Do not keep rerunning tests without progress.

## Parallel and long-running work

- Parallelize only when asked or when independent work shortens the critical path. Parallel
  writers need disjoint paths, their own worktree and their own Cargo target directory. One
  writer owns a shared seam.
- Planners own scope and write no code. Workers take one task to merge and return a short
  handoff; workers do not coordinate with each other.
- Maintenance roles may run on a schedule, each through normal small PRs: merge duplicate
  owners, refresh stale docs and matrix rows, move examples to standard packages, measure
  performance against the recorded baselines.
- Tool and test output stays terse: one grep-able line per failure, details in a file.

## Writing docs

- Write for the reader who acts next: plain words, short sentences, what works and what does
  not. Avoid internal jargon in user-facing text (admit, exact, bounded, authority, seam,
  claim-local).
- State a fact once, in its owner, and link to it. Do not copy paths, hashes, manifests or issue
  relations into prose.
- Decisions live in the repository (`docs/`, `rfcs/`), not only in chat or issues.
- Keep instruction files short: this file under 120 lines; a capability-matrix row under three
  lines, with detail in the linked `verify/` case README.

## Working on X: read first

| Working on | Read |
|---|---|
| Product priorities, deferred areas | [docs/roadmap.md](docs/roadmap.md) |
| Why the process looks like this | [AI-authored platform strategy](docs/development/ai-authored-platform-strategy.md) |
| Language syntax and semantics | [docs/language/](docs/language/core.md) |
| Standard packages | [packages README](crates/eqiora-api/packages/README.md) |
| Examples, gallery, tutorials | [examples/README.md](examples/README.md) |
| CPU, MPI, GPU backends and libraries | [library and accelerator strategy](docs/development/library-and-accelerator-strategy.md) |
| Architecture and crate boundaries | [docs/architecture.md](docs/architecture.md) |
| Evidence and oracles | [RFC 0089](rfcs/0089-resume-claim-local-evidence-development.md) |
| Python releases and wheels | [Python release policy](docs/development/python-release-policy.md) |
| CI, required checks, bypass | [CI topology](docs/development/ci-topology.md) |
| Viewer | [shared semantic viewer](docs/development/shared-semantic-viewer.md) |
| Terms | [glossary](docs/glossary.md) |

Add a rule only after a recurring failure shows that code, a test or a narrower document cannot
carry it, and delete the rule it replaces in the same change.
