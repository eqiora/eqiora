# Contributing to Eqiora

Thank you for contributing to Eqiora. Contributions are accepted under
Apache-2.0 and must be signed off under the Developer Certificate of Origin.

## Before starting

Read [`AGENTS.md`](AGENTS.md), the
[architecture summary](docs/architecture.md), and the
[glossary](docs/glossary.md). Read the
[high-risk and parallel development guide](docs/development/vertical-slice-development.md) only for
high-risk capability work or explicitly parallel writes.

- Ordinary, localized work goes straight into one small pull request: one
  author, the narrowest repository check that covers the change, one
  self-review, and done.
- Changes to scientific semantics, public or versioned compatibility, persisted
  or wire formats, security or trust policy, governance, or architecture limits
  need the relevant RFC or a review focused on that risk. The author may do that
  review and approve and merge the change; a second person is not required. A
  change is not high risk just because of its filename or a possible future use.
- Under [RFC 0089](rfcs/0089-resume-claim-local-evidence-development.md), registered
  evidence must be derived independently of the implementation and test the
  specific claim. Check the relevant values rather than pinning whole files or
  inventories; compare exact bytes only when byte identity is what you are claiming.
- Before 1.0, APIs and schemas should converge on one coherent design rather than
  keep old callers working. Update every consumer and piece of evidence in the
  repository in the same change, and delete the old names, schemas, aliases,
  overloads, decoders, and deprecation shims. Compatibility is kept only where
  there is an explicit stable or external interoperability promise.
- Do not add a Semantic Kernel node for UI or adapter convenience. First show
  why a typed named subgraph cannot express the concept.
- Open an Issue only when alternatives need coordination, the work has durable
  dependencies, or it cannot close in one pull request. Reuse an existing Issue
  when it owns the same outcome.

You may optionally record which AI agent produced an implementation, as
described in [RFC 0068](rfcs/0068-optional-implementation-agent-attestations.md).
This does not replace review, DCO sign-off, verification, or registered
evidence, and leaving it out never blocks a contribution.

## Development

Development uses [mise](https://mise.jdx.dev/), which installs the repository's
Rust, Python, Node, npm, and uv versions alongside the usual lockfiles and checks:

```bash
mise install
mise run setup
# Then run the narrowest repository-owned check for the changed surface.
# Use mise run fast only when no narrower task covers the delta.
# Use mise run affected for high-risk or uncertain integration closure.
```

To include a registered verification case, pass `--case <case-id>` after `--`
to either task. `mise tasks` lists the Studio and full-compatibility tasks.
Large verification scratch files go under `~/.cache/eqiora`, never the system
`/tmp`.

Rust development uses the latest stable toolchain with `rustfmt` and `clippy`;
public crates must also build with the minimum supported Rust version in
`Cargo.toml`. The [verification topology](docs/development/ci-topology.md) and
[local-verification guide](docs/development/local-verification.md) describe the
affected and complete gates.

The full Rust command set is below, for diagnosing failures or when you cannot
tell what a change affects. It does not replace the mise tasks and is not needed
for every change:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets
cargo xtask check-layers
cargo xtask check-facade
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
cargo deny check
```

Use a focused task while iterating. Run `affected` or the complete gate only
when you are unsure what a change affects, or when it touches releases,
compatibility, trust, or another high-risk area. Commands for Python, Studio,
physical GPU/MPI, and installed artifacts are documented next to those
components.

Tests should show the invariant, how it fails, and, where relevant, that
transactions roll back. Public errors use stable diagnostic codes rather than
bare strings.

A pull request that changes a capability must update its row in the
[capability matrix](docs/capability-matrix.md). Contract, execution,
verification, and maturity are rated separately: working code gets a
verification mark only when a reproducible `verify/` case supports exactly that
claim. Keep the stated limitations, and do not generalize from a single test
case.

## Architecture rules

- Dependencies point from higher layers to lower layers.
- `eqiora-core` depends only on external crates.
- Standard Ontology schemas may depend on core but never alter kernel meaning.
- The reference interpreter remains small and unoptimized.
- Unsafe Rust is denied by default and must be isolated behind a reviewed
  boundary with a documented safety invariant.
- Generated files are committed, and verification detects schema drift.

## Developer Certificate of Origin

Every commit must be signed off:

```bash
git commit -s -m "Describe the change"
```

The sign-off certifies the
[Developer Certificate of Origin 1.1](https://developercertificate.org/).
Use your real name and an email address you are entitled to use. Maintainers do
not merge unsigned commits.

## Pull requests

- Keep commits reviewable, and leave unrelated formatting, generated trees, and
  refactors out of the pull request.
- Link an Issue or RFC if one was needed; do not create one just to fill in a
  template.
- For a capability change, state what it does, what it does not cover, and how
  it was verified. For maintenance, state the behavior that changed and the check
  you ran.
- Mention any environment-dependent results you could not reproduce.
- Review is based on technical evidence; nobody's position replaces tests or
  conformance evidence.
