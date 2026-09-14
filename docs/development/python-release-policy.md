# Python distribution policy

This policy governs Python distribution artifacts. It does not widen Eqiora's
scientific or numerical claims, and a successful package upload is not
mathematical verification.

## Release identity

The Cargo workspace package version is the sole authored Eqiora version. The
public Rust `eqiora::VERSION`, private native-module version, public Python
`eqiora.__version__`, sdist, wheel metadata, and installed distribution all
derive from it. Cargo and Python use the same canonical stable SemVer identity;
prerelease and build-metadata forms reject. Python metadata must not repeat an
authored version. Publication requires the exact annotated tag derived by
prefixing `v` to the sole Cargo workspace version. A candidate records that tag, the full source
commit, and every source and wheel artifact by filename and SHA-256, and rejects
identity drift before publication.

The first wheel family is deliberately narrow:

- ordinary-GIL CPython 3.11, 3.12, 3.13, and 3.14;
- Linux x86-64 with a `manylinux_2_17` compatibility floor;
- one per-CPython wheel, because the NumPy C API is not an `abi3` claim.

macOS, Windows, other architectures, musllinux, free-threaded CPython, `abi3`,
GPU runtimes, bundled MPI, Conda, and system packages graduate independently
after installed-artifact evidence.

## Candidate construction and acceptance

A candidate is accepted only from a clean source commit. The release gate:

1. builds the source distribution;
2. rebuilds every wheel from that source distribution, not the checkout;
3. checks the declared wheel tags and metadata;
4. installs each wheel into an isolated environment outside the source tree;
5. runs exact Gmsh 4.15.2 meshing plus the base, NumPy ownership,
   async/cancellation, typing, PyTorch, JAX, and Matplotlib profiles within
   their declared boundaries;
6. replays the public base quick start on every wheel and the public framework
   quick starts on the exact framework interpreter before upload;
7. verifies NumPy 2.1.0 separately on CPython 3.12 while retaining the ordinary
   latest-resolution profiles;
8. records source identity, artifact hashes, build-tool versions, the observed
   NumPy floor, and passing profiles in the candidate manifest.

Artifact construction, wheel inspection, interpreter resolution, and shared
input hashing form one barrier. The immutable artifact family is then consumed
by isolated validation profiles under the verification lane's CPU and memory
budget. Each profile owns its environment, consumer tree, temporary directory,
and log. Heavy framework profiles do not overlap one another, but may overlap a
fitting light profile. All admitted work is joined; receipts and diagnostics are
merged in the frozen logical profile order, and the shared artifact family and
extracted source are re-hashed before the manifest is written. A profile may
therefore neither publish partial success nor mutate another profile's input.

Candidate and aggregate-gate scratch space is rooted below the invoking user's
home directory, including when the ambient system temporary directory points
at `/tmp`. The candidate admits a bounded internal 2-CPU, 4096-MiB profile
sub-budget; explicit candidate-specific environment settings may narrow or
raise it when the enclosing execution environment has been provisioned to
match.

The standard `release-tools` dependency group in `pyproject.toml` owns the exact
reviewed `uv` and Twine versions, and Dependabot's `uv` ecosystem proposes
upgrades to that source. Candidate execution neither trusts nor rejects an
ambient `uv`: it installs the declared binary wheel once below
`~/.cache/eqiora/tools/uv/`, validates the resulting executable, and reuses that
versioned cache entry. Updating either reviewed tool is an ordinary dependency
change followed by the complete distribution gate; a floating `latest` never
enters candidate identity.

The release-candidate workflow builds this complete candidate once for one source
commit and platform. Explicit registered-evidence runs can exercise the same
builder, but ordinary product-development CI does not build this full family.
The distribution, typing, PyTorch, JAX, and Matplotlib
validations then require their own closed check groups from that same manifest
and wheel family. The distinct evidence cases share one exact aggregate target,
so the verification runner may execute the target once while retaining a
report for each case. The focused adapter scripts remain developer diagnostics;
they are not a second candidate identity and may rebuild during standalone use.

Current candidates use `eqiora.python-distribution-candidate/v4`. The manifest
binds the complete sdist/four-wheel family and every required installed-wheel
profile without carrying a UI-host runtime, browser identity, or detached host
receipt. It is the only accepted manifest format; superseded pre-1.0 readers,
receipts, and transport paths are not retained.

Build into an empty directory outside the source tree:

```bash
python3 tools/release/python_candidate.py --out <candidate-directory>
```

The development-only `--skip-extras` mode may shorten iteration, but its
manifest is not eligible for upload. Add `--require-tag` at the publication
gate.

TestPyPI is the staging boundary. Upload uses a dedicated trusted-publishing
identity with the minimum OIDC permission and a protected release environment.
Hosted automation is transport for that upload, never the merge-quality gate.
The exact TestPyPI artifacts must be installed by version in clean
environments and replay the documented quick starts before promotion. A
TestPyPI upload, tag selection, or production PyPI release requires maintainer
release authority; ordinary feature work does not imply it.

## Dependencies and extras

NumPy is the only mandatory Python runtime dependency. Exact `gmsh==4.15.2` is
the optional `gmsh` extra and is launched only by the admitted automatic
meshing path; separating it preserves the base package's `manylinux_2_17`
floor. The Linux Gmsh wheel also requires `libGLU.so.1`. PyTorch, JAX,
Matplotlib, and exact `anywidget==0.11.0` remain optional extras, and importing
the base package must neither require nor import any of them. The wheel carries
the viewer JavaScript and CSS; the candidate does not install a separate
notebook host or browser runtime.
The first Matplotlib adapter uses exact release 3.11.1 with the headless Agg
backend. Its registered adapter profile runs on ordinary-GIL CPython 3.13;
the other wheel interpreters are not Matplotlib adapter compatibility
evidence. A wider compatibility range requires separate boundary evidence.

Dependency ranges are compatibility claims. Widening one requires an installed
wheel test at the new boundary; changing an exact framework baseline requires
its registered adapter evidence. The first candidate proves the NumPy 2.1.0
floor in one compatible installed-wheel environment; it does not claim every
NumPy version by CPython-version combination has been tested.

## Pre-1.0 API convergence

Before 1.0, public Python authoring names and behavior carry no backward-
compatibility or deprecation-period promise between releases. When a cleaner
final surface replaces an API, migrate every repository-owned caller, test,
example, and document atomically and remove the old name and path. Do not keep
an alias, overload, wrapper, warning shim, or parallel lifecycle solely for
compatibility.

Released wheels remain immutable. A breaking correction ships under a new
version and is stated plainly in its release notes; an existing wheel never
changes in place. [RFC 0083](../../rfcs/0083-current-model-artifact-epoch.md)
remains the authority for the bounded Model and Model Transaction compatibility
epoch reset and its exact rejection behavior. This authoring-API policy does
not reinterpret an old identifier, silently migrate persisted bytes, or
authorize unrelated wire changes.

Persisted Eqiora artifact codecs and versioned control contracts follow their
own compatibility rules. A Python package version never authorizes silent
migration, wire sniffing, or reinterpretation of those bytes.

## Yanking and security

Yank a Python release when users should not select it for new installations but
existing exact pins may still need resolution. Grounds include:

- a security vulnerability;
- data corruption or scientifically incorrect accepted output;
- a broken or materially mis-tagged wheel;
- metadata that installs an incompatible dependency set.

Publish a corrected version instead of replacing files under an existing
version. Use the private reporting path in
[`SECURITY.md`](../../SECURITY.md) for suspected vulnerabilities. Deprecation
alone is not a reason to yank, and yanking is not deletion or revocation.

Production PyPI promotion remains a separate public-release decision.
