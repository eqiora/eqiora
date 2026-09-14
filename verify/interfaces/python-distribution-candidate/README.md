# Python distribution candidate

This case closes one Python artifact family without turning package publication
into mathematical evidence. From one clean source commit it builds a source
distribution, extracts it, and builds four ordinary-GIL Linux x86-64 wheels
only from that extracted source.

Every wheel is installed outside the checkout and runs the base, NumPy
ownership, synchronous/awaitable execution, cancellation, public-smoke, and
strict-typing checks. CPython 3.13 additionally runs the PyTorch 2.14.0,
JAX/JAXLIB 0.11.0, and Matplotlib 3.11.1 profiles. A separate CPython 3.12
environment proves the declared NumPy 2.1.0 floor.

The v4 manifest binds the source commit, optional tags, tool versions, artifact
filenames and SHA-256 values, wheel tags, dependency profile, and passing
checks. Candidate construction and profile execution share no notebook-host or
browser authority. Colab is an example and documentation surface, not a
release-candidate runtime.

Execution is a two-stage handoff at one exact revision. `prepare` writes only
one sdist and four wheels. `finalize` admits those bytes, runs disjoint
home-backed profiles, rechecks the family, and writes exactly one manifest.
Only the five distribution artifacts are publishable.

## Falsifiers

The gate rejects a dirty source tree, incomplete or extra family, wrong wheel
tag or metadata, checkout-resolved import, missing license/stub material,
profile failure, wrong NumPy floor, changed artifact byte, profile mutation of
the family, or publication metadata mixed into the distribution directory.

## Boundary

This case does not upload anything. TestPyPI and PyPI additionally require the
tagged release workflows and trusted-publishing authority. macOS, Windows,
free-threaded CPython, abi3, GPU wheels, bundled MPI, and reproducible-build
certification remain outside this claim.
