# Reference strategy

The reference is structural and operational rather than a committed wheel
hash. Candidate hashes necessarily bind the exact source commit being tested,
so the gate generates and validates them within the run.

Acceptance is anchored to standard package contracts: Cargo-derived version,
PEP 621/639 metadata, PEP 561 typed-package contents, CPython stable-ABI wheel tags (PEP 384/425),
the manylinux compatibility floor, and isolated installed-artifact behavior.
The source distribution is the sole wheel input.
