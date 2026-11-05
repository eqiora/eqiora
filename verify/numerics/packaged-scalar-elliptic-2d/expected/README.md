# Acceptance

The [registered test](../../../../crates/eqiora/tests/standard_scalar_package.rs)
owns the convergence, conservation and outward-rounded Laplace error checks.
It also rejects nonfinite values starting from a certified finite solution,
incompatible units and unsupported boundary bindings. The
[case overview](../README.md) states the supported execution boundary.
