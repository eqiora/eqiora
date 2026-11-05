# Bundled scalar Poisson and Laplace equations

The [Scalar package](../../../crates/eqiora-api/packages/Eqiora.Scalar/README.md)
provides fixed-unit, two-dimensional equations and boundary components through
both exact package resolution and ordinary bundled source imports. It borrows
the caller's dimensionless potential; it does not create auxiliary fields.

The Poisson test uses the analytic sine solution on successively refined Q1
meshes. Its convergence and conservation checks reuse the established
[Cartesian Poisson reference](../cartesian-poisson-fem-fvm/README.md). Both the
default coefficient and an explicit coefficient are exercised.

The mixed-boundary Laplace test has the exact solutions `u = x/(1 m)` and
`u = -x/(1 m)`. Their outward constitutive fluxes have opposite signs. Each is
a valid problem. An independently assembled integer Q1 stencil checks the
computed field; its residual and a discrete comparison bound certify the nodal
error. Positive bilinear basis functions transfer the bound to the whole field.
The [test owner](../../../crates/eqiora/tests/standard_scalar_package.rs) derives
the stencil, matrix scaling and outward-rounded floating-point allowance.
Nonfinite falsifiers start from a certified exact solution.

Typed controls reject incompatible units, ambient dimensions, foreign parents,
insufficient authored regularity for a normal gradient, and a flux coefficient
that does not match the volume law. Closed source compilation does not acquire
implicit package access.

This case runs on one host CPU with the reference solver. It does not establish
FVM, MPI, GPU or unstructured-mesh execution, performance, generic component
units, Robin conditions or periodic conditions. Those methods and capabilities
remain with their existing owners.
