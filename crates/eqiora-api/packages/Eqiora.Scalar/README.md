# Eqiora.Scalar

The development package `Eqiora.Scalar@0.1.0` supplies two-dimensional scalar
Poisson and Laplace equations for a dimensionless potential on physical
coordinates measured in metres. These components do not represent a generic
unit-polymorphic temperature, voltage, or pressure field.

Import `Eqiora.Scalar.scalar as scalar`. `Poisson2d` adds
`-div(coefficient * grad(potential)) = source`; `Laplace2d` adds the homogeneous
equation without requiring a source field. Both borrow the caller's potential
and body. The dimensionless coefficient defaults to one and can be overridden.
The Poisson source is a typed expression on the body with dimension `1 / m^2`.

`FixedValue2d` takes a dimensionless expression on an exact boundary of that
body. `NormalFlux2d` takes an expression with dimension `1 / m` and imposes
`normal(coefficient * grad(potential)) = flux`, using the parent's outward
normal. This is the normal component of the constitutive gradient, with a
positive sign. It is not the negative-gradient physical heat flux. Use the
same coefficient in the volume and flux components; incompatible constitutive
fluxes reject during numerical lowering. Zero normal flux is supplied as
`flux = 0[1/m]`, without a separate boundary component.

The caller owns the potential, domain, boundary selections and boundary
coverage. The `2d` suffix states a required ambient dimension; this release
does not infer it. Meshes, numerical methods, solvers and execution targets
remain separate choices. Existing numerical methods determine which expressions
and boundaries can be solved. This package introduces no new backend support,
Robin or periodic conditions, or nonlinear diffusion method.
