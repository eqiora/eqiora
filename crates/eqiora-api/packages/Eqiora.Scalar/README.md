# Eqiora.Scalar

The development package `Eqiora.Scalar@0.5.0` supplies two-dimensional scalar
Poisson and Laplace equations for a dimensionless potential, and steady and
constant-capacity transient heat conduction for a temperature in kelvins. Coordinates are measured in metres.
Each component declares its physical units explicitly.

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

The volume components and `FixedValue2d` require at least authored `h1`
regularity. `NormalFlux2d` requires `smooth` for the normal gradient trace;
binding an `h1` potential rejects rather than strengthening it silently.

The caller owns the potential, domain, boundary selections and boundary
coverage. The `2d` suffix states a required ambient dimension; this release
does not infer it. Meshes, numerical methods, solvers and execution targets
remain separate choices. Existing numerical methods determine which expressions
and boundaries can be solved. This package introduces no new backend support,
Robin or periodic conditions, or nonlinear diffusion method.

## Steady heat conduction

`SteadyHeat2d` borrows the caller's `temperature: K` Field and body. Its physical
conservation Law is `flux -conductivity * grad(temperature); source heating;`:
the divergence of outward heat flux equals volumetric heating. Conductivity is
an explicit positive parameter in `W / (m * K)`, and `heating` is an expression
on the body in `W / m^3`. Supply `heating = 0[W/m^3]` for a source-free body.

`FixedTemperature2d` takes a `value` expression in `K` on a boundary of the
body. `OutwardHeatFlux2d` takes the same conductivity and a `flux` expression
in `W / m^2`, imposing `normal(-conductivity * grad(temperature)) = flux`.
Positive flux leaves the body; negative flux enters it. Supply
`flux = 0[W/m^2]` for insulation. In this physical Law, the numerical owner
checks that boundary conductivity matches the volume constitutive coefficient.

For example, with a caller-owned `temperature` and exact boundary `left`:

```eqiora
instance heat: scalar.SteadyHeat2d(
  body = body, temperature = temperature,
  conductivity = 2[W/(m*K)], heating = 0[W/m^3]
);
instance cold: scalar.FixedTemperature2d(
  body = body, face = left, temperature = temperature, value = 300[K]
);
```

The caller must cover the remaining boundaries. The volume and fixed-temperature
components require `h1`; the heat-flux component requires authored `smooth`
regularity. The initial imported thermal workflow covers Cartesian 2D Q1 on
CPU through the common Plan, with Plan and Result replay. It adds no verified
threaded, MPI or GPU thermal-package path. Convection, radiation, nonlinear
materials and coupled thermal interfaces are not supplied by these components.

## Transient heat conduction

`TransientHeat2d` borrows a caller-owned temperature **State**, rather than an
algebraic Field. It adds `storage capacity * temperature` to the same Fourier
flux and volumetric source as `SteadyHeat2d`. The constant volumetric `capacity`
parameter has units `J / (m^3 * K)`; conductivity and heating retain their units
above. The caller supplies the initial temperature and complete boundary data,
using `FixedTemperature2d` or `OutwardHeatFlux2d` with the same conductivity.

```eqiora
state temperature: K on body in h1;
initial { temperature = 300[K]; }
instance heat: scalar.TransientHeat2d(
  body = body, temperature = temperature,
  conductivity = 1[W/(m*K)], capacity = 1[J/(m^3*K)], heating = 1[W/m^3]
);
```

Supply positive capacity for physical heat storage. The component checks units
and the declared State role. Numerical methods retain their existing storage
acceptance rules; no new sign policy is introduced. Spatial and time methods
remain Plan choices. The maintained example uses serial CPU Q1 and backward Euler,
with independent mass/stiffness/load checks, initial data, restart and artifact replay.
Product tests also cover a temperature profile affine in space and linear in time,
with time-varying prescribed temperature and both signs of physical outward flux.
The four-cell warming example is a spatial approximation, not a continuum-exact
square temperature profile. State-dependent capacity,
material interfaces, threaded/MPI/GPU package execution and transient
sensitivities are outside this package evidence.

## Conservative transient advection–diffusion

`TransientAdvectionDiffusion2d` borrows a dimensionless caller-owned `scalar`
State on a two-dimensional body. It supplies
`∂scalar/∂t + div(scalar * velocity - diffusivity * grad(scalar)) = source`.
The two prescribed expressions `velocity_x` and `velocity_y` have units `m/s`;
`source` has units `1/s`, and constant positive `diffusivity` has units `m²/s`.
Velocity may vary in space and time, including nonzero curl and divergence.
It must be closed with respect to the solved State. For divergent velocity,
this conservative equation includes `scalar * div(velocity)`.

```eqiora
state concentration: 1 on body in h1;
initial { concentration = 0; }
instance transport: scalar.TransientAdvectionDiffusion2d(
  body = body, scalar = concentration, diffusivity = 1[m^2/s],
  velocity_x = 1[m/s], velocity_y = 2[m/s], source = 1[1/s]
);
```

Supply complete essential boundary data with `FixedValue2d`, binding its
`potential` to the same State. The maintained example is
[`examples/advection-diffusion`](../../../../examples/advection-diffusion/README.md).
The checked path is serial CPU, Cartesian 2D Q1 Galerkin and backward Euler,
with caller-owned initial data and Plan/State/Result replay and restart.
No stabilization, positivity guarantee, outflow/natural or periodic boundary
workflow, finite-volume local-conservation claim, MPI or GPU package evidence
is supplied. Advection-dominated problems may need a stabilized method.

## Convective heat transfer

`ConvectiveHeatTransfer2d` borrows the caller's smooth Kelvin temperature Field
and a boundary of the same two-dimensional body. Supply the same `conductivity`
(in W/(m K)) as the volume heat law, a heat-transfer `coefficient` in W/(m² K),
and an `ambient` Kelvin expression on the face. These inputs have no defaults.
The equation is
`normal(conductivity * grad(temperature)) + coefficient * trace(temperature) = coefficient * ambient`.
Thus physical outward heat flux is `coefficient * (temperature - ambient)`.
A zero coefficient gives insulation; heating and cooling use the same equation.

```eqiora
instance convection: scalar.ConvectiveHeatTransfer2d(
  body = body, face = right, temperature = temperature,
  conductivity = conductivity, coefficient = 2[W/(m^2*K)], ambient = 320[K]
);
```

The supported numerical path is steady affine Cartesian 2D Q1 on the serial CPU
reference provider. TPFA returns a typed unsupported diagnostic. No transient,
threaded, MPI or GPU convection execution is claimed. See the imported and
hand-written `ConvectiveHeatedBody` entries in
[`examples/heated-body`](../../../../examples/heated-body/README.md).
