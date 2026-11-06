# Prescribed scalar transport

Run `python examples/advection-diffusion/run.py` with an installed development
build. Published Eqiora 0.1.5 does not include Scalar 0.5.0.

The dimensionless concentration obeys the conservative equation
`∂c/∂t + div(c β - κ grad(c)) = q` on a one-metre square, with
`β=(1,2) m/s`, `κ=1 m²/s`, `q=1/s`, and zero initial and boundary values.
The example chooses serial CPU Q1 Galerkin, four cells, and backward Euler
with a quarter-second step. This coarse demonstration has one free coefficient.
Independent tensor-product integration gives `M=1/9`, `K=8/3`, `F=1/4` and
zero diagonal convection, so the first centre coefficient is `9/112`.
This is a discrete check, not a continuum-exact concentration profile.

To write the same equation yourself, replace only the `transport` instance with:

```eqiora
law transport on body {
  storage concentration;
  flux concentration * (1[m/s]*grad(coordinate(0)) + 2[m/s]*grad(coordinate(1)))
       - 1[m^2/s]*grad(concentration);
  source 1[1/s];
}
```

The standard boundary components can be mixed with this authored Law.
Velocity expressions can have curl and divergence; the equation retains
`c div(β)` when velocity is divergent. Supply closed prescribed expressions,
not expressions depending on concentration. This example covers full fixed
values, not outflow boundaries or stabilization. Q1 Galerkin does not promise
positive concentrations for advection-dominated problems. MPI, GPU, periodic
and finite-volume workflows are not established by this example.
