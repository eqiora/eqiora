# Specimen: a normalized 1D wavefunction

These two models describe a particle in a box: a fixed physical interval with zero
wavefunction at both ends, zero potential, positive mass, and positive `hbar`. The
stationary model and the time-dependent model share the same kinetic operator. The
page specifies the source and the mathematics; it does not mean that complex field
or eigenproblem execution is currently available.

```eqiora
operator kinetic(
  input second_partial: complex<m^(-5/2)>,
  input mass: kg,
  input hbar: J * s
): complex<J * m^(-1/2)> = -(hbar * hbar / (2 * mass)) * second_partial;

model StationaryBox(
  support line: interval(m),
  support ends: complete_exterior(parent = line),
  parameter mass: kg,
  parameter hbar: J * s
) {
  coordinate x: m on line from line;
  variable psi: complex<m^(-1/2)> on line;
  variable energy: J;

  relation endpoint_values on ends {
    trace(psi) = 0;
  }
  relation eigen_equation on line {
    kinetic(second_partial = partial(partial(psi, wrt = x), wrt = x), mass = mass, hbar = hbar)
      = energy * psi;
  }
  relation unit_norm {
    integral(math.abs2(psi), measure(line)) = 1;
  }
  form spectrum for eigen_equation {
    eigenpair(mode = psi, value = energy, metric = l2(measure(line)), normalization = unit_norm);
  }
}

model EvolvingBox(
  support line: interval(m),
  support ends: complete_exterior(parent = line),
  parameter length: m,
  parameter mass: kg,
  parameter hbar: J * s
) {
  coordinate x: m on line from line;
  state psi: complex<m^(-1/2)> on line;

  initial {
    psi = math.sqrt(1 / length)
      * (math.sin(math.pi * x / length) + math.sin(2 * math.pi * x / length));
  }
  relation endpoint_values on ends {
    trace(psi) = 0;
  }
  relation evolution on line {
    math.i * hbar * derivative(psi) =
      kinetic(second_partial = partial(partial(psi, wrt = x), wrt = x), mass = mass, hbar = hbar);
  }
  observable density: 1 / m on line = math.abs2(psi);
  observable probability: 1 = integral(math.abs2(psi), measure(line));
}
```

Bind `line` to the interval `[0, L]` with `L > 0`, and `ends` to both endpoints. In the
time-dependent model, bind `length` to the same L; if it differs from the geometry,
the initial condition no longer satisfies the boundary condition. Start time at zero.
Nothing is implied about periodicity, absorbing ends, stochastic paths, or a numerical
grid. `kinetic` is a pure operation evaluated pointwise; it does not read a
coordinate or take derivatives of its arguments.

Each spatial derivative divides the wave amplitude's dimension by length, so the
second partial has dimension `m^(-5/2)`, and both the kinetic term and
`i*hbar*derivative(psi)` have `J*m^(-1/2)`. Real initial data becomes complex without
losing its units. The density has units of `1/m`, so its integral over the line is
dimensionless.

## Eigenpair and normalization roles

`eigenpair(...)` is a form child with four required arguments. `mode` names the
unknown, `value` the real scalar eigenvalue, `metric` the inner product, and
`normalization` the relation that fixes the norm. It does not create a new mode,
eigenvalue, or copy of the normalization equation.

`l2(measure(line))` is the inner product that integrates `conj(a)*b` over that
measure, and `euclidean(space)` is the orthonormal metric of a declared finite space.
Both are typed constructors, not strings and not a guessed numerical mass matrix. The
referenced relation provides the norm target, so the zero vector is excluded by the
unit-norm equation rather than by the solver's starting vector.

The form includes the mode's endpoint conditions and everything the eigen-equation
depends on. Missing boundary conditions, an indefinite or singular metric, an
eigenvalue with the wrong dimension, or an unsupported Hermitian case is an error.
Numerically, the metric is represented by mass matrices; normalizing raw nodal
coefficients in the Euclidean norm is not equivalent. The number of modes, shifts,
search ranges, and algorithms belong in the study or Plan, not the model.

For this real, zero-potential operator with Dirichlet ends, integration by parts shows
that it is self-adjoint. Adding a potential or changing the boundary conditions later
must pass the same checks.

## Reference solutions

For positive integer n, differentiating `sin(n*pi*x/L)` gives:

```text
phi_n(x) = sqrt(2/L) * sin(n*pi*x/L)
E_n = hbar^2*pi^2*n^2/(2*mass*L^2)
integral_0^L |phi_n|^2 dx = 1
```

The modes vanish at the ends and are mutually orthogonal. The n=0 function is
identically zero and cannot be normalized, so it is not a ground state. A constant
potential V0 would add V0 to every energy without changing the modes, but it must be
written into the model.

The time-dependent model starts from `(phi_1 + phi_2)/sqrt(2)`. The exact solution is
`(phi_1*exp(-i*E_1*t/hbar) + phi_2*exp(-i*E_2*t/hbar))/sqrt(2)`;
substituting it into the evolution equation confirms the sign and the factor of hbar.
The total probability is one and the mean energy is `(E_1 + E_2)/2 = 5*E_1/2`.

The density is
`(phi_1^2 + phi_2^2)/2 + phi_1*phi_2*cos((E_2-E_1)*t/hbar)`.
At `x = L/4` it starts at `(3/2 + sqrt(2))/L` and reaches `(3/2 - sqrt(2))/L` at
`t = pi*hbar/(E_2-E_1)`. This interference checks the relative phase, which the total
probability alone cannot. Reversing the direction of time leaves this density
unchanged for real initial data, so detecting a flipped sign in the evolution
equation also needs the complex amplitude or the probability current.

A restart continues from the stored complex field, interval, boundaries, and time. It
must not reapply the initial superposition or renormalize a state that has drifted
numerically. How often results are written does not set the integrator's time steps,
and spatial and temporal accuracy must be checked for the chosen method.

Wrong wave-amplitude dimensions, missing endpoint conditions, a flipped kinetic sign,
and a missing hbar each break a different requirement above, as does normalizing
nodal values in the Euclidean norm. Tests should check those requirements directly
rather than compare eigenvector phases, which are arbitrary, or whole output files.
