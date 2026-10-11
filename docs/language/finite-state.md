# Example: a two-state Hamiltonian

This model uses [finite component spaces](finite-spaces.md), complex scalars, ordinary state
evolution, and observables. It is written in the target language: parts of it, such as
`identity` and product spaces mixing duals, are specified but not yet implemented (see
[finite spaces](finite-spaces.md)). Nothing in it relies on a quantum-specific solver.

```eqiora
space Levels = orthonormal(ground, excited);

model TwoState(
  parameter energy_scale: J,
  parameter hbar: J * s
) {
  let hamiltonian: map<complex<J>, Levels, Levels> =
    linear_map(Levels, Levels, [[0 [J], energy_scale], [energy_scale, 0 [J]]]);
  state psi: coordinates<complex<1>, Levels>;

  initial {
    psi = coordinates(Levels, [math.complex(1, 0), math.complex(0, 0)]);
  }
  relation evolution {
    math.i * hbar * derivative(psi) = apply(hamiltonian, psi);
  }

  observable ground_probability: 1 = math.abs2(psi.ground);
  observable excited_probability: 1 = math.abs2(psi.excited);
}

model StationaryTwoState(parameter energy_scale: J) {
  let hamiltonian: map<complex<J>, Levels, Levels> =
    linear_map(Levels, Levels, [[0 [J], energy_scale], [energy_scale, 0 [J]]]);
  variable mode: coordinates<complex<1>, Levels>;
  variable energy: J;
  relation stationary {
    apply(hamiltonian, mode) = energy * mode;
    inner(mode, mode) = 1;
  }
}
```

`energy_scale` and `hbar` must be positive and real. Both models are lumped and continuous, with
no spatial support, boundary, or clock. `psi` is dimensionless because its norm is a finite sum;
a normalized one-dimensional spatial wavefunction would instead have dimension `m^(-1/2)`.

The Hamiltonian is constant and Hermitian: its off-diagonal entries are equal and real, and its
diagonal is zero. The equation keeps `hbar` with dimension `J*s`, so both sides are coordinates
in joules; energy is never silently treated as an angular frequency.

In the stationary model, the normalization equation is what makes the mode nonzero; there is no
initial guess, state, or time. The equations alone do not turn this into an eigenproblem: a
spectral formulation has to identify the eigenvalue, eigenvector, and inner product explicitly.
The number of modes, shifts, tolerances, and algorithms are chosen when solving, not in the
Model.

## Independent solution

Write `e = energy_scale` and `b = hbar`, and measure `t` from the initial state. The exact
solution is:

```text
psi.ground(t)  = cos(e*t/b)
psi.excited(t) = -i*sin(e*t/b)
P_ground(t)    = cos(e*t/b)^2
P_excited(t)   = sin(e*t/b)^2
```

Substituting these into the evolution equations and the initial value confirms them. The total
probability is one and the expected energy is zero. At `t = pi*b/(2*e)` the excited probability
is one; at `t = pi*b/e` it is zero again and the state is the negative of the initial vector,
which has the same probabilities but is a different State.

The stationary eigenvalues are `+e` and `-e`, with normalized eigenvectors `[1, 1]/sqrt(2)` and
`[1, -1]/sqrt(2)`. The two are orthogonal, and multiplying each by the matrix gives its
eigenvalue times the vector. Any eigenvector multiplied by a unit complex phase is the same
eigenmode, so a test should compare residuals, normalization, and projectors rather than raw
vectors.

Changing `energy_scale` changes the frequency and the eigenvalues. Leaving out `hbar` is a
dimension error. Flipping the sign of `math.i` in the evolution equation changes the phase but
not these two probabilities, so only the complex components or a phase-sensitive observable can
catch that mistake.

## Other uses of finite spaces

The same map syntax works for two unrelated voltage channels:

```eqiora
space Channels = orthonormal(command, feedback);
parameter gains: map<1, Channels, Channels> =
  linear_map(Channels, Channels, [[2, 0], [0, 3]]);
let input_channels: coordinates<V, Channels> = coordinates(Channels, [1 [V], 2 [V]]);
let output_channels: coordinates<V, Channels> = apply(gains, input_channels);
```

The result is `[2 V, 6 V]`. Passing `Levels` coordinates instead is an error, even though both
spaces have two entries, and neither space can be replaced by `array<V, 2>` or a spatial
`vector<V, 2>`.

For a product of two spaces, declare two separate two-state spaces:

```eqiora
space Left = orthonormal(ground, excited);
space Right = orthonormal(ground, excited);
space Pair = product(Left, Right);
let left_h: map<J, Left, Left> = linear_map(Left, Left, [[0 [J], 1 [J]], [1 [J], 0 [J]]]);
let right_h: map<J, Right, Right> = linear_map(Right, Right, [[0 [J], 2 [J]], [2 [J], 0 [J]]]);
let pair_h: map<J, Pair, Pair> =
  tensor_product(left_h, identity(Right)) + tensor_product(identity(Left), right_h);
```

Its four eigenvalues are the pairwise sums `3 J, -1 J, 1 J, -3 J`, and its eigenvectors are
products of the factors' normalized eigenvectors. Applied to `[1, 0, 0, 0]`, with the right
factor varying fastest, it gives `[0, 2 J, 1 J, 0]`. Swapping the factors needs an explicit
permutation. The product is expanded in full, which is fine for four coordinates but is not a
scalable many-body representation.
