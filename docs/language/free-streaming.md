# Example: free streaming in one position and one velocity dimension

This model transports a scalar distribution function on the product of one position coordinate
and one velocity coordinate. It uses ordinary Relations and a periodic connection; there is no
special Vlasov keyword and no collision model. The language for it is specified, but running
the full product-space model is not yet supported.

```eqiora
connector StreamingBoundary {
  trace value: s / m^2;
  flux transport: 1 / m;
}

model FreeStreaming(
  support position: interval(m),
  support velocity: interval(m / s),
  parameter length: m,
  parameter speed_limit: m / s,
  parameter density: 1 / m
) {
  support phase: product(position, velocity);
  support left: boundary(parent = phase, factor = position, side = lower);
  support right: boundary(parent = phase, factor = position, side = upper);
  coordinate x: m on phase from position;
  coordinate v: m / s on phase from velocity;
  state distribution: s / m^2 on phase;
  port left_port: StreamingBoundary on left;
  port right_port: StreamingBoundary on right;

  initial {
    distribution = density / (2 * speed_limit)
      * (1 + 0.2 * math.cos(2 * math.pi * x / length));
  }
  relation transport on phase {
    derivative(distribution) + partial(v * distribution, wrt = x) = 0;
  }
  relation left_interface on left {
    left_port.value = trace(distribution);
    left_port.transport = -trace(v * distribution);
  }
  relation right_interface on right {
    right_port.value = trace(distribution);
    right_port.transport = trace(v * distribution);
  }
  connect periodic left_port, right_port;

  observable position_density: 1 / m on position = integral(distribution, measure(velocity));
  observable inventory: 1 = integral(distribution, measure(phase));
}
```

Bind position to `[0,L]`, velocity to `[-V,V]`, `length=L>0`, `speed_limit=V>0`, and a positive
line density n0. Time starts at zero. The periodic connection pairs the two position ends at the
same velocity: the velocity coordinate is unchanged, and the outward directions of the two ends
are opposite. The pairing comes from the topology of the product support, never from matching
shapes or nearby coordinates.

The connector's `trace` and `flux` are scalar quantities on those boundaries. Their dimensions
differ because the transported quantity is velocity times distribution. Both ends use the same
increasing-x coordinate, so the minus sign on the left makes its flux outward. The periodic
connection makes the two traces equal and the two outward fluxes cancel, instead of forcing zero
flux at each end.

There is no acceleration, so nothing is transported in velocity. Each characteristic keeps its
velocity, and neither velocity end has incoming characteristics that would need a boundary
value. The velocity range is finite; the model does not approximate a Gaussian tail or an
infinite domain. Adding acceleration would require a velocity flux and boundary treatment at the
velocity ends as well.

## Exact solution and moments

Let `k=2*pi/L`. The exact solution is:

```text
f(x,v,t) = n0/(2*V) * (1 + 0.2*cos(k*(x-v*t)))
```

It is periodic in x and matches the initial data. Differentiating gives
`partial_t f = (n0/(2*V))*0.2*k*v*sin(k*(x-v*t))` and
`v*partial_x f = -(n0/(2*V))*0.2*k*v*sin(k*(x-v*t))`, so the two terms cancel. Both have units
1/m^2. A velocity derivative would divide f by m/s instead and cannot stand in for the position
derivative.

Integrating over `[-V,V]` gives:

```text
n(x,t) = n0 * (1 + 0.2*cos(k*x)*sin(k*V*t)/(k*V*t)),  t != 0
n(x,0) = n0 * (1 + 0.2*cos(k*x))
integral_0^L n(x,t) dx = n0*L
```

At `t = 0` the density takes its analytic limit. At `t=L/(2*V)` the density is exactly n0
everywhere, even though the distribution in phase space still carries the transported
perturbation; nothing has relaxed or collided. The total stays n0*L, and the smallest value of
the distribution is `0.8*n0/(2*V)>0`. A numerical scheme does not automatically keep it positive.

By symmetry, the first velocity moment integrated over space is zero. The second is
`n0*L*V^2/3`, in m^2/s^2; an average would divide it by the total explicitly. Both moments are
integrals over velocity, not unweighted sums of samples.

## What to check

The analytic derivatives above test the units and signs of the coordinates. When the
distribution is unknown, its derivative stays symbolic until a numerical representation of the
product space is chosen. A finite-volume reconstruction, a finite-element basis, or another
representation brings its own accuracy and derivatives, but none of them turns the phase space
into a physical two-dimensional plate.

The compiler should reject velocity used in place of position, a pairing with a foreign
boundary, a trace with the wrong units, a flux sign reversed without reversing the orientation,
and inconsistent L and V bindings. Leaving out the periodic connection must not silently produce
an absorbing or reflecting boundary. Compare the whole distribution, not only density and total:
a solver that averages over velocity can match the density at one late time while getting the
distribution wrong.

A restart keeps the order and identity of the factors, the periodic topology, the distribution,
and the time. Mesh refinement and time steps are chosen in the Plan. How often output is written
does not affect the transport or the normalization.
