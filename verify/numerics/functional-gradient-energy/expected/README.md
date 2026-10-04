# Independent Q1 reference

Each cell has side h=1/2. In reference coordinates (s,t) in [0,1]², the central
corner hat on any cell is one of the reflections of N=s*t. Hence

```text
integral_cell |grad(N)|² dA = integral_0^1 integral_0^1 (s²+t²) ds dt = 2/3,
integral_cell N dA = h²/4 = 1/16.
```

Summing four cells gives K=8/3 and b=1/4. The eight boundary coefficients are zero,
the central coefficient is 3/32, and the energy is -3/256. Two-point tensor Gauss
quadrature integrates the Q1 energy exactly in each reference coordinate.

For the physical profile, u and w have dimension m, k has J/m², and f has J/m³.
Thus both the directional variation and F have dimension J. This is a two-
dimensional functional, not a three-dimensional elastic material or a stability
claim. The tested physical value type must be exactly real J.

The solver requests relative tolerance 1e-10 and absolute tolerance 1e-12.
With a single free coefficient, the load-scaled residual bound divided by K=8/3
is below 1e-10. The coefficient and energy assertions use absolute tolerance
1e-10 in their stated SI coordinates. At the stationary coefficient, the energy
perturbation is K*(delta u)²/2; the same tolerance also covers the few binary64
quadrature operations. These bounds come from this fixed discrete problem, not
from measured output. No continuum value is substituted for the Q1 energy.
