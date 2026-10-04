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

## Essential left side and three zero-flux sides

On the same four cells, the exact discrete solution is independent of y and
interpolates u=x−x²/2. Its values are 0, 3/8 and 1/2 at the three x coordinates;
its piecewise x slopes are 3/4 and 1/4. Direct integration gives internal energy
( (3/4)² + (1/4)² )/4 = 5/32 and load pairing
(0+2*(3/8)+1/2)/4 = 5/16, hence F=−5/32. This is the discrete Q1 energy,
not the continuum energy −1/6.

For local vertex order (00,10,01,11), the unit-diffusion element stiffness has
diagonal 2/3, edge-neighbor entries −1/6 and opposite-corner entries −1/3.
Assemble the four cells and delete the three x=0 nodes. Exact rational elimination
of this six-by-six matrix gives inverse infinity norm 234/49. The free load
vector, in y-major order, is (1/8,1/16,1/4,1/8,1/8,1/16), with squared two-norm
15/128. Thus the requested relative residual 1e−10 gives coefficient error at
most (234/49)*sqrt(15/128)*1e−10 < 1.7e−10. The coefficient assertions use 1e−9
to cover binary64 assembly and solve rounding. At stationarity energy error is
one half of delta-u transposed times K times delta-u; the 1e−9 energy tolerance
also covers the fixed quadrature arithmetic. These are bounds from the independent
Q1 matrix and load, not measured solution errors.
