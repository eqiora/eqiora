# Independent discrete reference

Put h=1/16. The Q1 solution interpolates u_x=x-x²/2, u_y=0 at the vertices.
On x-cell i its sole nonzero gradient component is g_i=1-(i+1/2)h.
Summing the square over the square domain gives

```text
E_h = 3*h*sum_i g_i² = 1-h²/4 = 1023/1024 N.
DE_h[eta] = 6*h*sum_i g_i = 3 N,  eta=(x,y) m.
```

At x=1, g=h/2, so the recovered outward traction is (3h,0) Pa,
while trace(u)=(1/2,0) m. Thus the boundary work is 3h/2=3/32 N.
Delta traction for eta is (6,0) Pa and trace(eta)=(1,y) m, giving
6/2+3h=51/16 N. The nonzero recovered traction is not replaced by the
zero natural-law datum. Two-point tensor Gauss quadrature is exact for these
Q1 polynomial densities on each cell and face.

The test checks the exact nodal displacement with absolute error budget
2e-11 m. For a Q1 interpolant whose two displacement components each satisfy
that budget delta, each gradient entry has error at most 2*delta/h=32*delta.
The Frobenius strain error is at most 64*delta. Cauchy–Schwarz with the exact
strain norm below sqrt(1/3) bounds the energy error by 222*delta+12288*delta².
The volume JVP error is at most 384*delta. On the right face the work error is
below 97*delta+384*delta² and the work JVP error is at most 390*delta.
All are below 8e-9 in coherent SI units. The fixed absolute 1e-8 checks reserve
the remainder for binary64 projection and quadrature. These budgets are set
from the shape-function bounds, not copied from measured output.
