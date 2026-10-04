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

## Authored first variation on four cells

For the complete-essential profile, the only free Q1 basis is the central hat N
in each displacement component. Its independent integrals are
integral N=1/4, integral N_x²=integral N_y²=4/3 and integral N_x*N_y=0.
Thus K_xx=K_yy=4*(lambda+3*mu)/3=44/3, K_xy=0, b=(6/4,0),
u_center=(9/88,0) and F=−b.u/2=−27/352 N.
The first variation is integral (2*mu*epsilon(u):epsilon(w) +
lambda*div(u)*div(w) − grad(q).w) dA. For the central hat this is the free
residual, bounded by 1.5e−10 from the requested solver tolerance. For the
inadmissible constant translation w=(1,0), all its derivatives vanish and
DF=−integral 6 dA=−6 N. The derivative assertions reserve 1e−9 for binary64
rounding. The central coefficient residual bound divided by 44/3 is below
1.1e−11 m; the coefficient and energy assertions use 1e−10 in SI units.
At stationarity the energy error is quadratic, delta-u.K.delta-u/2.

For the mixed-boundary profile, mu=3, lambda=0 and q=6x. With cell width h=1/2,
the Q1 x slopes are 3/4 and 1/4: the internal energy is 15/16 N and load pairing
is 30/16 N, giving F=−15/16 N. Nodal values interpolate x−x²/2.
For local vertex bits (x_i,y_i), the derivative products A_ab(i,j)=integral
N_i,a*N_j,b are sign(i,a)*sign(j,b)/4 if a!=b, and sign(i,a)*sign(j,a)
times 1/3 or 1/6 according to whether the other vertex bit agrees when a=b.
The elastic element matrix is mu*delta_ab*sum_k A_kk + mu*A_ba + lambda*A_ab.
Assembling four cells and deleting the six left-edge component DOFs gives a
12-by-12 matrix with exact inverse infinity norm 316103/59058. The free load
has norm 6*sqrt(15/128). The relative residual request 1e−10 therefore bounds
coefficient error below 1.1e−9 m; assertions use 2e−9 m, reserving the remainder
for assembly and solve rounding. The 1e−9 N energy tolerance covers the
quadratic stationary error and quadrature arithmetic. These bounds are derived
from the independent Q1 integrals, not measured solver output.

The same complete-essential functional has ordered second variation
`D²F[eta,zeta] = integral (2*mu*epsilon(eta):epsilon(zeta)
+ lambda*div(eta)*div(zeta)) dA`. The fixed conservative load drops out.
The central x-component hat paired with itself therefore gives `44/3 N`,
and paired with the central y-component hat gives `0`, using the independent
stiffness entries above. A constant translation has zero strain and yields zero
with either direction. Two-point Gauss exactly integrates these Q1 products;
the second action is independent of solved coefficients. The `1e-10 N`
tolerance reserves quadrature and binary64 arithmetic error on this four-cell
fixture. This checks consistency with the independently assembled stiffness;
it does not differentiate the implicit solve or infer nonlinear stability.
