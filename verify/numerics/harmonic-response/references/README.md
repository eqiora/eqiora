# Independent values and error bounds

For a grounded series RC circuit, `Vs-V=R*I`, `I=C*dV/dt`. With the declared
negative exponential and peak amplitudes, differentiation gives `-i*omega` and
`Vhat=Vshat/(1-i*omega*R*C)`, `Ihat=-i*omega*C*Vhat`.
For `R=1000 Ohm`, `C=1e-6 F`, `omega=1000/s`, `Vshat=1 V`, the exact amplitudes
are `(1+i)/2 V` and `(1-i)/2000 A`. Real reconstruction is
`Re(A)*cos(omega*t)+Im(A)*sin(omega*t)`. Four quarter-period samples integrate the
squared current exactly, giving mean resistor loss `R*|Ihat|²/2=1/4000 W`.
Using RMS in place of peak would double this value.

For cyclic frequency `f=250/s`, explicit `omega=2*pi*f` instead gives
`x=omega*R*C=pi/2` and `Vhat=(1+i*x)/(1+x²) V`. Omitting `2*pi` gives `x=1/4`;
the real voltage components differ by more than 0.5 V. Both are mathematically
valid angular-frequency requests; equal dimensions alone cannot identify a user's
intended cyclic-frequency role.

Adding `L=1 H` changes the denominator to `1-omega²*L*C-i*omega*R*C`.
At the above angular frequency, `Vhat=i V`, `Ihat=0.001 A`. The unscaled SI
matrix inverse infinity norm is `1+1000*sqrt(2)`. A `1e-14` relative / `1e-16`
absolute residual request leaves margin inside the fixed `1e-10` component bound.
The resistor-power bound is `3e-10 W`: propagating `1e-10 A` component errors
through the quarter-period sum contributes less than `2e-10 W`.

For the damped scalar wave `u_tt+u_t-div(grad(u))=f` with coherent SI coefficients
on `[0,1] m`, take `omega=1/s` and constant `U=2+i`. Then
`(-omega²-i*omega)U=-1-3i`, the left Dirichlet amplitude is `2+i`, and the right
outward flux is zero. Constant U belongs exactly to the two-cell Q1 space, so
the test's `1e-10` coefficient bound covers numerical solution error, not a
continuum approximation claim. Omitting either boundary Relation or either
explicit excitation must reject before assembly.

For the independently eliminated RC time equation, put `s=t/(R*C)`.
The exact initial-value solution is
`V(s)=0.5*(cos(s)+sin(s))+(V0-0.5)*exp(-s)` for `V0=0` or `1 V`.
Both have `|V''|,|V'''| < 1.21` in s coordinates. Implicit midpoint uses
`delta=h/(R*C)=0.001`. Its local defect is at most
`(1.21/24+1.21/8)*delta³/(1+delta/2)`. Summing with contraction factor
`q=(1-delta/2)/(1+delta/2)` bounds grid error by `0.202*delta²`.
Linear sample interpolation adds at most `1.21*delta²/8`; their sum is below
`3.54e-7 V`. The fixed `5e-7 V` acceptance bound leaves room for binary64
roundoff and tightly solved Newton corrections. The harmonic comparison adds
its separate `1e-10 V` bound. At twenty time constants the remaining transient
amplitude is below `1.04e-9 V`, giving a settled comparison bound of `5.02e-7 V`.
Positive R and C justify decay; no decay assertion is made for arbitrary systems.
