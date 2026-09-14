# Specimen: steady and transient heated bodies

The maintained [heated-body local package](../../examples/heated-body/README.md)
now executes through ordinary installed Python. Its [single mathematical source](../../examples/heated-body/src/main.eqi)
declares a steady heat Law on a unit-square cross-section, four complete 300 K
essential boundaries, and an authored weak form whose test vanishes on that
exterior. The [Python runner](../../examples/heated-body/run.py) supplies exact
Geometry and explicit Q1 and solver choices.

The independent four-cell Q1 oracle gives eight 300 K boundary coefficients and
one central coefficient of 300 + 3/32 K. Installed product tests check the free-row
heat balance, direct and exact-package execution, Plan/Result replay, and an
unchanged lock after moving the vendored project offline.

The temperature is an absolute Kelvin field; no shifted variable conceals the
nonzero boundary condition. This is a steady manufactured discrete problem,
not thermal time evolution or an exact continuum temperature profile.

The same source also exposes `TransientHeatedBody`, with constant positive
volumetric heat capacity and an explicit 300 K initial State. It retains the
complete boundary and advances through Q1, BackwardEuler and the ordinary
Plan/State/Run/Result APIs. On the same four-cell mesh, the interior mass is 1/9,
stiffness is 8/3 and heating load is 1/4 in coherent SI units per unit depth.
With unit capacity and step 1/24 s, the independent coefficient is
`300 + (3/32)*(1 - 2**(-n))` K after n steps. Each step's heat balance,
restart, replay and moved offline lock are tested independently.

The transient entry uses the shared compiled region equations; an authored
transient Law-to-Form correspondence certificate is outside this claim. Wider
capacities, material interfaces, 3D thermal execution and a curated thermal
standard package remain separate work.
