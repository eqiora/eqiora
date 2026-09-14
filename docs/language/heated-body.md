# Specimen: a steady heated body

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

Fixed-volume Law storage has separate language admission, but transient spatial
thermal execution is not connected. Consequently this executable specimen has
no initial condition or storage term. The former proposed three-dimensional
transient cube and `Eqiora.Thermal.Conduction` import have been removed rather
than presented as an executable standard package. Full transient boundary/initial
conditions and curated thermal package composition remain tracked by #896.
