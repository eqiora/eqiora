# Acceptance contract

The source mesh has 15 vertices, 16 cells, and two interface facets. The
target mesh has 17 vertices, 20 cells, and four interface facets. The target
interface contains every source P1 trace breakpoint, so refinement preserves
the accepted kinked trace rather than silently smoothing it. Both the
material solid overlap and current-spatial fluid overlap must contain at least
one source cell contributing to multiple target cells and at least one target
cell receiving contributions from multiple source cells.

Before transfer, the accepted source state must contain a nonzero MINI bubble,
a nonzero nonconstant absolute-pressure field, nonzero shared-interface
velocity, and nonzero solid displacement. These checks prevent a degenerate
constant or zero field from making an incorrect transfer appear exact.

The transfer occurs at exactly the accepted source time. Absolute material
displacement is projected first and must regenerate the target harmonic
geometry. Fluid and solid velocity share one constrained solve but retain
their distinct current-spatial and material integration charts. Pressure is
transferred absolutely. The accepted evidence must close solver residuals,
weak incompressibility, momentum, pressure moment, displacement continuity,
shared-velocity continuity, exterior velocity conditions, and harmonic replay
within the encoded bounds.

Repeating the same projection with a distinct computational L/U/P scale must
independently replay the same material overlap, current-overlap connectivity,
physical constraint roles and counts, solve reports, and dimensionless
acceptance contracts. The physical Fields are compared in the common
Realization units against the non-semantic observation bound registered in
`case.toml`; equality of normalized constraint coefficients, iterative stopping,
and coefficient bit identity across scale profiles are not claimed.

The transferred initial state must then advance through the ordinary target
finalizer by one positive time step. Its artifacts must replay as one immutable
V2 source prefix, one current remesh-origin state, and one current continuous target
state with a two-root content-addressed chain ending at generation 1.
