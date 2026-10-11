# Stokes shape optimization at fixed area

Status: planned. This page narrows, but does not advance, `optimization.stokes-cell-dissipation`;
the deferred Richardson physical-drag target stays with `optimization.cylinder-drag`. Publish it
first as an **Unverified product example**. Promote it to an accepted result only after its
candidate, independent oracle, falsifiers, and acceptance requirements pass under RFC 0089;
evidence work does not hold back the product example.

## What it shows

An accepted steady-Stokes solve drives one immutable shape-design history. The body starts as an
exact circle in a fixed square container. Two smooth, centred shape coordinates keep the body's
analytic area exactly constant. Every trial regenerates the geometry, moves one fixed-topology
body-fitted mesh harmonically, re-solves the all-Dirichlet problem, and evaluates the
finite-element viscous dissipation.

This is a two-dimensional dissipation design inside a bounded cell, not exterior minimum drag.
It claims an independently checked, complete discrete reduced gradient; at least one accepted
nonzero step that lowers the discrete dissipation; an immutable record of accepted and rejected
trials; and an objective ordering that still holds on a separate, pre-chosen refined mesh
topology. The last state is called the **accepted final iterate**, not an optimum, unless a later
stationarity criterion, fixed in advance and checked independently, passes.

It does not claim physical force or drag, a Richardson profile or exterior-flow approximation, a
continuous shape derivative, a mesh-independent or global optimum, general differentiable CAD,
remeshing, topology optimization, Navier--Stokes optimization, or aerodynamic airfoil
performance.

## Storyboard

| Presentation time | Content |
|---|---|
| 0--2 s | Fixed square, exact circular start, uniform outer velocity, no-slip body, viscosity, exact-area profile family, and two design coordinates |
| 2--11 s | Selected accepted iterates morph the exact geometry; pressure is the sole primary field while discrete dissipation, analytic area defect, and gradient-check status stay visible |
| 11--15 s | Complete-history summary, reference/refined objective ordering, and the exact termination reason finish beside the accepted final iterate |
| 15--18 s | Labelled diagram reset to the initial circle; design iterations are never played backward as physical time |

The video samples iterations and says so. It must not suggest that omitted accepted or rejected
trials never ran; rejected trials remain part of the recorded history.

## Evidence needed for acceptance

Two independent derivations fix and reconcile, before any implementation output is seen, the
exact polar-area identity, the complete discrete Stokes dissipation and its reduced derivative,
units, the gradient-comparison test, the sufficient-decrease rule, the refinement test, and the
termination labels. One derivation works from the analytic and discrete formulation; the other
builds the same finite-cell problem independently, without using Eqiora arrays, ordering, or
results.

The positive path first establishes an accepted Stokes state and dissipation, then passes
independently regenerated centred differences of the coefficients, accepts at least one nonzero
decreasing step, keeps the full trial lineage, and preserves the initial-to-final objective
ordering on the refined topology. Work stops if the two derivations disagree, or if any formula,
value, criterion, tolerance, or solver rule would have to be chosen after seeing implementation
output.

The minimum set of falsifiers includes a corrupted exact-area normalization, swapped harmonics, a
wrong body-normal sign, an outer side changed to inlet or traction, a missing pressure gauge, a
wrong dissipation factor, an incomplete state or geometry derivative, finite differences over a
stale polygon, reuse of the parent state without a fresh solve, aliased refinement, acceptance of
an invalid mesh, discarded rejected trials, and budget exhaustion mislabelled as stationarity. A
decreasing objective alone is not enough.

## Dependencies

- accepted exact analytic-source geometry and source-bound mesh identity;
- accepted fixed-topology harmonic coordinate motion and correspondence;
- the all-Dirichlet steady-Stokes Result and its pressure boundary;
- accepted reduced-differentiation and backtracking mechanics;
- a private native store for the full history, exposed read-only through
  `StokesDissipationDesignProjection`; and
- the shared typed presentation, media checks, accessibility, and publication code.

The future registered case is `optimization.stokes-cell-dissipation-2d`. It neither replaces nor
promotes `optimization.cylinder-drag`. This entry uses no force or drag capability and never
treats design trials as a physical Trajectory.

## Accessibility and promotion

The reduced-motion still overlays the initial and accepted final outlines without relying on
colour, and reports the change in discrete dissipation, the analytic area defect, the
gradient-check and refined-ordering status, and the actual termination reason. The text
alternative describes how the shape coordinates changed and states that the displayed iterations
are sampled from a complete, immutable history.

Promotion requires accepted exact-profile and fixed-topology evidence; the state, objective,
complete-gradient, stale-state, and refinement falsifiers; one accepted exact-head candidate that
includes every attempted trial; the installed-Python product projection; and the common
publication check.

## Primary sources

O. Pironneau's
[bounded Stokes variational result](https://doi.org/10.1017/S002211207300145X)
is the source of the fixed outer boundary, moving no-slip body, symmetric-gradient dissipation,
and fixed-volume descent. Its theorem does not provide Eqiora's two-dimensional discrete oracle
or optimizer values.

S. Richardson's
[two-dimensional exterior minimum-drag result](https://doi.org/10.1098/rspa.1995.0103)
belongs to the separate, deferred physical-drag target. This entry uses no Richardson profile
coefficient, effective-radius value, derivative density, or tolerance.
