# Three-dimensional Taylor--Green breakdown

Status: planned. This page is separate from, and does not advance, the analytic or manufactured
`fluid.taylor-green` candidate. Publish it first as an **Unverified product example**. Promote it
to an accepted result only after its candidate, independent oracle, falsifiers, and acceptance
requirements pass under RFC 0089; evidence work does not hold back the product example.

## What it shows

The visible title is **Eqiora 3D Taylor--Green numerical study**. The video follows one
full-cube, incompressible `Re=1600` Eqiora FVM/BDF1 computation in a triply periodic domain, from
the Taylor--Green initial state through vortex-sheet formation and small scales to peak physical
dissipation.

The accepted result publishes four scalar histories: dimensionless kinetic energy `E*`,
enstrophy `Z*`, physical viscous dissipation `epsilon_nu*`, and right-endpoint energy loss
`epsilon_E*`. With them come spatial and temporal refinement of those histories; mass, gauge,
periodicity, symmetry, and energy-ledger diagnostics; the lineage of the exact peak state; and a
deterministic sequence of vorticity-magnitude surfaces at `|omega|*=5`.

This is a numerical study checked by Eqiora's own analytic, conformance, ledger, and refinement
tests. It is not an exact solution after the initial state, a comparison with external numerical
histories, experimental validation, agreement with published histories, DNS certification, a
universal benchmark, turbulence-model validation, a singularity study, a convergence result for
local fields or surface geometry, a distributed or device result, or an HPC performance claim.

## Storyboard

| Presentation time | Content |
|---|---|
| 0--2 s | Periodic cube, exact initial velocity symmetries, `Re=1600`, Eqiora discretization, campaign envelope, and physical interval |
| 2--12 s | Fixed or slowly prescribed camera shows the deterministic `|omega|*=5` vorticity-magnitude surfaces through the exact accepted physical-dissipation peak state |
| 12--15 s | The exact panels “Eqiora histories,” “Spatial refinement,” “Temporal refinement,” and “Energy ledger and invariants” mark current time, accepted peak identity, and frozen scalar criteria |
| 15--18 s | Labelled neutral cut to the initial symmetric state; decay is never reversed |

There is no Q-criterion or `Q*`. Isovalue, scale, periodic chart, camera, clipping, and opacity
are fixed and never adjusted for individual frames.

## Evidence needed for acceptance

Two derivations, kept isolated from each other, fix every new initial formula, cell average,
integral, normalization, exact identity, discrete ledger term, acceptance criterion, expected
value, tolerance, solver criterion, and arithmetic floor before any Eqiora output is seen. One
agent may write both as long as neither reads the implementation or the other's output. Eqiora's
histories are results to be checked, never a source of expected values.

The reusable 3D periodic collocated class first passes one ordinary positive path, then its
seam, checkerboard, retained-flux, JVP, ledger, resource, and hidden-parity falsifiers. The full
Taylor--Green campaign must then pass mass and pressure-gauge checks, periodic replay, observed
symmetry plus a probe for a hidden symmetry outside the class, separation of physical and
numerical energy loss, closure of the discrete ledger, exact peak retention, complete lineage,
and the fixed spatial and temporal criteria for the scalar histories.

Self-refinement supports only that declared consistency and convergence of the scalar histories.
It says nothing about continuum truth after breakdown, independent trajectory truth after the
initial state, or convergence of local fields or surface geometry. A plausible history, a
matching peak, or a striking surface cannot override a failed check anywhere in the campaign.

No external numerical history after the initial state, whether from UCL, Southampton/OpenSBLI, a
Fourier code, or anywhere else, and no source peak, band, comparison curve, or comparison
statement is used for acceptance or publication.

## Dependencies

- the #149 dependency boundary and an accepted, executable 3D periodic reference capability;
- an accepted reusable collocated 3D periodic class with independent conformance evidence;
- the full-cube incompressible Eqiora FVM/BDF1 path and the complete pre-fixed campaign;
- immutable lineage for the Model, Realization, source, environment, states, Result, histories,
  peak, projection inputs, and scene;
- deterministic periodic vorticity-magnitude surface extraction and a minimal shared 3D
  presentation; and
- the common publication, accessibility, resource, and dossier checks.

The video does not justify a general volume viewer, a turbulence path, arbitrary periodic
topologies, a Reynolds-number family, a durable schema for large results, or unlimited result
size.

## Accessibility and promotion

The reduced-motion still shows the accepted peak-dissipation state, a section or silhouette that
reads without depth motion, and the full Eqiora energy history. Its text alternative describes how
the flow develops from the symmetric initial state and identifies the surface as a deterministic
presentation, not a quantitative accuracy check.

Every page and application that shows the result carries this disclosure:

> Eqiora-computed histories and surfaces are accepted by analytic initial
> checks, reusable-class conformance, balance and energy-ledger checks,
> symmetry checks, and precommitted spatial/temporal scalar-history
> refinement. They are not an external-history comparison, DNS certification,
> or claim of post-breakdown continuum truth or local surface convergence.

Promotion requires the accepted 3D periodic reference and reusable class, both derivations and
their review, the complete Eqiora campaign, immutable result checks, the deterministic vorticity
projection, and the common publication check. The dossier marks community numerical comparison
and experimental validation as not claimed.

## Primary source

M. E. Brachet et al.,
[“Small-scale structure of the Taylor--Green vortex”](https://doi.org/10.1017/S0022112083001159),
*Journal of Fluid Mechanics* 130, 1983, provides historical context only. No numerical history,
value, peak, or band from it is used for this claim.
