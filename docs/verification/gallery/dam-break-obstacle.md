# Dam break around an obstacle

Status: planned; this page does not advance `multiphase.dam-break`. Publish it first as an
**Unverified product example**. Promote it to an accepted result only after its candidate,
independent oracle, falsifiers, and acceptance requirements pass under RFC 0089; evidence work
does not hold back the product example.

## What it shows

A water column released under gravity in the Kleefsman et al. dam-break geometry hits a fixed
obstacle, and the run reports water-height and pressure-probe histories. The fluid state,
interface representation, obstacle load, and conservation diagnostics come from one result
lineage.

The entry demonstrates one interface-capturing workflow, with a descriptive side-by-side
comparison to the experiment. It claims no verified agreement, experimental validation, or
acceptance band. It also does not claim a resolved air phase unless the model actually solves
one, nor general impact-pressure accuracy, cavitation, compressible impact, arbitrary
fragmentation, or general free-surface CFD.

## Storyboard

| Presentation time | Content |
|---|---|
| 0--2 s | Tank, initial water column, gravity, obstacle, gauges, pressure sensors, and selected one- or two-fluid model |
| 2--12 s | Water volume fraction or level-set field is the sole primary field through release, impact, run-up, and bounded rebound |
| 12--15 s | One height gauge and one obstacle-pressure trace mark arrival and peak windows; mass defect remains visible |
| 15--18 s | Labelled neutral reset to the initial column; impact is never reversed |

The camera stays fixed so gauge and obstacle positions remain readable. Any interface threshold
used to draw the surface is recorded and fixed.

## Evidence needed for acceptance

Before comparing with the experiment, the prerequisite cases verify hydrostatic balance,
interface advection, boundedness, mass conservation, surface reconstruction, gravity forcing,
wall conditions, and pressure and force projection. Spatial, temporal, interface-thickness, and
sensor-sampling effects are studied separately.

The key observables are free-surface height at the selected gauges, obstacle-face pressure at
the selected sensors, first-impact arrival time, and total liquid-volume defect. The entry is
rejected if the arrival time falls outside its pre-registered experimental band or if the mass
defect exceeds its independently derived budget. A matching pressure spike or a plausible splash
cannot make up for either.

Hydrostatic, transport, and conservation identities use two independent oracles. Experimental
curves, sensor response, extraction, and uncertainty are handled by a separate evidence effort
and keep the source's conventions.

## Dependencies

- one explicitly selected VOF, level-set, or other interface-capturing model;
- gravity, pressure, interface transport, boundedness, and free-surface boundary semantics;
- obstacle pressure and force projections with sensor provenance;
- accepted adaptive or fixed trajectories, including conservative transfer if the mesh changes;
- synchronized interface, gauge, and pressure playback.

The model is chosen before implementation starts; this entry does not justify adding several
interchangeable multiphase formulations at once.

## Accessibility and promotion

The reduced-motion still shows the accepted impact state, the outline of the original column,
gauge and pressure locations, arrival time, mass defect, and a link to the evidence. The text
alternative describes the interface position and obstacle impact without relying on transparency
or colour.

Promotion requires accepted hydrostatic, transport, conservation, sensor, and
experimental-comparison evidence; an accepted free-surface trajectory; and the common
publication check.

## Primary source

K. M. T. Kleefsman et al.,
[“A Volume-of-Fluid based simulation method for wave impact problems”](https://doi.org/10.1016/j.jcp.2004.12.007),
*Journal of Computational Physics* 206, 2005. Redistributing the reference data needs an
explicit licence; otherwise the evidence package records how to obtain the data and its
checksums.
