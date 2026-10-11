# Turek--Hron FSI3

Status: planned; this page does not advance `multiphysics.turek-hron-fsi`. Publish it first as
an **Unverified product example**. Promote it to an accepted result only after its candidate,
independent oracle, falsifiers, and acceptance requirements pass under RFC 0089; evidence work
does not hold back the product example.

## What it shows

Two-way interaction between the laminar wake of a fixed cylinder and the elastic flag attached to
it, in the Turek--Hron FSI3 numerical benchmark. Fluid and solid states, the moving mesh,
conservative interface exchange, and the reported observables all come from one accepted lineage.

This is the gallery's coupled-physics entry. It claims comparison with a published numerical
benchmark, not experimental validation. FSI3 specifies a finite-strain solid; the case may not
use the current linear solid while keeping the FSI3 name.

It does not claim general FSI, equivalence of partitioned and monolithic coupling, production
preconditioning, remeshing, turbulence, or three-dimensional coupling.

## Storyboard

| Presentation time | Content |
|---|---|
| 0--2 s | Fluid and solid domains, inlet, wall/outlet conditions, material properties, interface, and point A |
| 2--12 s | Vorticity is the primary field; the accepted deformed interface and mesh motion remain visible without a second color field |
| 12--15 s | Point-A displacement trace and one interface balance defect mark the current phase |
| 15--18 s | Phase-matched return after a complete accepted oscillation |

The camera stays fixed so the flag amplitude is readable. Mesh lines appear only briefly, and
labelled, if they are needed to explain the ALE motion.

## Evidence needed for acceptance

The coupled case builds on already accepted evidence for transient non-box flow, finite-strain
structures, moving meshes and the geometric conservation law, and interface transfer. It then
compares the mean, amplitude, and frequency of point A's horizontal and vertical displacement,
together with drag and lift, in the source's conventions. Fluid mass, solid energy, geometric
conservation, and interface action/power defects are reported separately in the dossier.

The decisive test is the full set of point-A and force observables under spatial and temporal
refinement. If any component fails its pre-fixed criterion, the case is rejected, however
plausible the phase portrait looks. A linear-solid run is a diagnostic only; if it cannot be
distinguished from the finite-strain result within the accepted band, the case specification or
band goes back for review before implementation continues.

Derivable interface, energy, and manufactured component checks use two independent oracles.
Published FSI3 comparison bands are extracted and normalised independently of the implementer.

## Dependencies

- the accepted cylinder-wake fluid path, including force projection;
- finite-strain St. Venant--Kirchhoff solid dynamics;
- fixed-topology ALE mesh motion and geometric conservation on this geometry;
- conservative coupled iteration, with separate fluid, solid, and interface acceptance checks;
- durable coupled trajectories and synchronized field and trace playback.

No general FSI visualization protocol is needed; the gallery media support is extended only for
one moving 2D interface and one group of coupled observables.

## Accessibility and promotion

The reduced-motion still shows the state of maximum point-A displacement, the vorticity, the
interface direction, the displacement amplitude and frequency, and a link to the balance
evidence. The text alternative explains how the wake and the flag exchange energy.

Promotion requires all component cases, accepted FSI3 refinement and comparison evidence, a
coupled trajectory whose displayed fields share exact state identities, and the common
publication check.

## Primary source

S. Turek and J. Hron,
[“Proposal for Numerical Benchmarking of Fluid-Structure Interaction between an Elastic Object and Laminar Incompressible Flow”](https://doi.org/10.1007/3-540-34596-5_15),
in *Fluid--Structure Interaction*, LNCSE 53, 2006.
