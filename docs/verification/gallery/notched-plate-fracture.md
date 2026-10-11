# Notched-plate phase-field fracture

Status: planned; this page does not advance any fracture capability or benchmark. Publish it
first as an **Unverified product example**. Promote it to an accepted result only after its
candidate, independent oracle, falsifiers, and acceptance requirements pass under RFC 0089;
evidence work does not hold back the product example.

## What it shows

Quasi-static crack growth from one notch in a two-dimensional brittle plate, using one explicitly
selected phase-field fracture formulation. The load path, displacement, phase field,
irreversibility state, and energy terms come from one lineage.

The first case shows initiation and propagation, not dynamic branching. Its claim covers the
specific regularized model, length scale, mesh relation, loading protocol, irreversibility
treatment, and energy evidence chosen for the scientific case. It does not claim a sharp crack at
finite length scale, mesh-independent crack topology, material validation, fatigue, ductile
fracture, or general topology change.

## Storyboard

| Presentation time | Content |
|---|---|
| 0--2 s | Plate, notch, supports, imposed displacement, elastic and fracture parameters, phase-field length scale, and mesh relation |
| 2--11 s | Phase field is the sole primary field as the accepted load steps cross initiation and propagate the regularized crack |
| 11--15 s | Load--displacement and elastic/fracture energy histories mark peak load and current state |
| 15--18 s | Labelled neutral reset; damage never heals or plays backward |

Displacement may be drawn as geometry at a fixed, labelled scale, but not as a second colour
field.

## Evidence needed for acceptance

The prerequisite cases verify elasticity, phase-field regularization, variational derivatives,
the coupled or staggered solve actually used, irreversibility, and convergence as the mesh is
refined relative to the length scale. The dossier separates regularization error from spatial,
nonlinear-solve, and load-step error.

The key observables are load against imposed displacement, peak load, elastic and fracture
energies, irreversibility defect, and the regularized crack-surface measure. The entry is rejected
if damage decreases at any accepted material point or if the fracture-energy relation fails its
refinement check, fixed in advance, at fixed model length scale. A correct-looking crack path
cannot override either failure.

Variational derivatives, one-dimensional regularization profiles, and energy identities use two
independent oracles. Published notched-plate curves are a separate community comparison unless
licensed experimental data and a material calibration protocol are fixed.

## Dependencies

- an immutable phase-field fracture model with an explicit tension/compression split and
  irreversibility choice;
- accepted nonlinear or staggered solve semantics and load stepping;
- mesh/length-scale checks, and adaptivity only when its transfer preserves irreversibility and
  energy accounting;
- durable displacement, phase, history, and energy trajectories;
- synchronized 2D field and load-path playback.

The first case has one notch and one propagation mode. Branching, XFEM, cohesive zones,
remeshing, and finite-strain fracture are separate pieces of work.

## Accessibility and promotion

The reduced-motion still shows the accepted phase field after initiation, the notch and crack
direction, peak load, energy values, and a link to the evidence. The text alternative explains
that the diffuse band represents a regularized crack and names the length scale.

Promotion requires accepted regularization, derivative, irreversibility, energy, refinement, and
notched-plate evidence; an accepted coupled trajectory; and the common publication check.

## Primary source

C. Miehe, F. Welschinger, and M. Hofacker,
[“Thermodynamically consistent phase-field models of fracture”](https://doi.org/10.1002/nme.2861),
2010. The exact example, split, and irreversibility algorithm are fixed in the scientific
specification before implementation.
