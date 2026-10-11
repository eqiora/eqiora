# Thin cylindrical-shell collapse

Status: planned; this page does not advance any structural benchmark or claim shell support.
Publish it first as an **Unverified product example**. Promote it to an accepted result only
after its candidate, independent oracle, falsifiers, and acceptance requirements pass under
RFC 0089; evidence work does not hold back the product example.

## What it shows

The nonlinear load path of one thin circular cylinder with an explicit imperfection under axial
compression: mode initiation, peak load, and a limited post-buckling collapse interval. Geometry,
imperfection, constitutive law, end conditions, contact treatment, and load control are all shown
and versioned.

The claim is much narrower than "shell buckling is solved". NASA SP-8007 Rev 2 provides design
context and knockdown guidance, not an exact oracle. Experimental validation is not claimed until
the scientific case fixes one specimen dataset, including measured imperfections, that can be
redistributed or reproducibly acquired.

The first video need not show fracture. It does not infer a universal knockdown factor, general
shell accuracy, or general contact or plasticity from one specimen. If the accepted case has no
self-contact or plasticity, neither appears in the claim or the storyboard.

## Storyboard

| Presentation time | Content |
|---|---|
| 0--2 s | Radius, thickness, length, material, supports, axial load, and an honestly scaled imperfection preview |
| 2--11 s | Fixed camera with displacement magnitude or equivalent plastic strain as the sole primary field; load factor remains visible |
| 11--15 s | Load--end-shortening curve marks the current state, limit point, and accepted post-limit interval |
| 15--18 s | Labelled neutral cut back to the undeformed diagram; the collapse is never reversed |

Any deformation exaggeration is fixed and stated. The camera never moves to hide a competing mode
or a contact event.

## Evidence needed for acceptance

The prerequisite cases cover, separately:

- large-rotation objectivity and finite-strain continuum verification;
- shell patch, membrane, bending, and locking cases;
- eigenvalue buckling against a classical solution;
- limit-point and snap-through continuation;
- contact and plasticity verification, if either appears in the video;
- an imperfection-sensitive cylinder comparison and, separately, experimental validation once
  suitable data are fixed.

The key observable is the load--end-shortening path: linear eigenvalue load, nonlinear peak
load, post-limit branch, and the normalised knockdown ratio. The case is rejected if the
perfect-shell refinement sequence misses its independently fixed classical limit or if the
pre-registered imperfection study does not show the expected direction of sensitivity. Agreement
with a design knockdown band cannot make up for either failure.

Analytic limits and balance identities are derived twice, independently. External tests and the
NASA guidance keep their own uncertainty and applicability and are not turned into expected
values.

## Dependencies

- shell kinematics, sections, orientations, and locking control;
- geometric nonlinearity and an independently verified path-following method;
- imperfection identity and the lineage from parameters to geometry;
- plasticity and self-contact only if the chosen case needs them;
- accepted load-step trajectories, shell-surface result projection, and synchronized curve
  playback.

The scientific specification selects the specimen and its constitutive and contact scope before
implementation. This entry does not justify adding every nonlinear structural feature at once.

## Accessibility and promotion

The reduced-motion still pairs the critical deformed state with the full load--shortening curve
and marks the peak load. The text alternative names where the mode forms and whether yielding or
contact has begun.

Promotion requires every prerequisite case relevant to the chosen specimen, accepted
imperfection and load-path artifacts, a licensed plan for any external data used for
validation, and the common publication check.

## Primary source

NASA,
[SP-8007-2020 Rev 2, “Buckling of Thin-Walled Circular Cylinders”](https://ntrs.nasa.gov/citations/20205011530),
2020. This is design guidance and a map of sources; the validation specimen and data still have
to be chosen.
