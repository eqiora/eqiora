# Gallery publication rules

The gallery publishes real Eqiora Results. Each entry is either a clearly labelled unverified
product example or an accepted scientific result. The gallery does not store results, track
benchmarks, or decide capability status: executable evidence lives in the registered manifests
under [`verify/`](../../../verify/), case status in the
[benchmark roadmap](../benchmark-roadmap.md), and supported claims in the
[capability matrix](../../capability-matrix.md). The pages in this directory describe what each
entry should show and when it may be published.

Evidence development follows [RFC 0089](../../../rfcs/0089-resume-claim-local-evidence-development.md).
A new entry can ship first as a labelled product example and become an accepted scientific
result once reproducible, independently derived evidence supports its exact claim. Candidate
oracles and falsifiers check the specific quantities behind that claim, not whole files,
generated trees, or broad inventories.

The eight planned entries are:

1. [laminar cylinder wake](cylinder-wake.md);
2. [thin cylindrical-shell collapse](shell-collapse.md);
3. [Turek--Hron FSI3](turek-hron-fsi3.md);
4. [Stokes exact-area dissipation shape optimization](stokes-shape-optimization.md);
5. [three-dimensional Taylor--Green breakdown](taylor-green-3d.md);
6. [notched-plate phase-field fracture](notched-plate-fracture.md);
7. [dam break around an obstacle](dam-break-obstacle.md);
8. [full electric-motor multiphysics](electric-motor.md).

This is the planned publication order; implementation can proceed in parallel wherever the
contracts and code paths do not overlap. Each result builds on earlier ones: transient flow
establishes the time-series path, nonlinear structures follow, and FSI uses both. Shape
optimization reuses the exact geometry, fixed-topology harmonic mesh motion, the steady-Stokes
Result and pressure, reduced differentiation, immutable history, and the shared presentation
code; it needs no force or drag. The remaining entries add independent physics and scale.

## Heavy result production

Start a heavy candidate only when a gallery claim needs production-scale acceptance that ordinary
PR checks cannot provide. Until that candidate is accepted, production-resolution output remains
an unverified example.

Production-resolution solves, refinement campaigns, and full media encodes go through three
separate stages:

1. **PR conformance** runs small meshes, short trajectories, analytic or manufactured witnesses,
   admission mutants, and small renderer fixtures to exercise the normal implementation path
   cheaply. It never accepts the production gallery result.
2. **Exact-head scientific candidates** run the full scientific campaign for one final source
   revision in its declared trusted environment. The claim, independent oracle, tolerances, and
   stop conditions are fixed before the run, and the candidate must be accepted before the entry
   can become result-ready.
3. **Immutable publication projections** read an accepted, immutable Result or trajectory.
   Publication and replay retrieve and verify the exact media bytes; deploying the site never
   solves equations.

The current small fixed-reference FSI gallery case may keep its full 450-frame development build
because it is cheap. It counts as PR conformance, not as a precedent for production cost or as a
scientific candidate.

Each heavy candidate records the exact Model, Geometry, mesh family, correspondence, Realization,
Run, field and result identities, source revision, producer and runtime environment, solver and
library identities, independent oracle and registered evidence IDs, output inventory, and content
digests. The first real consumer also fixes its private delivery record and a durable external
storage location. The repository keeps the case contract, independently owned oracle or
reference inputs, compact observations and comparison summaries, receipts and manifests, digests,
and small conformance fixtures; bulk trajectories and media do not need to enter source history.

What a change invalidates depends on what it affects:

- a change to an equation, lowering, assembly, boundary law, mesh family, time integrator,
  solver-acceptance rule, scientific observable, benchmark, or oracle requires a new full
  candidate on the final affected head;
- a renderer, scene-profile, encoder, or accessibility change may produce new media from the
  unchanged accepted trajectory;
- documentation, site-shell, and unrelated product changes reuse the accepted result and media.

A successful candidate never carries over to another head by assertion. If the heavy environment
cannot run, the claim is narrowed or promotion stops, and the limitation stays visible. A small
conformance pass, a stale bundle, or a `not-selected` result cannot be relabelled as scientific
acceptance. Each scientific slice names the registered cases and invalidation inputs it affects;
there is no path-glob automation.

GitHub Actions, compiler, and workstation caches, Vault copies, and site copies can speed up or
mirror retrieval, but they are never the authoritative Result, media record, or evidence. A
shared runner or durable public artifact format waits until a second real heavy consumer exists
and passes the usual duplication and abstraction-budget review. This page does not choose an
object store, HPC provider, scheduler, archive format, signing scheme, retention policy, scene
graph, remote-execution API, or schedule.

## Publication classes

### Unverified product example

A new gallery entry may ship as an unverified product example when it has:

- a real Model--Plan--Run--Result workflow through the supported product interface;
- focused tests for that workflow, including failure cases;
- a reproducible invocation of the installed package with named inputs, quantities, and units;
- a visible **Unverified product example** label and a statement of what it does not show
  scientifically; and
- a reduced-motion still and a text alternative that convey the result without the video.

Such an example may use a production-resolution Result. It needs no registered case, oracle,
tolerance, falsifier, evidence dossier, candidate campaign, or accepted-result digest, and it
must not use "verified", "validated", or similar language.

### Accepted scientific result

An entry becomes an accepted result when its Model--Realization--Run--Result lineage,
independently derived registered evidence, public claim and non-claims, media integrity records,
and accessibility assets are complete, fixed for the candidate, and reproducible.

Synthetic or incomplete mockups are marked as development previews, which are different from
real product examples. Digests on media files, or a physically plausible look, never turn an
entry into evidence.

The first implementation may keep the delivery record private to the site build. A public
artifact schema is added only when a second real consumer needs it.

## Results and rendering

Videos are rendered from real Result data, never reconstructed from storyboard keyframes. An
accepted result shows only its unchanged accepted quantities with units. An unverified example
may show unit-bearing quantities from its workflow, together with its label and non-claims. The
renderer may interpolate colours, place labels, choose a camera, and build presentation-only
geometry such as deterministic streamlines, but none of that becomes an evidence quantity.

Selecting frames does not create a new solve. The selection records the source state
identities, never interpolates silently across topology changes, and states whether it samples,
decimates, or interpolates in time. Colour and deformation scales stay fixed for a whole video;
per-frame auto-ranging, unlabelled clipping, and unlabelled deformation exaggeration are not
allowed.

The result data stays available independently of the video. A video is a lossy presentation and
never replaces field data, comparison tables, convergence studies, or the reproducible execution
environment.

## Python workflow requirement

A workflow counts as complete only when one checked-in Python script imports the installed
`eqiora` package and uses the same Rust-owned Model, Geometry, Mesh, Realization, Run, Result, or
trajectory contracts as the product and Studio, without recreating missing meaning in Python. A
repository-only Rust executable, a private Studio bridge, or a re-authored demo dataset can help
during development but does not count. Focused Rust and Python tests cover the product path; they
are not scientific evidence, which comes separately from registered cases.

## Video structure

Each entry is a silent 12--18 second video:

1. about two seconds establish geometry, materials, and boundary conditions;
2. about eight to twelve seconds show one primary physical field at a time;
3. one or two key observables stay readable without pausing;
4. the closing frame returns to the poster composition.

Playback time is not physical time. Every video states its physical time, load, or
design-iteration interval and how that maps to playback. A periodic phenomenon may close on a
matching phase; an irreversible one must never run backwards and instead returns through a
labelled cut, fade, or diagram reset.

Every field names its physical quantity and unit. Signed fields use a diverging scale centred on
zero when zero is physically meaningful; nonnegative fields use a perceptually ordered scale.
Camera changes, mesh overlays, and field transitions appear only when the entry needs them.
Additional fields belong on the detailed page, not in the video.

## Delivery and accessibility

Each published entry provides:

- a lossless poster and, when worthwhile, a modern compressed poster;
- one efficient modern video and one widely supported fallback, with correct MIME types;
- intrinsic dimensions, duration, and frame rate, so loading causes no layout shift;
- responsive crops that keep the key observable and its unit;
- a text alternative and a short, ordered description of the visual changes.

Autoplay is off under `prefers-reduced-motion`. The still shown instead includes the geometry,
primary quantity, physical state, key observable, and either the unverified label or the
accepted-result link. Colour is never the only indication of state. Keyboard and screen-reader
users can reach the same details without going through the video controls.

Normal site asset-integrity checks apply to unverified examples but are not scientific evidence.
Accepted-result media keeps its own integrity records.

## Accepted-result dossier

Only accepted scientific results have an evidence dossier. It keeps four kinds of evidence apart
instead of reducing them to one badge:

- analytic or manufactured code verification;
- conservation, balance, and refinement evidence;
- accepted community numerical comparison;
- experimental validation within a stated validity range.

The dossier records the exact problem and source, normalization and sign conventions, lineage
identities, mesh and time-step studies, decisive falsifiers, execution environment, claim and
non-claims, media provenance, and reference-data licensing. Design guidance, such as a shell
knockdown-factor recommendation, is labelled as guidance and never used as an oracle. A numerical
community benchmark such as Turek--Hron FSI3 is never called experimental validation.

Dossiers are created or updated only from independently derived evidence for the specific
claim. An unverified example instead publishes its invocation, quantities, units, the scope of
its focused tests, and what it does not claim scientifically.

## Publication stages

An unverified product example goes through these stages:

1. **contracted** -- its page here fixes what it shows and what it does not claim;
2. **workflow-ready** -- the installed product workflow and its focused tests pass;
3. **result-ready** -- a real Result and the displayed quantities with units exist;
4. **publication-ready** -- the label, non-claims, media, and accessibility assets pass the
   normal site checks; and
5. **published** -- the example is deployed.

Accepted scientific results keep their evidence-ready, result-ready, and publication records. A
new entry joins that track only after all of its acceptance requirements pass.

Reaching one stage does not imply any later stage, and these planned entries do not change any
capability or benchmark status.
