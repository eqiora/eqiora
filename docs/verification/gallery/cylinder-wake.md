# Kármán vortex street: product example and verification status

The Reynolds-100 wake is an **Unverified product example**. It runs, produces an alternating
cylinder wake, and is published as a plain-Python script and a static-site walkthrough. It does
not advance the scientific `fluid.flow-past-cylinder` claim.

## The current product example

The sources use the Schäfer--Turek 2D-2 geometry: a
$2.2\,\mathrm{m}\times0.41\,\mathrm{m}$ channel with a cylinder of radius
$0.05\,\mathrm{m}$ centered at $(0.2\,\mathrm{m},0.2\,\mathrm{m})$. The
parabolic inlet has maximum speed $1.5\,\mathrm{m/s}$ and mean speed
$1\,\mathrm{m/s}$; density is $1\,\mathrm{kg/m^3}$ and dynamic viscosity is
$0.001\,\mathrm{Pa\,s}$. With cylinder diameter $D=0.1\,\mathrm{m}$, these
values give $\mathrm{Re}=\rho \bar U D/\mu=100$.

The workflow generates an unstructured triangular Gmsh mesh with a `0.02 m` target size and uses
MINI/P1 in space, Backward Euler with $\Delta t=0.01\,\mathrm{s}$, Newton iteration, and
SparseLU. A steady Stokes solution is the initial state. The default run advances 700 steps to
$t=7\,\mathrm{s}$ in chunks, then another 200 steps to $t=9\,\mathrm{s}$, keeping every second
State. The resulting 100 States cover the 2 s observation window at 0.02 s spacing.

Each retained State keeps its full lineage: Geometry, selection, Mesh, Model, Plan, Trajectory,
Field, unit and support. From State the workflow computes cell-average vorticity and
continuous-P1 pressure samples. Result reconstructs fluid force on the cylinder from the exact
Plan and the selected output’s immediate predecessor/current pair; force is not restart data.
The pressure probes use the benchmark front and rear points $(0.15,0.2)$ and $(0.25,0.2)$. It then derives

$$
C_D=\frac{2F_D}{\rho \bar U^2D},\qquad
C_L=\frac{2F_L}{\rho \bar U^2D},\qquad
\Delta p=p(0.15,0.2)-p(0.25,0.2),
$$

and estimates $\mathrm{St}=D/(\bar U T)$ from the median spacing of rising, mean-centred lift
crossings. The poster, reduced-motion still, and WebM/MP4 video show alternating signed vorticity
from that Result, alongside the sampled lift history.

These observations show that the workflow runs. They do not show agreement with the 2D-2
reference values, a fully developed periodic cycle, or numerical accuracy.

## Target scientific claim

The goal is to show that one exact channel-minus-circle Model runs as transient incompressible
Navier--Stokes flow, produces an alternating laminar wake, and publishes pressure, vorticity,
cylinder force, and time-series observables from one accepted lineage. The target is the
Schäfer--Turek 2D-2 configuration. The steady 2D-1 case and a smooth transient analytic or
manufactured case must be verified first, but neither replaces evidence for the wake itself.

The current example differs from the reference calculation:

- the reference uses $Q_2/P_1^{\mathrm{disc}}$ without stabilization and Crank--Nicolson time
  integration; Eqiora uses triangular MINI/P1 and first-order Backward Euler;
- Eqiora starts from a steady Stokes solution instead of the published zero-state procedure, and
  the 7--9 s window has not been shown to match the reference's fully developed measurement
  window;
- Eqiora imposes the full symmetric Newtonian traction at the outlet, whereas the FeatFlow
  definition writes the do-nothing condition with $\nu\nabla u-pI$.

These differences matter even if sampled values happen to lie near the published ones.

No spatial or temporal refinement study has been run. The `0.02 m` Gmsh target does not bound
the actual edge size, and the cell count does not correspond to a level of the reference $Q_2$
quadrilateral mesh family. The 0.01 s Backward Euler step can add numerical damping. The example
does not show turbulence, production scale, a benchmark-equivalent discretization, grid or
time-step independence, or general curved meshing, and visual similarity is not validation.

## Storyboard

| Presentation time | Content |
|---|---|
| 0--2 s | Exact circle, channel, named inlet/walls/outlet, velocity profile, viscosity, and Reynolds number |
| 2--11 s | Fixed-camera cell-average vorticity over selected States in the 7--9 s observation window |
| 11--15 s | Sampled lift trace and the current frame's shedding phase |
| 15--18 s | Return to the poster frame |

Vorticity is the only primary field. Pressure belongs in the detailed view and the poster
comparison, not in a simultaneous overlay.

## Evidence needed for acceptance

- a smooth transient case such as `fluid.taylor-green` verifies temporal and spatial accuracy
  before the non-box benchmark;
- Schäfer--Turek 2D-1 checks the force and pressure-difference conventions on the same geometry
  family;
- 2D-2 compares the periodic drag, lift, pressure difference, and shedding frequency with the
  accepted community ranges;
- mass balance, time-step refinement, spatial refinement, and the full force-balance defect are
  reported in the dossier.

The key observables are `C_D(t)`, `C_L(t)`, the front-minus-rear pressure difference, and the
Strouhal number in the benchmark normalization. The reference cycle runs from one minimum of
$C_L$ to the next. An accepted claim must show that the sampled window contains a stable,
repeatable cycle and that these values settle as space and time resolution are refined
independently. A convincing vortex street cannot replace coefficient, balance, and refinement
checks against requirements fixed in advance.

Expected community values and tolerances are maintained outside the implementation work.
Convergence and balance fixtures that involve derivations use two independent oracles.

## Dependencies

- non-box transient Navier--Stokes lowering over the accepted exact geometry and its mesh
  correspondence;
- independently derived force-sign and normalization checks on this boundary;
- a mesh family and a time-step family that vary one resolution axis at a time;
- a developed-cycle selector and stable $C_D$, $C_L$, pressure-difference, and Strouhal
  observations;
- mass-balance and force-balance checks for each accepted refinement level.

The first video needs only the renderer profile for this 2D scalar field and trace; it should not
introduce a general visualization schema.

## Accessibility and promotion

The reduced-motion still and text alternative describe alternating signed vorticity without
relying on colour. Numerical labels are marked as sampled, unverified product observations.

Promotion requires accepted smooth-transient, steady-cylinder, and periodic-wake evidence; an
accepted field trajectory with force and vorticity results; and the common publication check.

## Primary source

M. Schäfer and S. Turek,
[“Benchmark Computations of Laminar Flow Around a Cylinder”](https://doi.org/10.1007/978-3-322-89849-4_39),
1996. The maintained
[FeatFlow 2D-2 definition and comparison tables](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html)
record the periodic procedure and reported quantities.
