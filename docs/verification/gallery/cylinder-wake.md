# Kármán vortex-street product and verification boundary

Status: the unverified Reynolds-100 product example executes an alternating
cylinder wake and is available as a plain-Python source plus an
accessible static-site walkthrough. It does not advance the scientific
`fluid.flow-past-cylinder` claim.

## Current unverified product boundary

The checked-in sources use the Schäfer--Turek 2D-2 geometry: a
$2.2\,\mathrm{m}\times0.41\,\mathrm{m}$ channel with a cylinder of radius
$0.05\,\mathrm{m}$ centered at $(0.2\,\mathrm{m},0.2\,\mathrm{m})$. The
parabolic inlet has maximum speed $1.5\,\mathrm{m/s}$ and mean speed
$1\,\mathrm{m/s}$; density is $1\,\mathrm{kg/m^3}$ and dynamic viscosity is
$0.001\,\mathrm{Pa\,s}$. With cylinder diameter $D=0.1\,\mathrm{m}$, these
values give $\mathrm{Re}=\rho \bar U D/\mu=100$.

The product workflow generates an unstructured triangular Gmsh mesh with a
`0.02 m` characteristic target and uses MINI/P1 in space, Backward Euler with
$\Delta t=0.01\,\mathrm{s}$, Newton iteration, and SparseLU. A steady Stokes
result supplies the initial state. The default run advances 700 spin-up steps to
$t=7\,\mathrm{s}$ in bounded chunks, then advances another 200 steps through
$t=9\,\mathrm{s}$ and retains every second accepted State. The resulting 100
States span the 2 s observation window at 0.02 s output spacing.

Each retained State preserves Geometry, selection, Mesh, Model, Plan,
Trajectory, Field, unit, support, and observation lineage. It supplies
cell-average vorticity, the fluid-on-cylinder force, and continuous-P1 pressure
samples at the benchmark front and rear points $(0.15,0.2)$ and $(0.25,0.2)$.
The product derives

$$
C_D=\frac{2F_D}{\rho \bar U^2D},\qquad
C_L=\frac{2F_L}{\rho \bar U^2D},\qquad
\Delta p=p(0.15,0.2)-p(0.25,0.2),
$$

and estimates $\mathrm{St}=D/(\bar U T)$ from the median separation of rising
mean-centered lift crossings. The poster, reduced-motion still, and WebM/MP4
show alternating signed vorticity from the same product Result, accompanied by
the sampled lift history.

This remains an **Unverified product example**. The visible vortex street and
finite $C_D$, $C_L$, pressure-difference, and Strouhal observations demonstrate
the runnable Eqiora workflow. They do not establish agreement with the 2D-2
reference values, a fully developed periodic limit cycle, or numerical accuracy.

## Target experience and future public claim

The film shows that one exact channel-minus-circle Model executes as transient
incompressible Navier--Stokes flow, produces an alternating laminar wake, and
publishes pressure, vorticity, cylinder force, and time-series observables from
one accepted lineage.

The scientific target is the Schäfer--Turek 2D-2 configuration. The steady
2D-1 case and a smooth transient analytic or manufactured case are prerequisite
verification, not substitute evidence for the wake.

The product discretization differs from the reference calculation, which uses
$Q_2/P_1^{\mathrm{disc}}$ without stabilization and Crank--Nicolson time
integration. The Eqiora workflow instead uses triangular MINI/P1 and first-order
Backward Euler. Its steady-Stokes initialization also differs from the published
zero-state development procedure, and the 7--9 s product window has not been
shown equivalent to the reference procedure's fully developed measurement
window. Eqiora imposes the full symmetric Newtonian traction at the outlet,
whereas the maintained FeatFlow definition writes the do-nothing condition with
$\nu\nabla u-pI$. These distinctions remain visible even if sampled observables
happen to lie near published values.

No spatial or temporal refinement was performed for this product run. The
`0.02 m` Gmsh target is not a guaranteed maximum realized edge size and its cell
count is not equivalent to a level of the reference quadrilateral $Q_2$ mesh
family. The 0.01 s Backward Euler step can add numerical damping. The film does
not claim turbulent flow, production scale, benchmark-equivalent discretization,
grid independence, time-step independence, general curved meshing, or validation
from visual similarity.

## Storyboard

| Presentation time | Content |
|---|---|
| 0--2 s | Exact circle, channel, named inlet/walls/outlet, velocity profile, viscosity, and Reynolds number |
| 2--11 s | Fixed-camera cell-average vorticity over selected States in the 7--9 s observation window |
| 11--15 s | Sampled lift trace and the current frame's shedding phase |
| 15--18 s | Return to the poster frame |

Vorticity is the sole primary field. Pressure belongs in the detailed view and
poster comparison, not as a simultaneous overlay.

## Accepted-result evidence plan

Scientific promotion requires distinct evidence for these obligations:

- a smooth transient case such as `fluid.taylor-green` verifies temporal and
  spatial accuracy before the non-box benchmark;
- Schäfer--Turek 2D-1 checks the force and pressure-difference convention on
  the same geometry family;
- 2D-2 compares the periodic drag, lift, pressure difference, and shedding
  frequency with the accepted community ranges;
- mass balance, time-step refinement, spatial refinement, and a complete
  force-balance defect remain visible in the dossier.

The decisive observable family is `C_D(t)`, `C_L(t)`, front-minus-rear pressure
difference, and Strouhal number in the benchmark normalization. The reference
cycle begins at a minimum of $C_L$ and ends at its next minimum. A promoted
claim must show that the sampled window contains a stable repeatable cycle and
that independently varied space and time resolution approach stable values.
A plausible vortex street cannot substitute for coefficient, balance, and
refinement checks that meet their precommitted acceptance requirements.

Expected community values and tolerances are owned outside the implementation
lane. Derivation-bearing convergence and balance fixtures use the dual
independent oracle gate.

## Capability and artifact dependencies

- non-box transient Navier--Stokes lowering over the accepted exact geometry
  and its mesh correspondence;
- independently derived force-sign and normalization checks on this boundary;
- a mesh family and time-step family that vary one resolution axis at a time;
- a developed-cycle selector and stable $C_D$, $C_L$, pressure-difference, and
  Strouhal observations;
- mass-balance and force-balance checks for each accepted refinement member.

The first film may implement only the renderer profile required by this fixed
2D scalar field and trace. It must not introduce a universal visualization
schema.

## Accessibility and promotion

The reduced-motion still and text alternative describe alternating signed
vorticity without relying on color. Numerical labels remain identified as
sampled unverified product observations.

Promotion requires accepted smooth-transient, steady-cylinder, and periodic-
wake evidence; an accepted field trajectory with force and vorticity results;
and the common publication admission check.

## Primary source

M. Schäfer and S. Turek,
[“Benchmark Computations of Laminar Flow Around a Cylinder”](https://doi.org/10.1007/978-3-322-89849-4_39),
1996. The maintained
[FeatFlow 2D-2 definition and comparison tables](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html)
record the periodic procedure and reported quantities.
