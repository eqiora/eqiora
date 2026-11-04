//! Existing continuum entrypoints use the ordinary bundled project transport.
use std::collections::BTreeSet;
use std::fs;
use std::num::{NonZeroU16, NonZeroUsize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use eqiora::package::{DirectoryPackageStore, PackagedModelDocument, ResolutionRecordV1};
use eqiora::realization::{
    Discretization, DiscretizationMethod, ExecutionSchedule, MeshPolicy, QuadraturePolicy,
    RealizationCapabilities, RealizationPlan, RealizationRequest, RealizationRequirements,
    RealizationRevision, SemanticRevision, Space, Target, VectorLayoutKind, resolve,
};
use eqiora::sem::KernelProgram;
use eqiora::solver::{LinearSolver, REFERENCE_LINEAR_SOLVER, SolverPlan};
use eqiora_core::ScalarType;
use eqiora_numerics::{
    fluid::lower_steady_incompressible_stokes_cartesian_2d,
    solid::solve_resolved_isotropic_elasticity_cartesian_2d,
};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let parent = PathBuf::from(std::env::var_os("HOME").expect("home-backed test scratch"))
            .join(".cache/eqiora/bundled-entrypoint-tests");
        fs::create_dir_all(&parent).unwrap();
        let path = parent.join(format!(
            "{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn moved_offline(name: &str, source: &str) -> (PackagedModelDocument, ResolutionRecordV1) {
    let scratch = Scratch::new();
    let project = scratch.0.join("project");
    let store = scratch.0.join("store");
    fs::create_dir_all(project.join("src")).unwrap();
    fs::create_dir(&store).unwrap();
    fs::write(project.join("src/main.eqi"), source).unwrap();
    fs::write(
        project.join("eqiora.toml"),
        "[package]\nname = \"org.example.BundledEntry\"\nversion = \"1.0.0\"\nentry = \"main\"\n",
    )
    .unwrap();
    let lock =
        PackagedModelDocument::add_bundled_package_dependency_v1(&project, &store, name, "0.4.0")
            .unwrap();
    let original = PackagedModelDocument::compile_locked(
        &DirectoryPackageStore::open_ambient(&store).unwrap(),
        &lock,
        "Main",
    )
    .unwrap();
    let vendor = project.join("vendor");
    fs::create_dir(&vendor).unwrap();
    assert_eq!(
        PackagedModelDocument::vendor_local_package_project_v1(&project, &store, &vendor).unwrap(),
        lock
    );
    // A second publication keeps the accepted closure rather than replacing it.
    assert_eq!(
        PackagedModelDocument::vendor_local_package_project_v1(&project, &store, &vendor).unwrap(),
        lock
    );
    fs::remove_dir_all(store).unwrap();
    let moved = scratch.0.join("moved");
    fs::rename(project, &moved).unwrap();
    let vendor = moved.join("vendor");
    let reopened = PackagedModelDocument::open_local_package_project_v1(&moved, &vendor).unwrap();
    assert_eq!(reopened, lock);
    let replay = PackagedModelDocument::compile_locked(
        &DirectoryPackageStore::open_ambient(&vendor).unwrap(),
        &reopened,
        "Main",
    )
    .unwrap();
    assert_eq!(
        original.model().structural_fingerprint(),
        replay.model().structural_fingerprint()
    );
    (replay, reopened)
}

const FLUID: &str = r#"
import Eqiora.Fluid.fluid as fluid;
model Main() {
  domain body = box(0, 4, 0, 2);
  domain left = boundary(body, axis=0, side=lower);
  domain right = boundary(body, axis=0, side=upper);
  domain bottom = boundary(body, axis=1, side=lower);
  domain top = boundary(body, axis=1, side=upper);
  variable velocity: vector<m/s,2> on body in smooth;
  variable pressure: Pa on body in h1;
  variable force_potential: Pa on body;
  parameter force_gradient: Pa/m = 1;
  relation force on body { force_potential = force_gradient * coordinate(0); }
  instance governing: fluid.SteadyStokes2d(
    body=body, exterior=boundaries(left,right,bottom,top), velocity=velocity,
    pressure=pressure, force_potential=force_potential, dynamic_viscosity=2[Pa*s]
  );
  instance a: fluid.NoSlip2d(body=body,face=left);
  instance b: fluid.NoSlip2d(body=body,face=right);
  instance c: fluid.NoSlip2d(body=body,face=bottom);
  instance d: fluid.NoSlip2d(body=body,face=top);
  connect governing.mechanical[boundary=left], a.mechanical;
  connect governing.mechanical[boundary=right], b.mechanical;
  connect governing.mechanical[boundary=bottom], c.mechanical;
  connect governing.mechanical[boundary=top], d.mechanical;
}
"#;

#[test]
fn bundled_fluid_reopens_with_only_its_real_mechanics_dependency() {
    let (document, lock) = moved_offline("Eqiora.Fluid", FLUID);
    let names = lock
        .nodes()
        .iter()
        .map(|node| node.identity().name.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        names,
        BTreeSet::from([
            "org.example.BundledEntry",
            "Eqiora.Fluid",
            "Eqiora.Mechanics.Interfaces"
        ])
    );
    let edges = lock
        .edges()
        .iter()
        .map(|edge| (edge.declaring().name.as_str(), edge.target().name.as_str()))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        edges,
        BTreeSet::from([
            ("org.example.BundledEntry", "Eqiora.Fluid"),
            ("Eqiora.Fluid", "Eqiora.Mechanics.Interfaces")
        ])
    );
    let model =
        lower_steady_incompressible_stokes_cartesian_2d(document.model().program()).unwrap();
    assert_eq!(model.dynamic_viscosity(), 2.);
    assert_eq!(
        model
            .force_potential_expression()
            .evaluate(&[0.25, 0.75])
            .unwrap(),
        0.25
    );
}

#[test]
fn bundled_solid_reopens_and_runs_the_existing_nonzero_load_case() {
    // Reuse the established linear-load case, changing only the entrypoint's
    // existing component/argument names. No equation or physical load changes.
    let source = format!(
        "import Eqiora.Solid.solid as solid;\n{}",
        include_str!("../../../verify/solid/packaged-isotropic-balance-2d/models/linear-load.eqi")
            .replace("IsotropicBalanceWithPotential2d", "IsotropicBalance2d")
            .replace("mu = mu", "shear_modulus = mu")
            .replace("lambda = lambda", "first_lame_parameter = lambda")
    );
    let (document, lock) = moved_offline("Eqiora.Solid", &source);
    assert_eq!(
        lock.nodes()
            .iter()
            .map(|node| node.identity().name.as_str())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["org.example.BundledEntry", "Eqiora.Solid"])
    );
    assert_eq!(lock.edges().len(), 1);
    let program = document.model().program();
    let (_, solution) = solve_resolved_isotropic_elasticity_cartesian_2d(
        program,
        &solid_plan(program),
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap();
    assert!(
        solution
            .displacement()
            .values()
            .iter()
            .any(|value| *value != 0.)
    );
    // grad(x)=e_x on the unit square. Keep the established independent force
    // and reaction tolerances from packaged_isotropic_balance_2d.rs unchanged.
    for (actual, expected) in solution.integrated_body_force().into_iter().zip([1., 0.]) {
        assert!((actual - expected).abs() < 3e-14);
    }
    for component in 0..2 {
        assert!(
            (solution.boundary_reaction()[component] + solution.integrated_body_force()[component])
                .abs()
                < 2e-11
        );
    }
}

fn solid_plan(program: &KernelProgram) -> eqiora::realization::ResolvedRealization {
    let plan = RealizationPlan::new(
        Space::continuous_lagrange(NonZeroU16::MIN),
        Discretization::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshPolicy::GeneratedUniform {
                cells_per_axis: NonZeroUsize::new(8).unwrap(),
            },
            QuadraturePolicy::GaussLegendre {
                points_per_axis: NonZeroUsize::new(2).unwrap(),
            },
        ),
        SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1e-12,
            1e-14,
            NonZeroUsize::new(10_000).unwrap(),
        )
        .unwrap(),
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )
    .unwrap();
    resolve(
        &RealizationRequest::explicit(
            program.model(),
            SemanticRevision::new(program.revision().0),
            RealizationRevision::new(1),
            plan,
        ),
        RealizationRequirements::new(
            NonZeroUsize::new(2).unwrap(),
            ScalarType::F64,
            VectorLayoutKind::Replicated,
        ),
        &RealizationCapabilities::isotropic_elasticity_2d_reference(),
    )
    .unwrap()
}
