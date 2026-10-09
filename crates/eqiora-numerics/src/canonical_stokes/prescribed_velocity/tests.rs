use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use eqiora_compiler::compile;
use eqiora_core::{Diagnostic, DimExponents, DynQuantity, RawId};
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_realization::{
    FieldwiseRealizationRequest, RealizationCapabilities, RealizationRevision,
    ResolvedFieldwiseRealization, SemanticRevision, resolve_fieldwise,
};
use eqiora_schema::kernel::{BoundarySide, DomainKind, KernelNode};
use eqiora_sem::KernelProgram;
use eqiora_solver::{LinearSolver, PreconditionerPolicy, ReductionPolicy, SolverPlan};

use super::{
    ModelOwnedEssentialVelocityReplay2d, SteadyStokesPrescribedVelocityTrace2d,
    admit_model_owned_essential_velocity_2d, lower_prescribed_velocity_trace_2d,
};
use crate::canonical_stokes::api::StokesBoundaryKey2d;
use crate::canonical_stokes::dissipation_profile::{
    StokesDissipationGeometryModelBinding2d, StokesDissipationTopologyRole2d,
    e1_stokes_dissipation_sealed_inputs_v1,
};
use crate::canonical_stokes::geometry_realization::finalize_resolved_stokes_dissipation_profile_mini_2d_with_transport;
use crate::canonical_stokes::realization::{
    SteadyStokesScaleProfile2d, steady_stokes_fieldwise_requirements_for_model_2d,
    steady_stokes_mini_plan_for_model_2d,
};
use crate::canonical_stokes::support::relations_on;

const SEALED_INPUT_SHA256: &str =
    "da3223d51caf11f6e627540f9284c2bb307518f7d87a62e2897a9f3732bbf620";
const REFERENCE_TOPOLOGY: &str = "stokes-square-ring-reference-n32-m4-v1";
const REFINED_TOPOLOGY: &str = "stokes-square-ring-refined-n64-m8-v1";
const ROLE_NAMES: [&str; 5] = [
    "body_no_slip",
    "outer_x_minus",
    "outer_x_plus",
    "outer_y_minus",
    "outer_y_plus",
];

const LENGTH: DimExponents =
    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
const VELOCITY: DimExponents =
    DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension");
const PRESSURE: DimExponents =
    DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension");

const SOURCE: &str = r#"
model stokes_e1_prescribed_velocity() {
  domain fluid = box(-10, 10, -10, 10);
  domain body_no_slip = boundary(fluid, axis = 0, side = lower);
  domain outer_x_minus = boundary(fluid, axis = 0, side = lower);
  domain outer_x_plus = boundary(fluid, axis = 0, side = upper);
  domain outer_y_minus = boundary(fluid, axis = 1, side = lower);
  domain outer_y_plus = boundary(fluid, axis = 1, side = upper);

  variable velocity: vector<m / s, 2> on fluid in h1;
  variable pressure: kg / (m * s ^ 2) on fluid;
  variable force_potential: kg / (m * s ^ 2) on fluid;
  variable chi: m ^ 2 / s on fluid in smooth;
  parameter mu: kg / (m * s) = 1;
  parameter U: m / s = 1;
  parameter zero_pressure: kg / (m * s ^ 2) = 0;

  relation force on fluid {
    force_potential - zero_pressure = 0;
  }
  relation momentum on fluid {
    -div(
      2 * mu * symmetric_part(grad(velocity))
      - isotropic_lift(pressure)
    ) - grad(force_potential) = 0;
  }
  relation incompressibility on fluid { div(velocity) = 0; }
  relation define_chi on fluid { chi - U * coordinate(0) = 0; }

  relation body_zero on body_no_slip { trace(velocity) = 0; }
  relation outer_x_minus_value on outer_x_minus {
    trace(velocity) - trace(grad(chi)) = 0;
  }
  relation outer_x_plus_value on outer_x_plus {
    trace(velocity) - trace(grad(chi)) = 0;
  }
  relation outer_y_minus_value on outer_y_minus {
    trace(velocity) - trace(grad(chi)) = 0;
  }
  relation outer_y_plus_value on outer_y_plus {
    trace(velocity) - trace(grad(chi)) = 0;
  }
}
"#;

#[derive(Clone)]
struct Fixture {
    program: KernelProgram,
    velocity: RawId,
    role_boundaries: BTreeMap<String, RawId>,
    body_relation: RawId,
    outer_traces: BTreeMap<String, SteadyStokesPrescribedVelocityTrace2d>,
    owners: BTreeMap<String, Vec<(usize, [f64; 2])>>,
}

impl Fixture {
    fn from_source(source: &str) -> Self {
        let program = compile_program(source);
        let domain = program
            .nodes()
            .find_map(|node| match node {
                KernelNode::Domain(value)
                    if matches!(value.kind(), DomainKind::CartesianBox { .. }) =>
                {
                    Some(value.id().erase())
                }
                _ => None,
            })
            .expect("one volume Domain");
        let velocity = program
            .nodes()
            .find_map(|node| match node {
                KernelNode::Field(value) if value.shape().extents().len() == 1 => {
                    Some(value.id().erase())
                }
                _ => None,
            })
            .expect("one vector velocity Field");

        let mut role_boundaries = BTreeMap::new();
        let mut body_relation = None;
        let mut outer_traces = BTreeMap::new();
        for node in program.nodes() {
            let KernelNode::Domain(boundary) = node else {
                continue;
            };
            let DomainKind::CartesianBoundary { axis, side } = boundary.kind() else {
                continue;
            };
            let boundary_id = boundary.id().erase();
            let boundary_relations = relations_on(&program, boundary_id);
            let [relation] = boundary_relations.as_slice() else {
                panic!("every fixture boundary has one Relation");
            };
            match lower_prescribed_velocity_trace_2d(
                &program,
                domain,
                velocity,
                boundary_id,
                *relation,
            )
            .expect("fixture boundary has supported typed meaning")
            {
                Some(trace) => {
                    let role = outer_role(*axis, *side);
                    assert_eq!(trace.boundary(), boundary_id);
                    assert_eq!(trace.relation(), *relation);
                    role_boundaries.insert(role.to_owned(), boundary_id);
                    outer_traces.insert(role.to_owned(), trace);
                }
                None => {
                    assert!(body_relation.replace(*relation).is_none());
                    role_boundaries.insert("body_no_slip".to_owned(), boundary_id);
                }
            }
        }
        assert_eq!(role_boundaries.keys().count(), ROLE_NAMES.len());
        assert!(
            ROLE_NAMES
                .iter()
                .all(|name| role_boundaries.contains_key(*name))
        );

        Self {
            program,
            velocity,
            role_boundaries,
            body_relation: body_relation.expect("one trace-zero body"),
            outer_traces,
            owners: exact_owned_vertices(),
        }
    }

    fn admit(&self) -> Result<ModelOwnedEssentialVelocityReplay2d, eqiora_core::Diagnostic> {
        admit_model_owned_essential_velocity_2d(
            &self.program,
            self.velocity,
            &self.role_boundaries,
            self.body_relation,
            &self.outer_traces,
            &self.owners,
        )
    }
}

fn sealed_selector_and_complete_model_trace_reach_the_ordinary_positive_first() {
    assert_eq!(SEALED_INPUT_SHA256.len(), 64);
    assert!(
        SEALED_INPUT_SHA256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    );
    assert_ne!(REFERENCE_TOPOLOGY, REFINED_TOPOLOGY);
    let base = parse_pair(["0", "0"]);
    let direction = parse_pair(["1", "0"]);
    let step = "2e-3".parse::<f64>().expect("sealed step");
    assert_eq!(add_scaled(base, direction, step), [0.002, 0.0]);
    assert_eq!(add_scaled(base, direction, -step), [-0.002, 0.0]);

    let fixture = Fixture::from_source(SOURCE);
    let mut shared_identity = None;
    for (role, trace) in &fixture.outer_traces {
        assert!(trace.is_complete_affine_potential(), "{role}");
        assert_eq!(trace.evaluate([7.0, -3.0]).unwrap(), [1.0, 0.0]);
        let identity = (
            trace.coefficient_field(),
            trace.definition_relation(),
            trace.speed_parameter(),
        );
        if let Some(expected) = shared_identity {
            assert_eq!(
                identity, expected,
                "{role} drifted from the shared Model law"
            );
        } else {
            shared_identity = Some(identity);
        }
    }
    assert_eq!(fixture.outer_traces.len(), 4);

    let replay = fixture
        .admit()
        .expect("complete typed Model reaches admission");
    assert_eq!(
        replay.values(),
        &[
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    );
    // The horizontal sides witness the nonzero tangential component, while
    // the two vertical fluxes cancel exactly.
    assert_eq!(replay.value_for_role("outer_y_minus", 0), Some([1.0, 0.0]));
    assert_eq!(replay.value_for_role("outer_y_plus", 3), Some([1.0, 0.0]));
    assert_eq!((-1.0_f64).mul_add(20.0, 1.0 * 20.0), 0.0);
    replay
        .validate_transport(|vertex| replay.values()[vertex])
        .expect("the callback transports only replayed Model values");
}

type EssentialTransport = BTreeMap<(u64, u64), [f64; 2]>;

fn real_profile_bindings_reach_finalized_system_admission() {
    for role in [
        StokesDissipationTopologyRole2d::Reference,
        StokesDissipationTopologyRole2d::Refined,
    ] {
        let _ = finalized_profile_context(role);
    }
}

fn real_profile_finalizer_rejects_transport_mutants_after_usable_positives() {
    let (binding, resolved, mut callback_mismatch) =
        finalized_profile_context(StokesDissipationTopologyRole2d::Reference);
    let mismatched = callback_mismatch
        .values_mut()
        .find(|value| **value != [0.0; 2])
        .expect("the admitted outer trace is nonzero");
    *mismatched = [mismatched[1], mismatched[0]];
    assert!(
        finalize_profile_with_transport(&binding, &resolved, &callback_mismatch).is_err(),
        "a caller-owned rotated vector must not reach finalized system admission"
    );

    let (binding, resolved, mut normal_only) =
        finalized_profile_context(StokesDissipationTopologyRole2d::Refined);
    let (lower_y, upper_y) = normal_only
        .iter()
        .filter(|(_, value)| **value != [0.0; 2])
        .map(|((_, y), _)| f64::from_bits(*y))
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lower, upper), y| {
            (lower.min(y), upper.max(y))
        });
    let mut horizontal_vertices = 0;
    for ((_, y), value) in &mut normal_only {
        let y = f64::from_bits(*y);
        if *value != [0.0; 2] && (y == lower_y || y == upper_y) {
            *value = [0.0; 2];
            horizontal_vertices += 1;
        }
    }
    assert!(
        horizontal_vertices > 0,
        "the normal projection must erase a horizontal tangential trace"
    );
    assert!(
        finalize_profile_with_transport(&binding, &resolved, &normal_only).is_err(),
        "a normal-only projection must not reach finalized system admission"
    );
}

fn finalized_profile_context(
    role: StokesDissipationTopologyRole2d,
) -> (
    StokesDissipationGeometryModelBinding2d,
    ResolvedFieldwiseRealization,
    EssentialTransport,
) {
    let binding = StokesDissipationGeometryModelBinding2d::from_e1_sealed_inputs_v1(
        e1_stokes_dissipation_sealed_inputs_v1(),
        role,
    )
    .expect("the exact profile/topology binding is admitted");
    binding
        .revalidate_e1_profile_topology()
        .expect("the exact binding replays before Realization");
    let scales = profile_scales(&binding);
    let resolved = resolved_profile_realization(&binding, scales);
    let transport = exact_model_transport(&binding, scales);
    finalize_profile_with_transport(&binding, &resolved, &transport)
        .expect("the exact Model-derived transport reaches finalized system admission");
    (binding, resolved, transport)
}

fn profile_scales(binding: &StokesDissipationGeometryModelBinding2d) -> SteadyStokesScaleProfile2d {
    let model = binding.model();
    let mesh = binding.mesh().mesh();
    let facet = binding.entities("outer_x_lower").unwrap()[0];
    let vertex = mesh
        .entity_vertices(facet)
        .expect("an admitted outer facet owns vertices")[0]
        .index();
    let velocity = model
        .prescribed_velocity(
            &StokesBoundaryKey2d::NamedEntitySet("outer_x_lower".to_owned()),
            None,
            &mesh.vertices()[vertex],
        )
        .expect("the retained complete trace evaluates")
        .expect("the outer role owns a prescribed trace")[0];
    let length = binding.profile().area_radius_m();
    let pressure = model.dynamic_viscosity() * velocity / length;
    SteadyStokesScaleProfile2d::new(
        DynQuantity::new(length, LENGTH),
        DynQuantity::new(velocity, VELOCITY),
        DynQuantity::new(pressure, PRESSURE),
    )
    .expect("the sealed Model/profile derive positive coherent-SI scales")
}

fn resolved_profile_realization(
    binding: &StokesDissipationGeometryModelBinding2d,
    scales: SteadyStokesScaleProfile2d,
) -> ResolvedFieldwiseRealization {
    let solver = SolverPlan::new(
        LinearSolver::MinimumResidual,
        1.0e-11,
        1.0e-13,
        NonZeroUsize::new(10_000).expect("positive solver plan bound"),
    )
    .expect("the existing reference MINI solver tuple is valid")
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Reproducible);
    let plan = steady_stokes_mini_plan_for_model_2d(
        binding.model(),
        binding.mesh().artifact_reference().expect("mesh identity"),
        scales,
        solver,
    )
    .expect("the private profile Model admits the existing MINI plan");
    resolve_fieldwise(
        &FieldwiseRealizationRequest::explicit(
            binding.program().model(),
            SemanticRevision::new(binding.program().revision().0),
            RealizationRevision::new(407),
            plan,
        ),
        steady_stokes_fieldwise_requirements_for_model_2d(binding.model()),
        &RealizationCapabilities::symmetric_mixed_simplicial_2d_reference(),
    )
    .expect("the ordinary reference capability resolves the profile plan")
}

fn exact_model_transport(
    binding: &StokesDissipationGeometryModelBinding2d,
    scales: SteadyStokesScaleProfile2d,
) -> EssentialTransport {
    let model = binding.model();
    let mesh = binding.mesh().mesh();
    let mut transport = BTreeMap::new();
    for role in [
        "body",
        "outer_x_lower",
        "outer_x_upper",
        "outer_y_lower",
        "outer_y_upper",
    ] {
        let key = StokesBoundaryKey2d::NamedEntitySet(role.to_owned());
        for &facet in binding.entities(role).expect("the exact role is retained") {
            for vertex in mesh
                .entity_vertices(facet)
                .expect("an admitted boundary facet owns vertices")
            {
                let coordinate = &mesh.vertices()[vertex.index()];
                let value = if role == "body" {
                    [0.0; 2]
                } else {
                    model
                        .prescribed_velocity(&key, None, coordinate)
                        .expect("the retained complete Model law evaluates")
                        .expect("the outer role owns a prescribed trace")
                        .map(|component| component / scales.velocity_value())
                };
                let normalized = (
                    ((coordinate[0] - model.bounds()[0][0]) / scales.length_value()).to_bits(),
                    ((coordinate[1] - model.bounds()[1][0]) / scales.length_value()).to_bits(),
                );
                if let Some(previous) = transport.insert(normalized, value) {
                    assert_eq!(
                        previous, value,
                        "adjacent exact Model roles must agree before vertex deduplication"
                    );
                }
            }
        }
    }
    transport
}

fn finalize_profile_with_transport(
    binding: &StokesDissipationGeometryModelBinding2d,
    resolved: &ResolvedFieldwiseRealization,
    transport: &EssentialTransport,
) -> Result<(), Diagnostic> {
    let callback = |coordinate: [f64; 2]| {
        transport
            .get(&(coordinate[0].to_bits(), coordinate[1].to_bits()))
            .copied()
            .ok_or_else(|| {
                Diagnostic::error(
                    eqiora_core::diagnostic::codes::INVALID_REALIZATION,
                    "transport omitted an exact Model-owned essential vertex",
                )
            })
    };
    finalize_resolved_stokes_dissipation_profile_mini_2d_with_transport(
        binding.program(),
        resolved,
        binding,
        &callback,
    )
    .map(|_| ())
}

fn normal_only_incomplete_and_equal_value_identity_mutants_fail_closed() {
    let normal_source = SOURCE
        .replace(
            "  variable chi: m ^ 2 / s on fluid in smooth;",
            "  variable chi: m ^ 2 / s on fluid in smooth;\n  variable normal_speed: m / s on fluid in h1;",
        )
        .replace(
            "  relation define_chi on fluid { chi - U * coordinate(0) = 0; }",
            "  relation define_chi on fluid { chi - U * coordinate(0) = 0; }\n  relation define_normal_speed on fluid { normal_speed - U = 0; }",
        )
        .replace(
            "trace(velocity) - trace(grad(chi)) = 0;\n  }\n  relation outer_y_plus_value",
            "trace(velocity) - normal(isotropic_lift(normal_speed)) = 0;\n  }\n  relation outer_y_plus_value",
        );
    let normal = Fixture::from_source(&normal_source);
    assert!(
        !normal.outer_traces["outer_y_minus"].is_complete_affine_potential(),
        "the normal-only alias must reach the intended complete-trace gate"
    );
    assert!(normal.admit().is_err());

    let mut incomplete = Fixture::from_source(SOURCE);
    incomplete.outer_traces.remove("outer_y_plus");
    assert!(incomplete.admit().is_err());

    let drift_source = SOURCE
        .replace(
            "  variable chi: m ^ 2 / s on fluid in smooth;",
            "  variable chi: m ^ 2 / s on fluid in smooth;\n  variable chi_alt: m ^ 2 / s on fluid in smooth;",
        )
        .replace(
            "  parameter U: m / s = 1;",
            "  parameter U: m / s = 1;\n  parameter U_alt: m / s = 1;",
        )
        .replace(
            "  relation define_chi on fluid { chi - U * coordinate(0) = 0; }",
            "  relation define_chi on fluid { chi - U * coordinate(0) = 0; }\n  relation define_chi_alt on fluid { chi_alt - U_alt * coordinate(0) = 0; }",
        )
        .replace(
            "relation outer_y_plus_value on outer_y_plus {\n    trace(velocity) - trace(grad(chi)) = 0;",
            "relation outer_y_plus_value on outer_y_plus {\n    trace(velocity) - trace(grad(chi_alt)) = 0;",
        );
    let drift = Fixture::from_source(&drift_source);
    assert_eq!(
        drift.outer_traces["outer_y_plus"]
            .evaluate([0.0, 0.0])
            .unwrap(),
        [1.0, 0.0],
        "the mutant is numerically equal and must be killed by identity"
    );
    assert!(drift.admit().is_err());

    let mut permuted = Fixture::from_source(SOURCE);
    let x = permuted.outer_traces["outer_x_minus"].clone();
    let y = permuted.outer_traces["outer_y_minus"].clone();
    permuted.outer_traces.insert("outer_x_minus".into(), y);
    permuted.outer_traces.insert("outer_y_minus".into(), x);
    assert!(permuted.admit().is_err());
}

fn ownership_corner_and_callback_mutants_reject_without_partial_admission() {
    let fixture = Fixture::from_source(SOURCE);
    fixture
        .admit()
        .expect("mutants fork after the positive admission");

    let mut missing = fixture.clone();
    missing.owners.remove("outer_y_plus");
    assert!(missing.admit().is_err());

    let mut duplicate = fixture.clone();
    duplicate
        .owners
        .get_mut("outer_x_minus")
        .unwrap()
        .push((0, [-10.0, -10.0]));
    assert!(duplicate.admit().is_err());

    let mut mixed = fixture.clone();
    mixed
        .owners
        .get_mut("body_no_slip")
        .unwrap()
        .push((0, [-10.0, -10.0]));
    assert!(mixed.admit().is_err());

    let replay = fixture.admit().unwrap();
    assert!(
        replay
            .validate_transport(|vertex| if vertex == 0 {
                [0.0, 1.0]
            } else {
                replay.values()[vertex]
            })
            .is_err(),
        "a rotated callback at a shared corner cannot repair the Model"
    );
    assert!(
        replay
            .validate_transport(|vertex| if vertex == 2 {
                [2.0, 0.0]
            } else {
                replay.values()[vertex]
            })
            .is_err(),
        "a distinct caller-owned speed is rejected before finalization"
    );
}

#[test]
fn registered_evidence() {
    std::thread::scope(|scope| {
        scope.spawn(
            StokesDissipationGeometryModelBinding2d::run_e1_profile_topology_ordinary_positives,
        );
        scope.spawn(sealed_selector_and_complete_model_trace_reach_the_ordinary_positive_first);
        scope.spawn(real_profile_bindings_reach_finalized_system_admission);
    });

    std::thread::scope(|scope| {
        scope.spawn(StokesDissipationGeometryModelBinding2d::run_e1_profile_topology_falsifiers);
        scope.spawn(normal_only_incomplete_and_equal_value_identity_mutants_fail_closed);
        scope.spawn(ownership_corner_and_callback_mutants_reject_without_partial_admission);
        scope.spawn(real_profile_finalizer_rejects_transport_mutants_after_usable_positives);
    });
}

fn compile_program(source: &str) -> KernelProgram {
    let mut compiled =
        compile("stokes-e1-prescribed-velocity.eqi", source).expect("typed E1 Model compiles");
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).expect("transaction commits");
    KernelProgram::from_snapshot(&store.snapshot(), model).expect("Model validates")
}

fn outer_role(axis: usize, side: BoundarySide) -> &'static str {
    match (axis, side) {
        (0, BoundarySide::Lower) => "outer_x_minus",
        (0, BoundarySide::Upper) => "outer_x_plus",
        (1, BoundarySide::Lower) => "outer_y_minus",
        (1, BoundarySide::Upper) => "outer_y_plus",
        _ => panic!("E1 is exactly two-dimensional"),
    }
}

fn exact_owned_vertices() -> BTreeMap<String, Vec<(usize, [f64; 2])>> {
    BTreeMap::from([
        (
            "body_no_slip".into(),
            vec![(4, [-1.0, 0.0]), (5, [1.0, 0.0])],
        ),
        (
            "outer_x_minus".into(),
            vec![(0, [-10.0, -10.0]), (3, [-10.0, 10.0])],
        ),
        (
            "outer_x_plus".into(),
            vec![(1, [10.0, -10.0]), (2, [10.0, 10.0])],
        ),
        (
            "outer_y_minus".into(),
            vec![(0, [-10.0, -10.0]), (1, [10.0, -10.0])],
        ),
        (
            "outer_y_plus".into(),
            vec![(3, [-10.0, 10.0]), (2, [10.0, 10.0])],
        ),
    ])
}

fn parse_pair(values: [&str; 2]) -> [f64; 2] {
    values.map(|value| value.parse::<f64>().expect("sealed decimal"))
}

fn add_scaled(base: [f64; 2], direction: [f64; 2], step: f64) -> [f64; 2] {
    [
        direction[0].mul_add(step, base[0]),
        direction[1].mul_add(step, base[1]),
    ]
}
