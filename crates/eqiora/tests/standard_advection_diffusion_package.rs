//! Conservative prescribed advection through the ordinary Q1 Plan.
use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use eqiora::artifact::{
    CartesianMeshCellsV2, GeometryMeshCorrespondenceEnvelopeV1, MeshProductionLineageEnvelopeV1,
    ModelEnvelope,
};
use eqiora::compiler::StaticBindingValue;
use eqiora::geometry::GeometryGraph;
use eqiora::package::{
    SourceCompilation, SourceCompilationError, SourceImportContext, compile_source,
};
use eqiora::solver::{
    LinearSolver, REFERENCE_LINEAR_SOLVER, REFERENCE_SOLVER_PROVIDER, SolverPlan,
};
use eqiora_numerics::{
    AuthenticatedCommonMesh, CommonLinearRequest, CommonResult, CommonSolvePolicy,
    CommonSpatialPolicy, ResolvedCommonPlan,
};

const IMPORT: &str = "import Eqiora.Scalar.scalar as scalar;\n";
const SIGNATURE: &str = r#"
public component Main(
    support body: volume(ambient_dimension = 2),
    support left: boundary(parent = body),
    support right: boundary(parent = body),
    support bottom: boundary(parent = body),
    support top: boundary(parent = body)
) {
"#;

fn fixture(
    source: &str,
) -> Result<(SourceCompilation, AuthenticatedCommonMesh), SourceCompilationError> {
    let graph = GeometryGraph::new();
    let rectangle = graph.rectangle([0.0, 1.0], [0.0, 1.0]).unwrap();
    let edges = rectangle.boundaries();
    let geometry = graph
        .build(
            &rectangle,
            &BTreeMap::from([
                ("body".to_owned(), vec![rectangle.region().into()]),
                ("left".to_owned(), vec![edges[0].into()]),
                ("right".to_owned(), vec![edges[1].into()]),
                ("bottom".to_owned(), vec![edges[2].into()]),
                ("top".to_owned(), vec![edges[3].into()]),
            ]),
        )
        .unwrap();
    let body = geometry.entity_set("body").unwrap();
    let bindings = ["body", "left", "right", "bottom", "top"].map(|name| {
        (
            name,
            StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set(name).unwrap(),
                parent: (name != "body").then_some(body),
            },
        )
    });
    let document = compile_source(
        "thermal.eqi",
        source,
        Some("Main"),
        &bindings,
        SourceImportContext::Bundled,
    )?;
    let policy = CartesianMeshCellsV2::new([2, 2]).unwrap();
    let (mesh, correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_cartesian(&geometry, [2, 2])
            .unwrap();
    let production = MeshProductionLineageEnvelopeV1::from_structured_cartesian_v2_resources(
        &policy,
        &geometry,
        &mesh,
        &correspondence,
    )
    .unwrap();
    let owner =
        AuthenticatedCommonMesh::structured_cartesian(geometry, mesh, correspondence, production)
            .unwrap();
    Ok((document, owner))
}

fn resolve_temporal(source: &str, step: Option<f64>) -> Result<ResolvedCommonPlan, String> {
    let (document, owner) = fixture(source).map_err(|error| format!("{error:?}"))?;
    assert!(matches!(document, SourceCompilation::Packaged(_)));
    let model = ModelEnvelope::from_program(document.model().program()).unwrap();
    let linear = CommonLinearRequest::exact(
        SolverPlan::new(
            LinearSolver::BiConjugateGradientStabilized,
            1e-12,
            1e-14,
            NonZeroUsize::new(1000).unwrap(),
        )
        .unwrap()
        .with_reduction(eqiora::solver::ReductionPolicy::Reproducible),
        REFERENCE_SOLVER_PROVIDER,
    )
    .unwrap();
    ResolvedCommonPlan::resolve(
        &model,
        owner,
        CommonSpatialPolicy::Q1,
        CommonSolvePolicy::Linear(linear),
        None,
        step.map(eqiora_numerics::CommonTemporalPolicy::backward_euler)
            .transpose()
            .map_err(|error| error.to_string())?,
        &REFERENCE_LINEAR_SOLVER,
        None,
        None,
        None,
    )
    .map_err(|error| error.to_string())
}

use eqiora_numerics::{CommonState, CommonTrajectory, CommonTransientRunRequest};

fn model(manufactured: bool) -> String {
    let initial = if manufactured { "1" } else { "0" };
    let value = if manufactured {
        "1 + time()/1[s]*(1+coordinate(0)/1[m]+2*coordinate(1)/1[m]+coordinate(0)*coordinate(1)/1[m^2])"
    } else {
        "0"
    };
    let velocity_x = if manufactured {
        "(1+coordinate(0)/1[m]+2*coordinate(1)/1[m])*1[m/s]"
    } else {
        "1[m/s]"
    };
    let velocity_y = if manufactured {
        "(-1+3*coordinate(0)/1[m]+2*coordinate(1)/1[m])*1[m/s]"
    } else {
        "2[m/s]"
    };
    // Independently differentiate u=1+tF and div(beta*u): div(beta)=3/s,
    // curl(beta)=1/s, grad(F)=(1+Y,2+X), Laplacian(F)=0.
    let source = if manufactured {
        "(1+coordinate(0)/1[m]+2*coordinate(1)/1[m]+coordinate(0)*coordinate(1)/1[m^2]+time()/1[s]*((1+coordinate(0)/1[m]+2*coordinate(1)/1[m])*(1+coordinate(1)/1[m])+(-1+3*coordinate(0)/1[m]+2*coordinate(1)/1[m])*(2+coordinate(0)/1[m]))+3*(1+time()/1[s]*(1+coordinate(0)/1[m]+2*coordinate(1)/1[m]+coordinate(0)*coordinate(1)/1[m^2])))*1[1/s]"
    } else {
        "1[1/s]"
    };
    let boundaries = ["left", "right", "bottom", "top"].map(|face| format!("instance {face}_value: scalar.FixedValue2d(body=body, face={face}, potential=u, value={value});")).join("\n");
    format!(
        "{IMPORT}{SIGNATURE}
 state u: 1 on body in h1;
 initial {{u={initial};}}
 instance transport: scalar.TransientAdvectionDiffusion2d(body=body, scalar=u,
 diffusivity=1[m^2/s], velocity_x={velocity_x}, velocity_y={velocity_y}, source={source});
 {boundaries} }}"
    )
}

fn values(plan: &ResolvedCommonPlan, state: &CommonState) -> Vec<(Vec<f64>, f64)> {
    let owner = plan.authenticated_mesh().unwrap();
    let mesh = owner.cartesian_mesh().unwrap().mesh();
    let fields = plan
        .as_spatial()
        .unwrap()
        .state_fields(state)
        .unwrap()
        .collect::<Vec<_>>();
    assert_eq!(fields.len(), 1);
    assert_eq!(fields[0].3.len(), 9);
    fields[0]
        .2
        .iter()
        .zip(fields[0].3)
        .map(|(entity, value)| (mesh.vertex_coordinates(*entity).unwrap(), *value))
        .collect()
}

#[test]
fn prescribed_divergent_rotational_velocity_matches_q1_exact_manufactured_solution() {
    let plan = resolve_temporal(&model(true), Some(0.25)).unwrap();
    let bytes = plan.to_bytes().unwrap();
    let plan = ResolvedCommonPlan::from_bytes(&bytes, &REFERENCE_LINEAR_SOLVER, &[]).unwrap();
    let initial = plan
        .as_spatial()
        .unwrap()
        .initial_state(Some(0.), vec![])
        .unwrap();
    let initial = CommonState::from_bytes(&initial.to_bytes().unwrap(), &plan).unwrap();
    let run =
        CommonTransientRunRequest::from_steps(plan.clone(), initial, 3, vec![1, 2, 3]).unwrap();
    let std::ops::ControlFlow::Continue(outputs) = run
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("three steps")
    };
    let mut previous_error = 0.;
    for (step, (_, output)) in outputs.iter().enumerate() {
        for (point, value) in values(&plan, output.state()) {
            let (x, y) = (point[0], point[1]);
            let expected = 1. + (step + 1) as f64 * 0.25 * (1. + x + 2. * y + x * y);
            if point == [0.5, 0.5] {
                let error = value - expected;
                let residual = (59. / 18.) * error - (4. / 9.) * previous_error;
                assert!(
                    residual.abs() <= 1e-10,
                    "independent centre residual {residual}"
                );
                previous_error = error;
            }
            // Pre-run budget: one interior DOF, A=M/dt+K+C = 59/18;
            // 1e-10 absolute assembled residual implies error <= 18e-10/59.
            // History amplification is 8/59. For these polynomials, all local
            // integrand magnitudes <= 512. A conservative declared allowance
            // of 32768 operations gives gamma_32768*512 < 1.9e-9.
            // Review corrected the earlier 4096 operation undercount; no
            // expected value or acceptance threshold changed. This fixture
            // allowance is not a guarantee for every compiler path; the
            // independent residual gate above is decisive.
            // Include boundary/history arithmetic with a factor of 8:
            // (1e-10+8*1.9e-9)*(18/59)/(1-8/59) < 1e-8.
            assert!(
                (value - expected).abs() <= 1e-8,
                "{point:?}: {value} vs {expected}"
            );
        }
    }
    let restart = CommonTransientRunRequest::from_steps(
        plan.clone(),
        outputs[0].1.state().clone(),
        2,
        vec![2],
    )
    .unwrap();
    let std::ops::ControlFlow::Continue(resumed) = restart
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("restart")
    };
    assert_eq!(resumed[0].1.state(), outputs[2].1.state());
    let result = CommonResult::accept_trajectory(
        0.,
        CommonTrajectory::accept_spatial_transient(run, outputs).unwrap(),
    )
    .unwrap();
    let bytes = result.to_bytes().unwrap();
    assert_eq!(
        CommonResult::from_bytes(&bytes, &plan)
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes
    );
}

#[test]
fn independent_four_cell_diffusion_mass_load_oracle() {
    let plan = resolve_temporal(&model(false), Some(0.25)).unwrap();
    let initial = plan
        .as_spatial()
        .unwrap()
        .initial_state(Some(0.), vec![])
        .unwrap();
    let run = CommonTransientRunRequest::from_steps(plan.clone(), initial, 1, vec![1]).unwrap();
    let std::ops::ControlFlow::Continue(outputs) = run
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("step")
    };
    // Tensor-product Q1 integration: M=1/9,K=8/3,F=1/4,C_ii=0.
    // (M/dt+K)*u=F, hence u=9/112, independently of candidate output.
    for (point, value) in values(&plan, outputs[0].1.state()) {
        let expected = if point == [0.5, 0.5] { 9. / 112. } else { 0. };
        assert!((value - expected).abs() <= 1e-10, "{point:?}: {value}");
        if point == [0.5, 0.5] {
            assert!(((28. / 9.) * value - 0.25).abs() <= 1e-10);
        }
    }
}

#[test]
fn typed_transport_bindings_reject_wrong_units_dimension_and_unknown_velocity() {
    let good = model(false);
    for wrong in [
        good.replace("diffusivity=1[m^2/s]", "diffusivity=1[m/s]"),
        good.replace("velocity_x=1[m/s]", "velocity_x=1[1/s]"),
        good.replace("ambient_dimension = 2", "ambient_dimension = 3"),
    ] {
        let error = fixture(&wrong).expect_err("typed binding");
        assert!(
            error.diagnostics().is_some_and(|d| !d.is_empty()),
            "{error}"
        );
    }
    let nonlinear = good.replace("velocity_x=1[m/s]", "velocity_x=u*1[m/s]");
    assert!(resolve_temporal(&nonlinear, Some(0.25)).is_err());
}

#[test]
fn manufactured_witness_falsifies_missing_divergence_reversal_and_axis_swap() {
    let good = model(true);
    let x = "(1+coordinate(0)/1[m]+2*coordinate(1)/1[m])*1[m/s]";
    let y = "(-1+3*coordinate(0)/1[m]+2*coordinate(1)/1[m])*1[m/s]";
    let wrong_sources = [
        good.replace("+3*(1+time()/1[s]*(", "+0*(1+time()/1[s]*("),
        good.replace(
            &format!("velocity_x={x}, velocity_y={y}"),
            &format!("velocity_x=-({x}), velocity_y=-({y})"),
        ),
        good.replace(
            &format!("velocity_x={x}, velocity_y={y}"),
            &format!("velocity_x={y}, velocity_y={x}"),
        ),
    ];
    for wrong in wrong_sources {
        assert_ne!(wrong, good, "falsifier must alter the model");
        let plan = resolve_temporal(&wrong, Some(0.25)).unwrap();
        let initial = plan
            .as_spatial()
            .unwrap()
            .initial_state(Some(0.), vec![])
            .unwrap();
        let run = CommonTransientRunRequest::from_steps(plan.clone(), initial, 1, vec![1]).unwrap();
        let std::ops::ControlFlow::Continue(output) = run
            .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
            .unwrap()
        else {
            panic!("falsifier step")
        };
        let center = values(&plan, output[0].1.state())
            .into_iter()
            .find(|(p, _)| p == &[0.5, 0.5])
            .unwrap()
            .1;
        // Analytic centre is 1+(1/4)*(11/4)=27/16. A 1e-3 gap is
        // deliberately far above the previously declared 1e-8 budget.
        assert!(
            (center - 27. / 16.).abs() > 1e-3,
            "falsifier must fail the analytic witness"
        );
    }
}
