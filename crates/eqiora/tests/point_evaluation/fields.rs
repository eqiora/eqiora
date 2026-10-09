//! Affine values, first derivatives and deliberate Q1 derivative jumps through Result replay.
use super::*;
use eqiora::artifact::{
    CartesianMeshCellsV2, GeometryMeshCorrespondenceEnvelopeV1, MeshProductionLineageEnvelopeV1,
    ModelEnvelope,
};
use eqiora::geometry::GeometryGraph;
use eqiora::kernel::{KernelNode, ObservableReduction};
use eqiora::solver::{
    LinearSolver, REFERENCE_LINEAR_SOLVER, REFERENCE_SOLVER_PROVIDER, SolverPlan,
};
use eqiora_numerics::{
    AuthenticatedCommonMesh, CommonLinearPlan, CommonLinearRequest, CommonResult,
    CommonSolvePolicy, CommonSpatialPolicy, ResolvedCommonPlan,
};
use std::{collections::BTreeMap, num::NonZeroUsize};

const HEAT: &str = r#"
public component HeatedBody(
  support body: volume(ambient_dimension = 1),
  support left: boundary(parent = body),
  support right: boundary(parent = body)
) {
  variable temperature: K on body;
  parameter capacity: J / (K * m) = 3;
  parameter conductivity: W * m / K = 2;
  relation balance on body { -div(grad(temperature)) = 12[K/m^2]; }
  relation left_value on left { trace(temperature) = 300[K]; }
  relation right_value on right { trace(temperature) = 300[K]; }
  observable energy: J = integral(capacity * (temperature - 300[K]), measure(body));
  observable constant: K*m = integral(7[K], measure(body));
  observable left_flux: W = integral(normal(-conductivity * grad(temperature)), measure(left));
  observable right_flux: W = integral(normal(-conductivity * grad(temperature)), measure(right));
}
"#;

fn heat(source: &str) -> (ModelDocument, CommonLinearPlan, CommonResult) {
    heat_with_spatial(source, CommonSpatialPolicy::Q1)
}

fn heat_with_spatial(
    source: &str,
    spatial: CommonSpatialPolicy,
) -> (ModelDocument, CommonLinearPlan, CommonResult) {
    let graph = GeometryGraph::new();
    let interval = graph.interval([0.0, 1.0]).unwrap();
    let boundaries = interval.boundaries();
    let geometry = graph
        .build(
            &interval,
            &BTreeMap::from([
                ("body".to_owned(), vec![interval.region().into()]),
                ("left".to_owned(), vec![boundaries[0].into()]),
                ("right".to_owned(), vec![boundaries[1].into()]),
            ]),
        )
        .unwrap();
    let body = geometry.entity_set("body").unwrap();
    let bindings = [
        ("body", body, None),
        ("left", geometry.entity_set("left").unwrap(), Some(body)),
        ("right", geometry.entity_set("right").unwrap(), Some(body)),
    ]
    .map(|(name, selection, parent)| {
        (
            name,
            StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection,
                parent,
            },
        )
    });
    let document =
        ModelDocument::compile_selected("point-field.eqi", source, "HeatedBody", &bindings)
            .unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let policy = CartesianMeshCellsV2::new([4]).unwrap();
    let (mesh, correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_cartesian_box_v1(&geometry, policy.cells())
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
    let linear = CommonLinearRequest::exact(
        SolverPlan::new(
            if spatial == CommonSpatialPolicy::Q1 {
                LinearSolver::BiConjugateGradientStabilized
            } else {
                LinearSolver::ConjugateGradient
            },
            1e-12,
            1e-14,
            NonZeroUsize::new(100).unwrap(),
        )
        .unwrap()
        .with_reduction(eqiora::solver::ReductionPolicy::Reproducible),
        REFERENCE_SOLVER_PROVIDER,
    )
    .unwrap();
    let plan = eqiora_numerics::ResolvedCommonPlan::resolve(
        &model,
        owner,
        spatial,
        CommonSolvePolicy::Linear(linear),
        None,
        None,
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap()
    .as_linear()
    .unwrap()
    .clone();
    let result = plan.run_result(&REFERENCE_LINEAR_SOLVER).unwrap();
    (document, plan, result)
}

#[test]
fn point_observables_reconstruct_off_node_affine_q1_values() {
    let source = HEAT
        .replace("12[K/m^2]", "0[K/m^2]")
        .replace("relation right_value on right { trace(temperature) = 300[K]; }", "relation right_value on right { trace(temperature) = 302[K]; }")
        .replace("  observable energy:", "  coordinate x:m on body from body[0];\n  observable profile:K on body=temperature;\n  observable first:K=evaluate(profile,at=(x=0.125[m]));\n  observable second:K=evaluate(profile,at=(x=0.625[m]));\n  observable slope:K/m=evaluate(partial(temperature,wrt=x),at=(x=0.125[m]));\n  observable combined:K=first+second;\n  observable energy:");
    let (document, plan, result) = heat(&source);
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let replay = CommonResult::from_bytes(
        &result.to_bytes().unwrap(),
        &ResolvedCommonPlan::Linear(Box::new(plan)),
    )
    .unwrap();
    // T(x)=300+2*x and dT/dx=2, independently of the retained coefficients.
    for (name, expected, dimension) in [
        ("first", 300.25, [0, 0, 0, 0, 1, 0, 0]),
        ("second", 301.25, [0, 0, 0, 0, 1, 0, 0]),
        ("combined", 601.5, [0, 0, 0, 0, 1, 0, 0]),
        ("slope", 2.0, [0, -1, 0, 0, 1, 0, 0]),
    ] {
        let id = document.aliases()[&format!("definition.{name}")]
            .downcast()
            .unwrap();
        let observed = replay
            .observe(&model, id, &std::collections::HashMap::new())
            .unwrap();
        let value = observed.value().real_scalar_value().unwrap();
        assert_eq!(value.dim(), DimExponents::from_integers(dimension).unwrap());
        assert!(
            (value.value() - expected).abs() < 1e-11,
            "{name}: {value:?} != {expected}"
        );
    }
}

#[test]
fn point_observables_reconstruct_coordinate_slopes_and_require_interface_sides() {
    // -T''=12 and T(0)=T(1)=300 give nodal T=300+6*x*(1-x).
    // On the four uniform Q1 cells the slopes are 4.5, 1.5, -1.5, -4.5 K/m.
    // At x=1/2 the reconstruction therefore has distinct one-sided slopes.
    for (expression, expected) in [
        (
            "evaluate(partial(temperature,wrt=x),at=(x=0.125[m]))",
            Some(4.5),
        ),
        (
            "evaluate(partial(temperature,wrt=x),at=(x=0.5[m]),side=lower)",
            Some(1.5),
        ),
        (
            "evaluate(partial(temperature,wrt=x),at=(x=0.5[m]),side=upper)",
            Some(-1.5),
        ),
        ("evaluate(partial(temperature,wrt=x),at=(x=0.5[m]))", None),
    ] {
        let source = HEAT.replace("  observable energy:", &format!(
            "  coordinate x:m on body from body[0];\n  observable slope:K/m={expression};\n  observable energy:"
        ));
        let (document, _, result) = heat(&source);
        let id = document
            .program()
            .nodes()
            .find_map(|node| match node {
                KernelNode::Observable(definition)
                    if definition.reduction() == ObservableReduction::Value =>
                {
                    Some(definition.id())
                }
                _ => None,
            })
            .unwrap();
        let model = ModelEnvelope::from_program(document.program()).unwrap();
        let observed = result.observe(&model, id, &std::collections::HashMap::new());
        if let Some(expected) = expected {
            let value = observed.unwrap();
            let value = value.value().real_scalar_value().unwrap();
            assert_eq!(
                value.dim(),
                eqiora_core::DimExponents::from_integers([0, -1, 0, 0, 1, 0, 0]).unwrap()
            );
            assert!(
                (value.value() - expected).abs() < 1e-10,
                "{expression}: {value:?} != {expected}"
            );
        } else {
            assert!(observed.unwrap_err().message().contains("explicit side"));
        }
    }
}

#[test]
fn point_observables_reject_unavailable_reconstruction() {
    let source = HEAT.replace("  observable energy:", "  coordinate x:m on body from body[0];\n  observable point:K=evaluate(temperature,at=(x=0.125[m]));\n  observable energy:");
    let (document, _, result) = heat_with_spatial(&source, CommonSpatialPolicy::CellCenteredTpfa);
    let id = document
        .program()
        .nodes()
        .find_map(|node| match node {
            KernelNode::Observable(definition)
                if definition.reduction() == ObservableReduction::Value =>
            {
                Some(definition.id())
            }
            _ => None,
        })
        .unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    assert!(
        result
            .observe(&model, id, &std::collections::HashMap::new())
            .unwrap_err()
            .message()
            .contains("admitted Q1 space")
    );
}

#[test]
fn cartesian_point_reconstruction_preserves_coordinate_order_and_bilinear_slopes() {
    let source = super::support::common_scalar_plan::COMPONENT
        .replace("source_scale * math.sin", "0 * source_scale * math.sin")
        .replace(
            "trace(potential) - boundary_offset",
            "trace(potential) - coordinate(0)*(3[1/m]+2[1/m^2]*coordinate(1))",
        );
    let end = source.rfind('}').unwrap();
    let declarations = "
        coordinate x:m on square from square[0]; coordinate y:m on square from square[1];
        observable point:1=evaluate(potential,at=(y=0.625[m],x=0.125[m]));
        observable dx:1/m=evaluate(partial(potential,wrt=x),at=(x=0.125[m],y=0.625[m]));
        observable dy:1/m=evaluate(partial(potential,wrt=y),at=(y=0.625[m],x=0.125[m]));
    ";
    let source = format!("{}{}{}", &source[..end], declarations, &source[end..]);
    let (document, plan) = super::support::common_scalar_plan::document_and_plan_with_source(
        CommonSpatialPolicy::Q1,
        &source,
    );
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let result = plan.run_result(&REFERENCE_LINEAR_SOLVER).unwrap();
    // u=3*x+2*x*y is harmonic and exactly in Q1: at (1/8,5/8),
    // u=17/32, u_x=17/4 per metre and u_y=1/4 per metre.
    for (name, expected, power) in [
        ("point", 17.0 / 32.0, 0),
        ("dx", 17.0 / 4.0, -1),
        ("dy", 1.0 / 4.0, -1),
    ] {
        let id = document.aliases()[&format!("definition.{name}")]
            .downcast()
            .unwrap();
        let observed = result
            .observe(&model, id, &std::collections::HashMap::new())
            .unwrap();
        let value = observed.value().real_scalar_value().unwrap();
        assert_eq!(
            value.dim(),
            DimExponents::from_integers([0, power, 0, 0, 0, 0, 0]).unwrap()
        );
        assert!(
            (value.value() - expected).abs() < 1e-9,
            "{name}: {value:?} != {expected}"
        );
    }
}
