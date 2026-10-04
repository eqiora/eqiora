//! Separable kernels retain distinct source/target coordinates through the ordinary solve.
use super::*;
use eqiora::api::ModelDocument;
use std::collections::HashMap;

const SOURCE: &str =
    include_str!("../../../../verify/language/factor-integrals/models/nonlocal-interaction.eqi");

fn document(source: &str) -> ModelDocument {
    bounded_document(source, 2.0).unwrap()
}

fn bounded_document(
    source: &str,
    source_upper: f64,
) -> Result<ModelDocument, Vec<eqiora_core::Diagnostic>> {
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let interval = |upper| {
        StaticBindingValue::CoordinateInterval(
            AxisBounds::new(
                DynQuantity::new(0.0, length),
                DynQuantity::new(upper, length),
            )
            .unwrap(),
        )
    };
    ModelDocument::compile_selected(
        "nonlocal.eqi",
        source,
        "Interaction",
        &[
            ("target", interval(1.0)),
            ("source", interval(source_upper)),
        ],
    )
}

#[test]
fn nonlocal_separable_kernel_observation_and_integral_coupled_solve() {
    let document = document(SOURCE);
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let plan = resolve(&model).unwrap();
    let result = plan
        .run_result(&plan.initial_state(&[]).unwrap(), &FaerLinearSolver)
        .unwrap();
    let domain = |name| document.aliases()[name].downcast().unwrap();
    let rules = HashMap::from([(domain("source"), QuadratureRule::gauss_legendre(2).unwrap())]);
    // Independently, integral_0^2 (x*y)*(A*y) dy=8*A*x/3 in coherent SI.
    // Its target integral on [0,1] is 4*A/3, so the constraint fixes A=3/2.
    for x in [0.0, 0.25, 0.5, 1.0] {
        let observation = result
            .observe_at(
                &model,
                document.aliases()["action"].downcast().unwrap(),
                &[DynQuantity::new(x, length)],
                &rules,
            )
            .unwrap();
        assert!((observation.value().real_scalar_value().unwrap().value() - 4.0 * x).abs() < 1e-12);
    }
}

#[test]
fn nonlocal_integral_retains_residual_and_output_jvp_vjp_at_parameter_points() {
    use eqiora::api::DifferentiableProgram;
    use eqiora_numerics::ResolvedCommonPlan;
    let source = SOURCE
        .replace(
            "inventory=2[m]",
            "inventory=amount; inequality(amount>=0[m])",
        )
        .replace(
            "variable amplitude:1;",
            "variable amplitude:1; parameter amount:m=2[m];
        coordinate target_x:m on target from target;
        observable weighted:m=integral(target_x/1[m]*action,measure(target));
        observable readout:m=weighted;",
        );
    let document = document(&source);
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let linear = SolverPlan::new(
        LinearSolver::SparseLu,
        1e-12,
        1e-14,
        NonZeroUsize::new(8).unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Fast);
    let policy = CommonSolvePolicy::Newton {
        nonlinear: eqiora_realization::NonlinearSolvePlan::new(
            0.0,
            1e-12,
            NonZeroUsize::new(8).unwrap(),
            8,
        )
        .unwrap(),
        linear: CommonLinearRequest::exact(linear, FaerLinearSolver.provider()).unwrap(),
    };
    use eqiora_numerics::finite_constraints::{
        ConstraintRef, ConstraintTolerance, FiniteConstraintEnforcement,
    };
    // The existing accepted-point derivative owner admits strict-interior Newton Plans.
    let enforcement = FiniteConstraintEnforcement::strict_interior(vec![
        ConstraintTolerance::inequality(
            ConstraintRef::new(
                document.aliases()["prescribed_inventory"]
                    .downcast()
                    .unwrap(),
                1,
            ),
            DynQuantity::new(
                1e-10,
                DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap(),
            ),
        )
        .unwrap(),
    ])
    .unwrap();
    let plan =
        CommonAlgebraicPlan::resolve(&model, policy, Some(enforcement), None, &FaerLinearSolver)
            .unwrap();
    use eqiora_artifact::CanonicalModelArtifact;
    let seed = eqiora_numerics::CommonInitialField::scalar(
        model.artifact_reference().unwrap().artifact().clone(),
        document.aliases()["amplitude"].downcast().unwrap(),
        1.0,
    )
    .unwrap();
    let initial = plan.initial_state(&[seed]).unwrap();
    let program = DifferentiableProgram::compile(
        ResolvedCommonPlan::Algebraic(Box::new(plan)),
        &[document.parameter_ref("amount").unwrap()],
        &document.observable_ref("readout").unwrap(),
        Some(initial),
        &FaerLinearSolver,
    )
    .unwrap();
    let close = |actual: f64, expected: f64| {
        assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}")
    };
    // R(A,p)=4*A/3-p and O(A)=integral_0^1 x*(8*A*x/3) dx=8*A/9.
    // Therefore O(p)=2*p/3. Partial actions retain A and p as independent coordinates.
    for amount in [2.0, 4.0] {
        let point = program.evaluate(&[amount]).unwrap();
        close(point.accepted_unknowns()[0], 3.0 * amount / 4.0);
        close(point.primal().output()[0], 2.0 * amount / 3.0);
        close(point.residual_jvp(&[1.0], &[0.0]).unwrap()[0], 4.0 / 3.0);
        let (unknown, parameter) = point.residual_vjp(&[3.0]).unwrap();
        close(unknown[0], 4.0);
        close(parameter[0], -3.0);
        close(
            point.output_partial_jvp(&[1.0], &[0.0]).unwrap()[0],
            8.0 / 9.0,
        );
        let (unknown, parameter) = point.output_partial_vjp(&[9.0]).unwrap();
        close(unknown[0], 8.0);
        close(parameter[0], 0.0);
        close(point.jvp(&[3.0]).unwrap().tangent()[0], 2.0);
        close(point.vjp(&[3.0]).unwrap().input_cotangent()[0], 2.0);
    }
}

#[test]
fn nonsymmetric_kernel_uses_the_declared_pairing_and_each_domain_measure() {
    use eqiora_numerics::ResolvedCommonPlan;
    let source = "model Interaction(support target:interval(m), support source:interval(m)) {
        support pair:product(target,source);
        coordinate x:m on pair from target; coordinate y:m on pair from source;
        coordinate tx:m on target from target; coordinate sy:m on source from source;
        variable anchor:1; relation fixed {anchor=1;}
        let kernel:1/m on pair=(1+x/1[m]+2*y/1[m])/1[m];
        observable forward:1 on target=integral(kernel*y/1[m],measure(source));
        observable adjoint:1 on source=integral(kernel*x/1[m],measure(target));
        observable left:m=integral(tx/1[m]*forward,measure(target));
        observable right:m=integral(sy/1[m]*adjoint,measure(source));
    }";
    let document = document(source);
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let model =
        ModelEnvelope::from_json(&model.canonical_json().unwrap(), Default::default()).unwrap();
    let plan = resolve(&model).unwrap();
    let result = plan
        .run_result(&plan.initial_state(&[]).unwrap(), &FaerLinearSolver)
        .unwrap();
    let plan = ResolvedCommonPlan::from_bytes(
        &result.plan().to_bytes().unwrap(),
        &FaerLinearSolver,
        eqiora::time::TimeBackendIdentity::new("eqiora.test.time", "1"),
    )
    .unwrap();
    let result = CommonResult::from_bytes(&result.to_bytes().unwrap(), &plan).unwrap();
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let domain = |name| document.aliases()[name].downcast().unwrap();
    let rule = QuadratureRule::gauss_legendre(2).unwrap();
    // x is in [0,1], y in [0,2]. Integrating independent monomials gives
    // Ku=22/3+2*x and K*v=5/6+y for u=y, v=x. Both pairings are 13/3 m.
    // Swapping x and y inside K while retaining these exact source/target bindings
    // gives 7/6+y/2 and pairing 11/3 m, not the declared adjoint pairing.
    for (name, measure, points) in [
        (
            "forward",
            "source",
            vec![(0.0, 22.0 / 3.0), (0.5, 25.0 / 3.0), (1.0, 28.0 / 3.0)],
        ),
        (
            "adjoint",
            "target",
            vec![(0.0, 5.0 / 6.0), (0.5, 4.0 / 3.0), (2.0, 17.0 / 6.0)],
        ),
    ] {
        let rules = HashMap::from([(domain(measure), rule.clone())]);
        for (point, expected) in points {
            let value = result
                .observe_at(
                    &model,
                    document.aliases()[name].downcast().unwrap(),
                    &[DynQuantity::new(point, length)],
                    &rules,
                )
                .unwrap();
            let value = value.value().real_scalar_value().unwrap();
            assert_eq!(value.dim(), DimExponents::DIMENSIONLESS);
            assert!((value.value() - expected).abs() < 1e-12);
        }
    }
    let rules = HashMap::from([(domain("source"), rule.clone()), (domain("target"), rule)]);
    for name in ["left", "right"] {
        let value = result
            .observe(&model, document.aliases()[name].downcast().unwrap(), &rules)
            .unwrap();
        let value = value.value().real_scalar_value().unwrap();
        assert_eq!(value.dim(), length);
        assert!((value.value() - 13.0 / 3.0).abs() < 1e-12);
    }
}

#[test]
fn finite_atomic_source_sum_retains_weights_units_and_binder_identity() {
    let source = "operator kernel(input x:m,input y:m,input scale:1/m^3):1/m=x*y*scale;
    model Interaction(support target:interval(m), support source:interval(m)) {
        indexset Atoms=range(2);
        coordinate x:m on target from target;
        variable anchor:1; relation fixed {anchor=1;}
        observable action:1 on target=sum(
            kernel(x=x,y=(to_real(ordinal(i))+1)*1[m],scale=1[1/m^3])
            *(to_real(ordinal(i))+1)*(to_real(ordinal(i))+1)*1[m],over=(i in Atoms));
    }";
    let document = document(source);
    let renamed = super::nonlocal::document(
        &source
            .replace("ordinal(i)", "ordinal(j)")
            .replace("i in Atoms", "j in Atoms"),
    );
    assert!(document.structurally_equivalent(&renamed).unwrap());
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let plan = resolve(&model).unwrap();
    let result = plan
        .run_result(&plan.initial_state(&[]).unwrap(), &FaerLinearSolver)
        .unwrap();
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    // Two atoms at y=1,2 m have masses 1,2 m and samples u=1,2.
    // Their exact finite action is x*(1*1*1+2*2*2)=9*x in coherent SI.
    // This atomic measure differs from continuous Lebesgue integration on [0,2].
    for x in [0.0, 0.25, 1.0] {
        let value = result
            .observe_at(
                &model,
                document.aliases()["action"].downcast().unwrap(),
                &[DynQuantity::new(x, length)],
                &HashMap::new(),
            )
            .unwrap();
        assert!((value.value().real_scalar_value().unwrap().value() - 9.0 * x).abs() < 1e-12);
    }
}

#[test]
fn nonlocal_kernel_rejects_singular_coupling_and_excessive_quadrature_degree() {
    resolve(&ModelEnvelope::from_program(document(SOURCE).program()).unwrap()).unwrap();
    for (kernel, message) in [
        ("1[m]/(x-y)^2", "coordinate-independent denominators"),
        ("(x/1[m])*(y/1[m])^14/1[m]", "Gauss"),
    ] {
        let document = document(&SOURCE.replace("x*y/1[m^3]", kernel));
        let model = ModelEnvelope::from_program(document.program()).unwrap();
        let error = resolve(&model).unwrap_err();
        assert!(error.message().contains(message), "{error:?}");
    }
}

#[test]
fn nonlocal_unit_interval_action_is_x_over_three_and_rejects_foreign_measure() {
    let source = SOURCE.replace("inventory=2[m]", "amplitude=1");
    let document = bounded_document(&source, 1.0).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let plan = resolve(&model).unwrap();
    let result = plan
        .run_result(&plan.initial_state(&[]).unwrap(), &FaerLinearSolver)
        .unwrap();
    let domain = |name| document.aliases()[name].downcast().unwrap();
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let rule = QuadratureRule::gauss_legendre(2).unwrap();
    let rules = HashMap::from([(domain("source"), rule.clone())]);
    // K=(x/m)(y/m)/m and u=y/m. With source [0,1] m, integral y^2 dy = 1/3.
    for x in [0.0, 0.25, 1.0] {
        let value = result
            .observe_at(
                &model,
                document.aliases()["action"].downcast().unwrap(),
                &[DynQuantity::new(x, length)],
                &rules,
            )
            .unwrap();
        assert!((value.value().real_scalar_value().unwrap().value() - x / 3.0).abs() < 1e-12);
    }
    // Numerically identical intervals still have different exact measure identities.
    let foreign = HashMap::from([(domain("target"), rule)]);
    let error = result
        .observe_at(
            &model,
            document.aliases()["action"].downcast().unwrap(),
            &[DynQuantity::new(0.25, length)],
            &foreign,
        )
        .unwrap_err();
    assert!(
        error.message().contains("exact measure Domain"),
        "{error:?}"
    );
    for (from, to, gate) in [
        ("action:1 on target", "action:1 on source", "support"),
        ("kernel:1/m", "kernel:1", "type assertion"),
    ] {
        let errors = bounded_document(&source.replace(from, to), 1.0).unwrap_err();
        assert!(
            errors.iter().any(|error| error.message().contains(gate)),
            "{errors:?}"
        );
    }
}

#[test]
fn nonlocal_composition_has_a_bounded_dependency_depth() {
    let chained = |count| {
        let mut declarations = String::new();
        let mut previous = "action".to_owned();
        for i in 0..count {
            declarations.push_str(&format!("observable layer{i}:1 on target={previous};\n"));
            previous = format!("layer{i}");
        }
        SOURCE.replace(
            "observable inventory:m=integral(action,measure(target));",
            &format!("{declarations} observable inventory:m=integral({previous},measure(target));"),
        )
    };
    resolve(&ModelEnvelope::from_program(document(&chained(4)).program()).unwrap()).unwrap();
    let model = ModelEnvelope::from_program(document(&chained(40)).program()).unwrap();
    let error = resolve(&model).unwrap_err();
    assert!(
        error.message().contains("depth") || error.message().contains("bounded unannotated"),
        "{error:?}"
    );
}
