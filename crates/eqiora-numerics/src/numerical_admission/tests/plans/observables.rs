//! Independent Q1 integral and first-variation witnesses on an ordinary Result.
use super::*;
use eqiora_meshing::QuadratureRule;
use eqiora_schema::kernel::{KernelNode, ObservableReduction};

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

fn heat(source: &str) -> (ModelEnvelope, crate::CommonScalarPlan, crate::CommonResult) {
    let geometry = cartesian_interval();
    let body = geometry.entity_set("body").unwrap();
    let model = compile_model(
        "heat-observable.eqi",
        source,
        &geometry,
        "HeatedBody",
        &[
            ("body", body, None),
            (
                "left",
                geometry.entity_set("left").unwrap(),
                Some(("body", body)),
            ),
            (
                "right",
                geometry.entity_set("right").unwrap(),
                Some(("body", body)),
            ),
        ],
        &[],
    );
    let plan = resolve_scalar_box(
        &model,
        cartesian_box_resources(&geometry, &[4]),
        CommonSpatialPolicy::Q1,
    );
    let result = plan.run_result(&REFERENCE_LINEAR_SOLVER).unwrap();
    (model, plan, result)
}

#[test]
fn ordinary_observables_integrate_energy_and_oriented_flux_with_state_jvp() {
    check_energy_and_flux(HEAT);
}

#[test]
fn pure_operator_spatial_observable_preserves_field_state_jvp() {
    let source = format!(
        "operator stored(input c:J/(K*m),input t:K):J/m=c*(t-300[K]);\n{}",
        HEAT.replace(
            "capacity * (temperature - 300[K])",
            "stored(c=capacity,t=temperature)"
        )
    );
    check_energy_and_flux(&source);
}

fn check_energy_and_flux(source: &str) {
    let (model, plan, result) = heat(source);
    assert_eq!(
        plan.fields().len(),
        1,
        "derived expressions add no solve unknowns"
    );
    let quadrature = QuadratureRule::gauss_legendre(2).unwrap();
    let face = QuadratureRule::point();
    let program = plan.observation_program();
    let temperature = plan.fields().next().unwrap().0;
    let temperature_dimension = plan.fields().next().unwrap().1.dimension();
    let constant_direction = result
        .observable_state_tangent([(
            temperature,
            vec![DynQuantity::new(2.0, temperature_dimension); 5],
        )])
        .unwrap();
    let affine_direction = result
        .observable_state_tangent([(
            temperature,
            (0..5)
                .map(|index| DynQuantity::new(1.0 + 0.5 * f64::from(index), temperature_dimension))
                .collect(),
        )])
        .unwrap();
    let mut count = 0;
    for definition in program.nodes().filter_map(|node| match node {
        KernelNode::Observable(value) => Some(value),
        _ => None,
    }) {
        count += 1;
        let spatial = matches!(
            definition.reduction(),
            ObservableReduction::SpatialIntegral {
                measure: eqiora_schema::kernel::ObservableMeasure::Volume,
                ..
            }
        );
        let rule = if spatial { &quadrature } else { &face };
        let observation = result
            .observe(
                &model,
                definition.id(),
                &observation_rules(&model, definition.id(), rule),
            )
            .unwrap();
        let domain = definition.reduction().domain().unwrap();
        let foreign_domain = eqiora_core::Id::new();
        assert!(
            result
                .observe(
                    &model,
                    definition.id(),
                    &std::collections::HashMap::from([(foreign_domain, rule.clone())])
                )
                .is_err()
        );
        assert!(
            result
                .observe(
                    &model,
                    definition.id(),
                    &std::collections::HashMap::from([
                        (domain, rule.clone()),
                        (foreign_domain, rule.clone())
                    ])
                )
                .is_err()
        );
        assert_eq!(observation.result_identity(), result.identity());
        assert_eq!(
            observation.quadratures(),
            &observation_rules(&model, definition.id(), rule)
        );
        let value = observation.value().real_scalar_value().unwrap().value();
        let constant_jvp = result
            .observe_state_jvp(
                &model,
                definition.id(),
                &observation_rules(&model, definition.id(), rule),
                &constant_direction,
            )
            .unwrap()
            .real_scalar_value()
            .unwrap()
            .value();
        let affine_jvp = result
            .observe_state_jvp(
                &model,
                definition.id(),
                &observation_rules(&model, definition.id(), rule),
                &affine_direction,
            )
            .unwrap()
            .real_scalar_value()
            .unwrap()
            .value();
        let unit = definition.value_type().dimension();
        if unit == DimExponents::from_integers([1, 2, -2, 0, 0, 0, 0]).unwrap() {
            // Exact nodal solve T_i=300+6*x_i*(1-x_i), integrated Q1 hats:
            // integral(T-300)=15/16 K*m, times capacity3 =>45/16 J.
            assert!((value - 45.0 / 16.0).abs() < 1e-9);
            assert!((constant_jvp - 6.0).abs() < 1e-10);
            assert!((affine_jvp - 6.0).abs() < 1e-10);
        } else if spatial {
            assert!((value - 7.0).abs() < 1e-12);
            assert_eq!(constant_jvp, 0.0);
            assert_eq!(affine_jvp, 0.0);
        } else {
            // Discrete endpoint slope is +/-4.5 K/m; outward heat flux=9 W.
            assert!((value - 9.0).abs() < 1e-9);
            assert!(constant_jvp.abs() < 1e-12);
            // deltaT=1+2x: left/right outward flux variations are +4/-4 W.
            let domain = definition.reduction().domain().unwrap();
            let (_, boundary) = plan.observation_support(domain.erase()).unwrap();
            let expected = match boundary.unwrap().1 {
                eqiora_schema::kernel::BoundarySide::Lower => 4.0,
                eqiora_schema::kernel::BoundarySide::Upper => -4.0,
            };
            assert!((affine_jvp - expected).abs() < 1e-12);
        }
        assert!(
            result
                .observe(&model, definition.id(), &Default::default())
                .is_err()
        );
        let wrong = if spatial { &face } else { &quadrature };
        assert!(
            result
                .observe(
                    &model,
                    definition.id(),
                    &observation_rules(&model, definition.id(), wrong)
                )
                .is_err()
        );
    }
    assert_eq!(count, 4);
    assert!(
        result
            .observable_state_tangent([(
                temperature,
                vec![DynQuantity::new(1.0, DimExponents::DIMENSIONLESS); 5]
            )])
            .is_err()
    );
    assert!(
        result
            .observable_state_tangent([(
                temperature,
                vec![DynQuantity::new(1.0, temperature_dimension); 4]
            )])
            .is_err()
    );
    assert!(
        result
            .observable_state_tangent([(eqiora_core::Id::new(), vec![])])
            .is_err()
    );
    let foreign = heat(&HEAT.replace("= 3;", "= 4;"));
    let observable = program
        .nodes()
        .find_map(|node| match node {
            KernelNode::Observable(value) => Some(value.id()),
            _ => None,
        })
        .unwrap();
    assert!(
        result
            .observe(&foreign.0, observable, &Default::default())
            .is_err()
    );
    let foreign_direction = foreign.2.observable_state_tangent([]).unwrap();
    assert!(
        result
            .observe_state_jvp(
                &model,
                observable,
                &observation_rules(&model, observable, &quadrature),
                &foreign_direction
            )
            .is_err()
    );
}

#[test]
fn gradient_energy_and_its_state_direction_use_the_ordinary_result() {
    let source = HEAT
        .replace(
            "parameter capacity: J / (K * m) = 3;",
            "parameter stiffness: J*m/K^2 = 2;",
        )
        .replace(
            "capacity * (temperature - 300[K])",
            "stiffness*contract(grad(temperature),grad(temperature),axes=((0,0),))/2",
        );
    let (model, plan, result) = heat(&source);
    let energy_dimension = DimExponents::from_integers([1, 2, -2, 0, 0, 0, 0]).unwrap();
    let energy = plan
        .observation_program()
        .nodes()
        .find_map(|node| match node {
            KernelNode::Observable(value) if value.value_type().dimension() == energy_dimension => {
                Some(value.id())
            }
            _ => None,
        })
        .unwrap();
    let quadrature = QuadratureRule::gauss_legendre(2).unwrap();
    let value = result
        .observe(
            &model,
            energy,
            &observation_rules(&model, energy, &quadrature),
        )
        .unwrap();
    // Q1 nodal T=300+6*x*(1-x) gives slopes 4.5,1.5,-1.5,-4.5 K/m.
    // For k=2 J*m/K^2, integral(k*T_x^2/2)=45/4 J.
    assert!((value.value().real_scalar_value().unwrap().value() - 45.0 / 4.0).abs() < 1e-9);
    let (temperature, ty) = plan.fields().next().unwrap();
    // eta=x*(1-x) K has zero endpoint traces and eta_x=T_x/6.
    // Independently, delta E=integral(k*T_x*eta_x)=15/4 J.
    let direction = result
        .observable_state_tangent([(
            temperature,
            [0.0, 3.0 / 16.0, 1.0 / 4.0, 3.0 / 16.0, 0.0]
                .map(|value| DynQuantity::new(value, ty.dimension()))
                .to_vec(),
        )])
        .unwrap();
    let derivative = result
        .observe_state_jvp(
            &model,
            energy,
            &observation_rules(&model, energy, &quadrature),
            &direction,
        )
        .unwrap();
    assert_eq!(derivative.value_type().dimension(), energy_dimension);
    assert!((derivative.real_scalar_value().unwrap().value() - 15.0 / 4.0).abs() < 1e-9);
}

fn observation_rules(
    model: &eqiora_artifact::ModelEnvelope,
    observable: eqiora_core::Id<eqiora_core::entity::kinds::Observable>,
    rule: &eqiora_meshing::QuadratureRule,
) -> std::collections::HashMap<
    eqiora_core::Id<eqiora_core::entity::kinds::Domain>,
    eqiora_meshing::QuadratureRule,
> {
    let (transaction, _) = model.to_transaction().unwrap();
    let domain = transaction
        .ops()
        .iter()
        .find_map(|operation| match operation {
            eqiora_graph::Op::DefineKernelNode {
                node: eqiora_schema::kernel::KernelNode::Observable(definition),
            } if definition.id() == observable => definition.reduction().domain(),
            _ => None,
        })
        .unwrap();
    std::collections::HashMap::from([(domain, rule.clone())])
}
