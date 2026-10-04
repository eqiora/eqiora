use super::*;

fn prescribed_elastic_surface_source() -> String {
    SOURCE.replace(
        "  parameter length_scale: m\n",
        "  parameter length_scale: m,\n  parameter traction:vector<Pa,2>=tensor_value(frame=body,components=[6[Pa],0[Pa]])\n",
    ).replace(
        "  relation x_upper_free on x_upper {\n    normal(2 * mu * symmetric_part(grad(displacement))\n      + lambda * isotropic_lift(div(displacement))) = 0;",
        "  relation x_upper_free on x_upper {\n    normal(2 * mu * symmetric_part(grad(displacement))\n      + lambda * isotropic_lift(div(displacement))) = traction;",
    ).replace(
        "  relation load on body {",
        r#"  observable bulk:N=integral(
    mu*contract(symmetric_part(grad(displacement)),symmetric_part(grad(displacement)),axes=((0,0),(1,1)))
    +lambda*div(displacement)*div(displacement)/2
    -contract(grad(load_potential),displacement,axes=((0,0),)),measure(body));
  observable surface:N=integral(-contract(traction,trace(displacement),axes=((0,0),)),measure(x_upper));
  observable total:N=bulk+surface;
  relation load on body {"#,
    )
}

#[test]
fn prescribed_elastic_surface_work_has_an_ordinary_positive_result() {
    let source = prescribed_elastic_surface_source();
    let accepted = accepted_source_on(&source, 0.0, 2);
    check_loaded_result(&accepted);
}

fn check_loaded_result(accepted: &Accepted) {
    let (_, values, _) = accepted.result.field_block(0, 0).unwrap();
    // mu=3, lambda=0, f=(6,0), t_right=(6,0): u=(2x-x²/2,0).
    // The Q1 nodal solution is exact. The existing independent inverse norm
    // 316103/59058 and ||b||²=783/32 bound coefficient error by 2.65e-9 m.
    for (vertex, value) in values.as_chunks::<2>().0.iter().enumerate() {
        let x = accepted
            .mesh
            .mesh()
            .vertex_coordinates(MeshEntity::new(0, vertex))
            .unwrap()[0];
        assert!((value[0] - (2.0 * x - x * x / 2.0)).abs() < 3e-9);
        assert!(value[1].abs() < 3e-9);
    }
    let model = ModelEnvelope::from_program(accepted.document.program()).unwrap();
    let mut rules = std::collections::HashMap::new();
    let mut total = None;
    for node in accepted.document.program().nodes() {
        if let eqiora::kernel::KernelNode::Observable(value) = node {
            match value.reduction() {
                eqiora::kernel::ObservableReduction::Value => total = Some(value.id()),
                eqiora::kernel::ObservableReduction::SpatialIntegral {
                    domain, measure, ..
                } => {
                    let dimension = match measure {
                        eqiora::kernel::ObservableMeasure::Volume => 2,
                        eqiora::kernel::ObservableMeasure::Boundary => 1,
                    };
                    let rule = eqiora::meshing::QuadratureRule::tensor_product_gauss_legendre(
                        dimension, 2,
                    )
                    .unwrap();
                    let observed = accepted
                        .result
                        .observe(
                            &model,
                            value.id(),
                            &std::collections::HashMap::from([(domain, rule.clone())]),
                        )
                        .unwrap();
                    let expected = if dimension == 2 { 33.0 / 16.0 } else { -9.0 };
                    // Each separate work term is linear/Lipschitz in the nodal error.
                    assert!((observed.value().component(0).unwrap().0 - expected).abs() < 1e-7);
                    rules.insert(domain, rule);
                }
            }
        }
    }
    // Internal energy=3*(37/16)=111/16; body work=6*(13/16)=39/8;
    // surface work=-6*(3/2)=-9, hence total=-111/16 N.
    // The total's stationary coefficient error is quadratic; 1e-9 reserves
    // quadrature and binary64 rounding on this four-cell fixture.
    let total = total.unwrap();
    let observed = accepted.result.observe(&model, total, &rules).unwrap();
    assert!((observed.value().real_scalar_value().unwrap().value() + 111.0 / 16.0).abs() < 1e-9);
    let (reaction, body, _, _) = accepted.result.elasticity_observation().unwrap();
    assert!((body[0] - 6.0).abs() < 1e-12 && body[1].abs() < 1e-12);
    assert!((reaction[0] + 12.0).abs() < 1e-7 && reaction[1].abs() < 1e-7);
    let displacement = accepted
        .document
        .program()
        .nodes()
        .find_map(|node| match node {
            eqiora::kernel::KernelNode::Field(field) if field.value_type().shape().rank() == 1 => {
                Some(field)
            }
            _ => None,
        })
        .unwrap();
    for (admissible, first_expected, second_expected) in [(true, 0.0, 6.0), (false, -12.0, 0.0)] {
        let coefficients = (0..values.len())
            .map(|i| {
                let x = accepted
                    .mesh
                    .mesh()
                    .vertex_coordinates(MeshEntity::new(0, i / 2))
                    .unwrap()[0];
                eqiora::DynQuantity::new(
                    if i % 2 == 1 {
                        0.0
                    } else if admissible {
                        x
                    } else {
                        1.0
                    },
                    displacement.dimension(),
                )
            })
            .collect::<Vec<_>>();
        let direction = accepted
            .result
            .observable_state_tangent([(displacement.id(), coefficients)])
            .unwrap();
        let first = accepted
            .result
            .observe_state_jvp(&model, total, &rules, &direction)
            .unwrap();
        // eta=(x,0) is admissible; translation is arbitrary and gives -6 body -6 surface.
        assert!((first.component(0).unwrap().0 - first_expected).abs() < 1e-7);
        let second = accepted
            .result
            .observe_state_second_variation(
                &model,
                total,
                &rules,
                displacement.id(),
                [&direction, &direction],
            )
            .unwrap();
        // D²F[eta,eta]=integral 6*eta_x,x²: 6 for eta=x and 0 for translation.
        assert!((second.component(0).unwrap().0 - second_expected).abs() < 1e-12);
    }
    let bytes = accepted.result.to_bytes().unwrap();
    let replayed = CommonResult::from_bytes(&bytes, accepted.result.plan()).unwrap();
    assert_eq!(
        replayed.observe(&model, total, &rules).unwrap().value(),
        observed.value()
    );
}

fn authored_source() -> String {
    let source = prescribed_elastic_surface_source();
    format!(
        "{}\n{}\n}}",
        source.strip_suffix('}').unwrap(),
        r#"
  form stationary for balance {
    test w:m for displacement zero_on x_lower;
    variation(total,wrt=displacement,direction=w,holding=(mu,lambda,load_potential,traction))=0;
  }
"#
    )
}

#[test]
fn prescribed_elastic_surface_work_has_an_authored_positive_result() {
    let accepted = accepted_source_on(&authored_source(), 0.0, 2);
    check_loaded_result(&accepted);
    let bytes = accepted.result.plan().to_bytes().unwrap();
    let replayed = eqiora_numerics::ResolvedCommonPlan::from_bytes(
        &bytes,
        &REFERENCE_LINEAR_SOLVER,
        eqiora::time::TimeBackendIdentity::new("eqiora.test.time", "1"),
    )
    .unwrap();
    assert_eq!(replayed.to_bytes().unwrap(), bytes);
    let result = replayed
        .as_elasticity()
        .unwrap()
        .run_result(&REFERENCE_LINEAR_SOLVER)
        .unwrap();
    assert_eq!(
        result.field_block(0, 0).unwrap().1,
        accepted.result.field_block(0, 0).unwrap().1
    );
    let (_, values, _) = accepted.result.field_block(0, 0).unwrap();
    for (vertex, value) in values.as_chunks::<2>().0.iter().enumerate() {
        let x = accepted
            .mesh
            .mesh()
            .vertex_coordinates(MeshEntity::new(0, vertex))
            .unwrap()[0];
        assert!((value[0] - (2.0 * x - x * x / 2.0)).abs() < 3e-9);
        assert!(value[1].abs() < 3e-9);
    }
    assert_eq!(
        accepted.result.plan().formulation().unwrap().requested(),
        eqiora_numerics::FormulationSelectionMode::Authored
    );
}

#[test]
fn prescribed_elastic_surface_work_retains_signs_sides_and_nominal_data() {
    let source = authored_source();
    let normal = "normal(2 * mu * symmetric_part(grad(displacement))\n      + lambda * isotropic_lift(div(displacement)))";
    let reversed = source.replace(
        &format!("{normal} = traction;"),
        &format!("traction = {normal};"),
    );
    check_loaded_result(&accepted_source_on(&reversed, 0.0, 2));
    let foreign = source.replace("  parameter traction:", "  parameter other_traction:vector<Pa,2>=tensor_value(frame=body,components=[6[Pa],0[Pa]]),\n  parameter traction:")
        .replace("-contract(traction,trace(displacement)", "-contract(other_traction,trace(displacement)")
        .replace("load_potential,traction)", "load_potential,other_traction)");
    for changed in [
        source
            .replace("total:N=bulk+surface", "total:N=bulk")
            .replace("load_potential,traction)", "load_potential)"),
        source.replace("total:N=bulk+surface", "total:N=bulk-surface"),
        source.replace("measure(x_upper)", "measure(y_upper)"),
        foreign,
    ] {
        let error = try_accepted_source_on(&changed, 0.0, 2)
            .err()
            .expect("wrong surface work must reject");
        assert!(
            error
                .message()
                .contains("differs from the admitted elastic strong-law weak residual"),
            "{error:?}"
        );
    }
    let wrong_stress = source.replace(
        &format!("{normal} = traction;"),
        &format!("{} = traction;", normal.replace("2 * mu", "3 * mu")),
    );
    let error = try_accepted_source_on(&wrong_stress, 0.0, 2).err().unwrap();
    assert!(
        error
            .message()
            .contains("boundary and volume isotropic stress coefficients differ"),
        "{error:?}"
    );
}

#[test]
fn prescribed_elastic_surface_loads_use_the_exact_cartesian_side() {
    let source = authored_source();
    let rotated = source
        .replace("x_lower", "TEMP_lower")
        .replace("x_upper", "TEMP_upper")
        .replace("y_lower", "x_lower")
        .replace("y_upper", "x_upper")
        .replace("TEMP_lower", "y_lower")
        .replace("TEMP_upper", "y_upper")
        .replace("coordinate(0)", "coordinate(1)");
    let mirrored = source
        .replace("x_lower", "TEMP")
        .replace("x_upper", "x_lower")
        .replace("TEMP", "x_upper");
    for (source, traction, axis, upper) in [
        (rotated, [0.0, 6.0], 1, true),
        (mirrored, [-6.0, 0.0], 0, false),
    ] {
        let accepted = try_accepted_source_on_with_traction(&source, 0.0, 2, traction).unwrap();
        let (_, values, _) = accepted.result.field_block(0, 0).unwrap();
        for (vertex, values) in values.as_chunks::<2>().0.iter().enumerate() {
            let x = accepted
                .mesh
                .mesh()
                .vertex_coordinates(MeshEntity::new(0, vertex))
                .unwrap()[axis];
            let expected = if upper {
                2.0 * x - x * x / 2.0
            } else {
                -(1.0 - x) * (1.0 - x) / 2.0
            };
            assert!((values[axis] - expected).abs() < 3e-9);
            assert!(values[1 - axis].abs() < 3e-9);
        }
    }
}
