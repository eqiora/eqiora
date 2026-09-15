use super::*;

#[test]
fn discarded_zero_rejects_coordinate_dependent_multipliers() {
    let source = "model Scalar() {
        domain body = box(0, 1, 0, 1);

        variable u: 1 on body;
        relation balance on body {
            -div(grad(u)) + (1 / coordinate(0) ^ 2) * 0 = 0;
        }
    }";
    assert!(
        derive(source)
            .unwrap_err()
            .message()
            .contains("discarded zero requires a coordinate-independent multiplier")
    );
}

#[test]
fn uniform_load_gradients_validate_their_parameter_values_before_erasure() {
    for expression in ["load_scale / divisor", "0 * load_scale / divisor"] {
        let source = MIXED
            .replace("state v:", "parameter load_scale: kg / (m * s ^ 2) = 1; parameter divisor: 1 = 0; variable load: kg / (m * s ^ 2) on body; state v:")
            .replace("relation balance", &format!("relation load_definition on body {{ load - {expression} = 0; }} relation balance"))
            .replace("isotropic_lift(p)) = 0", "isotropic_lift(p)) - grad(load) = 0");
        assert!(
            derive(&source)
                .unwrap_err()
                .message()
                .contains("non-finite")
        );
        let valid = source.replace("divisor: 1 = 0", "divisor: 1 = 4");
        let form = derive(&valid).unwrap();
        let coefficient = form
            .roles
            .relations
            .values()
            .find(|role| matches!(role.kind, Role::Coefficient { .. }))
            .unwrap();
        // The defined Field and both Parameters remain exact dependencies even
        // though the coordinate-independent load has a vanishing gradient.
        assert_eq!(coefficient.dependencies.len(), 3);
    }
}

#[test]
fn exact_plan_bindings_reject_wrong_units_spaces_state_pairs_and_coverage() {
    let form = derive(ELIMINATED).unwrap();
    let reference = ReferenceCell::simplex(2).unwrap();
    let (fields, rows, time) = inputs(&form, false);
    let mut wrong_fields = fields.clone();
    wrong_fields[0].scale = DynQuantity::new(1.0, DimExponents::DIMENSIONLESS);
    assert!(
        form.bind(reference, &wrong_fields, &rows, Some(&time))
            .unwrap_err()
            .message()
            .contains("Field scale")
    );
    let mut wrong_rows = rows.clone();
    *wrong_rows.values_mut().next().unwrap() = DynQuantity::new(1.0, DimExponents::DIMENSIONLESS);
    assert!(
        form.bind(reference, &fields, &wrong_rows, Some(&time))
            .unwrap_err()
            .message()
            .contains("residual units")
    );
    assert!(
        form.bind(reference, &[], &rows, Some(&time))
            .unwrap_err()
            .message()
            .contains("coverage")
    );
    assert!(
        form.bind(reference, &fields, &rows, None)
            .unwrap_err()
            .message()
            .contains("kinematic pair")
    );
    let mut wrong_time = time.clone();
    wrong_time.step = DynQuantity::new(1.0, dim([0, 1, 0, 0, 0, 0, 0]));
    assert!(
        form.bind(reference, &fields, &rows, Some(&wrong_time))
            .unwrap_err()
            .message()
            .contains("finite time")
    );
    wrong_time = time.clone();
    wrong_time.states.clear();
    assert!(
        form.bind(reference, &fields, &rows, Some(&wrong_time))
            .unwrap_err()
            .message()
            .contains("kinematic pair")
    );
    wrong_time = time.clone();
    wrong_time.states.push(time.states[0]);
    assert!(
        form.bind(reference, &fields, &rows, Some(&wrong_time))
            .unwrap_err()
            .message()
            .contains("Model kinematics")
    );
    wrong_time = time.clone();
    let state = time.states[0];
    wrong_time.states[0] = BackwardEulerStateBinding::new(
        state.pair(),
        Space::simplex_p1_bubble(),
        state.state_scale(),
    );
    assert!(
        form.bind(reference, &fields, &rows, Some(&wrong_time))
            .unwrap_err()
            .message()
            .contains("discrete space")
    );
    wrong_time = time.clone();
    wrong_time.states[0] = BackwardEulerStateBinding::new(
        BackwardEulerStatePair::new(
            state.pair().relation(),
            state.pair().rate(),
            state.pair().state(),
        )
        .unwrap(),
        p1(),
        state.state_scale(),
    );
    assert!(
        form.bind(reference, &fields, &rows, Some(&wrong_time))
            .unwrap_err()
            .message()
            .contains("Model kinematics")
    );
    assert!(
        form.bind(
            ReferenceCell::simplex(3).unwrap(),
            &fields,
            &rows,
            Some(&time)
        )
        .unwrap_err()
        .message()
        .contains("dimension")
    );
}

#[test]
fn previous_data_is_exactly_consumed_physical_fields_and_checked_before_evaluation() {
    let form = derive(MIXED).unwrap();
    let bound = bound(&form, false);
    let required = bound.previous_fields();
    assert_eq!(required.len(), 1);
    let field = *required.keys().next().unwrap();
    let quadrature = simplex_duffy_gauss_legendre(2, 4).unwrap();
    let valid = BTreeMap::from([(field, vec![0.0; 6])]);
    for wrong in [
        BTreeMap::new(),
        BTreeMap::from([(field, vec![0.0; 5])]),
        BTreeMap::from([(field, vec![f64::NAN; 6])]),
    ] {
        assert!(
            bound
                .prepare_cell(&geometry(), &quadrature)
                .and_then(|cell| cell.evaluate(&wrong))
                .unwrap_err()
                .message()
                .contains("exact consumed Field coverage")
        );
    }
    let mut extra = valid.clone();
    let pressure = bound
        .fields()
        .iter()
        .find(|layout| layout.field != field)
        .unwrap();
    extra.insert(pressure.field, vec![0.0; 3]);
    assert!(
        bound
            .prepare_cell(&geometry(), &quadrature)
            .and_then(|cell| cell.evaluate(&extra))
            .unwrap_err()
            .message()
            .contains("exact consumed Field coverage")
    );
    bound
        .prepare_cell(&geometry(), &quadrature)
        .and_then(|cell| cell.evaluate(&valid))
        .unwrap();
    let (fields, rows, _) = inputs(&form, false);
    assert!(
        form.bind(ReferenceCell::simplex(2).unwrap(), &fields, &rows, None)
            .unwrap_err()
            .message()
            .contains("explicit Backward Euler")
    );
}

#[test]
fn nonlinear_and_coordinate_dependent_outer_multipliers_are_rejected() {
    let nonlinear = MIXED
        .replace(
            "parameter density",
            "parameter reference: kg / (m * s ^ 2) = 1; parameter density",
        )
        .replace(
            "2 * viscosity * symmetric_part",
            "2 * viscosity * (1 + p / reference) * symmetric_part",
        );
    assert!(
        derive(&nonlinear)
            .unwrap_err()
            .message()
            .contains("nonlinear unknown-dependent")
    );
    let product = MIXED
        .replace(
            "parameter density",
            "parameter length: m = 1; parameter density",
        )
        .replace("- div(2", "- (coordinate(0) / length) * div(2");
    assert!(
        derive(&product)
            .unwrap_err()
            .message()
            .contains("product-rule")
    );
}

#[test]
fn coefficient_chains_and_mixed_rows_ignore_names_and_declaration_order() {
    let with_data = MIXED.replace("state v:", "variable first: kg / (m * s ^ 2) on body; variable load: kg / (m * s ^ 2) on body; parameter zero: kg / (m * s ^ 2) = 0; state v:")
        .replace("relation balance", "relation first_definition on body { first - zero = 0; } relation load_definition on body { load - first = 0; } relation balance")
        .replace("isotropic_lift(p)) = 0", "isotropic_lift(p)) - grad(load) = 0");
    let reordered = with_data.replace(
        "relation first_definition on body { first - zero = 0; } relation load_definition on body { load - first = 0; }",
        "relation load_definition on body { load - first = 0; } relation first_definition on body { first - zero = 0; }",
    ).replace("state v: vector<m / s, 2> on body;\n variable p: kg / (m * s ^ 2) on body;",
        "variable p: kg / (m * s ^ 2) on body; state v: vector<m / s, 2> on body;")
        .replace(" v:", " motion:").replace("(v)", "(motion)").replace(" p:", " multiplier:").replace("(p)", "(multiplier)");
    assert_ne!(with_data, reordered);
    let normalized = |source: &str| {
        let form = derive(source).unwrap();
        assert_eq!(form.roles.relations.len(), 4);
        let bound = bound(&form, false);
        let velocity = bound
            .fields()
            .iter()
            .find(|layout| layout.components == 2)
            .unwrap();
        let pressure = bound
            .fields()
            .iter()
            .find(|layout| layout.components == 1)
            .unwrap();
        let order = velocity
            .range
            .clone()
            .chain(pressure.range.clone())
            .collect::<Vec<_>>();
        let local = bound
            .prepare_cell(&geometry(), &simplex_duffy_gauss_legendre(2, 4).unwrap())
            .and_then(|cell| cell.evaluate(&BTreeMap::from([(velocity.field, vec![1.0; 6])])))
            .unwrap();
        let matrix = local.matrix();
        order
            .iter()
            .flat_map(|row| order.iter().map(move |column| matrix[row * 9 + column]))
            .collect::<Vec<_>>()
    };
    let original = normalized(&with_data);
    for (a, b) in original.into_iter().zip(normalized(&reordered)) {
        close(a, b);
    }
}
