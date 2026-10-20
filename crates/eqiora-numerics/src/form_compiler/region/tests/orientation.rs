use super::*;

#[test]
fn mixed_volume_orientation_derives_transpose_signs_and_preserves_whole_rows() {
    let form = derive(MIXED).unwrap();
    let (velocity, pressure) = form.constraint_pairs().next().unwrap();
    let signs = form.volume_test_orientations().unwrap();
    assert!(form.has_zero_multiplier_block());
    assert_eq!(signs[&velocity], 1.0);
    assert_eq!(signs[&pressure], -1.0);
    let mut reversed = form.clone();
    let row = reversed
        .rows
        .iter_mut()
        .find(|row| row.tested == pressure)
        .unwrap();
    for term in &mut row.terms {
        term.coefficient = term.coefficient.clone().multiply(Data::constant(2, -1.0));
    }
    // Reversing div(v)=0 changes the required test sign, not its mathematics.
    assert_eq!(reversed.volume_test_orientations().unwrap()[&pressure], 1.0);

    let (fields, rows) = form
        .si_bindings(|field| {
            Ok(if field == velocity {
                Space::simplex_p1_bubble()
            } else {
                p1()
            })
        })
        .unwrap();
    let (_, _, time) = inputs(&form, true);
    let bound = form
        .bind(
            ReferenceCell::simplex(2).unwrap(),
            &fields,
            &rows,
            Some(&time),
        )
        .unwrap();
    let rule = simplex_duffy_gauss_legendre(2, 4).unwrap();
    let local = bound
        .prepare_affine_actions::<Native>(&geometry(), None, &rule)
        .map(|(primal, _)| primal)
        .unwrap();
    let n = local.rhs().len();
    for i in 0..n {
        for j in 0..n {
            assert!((local.matrix()[i * n + j] - local.matrix()[j * n + i]).abs() < 1e-12);
        }
    }
    // A source in the constraint must reverse together with its operator.
    let mut forced = form.clone();
    forced
        .rows
        .iter_mut()
        .find(|row| row.tested == pressure)
        .unwrap()
        .forcing = vec![Data::constant(2, 3.0)];
    let forced = forced
        .bind(
            ReferenceCell::simplex(2).unwrap(),
            &fields,
            &rows,
            Some(&time),
        )
        .unwrap();
    let local = forced
        .prepare_affine_actions::<Native>(&geometry(), None, &rule)
        .map(|(primal, _)| primal)
        .unwrap();
    let pressure = forced
        .fields()
        .iter()
        .find(|field| field.field == pressure)
        .unwrap();
    // Unit right triangle has area 1/2 and integral of each P1 basis is 1/6.
    for &value in &local.rhs()[pressure.range.clone()] {
        assert!((value + 0.5).abs() < 1e-12);
    }
}

#[test]
fn volume_orientation_declines_missing_or_nonreciprocal_cross_blocks() {
    let form = derive(MIXED).unwrap();
    let (_, pressure) = form.constraint_pairs().next().unwrap();
    for mode in 0..3 {
        let mut bad = form.clone();
        let row = bad
            .rows
            .iter_mut()
            .find(|row| row.tested == pressure)
            .unwrap();
        match mode {
            0 => row.terms.clear(),
            1 => row.terms[0].coefficient = Data::constant(2, 2.0),
            _ => row.terms[0].derivative = true,
        }
        assert!(bad.volume_test_orientations().is_none());
    }
    let mut stabilized = form.clone();
    let row = stabilized
        .rows
        .iter_mut()
        .find(|row| row.tested == pressure)
        .unwrap();
    row.terms.push(Term {
        trial: pressure,
        derivative: false,
        pairing: Pairing::Value,
        coefficient: Data::constant(2, 1.0),
        positive_diffusion: false,
    });
    // Symmetry alone does not prove saddle indefiniteness after a pressure penalty.
    assert!(stabilized.volume_test_orientations().is_some());
    assert!(!stabilized.has_zero_multiplier_block());
}

#[test]
fn eliminated_states_preserve_reciprocity_only_at_matching_timestep_powers() {
    let elastic = derive(ELIMINATED).unwrap();
    assert!(
        elastic
            .volume_test_orientations()
            .unwrap()
            .values()
            .all(|sign| *sign == 1.0)
    );
    let form = derive(
        "model CoupledRates() {
        domain body = box(0, 1, 0, 1);
        parameter rho: kg / m ^ 3 = 3;
        parameter k: kg / (m ^ 3 * s ^ 2) = 2;
        state d: vector<m, 2> on body;
        state e: vector<m, 2> on body;
        state u: vector<m / s, 2> on body;
        state v: vector<m / s, 2> on body;
        relation first_pair on body { derivative(d) - u = 0; }
        relation second_pair on body { derivative(e) - v = 0; }
        relation first_balance on body { rho * derivative(u) + k * e = 0; }
        relation second_balance on body { rho * derivative(v) + k * d = 0; }
    }",
    )
    .unwrap();
    assert!(form.volume_test_orientations().is_some());
    let states = form
        .roles
        .relations
        .values()
        .filter_map(|role| match role.kind {
            Role::Kinematic { state, .. } => Some(state),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    for mutation in 0..3 {
        let mut invalid = form.clone();
        let row = &mut invalid.rows[0];
        let index = row
            .terms
            .iter()
            .position(|term| states.contains(&term.trial))
            .unwrap();
        match mutation {
            0 => row.terms[index].derivative = true,
            1 => {
                row.terms[index].coefficient = row.terms[index]
                    .coefficient
                    .clone()
                    .multiply(Data::constant(2, 2.0))
            }
            _ => {
                row.terms.remove(index);
            }
        }
        assert!(invalid.volume_test_orientations().is_none());
    }
}

#[test]
fn congruence_binding_applies_field_scales_to_both_matrix_axes() {
    use eqiora_realization::{AlgebraicBlockScale, SymmetricCongruenceScaling};
    use eqiora_solver::AlgebraicBlock;
    let form = derive(MIXED).unwrap();
    let (velocity, _) = form.constraint_pairs().next().unwrap();
    let scale_value = |field| if field == velocity { 2.0 } else { 3.0 };
    let functional_dimension = DimExponents::from_integers([1, 1, -3, 0, 0, 0, 0]).unwrap();
    let scaling = SymmetricCongruenceScaling::new(
        form.fields().map(|(field, ty)| {
            AlgebraicBlockScale::new(
                AlgebraicBlock::Field(field.downcast().unwrap()),
                PositivePhysicalScale::new(DynQuantity::new(scale_value(field), ty.dimension()))
                    .unwrap(),
            )
        }),
        PositivePhysicalScale::new(DynQuantity::new(5.0, functional_dimension)).unwrap(),
    )
    .unwrap();
    let space = |field| {
        Ok(if field == velocity {
            Space::simplex_p1_bubble()
        } else {
            p1()
        })
    };
    let (unit_fields, unit_rows) = form.si_bindings(space).unwrap();
    let (scaled_fields, scaled_rows) = form
        .congruence_bindings(space, &scaling, &form.volume_test_orientations().unwrap())
        .unwrap();
    let (_, _, time) = inputs(&form, true);
    let reference = ReferenceCell::simplex(2).unwrap();
    let unit = form
        .bind(reference, &unit_fields, &unit_rows, Some(&time))
        .unwrap();
    let scaled = form
        .bind(reference, &scaled_fields, &scaled_rows, Some(&time))
        .unwrap();
    let rule = simplex_duffy_gauss_legendre(2, 4).unwrap();
    let unscaled_local = unit
        .prepare_affine_actions::<Native>(&geometry(), None, &rule)
        .map(|(primal, _)| primal)
        .unwrap();
    let scaled_local = scaled
        .prepare_affine_actions::<Native>(&geometry(), None, &rule)
        .map(|(primal, _)| primal)
        .unwrap();
    let n = unscaled_local.rhs().len();
    let scales = unit
        .fields()
        .iter()
        .flat_map(|field| std::iter::repeat_n(scale_value(field.field), field.range.len()))
        .collect::<Vec<_>>();
    // Substituting x = S x_hat and testing with S/W gives S A S/W.
    // This identity is derived before evaluating either assembled matrix.
    for i in 0..n {
        for j in 0..n {
            let expected = scales[i] * unscaled_local.matrix()[i * n + j] * scales[j] / 5.0;
            assert!((scaled_local.matrix()[i * n + j] - expected).abs() < 1e-11);
        }
    }
}
