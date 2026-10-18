use super::*;

#[test]
fn determinant_mixed_action_keeps_the_complete_scaled_row() {
    use eqiora_core::{DimExponents, DynQuantity};
    let constant = |x| Instruction::Constant(DynQuantity::new(x, DimExponents::DIMENSIONLESS));
    // A(p,q)=[[1+B²p, s²p/2], [B²q, 1+s²q]], B=2^700, s=1/B.
    // det(A)=1+B²p+s²q+pq/2. At p=q=0 all primal intermediates
    // are finite and D_p D_q det(A)=1/2 exactly. Each directional row
    // contains both 2^1400 and 2^-1400: normalizing it into f64 loses
    // a nonzero entry even though the requested mixed action fits.
    let instructions = [
        Instruction::Read(SymbolSlot(0)),
        Instruction::Read(SymbolSlot(1)),
        constant(2f64.powi(700)),
        constant(2f64.powi(-700)),
        constant(0.5),
        constant(1.),
        Instruction::Mul(ValueId(0), ValueId(2)),
        Instruction::Mul(ValueId(6), ValueId(2)),
        Instruction::Mul(ValueId(0), ValueId(3)),
        Instruction::Mul(ValueId(8), ValueId(3)),
        Instruction::Mul(ValueId(9), ValueId(4)),
        Instruction::Mul(ValueId(1), ValueId(2)),
        Instruction::Mul(ValueId(11), ValueId(2)),
        Instruction::Mul(ValueId(1), ValueId(3)),
        Instruction::Mul(ValueId(13), ValueId(3)),
        Instruction::Add(ValueId(5), ValueId(7)),
        Instruction::Mul(ValueId(10), ValueId(5)),
        Instruction::Mul(ValueId(12), ValueId(5)),
        Instruction::Add(ValueId(5), ValueId(14)),
        Instruction::MapInvariant {
            start: ValueId(15),
            extent: 2,
            output: MapOutput::Determinant,
        },
    ];
    let action = ScalarLinearization::bind(
        &instructions,
        &[ValueId(19)],
        2,
        &[0., 0.],
        &[DifferentiationRole::Parameter; 2],
    )
    .unwrap();
    let mut output = [0.];
    action.primal(&mut output).unwrap();
    assert_eq!(output, [1.]);
    action
        .mixed_jvp(
            RelationTangent::Parameter(&[1., 0.]),
            RelationTangent::Parameter(&[0., 1.]),
            RelationTangent::Parameter(&[0., 0.]),
            &mut output,
        )
        .unwrap();
    assert_eq!(output, [0.5]);
}

#[test]
fn triangular_inverse_and_cofactor_retain_small_elimination_products() {
    // A=s*I+t*N, N01=N12=1 and N³=0. Thus
    // A^-1=I/s-t*N/s²+t²*N²/s³, and C20=t².
    let power = |exponent: i32| {
        if exponent < -1074 {
            0.
        } else if exponent < -1022 {
            f64::from_bits(1u64 << (1074 + exponent))
        } else {
            f64::from_bits(((exponent + 1023) as u64) << 52)
        }
    };
    for (s_exp, t_exp) in [(-500, -1050), (300, -400)] {
        let (s, t) = (power(s_exp), power(t_exp));
        let mut map = MapEvaluation::new(&[s, t, 0., 0., s, t, 0., 0., s], 3).unwrap();
        assert_eq!(
            map.value(MapOutput::Inverse(2)).unwrap(),
            power(2 * t_exp - 3 * s_exp)
        );
        assert_eq!(map.value(MapOutput::Cofactor(6)).unwrap(), power(2 * t_exp));
    }
}

#[test]
fn inverse_weighted_actions_retain_range_until_the_requested_output() {
    use eqiora_core::{DimExponents, DynQuantity};
    let constant = |x| Instruction::Constant(DynQuantity::new(x, DimExponents::DIMENSIONLESS));
    // A(p)=s*(1+p)*I has condition number one, y=B00=1/(s*(1+p)).
    // At p=0, y=1/s and y'=-1/s are exactly representable, even when
    // the unit partial -1/s² overflows or underflows before contraction.
    let mut failures = Vec::new();
    for exponent in [-400, 400, -700, 700] {
        let scale = 2f64.powi(exponent);
        let instructions = [
            Instruction::Read(SymbolSlot(0)),
            constant(1.),
            constant(scale),
            Instruction::Add(ValueId(0), ValueId(1)),
            Instruction::Mul(ValueId(2), ValueId(3)),
            constant(0.),
            constant(0.),
            Instruction::Mul(ValueId(2), ValueId(3)),
            Instruction::MapInvariant {
                start: ValueId(4),
                extent: 2,
                output: MapOutput::Inverse(0),
            },
        ];
        let action = ScalarLinearization::bind(
            &instructions,
            &[ValueId(8)],
            1,
            &[0.],
            &[DifferentiationRole::Parameter],
        )
        .unwrap();
        let (mut primal, mut forward, mut reverse) = ([0.], [0.], [0.]);
        action.primal(&mut primal).unwrap();
        assert_eq!(primal, [1. / scale]);
        let jvp = action.jvp(RelationTangent::Parameter(&[1.]), &mut forward);
        let vjp = action.vjp(&[1.], RelationCotangent::Parameter(&mut reverse));
        let mut mixed = [0.];
        let second = action.mixed_jvp(
            RelationTangent::Parameter(&[1.]),
            RelationTangent::Parameter(&[1.]),
            RelationTangent::Parameter(&[0.]),
            &mut mixed,
        );
        if jvp.is_err()
            || vjp.is_err()
            || second.is_err()
            || forward != [-1. / scale]
            || reverse != [-1. / scale]
            || mixed != [2. / scale]
        {
            failures.push(format!(
                "scale=2^{exponent}; JVP: {jvp:?}, {forward:?}; VJP: {vjp:?}, {reverse:?}; mixed: {second:?}, {mixed:?}"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn selected_cofactor_and_determinant_actions_do_not_require_other_cofactors() {
    // diag(s,s,0) has cofactors diag(0,0,s²). Only the last exceeds
    // binary64. Its first cofactor is zero, with dC00/dA22=s.
    let scale = 2f64.powi(700);
    let mut map = MapEvaluation::new(&[scale, 0., 0., 0., scale, 0., 0., 0., 0.], 3).unwrap();
    assert_eq!(map.value(MapOutput::Cofactor(0)).unwrap(), 0.);
    assert_eq!(
        map.gradient(MapOutput::Determinant, &|index| index == 0)
            .unwrap()
            .into_iter()
            .map(Scaled::value)
            .collect::<Vec<_>>(),
        [0.; 9]
    );
    let gradient = map
        .gradient(MapOutput::Cofactor(0), &|index| index == 8)
        .unwrap();
    // The selected 1x1 minor is the exact power of two s.
    assert_eq!(gradient[8].value(), scale);
    assert!(map.value(MapOutput::Cofactor(8)).is_err());
    assert!(
        !map.gradient(MapOutput::Determinant, &|_| true).unwrap()[8]
            .value()
            .is_finite()
    );
    assert_eq!(map.value(MapOutput::Cofactor(0)).unwrap(), 0.);
}

#[test]
fn inverse_actions_do_not_materialize_unrequested_matrix_partials() {
    // A(p)=[[p,s],[s,0]] has B00=0 identically. The inverse is well
    // conditioned at p=0 and p=s, while dB00/dA11=-1/s² need not fit f64.
    let instructions = [
        Instruction::Read(SymbolSlot(0)),
        Instruction::Read(SymbolSlot(1)),
        Instruction::Read(SymbolSlot(2)),
        Instruction::Read(SymbolSlot(3)),
        Instruction::MapInvariant {
            start: ValueId(0),
            extent: 2,
            output: MapOutput::Inverse(0),
        },
    ];
    for exponent in [-400, -700] {
        let scale = 2f64.powi(exponent);
        for parameter in [0., scale] {
            let action = ScalarLinearization::bind(
                &instructions,
                &[ValueId(4)],
                4,
                &[parameter, scale, scale, 0.],
                &[
                    DifferentiationRole::Parameter,
                    DifferentiationRole::Frozen,
                    DifferentiationRole::Frozen,
                    DifferentiationRole::Frozen,
                ],
            )
            .unwrap();
            let (mut primal, mut forward, mut reverse) = ([0.], [0.], [0.]);
            action.primal(&mut primal).unwrap();
            let jvp = action.jvp(RelationTangent::Parameter(&[1.]), &mut forward);
            let vjp = action.vjp(&[1.], RelationCotangent::Parameter(&mut reverse));
            assert!(jvp.is_ok() && vjp.is_ok(), "JVP: {jvp:?}; VJP: {vjp:?}");
            assert_eq!((primal, forward, reverse), ([0.], [0.], [0.]));
        }
    }
}

#[test]
fn inverse_projection_retains_primal_checks_without_unused_derivative_outputs() {
    use eqiora_core::{DimExponents, DynQuantity};
    let constant = |x| Instruction::Constant(DynQuantity::new(x, DimExponents::DIMENSIONLESS));
    for exponent in [-400, -700] {
        let scale = 2f64.powi(exponent);
        let mut instructions = vec![
            Instruction::Read(SymbolSlot(0)),
            constant(scale),
            constant(0.),
            constant(1.),
            Instruction::Mul(ValueId(0), ValueId(3)),
            Instruction::Mul(ValueId(0), ValueId(2)),
            Instruction::Add(ValueId(2), ValueId(4)),
            Instruction::Add(ValueId(1), ValueId(5)),
            Instruction::Add(ValueId(1), ValueId(5)),
            Instruction::Add(ValueId(2), ValueId(5)),
        ];
        for component in 0..4 {
            instructions.push(Instruction::MapInvariant {
                start: ValueId(6),
                extent: 2,
                output: MapOutput::Inverse(component),
            });
        }
        // Preserve the ordinary expansion of e^T B e, e=[1,0], including
        // zero products. y=p+e^T B e=p, so both derivative products are one.
        instructions.extend([
            Instruction::Mul(ValueId(10), ValueId(3)),
            Instruction::Mul(ValueId(11), ValueId(2)),
            Instruction::Add(ValueId(14), ValueId(15)),
            Instruction::Mul(ValueId(12), ValueId(3)),
            Instruction::Mul(ValueId(13), ValueId(2)),
            Instruction::Add(ValueId(17), ValueId(18)),
            Instruction::Mul(ValueId(16), ValueId(3)),
            Instruction::Mul(ValueId(19), ValueId(2)),
            Instruction::Add(ValueId(20), ValueId(21)),
            Instruction::Add(ValueId(0), ValueId(22)),
        ]);
        for parameter in [0., scale] {
            let action = ScalarLinearization::bind(
                &instructions,
                &[ValueId(23)],
                1,
                &[parameter],
                &[DifferentiationRole::Parameter],
            )
            .unwrap();
            let (mut primal, mut forward, mut reverse) = ([0.], [0.], [0.]);
            action.primal(&mut primal).unwrap();
            let jvp = action.jvp(RelationTangent::Parameter(&[1.]), &mut forward);
            let vjp = action.vjp(&[1.], RelationCotangent::Parameter(&mut reverse));
            assert!(jvp.is_ok() && vjp.is_ok(), "JVP: {jvp:?}; VJP: {vjp:?}");
            assert_eq!((primal, forward, reverse), ([parameter], [1.], [1.]));
        }
    }
}

#[test]
fn typed_maps_preserve_coefficient_dimensions() {
    use eqiora_core::{DimExponents, DynQuantity, ValueLiteral};
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    for n in [1u32, 3] {
        let values = (0..n * n)
            .map(|i| {
                Some(
                    ValueLiteral::try_from(DynQuantity::new(
                        if i / n == i % n { 2. } else { 0. },
                        length,
                    ))
                    .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let mut cache = MapCache::default();
        let det = cache
            .typed(ValueId(0), n, MapOutput::Determinant, &values, values.len())
            .unwrap()
            .real_scalar_value()
            .unwrap();
        // The diagonal determinant is 2^n. Its floating evaluation need not
        // be bit-exact; allow a small rounding budget, independently of units.
        assert!((det.value() / 2f64.powi(n as i32) - 1.).abs() < 16. * f64::EPSILON);
        assert_eq!(det.dim(), length.pow(n as i32, 1).unwrap());
        for i in 0..n * n {
            let inverse = cache
                .typed(ValueId(0), n, MapOutput::Inverse(i), &values, values.len())
                .unwrap()
                .real_scalar_value()
                .unwrap();
            assert_eq!(inverse.value(), if i / n == i % n { 0.5 } else { 0. });
            assert_eq!(inverse.dim(), length.pow(-1, 1).unwrap());
            let cofactor = cache
                .typed(ValueId(0), n, MapOutput::Cofactor(i), &values, values.len())
                .unwrap()
                .real_scalar_value()
                .unwrap();
            let expected = if i / n == i % n {
                2f64.powi(n as i32 - 1)
            } else {
                0.
            };
            assert!((cofactor.value() - expected).abs() < 16. * f64::EPSILON * expected.max(1.));
            assert_eq!(cofactor.dim(), length.pow(n as i32 - 1, 1).unwrap());
        }
        if n > 1 {
            let mut mismatched = values.clone();
            mismatched[1] = Some(
                ValueLiteral::try_from(DynQuantity::new(0., DimExponents::DIMENSIONLESS)).unwrap(),
            );
            assert!(
                MapCache::default()
                    .typed(
                        ValueId(0),
                        n,
                        MapOutput::Determinant,
                        &mismatched,
                        mismatched.len()
                    )
                    .unwrap_err()
                    .message()
                    .contains("units differ")
            );
        }
    }
}

#[test]
fn orientation_does_not_require_a_representable_volume_factor() {
    use eqiora_schema::kernel::CoordinateMapFactor::{Orientation, VolumeScale};
    for n in [2, 6, 9, 16] {
        let mut entries = vec![0.0; n * n];
        for i in 0..n {
            entries[i * n + i] = 1e-200;
        }
        entries[0] = -1e-200;
        assert_eq!(
            ScalarOperatorIr::coordinate_map_factor(&entries, n, Orientation).unwrap(),
            -1.0
        );
        assert!(
            ScalarOperatorIr::coordinate_map_factor(&entries, n, VolumeScale)
                .unwrap_err()
                .message()
                .contains("underflows")
        );
    }
}

#[test]
fn normalization_cannot_turn_nonzero_coefficients_into_singular_success() {
    // Independent diagonal multiplication is finite even though normalization
    // by the largest entry would underflow the second entry in plain binary64.
    let (large, small) = (1e308, 1e-308);
    let mut map = MapEvaluation::new(&[large, 0., 0., small], 2).unwrap();
    assert!((map.value(MapOutput::Determinant).unwrap() - large * small).abs() < 4. * f64::EPSILON);
    // Retaining range does not waive the inverse conditioning requirement.
    assert!(
        map.value(MapOutput::Inverse(0))
            .unwrap_err()
            .message()
            .contains("ill-conditioned")
    );
}

#[test]
fn determinant_cofactors_survive_intermediate_underflow() {
    let mut map =
        MapEvaluation::new(&[1e-200, 0.0, 0.0, 0.0, 1e-200, 0.0, 0.0, 0.0, 1e100], 3).unwrap();
    let gradient = map.gradient(MapOutput::Determinant, &|_| true).unwrap();
    // Independently, diagonal cofactors are products of the other entries:
    // 1e-100, 1e-100, 1e-400 (the last is unrepresentable in binary64).
    for index in [0, 4] {
        assert!((gradient[index].value() / 1e-100 - 1.0).abs() < 1e-12);
    }
    for index in [1, 2, 3, 5, 6, 7, 8] {
        assert_eq!(gradient[index].value(), 0.0);
    }
}

#[test]
fn factorization_work_budget_precedes_arithmetic_and_has_an_admitted_neighbor() {
    // The reference work bound is 2*n^3, not a mathematical extent rule:
    // 2*79^3=986078 <=1e6, while 2*80^3=1024000 >1e6.
    let n = 79;
    let identity = (0..n * n)
        .map(|i| f64::from(i / n == i % n))
        .collect::<Vec<_>>();
    assert_eq!(
        MapEvaluation::new(&identity, n)
            .unwrap()
            .value(MapOutput::Determinant)
            .unwrap(),
        1.0
    );
    let error = match MapEvaluation::new(&vec![f64::NAN; 80 * 80], 80) {
        Err(error) => error,
        Ok(_) => panic!("work budget must reject before numerical work"),
    };
    assert!(
        error
            .message()
            .contains("factorization exceeds one million component products")
    );
}

#[test]
fn retained_cofactor_partials_follow_the_exact_signed_minor() {
    let mut map = MapEvaluation::new(&[2., 1., 7., 3., 4., 8., 5., 6., 9.], 3).unwrap();
    // C_(1,2)=-(a00*a21-a01*a20); its four nonzero partials have
    // different signs and positions in the original nonsymmetric 3x3 map.
    let output = MapOutput::Cofactor(5);
    assert!((map.value(output).unwrap() + 7.).abs() < 1e-12);
    for (actual, expected) in map
        .gradient(output, &|_| true)
        .unwrap()
        .into_iter()
        .zip([-6., 5., 0., 0., 0., 0., 1., -2., 0.])
    {
        assert!((actual.value() - expected).abs() < 1e-12);
    }
    let h = [1., 2., 0., 0., 0., 0., -1., 3., 0.];
    let k = [2., -1., 0., 0., 0., 0., 3., -2., 0.];
    // -(h00*k21+k00*h21-h01*k20-k01*h20)=3.
    assert!(
        (map.curvature(output, &h.map(Scaled::new), &k.map(Scaled::new))
            .unwrap()
            .value()
            - 3.)
            .abs()
            < 1e-12
    );
    let mut scalar = MapEvaluation::new(&[0.], 1).unwrap();
    assert_eq!(scalar.value(MapOutput::Cofactor(0)).unwrap(), 1.);
    assert_eq!(
        scalar
            .gradient(MapOutput::Cofactor(0), &|_| true)
            .unwrap()
            .into_iter()
            .map(Scaled::value)
            .collect::<Vec<_>>(),
        [0.]
    );
    assert_eq!(
        scalar
            .curvature(
                MapOutput::Cofactor(0),
                &[Scaled::new(1.)],
                &[Scaled::new(2.)]
            )
            .unwrap()
            .value(),
        0.
    );
}
