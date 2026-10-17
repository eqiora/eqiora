use super::*;
use eqiora_schema::kernel::UnaryMathFunction;

#[test]
fn mixed_composition_preserves_unknown_parameter_and_frozen_roles() {
    // f(x,p,c)=(x*p)^3+c; d_x d_p f=9*x²*p²=324 at (2,3).
    let instructions = [
        Instruction::Read(SymbolSlot(0)),
        Instruction::Read(SymbolSlot(1)),
        Instruction::Read(SymbolSlot(2)),
        Instruction::Mul(ValueId(0), ValueId(1)),
        Instruction::PowI(ValueId(3), 3),
        Instruction::Add(ValueId(4), ValueId(2)),
    ];
    let roots = [ValueId(5)];
    let linearization = ScalarLinearization::bind(
        &instructions,
        &roots,
        3,
        &[2., 3., 7.],
        &[
            DifferentiationRole::Unknown,
            DifferentiationRole::Parameter,
            DifferentiationRole::Frozen,
        ],
    )
    .unwrap();
    let mut output = [0.];
    linearization
        .mixed_jvp(
            RelationTangent::Unknown(&[1.]),
            RelationTangent::Parameter(&[1.]),
            RelationTangent::Both {
                unknown: &[0.],
                parameter: &[0.],
            },
            &mut output,
        )
        .unwrap();
    assert_eq!(output, [324.]);
}

fn product(
    operation: Instruction,
    point: &[f64],
    first: &[f64],
    second: &[f64],
    mixed: &[f64],
) -> Result<f64, Diagnostic> {
    let mut instructions = (0..point.len())
        .map(|i| Instruction::Read(SymbolSlot(u32::try_from(i).unwrap())))
        .collect::<Vec<_>>();
    instructions.push(operation);
    let roots = [ValueId(u32::try_from(point.len()).unwrap())];
    let roles = vec![DifferentiationRole::Unknown; point.len()];
    let linearization =
        ScalarLinearization::bind(&instructions, &roots, point.len(), point, &roles)?;
    let mut output = [0.];
    linearization.mixed_jvp(
        RelationTangent::Unknown(first),
        RelationTangent::Unknown(second),
        RelationTangent::Unknown(mixed),
        &mut output,
    )?;
    Ok(output[0])
}

fn check(
    operation: Instruction,
    point: &[f64],
    first: &[f64],
    second: &[f64],
    mixed: &[f64],
    expected: f64,
) {
    for (left, right) in [(first, second), (second, first)] {
        let actual = product(operation, point, left, right, mixed).unwrap();
        assert!(
            (actual - expected).abs() <= 128. * f64::EPSILON * (1. + expected.abs()),
            "{operation:?}: {actual} != {expected}"
        );
    }
}

#[test]
fn mixed_polynomial_and_quotient_actions_include_input_curvature() {
    // Coefficients of st in f(x + s*h + t*k + st*g), expanded by hand.
    check(
        Instruction::Mul(ValueId(0), ValueId(1)),
        &[2., 3.],
        &[5., 7.],
        &[11., 13.],
        &[17., 19.],
        231.,
    );
    check(
        Instruction::Div(ValueId(0), ValueId(1)),
        &[2., 4.],
        &[3., 5.],
        &[7., 11.],
        &[13., 17.],
        5. / 16.,
    );
    check(
        Instruction::PowI(ValueId(0), 3),
        &[2.],
        &[3.],
        &[5.],
        &[7.],
        264.,
    );
    check(
        Instruction::PowI(ValueId(0), 2),
        &[0.],
        &[3.],
        &[5.],
        &[7.],
        30.,
    );
    check(
        Instruction::PowI(ValueId(0), 3),
        &[0.],
        &[3.],
        &[5.],
        &[7.],
        0.,
    );
}

#[test]
fn mixed_elementary_actions_match_closed_form_derivatives() {
    for (function, point, first, second, mixed, expected) in [
        (UnaryMathFunction::Sqrt, 4., 2., 3., 5., 17. / 16.),
        (UnaryMathFunction::Log, 2., 3., 5., 7., -1. / 4.),
        (UnaryMathFunction::Sin, 0., 3., 5., 7., 7.),
        (UnaryMathFunction::Cos, 0., 3., 5., 7., -15.),
        (UnaryMathFunction::Exp, 0., 3., 5., 7., 22.),
    ] {
        check(
            Instruction::Math(function, ValueId(0)),
            &[point],
            &[first],
            &[second],
            &[mixed],
            expected,
        );
    }
}

#[test]
fn mixed_complex_quotient_matches_rational_series_coefficient() {
    // st coefficient: mz/w -(dz*ew+ez*dw+z*mw)/w² +2*z*dw*ew/w³.
    // Exact rational arithmetic gives (459 - 212i)/125.
    // A common numerator/denominator scale cancels from every series term.
    // Squaring the physical denominator would overflow or underflow here.
    for scale in [1., 1e-200, 1e200] {
        for (imaginary, expected) in [(false, 459. / 125.), (true, -212. / 125.)] {
            check(
                Instruction::ComplexDiv {
                    operands: [ValueId(0), ValueId(1), ValueId(2), ValueId(3)],
                    imaginary,
                },
                &[1., 2., 2., 1.].map(|value| value * scale),
                &[3., -1., 1., 2.].map(|value| value * scale),
                &[-2., 4., 2., -1.].map(|value| value * scale),
                &[1., -3., -1., 2.].map(|value| value * scale),
                expected,
            );
        }
    }
}

#[test]
fn mixed_map_actions_match_two_by_two_polynomial_and_rational_expansions() {
    let map = |component| Instruction::MapInvariant {
        start: ValueId(0),
        extent: 2,
        component,
    };
    let h = [1., 2., -1., 3.];
    let k = [2., -3., 4., 1.];
    let g = [-2., 1., 3., -1.];
    // det(A)=ad-bc, inverse(A)[0,0]=d/(ad-bc).
    check(map(None), &[2., 1., 3., 4.], &h, &k, &g, -20.);
    check(map(Some(0)), &[2., 1., 3., 4.], &h, &k, &g, 29. / 5.);
    // A singular determinant remains differentiable.
    check(map(None), &[1., 2., 2., 4.], &h, &k, &g, -21.);
    assert!(
        product(
            map(Some(0)),
            &[1., 2., 2., 4.],
            &[0.; 4],
            &[0.; 4],
            &[0.; 4]
        )
        .is_err()
    );
}

#[test]
fn mixed_actions_keep_domain_and_direction_admission() {
    for function in [UnaryMathFunction::Sqrt, UnaryMathFunction::Log] {
        assert!(
            product(
                Instruction::Math(function, ValueId(0)),
                &[0.],
                &[0.],
                &[0.],
                &[0.]
            )
            .is_err()
        );
    }
    let square = Instruction::PowI(ValueId(0), 2);
    assert!(product(square, &[1.], &[], &[0.], &[0.]).is_err());
    assert!(product(square, &[1.], &[0.], &[f64::NAN], &[0.]).is_err());
    assert!(product(square, &[1.], &[0.], &[0.], &[f64::INFINITY]).is_err());
}
