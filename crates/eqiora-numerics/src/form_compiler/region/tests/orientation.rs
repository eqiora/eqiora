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
    let local = bound.prepare_affine(&geometry(), &rule).unwrap();
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
    let local = forced.prepare_affine(&geometry(), &rule).unwrap();
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
