//! Explicit Helmholtz and zero-reaction diffusion specializations of the same owner.
use super::*;

pub(super) fn check() {
    // -6 u'' + (-1+i)u = (-4-2i)+(-1+3i)x, u=1+3i+(2-i)x.
    // The right outward flux is 6*(2-i)=12-6i. No harmonic ansatz is inferred.
    let helmholtz = replace_source(&[
        ("math.complex(6[m^2],6[m^2])", "math.complex(6[m^2],0[m^2])"),
        ("math.complex(1,1)", "math.complex(-1,1)"),
        ("math.complex(-2,4)", "math.complex(-4,-2)"),
        (
            "math.complex(3[1/m],1[1/m])",
            "math.complex(-1[1/m],3[1/m])",
        ),
        ("math.complex(18[m],6[m])", "math.complex(12[m],-6[m])"),
    ]);
    let plan = replay_plan(resolve(&helmholtz).unwrap(), &REFERENCE_LINEAR_SOLVER);
    let result = plan
        .as_linear()
        .unwrap()
        .run_result(&REFERENCE_LINEAR_SOLVER)
        .unwrap();
    let (_, values, shape) = result.field_block(0, 0).unwrap();
    assert_eq!(shape, &[3]);
    assert_eq!(values.len(), 6);
    for (actual, expected) in
        values
            .as_chunks::<2>()
            .0
            .iter()
            .zip([C::new(1., 3.), C::new(7., 0.), C::new(13., -3.)])
    {
        assert!((C::new(actual[0], actual[1]) - expected).norm() < 1e-10);
    }
    let bytes = result.to_bytes().unwrap();
    assert_eq!(
        crate::CommonResult::from_bytes(&bytes, &plan)
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes
    );

    // -6 u''=0, u(0)=1, 6 u'(6)=12 has the unique affine solution 1+2x.
    let diffusion = replace_source(&[
        ("complex<m^2>", "m^2"),
        ("complex<1/m>", "1/m"),
        ("complex<m>", "m"),
        ("complex<1>", "1"),
        ("math.complex(6[m^2],6[m^2])", "6[m^2]"),
        ("math.complex(1,1)", "0"),
        ("math.complex(-2,4)", "0"),
        ("math.complex(3[1/m],1[1/m])", "0[1/m]"),
        ("math.complex(18[m],6[m])", "12[m]"),
        ("math.complex(1,3)", "1"),
    ]);
    let plan = replay_plan(resolve(&diffusion).unwrap(), &REFERENCE_LINEAR_SOLVER);
    let result = plan
        .as_linear()
        .unwrap()
        .run_result(&REFERENCE_LINEAR_SOLVER)
        .unwrap();
    let (_, values, shape) = result.field_block(0, 0).unwrap();
    assert_eq!(shape, &[3]);
    assert_eq!(values.len(), 3);
    for (actual, expected) in values.iter().zip([1., 7., 13.]) {
        assert!((actual - expected).abs() < 1e-10);
    }
}

fn replace_source(replacements: &[(&str, &str)]) -> String {
    replacements
        .iter()
        .copied()
        .fold(SOURCE.to_owned(), |source, (from, to)| {
            assert!(
                source.contains(from),
                "profile lost its source coefficient: {from}"
            );
            source.replace(from, to)
        })
}
