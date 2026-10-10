mod solve;
use super::*;
use crate::form_compiler::region::RegionFieldBinding;
use eqiora_core::{DimExponents, DynQuantity};
use eqiora_meshing::simplex_duffy_gauss_legendre;
use eqiora_realization::Space;
use num_complex::Complex64 as C;

fn source(complex: bool, face: bool, first_law: Option<&str>, boundaries: usize) -> String {
    let (scalar, parameter) = if complex {
        ("complex<1>", "complex<m^2> = math.complex(2[m^2], -1[m^2])")
    } else {
        ("1", "m^2 = 2[m^2]")
    };
    let differential = if face {
        "-grad(div(u))"
    } else {
        "curl(curl(u))"
    };
    let natural = if face {
        "normal(a*isotropic_lift(div(u))) = 0"
    } else {
        "tangential_trace(-a*curl(u)) = 0"
    };
    let mut source = format!(
        "model MomentBlock() {{
        domain body = box(0,2,0,3,0,4);
        parameter a: {parameter};
        variable potential: m^2 on body in smooth;
        relation potential_definition on body {{ potential = coordinate(1)^2; }}
        variable u: vector<{scalar},3> on body in smooth;
        relation balance on body {{ a*({differential}) = 0; }}
    "
    );
    for index in 0..boundaries {
        let side = if index % 2 == 0 { "lower" } else { "upper" };
        let law = if index == 0 {
            first_law.unwrap_or(natural)
        } else {
            natural
        };
        source += &format!(
            "domain side{index} = boundary(body,axis={},side={side});
            relation law{index} on side{index} {{ {law}; }}",
            index / 2
        );
    }
    source + "}"
}

fn derive<S: Coefficient>(source: &str) -> Result<CompiledLinearBlockForm<S>, Diagnostic> {
    let program = super::program(source);
    let domain = program
        .nodes()
        .find_map(|node| match node {
            KernelNode::Domain(domain)
                if matches!(domain.kind(), DomainKind::CartesianBox { .. }) =>
            {
                Some(domain.id().erase())
            }
            _ => None,
        })
        .unwrap();
    CompiledLinearBlockForm::derive(&program, domain, 3, &BTreeSet::new())
}

fn bind<S: Coefficient>(
    form: &CompiledLinearBlockForm<S>,
    face: bool,
) -> Result<BoundRegionForm<S>, Diagnostic> {
    // Residual units are 1, volume has units m³, and moment test functions
    // have units m⁻¹ or m⁻². These exponents are independently specified.
    let power = if face { 2 } else { 1 };
    let expected = form.bind_volume(
        ReferenceCell::simplex(3)?,
        &[RegionFieldBinding {
            field: form.fields()[0].0,
            space: if face {
                Space::tetrahedral_face()
            } else {
                Space::tetrahedral_edge()
            },
            scale: DynQuantity::new(
                1.,
                DimExponents::from_integers([0, power, 0, 0, 0, 0, 0]).unwrap(),
            ),
        }],
        &BTreeMap::from([(
            form.relations[0],
            DynQuantity::new(
                1.,
                DimExponents::from_integers([0, power - 3, 0, 0, 0, 0, 0]).unwrap(),
            ),
        )]),
    )?;
    let actual = form.bind_space(
        ReferenceCell::simplex(3)?,
        if face {
            Space::tetrahedral_face()
        } else {
            Space::tetrahedral_edge()
        },
    )?;
    assert_eq!(actual, expected);
    Ok(actual)
}

#[test]
fn si_binding_retains_physical_field_units_in_moment_functionals() {
    for face in [false, true] {
        let source = source(false, face, None, 6).replace("vector<1,3>", "vector<m,3>");
        let form = derive::<f64>(&source).unwrap();
        let (space, power) = if face {
            (Space::tetrahedral_face(), 2)
        } else {
            (Space::tetrahedral_edge(), 1)
        };
        let reference = ReferenceCell::simplex(3).unwrap();
        // A metre-valued vector has edge integrals in m² or face fluxes in m³.
        // Residual m times volume m³ times reciprocal moment-basis units gives
        // row multipliers m⁻³ (edge) or m⁻² (face), independently of tabulation.
        let expected = form
            .bind_volume(
                reference,
                &[RegionFieldBinding {
                    field: form.fields()[0].0,
                    space,
                    scale: DynQuantity::new(
                        1.,
                        DimExponents::from_integers([0, power + 1, 0, 0, 0, 0, 0]).unwrap(),
                    ),
                }],
                &BTreeMap::from([(
                    form.relations[0],
                    DynQuantity::new(
                        1.,
                        DimExponents::from_integers([0, power - 4, 0, 0, 0, 0, 0]).unwrap(),
                    ),
                )]),
            )
            .unwrap();
        assert_eq!(form.bind_space(reference, space).unwrap(), expected);
        let incompatible = if face {
            Space::tetrahedral_edge()
        } else {
            Space::tetrahedral_face()
        };
        assert!(form.bind_space(reference, incompatible).is_err());
        assert!(
            form.bind_space(ReferenceCell::hypercube(3).unwrap(), space)
                .is_err()
        );
    }
}

#[test]
fn complete_linear_blocks_keep_vector_spaces_boundary_inventory_and_complex_phase() {
    let reference = ReferenceCell::simplex(3).unwrap();
    let geometry = AffineGeometryMap::new(
        reference,
        3,
        vec![0.; 3],
        vec![2., 0., 0., 0., 3., 0., 0., 0., 4.],
    )
    .unwrap();
    let rule = simplex_duffy_gauss_legendre(3, 3).unwrap();
    for face in [false, true] {
        let real = derive::<f64>(&source(false, face, None, 6)).unwrap();
        let complex = derive::<C>(&source(true, face, None, 6)).unwrap();
        assert_eq!(real.boundary_laws()[&real.fields()[0].0].len(), 6);
        assert!(!real.is_transient());
        let nodal = real.volume().unwrap();
        assert_eq!(nodal.reference_cell(), ReferenceCell::hypercube(3).unwrap());
        let [layout] = nodal.fields() else {
            panic!("one exact vector Field");
        };
        assert_eq!(layout.field, real.fields()[0].0);
        assert_eq!(layout.value_type, real.fields()[0].1);
        assert_eq!(layout.components, 3);
        assert_eq!(layout.range, 0..24); // Eight Q1 vertices, three components each.
        assert_eq!(
            layout.space,
            Space::continuous_lagrange(std::num::NonZeroU16::MIN)
        );
        let real = bind(&real, face)
            .unwrap()
            .prepare_cell(&geometry, &rule)
            .and_then(|cell| cell.evaluate(&BTreeMap::new()))
            .unwrap();
        let complex = bind(&complex, face)
            .unwrap()
            .prepare_cell(&geometry, &rule)
            .and_then(|cell| cell.evaluate(&BTreeMap::new()))
            .unwrap();
        let coefficients = if face {
            vec![0., 0., 0., 12.]
        } else {
            vec![0., 0., 0., 6., 0., 0.]
        };
        assert_eq!(real.rows(), coefficients.len());
        let action = |matrix: &eqiora_assembly::LocalContribution<C>| {
            matrix
                .matrix()
                .chunks_exact(coefficients.len())
                .map(|row| {
                    row.iter()
                        .zip(&coefficients)
                        .map(|(a, u)| a * *u)
                        .sum::<C>()
                })
                .collect::<Vec<_>>()
        };
        let energy = action(&complex)
            .iter()
            .zip(&coefficients)
            .map(|(a, u)| a * *u)
            .sum::<C>();
        // Volume 4: rotation curl magnitude 2 gives 16; radial divergence 3 gives 36.
        let expected = C::new(2., -1.) * if face { 36. } else { 16. };
        assert!((energy - expected).norm() <= 4096. * f64::EPSILON * expected.norm());
        for (a, b) in real.matrix().iter().zip(complex.matrix()) {
            assert!((*b - C::new(1., -0.5) * *a).norm() <= 4096. * f64::EPSILON * a.abs().max(1.));
        }
    }
}

#[test]
fn moment_blocks_reject_incomplete_essential_and_nonzero_natural_boundary_profiles() {
    for face in [false, true] {
        assert!(derive::<f64>(&source(false, face, None, 5)).is_err());
        let trace = derive::<f64>(&source(false, face, Some("trace(u) = 0"), 6)).unwrap();
        assert!(
            bind(&trace, face)
                .unwrap_err()
                .message()
                .contains("homogeneous natural")
        );
    }
    let nonzero = derive::<f64>(&source(
        false,
        false,
        Some("tangential_trace(-a*curl(u)) = trace(grad(potential))"),
        6,
    ))
    .unwrap();
    assert!(
        bind(&nonzero, false)
            .unwrap_err()
            .message()
            .contains("homogeneous natural")
    );
    let reversed_sign = source(false, false, Some("tangential_trace(a*curl(u)) = 0"), 6);
    let reversed_zero = derive::<f64>(&reversed_sign).unwrap();
    assert!(bind(&reversed_zero, false).is_ok());
    let reversed_nonzero = source(
        false,
        false,
        Some("tangential_trace(a*curl(u)) = trace(grad(potential))"),
        6,
    );
    assert!(derive::<f64>(&reversed_nonzero).is_err());
}

#[test]
fn vector_blocks_cannot_hide_exterior_laws_as_unadmitted_interfaces() {
    let program = super::program(&source(false, false, None, 6));
    let domain = program
        .nodes()
        .find_map(|node| match node {
            KernelNode::Domain(domain)
                if matches!(domain.kind(), DomainKind::CartesianBox { .. }) =>
            {
                Some(domain.id().erase())
            }
            _ => None,
        })
        .unwrap();
    let interfaces = program
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Domain(domain)
                if matches!(domain.kind(), DomainKind::CartesianBoundary { .. }) =>
            {
                Some(domain.id().erase())
            }
            _ => None,
        })
        .collect();
    let error =
        CompiledLinearBlockForm::<f64>::derive(&program, domain, 3, &interfaces).unwrap_err();
    assert!(error.message().contains("interface boundary quotients"));
}
