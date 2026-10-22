use super::*;
use crate::parameter_point::ParameterPoint;
use eqiora_meshing::{EntityIncidence, MeshEntity, OrientationCode, QuadratureRule};
use num_complex::Complex64 as C;

const HELMHOLTZ: &str = "model Wave() {
 domain body = box(0, 6);
 domain wall = boundary(body, axis=0, side=lower);
 parameter a: complex<m^2> = math.complex(6[m^2], 6[m^2]);
 parameter q: complex<1> = math.complex(3, -1);
 parameter f: complex<1> = math.complex(1, 3);
 variable u: complex<1> on body in smooth;
 relation balance on body { -div(a*grad(u)) + q*u = f; }
 relation law on wall { normal(a*grad(u)) = math.complex(-2[m], 4[m]); }
}";

fn fixture(source: &str) -> (KernelProgram, BTreeMap<String, RawId>) {
    let (transaction, model, symbols) = compile("complex-region.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let ids = symbols
        .iter()
        .map(|(name, id)| (name.to_owned(), id))
        .collect();
    (program, ids)
}

fn scalar_bound(form: &CompiledRegionForm<C>) -> Result<BoundRegionForm<C>, Diagnostic> {
    let (fields, rows) = form.si_bindings(|_| {
        Ok(eqiora_realization::Space::continuous_lagrange(
            std::num::NonZeroU16::MIN,
        ))
    })?;
    form.bind(
        ReferenceCell::hypercube(form.dimension)?,
        &fields,
        &rows,
        None,
    )
}

fn interval() -> AffineGeometryMap {
    AffineGeometryMap::new(ReferenceCell::hypercube(1).unwrap(), 1, vec![3.], vec![3.]).unwrap()
}

fn complex_close(actual: C, expected: C) {
    // Fixed low-order binary64 quadrature; no solver/refinement claim.
    assert!(
        (actual - expected).norm() < 1e-12,
        "{actual:?} != {expected:?}"
    );
}

#[test]
fn additive_complex_diffusion_preserves_flux_phase_rebinding_and_partials() {
    for equation in [
        "-div(a*grad(u)) + q*u = f;",
        "div(a*grad(u)) - q*u = -f;",
        "-div(2*a*grad(u)) + div(a*grad(u)) + q*u = f;",
        "div(a*grad(u)) - div(2*a*grad(u)) + q*u = f;",
        "div(2*a*grad(u)) - div(a*grad(u)) - q*u = -f;",
        "-div((2*a-a)*grad(u)) + q*u = f;",
    ] {
        for flux in ["a*grad(u)", "2*a*grad(u)-a*grad(u)"] {
            let source = HELMHOLTZ
                .replace("-div(a*grad(u)) + q*u = f;", equation)
                .replace("normal(a*grad(u))", &format!("normal({flux})"));
            let (program, ids) = fixture(&source);
            let form = CompiledRegionForm::<C>::derive(&program, ids["body"], 1).unwrap();
            let rule = QuadratureRule::tensor_product_gauss_legendre(1, 2).unwrap();
            let local = scalar_bound(&form)
                .unwrap()
                .prepare_affine_actions::<Native>(&interval(), None, &rule, AffineInput::Current)
                .unwrap()
                .0;
            // On [0,6], K=a/6[[1,-1],[-1,1]], M=q[[2,1],[1,2]], load=3f.
            for (actual, expected) in local.matrix().iter().zip([
                C::new(7., -1.),
                C::new(2., -2.),
                C::new(2., -2.),
                C::new(7., -1.),
            ]) {
                complex_close(*actual, expected);
            }
            for value in local.rhs() {
                complex_close(*value, C::new(3., 9.));
            }
            let law = form
                .boundary_laws(&program, ids["wall"], ids["law"], None)
                .unwrap()
                .remove(0);
            complex_close(law.evaluate(&[0.], &[-1.]).unwrap()[0], C::new(-2., 4.));
            let diffusion = &form.rows[0]
                .terms
                .iter()
                .find(|term| term.pairing == Pairing::Gradient)
                .unwrap()
                .coefficient;
            let point = ParameterPoint::bind(&program, &[ids["a"].downcast().unwrap()], &[9., -3.])
                .unwrap();
            complex_close(
                diffusion
                    .bind_inputs(&point)
                    .unwrap()
                    .evaluate(&[3.])
                    .unwrap(),
                C::new(9., -3.),
            );
            for imaginary in [false, true] {
                let partial = diffusion
                    .derivative(
                        &|coordinate| {
                            f64::from(
                                coordinate.symbol()
                                    == eqiora_schema::kernel::SymbolRef::Parameter(
                                        ids["a"].downcast().unwrap(),
                                    )
                                    && coordinate.is_imaginary() == imaginary,
                            )
                        },
                        1,
                    )
                    .unwrap();
                complex_close(
                    partial.evaluate(&[3.]).unwrap(),
                    if imaginary {
                        C::new(0., 1.)
                    } else {
                        C::new(1., 0.)
                    },
                );
            }
        }
    }
}

#[test]
fn conservative_projection_preserves_complex_coefficients_and_rejects_storage() {
    let (program, ids) = fixture(HELMHOLTZ);
    let form = CompiledRegionForm::<C>::derive(&program, ids["body"], 1).unwrap();
    let row = form.conservative_system().unwrap();
    assert_eq!(row.fields().collect::<Vec<_>>(), vec![(ids["u"], 1)]);
    for x in [0., 2., 6.] {
        complex_close(row.diffusion(0, 0, 0, &[x]).unwrap(), C::new(6., 6.));
    }
    let stored = HELMHOLTZ.replace("variable u:", "state u:").replace(
        "-div(a*grad(u)) + q*u = f;",
        "-div(a*grad(u)) + q*u + 1[s]*derivative(u) = f;",
    );
    let (program, ids) = fixture(&stored);
    let form = CompiledRegionForm::<C>::derive(&program, ids["body"], 1).unwrap();
    assert!(form.conservative_system().is_err());
}

#[test]
fn conservative_real_coordinates_preserve_linear_and_conjugated_reactions() {
    for conjugated in [false, true] {
        let source = if conjugated {
            HELMHOLTZ.replace("q*u", "q*math.conj(u)")
        } else {
            HELMHOLTZ.to_owned()
        };
        let (program, ids) = fixture(&source);
        let form = CompiledRegionForm::<C>::derive(&program, ids["body"], 1).unwrap();
        let native = form.conservative_system().unwrap();
        let (fields, rows) = form.si_bindings(|_| Ok(Space::cell_constant())).unwrap();
        let bound = form
            .bind_conservative(ReferenceCell::hypercube(1).unwrap(), &fields, &rows, None)
            .unwrap();
        let rule = QuadratureRule::gauss_legendre(1).unwrap();
        assert_eq!(
            bound
                .prepare_affine_actions::<Native>(&interval(), None, &rule, AffineInput::Current)
                .is_err(),
            conjugated
        );
        let (volume, _) = bound
            .prepare_affine_actions::<super::super::coordinates::Real>(
                &interval(),
                None,
                &rule,
                AffineInput::Current,
            )
            .unwrap();
        let row = native.project::<super::super::coordinates::Real>().unwrap();
        assert_eq!(row.components(), 2);
        assert_eq!(row.fields().collect::<Vec<_>>(), vec![(ids["u"], 1)]);
        // Multiplication by 6+6i sends (x,y) to (6x-6y,6x+6y).
        // (3-i)conj(x+iy) sends (x,y) to (3x-y,-x-3y).
        let reaction = if conjugated {
            [[3., -1.], [-1., -3.]]
        } else {
            [[3., 1.], [-1., 3.]]
        };
        // A constant test/trial pair integrates to the interval length 6.
        for i in 0..2 {
            assert!((volume.rhs()[i] - 6. * [1., 3.][i]).abs() < 1e-12);
            for j in 0..2 {
                assert!((volume.matrix()[2 * i + j] - 6. * reaction[i][j]).abs() < 1e-12);
            }
        }
        for x in [0., 2., 6.] {
            for i in 0..2 {
                for j in 0..2 {
                    assert_eq!(
                        row.diffusion(i, j, 0, &[x]).unwrap(),
                        [[6., -6.], [6., 6.]][i][j]
                    );
                }
            }
        }
    }
}

#[test]
fn complex_helmholtz_region_retains_phase_reaction_and_imaginary_load() {
    let (program, ids) = fixture(HELMHOLTZ);
    let form = CompiledRegionForm::<C>::derive(&program, ids["body"], 1).unwrap();
    let bound = scalar_bound(&form).unwrap();
    let rule = QuadratureRule::gauss_legendre(2).unwrap();
    let local = bound
        .prepare_affine_actions::<Native>(&interval(), None, &rule, AffineInput::Current)
        .map(|(primal, _)| primal)
        .unwrap();
    // Independently on [0,6]: K=a/6[[1,-1],[-1,1]],
    // M=q[[2,1],[1,2]], load=3f at each endpoint.
    for (actual, expected) in local.matrix().iter().zip([
        C::new(7., -1.),
        C::new(2., -2.),
        C::new(2., -2.),
        C::new(7., -1.),
    ]) {
        complex_close(*actual, expected);
    }
    for value in local.rhs() {
        complex_close(*value, C::new(3., 9.));
    }
    assert_ne!(local.matrix()[1], local.matrix()[2].conj());
    assert!(CompiledRegionForm::<f64>::derive(&program, ids["body"], 1).is_err());

    let mut prepared = bound.prepare_cell(&interval(), &rule).unwrap();
    prepared
        .add_load(&[C::new(-2., 4.), C::new(0., 0.)])
        .unwrap();
    let cached = prepared.evaluate(&BTreeMap::new()).unwrap();
    complex_close(cached.matrix()[0], C::new(7., -1.));
    complex_close(cached.matrix()[1], C::new(2., -2.));
    complex_close(cached.rhs()[0], C::new(1., 13.));
    complex_close(cached.rhs()[1], C::new(3., 9.));

    let rebound = form
        .bind_inputs(
            &ParameterPoint::bind(
                &program,
                &[
                    ids["a"].downcast().unwrap(),
                    ids["q"].downcast().unwrap(),
                    ids["f"].downcast().unwrap(),
                ],
                &[6., 6., 1., 2., 1., 3.],
            )
            .unwrap(),
        )
        .unwrap();
    let rebound = scalar_bound(&rebound)
        .unwrap()
        .prepare_affine_actions::<Native>(&interval(), None, &rule, AffineInput::Current)
        .map(|(primal, _)| primal)
        .unwrap();
    complex_close(rebound.matrix()[0], C::new(3., 5.));
    complex_close(rebound.matrix()[1], C::new(0., 1.));
    complex_close(
        bound
            .prepare_affine_actions::<Native>(&interval(), None, &rule, AffineInput::Current)
            .map(|(primal, _)| primal)
            .unwrap()
            .matrix()[0],
        C::new(7., -1.),
    );
}

#[test]
fn complex_boundary_law_preserves_constitutive_phase_and_parent_orientation() {
    let (program, ids) = fixture(HELMHOLTZ);
    let form = CompiledRegionForm::<C>::derive(&program, ids["body"], 1).unwrap();
    let law = form
        .boundary_laws(&program, ids["wall"], ids["law"], None)
        .unwrap()
        .remove(0);
    assert_eq!(law.evaluate(&[0.], &[-1.]).unwrap(), [C::new(-2., 4.)]);
    let bound = scalar_bound(&form).unwrap();
    for (side, x, sign) in [(0, 0., -1.), (1, 6., 1.)] {
        let facet = AffineGeometryMap::new(ReferenceCell::point(), 1, vec![x], vec![]).unwrap();
        let incidence = EntityIncidence {
            entity: MeshEntity::new(1, 0),
            local_ordinal: side,
            orientation: OrientationCode::new(0),
        };
        let contribution = bound
            .evaluate_natural_facet(
                ids["u"],
                &interval(),
                (&facet, incidence, &[side]),
                &QuadratureRule::point(),
                |_, normal| Ok(vec![C::new(2., -4.) * normal[0]]),
            )
            .unwrap();
        complex_close(contribution.rhs()[side], C::new(2., -4.) * sign);
        complex_close(contribution.rhs()[1 - side], C::new(0., 0.));
    }
    // Only change the boundary's phase; the strong volume law is unchanged.
    let wrong = HELMHOLTZ.replace("normal(a*grad(u))", "normal(math.conj(a)*grad(u))");
    let (program, ids) = fixture(&wrong);
    let form = CompiledRegionForm::<C>::derive(&program, ids["body"], 1).unwrap();
    assert!(
        form.boundary_laws(&program, ids["wall"], ids["law"], None)
            .is_err()
    );
}

#[test]
fn complex_region_assembly_eliminates_fixed_phase_without_conjugating_trial() {
    use crate::region_assembly::{PreparedRegionAssembly, RegionAssemblyCell};
    use eqiora_assembly::{
        AssemblyBackend, AssemblyMap, AssemblyPacketSetIdentityV1, AssemblyPlan, AssemblyTarget,
        DofId, LocalUnknown, REFERENCE_ASSEMBLY_BACKEND, TargetAssemblyMap,
    };
    let (program, ids) = fixture(HELMHOLTZ);
    let form = CompiledRegionForm::<C>::derive(&program, ids["body"], 1).unwrap();
    let plan = AssemblyPlan::new(vec![AssemblyTarget::new(1).unwrap()]).unwrap();
    let target = plan.target_id(0).unwrap();
    let work = PreparedRegionAssembly::<_>::new(
        AssemblyPacketSetIdentityV1::from_sha256([91; 32]),
        &plan,
        vec![(
            scalar_bound(&form).unwrap(),
            QuadratureRule::gauss_legendre(2).unwrap(),
        )],
        &[ids["body"]],
        vec![RegionAssemblyCell {
            previous_geometry: None,
            orientation: vec![1; 2],
            index: 0,
            geometry: interval(),
            mappings: vec![TargetAssemblyMap::new(
                target,
                AssemblyMap::new(
                    vec![None, Some(DofId::new(0))],
                    vec![
                        LocalUnknown::Fixed(C::new(1., 2.)),
                        LocalUnknown::Free(DofId::new(0)),
                    ],
                )
                .unwrap(),
            )]
            .into(),
            previous: BTreeMap::new(),
        }],
        vec![],
    )
    .unwrap();
    let assembled = REFERENCE_ASSEMBLY_BACKEND.assemble(&plan, &work).unwrap();
    let system = assembled.system(target).unwrap();
    // The free row is (7-i) u_1 = 3+9i - (2-2i)(1+2i) = -3+7i.
    assert_eq!(system.matrix().values().len(), 1);
    complex_close(system.matrix().values()[0], C::new(7., -1.));
    complex_close(system.rhs()[0], C::new(-3., 7.));
}

#[test]
fn conservative_conjugated_gradient_keeps_spatial_axes_and_prescribed_flux() {
    let source = HELMHOLTZ.replace("box(0, 6)", "box(0, 2, 0, 3)")
        .replace("variable u:", "parameter g:vector<complex<m>,2>=tensor_value(frame=body,components=[math.complex(1[m],2[m]),math.complex(3[m],4[m])]); variable u:")
        .replace("a*grad(u)", "a*grad(u)+1[m^2]*math.conj(grad(u))+g");
    let (program, ids) = fixture(&source);
    let form = CompiledRegionForm::<C>::derive(&program, ids["body"], 2).unwrap();
    let row = form
        .conservative_system()
        .unwrap()
        .project::<super::super::coordinates::Real>()
        .unwrap();
    assert!(!row.has_positive_diffusion());
    for axis in 0..2 {
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(
                    row.diffusion(i, j, axis, &[1., 2.]).unwrap(),
                    [[7., -6.], [6., 5.]][i][j]
                );
            }
            assert_eq!(
                row.prescribed_flux(i, axis, &[1., 2.]).unwrap(),
                [[1., 2.], [3., 4.]][axis][i]
            );
        }
    }
}

#[test]
fn composed_tensor_gradient_keeps_transposition_inside_the_physical_components() {
    let source = r#"
model TensorGradient() {
    domain body = box(0, 2, 0, 3);
    variable u:tensor<complex<1>,2,2> on body in smooth;
    relation balance on body {
        -div(math.complex(4,3)*grad(u)
            +math.complex(2,3)*grad(transpose(math.conj(u))))=0;
    }
}
"#;
    let (program, ids) = fixture(source);
    let form = CompiledRegionForm::<C>::derive(&program, ids["body"], 2).unwrap();
    let row = form
        .conservative_system()
        .unwrap()
        .project::<super::super::coordinates::Real>()
        .unwrap();
    // (4+3i)*U+(2+3i)*conj(U)^T, with each tensor entry's real and
    // imaginary coordinates adjacent. Transposition exchanges 01 and 10;
    // neither spatial derivative axis participates in that permutation.
    let expected = [
        [6., 0., 0., 0., 0., 0., 0., 0.],
        [6., 2., 0., 0., 0., 0., 0., 0.],
        [0., 0., 4., -3., 2., 3., 0., 0.],
        [0., 0., 3., 4., 3., -2., 0., 0.],
        [0., 0., 2., 3., 4., -3., 0., 0.],
        [0., 0., 3., -2., 3., 4., 0., 0.],
        [0., 0., 0., 0., 0., 0., 6., 0.],
        [0., 0., 0., 0., 0., 0., 6., 2.],
    ];
    assert_eq!(row.components(), 8);
    assert!(!row.has_positive_diffusion());
    for axis in 0..2 {
        for (test, values) in expected.iter().enumerate() {
            for (trial, value) in values.iter().enumerate() {
                assert_eq!(row.diffusion(test, trial, axis, &[1., 2.]).unwrap(), *value);
            }
        }
    }
}
