use super::*;
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
    let ids = ["body", "wall", "balance", "law", "a", "q", "f", "u"]
        .into_iter()
        .map(|name| (name.to_owned(), symbols.get(name).unwrap()))
        .collect();
    (program, ids)
}

fn scalar_bound(form: &CompiledRegionForm<C>) -> Result<BoundRegionForm<C>, Diagnostic> {
    let (fields, rows) = form.si_bindings(eqiora_realization::Space::continuous_lagrange(
        std::num::NonZeroU16::MIN,
    ))?;
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
fn complex_helmholtz_region_retains_phase_reaction_and_imaginary_load() {
    let (program, ids) = fixture(HELMHOLTZ);
    let form = CompiledRegionForm::<C>::derive(&program, ids["body"], 1).unwrap();
    let bound = scalar_bound(&form).unwrap();
    let rule = QuadratureRule::gauss_legendre(2).unwrap();
    let local = bound.prepare_affine(&interval(), &rule).unwrap();
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
        .bind_parameter_point(
            &[
                ids["a"].downcast().unwrap(),
                ids["q"].downcast().unwrap(),
                ids["f"].downcast().unwrap(),
            ],
            &[C::new(6., 6.), C::new(1., 2.), C::new(1., 3.)],
        )
        .unwrap();
    let rebound = scalar_bound(&rebound)
        .unwrap()
        .prepare_affine(&interval(), &rule)
        .unwrap();
    complex_close(rebound.matrix()[0], C::new(3., 5.));
    complex_close(rebound.matrix()[1], C::new(0., 1.));
    complex_close(
        bound.prepare_affine(&interval(), &rule).unwrap().matrix()[0],
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
    let work = PreparedRegionAssembly::new(
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
            )],
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
