use super::*;
use crate::form_compiler::{derive_candidate_with_dimension, linear::CompiledLinearBlockForm};
use crate::region_assembly::{PreparedRegionAssembly, RegionAssemblyCell};
use eqiora_assembly::{
    AssemblyBackend, AssemblyMap, AssemblyPacket, AssemblyPacketSetIdentityV1, AssemblyPlan,
    AssemblyTarget, DofId, LocalUnknown, PacketLinearSystem, REFERENCE_ASSEMBLY_BACKEND,
    TargetAssemblyMap,
};
use eqiora_meshing::{
    AffineGeometryMap, EntityIncidence, MeshEntity, MeshGeometry, OrientationCode, QuadratureRule,
    ReferenceCell,
};
use eqiora_solver::{LinearOperator, LinearOperatorOrientation as O, OrientedLinearOperator};

pub(super) fn check(program: &KernelProgram, projection: &AuthoredFormulationProjection) {
    let domain = eqiora_core::Id::<eqiora_core::entity::kinds::Domain>::from_ulid(
        projection.domain_ulid().unwrap().parse().unwrap(),
    )
    .erase();
    let derived = derive_candidate_with_dimension(program, domain, 1)
        .unwrap()
        .unwrap();
    crate::form_compiler::admit_authored_scalar_primal_form(projection, program, &derived).unwrap();
    let form = CompiledLinearBlockForm::<C>::derive(program, domain, 1, &BTreeSet::new()).unwrap();
    let field = form.fields()[0].0;
    let law = form.boundary_laws()[&field]
        .values()
        .find(|law| law.trace_field.is_none())
        .unwrap();
    let bound = form.volume().unwrap();
    let mesh = CartesianMesh::from_axes(vec![vec![0., 3., 6.]]).unwrap();
    let plan = AssemblyPlan::new(vec![AssemblyTarget::new(2).unwrap()]).unwrap();
    let target = plan.target_id(0).unwrap();
    let maps = [
        AssemblyMap::new(
            vec![None, Some(DofId::new(0))],
            vec![
                LocalUnknown::Fixed(C::new(1., 3.)),
                LocalUnknown::Free(DofId::new(0)),
            ],
        )
        .unwrap(),
        AssemblyMap::new(
            vec![Some(DofId::new(0)), Some(DofId::new(1))],
            vec![
                LocalUnknown::Free(DofId::new(0)),
                LocalUnknown::Free(DofId::new(1)),
            ],
        )
        .unwrap(),
    ];
    let cell = |index| mesh.geometry_map(MeshEntity::new(1, index)).unwrap();
    let facet = AffineGeometryMap::new(ReferenceCell::point(), 1, vec![6.], vec![]).unwrap();
    let local = bound
        .evaluate_natural_facet(
            field,
            &cell(1),
            (
                &facet,
                EntityIncidence {
                    entity: MeshEntity::new(1, 1),
                    local_ordinal: 1,
                    orientation: OrientationCode::new(0),
                },
                &[1],
            ),
            &QuadratureRule::point(),
            |point, normal| law.evaluate(point, normal),
        )
        .unwrap();
    let boundary =
        AssemblyPacket::new(local, vec![TargetAssemblyMap::new(target, maps[1].clone())]).unwrap();
    let work = PreparedRegionAssembly::new(
        AssemblyPacketSetIdentityV1::Unbound,
        &plan,
        vec![(bound, QuadratureRule::gauss_legendre(2).unwrap())],
        &[domain; 2],
        (0..2)
            .map(|index| RegionAssemblyCell {
                previous_geometry: None,
                orientation: vec![1; maps[index].unknowns().len()],
                index,
                geometry: cell(index),
                mappings: vec![TargetAssemblyMap::new(target, maps[index].clone())],
                previous: BTreeMap::new(),
            })
            .collect(),
        vec![boundary],
    )
    .unwrap();
    let assembled = REFERENCE_ASSEMBLY_BACKEND.assemble(&plan, &work).unwrap();
    let csr = assembled.system(target).unwrap();
    let packets = PacketLinearSystem::from_work(&plan, target, &work).unwrap();
    // On h=3: K=a/3[[1,-1],[-1,1]], M=q/2[[2,1],[1,2]].
    // Eliminate u(0)=1+3i and add the exact outward flux 18+6i.
    for rhs in [csr.rhs(), packets.right_hand_side()] {
        assert_eq!(rhs.len(), 2);
        for (actual, expected) in rhs.iter().zip([C::new(18., 27.), C::new(37.5, 19.5)]) {
            close(*actual, expected);
        }
    }
    assert_eq!(csr.matrix().values().len(), 4);
    for (actual, expected) in csr.matrix().values().iter().zip([
        C::new(6., 6.),
        C::new(-1.5, -1.5),
        C::new(-1.5, -1.5),
        C::new(3., 3.),
    ]) {
        close(*actual, expected);
    }
    let input = [C::new(2., -1.), C::new(-1., 3.)];
    for (orientation, expected) in [
        (O::Normal, [C::new(24., 3.), C::new(-16.5, 4.5)]),
        (O::Transposed, [C::new(24., 3.), C::new(-16.5, 4.5)]),
        (
            O::ConjugateTransposed,
            [C::new(3., -24.), C::new(4.5, 16.5)],
        ),
    ] {
        for operator in [
            csr.matrix() as &dyn OrientedLinearOperator<Scalar = C>,
            packets.operator(),
        ] {
            let mut actual = [C::new(0., 0.); 2];
            operator
                .apply_oriented(orientation, &input, &mut actual)
                .unwrap();
            for (actual, expected) in actual.into_iter().zip(expected) {
                close(actual, expected);
            }
        }
    }
    let mut diagonal = [C::new(0., 0.); 2];
    packets.operator().diagonal(&mut diagonal).unwrap();
    for (actual, expected) in diagonal.into_iter().zip([C::new(6., 6.), C::new(3., 3.)]) {
        close(actual, expected);
    }
}
fn close(actual: C, expected: C) {
    assert!(
        (actual - expected).norm() < 1e-12,
        "{actual:?} != {expected:?}"
    );
}
