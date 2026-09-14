use super::*;
use eqiora_schema::kernel::ExprNode;

fn fixture(flux: &str) -> (KernelProgram, CompiledRegionForm, BTreeMap<String, RawId>) {
    let source = format!("model Stress() {{
        domain body = box(0, 1, 0, 1);
        domain left = boundary(body, axis=0, side=lower);
        domain right = boundary(body, axis=0, side=upper);
        parameter mu: 1 = 2;
        parameter lambda: 1 = 5;
        parameter p0: 1 = 3;
        parameter other0: 1 = 3;
        variable p: 1 on body;
        variable other: 1 on body;
        relation pressure on body {{ p = p0; }}
        relation other_pressure on body {{ other = other0; }}
        variable u: vector<m, 2> on body;
        relation balance on body {{
            -div(2*mu*symmetric_part(grad(u)) + lambda*isotropic_lift(div(u)) + isotropic_lift(p)) = 0;
        }}
        relation left_law on left {{ normal({flux}) = 0; }}
        relation right_law on right {{ normal({flux}) = 0; }}
    }}");
    let (transaction, model, symbols) =
        compile("flux.eqi", &source).unwrap().remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let ids = ["body", "left", "right", "left_law", "right_law", "u"]
        .into_iter()
        .map(|name| (name.to_owned(), symbols.get(name).unwrap()))
        .collect::<BTreeMap<_, _>>();
    let form = CompiledRegionForm::derive(&program, ids["body"], 2).unwrap();
    (program, form, ids)
}

fn check(
    program: &KernelProgram,
    form: &CompiledRegionForm,
    ids: &BTreeMap<String, RawId>,
    side: &str,
) -> Result<(), Diagnostic> {
    let relation = ids[&format!("{side}_law")];
    let typed = typed_relation(program, relation).unwrap();
    let normal = typed.expression().roots()[0];
    assert!(matches!(
        typed.expression().node(normal),
        Some(ExprNode::NormalComponent(_))
    ));
    form.require_boundary_flux(program, ids[side], relation, ids["u"], normal)
}

#[test]
fn vector_stress_boundary_matches_complete_exact_operator_and_parameter_inventory() {
    let exact = "2*mu*symmetric_part(grad(u)) + lambda*isotropic_lift(div(u)) + isotropic_lift(p)";
    let (program, form, ids) = fixture(exact);
    check(&program, &form, &ids, "left").unwrap();
    check(&program, &form, &ids, "right").unwrap();
    // The uniform term is absent from the volume contraction, but remains a
    // mandatory member of the flux on both exact opposite parent boundaries.
    assert_eq!(form.rows[0].terms.len(), 2);
    assert_eq!(form.rows[0].flux.len(), 3);
    for changed in [
        "2*mu*symmetric_part(grad(u)) + lambda*isotropic_lift(div(u))",
        "2*mu*symmetric_part(grad(u)) + lambda*isotropic_lift(div(u)) + isotropic_lift(other)",
        "2*mu*grad(u) + lambda*isotropic_lift(div(u)) + isotropic_lift(p)",
    ] {
        let (program, form, ids) = fixture(changed);
        assert!(check(&program, &form, &ids, "left").is_err());
    }
    let typed = typed_relation(&program, ids["left_law"]).unwrap();
    assert!(
        form.require_boundary_flux(
            &program,
            ids["right"],
            ids["left_law"],
            ids["u"],
            typed.expression().roots()[0]
        )
        .is_err()
    );
}

#[test]
fn uniform_stress_weak_volume_and_oriented_facet_loads_cancel_exactly() {
    let (_, form, ids) =
        fixture("2*mu*symmetric_part(grad(u)) + lambda*isotropic_lift(div(u)) + isotropic_lift(p)");
    let (mut fields, mut rows, _) = inputs(&form, false);
    for field in &mut fields {
        field.scale = DynQuantity::new(1.0, field.scale.dim());
    }
    for scale in rows.values_mut() {
        *scale = DynQuantity::new(1.0, scale.dim());
    }
    let bound = form
        .bind(ReferenceCell::simplex(2).unwrap(), &fields, &rows, None)
        .unwrap();
    let cell = geometry();
    let volume = bound
        .evaluate(
            &cell,
            &simplex_duffy_gauss_legendre(2, 2).unwrap(),
            &BTreeMap::new(),
        )
        .unwrap();
    // Independent weak action: area=1/2, p=3 and gradients (-1,-1),(1,0),(0,1).
    // Known stress enters the left side; the assembled RHS has its negative.
    let stress = [-1.5, -1.5, 1.5, 0.0, 0.0, 1.5];
    for (rhs, stress) in volume.rhs().iter().zip(stress) {
        close(*rhs, -stress);
    }
    let mut traction = [0.0; 6];
    let mut wrong_normal = [0.0; 6];
    use eqiora_meshing::{MeshEntity, MeshGeometry, MeshQualityGate, MeshTopology, SimplicialMesh};
    let mesh = SimplicialMesh::new(
        2,
        vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
        vec![vec![0, 1, 2]],
        MeshQualityGate::new(0.05).unwrap(),
    )
    .unwrap();
    let rule = simplex_duffy_gauss_legendre(1, 2).unwrap();
    for index in 0..mesh.entity_count(1).unwrap() {
        let entity = MeshEntity::new(1, index);
        let facet = mesh.geometry_map(entity).unwrap();
        let incident = mesh.incidence(entity, 2).unwrap();
        let [incidence] = incident.as_slice() else {
            panic!("one parent cell")
        };
        let parent_vertices = mesh
            .entity_vertices(entity)
            .unwrap()
            .iter()
            .map(|vertex| vertex.index())
            .collect::<Vec<_>>();
        let local = bound
            .evaluate_natural_facet(
                ids["u"],
                &cell,
                (&facet, *incidence, &parent_vertices),
                &rule,
                |_, normal| Ok(normal.iter().map(|normal| 3.0 * normal).collect()),
            )
            .unwrap();
        let reversed = bound
            .evaluate_natural_facet(
                ids["u"],
                &cell,
                (&facet, *incidence, &parent_vertices),
                &rule,
                |_, normal| Ok(normal.iter().map(|normal| -3.0 * normal).collect()),
            )
            .unwrap();
        for i in 0..6 {
            traction[i] += local.rhs()[i];
            wrong_normal[i] += reversed.rhs()[i];
        }
    }
    for i in 0..6 {
        close(traction[i], stress[i]);
        close(volume.rhs()[i] + traction[i], 0.0);
    }
    assert!(
        volume
            .rhs()
            .iter()
            .zip(wrong_normal)
            .any(|(volume, boundary)| (volume + boundary).abs() > 1.0)
    );
    // For all-essential zero displacement, full residual reactions retain the
    // known stress instead of reporting zero from its zero strong divergence.
    use crate::region_assembly::{DomainReactions, PreparedRegionAssembly, RegionAssemblyCell};
    use eqiora_assembly::{
        AssemblyPacketSetIdentityV1, AssemblyPlan, AssemblyTarget, TargetAssemblyMap,
    };
    let constraints =
        crate::constrained_dofs::ConstrainedDofLayout::new(vec![Some(0.0); 6]).unwrap();
    let plan = AssemblyPlan::new(vec![AssemblyTarget::new(6).unwrap()]).unwrap();
    let target = plan.target_id(0).unwrap();
    let work = PreparedRegionAssembly::new(
        AssemblyPacketSetIdentityV1::from_sha256([73; 32]),
        &plan,
        vec![(bound, simplex_duffy_gauss_legendre(2, 2).unwrap())],
        &[form.domain()],
        vec![RegionAssemblyCell {
            index: 0,
            geometry: cell,
            mappings: vec![TargetAssemblyMap::new(
                target,
                constraints.full_map(&(0..6).collect::<Vec<_>>()).unwrap(),
            )],
            previous: BTreeMap::new(),
        }],
        vec![],
    )
    .unwrap();
    let reactions =
        DomainReactions::prepare(&work, target, 6, &[form.domain()], &(0..6).collect()).unwrap();
    for (actual, expected) in reactions.recover(&[0.0; 6]).unwrap().values[&form.domain()]
        .iter()
        .zip(stress)
    {
        close(*actual, expected);
    }
}

#[test]
fn intrinsic_facet_trace_preserves_parent_permutation_and_zero_bubble() {
    use eqiora_meshing::{EntityIncidence, MeshEntity, OrientationCode, ReferenceTopology};
    let (_, form, ids) =
        fixture("2*mu*symmetric_part(grad(u)) + lambda*isotropic_lift(div(u)) + isotropic_lift(p)");
    for (reference, bubble) in [
        (ReferenceCell::simplex(2).unwrap(), false),
        (ReferenceCell::simplex(2).unwrap(), true),
        (ReferenceCell::hypercube(2).unwrap(), false),
    ] {
        let (mut fields, mut rows, _) = inputs(&form, bubble);
        fields
            .iter_mut()
            .for_each(|field| field.scale = DynQuantity::new(1.0, field.scale.dim()));
        rows.values_mut()
            .for_each(|row| *row = DynQuantity::new(1.0, row.dim()));
        let bound = form.bind(reference, &fields, &rows, None).unwrap();
        let (origin, jacobian) =
            if reference.family() == eqiora_meshing::ReferenceCellFamily::Hypercube {
                (vec![11.5, 21.0], vec![1.5, 0.0, 0.0, 1.0])
            } else {
                (vec![10.0, 20.0], vec![3.0, 0.0, 0.0, 2.0])
            };
        let cell = AffineGeometryMap::new(reference, 2, origin, jacobian).unwrap();
        // Canonical facet vertex order is reversed relative to the parent.
        let facet =
            AffineGeometryMap::from_simplex_vertices(vec![vec![13.0, 20.0], vec![10.0, 20.0]])
                .unwrap();
        let topology = ReferenceTopology::new(reference).unwrap();
        let ordinal = (0..topology.entity_count(1).unwrap())
            .find(|ordinal| topology.entity(1, *ordinal).unwrap().vertex_ordinals() == [0, 1])
            .unwrap();
        let incidence = EntityIncidence {
            entity: MeshEntity::new(2, 0),
            local_ordinal: ordinal,
            orientation: OrientationCode::new(1),
        };
        let rule = simplex_duffy_gauss_legendre(1, 2).unwrap();
        let local = bound
            .evaluate_natural_facet(
                ids["u"],
                &cell,
                (&facet, incidence, &[1, 0]),
                &rule,
                |point, normal| {
                    assert_eq!(normal, [0.0, -1.0]);
                    Ok(vec![point[0] - 10.0, 0.0])
                },
            )
            .unwrap();
        // x=13-3t, ds=3dt: N0=t and N1=1-t. Their integrals
        // against x-10 are 9*integral(t(1-t))=3/2 and 9/3=3.
        for (index, value) in local.rhs().iter().enumerate() {
            close(
                *value,
                match index {
                    0 => 1.5,
                    2 => 3.0,
                    _ => 0.0,
                },
            );
        }
        assert!(
            bound
                .evaluate_natural_facet(
                    ids["u"],
                    &cell,
                    (&facet, incidence, &[0, 2]),
                    &rule,
                    |_, _| Ok(vec![0.0, 0.0])
                )
                .is_err()
        );
        let wrong = EntityIncidence {
            local_ordinal: (ordinal + 1) % topology.entity_count(1).unwrap(),
            ..incidence
        };
        assert!(
            bound
                .evaluate_natural_facet(
                    ids["u"],
                    &cell,
                    (&facet, wrong, &[1, 0]),
                    &rule,
                    |_, _| Ok(vec![0.0, 0.0])
                )
                .is_err()
        );
    }
}

#[test]
fn rounded_cartesian_midpoint_does_not_move_a_topological_facet_outside() {
    use eqiora_meshing::{CartesianMesh, MeshEntity, MeshGeometry, MeshTopology};
    let (_, form, ids) =
        fixture("2*mu*symmetric_part(grad(u)) + lambda*isotropic_lift(div(u)) + isotropic_lift(p)");
    let (mut fields, mut rows, _) = inputs(&form, false);
    fields
        .iter_mut()
        .for_each(|field| field.scale = DynQuantity::new(1.0, field.scale.dim()));
    rows.values_mut()
        .for_each(|row| *row = DynQuantity::new(1.0, row.dim()));
    let bound = form
        .bind(ReferenceCell::hypercube(2).unwrap(), &fields, &rows, None)
        .unwrap();
    let mesh = CartesianMesh::from_axes(vec![
        vec![0.0, 0.5, 1.0],
        vec![0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
    ])
    .unwrap();
    let entity = (0..mesh.entity_count(1).unwrap())
        .map(|index| MeshEntity::new(1, index))
        .find(|entity| mesh.geometry_map(*entity).unwrap().origin() == [0.25, 1.0])
        .unwrap();
    let facet = mesh.geometry_map(entity).unwrap();
    let incidence = mesh.incidence(entity, 2).unwrap()[0];
    let cell = mesh.geometry_map(incidence.entity).unwrap();
    // Binary64 midpoint rounding made the removed physical inverse exceed 1.
    let reconstructed =
        cell.inverse_jacobian().unwrap()[3] * (facet.origin()[1] - cell.origin()[1]);
    assert!(reconstructed > 1.0);
    let cell_vertices = mesh.entity_vertices(incidence.entity).unwrap();
    let parent_vertices = mesh
        .entity_vertices(entity)
        .unwrap()
        .iter()
        .map(|vertex| {
            cell_vertices
                .iter()
                .position(|parent| parent == vertex)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let packet = bound
        .evaluate_natural_facet(
            ids["u"],
            &cell,
            (&facet, incidence, &parent_vertices),
            &eqiora_meshing::QuadratureRule::tensor_product_gauss_legendre(1, 2).unwrap(),
            |_, normal| {
                assert_eq!(normal, [0.0, 1.0]);
                Ok(vec![1.0, 0.0])
            },
        )
        .unwrap();
    // Constant unit datum on a half-unit edge: each endpoint receives length/2.
    for (index, value) in packet.rhs().iter().enumerate() {
        close(*value, if index == 4 || index == 6 { 0.25 } else { 0.0 });
    }
}
