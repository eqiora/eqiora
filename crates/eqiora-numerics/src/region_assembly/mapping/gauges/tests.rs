use super::*;
use eqiora_assembly::LocalUnknown;
use eqiora_core::{DimExponents, Id, ScalarDomain, ValueType, entity::kinds};
use eqiora_meshing::CartesianMesh;
use eqiora_realization::Space;

fn fixture() -> (RegionDofMap<f64>, [AlgebraicConstraint; 2], FieldDof) {
    let domain = Id::<kinds::Domain>::new().erase();
    let fields = [Id::<kinds::Field>::new(), Id::<kinds::Field>::new()];
    let layouts = fields
        .iter()
        .enumerate()
        .map(|(index, field)| RegionFieldLayout {
            field: field.erase(),
            value_type: ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap(),
            space: Space::continuous_lagrange(std::num::NonZeroU16::MIN),
            range: 2 * index..2 * (index + 1),
            components: 1,
            scale: 2.0,
        })
        .collect();
    let fixed = FieldDof {
        field: fields[0].erase(),
        entity: MeshEntity::new(0, 0),
        slot: 0,
        component: 0,
    };
    let mesh = CartesianMesh::from_axes(vec![vec![0.0, 0.5, 1.0]]).unwrap();
    let mapping = RegionDofMap::new(
        &mesh,
        &BTreeMap::from([(domain, layouts)]),
        ReferenceCell::hypercube(1).unwrap(),
        &[domain; 2],
        &[],
        &BTreeMap::from([(fixed, 6.0)]),
    )
    .unwrap();
    (
        mapping,
        fields.map(|field| AlgebraicConstraint::ZeroIntegral { field }),
        fixed,
    )
}

#[test]
fn plural_gauges_share_exact_cell_coordinates_and_preserve_physical_recovery() {
    let (base, constraints, fixed) = fixture();
    let mapping = base
        .clone()
        .with_algebraic_constraints(&constraints)
        .unwrap();
    assert_eq!(
        mapping,
        base.clone()
            .with_algebraic_constraints(&[constraints[1], constraints[0]])
            .unwrap()
    );
    assert_eq!(mapping.full_count(), base.full_count() + 2);
    assert_eq!(mapping.free_count(), base.free_count() + 2);
    for key in base.keys() {
        assert_eq!(mapping.global_dof(key), base.global_dof(key));
        assert_eq!(mapping.free_dof(key), base.free_dof(key));
    }
    for constraint in constraints {
        let first = mapping.constraint_map(0, constraint, true).unwrap();
        let second = mapping.constraint_map(1, constraint, true).unwrap();
        assert_eq!(first.unknowns().last(), second.unknowns().last());
        assert_eq!(first.unknowns()[1], second.unknowns()[0]);
        assert_eq!(
            first.unknowns().last(),
            Some(&LocalUnknown::Free(
                mapping.constraint_free_dof(constraint).unwrap()
            ))
        );
    }
    assert_eq!(
        mapping
            .constraint_map(0, constraints[0], true)
            .unwrap()
            .unknowns()[0],
        LocalUnknown::Fixed(3.0)
    );
    let mut values = vec![4.0; mapping.free_count()];
    values[mapping.constraint_free_dof(constraints[0]).unwrap().index()] = -19.0;
    values[mapping.constraint_free_dof(constraints[1]).unwrap().index()] = 23.0;
    let fields = mapping
        .recover(
            &values,
            &constraints.map(|constraint| constraint.field().erase()),
        )
        .unwrap();
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[&fixed.field].coefficients[&fixed], 6.0);
    assert!(
        fields
            .values()
            .flat_map(|field| &field.coefficients)
            .all(|(key, value)| *key == fixed || *value == 8.0)
    );
    let full = mapping.lift(&values, false).unwrap();
    assert_eq!(mapping.restrict(&full).unwrap(), values);
    let changed = mapping
        .with_prescribed(&BTreeMap::from([(fixed, 10.0)]))
        .unwrap();
    assert_eq!(
        changed.constraint_global_dof(constraints[0]),
        mapping.constraint_global_dof(constraints[0])
    );
    assert_eq!(
        changed
            .constraint_map(0, constraints[0], true)
            .unwrap()
            .unknowns()[0],
        LocalUnknown::Fixed(5.0)
    );
}

#[test]
fn duplicate_foreign_and_rebound_constraints_reject_without_mutating_the_map() {
    let (base, constraints, _) = fixture();
    let foreign = AlgebraicConstraint::ZeroIntegral { field: Id::new() };
    assert!(
        base.clone()
            .with_algebraic_constraints(&[constraints[0]; 2])
            .is_err()
    );
    assert!(base.clone().with_algebraic_constraints(&[foreign]).is_err());
    assert!(base.constraint_map(0, constraints[0], true).is_err());
    let mapping = base
        .clone()
        .with_algebraic_constraints(&constraints)
        .unwrap();
    assert!(
        mapping
            .clone()
            .with_algebraic_constraints(&constraints)
            .is_err()
    );
    assert!(mapping.constraint_map(2, constraints[0], true).is_err());
    assert_eq!(base.full_count(), 6);
}

#[test]
fn independent_integral_rows_close_both_constant_modes_and_missing_rows_reject() {
    use eqiora_assembly::{
        AssemblyBackend, AssemblyPacket, AssemblyPlan, AssemblyTarget, IndexedAssemblyWork,
        LocalContribution, REFERENCE_ASSEMBLY_BACKEND, TargetAssemblyMap,
    };
    let (base, constraints, _) = fixture();
    let mapping = base
        .with_prescribed(&BTreeMap::new())
        .unwrap()
        .with_algebraic_constraints(&constraints)
        .unwrap();
    let plan = AssemblyPlan::new(vec![AssemblyTarget::new(mapping.free_count()).unwrap()]).unwrap();
    let multipliers = constraints
        .map(|constraint| constraint.field().erase())
        .into_iter()
        .collect();
    for complete in [true, false] {
        let work = IndexedAssemblyWork::new(4, |packet| {
            // Two length-1/2 linear elements per scalar Field: stiffness
            // 2*[1,-1;-1,1] and exact basis integrals 1/4 each.
            let weight = if complete || packet / 2 == 0 {
                0.25
            } else {
                0.0
            };
            // Keep the uncoupled auxiliary row nonempty so the negative
            // example reaches constant-mode closure, rather than the earlier
            // empty-row assembly guard. Its diagonal cannot act on a physical
            // constant mode because that mode has zero auxiliary coordinates.
            let auxiliary = if complete || packet / 2 == 0 {
                0.0
            } else {
                1.0
            };
            AssemblyPacket::new(
                LocalContribution::new(
                    3,
                    3,
                    vec![
                        2.0, -2.0, weight, -2.0, 2.0, weight, weight, weight, auxiliary,
                    ],
                    vec![0.0; 3],
                )?,
                vec![TargetAssemblyMap::new(
                    plan.target_id(0).unwrap(),
                    mapping.constraint_map(packet % 2, constraints[packet / 2], true)?,
                )],
            )
        });
        let assembled = REFERENCE_ASSEMBLY_BACKEND.assemble(&plan, &work).unwrap();
        let closure =
            mapping.require_multiplier_closure(assembled.systems()[0].matrix(), &multipliers);
        assert_eq!(closure.is_ok(), complete);
    }
}
