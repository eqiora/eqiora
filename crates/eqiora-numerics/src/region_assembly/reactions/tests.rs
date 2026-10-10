use super::*;
use eqiora_assembly::{
    AssemblyPacket, AssemblyPlan, AssemblyTarget, IndexedAssemblyWork, LocalContribution,
    TargetAssemblyMap,
};

#[test]
fn volume_load_is_not_obtained_by_subtracting_large_boundary_traction() {
    let domain = eqiora_core::Id::<eqiora_core::entity::kinds::Domain>::new().erase();
    let constraints = crate::constrained_dofs::ConstrainedDofLayout::new(vec![Some(0.0)]).unwrap();
    let plan = AssemblyPlan::new(vec![AssemblyTarget::new(1).unwrap()]).unwrap();
    let target = plan.target_id(0).unwrap();
    let work = IndexedAssemblyWork::new(2, |index| {
        AssemblyPacket::new(
            LocalContribution::new(1, 1, vec![0.0], vec![if index == 0 { 1.0 } else { 1e20 }])?,
            vec![TargetAssemblyMap::new(target, constraints.full_map(&[0])?)],
        )
    });
    let rows = BTreeSet::from([0]);
    let reactions =
        DomainReactions::prepare(&work, target, 1, &[domain, domain], 1, &rows).unwrap();
    assert_eq!(reactions.volume_loads[&domain], [1.0]);
    assert_eq!(reactions.recover(&[0.0]).unwrap().values[&domain], [-1e20]);
    assert!(DomainReactions::prepare(&work, target, 1, &[domain, domain], 3, &rows).is_err());
}
