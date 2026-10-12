use super::*;

#[test]
fn harmonic_policy_canonicalizes_complete_inventories_and_rejects_ambiguity() {
    let domains = [Id::new(), Id::new(), Id::new(), Id::new()];
    let fields = [Id::new(), Id::new()];
    let interfaces = [Id::new(), Id::new()];
    let quality = AleGeometryQualityGate::new(0.05).unwrap();
    let solver = SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1e-12,
        1e-14,
        NonZeroUsize::new(100).unwrap(),
    )
    .unwrap();
    let make = |harmonic: Vec<_>, drivers: Vec<_>, interfaces: Vec<_>| {
        P1HarmonicMeshMotionPolicy::new(harmonic, drivers, interfaces, quality, solver)
    };
    let drivers = vec![(domains[2], fields[0]), (domains[3], fields[1])];
    let policy = make(domains[..2].to_vec(), drivers.clone(), interfaces.to_vec()).unwrap();
    assert_eq!(
        policy,
        make(
            domains[..2].iter().rev().copied().collect(),
            drivers.iter().rev().copied().collect(),
            interfaces.iter().rev().copied().collect(),
        )
        .unwrap()
    );
    for (harmonic, drivers, connections) in [
        (vec![], drivers.clone(), interfaces.to_vec()),
        (domains[..2].to_vec(), vec![], interfaces.to_vec()),
        (domains[..2].to_vec(), drivers.clone(), vec![]),
        (
            vec![domains[0], domains[0]],
            drivers.clone(),
            interfaces.to_vec(),
        ),
        (vec![domains[2]], drivers.clone(), interfaces.to_vec()),
        (
            vec![domains[0]],
            vec![drivers[0], drivers[0]],
            interfaces.to_vec(),
        ),
        (
            vec![domains[0]],
            vec![(domains[2], fields[0]), (domains[3], fields[0])],
            interfaces.to_vec(),
        ),
        (
            vec![domains[0]],
            drivers.clone(),
            vec![interfaces[0], interfaces[0]],
        ),
    ] {
        assert!(make(harmonic, drivers, connections).is_err());
    }
}
