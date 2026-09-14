use std::collections::BTreeSet;

use eqiora_compiler::compile;
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_schema::kernel::{DomainKind, KernelNode};
use eqiora_sem::KernelProgram;

use super::elasticity::derive_cartesian_q1_elasticity_form_2d;
use super::scalar::derive_candidate;
use super::vocabulary::{
    BoundaryTreatment, FormulationKind, FormulationRule, PrimalGalerkinCorrespondence,
};

const POISSON: &str =
    include_str!("../../../../crates/eqiora-api/packages/org.example.poisson/src/main.eqi");
const ELASTICITY: &str =
    include_str!("../../../../verify/solid/isotropic-elasticity-2d/models/linear-load.eqi");

#[test]
fn poisson_and_elasticity_share_one_typed_primal_galerkin_vocabulary() {
    let poisson_program = compile_program("poisson.eqi", POISSON);
    let poisson_domain = poisson_program
        .nodes()
        .find_map(|node| match node {
            KernelNode::Domain(domain)
                if matches!(domain.kind(), DomainKind::CartesianBox { .. }) =>
            {
                Some(domain.id().erase())
            }
            _ => None,
        })
        .expect("Poisson has one Cartesian volume");
    let poisson = derive_candidate(&poisson_program, poisson_domain)
        .expect("Poisson derivation is valid")
        .expect("Poisson selects the shared formulation");

    let elasticity_program = compile_program("elasticity.eqi", ELASTICITY);
    let elasticity = derive_cartesian_q1_elasticity_form_2d(&elasticity_program)
        .expect("elasticity derivation is valid");

    assert_shared_primal_galerkin(poisson.correspondence());
    assert_shared_primal_galerkin(elasticity.correspondence());
    assert_ne!(
        poisson.correspondence().law,
        elasticity.correspondence().law
    );
}

fn assert_shared_primal_galerkin(correspondence: &PrimalGalerkinCorrespondence) {
    assert_eq!(
        correspondence.formulation.kind,
        FormulationKind::PrimalGalerkin
    );
    assert_eq!(
        correspondence.formulation.boundary_treatment,
        BoundaryTreatment::CompleteEssential
    );
    assert_eq!(correspondence.formulation.trial, correspondence.law.unknown);
    assert_eq!(correspondence.formulation.test, correspondence.law.unknown);
    assert_eq!(
        correspondence.formulation.rules,
        [
            FormulationRule::TestPairing,
            FormulationRule::DivergenceByParts,
            FormulationRule::ZeroTestTraceDischarge,
            FormulationRule::SourcePairing,
        ]
    );
    assert_eq!(correspondence.law.relations.len(), 5);
    assert_eq!(
        correspondence
            .law
            .relations
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len(),
        correspondence.law.relations.len()
    );
}

fn compile_program(name: &str, source: &str) -> KernelProgram {
    let mut compiled = compile(name, source).expect("source compiles");
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).expect("transaction commits");
    KernelProgram::from_snapshot(&store.snapshot(), model).expect("kernel program projects")
}
