use super::*;
use eqiora_core::{Id, entity::kinds};
use eqiora_schema::kernel::{ExprDagBuilder, SymbolRef};

fn relation() -> RawId {
    Id::<kinds::Relation>::new().erase()
}

#[test]
fn scalar_replay_rejects_foreign_resources_missing_terms_and_wrong_roles() {
    let field = Id::<kinds::Field>::new();
    let mut dag = ExprDagBuilder::new();
    let node = dag.symbol(SymbolRef::Field(field)).unwrap();
    let divergence = dag.neg(node).unwrap();
    let root = dag.add(node, divergence).unwrap();
    let boundaries = [
        BoundarySource {
            domain: Id::<kinds::Domain>::new().erase(),
            relation: relation(),
            trace_node: node,
        },
        BoundarySource {
            domain: Id::<kinds::Domain>::new().erase(),
            relation: relation(),
            trace_node: divergence,
        },
    ];
    let source = PrimalGalerkinSource {
        domain: Id::<kinds::Domain>::new().erase(),
        unknown: field.erase(),
        volume_relation: relation(),
        root,
        divergence,
        divergence_sign: WeakSign::Positive,
        source: node,
        boundaries: &boundaries,
    };
    let valid = PrimalGalerkinCorrespondence::derive(source);
    valid.replay(source).unwrap();
    let reject = |mutate: fn(&mut PrimalGalerkinCorrespondence)| {
        let mut candidate = valid.clone();
        mutate(&mut candidate);
        assert!(candidate.replay(source).is_err());
    };
    reject(|c| c.law.domain = relation());
    reject(|c| c.law.unknown = relation());
    reject(|c| c.law.relations[0] = relation());
    reject(|c| {
        c.law.relations.pop();
    });
    reject(|c| c.law.relations.swap(1, 2));
    reject(|c| c.formulation.zero_on.clear());
    reject(|c| c.formulation.zero_on[0] = relation());
    reject(|c| c.formulation.assumptions.clear());
    reject(|c| c.formulation.assumptions.push("weak-implies-strong"));
    reject(|c| c.formulation.kind = FormulationKind::MixedGalerkin);
    reject(|c| c.formulation.trial = relation());
    reject(|c| c.formulation.test = relation());
    reject(|c| c.formulation.boundary_treatment = BoundaryTreatment::ExplicitTraceFluxLaws);
    reject(|c| c.formulation.rules.swap(0, 1));
    reject(|c| {
        c.entries[1].slot = WeakTermSlot::Bilinear {
            test: MatrixSlot::Trial,
            trial: MatrixSlot::Test,
        }
    });
    reject(|c| {
        c.entries[2].slot = WeakTermSlot::Linear {
            test: MatrixSlot::Test,
        }
    });
    reject(|c| c.entries.swap(2, 3));
    reject(|c| {
        c.entries.push(c.entries[0]);
    });
    for index in 0..valid.entries.len() {
        // Every retained occurrence is independently required: deleting any one,
        // substituting a foreign relation/node, unknown rule, or reversed sign fails.
        let mut candidate = valid.clone();
        candidate.entries.remove(index);
        assert!(candidate.replay(source).is_err(), "missing entry {index}");
        let mut candidate = valid.clone();
        candidate.entries[index].relation = relation();
        assert!(
            candidate.replay(source).is_err(),
            "foreign relation {index}"
        );
        let mut candidate = valid.clone();
        candidate.entries[index].source_node = dag.neg(root).unwrap();
        assert!(
            candidate.replay(source).is_err(),
            "foreign occurrence {index}"
        );
        let mut candidate = valid.clone();
        candidate.entries[index].rule_id = "unsupported-rule";
        assert!(candidate.replay(source).is_err(), "unknown rule {index}");
        let mut candidate = valid.clone();
        candidate.entries[index].sign = match candidate.entries[index].sign {
            WeakSign::Positive => WeakSign::Negative,
            WeakSign::Negative => WeakSign::Positive,
        };
        assert!(candidate.replay(source).is_err(), "reversed sign {index}");
    }
    // Source drift must also fail: a consistent old certificate cannot admit new input.
    assert!(
        valid
            .replay(PrimalGalerkinSource {
                root: node,
                ..source
            })
            .is_err()
    );
    assert!(
        valid
            .replay(PrimalGalerkinSource {
                boundaries: &boundaries[..1],
                ..source
            })
            .is_err()
    );
}

#[test]
fn conservative_replay_binds_every_physical_role_and_closed_rule() {
    let boundaries = [relation(), relation()];
    let source = IntegralConservativeSource {
        domain: Id::<kinds::Domain>::new().erase(),
        velocity: Id::<kinds::Field>::new().erase(),
        pressure: Id::<kinds::Field>::new().erase(),
        source: Id::<kinds::Field>::new().erase(),
        source_definition: relation(),
        momentum_relation: relation(),
        incompressibility_relation: relation(),
        boundary_relations: &boundaries,
    };
    let valid = IntegralConservativeCorrespondence::derive(source);
    valid.replay(source).unwrap();
    let reject = |mutate: fn(&mut IntegralConservativeCorrespondence)| {
        let mut candidate = valid.clone();
        mutate(&mut candidate);
        assert!(candidate.replay(source).is_err());
    };
    reject(|c| c.law.domain = relation());
    reject(|c| c.law.velocity = relation());
    reject(|c| c.law.pressure = relation());
    reject(|c| c.law.source = relation());
    reject(|c| c.law.source_definition = relation());
    reject(|c| c.law.momentum_relation = relation());
    reject(|c| c.law.incompressibility_relation = relation());
    reject(|c| {
        c.law.boundary_relations.pop();
    });
    reject(|c| c.law.boundary_relations.push(relation()));
    reject(|c| c.law.boundary_relations.swap(0, 1));
    reject(|c| c.formulation.kind = FormulationKind::PrimalGalerkin);
    reject(|c| c.formulation.domain = relation());
    reject(|c| c.formulation.momentum_unknown = relation());
    reject(|c| c.formulation.pressure_role = relation());
    reject(|c| c.formulation.boundary_treatment = BoundaryTreatment::CompleteHomogeneousEssential);
    for index in 0..valid.formulation.rules.len() {
        let mut candidate = valid.clone();
        candidate.formulation.rules[index] = valid.formulation.rules[(index + 1) % 7];
        assert!(
            candidate.replay(source).is_err(),
            "omitted/replaced rule {index}"
        );
    }
    assert!(
        valid
            .replay(IntegralConservativeSource {
                source_definition: relation(),
                ..source
            })
            .is_err()
    );
    assert!(
        valid
            .replay(IntegralConservativeSource {
                boundary_relations: &[],
                ..source
            })
            .is_err()
    );
}
