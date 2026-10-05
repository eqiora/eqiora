use eqiora_core::entity::kinds;
use eqiora_core::{Id, OntologyId, ValueType};
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_schema::kernel::{
    ActivationDef, ClockDomainDef, ExprDagBuilder, FieldDef, KernelNode, RationalTime, RelationDef,
    SymbolRef,
};
use eqiora_schema::{Model, ModelView};
use eqiora_sem::KernelProgram;

#[test]
fn native_kernel_admission_rejects_time_operators_outside_their_activation() {
    for periodic in [false, true] {
        let field = Id::<kinds::Field>::new();
        let relation = Id::<kinds::Relation>::new();
        let activation = Id::<kinds::Activation>::new();
        let clock = Id::<kinds::ClockDomain>::new();
        let model = OntologyId::<Model>::new();
        let mut expression = ExprDagBuilder::new();
        let root = expression
            .symbol(if periodic {
                SymbolRef::Derivative(field, std::num::NonZeroU32::MIN)
            } else {
                SymbolRef::Pre(field)
            })
            .unwrap();
        let mut nodes = vec![
            KernelNode::from(FieldDef::new(
                field,
                ValueType::scalar(
                    eqiora_core::ScalarDomain::Real,
                    eqiora_core::DimExponents::DIMENSIONLESS,
                )
                .expect("valid scalar type"),
                eqiora_schema::kernel::FieldRole::State,
            )),
            KernelNode::from(
                RelationDef::new(
                    relation,
                    {
                        let equation_zero = expression
                            .constant(eqiora_core::DynQuantity::new(
                                0.0,
                                if periodic {
                                    eqiora_core::DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0])
                                        .unwrap()
                                } else {
                                    eqiora_core::DimExponents::DIMENSIONLESS
                                },
                            ))
                            .unwrap();
                        expression.finish([root, equation_zero])
                    }
                    .unwrap(),
                )
                .unwrap(),
            ),
            KernelNode::from(if periodic {
                ActivationDef::periodic(activation)
            } else {
                ActivationDef::continuous(activation)
            }),
        ];
        if periodic {
            nodes.push(KernelNode::from(
                ClockDomainDef::periodic(
                    clock,
                    RationalTime::new(1, 1).unwrap(),
                    RationalTime::ZERO,
                )
                .unwrap(),
            ));
        }
        let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
        let mut transaction = Transaction::new("activation falsifier without source compiler");
        for node in nodes {
            transaction.push(Op::DefineKernelNode { node });
        }
        transaction.push(Op::Connect {
            from: relation.erase(),
            to: field.erase(),
            edge: EdgeKind::DependsOn,
        });
        transaction.push(Op::Connect {
            from: activation.erase(),
            to: relation.erase(),
            edge: EdgeKind::Activates,
        });
        if periodic {
            transaction.push(Op::Connect {
                from: activation.erase(),
                to: clock.erase(),
                edge: EdgeKind::ClockedBy,
            });
        }
        transaction.push(Op::DefineOntologyView {
            view: ModelView::new(model, members, []).unwrap().into(),
        });
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let diagnostics = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap_err();
        let expected = if periodic {
            "clocked Relation cannot read Derivative symbols"
        } else {
            "continuous Relation cannot read Pre or Next symbols"
        };
        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message() == expected),
            "{diagnostics:?}"
        );
    }
}
