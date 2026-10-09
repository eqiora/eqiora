//! Closed record membership and shared temporal ownership during Model admission.
use super::*;

pub(super) fn validate(
    nodes: &BTreeMap<RawId, KernelNode>,
    edges: &[Edge],
    diagnostics: &mut Vec<Diagnostic>,
) {
    for (&owner, node) in nodes {
        match node {
            KernelNode::Record(record) => {
                for (_, value_type) in record.members() {
                    super::nominal_values::check(owner, value_type, nodes, diagnostics);
                }
            }
            KernelNode::RecordInstance(instance) => {
                let Some(KernelNode::Record(record)) = nodes.get(&instance.definition().erase())
                else {
                    diagnostics.push(kernel_error(
                        owner,
                        "record instance requires its exact selected Record declaration",
                    ));
                    continue;
                };
                let expression = instance.expression();
                if expression.roots().len() != record.members().len() {
                    diagnostics.push(kernel_error(
                        owner,
                        "record instance must bind every declared member exactly once",
                    ));
                    continue;
                }
                for node in expression.nodes() {
                    if let ExprNode::Constant(value) = node {
                        super::nominal_values::check_literal(owner, value, nodes, diagnostics);
                    }
                }
                let inferred = TypedResidual::<()>::infer(
                    expression.clone(),
                    None,
                    |_| None,
                    RootContract::ValueRoots,
                    |symbol| {
                        let value_type = match symbol {
                            SymbolRef::Parameter(id) => match nodes.get(&id.erase()) {
                                Some(KernelNode::Parameter(value)) => value.value_type(),
                                _ => return Err(()),
                            },
                            SymbolRef::Field(id) => match nodes.get(&id.erase()) {
                                Some(KernelNode::Field(value)) => value.value_type(),
                                _ => return Err(()),
                            },
                            _ => return Err(()),
                        };
                        Ok(ExpressionType::new(value_type.clone(), None))
                    },
                );
                let Ok(inferred) = inferred else {
                    diagnostics.push(kernel_error(owner, "record member expression requires exact selected static Parameters or owned bus Fields and valid typed operations"));
                    continue;
                };
                for ((_, expected_type), root) in record.members().iter().zip(expression.roots()) {
                    if inferred
                        .node_type(*root)
                        .is_none_or(|value| &value.value_type != expected_type)
                    {
                        diagnostics.push(kernel_error(owner, "record member disagrees with its declared type or ordered member identity"));
                    }
                }
                let has_fields = expression
                    .nodes()
                    .iter()
                    .any(|node| matches!(node, ExprNode::Symbol(SymbolRef::Field(_))));
                if !has_fields {
                    continue;
                }
                let mut temporal_owner = None;
                let mut seen = BTreeSet::new();
                for root in expression.roots() {
                    let Some(ExprNode::Symbol(SymbolRef::Field(member))) = expression.node(*root)
                    else {
                        diagnostics.push(kernel_error(owner, "record bus roots must be direct owned Fields; static members cannot mix with bus ownership"));
                        continue;
                    };
                    let Some(KernelNode::Field(field)) = nodes.get(&member.erase()) else {
                        continue;
                    };
                    if !seen.insert(member.erase()) {
                        diagnostics.push(kernel_error(
                            owner,
                            "record bus requires distinct owned member Fields",
                        ));
                    }
                    let temporal = (
                        field.role(),
                        edge_targets(edges, member.erase(), EdgeKind::ClockedBy),
                    );
                    if temporal.1.len() > 1 || temporal.1.iter().any(|clock| !matches!(nodes.get(clock), Some(KernelNode::ClockDomain(value)) if matches!(value.kind(), ClockKind::Periodic { .. }))) {
                        diagnostics.push(kernel_error(owner, "record member requires at most one exact periodic ClockDomain"));
                    }
                    if temporal_owner
                        .as_ref()
                        .is_some_and(|expected| expected != &temporal)
                    {
                        diagnostics.push(kernel_error(
                            owner,
                            "record bus members must share one exact temporal ownership",
                        ));
                    } else {
                        temporal_owner = Some(temporal);
                    }
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{ScalarDomain, ValueType};
    use eqiora_graph::{GraphStore, InMemoryGraphStore, Op, Transaction};
    use eqiora_schema::kernel::{
        ClockDomainDef, FieldDef, FieldRole, RationalTime, RecordDef, RecordInstanceDef,
    };

    fn fixture(other_clock: bool) -> (BTreeMap<RawId, KernelNode>, Vec<Edge>, RawId) {
        let ty = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap();
        let record = RecordDef::new(
            Id::new(),
            vec![
                ("sensor".into(), ty.clone()),
                ("command".into(), ty.clone()),
            ],
        )
        .unwrap();
        let first = FieldDef::new(Id::new(), ty.clone(), FieldRole::State);
        let second = FieldDef::new(Id::new(), ty, FieldRole::State);
        let mut expression = eqiora_schema::kernel::ExprDagBuilder::new();
        let roots = [
            expression.symbol(SymbolRef::Field(first.id())).unwrap(),
            expression.symbol(SymbolRef::Field(second.id())).unwrap(),
        ];
        let instance =
            RecordInstanceDef::new(Id::new(), record.id(), expression.finish(roots).unwrap())
                .unwrap();
        let instance_id = instance.id().erase();
        let clock = ClockDomainDef::periodic(
            Id::new(),
            RationalTime::new(1, 1).unwrap(),
            RationalTime::new(0, 1).unwrap(),
        )
        .unwrap();
        let foreign_clock = ClockDomainDef::periodic(
            Id::new(),
            RationalTime::new(1, 1).unwrap(),
            RationalTime::new(0, 1).unwrap(),
        )
        .unwrap();
        let edges = [
            (first.id().erase(), clock.id().erase()),
            (
                second.id().erase(),
                if other_clock {
                    foreign_clock.id().erase()
                } else {
                    clock.id().erase()
                },
            ),
        ];
        let nodes: BTreeMap<_, _> = [
            record.into(),
            first.into(),
            second.into(),
            instance.into(),
            clock.into(),
            foreign_clock.into(),
        ]
        .into_iter()
        .map(|node: KernelNode| (node.id(), node))
        .collect();
        let mut transaction = Transaction::new("record clocks");
        for node in nodes.values() {
            transaction.push(Op::DefineKernelNode { node: node.clone() });
        }
        for (from, to) in edges {
            transaction.push(Op::Connect {
                from,
                to,
                edge: EdgeKind::ClockedBy,
            });
        }
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let edges = store.snapshot().edges().copied().collect();
        (nodes, edges, instance_id)
    }

    #[test]
    fn homogeneous_bus_rejects_equal_period_foreign_clock() {
        let (nodes, edges, _) = fixture(false);
        let mut errors = Vec::new();
        validate(&nodes, &edges, &mut errors);
        assert!(errors.is_empty(), "{errors:?}");
        let (nodes, edges, _) = fixture(true);
        validate(&nodes, &edges, &mut errors);
        assert!(
            errors.iter().any(|error| error
                .message()
                .contains("share one exact temporal ownership")),
            "{errors:?}"
        );
    }

    #[test]
    fn record_instance_requires_its_selected_declaration_and_complete_typed_members() {
        let (mut nodes, edges, id) = fixture(false);
        let KernelNode::RecordInstance(instance) = nodes[&id].clone() else {
            panic!("instance")
        };
        let mut errors = Vec::new();
        nodes.remove(&instance.definition().erase());
        validate(&nodes, &edges, &mut errors);
        assert!(errors.iter().any(|error| {
            error
                .message()
                .contains("exact selected Record declaration")
        }));
        let (mut nodes, edges, id) = fixture(false);
        let KernelNode::RecordInstance(instance) = nodes[&id].clone() else {
            panic!("instance")
        };
        let Some(ExprNode::Symbol(SymbolRef::Field(member))) =
            instance.expression().node(instance.expression().roots()[0])
        else {
            panic!("member")
        };
        let member = member.erase();
        nodes.insert(
            member,
            FieldDef::new(
                member.downcast().unwrap(),
                ValueType::boolean(),
                FieldRole::State,
            )
            .into(),
        );
        errors.clear();
        validate(&nodes, &edges, &mut errors);
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains("declared type"))
        );
    }
}
