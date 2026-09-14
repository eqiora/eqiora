//! On-demand numerical projection of admitted equation sides; never persisted.
use super::*;
use eqiora_core::ScalarDomain;
use eqiora_schema::kernel::ExprDagBuilder;

impl KernelProgram {
    /// Derive numerical residuals while retaining original node indices and sharing.
    /// Discrete equations have no subtraction projection and must use typed execution.
    ///
    /// # Errors
    /// Rejects absent Relations, non-numerical equation sides, or failed typed projection.
    pub fn numerical_residuals(&self, relation: RawId) -> Result<ExprDag, Diagnostic> {
        let Some(KernelNode::Relation(definition)) = self.node(relation) else {
            return Err(kernel_error(
                relation,
                "numerical projection requires a Relation",
            ));
        };
        if definition.has_constraints() {
            return Err(kernel_error(
                relation,
                "inequalities and complementarity require an explicit constrained realization",
            ));
        }
        let scope = edge_targets(&self.edges, relation, EdgeKind::AppliesOn)
            .first()
            .copied();
        let typed = self
            .type_derived_residual(
                definition.expression().clone(),
                relation,
                scope,
                if definition.is_initial() {
                    RootContract::InitialConditions
                } else {
                    RootContract::EquationSides
                },
            )
            .map_err(|errors| {
                errors
                    .into_iter()
                    .next()
                    .expect("typing failure has diagnostics")
            })?;
        for (left, right) in definition.equation_sides() {
            for id in [left, right] {
                let value = &typed
                    .node_type(id)
                    .expect("validated equation side")
                    .value_type;
                if !matches!(
                    value.scalar_domain(),
                    ScalarDomain::Real | ScalarDomain::Complex
                ) {
                    return Err(Diagnostic::error(
                        codes::NOT_IMPLEMENTED,
                        "discrete equation sides have no numerical residual projection",
                    )
                    .with_graph_path(kernel_path(relation)));
                }
            }
        }
        let mut builder = ExprDagBuilder::from_dag(definition.expression());
        let mut roots = Vec::with_capacity(definition.equation_sides().len());
        for (left, right) in definition.equation_sides() {
            let left_type = typed.node_type(left).expect("validated side");
            let right_type = typed.node_type(right).expect("validated side");
            let same =
                typing::additive(left_type, right_type).is_ok_and(|result| result == *left_type);
            let zero = matches!(definition.expression().nodes().get(right.index() as usize),Some(ExprNode::Constant(value)) if value.is_zero());
            roots.push(if same && zero {
                left
            } else {
                builder.sub(left, right)?
            });
        }
        builder.finish(roots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{ValueLiteral, ValueType};
    use eqiora_graph::{GraphStore, InMemoryGraphStore, Op, Transaction};
    use eqiora_schema::kernel::RelationDef;
    use eqiora_schema::{Model, ModelView};

    fn program(expression: ExprDag) -> (KernelProgram, RawId) {
        let relation = Id::<kinds::Relation>::new();
        let model = OntologyId::<Model>::new();
        let mut transaction = Transaction::new("typed equation projection");
        transaction.push(Op::DefineKernelNode {
            node: RelationDef::initial(relation, expression).unwrap().into(),
        });
        transaction.push(Op::DefineOntologyView {
            view: ModelView::new(model, [relation.erase()], [])
                .unwrap()
                .into(),
        });
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        (
            KernelProgram::from_snapshot(&store.snapshot(), model).unwrap(),
            relation.erase(),
        )
    }

    #[test]
    fn projection_preserves_arena_sharing_and_complex_zero_promotion() {
        let real = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
            .expect("valid scalar type");
        let complex = ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
            .expect("valid scalar type");
        let mut builder = ExprDagBuilder::new();
        let left = builder
            .constant(ValueLiteral::from_real(real.clone(), 2.).unwrap())
            .unwrap();
        let real_zero = builder
            .constant(ValueLiteral::from_real(real, 0.).unwrap())
            .unwrap();
        let complex_zero = builder
            .constant(ValueLiteral::from_real(complex.clone(), 0.).unwrap())
            .unwrap();
        let original = builder
            .finish([left, real_zero, left, complex_zero])
            .unwrap();
        let (program, relation) = program(original.clone());
        let residuals = program.numerical_residuals(relation).unwrap();
        assert_eq!(
            &residuals.nodes()[..original.nodes().len()],
            original.nodes()
        );
        assert_eq!(residuals.nodes().len(), original.nodes().len() + 1);
        assert_eq!(residuals.roots()[0], left);
        assert!(
            matches!(residuals.nodes().last(),Some(ExprNode::Sub(a,b)) if *a==left && *b==complex_zero)
        );
        let typed = program
            .typed_relation_residual(relation.downcast().unwrap())
            .unwrap();
        assert_eq!(
            typed.node_type(residuals.roots()[1]).unwrap().value_type,
            complex
        );
    }

    #[test]
    fn boolean_and_integer_equations_have_no_numeric_projection() {
        let integer = ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
            .expect("valid scalar type");
        for value in [
            ValueLiteral::boolean(false),
            ValueLiteral::from_integer(integer, 0).unwrap(),
        ] {
            let mut builder = ExprDagBuilder::new();
            let side = builder.constant(value).unwrap();
            let (program, relation) = program(builder.finish([side, side]).unwrap());
            assert!(program.numerical_residuals(relation).is_err());
        }
    }
}
