//! Retain native types and symbol hypotheses beside one boundary operand DAG.
use super::super::rejection;
use eqiora_core::{Diagnostic, RawId};
use eqiora_schema::kernel::pure_operator::PureOperatorDefinition;
use eqiora_schema::kernel::typing::{ExpressionType, TraceRegularityChecker};
use eqiora_schema::kernel::{ExprDagBuilder, ExprId, ExprNode, SpatialRegularity};

pub(super) type Ty = ExpressionType<RawId>;

#[derive(Default)]
pub(super) struct NativeTrace {
    builder: ExprDagBuilder,
    types: Vec<Ty>,
    assertions: Vec<SpatialRegularity>,
}

impl NativeTrace {
    pub(super) fn ty(&self, id: ExprId) -> &Ty {
        &self.types[id.index() as usize]
    }

    pub(super) fn push(
        &mut self,
        node: ExprNode,
        ty: Ty,
        assertion: SpatialRegularity,
    ) -> Result<ExprId, Diagnostic> {
        let id = self.builder.push(node)?;
        self.retain(id, ty, assertion);
        Ok(id)
    }

    pub(super) fn operation(&mut self, node: ExprNode, ty: Ty) -> Result<ExprId, Diagnostic> {
        self.push(node, ty, SpatialRegularity::Unspecified)
    }

    pub(super) fn pure(
        &mut self,
        definition: &PureOperatorDefinition,
        arguments: &[ExprId],
    ) -> Result<ExprId, Diagnostic> {
        let types = arguments
            .iter()
            .map(|id| self.ty(*id).clone())
            .collect::<Vec<_>>();
        let ty = definition
            .instantiate(&types)
            .map_err(|error| rejection(&error.to_string()))?
            .result_type()
            .clone();
        let id = self
            .builder
            .pure_operator(definition, arguments.iter().copied())?;
        self.retain(id, ty, SpatialRegularity::Unspecified);
        Ok(id)
    }

    fn retain(&mut self, id: ExprId, ty: Ty, assertion: SpatialRegularity) {
        debug_assert_eq!(id.index() as usize, self.types.len());
        self.types.push(ty);
        self.assertions.push(assertion);
    }

    pub(super) fn check(self, root: ExprId) -> Result<(), Diagnostic> {
        let dag = self.builder.finish([root])?;
        let mut checker = TraceRegularityChecker::default();
        for (index, node) in dag.nodes().iter().enumerate() {
            checker
                .check_node(
                    node,
                    &self.types[index],
                    |id| &self.types[id.index() as usize],
                    |_| self.assertions[index],
                    |digest| {
                        dag.definition(digest)
                            .expect("builder retains closed definitions")
                    },
                )
                .map_err(|error| rejection(&format!("Field trace regularity: {error}")))?;
        }
        Ok(())
    }
}
