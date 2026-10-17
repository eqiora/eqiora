//! Prescribed components use the existing typed component Operator IR.
use super::*;
use eqiora_core::ScalarDomain;
use eqiora_ir::{
    ComponentScalarRow, ComponentScalarization, DifferentiationRole, LinearizedRelation,
    RelationTangent,
};

#[derive(Debug, Clone, PartialEq)]
pub(super) struct PointwiseData<S: Coefficient> {
    row: ComponentScalarRow,
    inputs: Vec<Input<S>>,
    tangent: Option<Vec<Input<S>>>,
}

impl<S: Coefficient> PointwiseData<S> {
    pub(super) fn lower(
        context: &Context<'_, S>,
        id: ExprId,
        depth: usize,
    ) -> Result<Components<S>, Diagnostic> {
        let typed = super::super::super::scalar::typed_relation(context.program, context.owner)?;
        if typed.expression() != context.dag {
            return Err(super::super::invalid(
                "pointwise coefficient requires its exact typed expression",
            ));
        }
        let ty = &typed
            .node_type(id)
            .ok_or_else(|| super::super::invalid("missing typed coefficient"))?
            .value_type;
        if !matches!(
            ty.scalar_domain(),
            ScalarDomain::Real | ScalarDomain::Complex
        ) {
            return Err(super::super::invalid(
                "pointwise coefficient requires its complete numeric scalar domain",
            ));
        }
        let lowered = ComponentScalarization::lower_spatial_selected(&typed, &[id], None)?;
        let rows = lowered
            .rows()
            .iter()
            .map(|row| {
                Self::from_row(context, row, depth)
                    .map(|data| Data(Arc::new(Node::Pointwise(data))))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Components::new(ty.clone(), rows)
    }

    fn from_row(
        context: &Context<'_, S>,
        row: &ComponentScalarRow,
        depth: usize,
    ) -> Result<Self, Diagnostic> {
        let inputs = row
            .symbols()
            .iter()
            .map(|coordinate| Input::from_coordinate(context, coordinate, depth))
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        Ok(Self {
            row: row.clone(),
            inputs,
            tangent: None,
        })
    }

    pub(super) fn bind_parameter_point(
        &self,
        point: &crate::parameter_point::ParameterPoint,
    ) -> Result<Self, Diagnostic> {
        let bind = |inputs: &[Input<S>]| {
            inputs
                .iter()
                .zip(self.row.symbols())
                .map(|(input, coordinate)| input.bind(coordinate, point))
                .collect::<Result<Vec<_>, _>>()
        };
        Ok(Self {
            row: self.row.clone(),
            inputs: bind(&self.inputs)?,
            tangent: self.tangent.as_deref().map(bind).transpose()?,
        })
    }
    pub(super) fn spatial(&self) -> bool {
        self.inputs
            .iter()
            .chain(self.tangent.iter().flatten())
            .any(Input::spatial)
    }
    pub(super) fn coordinate_derivative(
        &self,
        axis: usize,
        dimension: usize,
    ) -> Result<Self, Diagnostic> {
        if self.tangent.is_some() {
            return Err(super::super::invalid(
                "pointwise coefficient admits one spatial derivative",
            ));
        }
        let tangent = self
            .inputs
            .iter()
            .map(|input| input.derivative(axis, dimension))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            row: self.row.clone(),
            inputs: self.inputs.clone(),
            tangent: Some(tangent),
        })
    }
    pub(super) fn evaluate(&self, point: &[f64]) -> Result<S, Diagnostic> {
        let evaluate = |inputs: &[Input<S>]| {
            inputs
                .iter()
                .map(|input| input.evaluate(point))
                .collect::<Result<Vec<_>, _>>()
        };
        let inputs = evaluate(&self.inputs)?;
        let value = if let Some(tangent) = &self.tangent {
            let tangent = evaluate(tangent)?;
            let roles = vec![DifferentiationRole::Unknown; inputs.len()];
            let action = self.row.linearize(&inputs, &roles)?;
            let mut result = [0.0];
            action.jvp(RelationTangent::Unknown(&tangent), &mut result)?;
            result[0]
        } else {
            self.row.evaluate(&inputs)?
        };
        Ok(<S as From<f64>>::from(value))
    }
}
