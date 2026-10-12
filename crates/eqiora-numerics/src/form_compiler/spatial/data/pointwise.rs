//! Prescribed real scalar calculus uses the existing component Operator IR.
use super::*;
use eqiora_core::{Id, ScalarDomain, ValueFrame, entity::kinds};
use eqiora_ir::{
    ComponentScalarRow, ComponentScalarization, DifferentiationRole, LinearizedRelation,
    RelationTangent,
};

#[derive(Debug, Clone, PartialEq)]
pub(super) struct PointwiseData<S: Coefficient> {
    row: ComponentScalarRow,
    inputs: Vec<Data<S>>,
    tangent: Option<Vec<Data<S>>>,
}

impl<S: Coefficient> PointwiseData<S> {
    pub(super) fn new(
        context: &Context<'_, S>,
        id: ExprId,
        depth: usize,
    ) -> Result<Self, Diagnostic> {
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
        if ty.scalar_domain() != ScalarDomain::Real
            || !ty.shape().is_scalar()
            || ty.frame() != ValueFrame::Invariant
            || ty.array_rank() != 0
        {
            return Err(super::super::invalid(
                "pointwise coefficient requires an invariant real scalar",
            ));
        }
        let lowered = ComponentScalarization::lower_selected(&typed, &[id])?;
        let [row] = lowered.rows() else {
            return Err(super::super::invalid(
                "pointwise coefficient requires one scalar row",
            ));
        };
        let inputs = row
            .symbols()
            .iter()
            .map(|coordinate| {
                if coordinate.is_imaginary() || !coordinate.component_index().is_empty() {
                    return Err(super::super::invalid(
                        "pointwise coefficient requires real scalar inputs",
                    ));
                }
                let index = context.dag.nodes().iter().position(|node|
                matches!(node, ExprNode::Symbol(symbol) if *symbol == coordinate.symbol())
            ).ok_or_else(|| super::super::invalid("pointwise input has no retained symbol"))?;
                // Context admits only prescribed Fields, exact Parameters, physical
                // coordinates and explicitly bound Time. Unknowns remain rejected.
                context.data(
                    context.dag.node_id(index as u32).expect("symbol index"),
                    depth + 1,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            row: row.clone(),
            inputs,
            tangent: None,
        })
    }

    pub(super) fn bind_parameter_point(
        &self,
        fields: &[Id<kinds::Parameter>],
        values: &[S],
    ) -> Result<Self, Diagnostic> {
        let bind = |inputs: &[Data<S>]| {
            inputs
                .iter()
                .map(|input| input.bind_parameter_point(fields, values))
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
            .any(Data::spatial)
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
            .map(|input| input.coordinate_derivative(axis, dimension))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            row: self.row.clone(),
            inputs: self.inputs.clone(),
            tangent: Some(tangent),
        })
    }

    pub(super) fn evaluate(&self, point: &[f64]) -> Result<S, Diagnostic> {
        let evaluate = |inputs: &[Data<S>]| {
            inputs
                .iter()
                .map(|input| {
                    let value = input.evaluate(point)?;
                    if value.im() != 0.0 {
                        return Err(super::super::invalid(
                            "pointwise coefficient input is not real",
                        ));
                    }
                    Ok(value.re())
                })
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
