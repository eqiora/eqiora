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
    dimension: usize,
    // Entry mask identifies the differentiated directions: value, first,
    // second, mixed. These are derivative axes, never Field component axes.
    inputs: Vec<Vec<Input<S>>>,
    active: Vec<bool>,
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
            dimension: context.dimension,
            active: vec![false; inputs.len()],
            inputs: vec![inputs],
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
            dimension: self.dimension,
            active: self.active.clone(),
            inputs: self
                .inputs
                .iter()
                .map(|inputs| bind(inputs))
                .collect::<Result<_, _>>()?,
        })
    }
    pub(super) fn expression_view(&self) -> Option<eqiora_ir::ScalarExpressionView<'_>> {
        (self.inputs.len() == 1).then(|| self.row.expression_view())
    }
    pub(super) fn input(&self, ordinal: usize) -> &Input<S> {
        &self.inputs[0][ordinal]
    }
    pub(super) fn dimension(&self) -> usize {
        self.dimension
    }

    pub(super) fn polynomial_degree(&self) -> Option<usize> {
        // Derivative input blocks can have different spatial degrees. Their
        // maximum gives a conservative bound for every retained direction.
        let inputs = (0..self.row.symbols().len())
            .map(|index| {
                self.inputs.iter().try_fold(0, |degree, block| {
                    Some(degree.max(block[index].polynomial_degree()?))
                })
            })
            .collect::<Vec<_>>();
        self.row.polynomial_degree(&inputs)
    }
    pub(super) fn spatial(&self) -> bool {
        // Deriving a uniform Jacobian retains coordinate reads for primal
        // domain checks. A structural degree-zero proof still establishes that
        // its value is spatially constant; sampled Parameter values never do.
        self.inputs.iter().flatten().any(Input::spatial) && self.polynomial_degree() != Some(0)
    }
    pub(super) fn constant_in_direction(
        &self,
        direction: Direction<'_>,
    ) -> Result<bool, Diagnostic> {
        for inputs in &self.inputs {
            for (input, coordinate) in inputs.iter().zip(self.row.symbols()) {
                if !input.constant_in_direction(coordinate, direction)? {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
    pub(super) fn derivative(
        &self,
        direction: Direction<'_>,
        dimension: usize,
    ) -> Result<Self, Diagnostic> {
        if self.inputs.len() >= 4 {
            return Err(super::super::invalid(
                "pointwise coefficient supports up to two input derivatives",
            ));
        }
        let mut inputs = self.inputs.clone();
        let active = self
            .active
            .iter()
            .zip(&self.inputs[0])
            .zip(self.row.symbols())
            .map(|((&active, input), coordinate)| {
                Ok(active || !input.constant_in_direction(coordinate, direction)?)
            })
            .collect::<Result<_, Diagnostic>>()?;
        for values in &self.inputs {
            inputs.push(
                values
                    .iter()
                    .zip(self.row.symbols())
                    .map(|(input, coordinate)| input.derivative(coordinate, direction, dimension))
                    .collect::<Result<_, _>>()?,
            );
        }
        Ok(Self {
            row: self.row.clone(),
            dimension: self.dimension,
            inputs,
            active,
        })
    }
    pub(super) fn evaluate(&self, point: &[f64]) -> Result<S, Diagnostic> {
        validate_point(point, self.dimension)?;
        let evaluate = |inputs: &[Input<S>]| {
            inputs
                .iter()
                .map(|input| input.evaluate(point))
                .collect::<Result<Vec<_>, _>>()
        };
        let inputs = self
            .inputs
            .iter()
            .map(|inputs| evaluate(inputs))
            .collect::<Result<Vec<_>, _>>()?;
        let value = if inputs.len() == 1 {
            self.row.evaluate(&inputs[0])?
        } else {
            let roles = self
                .active
                .iter()
                .map(|&active| {
                    if active {
                        DifferentiationRole::Unknown
                    } else {
                        DifferentiationRole::Frozen
                    }
                })
                .collect::<Vec<_>>();
            let action = self.row.linearize(&inputs[0], &roles)?;
            let directions = inputs[1..]
                .iter()
                .map(|values| {
                    values
                        .iter()
                        .zip(&self.active)
                        .filter_map(|(&value, &active)| active.then_some(value))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let mut result = [0.];
            if inputs.len() == 2 {
                action.jvp(RelationTangent::Unknown(&directions[0]), &mut result)?;
            } else {
                action.mixed_jvp(
                    RelationTangent::Unknown(&directions[0]),
                    RelationTangent::Unknown(&directions[1]),
                    RelationTangent::Unknown(&directions[2]),
                    &mut result,
                )?;
            }
            result[0]
        };
        Ok(<S as From<f64>>::from(value))
    }
}
