//! Differential factors reuse component JVPs and the finite-map LU owner.
use super::*;
use eqiora_ir::{ComponentScalarization, DifferentiationRole, LinearizedRelation, RelationTangent};
use eqiora_schema::kernel::{CoordinateMapFactor, typing::RootContract};

impl Evaluator<'_, '_> {
    pub(super) fn map_factor(
        &mut self,
        id: ExprId,
        factor: CoordinateMapFactor,
        source: &[ExprId],
        at: &[(ExprId, ExprId)],
        point: Option<&EvaluationPoint>,
    ) -> Result<ValueLiteral, Diagnostic> {
        let missing = || {
            Diagnostic::error(
                codes::NOT_IMPLEMENTED,
                "coordinate Jacobian requires its exact Model and source point",
            )
        };
        let program = self.program.ok_or_else(missing)?;
        let point = point.ok_or_else(missing)?;
        if point.side().is_some() {
            return Err(Diagnostic::error(
                codes::NOT_IMPLEMENTED,
                "one-sided Jacobian requires an admitted orientation transform",
            ));
        }
        let n = source.len();
        let work = n
            .checked_mul(n)
            .and_then(|n| n.checked_mul(self.expression.nodes().len()))
            .ok_or_else(component_budget_error)?;
        self.point_work = self
            .point_work
            .checked_add(work)
            .filter(|work| *work <= 1_000_000)
            .ok_or_else(component_budget_error)?;
        let coordinates = source
            .iter()
            .map(|id| match self.expression.node(*id) {
                Some(ExprNode::Symbol(
                    symbol @ SymbolRef::Coordinate {
                        support,
                        factor,
                        axis,
                    },
                )) => {
                    point.coordinate(*support, *factor, *axis)?;
                    Ok(*symbol)
                }
                _ => Err(Diagnostic::error(
                    codes::INVALID_EXPRESSION_DAG,
                    "Jacobian source is not an exact coordinate",
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let typed = program
            .type_derived_residual(
                self.expression.clone(),
                self.owner,
                None,
                RootContract::Observable,
            )
            .map_err(|errors| errors.into_iter().next().expect("failed inference"))?;
        let roots = at.iter().map(|(_, mapped)| *mapped).collect::<Vec<_>>();
        let lowered = ComponentScalarization::lower_selected(&typed, &roots)?;
        if lowered.rows().len() != n
            || lowered
                .rows()
                .iter()
                .any(|row| row.is_imaginary() || !row.component_index().is_empty())
        {
            return Err(Diagnostic::error(
                codes::INVALID_EXPRESSION_DAG,
                "Jacobian requires one real scalar per target coordinate",
            ));
        }
        let mut entries = Vec::with_capacity(n * n);
        let mut mapped = Vec::with_capacity(n);
        for (row, root) in lowered.rows().iter().zip(&roots) {
            let mut inputs = Vec::new();
            let mut roles = Vec::new();
            let mut active = Vec::new();
            for coordinate in row.symbols() {
                let symbol = coordinate.symbol();
                let value = if let SymbolRef::Coordinate {
                    support,
                    factor,
                    axis,
                } = symbol
                {
                    roles.push(DifferentiationRole::Unknown);
                    active.push(symbol);
                    point.coordinate(support, factor, axis)?
                } else {
                    let spatial =
                        self.expression
                            .nodes()
                            .iter()
                            .zip(typed.node_types())
                            .any(|(node, ty)| {
                                matches!(node, ExprNode::Symbol(found) if *found == symbol)
                                    && ty.support.is_some()
                            });
                    if spatial {
                        return Err(Diagnostic::error(
                            codes::NOT_IMPLEMENTED,
                            "Jacobian of a spatial Field requires an admitted reconstruction derivative",
                        ));
                    }
                    roles.push(DifferentiationRole::Frozen);
                    (self.resolve)(EvaluationInput::Value(symbol), Some(point))?
                };
                if coordinate.is_imaginary() || !coordinate.component_index().is_empty() {
                    return Err(Diagnostic::error(
                        codes::NOT_IMPLEMENTED,
                        "coordinate Jacobian input requires the admitted real scalar profile",
                    ));
                }
                inputs.push(value.real_scalar_value().ok_or_else(missing)?.value());
            }
            let linearized = row.linearize(&inputs, &roles)?;
            for coordinate in &coordinates {
                let direction = active
                    .iter()
                    .map(|symbol| f64::from(symbol == coordinate))
                    .collect::<Vec<_>>();
                let mut derivative = [0.0];
                linearized.jvp(RelationTangent::Unknown(&direction), &mut derivative)?;
                entries.push(derivative[0]);
            }
            mapped.push(
                ValueLiteral::from_real(
                    typed
                        .node_type(*root)
                        .expect("typed map")
                        .value_type
                        .clone(),
                    row.evaluate(&inputs)?,
                )
                .map_err(discrete_error)?,
            );
        }
        EvaluationPoint::bind(program, self.expression, at, &mapped, None)?;
        let value = eqiora_ir::coordinate_map_factor(&entries, n, factor)?;
        ValueLiteral::from_real(
            typed
                .node_type(id)
                .expect("typed factor")
                .value_type
                .clone(),
            value,
        )
        .map_err(discrete_error)
    }
}
