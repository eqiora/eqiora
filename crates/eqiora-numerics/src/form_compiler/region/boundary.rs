//! Boundary equations selected by exact tested rows and constitutive operators.

use eqiora_core::{Id, entity::kinds};
use eqiora_graph::EdgeKind;
use eqiora_schema::kernel::{ExprId, ExprNode, SymbolRef};

use crate::additive_residual::AdditiveResidualView;
use crate::canonical_boundary::{BoundaryRelationBinding, PhysicalBoundaryQuantity};

use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RegionBoundaryLaw<S: Coefficient> {
    pub(crate) binding: BoundaryRelationBinding,
    pub(crate) tested: RawId,
    pub(crate) trace_field: Option<RawId>,
    pub(crate) quantity: PhysicalBoundaryQuantity,
    pub(crate) dependencies: BTreeSet<RawId>,
    pub(crate) operator: ExprId,
    pub(crate) datum_expression: Option<ExprId>,
    datum: PrescribedDatum<S>,
}

impl<S: Coefficient> RegionBoundaryLaw<S> {
    pub(in crate::form_compiler) fn on_uniform_chart(
        &mut self,
        chart: &super::super::linear::motion::UniformChart,
    ) -> Result<(), Diagnostic> {
        if self.quantity != PhysicalBoundaryQuantity::Trace {
            return Err(invalid(
                "moving scalar storage requires essential boundary data",
            ));
        }
        let PrescribedDatum::Components(values) = &mut self.datum else {
            return Err(invalid("moving scalar trace requires component data"));
        };
        for value in values {
            *value = value.on_uniform_chart(chart);
        }
        Ok(())
    }

    pub(crate) fn evaluate(&self, point: &[f64], normal: &[f64]) -> Result<Vec<S>, Diagnostic> {
        self.datum.evaluate(point, normal)
    }

    pub(crate) fn bind_parameter_point(
        &mut self,
        fields: &[Id<kinds::Parameter>],
        values: &[S],
    ) -> Result<(), Diagnostic> {
        self.datum.bind_parameter_point(fields, values)
    }
}

impl<S: Coefficient> CompiledRegionForm<S> {
    pub(in crate::form_compiler) fn boundary_laws(
        &self,
        program: &KernelProgram,
        boundary: RawId,
        relation: RawId,
        time_s: Option<f64>,
    ) -> Result<Vec<RegionBoundaryLaw<S>>, Diagnostic> {
        if crate::canonical::boundary_parent(program, boundary) != Some(self.domain)
            || !crate::canonical::relations_on(program, boundary).contains(&relation)
        {
            return Err(invalid("boundary law has foreign exact support"));
        }
        continuous_activations(program, &BTreeSet::from([relation]))?;
        let typed = typed_relation(program, relation)?;
        let dag = typed.expression();
        super::super::scalar::require_closed_roots(dag, relation)?;
        let dependencies = dag
            .nodes()
            .iter()
            .filter_map(|node| match node {
                ExprNode::Symbol(SymbolRef::Field(id) | SymbolRef::Derivative(id, _)) => {
                    Some(id.erase())
                }
                ExprNode::Symbol(SymbolRef::Parameter(id)) => Some(id.erase()),
                ExprNode::Symbol(SymbolRef::Coordinate { support, .. }) => Some(support.erase()),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        let graph = program
            .edges()
            .iter()
            .filter(|edge| edge.from() == relation && edge.kind() == EdgeKind::DependsOn)
            .map(|edge| edge.to())
            .collect();
        if dependencies != graph {
            return Err(invalid(
                "boundary dependency inventory differs from its exact equation",
            ));
        }

        let coefficients = super::super::linear::coefficients_at_time(
            program,
            self.dimension,
            &self.roles,
            time_s,
        )?;
        let context = Context {
            time_s,
            program,
            dag,
            owner: relation,
            dimension: self.dimension,
            coefficients: &coefficients,
        };
        let mut laws = Vec::new();
        for root in dag.roots() {
            // Preserve one prescribed scalar datum, including sums of coordinates.
            // Unknown-dependent or trace expressions cannot pass this coefficient gate.
            let view = AdditiveResidualView::derive_preserving(dag, *root, relation, &|id| {
                context.data(id, 0).is_ok()
            })?;
            let mut operators = Vec::new();
            for leaf in view.leaves() {
                match dag.node(leaf.value()) {
                    Some(ExprNode::Trace { value, .. }) => {
                        let Some(ExprNode::Symbol(SymbolRef::Field(field))) = dag.node(*value)
                        else {
                            continue;
                        };
                        let field = field.erase();
                        let tested = self
                            .roles
                            .relations
                            .values()
                            .find_map(|role| match role.kind {
                                Role::Kinematic { state, rate } if state == field => Some(rate),
                                _ => None,
                            })
                            .unwrap_or(field);
                        if let Some(row) = self
                            .rows
                            .iter()
                            .find(|row| row.tested == tested && !row.flux.is_empty())
                        {
                            operators.push((
                                leaf,
                                row,
                                Some(field),
                                PhysicalBoundaryQuantity::Trace,
                            ));
                        }
                    }
                    Some(ExprNode::NormalComponent { .. }) => {
                        for row in self.rows.iter().filter(|row| !row.flux.is_empty()) {
                            if self
                                .require_boundary_flux(
                                    program,
                                    boundary,
                                    relation,
                                    row.tested,
                                    leaf.value(),
                                    view.leaves().len() == 1,
                                )
                                .is_ok()
                            {
                                operators.push((leaf, row, None, PhysicalBoundaryQuantity::Flux));
                            }
                        }
                    }
                    _ => {}
                }
            }
            let [(operator, row, trace_field, quantity)] = operators.as_slice() else {
                return Err(view.mismatch("boundary requires one uniquely matched tested-row trace or complete constitutive flux"));
            };
            let values = view
                .leaves()
                .iter()
                .filter(|leaf| leaf.value() != operator.value())
                .collect::<Vec<_>>();
            let datum_expression = match values.as_slice() {
                [] => None,
                [value] => Some(value.value()),
                _ => {
                    return Err(
                        view.mismatch("boundary datum must be the sole term beside its operator")
                    );
                }
            };
            let mut datum =
                PrescribedDatum::derive(&context, &typed, &row.value_type, datum_expression)?;
            if values
                .first()
                .is_some_and(|value| value.sign() == operator.sign())
            {
                let negative = Data::constant(self.dimension, <S as From<f64>>::from(-1.0));
                match &mut datum {
                    PrescribedDatum::ParameterComponents { values, .. } => {
                        for value in values {
                            *value = <S as From<f64>>::from(-1.0) * *value;
                        }
                    }
                    PrescribedDatum::Components(components) => {
                        for value in components {
                            *value = value.clone().multiply(negative.clone());
                        }
                    }
                    PrescribedDatum::NormalMultiple(value) => {
                        *value = value.clone().multiply(negative)
                    }
                }
            }
            laws.push(RegionBoundaryLaw {
                binding: BoundaryRelationBinding::new(boundary, relation),
                tested: row.tested,
                trace_field: *trace_field,
                quantity: *quantity,
                dependencies: dependencies.clone(),
                operator: operator.value(),
                datum_expression,
                datum,
            });
        }
        Ok(laws)
    }
}
