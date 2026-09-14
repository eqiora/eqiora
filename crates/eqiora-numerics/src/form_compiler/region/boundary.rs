//! Boundary equations selected by exact tested rows and constitutive operators.

use eqiora_core::{Id, entity::kinds};
use eqiora_graph::EdgeKind;
use eqiora_ir::{OperatorApplicationProof, StandardPureOperator};
use eqiora_schema::kernel::{ExprId, ExprNode, SymbolRef};

use crate::additive_residual::AdditiveResidualView;
use crate::canonical_boundary::{BoundaryRelationBinding, PhysicalBoundaryQuantity};

use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RegionBoundaryLaw {
    pub(crate) binding: BoundaryRelationBinding,
    pub(crate) tested: RawId,
    pub(crate) trace_field: Option<RawId>,
    pub(crate) quantity: PhysicalBoundaryQuantity,
    pub(crate) dependencies: BTreeSet<RawId>,
    pub(crate) operator: ExprId,
    pub(crate) datum_expression: Option<ExprId>,
    datum: BoundaryDatum,
}

#[derive(Debug, Clone, PartialEq)]
enum BoundaryDatum {
    Components(Vec<Data>),
    NormalMultiple(Data),
}

impl RegionBoundaryLaw {
    pub(crate) fn evaluate(&self, point: &[f64], normal: &[f64]) -> Result<Vec<f64>, Diagnostic> {
        match &self.datum {
            BoundaryDatum::Components(values) => {
                values.iter().map(|value| value.evaluate(point)).collect()
            }
            BoundaryDatum::NormalMultiple(value) => {
                let value = value.evaluate(point)?;
                if normal.len() != point.len() || normal.iter().any(|value| !value.is_finite()) {
                    return Err(invalid("boundary datum requires the exact parent normal"));
                }
                Ok(normal.iter().map(|normal| normal * value).collect())
            }
        }
    }

    pub(crate) fn bind_parameter_point(
        &mut self,
        fields: &[Id<kinds::Parameter>],
        values: &[f64],
    ) -> Result<(), Diagnostic> {
        match &mut self.datum {
            BoundaryDatum::Components(components) => {
                for component in components {
                    *component = component.bind_parameter_point(fields, values)?;
                }
            }
            BoundaryDatum::NormalMultiple(value) => {
                *value = value.bind_parameter_point(fields, values)?
            }
        }
        Ok(())
    }
}

impl CompiledRegionForm {
    pub(in crate::form_compiler) fn boundary_law(
        &self,
        program: &KernelProgram,
        boundary: RawId,
        relation: RawId,
    ) -> Result<RegionBoundaryLaw, Diagnostic> {
        if crate::canonical::boundary_parent(program, boundary) != Some(self.domain)
            || !crate::canonical::relations_on(program, boundary).contains(&relation)
        {
            return Err(invalid("boundary law has foreign exact support"));
        }
        continuous_activations(program, &BTreeSet::from([relation]))?;
        let typed = typed_relation(program, relation)?;
        let dag = typed.expression();
        require_closed_dag(dag, relation)?;
        let dependencies = dag
            .nodes()
            .iter()
            .filter_map(|node| match node {
                ExprNode::Symbol(SymbolRef::Field(id) | SymbolRef::Derivative(id)) => {
                    Some(id.erase())
                }
                ExprNode::Symbol(SymbolRef::Parameter(id)) => Some(id.erase()),
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

        let [root] = dag.roots() else {
            return Err(invalid("closed boundary law requires one residual root"));
        };
        let view = AdditiveResidualView::derive(dag, *root, relation)?;
        let mut operators = Vec::new();
        for leaf in view.leaves() {
            match dag.node(leaf.value()) {
                Some(ExprNode::Trace(value)) => {
                    let Some(ExprNode::Symbol(SymbolRef::Field(field))) = dag.node(*value) else {
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
                        operators.push((leaf, row, Some(field), PhysicalBoundaryQuantity::Trace));
                    }
                }
                Some(ExprNode::NormalComponent(_)) => {
                    for row in self.rows.iter().filter(|row| !row.flux.is_empty()) {
                        if self
                            .require_boundary_flux(
                                program,
                                boundary,
                                relation,
                                row.tested,
                                leaf.value(),
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
        let coefficients =
            super::super::linear::coefficients(program, self.dimension, &self.roles)?;
        let context = Context {
            program,
            dag,
            owner: relation,
            dimension: self.dimension,
            coefficients: &coefficients,
        };
        let count = components(&row.value_type, self.dimension)?;
        let mut datum = match datum_expression {
            None => BoundaryDatum::Components(vec![Data::constant(self.dimension, 0.0); count]),
            Some(id) if row.value_type.shape().is_scalar() => {
                BoundaryDatum::Components(vec![context.data(id, 0)?])
            }
            Some(id) => {
                let inner = match dag.node(id) {
                    Some(ExprNode::Trace(value)) => *value,
                    _ => id,
                };
                match dag.node(inner) {
                    Some(ExprNode::Gradient(potential)) => {
                        let potential = context.data(*potential, 0)?;
                        let primal = potential
                            .clone()
                            .multiply(Data::constant(self.dimension, 0.0));
                        BoundaryDatum::Components(
                            (0..count)
                                .map(|axis| {
                                    Ok(primal.clone().add(
                                        potential.coordinate_derivative(axis, self.dimension)?,
                                    ))
                                })
                                .collect::<Result<_, Diagnostic>>()?,
                        )
                    }
                    Some(ExprNode::NormalComponent(tensor)) => {
                        let proof = OperatorApplicationProof::classify(
                            &typed,
                            *tensor,
                            StandardPureOperator::IsotropicLift,
                        )
                        .map_err(|_| {
                            invalid("boundary normal datum lacks an exact isotropic-lift proof")
                        })?
                        .ok_or_else(|| {
                            invalid("boundary normal datum requires an isotropic lift")
                        })?;
                        BoundaryDatum::NormalMultiple(context.data(proof.operand(), 0)?)
                    }
                    _ => {
                        return Err(invalid(
                            "vector boundary datum requires a potential gradient or isotropic normal lift",
                        ));
                    }
                }
            }
        };
        if values
            .first()
            .is_some_and(|value| value.sign() == operator.sign())
        {
            let negative = Data::constant(self.dimension, -1.0);
            match &mut datum {
                BoundaryDatum::Components(components) => {
                    for value in components {
                        *value = value.clone().multiply(negative.clone());
                    }
                }
                BoundaryDatum::NormalMultiple(value) => *value = value.clone().multiply(negative),
            }
        }
        Ok(RegionBoundaryLaw {
            binding: BoundaryRelationBinding::new(boundary, relation),
            tested: row.tested,
            trace_field: *trace_field,
            quantity: *quantity,
            dependencies,
            operator: operator.value(),
            datum_expression,
            datum,
        })
    }
}
