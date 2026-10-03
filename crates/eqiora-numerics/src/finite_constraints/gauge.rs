//! Explicit finite constant-shift reference over the existing typed affine lowering.
use super::*;
use crate::nullspace::{NullspaceConstraint, NullspaceEvidence, NullspaceLinearSolution};
use crate::physical_network::AffineCsrStorage;
use eqiora_compiler::{AuthoredFormExpressionV1 as E, AuthoredFormulationProjection};
use eqiora_core::{ScalarDomain, ValueFrame};
use eqiora_ir::ComponentScalarization;
use eqiora_schema::kernel::{ExprDagBuilder, KernelNode};
use eqiora_solver::{CanonicalCsrSystemView, LinearOperatorProperties};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FiniteGauge {
    system: CanonicalCsrSystemView,
    reference: NullspaceConstraint,
    bordered_load_norm: f64,
}
impl FiniteGauge {
    pub(crate) fn residual_target(&self, plan: SolverPlan) -> Result<f64, Diagnostic> {
        plan.residual_target(self.bordered_load_norm)
    }

    pub(crate) fn solve(
        &self,
        request: LinearSolveRequest<'_>,
    ) -> Result<NullspaceLinearSolution, Diagnostic> {
        crate::nullspace::solve_canonical_with_nullspace(request, &self.system, &self.reference)
    }
    pub(crate) fn assess(
        &self,
        values: &[f64],
        multiplier: f64,
        plan: SolverPlan,
    ) -> Result<NullspaceEvidence, Diagnostic> {
        crate::nullspace::assess_canonical_with_nullspace(
            &self.system,
            &self.reference,
            values,
            multiplier,
            plan,
        )
    }
}
impl FiniteConstraintProblem {
    pub(crate) fn gauge(
        &self,
        form: &AuthoredFormulationProjection,
    ) -> Result<FiniteGauge, Diagnostic> {
        let n = self.symbols.len();
        if form.finite_space().is_none()
            || self.enforcement.is_some()
            || self.coordinate_count() != n
            || form.trial_ulids().len() != n
            || form.equations().len() != self.relations.len()
            || self.relations.len() != n
            || form.gauge_field_ulids() != Some(form.trial_ulids())
        {
            return Err(invalid(
                "finite gauge requires an exact complete real scalar equality closure with explicit reference",
            ));
        }
        let load_dimension = self
            .relations
            .first()
            .and_then(|relation| relation.dimensions.first())
            .map(|pair| pair.0)
            .ok_or_else(|| invalid("finite gauge has no original equality dimension"))?;
        let mut storage = AffineCsrStorage::new(n, n)?;
        let mut rhs_terms = vec![None; n];
        // Bind equation i to declared trial i, then emit rows in the existing
        // canonical Field order. No incidental Relation-ID ordering defines symmetry.
        for symbol in &self.symbols {
            let SymbolRef::Field(field) = symbol else {
                return Err(invalid("finite gauge requires Fields"));
            };
            let Some(KernelNode::Field(definition)) = self.kernel.node(field.erase()) else {
                return Err(invalid("missing gauge Field"));
            };
            if definition.value_type().scalar_domain() != ScalarDomain::Real
                || definition.value_type().frame() != ValueFrame::Invariant
                || !definition.shape().is_scalar()
                || definition.dimension() != self.dimensions[0]
            {
                return Err(invalid(
                    "finite constant-shift coordinates require invariant real scalars with one dimension",
                ));
            }
            let row = form
                .trial_ulids()
                .iter()
                .position(|id| id == &field.ulid().to_string())
                .ok_or_else(|| invalid("gauge omits an original Field"))?;
            let (id, authored_left, authored_right) = &form.equations()[row];
            let relation = self
                .relations
                .iter()
                .find(|relation| relation.id.ulid().to_string() == *id)
                .ok_or_else(|| invalid("gauge equation refers to a foreign Relation"))?;
            if relation.conditions.as_slice() != [RelationConditionKind::Equality]
                || relation
                    .dimensions
                    .first()
                    .is_none_or(|pair| pair.0 != load_dimension)
            {
                return Err(invalid(
                    "finite gauge requires one equality per original Relation with a common load dimension",
                ));
            }
            let [left, right] = relation.expression.roots() else {
                return Err(invalid("finite gauge requires scalar operand pairs"));
            };
            let left_projection = E::from_expression(&relation.expression, *left)?;
            let right_projection = E::from_expression(&relation.expression, *right)?;
            if left_projection.as_ref() != Some(authored_left)
                || right_projection.as_ref() != Some(authored_right)
            {
                return Err(invalid(
                    "finite authored equation differs from the live original operands",
                ));
            }
            // Keep the first finite profile explicit: homogeneous affine LHS,
            // Field-independent RHS. Both facts come from the existing affine IR.
            for (root, lhs) in [(*left, true), (*right, false)] {
                let expression = ExprDagBuilder::from_dag(&relation.expression).finish([root])?;
                let typed = coordinates::typed_expression(&self.kernel, &expression)?;
                let scalarized = ComponentScalarization::lower(&typed)?;
                let [row] = scalarized.rows() else {
                    return Err(invalid("finite gauge operand is not a real scalar"));
                };
                let affine = row.bind_affine(&self.coordinates, &self.bindings)?;
                if (lhs && affine.offsets().iter().any(|value| *value != 0.0))
                    || (!lhs && affine.coefficients().iter().any(|value| *value != 0.0))
                {
                    return Err(invalid(
                        "finite gauge requires homogeneous affine left operands and independent loads",
                    ));
                }
            }
            rhs_terms[row] = Some(authored_right.clone());
            let expression = preparation::branch_expression(relation, 0, &mut 0)?
                .ok_or_else(|| invalid("missing finite equality"))?;
            let typed = coordinates::typed_expression(&self.kernel, &expression)?;
            for row in ComponentScalarization::lower(&typed)?.rows() {
                storage.append(&row.bind_affine(&self.coordinates, &self.bindings)?)?;
            }
        }
        storage.finish()?;
        let system = CanonicalCsrSystemView::new(&storage, LinearOperatorProperties::Symmetric)?;
        let balance = rhs_terms
            .into_iter()
            .map(|term| term.expect("complete unique trial coverage"))
            .reduce(|left, right| E::Add {
                left: Box::new(left),
                right: Box::new(right),
            })
            .ok_or_else(|| invalid("empty gauge"))?;
        let (left, right) = form
            .gauge_compatibility()
            .ok_or_else(|| invalid("missing finite compatibility declaration"))?;
        if !crate::form_compiler::equivalent_authored_expression(left, &balance)
            || !matches!(right, E::Number { value } if *value == 0.0)
        {
            return Err(invalid(
                "finite compatibility differs from the sum of original loads",
            ));
        }
        let (left, right) = form
            .gauge_reference()
            .ok_or_else(|| invalid("missing finite reference declaration"))?;
        let E::Field { ulid } = left else {
            return Err(invalid(
                "finite reference currently fixes one explicit coordinate",
            ));
        };
        let coordinate = self
            .symbols
            .iter()
            .position(
                |symbol| matches!(symbol, SymbolRef::Field(id) if id.ulid().to_string() == *ulid),
            )
            .ok_or_else(|| invalid("finite reference names a foreign Field"))?;
        let value = match right {
            E::Parameter { ulid } => self
                .kernel
                .nodes()
                .find_map(|node| match node {
                    KernelNode::Parameter(parameter)
                        if parameter.id().ulid().to_string() == *ulid =>
                    {
                        parameter.real_scalar_value()
                    }
                    _ => None,
                })
                .filter(|quantity| quantity.dim() == self.dimensions[coordinate])
                .ok_or_else(|| {
                    invalid("reference requires an exact real Parameter with the Field dimension")
                })?
                .value(),
            E::Number { value }
                if *value == 0.0 || self.dimensions[coordinate] == DimExponents::DIMENSIONLESS =>
            {
                *value
            }
            E::Neg { value } if self.dimensions[coordinate] == DimExponents::DIMENSIONLESS => {
                match **value {
                    E::Number { value } => -value,
                    _ => {
                        return Err(invalid(
                            "finite reference requires a literal or exact Parameter",
                        ));
                    }
                }
            }
            _ => {
                return Err(invalid(
                    "finite reference requires a literal or exact Parameter",
                ));
            }
        };
        let mut weights = vec![0.; n];
        weights[coordinate] = 1.;
        use eqiora_solver::{
            FixedOrderInnerProduct, ReplicatedLinearExecution, SERIAL_LINEAR_EXECUTION,
        };
        let mut load = system.right_hand_side().to_vec();
        load.push(value);
        let bordered_load_norm = SERIAL_LINEAR_EXECUTION
            .inner_product(FixedOrderInnerProduct::new(&load, &load)?)?
            .sqrt();
        Ok(FiniteGauge {
            bordered_load_norm,
            system,
            reference: NullspaceConstraint::new(vec![1.; n], weights, value)?,
        })
    }
}
