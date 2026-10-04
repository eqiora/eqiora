use super::super::*;
use crate::form_compiler::scalar::typed_relation;
use eqiora_compiler::{AuthoredFormExpressionV1, AuthoredFormulationProjection};
use eqiora_core::Diagnostic;
use eqiora_sem::KernelProgram;

impl PrimalGalerkinCorrespondence {
    pub(in crate::form_compiler) fn replay_authored(
        &self,
        projection: &AuthoredFormulationProjection,
        program: &KernelProgram,
        dimension: usize,
    ) -> Result<bool, Diagnostic> {
        let mut variation_dimensions = Vec::new();
        let mut remaining = 65536usize;
        for (_, left, right) in projection.equations() {
            let mut pending = vec![left, right];
            while let Some(expression) = pending.pop() {
                remaining = remaining.checked_sub(1).ok_or_else(|| {
                    rejection_with(
                        projection,
                        "variation sum exceeds the bounded replay inventory",
                    )
                })?;
                match expression {
                    AuthoredFormExpressionV1::Add { left, right }
                    | AuthoredFormExpressionV1::Sub { left, right } => {
                        pending.push(left);
                        pending.push(right);
                    }
                    AuthoredFormExpressionV1::Neg { value } => pending.push(value),
                    _ => {}
                }
                if let AuthoredFormExpressionV1::Variation {
                    functional_ulid, ..
                } = expression
                {
                    let id = functional_ulid
                        .parse::<ulid::Ulid>()
                        .map(eqiora_core::Id::<eqiora_core::entity::kinds::Observable>::from_ulid)
                        .map_err(|_| {
                            rejection_with(projection, "variation Observable identity is invalid")
                        })?;
                    let Some(eqiora_schema::kernel::KernelNode::Observable(functional)) =
                        program.node(id.erase())
                    else {
                        return Err(rejection_with(
                            projection,
                            "variation Observable is outside the live Model",
                        ));
                    };
                    let typed = program.typed_observable(id).map_err(|_| {
                        rejection_with(projection, "variation energy has invalid live types")
                    })?;
                    expression.check_functional_variation(functional, &typed)?;
                    variation_dimensions.push(functional.value_type().dimension());
                }
            }
        }
        let has_variation = !variation_dimensions.is_empty();
        let test_dimension = if has_variation {
            let Some(eqiora_schema::kernel::KernelNode::Field(field)) =
                program.node(self.law.unknown)
            else {
                return Err(rejection_with(
                    projection,
                    "variation trial is not a live Field",
                ));
            };
            field.dimension()
        } else {
            eqiora_core::DimExponents::DIMENSIONLESS
        };
        self.replay_authored_restriction(projection, test_dimension)
            .map_err(|message| rejection_with(projection, message))?;
        let typed = typed_relation(program, self.law.relations[0])?;
        let dag = typed.expression();
        if has_variation {
            let invalid_dimension = || {
                rejection_with(
                    projection,
                    "variation dimension differs from the strong-law test pairing",
                )
            };
            let spatial_dimension = i32::try_from(dimension).map_err(|_| invalid_dimension())?;
            let measure =
                eqiora_core::DimExponents::from_integers([0, spatial_dimension, 0, 0, 0, 0, 0])
                    .ok_or_else(invalid_dimension)?;
            let paired_dimension = typed
                .node_type(dag.roots()[0])
                .ok_or_else(invalid_dimension)?
                .dimension()
                .mul(test_dimension)
                .and_then(|dimension| dimension.mul(measure))
                .ok_or_else(invalid_dimension)?;
            if variation_dimensions
                .iter()
                .any(|dimension| *dimension != paired_dimension)
            {
                return Err(invalid_dimension());
            }
        }
        Ok(has_variation)
    }
}

fn rejection_with(projection: &AuthoredFormulationProjection, message: &str) -> Diagnostic {
    Diagnostic::error(
        eqiora_core::diagnostic::codes::INVALID_DISCRETIZATION,
        format!(
            "authored primal Formulation rejected: {message} (source identity {})",
            projection.source_identity()
        ),
    )
}
