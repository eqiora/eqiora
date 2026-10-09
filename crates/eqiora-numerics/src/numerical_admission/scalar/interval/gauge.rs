//! Match the authored compatibility integral to retained source and boundary terms.
use super::*;
use eqiora_compiler::AuthoredFormExpressionV1 as E;
use eqiora_schema::kernel::KernelNode;
use eqiora_solver::AlgebraicConstraint;

pub(in crate::numerical_admission) fn admit(
    program: &KernelProgram,
    equations: &ExecutableLinearEquations<f64>,
    authored: &AuthoredFormulationProjection,
) -> Result<Option<AlgebraicConstraint>, Diagnostic> {
    let Some(fields) = authored.gauge_field_ulids() else {
        return Ok(None);
    };
    let [field] = fields else {
        return Err(invalid("interval gauge requires exactly one Field"));
    };
    let descriptor = equations.conservation_descriptor(program)?;
    let region = descriptor.regions().next().expect("single scalar region");
    if region.dimensions() != 1 || field.as_str() != region.field().ulid().to_string() {
        return Err(invalid(
            "constant gauge requires the exact one-dimensional scalar Field",
        ));
    }
    let domain = region.domain().ulid().to_string();
    let expected_reference = E::Integrate {
        domain_ulid: domain.clone(),
        integrand: Box::new(E::Field {
            ulid: field.clone(),
        }),
    };
    let zero = E::Number { value: 0.0 };
    let same = crate::form_compiler::equivalent_authored_expression;
    let Some((left, right)) = authored.gauge_reference() else {
        return Err(invalid("gauge has no reference condition"));
    };
    if !same(left, &expected_reference) || !same(right, &zero) {
        return Err(invalid(
            "scalar interval gauge currently requires the exact zero spatial integral",
        ));
    }
    let source = region
        .source()
        .ok_or_else(|| invalid("gauge compatibility requires the retained source term"))?;
    let project = |relation, expression| {
        let Some(KernelNode::Relation(definition)) = program.node(relation) else {
            return Err(invalid("gauge condition lost its original Relation"));
        };
        E::from_expression(definition.expression(), expression)?
            .ok_or_else(|| invalid("gauge condition exceeds the admitted expression profile"))
    };
    let mut balance = E::Integrate {
        domain_ulid: domain,
        integrand: Box::new(project(
            source.lineage().relation(),
            source.lineage().expression(),
        )?),
    };
    for side in [BoundarySide::Lower, BoundarySide::Upper] {
        let law = region
            .exterior_at(0, side)
            .ok_or_else(|| invalid("gauge requires both exact endpoints"))?
            .law();
        let datum = match law {
            ScalarExteriorLaw::ZeroOutwardFlux { .. } => continue,
            ScalarExteriorLaw::PrescribedOutwardFlux { .. } => {
                let lineage = law.lineage();
                let value = project(
                    lineage.relation(),
                    lineage
                        .datum_expression()
                        .ok_or_else(|| invalid("missing boundary datum"))?,
                )?;
                if lineage.datum_negative() {
                    E::Neg {
                        value: Box::new(value),
                    }
                } else {
                    value
                }
            }
            _ => {
                return Err(invalid(
                    "constant gauge requires two natural boundaries without a hidden essential reference",
                ));
            }
        };
        // The existing descriptor and assembler use n*k*grad(u). Physical
        // flux is its negative, hence integral(source) + conormal loads = 0.
        balance = E::Add {
            left: Box::new(balance),
            right: Box::new(datum),
        };
    }
    let Some((left, right)) = authored.gauge_compatibility() else {
        return Err(invalid("gauge has no compatibility condition"));
    };
    if !same(left, &balance) || !same(right, &zero) {
        return Err(invalid(
            "authored gauge compatibility differs from the original source and signed endpoint loads",
        ));
    }
    Ok(Some(AlgebraicConstraint::ZeroIntegral {
        field: region.field().downcast().expect("scalar Field"),
    }))
}
