//! Numerical admission of exact retained physical Law terms.
//!
//! The physical source and accumulation are selected by their owned identities;
//! only the requested affine diffusion realization is recognized here.

use super::*;
use eqiora_schema::kernel::ConservationTerms;

pub(super) fn retained_balance(
    program: &KernelProgram,
    relation: RawId,
    field: RawId,
    dimensions: usize,
    bounds: &[[f64; 2]],
    terms: ConservationTerms,
) -> Result<
    (
        DimExponents,
        Option<ScalarStorageMeaning>,
        ScalarFluxMeaning,
        Option<VolumetricSourceMeaning>,
    ),
    Diagnostic,
> {
    require_continuous_relation(program, relation)?;
    let typed = typed_relation(program, relation)?;
    let expression = typed.expression();
    let flux = terms.flux();
    // This realization admits F = -k grad(u), k positive affine. The sign is
    // physical: reversing F changes ellipticity and is not absorbed into a
    // residual-wide sign convention.
    let (coefficient, outward_negative) =
        signed_flux_coefficient(program, expression, flux, field, relation, dimensions)?;
    if !outward_negative {
        return Err(lowering_error(
            relation,
            "scalar diffusion requires physical outward flux -k grad(field) with positive k",
        ));
    }
    validate_positive_affine_coefficient(&coefficient, bounds, relation)?;
    let storage = terms
        .storage()
        .map(|(stored, _accumulation)| {
            let is_field = |id| matches!(expression.node(id), Some(ExprNode::Symbol(SymbolRef::Field(id))) if id.erase() == field);
            let coefficient = match expression.node(stored) {
                Some(ExprNode::Symbol(SymbolRef::Field(id))) if id.erase() == field => None,
                Some(ExprNode::Mul(left, right)) if is_field(*left) => Some(*right),
                Some(ExprNode::Mul(left, right)) if is_field(*right) => Some(*left),
                _ => return Err(lowering_error(relation, "first retained Law storage requires one constant coefficient times the exact Field")),
            };
            // Storage correspondence has already been independently checked by
            // Kernel admission; numerical selection reads the physical storage.
            super::balance::storage_coefficient(program, expression, coefficient,
                stored, relation, dimensions)
        })
        .transpose()?;
    if contains_state_symbol(expression, terms.source()) {
        return Err(lowering_error(
            relation,
            "first scalar-conservation realization requires a state-independent physical source",
        ));
    }
    let source_dimension = node_dimension(&typed, terms.source(), relation)?;
    let source = VolumetricSourceMeaning {
        expression: spatial_expression::lower(
            program,
            expression,
            terms.source(),
            relation,
            dimensions,
        )?,
        dimension: source_dimension,
        integrated_dimension: integrate_dimension(source_dimension, dimensions, relation)?,
        lineage: ScalarTermLineage {
            relation,
            expression: terms.source(),
        },
    };
    Ok((
        source_dimension,
        storage,
        ScalarFluxMeaning {
            coefficient,
            lineage: ScalarTermLineage {
                relation,
                expression: flux,
            },
        },
        Some(source),
    ))
}

/// Remove only exact unary signs from a scalar-times-gradient expression.
/// Parameter identities and the remaining coefficient tape stay unchanged.
pub(super) fn signed_flux_coefficient(
    program: &KernelProgram,
    expression: &ExprDag,
    value: ExprId,
    field: RawId,
    owner: RawId,
    dimensions: usize,
) -> Result<(ScalarSpatialExpression, bool), Diagnostic> {
    fn has_gradient(expression: &ExprDag, value: ExprId, field: RawId) -> bool {
        match expression.node(value) {
            Some(ExprNode::Neg(value)) => has_gradient(expression, *value, field),
            Some(ExprNode::Mul(left, right)) => {
                has_gradient(expression, *left, field) || has_gradient(expression, *right, field)
            }
            Some(ExprNode::Gradient(argument)) => matches!(expression.node(*argument),
                Some(ExprNode::Symbol(SymbolRef::Field(id))) if id.erase() == field),
            _ => false,
        }
    }
    if let Some(ExprNode::Neg(value)) = expression.node(value) {
        let (coefficient, negative) =
            signed_flux_coefficient(program, expression, *value, field, owner, dimensions)?;
        return Ok((coefficient, !negative));
    }
    if let Some(ExprNode::Mul(left, right)) = expression.node(value) {
        let left_gradient = has_gradient(expression, *left, field);
        let right_gradient = has_gradient(expression, *right, field);
        if left_gradient != right_gradient {
            let (gradient, mut factor) = if left_gradient {
                (*left, *right)
            } else {
                (*right, *left)
            };
            let (coefficient, mut negative) =
                signed_flux_coefficient(program, expression, gradient, field, owner, dimensions)?;
            while let Some(ExprNode::Neg(inner)) = expression.node(factor) {
                negative = !negative;
                factor = *inner;
            }
            let factor = spatial_expression::lower(program, expression, factor, owner, dimensions)?;
            return Ok((
                if left_gradient {
                    coefficient.multiply(factor)
                } else {
                    factor.multiply(coefficient)
                },
                negative,
            ));
        }
    }
    lower_flux_coefficient(program, expression, value, field, owner, dimensions)
        .map(|value| (value, false))
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_graph::{GraphStore, InMemoryGraphStore};

    fn native_law(reversed_flux: bool) -> KernelProgram {
        native_law_with_boundary(reversed_flux, "trace(u) = 0;")
    }

    fn native_law_with_boundary(reversed_flux: bool, boundary: &str) -> KernelProgram {
        let source = law_source(reversed_flux, boundary);
        admit_source(&source)
    }

    fn law_source(reversed_flux: bool, boundary: &str) -> String {
        r#"
model Balance() {
  domain body = box(0, 1);
  domain lower_face = boundary(body, axis = 0, side = lower);
  domain upper_face = boundary(body, axis = 0, side = upper);
  variable u: 1 on body;
  parameter k: 1 = 2;
  parameter q: 1 / m ^ 2 = 4;
  parameter outward: 1 / m = 2;
  parameter transfer: 1 / m = 3;
  law balance on body { flux PHYSICAL_FLUX; source q; }
  relation lower_value on lower_face { trace(u) = 0; }
  relation upper_value on upper_face { BOUNDARY }
}
"#
        .replace("BOUNDARY", boundary)
        .replace(
            "PHYSICAL_FLUX",
            if reversed_flux {
                "k * grad(u)"
            } else {
                "-k * grad(u)"
            },
        )
    }

    fn admit_source(source: &str) -> KernelProgram {
        let compiled = eqiora_compiler::compile("native-law.eqi", source)
            .unwrap()
            .remove(0);
        let (transaction, model, _) = compiled.into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
    }

    #[test]
    fn retained_source_law_preserves_physical_source_and_flux_sign() {
        let program = native_law(false);
        let descriptor = recognize_scalar_conservation(&program).unwrap();
        let region = &descriptor.regions[0];
        assert_eq!(region.flux.coefficient.constant_value(), Some(2.0));
        assert_eq!(
            region.source.as_ref().unwrap().expression.constant_value(),
            Some(4.0)
        );
        assert!(region.storage.is_none());
        // A reversed physical flux is well-typed mathematics, but fails the
        // positive diffusion realization at its own numerical admission gate.
        assert!(recognize_scalar_conservation(&native_law(true)).is_err());
    }
    #[test]
    fn retained_linear_storage_uses_checked_physical_value_not_derivative_syntax() {
        let source = law_source(false, "trace(u) = 0;")
            .replace("variable u", "state u")
            .replace(
                "law balance",
                "parameter c: s / m ^ 2 = 3; initial { u=0; } law balance",
            )
            .replace("{ flux", "{ storage c * u; flux");
        let program = admit_source(&source);
        let descriptor = recognize_scalar_conservation(&program).unwrap();
        assert_eq!(
            descriptor.regions[0]
                .storage
                .as_ref()
                .unwrap()
                .coefficient
                .constant_value(),
            Some(3.0)
        );
        for unsupported in [
            source.replace("storage c * u;", "storage c * u * u;"),
            source.replace(
                "parameter c: s / m ^ 2 = 3;",
                "parameter c: s / m ^ 2 = -3;",
            ),
        ] {
            // Both remain valid physical Laws; this realization has a narrower
            // positive, constant linear-storage contract.
            let admitted = admit_source(&unsupported);
            assert!(recognize_scalar_conservation(&admitted).is_err());
        }
    }

    #[test]
    fn physical_outward_boundary_values_convert_to_diffusion_conormal_once() {
        let program = native_law_with_boundary(false, "normal(-k * grad(u)) = outward;");
        let descriptor = recognize_scalar_conservation(&program).unwrap();
        let law = &descriptor.regions[0].exterior[&(0, BoundarySide::Upper)].law;
        assert!(
            matches!(law, ScalarExteriorLaw::PrescribedOutwardFlux { value, .. } if value.constant_value() == Some(-2.0))
        );
        let robin = native_law_with_boundary(
            false,
            "normal(-k * grad(u)) - transfer * trace(u) = outward;",
        );
        let descriptor = recognize_scalar_conservation(&robin).unwrap();
        let law = &descriptor.regions[0].exterior[&(0, BoundarySide::Upper)].law;
        assert!(
            matches!(law, ScalarExteriorLaw::Robin { trace_coefficient, value, .. }
            if trace_coefficient.constant_value() == Some(3.0) && value.constant_value() == Some(-2.0))
        );
    }
}
