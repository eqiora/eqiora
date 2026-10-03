//! Resolve mathematical source types without choosing numerical storage.

use eqiora_core::{Diagnostic, ValueShape, diagnostic::codes};
use eqiora_core::{ValueFrame, ValueType};
use eqiora_lang::{ValueTypeSyntax, ValueTypeSyntaxKind};
use eqiora_schema::kernel::typing::SpatialSupport;

use crate::{diagnostics::source_error, dimensions::lower_dimension};

pub(crate) fn lower_scalar_type(
    file: &str,
    syntax: &ValueTypeSyntax,
) -> Result<ValueType, Diagnostic> {
    let non_scalar = || {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            syntax.range(),
            "scalar physical quantities require scalar mathematical types",
        )
    };
    if matches!(
        syntax.kind(),
        ValueTypeSyntaxKind::Array { .. }
            | ValueTypeSyntaxKind::Vector { .. }
            | ValueTypeSyntaxKind::Tensor { .. }
    ) {
        return Err(non_scalar());
    }
    let value = lower_value_type::<()>(file, syntax, None)?;
    if !value.shape().is_scalar() {
        return Err(non_scalar());
    }
    Ok(value)
}

pub(crate) fn lower_value_type<I>(
    file: &str,
    syntax: &ValueTypeSyntax,
    support: Option<&SpatialSupport<I>>,
) -> Result<ValueType, Diagnostic> {
    let invalid =
        |message: String| source_error(codes::LANGUAGE_TYPE_ERROR, file, syntax.range(), message);
    match syntax.kind() {
        ValueTypeSyntaxKind::Named(name) => {
            if let Some(value) = syntax.resolved_nominal() {
                return Ok(value.clone());
            }
            let expression = eqiora_lang::SourceAstFactory::expression(
                if name.segments().len() == 1 {
                    eqiora_lang::ExprKind::Name(name.as_str().to_owned())
                } else {
                    eqiora_lang::ExprKind::Path(name.clone())
                },
                syntax.range(),
            )
            .map_err(|error| invalid(error.message().to_owned()))?;
            ValueType::scalar(
                eqiora_core::ScalarDomain::Real,
                lower_dimension(file, &expression)?,
            )
            .map_err(|error| invalid(error.to_string()))
        }
        ValueTypeSyntaxKind::Coordinates { .. }
        | ValueTypeSyntaxKind::LinearMap { .. }
        | ValueTypeSyntaxKind::Counts(_)
        | ValueTypeSyntaxKind::Index(_) => syntax.resolved_nominal().cloned().ok_or_else(|| {
            invalid("nominal value type requires its exact lexical declaration binding".into())
        }),
        ValueTypeSyntaxKind::Scalar { domain, dimension } => {
            ValueType::scalar(*domain, lower_dimension(file, dimension)?)
                .map_err(|error| invalid(error.to_string()))
        }
        ValueTypeSyntaxKind::Array { element, extent } => {
            let extent = crate::hierarchy::closed_index(extent)?;
            lower_value_type(file, element, support)?
                .array(extent)
                .map_err(|error| invalid(error.to_string()))
        }
        ValueTypeSyntaxKind::Vector { scalar, extent } => {
            spatial_type(file, syntax, scalar, &[*extent], support)
        }
        ValueTypeSyntaxKind::Tensor { scalar, extents } => {
            spatial_type(file, syntax, scalar, extents, support)
        }
    }
}

pub(crate) fn component_dimension(
    file: &str,
    syntax: &ValueTypeSyntax,
) -> Result<eqiora_core::DimExponents, Diagnostic> {
    match syntax.kind() {
        ValueTypeSyntaxKind::Vector { scalar, .. } | ValueTypeSyntaxKind::Tensor { scalar, .. } => {
            component_dimension(file, scalar)
        }
        ValueTypeSyntaxKind::Array { element, .. } => component_dimension(file, element),
        _ => lower_value_type::<()>(file, syntax, None).map(|value| value.dimension()),
    }
}

fn spatial_type<I>(
    file: &str,
    syntax: &ValueTypeSyntax,
    scalar: &ValueTypeSyntax,
    extents: &[u32],
    support: Option<&SpatialSupport<I>>,
) -> Result<ValueType, Diagnostic> {
    let invalid =
        |message: String| source_error(codes::LANGUAGE_TYPE_ERROR, file, syntax.range(), message);
    let support = support.ok_or_else(|| {
        invalid("spatial value type requires an exact support to determine its frame".into())
    })?;
    let dimension = u32::try_from(support.dimensions())
        .ok()
        .filter(|n| *n > 0)
        .ok_or_else(|| invalid("support has no valid ambient spatial dimension".into()))?;
    if extents.iter().any(|extent| *extent != dimension) {
        return Err(invalid(format!(
            "spatial type extents must match support ambient dimension {dimension}",
        )));
    }
    let scalar = lower_value_type(file, scalar, Some(support))?;
    let shape =
        ValueShape::new(extents.iter().copied()).map_err(|error| invalid(error.to_string()))?;
    ValueType::shaped(
        scalar.scalar_domain(),
        scalar.dimension(),
        shape,
        ValueFrame::SpatialCartesian,
    )
    .map_err(|error| invalid(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{DimExponents, ScalarDomain};
    use eqiora_lang::{Item, parse};

    fn syntax(value: &str) -> ValueTypeSyntax {
        let source = format!("model M() {{ parameter value: {value} = 0; }}");
        let document = parse("types.eqi", &source).into_document().unwrap();
        let Item::Parameter(parameter) = &document.models()[0].items()[0] else {
            panic!("parameter");
        };
        parameter.value_type().clone()
    }

    #[test]
    fn source_types_resolve_domains_dimensions_and_component_roles_together() {
        let volume = SpatialSupport::Volume {
            domain: "body",
            dimensions: 3,
        };
        for (source, domain, dimension, extents, array_rank, frame) in [
            (
                "Pa",
                ScalarDomain::Real,
                [1, -1, -2, 0, 0, 0, 0],
                vec![],
                0,
                ValueFrame::Invariant,
            ),
            (
                "complex<V>",
                ScalarDomain::Complex,
                [1, 2, -3, -1, 0, 0, 0],
                vec![],
                0,
                ValueFrame::Invariant,
            ),
            (
                "vector<V, 3>",
                ScalarDomain::Real,
                [1, 2, -3, -1, 0, 0, 0],
                vec![3],
                0,
                ValueFrame::SpatialCartesian,
            ),
            (
                "tensor<complex<Pa>, 3, 3>",
                ScalarDomain::Complex,
                [1, -1, -2, 0, 0, 0, 0],
                vec![3, 3],
                0,
                ValueFrame::SpatialCartesian,
            ),
            (
                "array<vector<complex<V>, 3>, 2>",
                ScalarDomain::Complex,
                [1, 2, -3, -1, 0, 0, 0],
                vec![2, 3],
                1,
                ValueFrame::SpatialCartesian,
            ),
            (
                "array<array<V, 3>, 2>",
                ScalarDomain::Real,
                [1, 2, -3, -1, 0, 0, 0],
                vec![2, 3],
                2,
                ValueFrame::Invariant,
            ),
        ] {
            let value = lower_value_type("types.eqi", &syntax(source), Some(&volume)).unwrap();
            assert_eq!(value.scalar_domain(), domain, "{source}");
            assert_eq!(
                value.dimension(),
                DimExponents::from_integers(dimension).unwrap()
            );
            assert_eq!(value.shape(), &ValueShape::new(extents).unwrap());
            assert_eq!(value.array_rank(), array_rank);
            assert_eq!(value.frame(), frame);
        }
        let wavefunction =
            lower_value_type::<()>("types.eqi", &syntax("complex<m^(-1/2)>"), None).unwrap();
        let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
        assert_eq!(wavefunction.dimension().pow(2, 1), length.pow(-1, 1));
    }

    #[test]
    fn spatial_extents_cannot_choose_a_frame_or_override_support() {
        let volume = SpatialSupport::Volume {
            domain: "body",
            dimensions: 2,
        };
        for (source, support) in [
            ("vector<V, 2>", None),
            ("vector<V, 3>", Some(&volume)),
            ("tensor<V, 2, 3>", Some(&volume)),
        ] {
            let syntax = syntax(source);
            let error = lower_value_type("types.eqi", &syntax, support).unwrap_err();
            assert_eq!(error.code(), codes::LANGUAGE_TYPE_ERROR);
            let span = error.source_span().unwrap();
            assert_eq!(span.file, "types.eqi");
            assert_eq!(
                (span.start, span.end),
                (syntax.range().start(), syntax.range().end())
            );
        }
        assert!(lower_value_type::<()>("types.eqi", &syntax("array<V, 3>"), None).is_ok());
    }
}
