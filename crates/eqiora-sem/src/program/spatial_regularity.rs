//! Admission of authored spatial regularity, independent of numerical spaces.
use std::collections::BTreeMap;

use eqiora_core::{Diagnostic, RawId, ScalarDomain, ValueFrame};
use eqiora_schema::kernel::typing::{SpatialSupport, TypedResidual, TypedResidualError};
use eqiora_schema::kernel::{FieldDef, KernelNode, SpatialRegularity, SymbolRef};

use super::{kernel_error, type_violation_diagnostic};

pub(super) fn validate_field(
    field: &FieldDef,
    support: Option<&SpatialSupport<RawId>>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let regularity = field.spatial_regularity();
    if regularity == SpatialRegularity::Unspecified {
        return;
    }
    let id = field.id().erase();
    let Some(SpatialSupport::Volume { dimensions, .. }) = support else {
        diagnostics.push(kernel_error(
            id,
            "authored spatial regularity requires one admitted physical volume support",
        ));
        return;
    };
    let value_type = field.value_type();
    if !matches!(
        value_type.scalar_domain(),
        ScalarDomain::Real | ScalarDomain::Complex
    ) {
        diagnostics.push(kernel_error(
            id,
            "spatial regularity requires real or complex components",
        ));
    }
    let rank = field.shape().extents().len();
    let cartesian = field.frame() == ValueFrame::SpatialCartesian && value_type.array_rank() == 0;
    let valid = match regularity {
        SpatialRegularity::HCurl => cartesian && rank == 1 && matches!(dimensions, 2 | 3),
        SpatialRegularity::HDiv => cartesian && rank >= 1,
        _ => true,
    };
    if !valid {
        diagnostics.push(kernel_error(id, match regularity {
            SpatialRegularity::HCurl => "H(curl) requires a Cartesian vector in two or three dimensions without outer arrays",
            _ => "H(div) requires a Cartesian tensor with a last spatial axis and no outer arrays",
        }));
    }
}

pub(super) fn validate_traces(
    typed: &TypedResidual<RawId>,
    owner: RawId,
    nodes: &BTreeMap<RawId, KernelNode>,
) -> Result<(), Vec<Diagnostic>> {
    typed
        .validate_trace_regularity(|symbol| {
            // The assertion holds for every value of this Field, including event sides.
            // It says nothing about the spatial regularity of its time derivatives.
            let id = match symbol {
                SymbolRef::Field(id) | SymbolRef::Pre(id) | SymbolRef::Next(id) => id,
                _ => return SpatialRegularity::Unspecified,
            };
            match nodes.get(&id.erase()) {
                Some(KernelNode::Field(field)) => field.spatial_regularity(),
                _ => SpatialRegularity::Unspecified,
            }
        })
        .map_err(|errors| {
            errors
                .into_iter()
                .map(|error| match error {
                    TypedResidualError::Type { node_index, error } => {
                        type_violation_diagnostic(owner, node_index, &error)
                    }
                    TypedResidualError::Symbol { error, .. } => match error {},
                })
                .collect()
        })
}
