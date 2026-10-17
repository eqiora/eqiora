//! One typed Parameter point for finite residuals and spatial coefficients.
use eqiora_core::{Diagnostic, Id, ScalarDomain, ValueLiteral, entity::kinds};
use eqiora_ir::ScalarSymbolCoordinate;
use eqiora_schema::kernel::{KernelNode, SymbolRef};
use eqiora_sem::KernelProgram;

pub(crate) fn coordinates(
    kernel: &KernelProgram,
    selected: &[Id<kinds::Parameter>],
) -> Result<Vec<ScalarSymbolCoordinate>, Diagnostic> {
    let mut coordinates = Vec::new();
    for (index, id) in selected.iter().enumerate() {
        if selected[..index].contains(id) {
            return Err(invalid("Parameter selection repeats an exact identity"));
        }
        let Some(KernelNode::Parameter(parameter)) = kernel.node(id.erase()) else {
            return Err(invalid("Parameter is outside the exact Model"));
        };
        coordinates.extend(ScalarSymbolCoordinate::for_value(
            SymbolRef::Parameter(*id),
            parameter.value_type(),
        )?);
    }
    Ok(coordinates)
}

/// A validated typed point; construction owns identity, shape and finiteness checks.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct ParameterPoint {
    values: Vec<(Id<kinds::Parameter>, ValueLiteral)>,
}

impl ParameterPoint {
    pub(crate) fn candidates(&self) -> &[(Id<kinds::Parameter>, ValueLiteral)] {
        &self.values
    }

    pub(crate) fn get(&self, id: Id<kinds::Parameter>) -> Option<&ValueLiteral> {
        self.values
            .iter()
            .find(|(candidate, _)| *candidate == id)
            .map(|(_, value)| value)
    }

    /// Selected values use Parameter order, then row-major components and real/imaginary parts.
    /// Unselected Parameters retain the canonical Model value, never an earlier point.
    pub(crate) fn bind(
        kernel: &KernelProgram,
        selected: &[Id<kinds::Parameter>],
        values: &[f64],
    ) -> Result<Self, Diagnostic> {
        if coordinates(kernel, selected)?.len() != values.len()
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(invalid(
                "Parameter point requires complete finite typed coordinates",
            ));
        }
        let mut replacements = std::collections::HashMap::new();
        let mut cursor = 0;
        for id in selected {
            let Some(KernelNode::Parameter(parameter)) = kernel.node(id.erase()) else {
                unreachable!("validated Parameter");
            };
            let ty = parameter.value_type();
            let width = if ty.scalar_domain() == ScalarDomain::Complex {
                2
            } else {
                1
            };
            let length = ty
                .shape()
                .component_count()
                .expect("validated numeric shape")
                * width;
            let value = ValueLiteral::new(
                ty.clone(),
                values[cursor..cursor + length]
                    .chunks_exact(width)
                    .map(|part| (part[0], part.get(1).copied().unwrap_or(0.))),
            )
            .map_err(|error| invalid(error.to_string()))?;
            cursor += length;
            replacements.insert(*id, value);
        }
        Ok(Self {
            values: kernel
                .nodes()
                .filter_map(|node| match node {
                    KernelNode::Parameter(parameter) => Some((
                        parameter.id(),
                        replacements
                            .remove(&parameter.id())
                            .unwrap_or_else(|| parameter.value().clone()),
                    )),
                    _ => None,
                })
                .collect(),
        })
    }
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(eqiora_core::diagnostic::codes::INVALID_REALIZATION, message)
}
