//! Exact Parameter identities retain all typed numerical coordinates at each point.
use super::*;
use crate::spatial_design::SpatialDesignCoordinate;
use eqiora_core::{ScalarDomain, ValueLiteral};
use eqiora_ir::ScalarSymbolCoordinate;
use eqiora_schema::kernel::KernelNode;

impl FiniteConstraintProblem {
    pub(crate) fn parameter_coordinates(
        &self,
        selected: &[Id<kinds::Parameter>],
    ) -> Result<Vec<ScalarSymbolCoordinate>, Diagnostic> {
        let mut coordinates = Vec::new();
        for (index, id) in selected.iter().enumerate() {
            if selected[..index].contains(id) {
                return Err(invalid(
                    "finite Parameter selection repeats an exact identity",
                ));
            }
            let Some(KernelNode::Parameter(parameter)) = self.kernel.node(id.erase()) else {
                return Err(invalid("finite Parameter is outside the exact Model"));
            };
            coordinates.extend(ScalarSymbolCoordinate::for_value(
                SymbolRef::Parameter(*id),
                parameter.value_type(),
            )?);
        }
        Ok(coordinates)
    }

    /// Canonical Model defaults in selected-Parameter, row-major, real/imaginary order.
    pub(crate) fn parameter_point(
        &self,
        selected: &[Id<kinds::Parameter>],
    ) -> Result<(Vec<SpatialDesignCoordinate>, Vec<f64>), Diagnostic> {
        let mut design = Vec::new();
        let mut values = Vec::new();
        for coordinate in self.parameter_coordinates(selected)? {
            let SymbolRef::Parameter(id) = coordinate.symbol() else {
                unreachable!("selected Parameter coordinate");
            };
            let Some(KernelNode::Parameter(parameter)) = self.kernel.node(id.erase()) else {
                unreachable!("validated Parameter");
            };
            let component = parameter
                .value_type()
                .shape()
                .extents()
                .iter()
                .zip(coordinate.component_index())
                .fold(0usize, |flat, (extent, index)| {
                    flat * extent.get() as usize + *index as usize
                });
            design.push(SpatialDesignCoordinate::ModelParameter {
                parameter: id,
                component,
                imaginary: coordinate.is_imaginary(),
            });
            values.push(coordinates::component(parameter.value(), &coordinate)?);
        }
        Ok((design, values))
    }

    /// Bind complete real/complex Parameter values without replacing the Model.
    pub(crate) fn at_parameters(
        &self,
        selected: &[Id<kinds::Parameter>],
        values: &[f64],
    ) -> Result<Self, Diagnostic> {
        if self.parameter_coordinates(selected)?.len() != values.len()
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(invalid(
                "finite Parameter point requires complete finite typed coordinates",
            ));
        }
        let mut point = self.clone();
        point.parameter_candidates.clear();
        let mut cursor = 0;
        for id in selected {
            let Some(KernelNode::Parameter(parameter)) = self.kernel.node(id.erase()) else {
                unreachable!("validated Parameter");
            };
            let ty = parameter.value_type();
            let mut components = Vec::new();
            for _ in 0..ty
                .shape()
                .component_count()
                .expect("validated numeric shape")
            {
                let real = values[cursor];
                cursor += 1;
                let imaginary = if ty.scalar_domain() == ScalarDomain::Complex {
                    let imaginary = values[cursor];
                    cursor += 1;
                    imaginary
                } else {
                    0.
                };
                components.push((real, imaginary));
            }
            point.parameter_candidates.push((
                *id,
                ValueLiteral::new(ty.clone(), components)
                    .map_err(|error| invalid(error.to_string()))?,
            ));
        }
        // Reset unselected bindings to canonical Model values too: a new point
        // must not retain candidates from a previous evaluation.
        for (coordinate, binding) in &mut point.bindings {
            let SymbolRef::Parameter(id) = coordinate.symbol() else {
                continue;
            };
            let Some(KernelNode::Parameter(parameter)) = self.kernel.node(id.erase()) else {
                return Err(invalid("finite binding is outside the exact Model"));
            };
            let value = point
                .parameter_candidates
                .iter()
                .find(|(candidate, _)| *candidate == id)
                .map_or(parameter.value(), |(_, value)| value);
            *binding = coordinates::component(value, coordinate)?;
        }
        Ok(point)
    }
}
