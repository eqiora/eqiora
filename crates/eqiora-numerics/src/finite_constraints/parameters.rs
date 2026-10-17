//! Exact Parameter identities retain all typed numerical coordinates at each point.
use super::*;
use crate::spatial_design::SpatialDesignCoordinate;
use eqiora_schema::kernel::KernelNode;

impl FiniteConstraintProblem {
    /// Canonical Model defaults in selected-Parameter, row-major, real/imaginary order.
    pub(crate) fn parameter_point(
        &self,
        selected: &[Id<kinds::Parameter>],
    ) -> Result<(Vec<SpatialDesignCoordinate>, Vec<f64>), Diagnostic> {
        let mut design = Vec::new();
        let mut values = Vec::new();
        for coordinate in crate::parameter_point::coordinates(&self.kernel, selected)? {
            let SymbolRef::Parameter(id) = coordinate.symbol() else {
                unreachable!("selected Parameter coordinate");
            };
            let Some(KernelNode::Parameter(parameter)) = self.kernel.node(id.erase()) else {
                unreachable!("validated Parameter");
            };
            let component = parameter
                .value_type()
                .shape()
                .flat_index(coordinate.component_index())
                .ok_or_else(|| {
                    invalid("Parameter component coordinate differs from its exact shape")
                })?;
            design.push(SpatialDesignCoordinate::ModelParameter {
                parameter: id,
                component,
                imaginary: coordinate.is_imaginary(),
            });
            values.push(coordinate.read_value(parameter.value())?);
        }
        Ok((design, values))
    }

    /// Bind complete real/complex Parameter values without replacing the Model.
    pub(crate) fn at_parameters(
        &self,
        selected: &[Id<kinds::Parameter>],
        values: &[f64],
    ) -> Result<Self, Diagnostic> {
        let mut point = self.clone();
        point.parameter_candidates =
            crate::parameter_point::ParameterPoint::bind(&self.kernel, selected, values)?;
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
                .get(id)
                .unwrap_or(parameter.value());
            *binding = coordinate.read_value(value)?;
        }
        Ok(point)
    }
}
