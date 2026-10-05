//! Compose existing scalar products using exact source-coordinate correspondence.
use super::*;
use crate::{
    DifferentiationRole, LinearizedRelation, RelationCotangent, RelationTangent,
    ScalarLinearization,
};

#[derive(Debug)]
struct Row<'a> {
    action: ScalarLinearization<'a>,
    unknown: Vec<usize>,
    parameter: Vec<usize>,
}

/// Point-bound real differential of shaped real or complex residuals.
///
/// Coordinates retain their source symbols. The pairing is Euclidean in real
/// coordinates, equivalently Re(sum(conj(a)*b)) for complex components.
/// This adapter composes the scalar derivative graph; it assumes no holomorphy.
#[derive(Debug)]
pub struct ComponentLinearization<'a> {
    rows: Vec<Row<'a>>,
    unknown: Vec<ScalarSymbolCoordinate>,
    parameter: Vec<ScalarSymbolCoordinate>,
}

impl ComponentScalarization {
    /// Bind one point and differentiation role per exact source coordinate.
    /// The resolver is called once per distinct coordinate, including frozen inputs.
    /// Active coordinate order is first appearance in the retained residual rows.
    ///
    /// # Errors
    /// Rejects missing/nonfinite inputs and unsupported derivative operations.
    pub fn linearize(
        &self,
        mut resolve: impl FnMut(&ScalarSymbolCoordinate) -> Option<(f64, DifferentiationRole)>,
    ) -> Result<ComponentLinearization<'_>, Diagnostic> {
        let mut bound = HashMap::new();
        let mut unknown = Vec::new();
        let mut parameter = Vec::new();
        let mut rows = Vec::new();
        for row in &self.rows {
            let mut inputs = Vec::new();
            let mut roles = Vec::new();
            let mut unknown_map = Vec::new();
            let mut parameter_map = Vec::new();
            for coordinate in row.symbols() {
                let (value, role, index) = match bound.get(coordinate) {
                    Some(value) => *value,
                    None => {
                        let (value, role) = resolve(coordinate).ok_or_else(|| {
                            invalid_component_ir("missing component linearization input")
                        })?;
                        let index = match role {
                            DifferentiationRole::Unknown => {
                                unknown.push(coordinate.clone());
                                unknown.len() - 1
                            }
                            DifferentiationRole::Parameter => {
                                parameter.push(coordinate.clone());
                                parameter.len() - 1
                            }
                            DifferentiationRole::Frozen => 0,
                        };
                        bound.insert(coordinate.clone(), (value, role, index));
                        (value, role, index)
                    }
                };
                inputs.push(value);
                roles.push(role);
                match role {
                    DifferentiationRole::Unknown => unknown_map.push(index),
                    DifferentiationRole::Parameter => parameter_map.push(index),
                    DifferentiationRole::Frozen => {}
                }
            }
            rows.push(Row {
                action: row.linearize(&inputs, &roles)?,
                unknown: unknown_map,
                parameter: parameter_map,
            });
        }
        Ok(ComponentLinearization {
            rows,
            unknown,
            parameter,
        })
    }
}

impl ComponentLinearization<'_> {
    /// Exact coordinates of the implicit-state tangent and cotangent.
    #[must_use]
    pub fn unknown_coordinates(&self) -> &[ScalarSymbolCoordinate] {
        &self.unknown
    }
    /// Exact coordinates of the selected Parameter tangent and cotangent.
    #[must_use]
    pub fn parameter_coordinates(&self) -> &[ScalarSymbolCoordinate] {
        &self.parameter
    }
}

fn check(values: &[f64], len: usize) -> Result<(), Diagnostic> {
    if values.len() != len || values.iter().any(|v| !v.is_finite()) {
        return Err(invalid_component_ir(
            "component derivative input cardinality or finite-value mismatch",
        ));
    }
    Ok(())
}
fn output_len(len: usize, expected: usize) -> Result<(), Diagnostic> {
    if len != expected {
        return Err(invalid_component_ir(
            "component derivative output cardinality mismatch",
        ));
    }
    Ok(())
}

impl LinearizedRelation<f64> for ComponentLinearization<'_> {
    fn unknown_dimension(&self) -> usize {
        self.unknown.len()
    }
    fn parameter_dimension(&self) -> usize {
        self.parameter.len()
    }
    fn residual_dimension(&self) -> usize {
        self.rows.len()
    }
    fn primal(&self, output: &mut [f64]) -> Result<(), Diagnostic> {
        output_len(output.len(), self.rows.len())?;
        for (row, value) in self.rows.iter().zip(output) {
            row.action.primal(std::slice::from_mut(value))?;
        }
        Ok(())
    }
    fn jvp(&self, tangent: RelationTangent<'_, f64>, output: &mut [f64]) -> Result<(), Diagnostic> {
        let (unknown, parameter) = match tangent {
            RelationTangent::Unknown(v) => (Some(v), None),
            RelationTangent::Parameter(v) => (None, Some(v)),
            RelationTangent::Both { unknown, parameter } => (Some(unknown), Some(parameter)),
        };
        if let Some(v) = unknown {
            check(v, self.unknown.len())?;
        }
        if let Some(v) = parameter {
            check(v, self.parameter.len())?;
        }
        output_len(output.len(), self.rows.len())?;
        for (row, value) in self.rows.iter().zip(output) {
            let u = row
                .unknown
                .iter()
                .map(|&i| unknown.map_or(0., |v| v[i]))
                .collect::<Vec<_>>();
            let p = row
                .parameter
                .iter()
                .map(|&i| parameter.map_or(0., |v| v[i]))
                .collect::<Vec<_>>();
            row.action.jvp(
                RelationTangent::Both {
                    unknown: &u,
                    parameter: &p,
                },
                std::slice::from_mut(value),
            )?;
        }
        Ok(())
    }
    fn vjp(&self, weights: &[f64], output: RelationCotangent<'_, f64>) -> Result<(), Diagnostic> {
        check(weights, self.rows.len())?;
        let (mut unknown, mut parameter) = match output {
            RelationCotangent::Unknown(v) => (Some(v), None),
            RelationCotangent::Parameter(v) => (None, Some(v)),
            RelationCotangent::Both { unknown, parameter } => (Some(unknown), Some(parameter)),
        };
        if let Some(v) = &mut unknown {
            output_len(v.len(), self.unknown.len())?;
            v.fill(0.);
        }
        if let Some(v) = &mut parameter {
            output_len(v.len(), self.parameter.len())?;
            v.fill(0.);
        }
        for (row, &weight) in self.rows.iter().zip(weights) {
            let mut u = vec![0.; row.unknown.len()];
            let mut p = vec![0.; row.parameter.len()];
            row.action.vjp(
                &[weight],
                RelationCotangent::Both {
                    unknown: &mut u,
                    parameter: &mut p,
                },
            )?;
            if let Some(v) = &mut unknown {
                for (&i, value) in row.unknown.iter().zip(u) {
                    v[i] += value;
                }
            }
            if let Some(v) = &mut parameter {
                for (&i, value) in row.parameter.iter().zip(p) {
                    v[i] += value;
                }
            }
        }
        if let Some(v) = unknown {
            check(v, self.unknown.len())?;
        }
        if let Some(v) = parameter {
            check(v, self.parameter.len())?;
        }
        Ok(())
    }
}
