//! Method-owned scalar integral constraints through the common quadrature owner.

use eqiora_assembly::LocalContribution;
use eqiora_core::Diagnostic;
use eqiora_meshing::{AffineGeometryMap, GeometryMap, QuadratureRule};
use eqiora_realization::Space;

use super::bilinear::Pairing;
use super::region::integration::{IntegralTerm, integrate};

/// Local occurrence of one global zero-integral multiplier. The scalar Field
/// coefficients precede the single multiplier; global ownership and scaling
/// remain with the caller's realized algebraic constraint and DOF map.
pub(crate) fn zero_integral_constraint(
    space: Space,
    geometry: &AffineGeometryMap,
    quadrature: &QuadratureRule,
) -> Result<LocalContribution<f64>, Diagnostic> {
    integrate(
        geometry.reference_cell(),
        &[(space, 1), (Space::cell_constant(), 1)],
        &[
            IntegralTerm {
                row: 0,
                column: 1,
                pairing: Pairing::Value,
                trial_scale: 1.0,
            },
            IntegralTerm {
                row: 1,
                column: 0,
                pairing: Pairing::Value,
                trial_scale: 1.0,
            },
        ],
        geometry,
        quadrature,
        |_, coefficients, _, _| {
            coefficients.fill(1.0);
            Ok(())
        },
    )
}

#[cfg(test)]
mod tests;
