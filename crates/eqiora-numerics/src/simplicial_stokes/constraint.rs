use eqiora_assembly::LocalContribution;
use eqiora_core::Diagnostic;
use eqiora_meshing::{AffineGeometryMap, QuadratureRule};

use super::acceptance::require_local_geometry;
use super::{CONSTRAINT_LOCAL_DOF_COUNT, CONSTRAINT_LOCAL_GAUGE, DIMENSION, P1_BASIS_COUNT};
use crate::discrete_space::{DiscreteSpace, SimplexP1Space};
use crate::operator::LocalOperator;

/// Cell-local occurrence of one global zero-integral pressure constraint.
pub(crate) struct MiniPressureMeanConstraintCell;

impl LocalOperator<AffineGeometryMap> for MiniPressureMeanConstraintCell {
    fn evaluate(
        &self,
        geometry: &AffineGeometryMap,
        quadrature: &QuadratureRule,
    ) -> Result<LocalContribution, Diagnostic> {
        let pressure_integrals = integrated_pressure_basis(geometry, quadrature)?;
        let mut matrix = vec![0.0; CONSTRAINT_LOCAL_DOF_COUNT * CONSTRAINT_LOCAL_DOF_COUNT];
        for (pressure, value) in pressure_integrals.into_iter().enumerate() {
            matrix[pressure * CONSTRAINT_LOCAL_DOF_COUNT + CONSTRAINT_LOCAL_GAUGE] = value;
            matrix[CONSTRAINT_LOCAL_GAUGE * CONSTRAINT_LOCAL_DOF_COUNT + pressure] = value;
        }
        LocalContribution::new(
            CONSTRAINT_LOCAL_DOF_COUNT,
            CONSTRAINT_LOCAL_DOF_COUNT,
            matrix,
            vec![0.0; CONSTRAINT_LOCAL_DOF_COUNT],
        )
    }
}

fn integrated_pressure_basis(
    geometry: &AffineGeometryMap,
    quadrature: &QuadratureRule,
) -> Result<[f64; P1_BASIS_COUNT], Diagnostic> {
    require_local_geometry(geometry, quadrature)?;
    let pressure_space = SimplexP1Space::new(DIMENSION)?;
    let mut integrated = [0.0; P1_BASIS_COUNT];
    for point in quadrature.points() {
        let basis = pressure_space.tabulate(&point.coordinates)?;
        let scale = point.weight * geometry.measure_scale();
        for (value, basis_value) in integrated.iter_mut().zip(basis.values()) {
            *value += scale * basis_value;
        }
    }
    Ok(integrated)
}

#[cfg(test)]
mod tests {
    use eqiora_meshing::triangle_duffy_gauss_legendre;

    use super::*;
    use crate::operator::LocalOperator;

    #[test]
    fn pressure_constraint_is_one_independent_symmetric_local_relation() {
        let geometry = AffineGeometryMap::new(
            eqiora_meshing::ReferenceCell::simplex(DIMENSION).unwrap(),
            DIMENSION,
            vec![0.0, 0.0],
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let local = MiniPressureMeanConstraintCell
            .evaluate(&geometry, &triangle_duffy_gauss_legendre(3).unwrap())
            .unwrap();
        assert_eq!(local.rows(), CONSTRAINT_LOCAL_DOF_COUNT);
        assert!(local.rhs().iter().all(|value| *value == 0.0));
        for pressure in 0..P1_BASIS_COUNT {
            assert!(
                (local.entry(pressure, CONSTRAINT_LOCAL_GAUGE).unwrap() - 1.0 / 6.0).abs()
                    < 2.0e-15
            );
            assert_eq!(
                local.entry(pressure, CONSTRAINT_LOCAL_GAUGE),
                local.entry(CONSTRAINT_LOCAL_GAUGE, pressure)
            );
        }
        for row in 0..CONSTRAINT_LOCAL_DOF_COUNT {
            for column in 0..CONSTRAINT_LOCAL_DOF_COUNT {
                if row != CONSTRAINT_LOCAL_GAUGE && column != CONSTRAINT_LOCAL_GAUGE {
                    assert_eq!(local.entry(row, column), Some(0.0));
                }
            }
        }
    }
}
