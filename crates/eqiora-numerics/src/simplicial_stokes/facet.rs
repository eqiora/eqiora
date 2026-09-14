use eqiora_assembly::LocalContribution;
use eqiora_core::Diagnostic;
use eqiora_meshing::{AffineGeometryMap, QuadratureRule};

use super::acceptance::require_facet_geometry;
use super::{COMPONENTS, FACET_BASIS_COUNT, FACET_LOCAL_DOF_COUNT};
use crate::discrete_space::{DiscreteSpace, SimplexP1Space};
use crate::operator::LocalOperator;

/// One constant prescribed-traction action on a P1 boundary trace.
pub(crate) struct MiniConstantTractionFacet {
    pub(crate) traction: [f64; COMPONENTS],
}

impl LocalOperator<AffineGeometryMap> for MiniConstantTractionFacet {
    fn evaluate(
        &self,
        geometry: &AffineGeometryMap,
        quadrature: &QuadratureRule,
    ) -> Result<LocalContribution, Diagnostic> {
        let rhs = integrated_traction_action(self.traction, geometry, quadrature)?;
        LocalContribution::new(
            FACET_LOCAL_DOF_COUNT,
            FACET_LOCAL_DOF_COUNT,
            vec![0.0; FACET_LOCAL_DOF_COUNT * FACET_LOCAL_DOF_COUNT],
            rhs,
        )
    }
}

fn integrated_traction_action(
    traction: [f64; COMPONENTS],
    geometry: &AffineGeometryMap,
    quadrature: &QuadratureRule,
) -> Result<Vec<f64>, Diagnostic> {
    require_facet_geometry(geometry, quadrature)?;
    let space = SimplexP1Space::new(1)?;
    let mut action = vec![0.0; FACET_LOCAL_DOF_COUNT];
    for point in quadrature.points() {
        let basis = space.tabulate(&point.coordinates)?;
        let scale = point.weight * geometry.measure_scale();
        for local_basis in 0..FACET_BASIS_COUNT {
            for component in 0..COMPONENTS {
                action[facet_velocity(local_basis, component)] +=
                    scale * basis.values()[local_basis] * traction[component];
            }
        }
    }
    Ok(action)
}

const fn facet_velocity(basis: usize, component: usize) -> usize {
    basis * COMPONENTS + component
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_traction_produces_exact_p1_endpoint_actions() {
        let geometry = AffineGeometryMap::new(
            eqiora_meshing::ReferenceCell::simplex(1).unwrap(),
            2,
            vec![4.0, 0.0],
            vec![0.0, 2.0],
        )
        .unwrap();
        let quadrature = eqiora_meshing::simplex_centroid_rule(1).unwrap();
        let local = MiniConstantTractionFacet {
            traction: [-4.5, 0.0],
        }
        .evaluate(&geometry, &quadrature)
        .unwrap();
        assert_eq!(local.rows(), FACET_LOCAL_DOF_COUNT);
        assert!(local.matrix().iter().all(|value| *value == 0.0));
        assert_eq!(local.rhs(), &[-4.5, 0.0, -4.5, 0.0]);
    }
}
