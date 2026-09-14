//! GCL-compatible differential ALE action on one affine MINI fluid cell.
//!
//! The local relation consumes one sealed moving-geometry action.  Its JVP is
//! evaluated from a state direction and an affine-geometry direction in one
//! pass, including the induced mesh-velocity, inverse-map, measure, and GCL
//! derivatives.  No finite difference participates in the production action.

use eqiora_core::Diagnostic;
use eqiora_meshing::FixedTopologyCellGeometryAction;
use eqiora_meshing::{AffineGeometryLinearization, QuadratureRule};

use crate::simplicial_mini_transient::{
    MiniGeometryDirection, MiniTransientCell, MiniTransientDirection, MiniTransport,
};

#[cfg(test)]
const DIMENSION: usize = 2;
#[cfg(test)]
pub(super) const COMPONENTS: usize = 2;
#[cfg(test)]
pub(super) const P1_BASIS_COUNT: usize = 3;
#[cfg(test)]
pub(super) const VELOCITY_BASIS_COUNT: usize = 4;
#[cfg(test)]
pub(super) const LOCAL_VELOCITY_DOF_COUNT: usize = VELOCITY_BASIS_COUNT * COMPONENTS;
#[cfg(test)]
pub(super) const LOCAL_PRESSURE_OFFSET: usize = LOCAL_VELOCITY_DOF_COUNT;
#[cfg(test)]
pub(super) const CELL_LOCAL_DOF_COUNT: usize = LOCAL_VELOCITY_DOF_COUNT + P1_BASIS_COUNT;

/// Primal data for one current-geometry fluid cell.
pub(super) struct AleMiniFluidCell<'a, const D: usize> {
    pub(super) geometry: &'a FixedTopologyCellGeometryAction<D>,
    pub(super) density: f64,
    pub(super) viscosity: f64,
    pub(super) time_step: f64,
    pub(super) previous_velocity: &'a [[f64; D]],
    pub(super) current_velocity: &'a [[f64; D]],
    pub(super) current_pressure: &'a [f64],
}

#[cfg(test)]
pub(super) type AleMiniFluidCell2d<'a> = AleMiniFluidCell<'a, 2>;

/// One exact direction through state and current absolute geometry.
pub(super) struct AleMiniFluidDirection<'a, const D: usize> {
    pub(super) current_velocity: &'a [[f64; D]],
    pub(super) current_pressure: &'a [f64],
    pub(super) current_geometry: &'a AffineGeometryLinearization,
}

#[cfg(test)]
pub(super) type AleMiniFluidDirection2d<'a> = AleMiniFluidDirection<'a, 2>;

/// Residual and analytic JVP evaluated at the identical primal point.
pub(super) struct AleMiniFluidEvaluation {
    residual: Vec<f64>,
    jvp: Vec<f64>,
}

impl AleMiniFluidEvaluation {
    pub(super) fn residual(&self) -> &[f64] {
        &self.residual
    }

    pub(super) fn jvp(&self) -> &[f64] {
        &self.jvp
    }
}

impl<const D: usize> AleMiniFluidCell<'_, D> {
    /// Evaluate the ALE primal residual without constructing a tangent action.
    pub(super) fn residual(&self, quadrature: &QuadratureRule) -> Result<Vec<f64>, Diagnostic> {
        MiniTransientCell::<D> {
            geometry: self.geometry.current_map(),
            transport: MiniTransport::SkewRelativeGcl(self.geometry),
            density: self.density,
            viscosity: self.viscosity,
            time_step: self.time_step,
            previous_velocity: self.previous_velocity,
            current_velocity: self.current_velocity,
            current_pressure: self.current_pressure,
        }
        .residual(quadrature)
    }

    /// Evaluate the differential ALE residual and its exact directional action.
    pub(super) fn evaluate(
        &self,
        direction: AleMiniFluidDirection<'_, D>,
        quadrature: &QuadratureRule,
    ) -> Result<AleMiniFluidEvaluation, Diagnostic> {
        let (residual, jvp) = MiniTransientCell::<D> {
            geometry: self.geometry.current_map(),
            transport: MiniTransport::SkewRelativeGcl(self.geometry),
            density: self.density,
            viscosity: self.viscosity,
            time_step: self.time_step,
            previous_velocity: self.previous_velocity,
            current_velocity: self.current_velocity,
            current_pressure: self.current_pressure,
        }
        .evaluate(
            MiniTransientDirection {
                current_velocity: direction.current_velocity,
                current_pressure: direction.current_pressure,
                current_geometry: MiniGeometryDirection::Endpoint(direction.current_geometry),
            },
            quadrature,
        )?
        .into_parts();
        Ok(AleMiniFluidEvaluation { residual, jvp })
    }
}

#[cfg(test)]
pub(super) const fn local_velocity(basis: usize, component: usize) -> usize {
    basis * COMPONENTS + component
}

#[cfg(test)]
mod tests {
    use eqiora_meshing::{
        FixedTopologyGeometryAction2d, FixedTopologyGeometryState2d, MeshQualityGate,
        SimplicialMesh, simplex_duffy_gauss_legendre, triangle_duffy_gauss_legendre,
    };
    use eqiora_meshing::{FixedTopologyGeometryAction3d, FixedTopologyGeometryState3d};

    use super::*;
    use crate::discrete_space::{DiscreteSpace, SimplexP1BubbleSpace};
    use crate::simplicial_stokes::SimplicialMiniVelocityField2d;

    const STEP: f64 = 0.2;

    fn reference() -> SimplicialMesh {
        SimplicialMesh::new(
            DIMENSION,
            vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![vec![0, 1, 2]],
            MeshQualityGate::new(0.05).unwrap(),
        )
        .unwrap()
    }

    fn current_coordinates() -> Vec<Vec<f64>> {
        vec![vec![0.02, -0.01], vec![1.06, 0.03], vec![0.04, 0.96]]
    }

    fn coordinate_direction() -> [[f64; DIMENSION]; P1_BASIS_COUNT] {
        [[0.03, -0.02], [-0.01, 0.04], [0.02, 0.01]]
    }

    fn geometry_linearization(
        action: &FixedTopologyGeometryAction2d,
        vertex_direction: &[[f64; DIMENSION]; P1_BASIS_COUNT],
    ) -> AffineGeometryLinearization {
        let origin = vertex_direction[0].to_vec();
        let jacobian = vec![
            vertex_direction[1][0] - vertex_direction[0][0],
            vertex_direction[2][0] - vertex_direction[0][0],
            vertex_direction[1][1] - vertex_direction[0][1],
            vertex_direction[2][1] - vertex_direction[0][1],
        ];
        AffineGeometryLinearization::new(
            action.cell(0).unwrap().current_map().clone(),
            origin,
            jacobian,
        )
        .unwrap()
    }

    fn evaluate_residual(
        current_coordinates: Vec<Vec<f64>>,
        current_velocity: &[[f64; COMPONENTS]; VELOCITY_BASIS_COUNT],
        current_pressure: &[f64; P1_BASIS_COUNT],
    ) -> [f64; CELL_LOCAL_DOF_COUNT] {
        let reference = reference();
        let previous = FixedTopologyGeometryState2d::reference(&reference).unwrap();
        let current = FixedTopologyGeometryState2d::new(&reference, current_coordinates).unwrap();
        let action =
            FixedTopologyGeometryAction2d::new(&reference, &previous, &current, STEP).unwrap();
        let stationary =
            AffineGeometryLinearization::stationary(action.cell(0).unwrap().current_map().clone())
                .unwrap();
        let zero_velocity = [[0.0; COMPONENTS]; VELOCITY_BASIS_COUNT];
        let zero_pressure = [0.0; P1_BASIS_COUNT];
        let previous_velocity = [[0.15, -0.05], [0.10, 0.02], [0.12, -0.03], [0.01, 0.02]];
        AleMiniFluidCell2d {
            geometry: action.cell(0).unwrap(),
            density: 1.3,
            viscosity: 0.08,
            time_step: STEP,
            previous_velocity: &previous_velocity,
            current_velocity,
            current_pressure,
        }
        .evaluate(
            AleMiniFluidDirection2d {
                current_velocity: &zero_velocity,
                current_pressure: &zero_pressure,
                current_geometry: &stationary,
            },
            &triangle_duffy_gauss_legendre(5).unwrap(),
        )
        .unwrap()
        .residual
        .try_into()
        .expect("2D ALE MINI residual owns eleven entries")
    }

    #[test]
    fn primal_only_residual_matches_directional_projection_bits() {
        let reference = reference();
        let previous = FixedTopologyGeometryState2d::reference(&reference).unwrap();
        let current = FixedTopologyGeometryState2d::new(&reference, current_coordinates()).unwrap();
        let action =
            FixedTopologyGeometryAction2d::new(&reference, &previous, &current, STEP).unwrap();
        let geometry_direction = geometry_linearization(&action, &coordinate_direction());
        let previous_velocity = [[0.15, -0.05], [0.10, 0.02], [0.12, -0.03], [0.01, 0.02]];
        let current_velocity = [[0.18, -0.01], [0.11, 0.04], [0.09, -0.02], [0.03, 0.01]];
        let velocity_direction = [[0.02, -0.01], [-0.03, 0.02], [0.01, 0.04], [-0.02, 0.03]];
        let pressure = [0.12, -0.08, 0.03];
        let pressure_direction = [-0.02, 0.04, 0.01];
        let cell = AleMiniFluidCell2d {
            geometry: action.cell(0).unwrap(),
            density: 1.3,
            viscosity: 0.08,
            time_step: STEP,
            previous_velocity: &previous_velocity,
            current_velocity: &current_velocity,
            current_pressure: &pressure,
        };
        let projected = cell
            .evaluate(
                AleMiniFluidDirection2d {
                    current_velocity: &velocity_direction,
                    current_pressure: &pressure_direction,
                    current_geometry: &geometry_direction,
                },
                &triangle_duffy_gauss_legendre(5).unwrap(),
            )
            .unwrap();
        let primal = cell
            .residual(&triangle_duffy_gauss_legendre(5).unwrap())
            .unwrap();
        assert_eq!(
            primal
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            projected
                .residual()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn analytic_state_and_geometry_jvp_matches_centered_reassembly() {
        let reference = reference();
        let previous = FixedTopologyGeometryState2d::reference(&reference).unwrap();
        let current = FixedTopologyGeometryState2d::new(&reference, current_coordinates()).unwrap();
        let action =
            FixedTopologyGeometryAction2d::new(&reference, &previous, &current, STEP).unwrap();
        let geometry_direction = geometry_linearization(&action, &coordinate_direction());
        let previous_velocity = [[0.15, -0.05], [0.10, 0.02], [0.12, -0.03], [0.01, 0.02]];
        let current_velocity = [[0.18, -0.01], [0.11, 0.04], [0.09, -0.02], [0.03, 0.01]];
        let velocity_direction = [[0.02, -0.01], [-0.03, 0.02], [0.01, 0.04], [-0.02, 0.03]];
        let pressure = [0.12, -0.08, 0.03];
        let pressure_direction = [-0.02, 0.04, 0.01];
        let evaluated = AleMiniFluidCell2d {
            geometry: action.cell(0).unwrap(),
            density: 1.3,
            viscosity: 0.08,
            time_step: STEP,
            previous_velocity: &previous_velocity,
            current_velocity: &current_velocity,
            current_pressure: &pressure,
        }
        .evaluate(
            AleMiniFluidDirection2d {
                current_velocity: &velocity_direction,
                current_pressure: &pressure_direction,
                current_geometry: &geometry_direction,
            },
            &triangle_duffy_gauss_legendre(5).unwrap(),
        )
        .unwrap();

        let epsilon = f64::EPSILON.cbrt();
        let perturbed_residual = |sign: f64| {
            let coordinates = current_coordinates()
                .into_iter()
                .zip(coordinate_direction())
                .map(|(coordinate, direction)| {
                    coordinate
                        .into_iter()
                        .zip(direction)
                        .map(|(value, direction)| value + sign * epsilon * direction)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let velocity = std::array::from_fn(|basis| {
                std::array::from_fn(|component| {
                    current_velocity[basis][component]
                        + sign * epsilon * velocity_direction[basis][component]
                })
            });
            let pressure = std::array::from_fn(|basis| {
                pressure[basis] + sign * epsilon * pressure_direction[basis]
            });
            evaluate_residual(coordinates, &velocity, &pressure)
        };
        let plus = perturbed_residual(1.0);
        let minus = perturbed_residual(-1.0);
        let centered = std::array::from_fn::<_, CELL_LOCAL_DOF_COUNT, _>(|row| {
            (plus[row] - minus[row]) / (2.0 * epsilon)
        });
        let error = centered
            .iter()
            .zip(evaluated.jvp())
            .map(|(centered, analytic)| (centered - analytic).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale = evaluated
            .jvp()
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        assert!(error < 2.0e-7 * (1.0 + scale), "{error:e} versus {scale:e}");
        assert!(evaluated.residual().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn moving_action_residual_and_jvp_have_stable_bits() {
        let reference = reference();
        let previous = FixedTopologyGeometryState2d::reference(&reference).unwrap();
        let current = FixedTopologyGeometryState2d::new(&reference, current_coordinates()).unwrap();
        let action =
            FixedTopologyGeometryAction2d::new(&reference, &previous, &current, STEP).unwrap();
        let geometry_direction = geometry_linearization(&action, &coordinate_direction());
        let previous_velocity = [[0.15, -0.05], [0.10, 0.02], [0.12, -0.03], [0.01, 0.02]];
        let current_velocity = [[0.18, -0.01], [0.11, 0.04], [0.09, -0.02], [0.03, 0.01]];
        let velocity_direction = [[0.02, -0.01], [-0.03, 0.02], [0.01, 0.04], [-0.02, 0.03]];
        let pressure = [0.12, -0.08, 0.03];
        let pressure_direction = [-0.02, 0.04, 0.01];
        let evaluated = AleMiniFluidCell2d {
            geometry: action.cell(0).unwrap(),
            density: 1.3,
            viscosity: 0.08,
            time_step: STEP,
            previous_velocity: &previous_velocity,
            current_velocity: &current_velocity,
            current_pressure: &pressure,
        }
        .evaluate(
            AleMiniFluidDirection2d {
                current_velocity: &velocity_direction,
                current_pressure: &pressure_direction,
                current_geometry: &geometry_direction,
            },
            &triangle_duffy_gauss_legendre(5).unwrap(),
        )
        .unwrap();

        let expected_residual = [
            4_585_640_822_926_282_759,
            4_585_782_206_638_293_190,
            4_568_732_025_624_342_933,
            4_580_915_278_835_764_599,
            4_563_960_786_078_317_055,
            4_570_393_933_893_647_435,
            4_576_218_749_639_780_020,
            4_580_781_436_043_901_657,
            4_571_312_628_871_897_137,
            4_581_167_946_008_464_539,
            4_579_619_908_696_549_717,
        ];
        let expected_jvp = [
            13_790_422_335_717_904_274,
            4_582_648_534_297_012_687,
            13_805_673_244_579_928_710,
            4_585_239_383_242_535_742,
            13_794_306_283_591_360_986,
            4_585_699_131_716_682_837,
            13_804_380_131_759_224_201,
            4_590_741_557_917_501_196,
            13_794_003_961_654_994_656,
            13_796_391_950_321_411_587,
            4_572_939_208_961_313_291,
        ];
        let actual_residual = std::array::from_fn::<_, CELL_LOCAL_DOF_COUNT, _>(|row| {
            evaluated.residual()[row].to_bits()
        });
        let actual_jvp =
            std::array::from_fn::<_, CELL_LOCAL_DOF_COUNT, _>(|row| evaluated.jvp()[row].to_bits());
        assert_eq!(actual_residual, expected_residual);
        assert_eq!(actual_jvp, expected_jvp);
    }

    #[test]
    fn stationary_geometry_is_exactly_the_fixed_domain_local_action() {
        let mesh = reference();
        let state = FixedTopologyGeometryState2d::reference(&mesh).unwrap();
        let action = FixedTopologyGeometryAction2d::new(&mesh, &state, &state, STEP).unwrap();
        let stationary =
            AffineGeometryLinearization::stationary(action.cell(0).unwrap().current_map().clone())
                .unwrap();
        let previous_coefficients = [[0.15, -0.05], [0.10, 0.02], [0.12, -0.03], [0.01, 0.02]];
        let current_coefficients = [[0.18, -0.01], [0.11, 0.04], [0.09, -0.02], [0.03, 0.01]];
        let pressure = [0.12, -0.08, 0.03];
        let previous = SimplicialMiniVelocityField2d::new(
            mesh.clone(),
            previous_coefficients[..P1_BASIS_COUNT].to_vec(),
            vec![previous_coefficients[P1_BASIS_COUNT]],
        )
        .unwrap();
        let current = SimplicialMiniVelocityField2d::new(
            mesh.clone(),
            current_coefficients[..P1_BASIS_COUNT].to_vec(),
            vec![current_coefficients[P1_BASIS_COUNT]],
        )
        .unwrap();
        let quadrature = triangle_duffy_gauss_legendre(5).unwrap();
        let vertices = mesh
            .entity_vertices(eqiora_meshing::MeshEntity::new(DIMENSION, 0))
            .unwrap();
        let fixed_geometry = action.cell(0).unwrap().current_map();
        let fixed_quadrature =
            crate::simplicial_mini_transient::MiniFixedGeometryQuadrature::prepare(
                fixed_geometry,
                &quadrature,
            )
            .unwrap();
        let fixed = crate::simplicial_navier_stokes::element::MiniNavierStokesCell {
            cell: 0,
            vertices: &vertices,
            density: 1.3,
            viscosity: 0.08,
            time_step: STEP,
            previous_velocity: &previous,
            candidate_velocity: &current,
            candidate_pressure: &pressure,
            body_force: &|_| Ok([0.0; COMPONENTS]),
        }
        .linearize_prepared(fixed_geometry, &fixed_quadrature)
        .unwrap();
        let zero_velocity = [[0.0; COMPONENTS]; VELOCITY_BASIS_COUNT];
        let zero_pressure = [0.0; P1_BASIS_COUNT];
        let ale = AleMiniFluidCell2d {
            geometry: action.cell(0).unwrap(),
            density: 1.3,
            viscosity: 0.08,
            time_step: STEP,
            previous_velocity: &previous_coefficients,
            current_velocity: &current_coefficients,
            current_pressure: &pressure,
        }
        .evaluate(
            AleMiniFluidDirection2d {
                current_velocity: &zero_velocity,
                current_pressure: &zero_pressure,
                current_geometry: &stationary,
            },
            &quadrature,
        )
        .unwrap();
        for (ale, fixed) in ale.residual().iter().zip(fixed.residual()) {
            assert_eq!(ale.to_bits(), fixed.to_bits());
        }
    }

    #[test]
    fn stationary_tetrahedron_is_exactly_the_fixed_domain_local_action() {
        const D3: usize = 3;
        let mesh = SimplicialMesh::new(
            D3,
            vec![
                vec![0.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
            vec![vec![0, 1, 2, 3]],
            MeshQualityGate::new(0.05).unwrap(),
        )
        .unwrap();
        let state = FixedTopologyGeometryState3d::reference(&mesh).unwrap();
        let action = FixedTopologyGeometryAction3d::new(&mesh, &state, &state, STEP).unwrap();
        let stationary =
            AffineGeometryLinearization::stationary(action.cell(0).unwrap().current_map().clone())
                .unwrap();
        let previous_velocity = [
            [0.15, -0.05, 0.04],
            [0.10, 0.02, -0.01],
            [0.12, -0.03, 0.06],
            [0.08, 0.01, -0.02],
            [0.01, 0.02, -0.03],
        ];
        let current_velocity = [
            [0.18, -0.01, 0.03],
            [0.11, 0.04, -0.02],
            [0.09, -0.02, 0.05],
            [0.07, 0.03, -0.01],
            [0.03, 0.01, 0.02],
        ];
        let pressure = [0.12, -0.08, 0.03, -0.02];
        let zero_velocity = [[0.0; D3]; D3 + 2];
        let zero_pressure = [0.0; D3 + 1];
        let quadrature = simplex_duffy_gauss_legendre(D3, 7).unwrap();

        let fixed = MiniTransientCell::<D3> {
            geometry: action.cell(0).unwrap().current_map(),
            transport: MiniTransport::SkewStationary,
            density: 1.3,
            viscosity: 0.08,
            time_step: STEP,
            previous_velocity: &previous_velocity,
            current_velocity: &current_velocity,
            current_pressure: &pressure,
        }
        .evaluate(
            MiniTransientDirection {
                current_velocity: &zero_velocity,
                current_pressure: &zero_pressure,
                current_geometry: MiniGeometryDirection::Zero,
            },
            &quadrature,
        )
        .unwrap()
        .into_parts()
        .0;
        let ale = AleMiniFluidCell::<D3> {
            geometry: action.cell(0).unwrap(),
            density: 1.3,
            viscosity: 0.08,
            time_step: STEP,
            previous_velocity: &previous_velocity,
            current_velocity: &current_velocity,
            current_pressure: &pressure,
        }
        .evaluate(
            AleMiniFluidDirection {
                current_velocity: &zero_velocity,
                current_pressure: &zero_pressure,
                current_geometry: &stationary,
            },
            &quadrature,
        )
        .unwrap();

        assert_eq!(
            ale.residual()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            fixed
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn geometric_correction_preserves_a_constant_free_stream_for_zero_trace_tests() {
        let mesh = reference();
        let previous_geometry = FixedTopologyGeometryState2d::reference(&mesh).unwrap();
        let current_geometry =
            FixedTopologyGeometryState2d::new(&mesh, current_coordinates()).unwrap();
        let action =
            FixedTopologyGeometryAction2d::new(&mesh, &previous_geometry, &current_geometry, STEP)
                .unwrap();
        let cell_geometry = action.cell(0).unwrap();
        assert!(cell_geometry.current_velocity_divergence().abs() > 1.0e-3);

        let transported = [0.7, -0.4];
        let coefficients = [transported, transported, transported, [0.0; COMPONENTS]];
        let pressure = [0.0; P1_BASIS_COUNT];
        let stationary =
            AffineGeometryLinearization::stationary(cell_geometry.current_map().clone()).unwrap();
        let zero_velocity = [[0.0; COMPONENTS]; VELOCITY_BASIS_COUNT];
        let zero_pressure = [0.0; P1_BASIS_COUNT];
        let quadrature = triangle_duffy_gauss_legendre(5).unwrap();
        let density = 1.3;
        let evaluated = AleMiniFluidCell2d {
            geometry: cell_geometry,
            density,
            viscosity: 0.08,
            time_step: STEP,
            previous_velocity: &coefficients,
            current_velocity: &coefficients,
            current_pressure: &pressure,
        }
        .evaluate(
            AleMiniFluidDirection2d {
                current_velocity: &zero_velocity,
                current_pressure: &zero_pressure,
                current_geometry: &stationary,
            },
            &quadrature,
        )
        .unwrap();

        let bubble_space = SimplexP1BubbleSpace::new(DIMENSION).unwrap();
        let bubble_integral = quadrature
            .points()
            .iter()
            .map(|point| {
                point.weight
                    * cell_geometry.current_map().measure_scale()
                    * bubble_space.tabulate(&point.coordinates).unwrap().values()[P1_BASIS_COUNT]
            })
            .sum::<f64>();
        for (component, transported) in transported.into_iter().enumerate() {
            let row = local_velocity(P1_BASIS_COUNT, component);
            let correction_without_which_the_residual_is_nonzero = 0.5
                * density
                * cell_geometry.current_velocity_divergence()
                * transported
                * bubble_integral;
            assert!(correction_without_which_the_residual_is_nonzero.abs() > 1.0e-4);
            assert!(evaluated.residual()[row].abs() < 1.0e-13);
        }
        assert!(
            evaluated.residual()[LOCAL_PRESSURE_OFFSET..]
                .iter()
                .all(|value| value.abs() < 1.0e-13)
        );
    }
}
