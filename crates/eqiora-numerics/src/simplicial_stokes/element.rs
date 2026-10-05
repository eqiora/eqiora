use eqiora_assembly::LocalContribution;
use eqiora_core::Diagnostic;
use eqiora_meshing::{AffineGeometryMap, GeometryMap, QuadratureRule};

use super::acceptance::require_local_geometry;
use super::{
    CELL_LOCAL_DOF_COUNT, COMPONENTS, DIMENSION, LOCAL_PRESSURE_OFFSET, P1_BASIS_COUNT,
    SimplicialMiniVelocityField2d, VELOCITY_BASIS_COUNT, invalid,
};
use crate::affine_fem::physical_gradient;
use crate::continuum_kinematics::symmetric_gradient_bilinear_entry;
use crate::discrete_space::{DiscreteSpace, SimplexP1BubbleSpace, SimplexP1Space};
use crate::operator::LocalOperator;
use crate::simplicial_elliptic::SimplicialP1Field;

pub(super) struct MiniStokesCell<'a, F> {
    pub(super) viscosity: f64,
    pub(super) body_force: &'a F,
}

impl<F> LocalOperator<AffineGeometryMap> for MiniStokesCell<'_, F>
where
    F: Fn([f64; DIMENSION]) -> Result<[f64; COMPONENTS], Diagnostic> + Sync,
{
    fn evaluate(
        &self,
        geometry: &AffineGeometryMap,
        quadrature: &QuadratureRule,
    ) -> Result<LocalContribution<f64>, Diagnostic> {
        require_local_geometry(geometry, quadrature)?;
        let inverse = geometry.inverse_jacobian()?;
        let spaces = MiniSpaces::new()?;
        let mut matrix = vec![0.0; CELL_LOCAL_DOF_COUNT * CELL_LOCAL_DOF_COUNT];
        let mut rhs = vec![0.0; CELL_LOCAL_DOF_COUNT];
        for point in quadrature.points() {
            let basis = spaces.tabulate(&point.coordinates)?;
            let gradients = physical_gradients(&basis, &inverse);
            let mut coordinates = [0.0; DIMENSION];
            geometry.map_point(&point.coordinates, &mut coordinates)?;
            let force = (self.body_force)(coordinates)?;
            if force.iter().any(|value| !value.is_finite()) {
                return Err(invalid("MINI Stokes body force is non-finite"));
            }
            let scale = point.weight * geometry.measure_scale();
            for row_basis in 0..VELOCITY_BASIS_COUNT {
                for row_component in 0..COMPONENTS {
                    let row = local_velocity(row_basis, row_component);
                    rhs[row] += scale * force[row_component] * basis.values[row_basis];
                    for column_basis in 0..VELOCITY_BASIS_COUNT {
                        for column_component in 0..COMPONENTS {
                            let column = local_velocity(column_basis, column_component);
                            matrix[row * CELL_LOCAL_DOF_COUNT + column] += scale
                                * self.viscosity
                                * symmetric_gradient_bilinear_entry(
                                    &gradients[row_basis],
                                    row_component,
                                    &gradients[column_basis],
                                    column_component,
                                );
                        }
                    }
                    for pressure_basis in 0..P1_BASIS_COUNT {
                        let pressure = LOCAL_PRESSURE_OFFSET + pressure_basis;
                        let coupling = -scale
                            * basis.pressure_values[pressure_basis]
                            * gradients[row_basis][row_component];
                        matrix[row * CELL_LOCAL_DOF_COUNT + pressure] += coupling;
                        matrix[pressure * CELL_LOCAL_DOF_COUNT + row] += coupling;
                    }
                }
            }
        }
        LocalContribution::new(CELL_LOCAL_DOF_COUNT, CELL_LOCAL_DOF_COUNT, matrix, rhs)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MiniBasis {
    pub(crate) values: [f64; VELOCITY_BASIS_COUNT],
    pub(crate) pressure_values: [f64; P1_BASIS_COUNT],
    reference_gradients: [[f64; DIMENSION]; VELOCITY_BASIS_COUNT],
}

pub(crate) struct MiniSpaces {
    velocity: SimplexP1BubbleSpace,
    pressure: SimplexP1Space,
}

impl MiniSpaces {
    pub(crate) fn new() -> Result<Self, Diagnostic> {
        Ok(Self {
            velocity: SimplexP1BubbleSpace::new(DIMENSION)?,
            pressure: SimplexP1Space::new(DIMENSION)?,
        })
    }

    pub(crate) fn tabulate(&self, reference: &[f64]) -> Result<MiniBasis, Diagnostic> {
        let velocity = self.velocity.tabulate(reference)?;
        let pressure = self.pressure.tabulate(reference)?;
        Ok(MiniBasis {
            values: velocity
                .values()
                .try_into()
                .expect("2D P1-bubble has four basis values"),
            pressure_values: pressure
                .values()
                .try_into()
                .expect("2D P1 has three basis values"),
            reference_gradients: std::array::from_fn(|basis| {
                velocity
                    .gradient(basis)
                    .expect("accepted P1-bubble basis index exists")
                    .try_into()
                    .expect("2D reference gradient has two entries")
            }),
        })
    }
}

pub(crate) fn physical_gradients(
    basis: &MiniBasis,
    inverse: &[f64],
) -> [[f64; DIMENSION]; VELOCITY_BASIS_COUNT] {
    std::array::from_fn(|index| {
        physical_gradient(&basis.reference_gradients[index], inverse, DIMENSION)
            .try_into()
            .expect("2D physical gradient has two entries")
    })
}

fn local_velocity(basis: usize, component: usize) -> usize {
    basis * COMPONENTS + component
}

pub(super) fn evaluate_fields(
    velocity: &SimplicialMiniVelocityField2d,
    pressure: &SimplicialP1Field,
    cell: usize,
    vertices: &[eqiora_meshing::MeshEntity],
    basis: &[f64; VELOCITY_BASIS_COUNT],
    pressure_basis: &[f64; P1_BASIS_COUNT],
    gradients: &[[f64; DIMENSION]; VELOCITY_BASIS_COUNT],
) -> ([f64; COMPONENTS], [[f64; DIMENSION]; COMPONENTS], f64) {
    let mut value = [0.0; COMPONENTS];
    let mut gradient = [[0.0; DIMENSION]; COMPONENTS];
    let mut pressure_value = 0.0;
    for local in 0..P1_BASIS_COUNT {
        let vertex = vertices[local].index();
        for component in 0..COMPONENTS {
            value[component] += basis[local] * velocity.vertex_values[vertex][component];
            for axis in 0..DIMENSION {
                gradient[component][axis] +=
                    gradients[local][axis] * velocity.vertex_values[vertex][component];
            }
        }
        pressure_value += pressure_basis[local] * pressure.vertex_values()[vertex];
    }
    for component in 0..COMPONENTS {
        value[component] += basis[3] * velocity.cell_bubble_values[cell][component];
        for axis in 0..DIMENSION {
            gradient[component][axis] +=
                gradients[3][axis] * velocity.cell_bubble_values[cell][component];
        }
    }
    (value, gradient, pressure_value)
}

#[cfg(test)]
mod tests {
    use super::super::LOCAL_VELOCITY_DOF_COUNT;
    use super::*;

    #[test]
    fn mini_spaces_use_the_shared_p1_and_bubble_contracts() {
        let spaces = MiniSpaces::new().unwrap();
        for (point, expected) in [
            ([0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            ([1.0, 0.0], [0.0, 1.0, 0.0, 0.0]),
            ([0.0, 1.0], [0.0, 0.0, 1.0, 0.0]),
            (
                [1.0 / 3.0, 1.0 / 3.0],
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0],
            ),
        ] {
            let basis = spaces.tabulate(&point).unwrap();
            for (value, expected) in basis.values.iter().zip(expected) {
                assert!((value - expected).abs() < 2.0e-15);
            }
            for (value, expected) in basis
                .pressure_values
                .iter()
                .zip(&expected[..P1_BASIS_COUNT])
            {
                assert!((value - expected).abs() < 2.0e-15);
            }
        }
    }

    #[test]
    fn local_viscous_form_is_the_symmetric_gradient_not_a_vector_laplacian() {
        let geometry = AffineGeometryMap::new(
            eqiora_meshing::ReferenceCell::simplex(DIMENSION).unwrap(),
            DIMENSION,
            vec![0.0, 0.0],
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let quadrature = eqiora_meshing::triangle_duffy_gauss_legendre(3).unwrap();
        let zero_force = |_| Ok([0.0, 0.0]);
        let local = MiniStokesCell {
            viscosity: 2.0,
            body_force: &zero_force,
        }
        .evaluate(&geometry, &quadrature)
        .unwrap();
        assert_eq!(local.rows(), CELL_LOCAL_DOF_COUNT);

        let mut rotation = [0.0; LOCAL_VELOCITY_DOF_COUNT];
        rotation[local_velocity(1, 1)] = 1.0;
        rotation[local_velocity(2, 0)] = -1.0;
        assert!(velocity_energy(&local, &rotation).abs() < 2.0e-14);

        let mut shear = [0.0; LOCAL_VELOCITY_DOF_COUNT];
        shear[local_velocity(2, 0)] = 1.0;
        assert!((velocity_energy(&local, &shear) - 1.0).abs() < 2.0e-14);

        let crossed = local
            .entry(local_velocity(2, 0), local_velocity(1, 1))
            .unwrap();
        assert!((crossed - 1.0).abs() < 2.0e-14);
    }

    fn velocity_energy(
        local: &LocalContribution<f64>,
        coefficients: &[f64; LOCAL_VELOCITY_DOF_COUNT],
    ) -> f64 {
        coefficients
            .iter()
            .enumerate()
            .map(|(row, left)| {
                coefficients
                    .iter()
                    .enumerate()
                    .map(|(column, right)| left * local.entry(row, column).unwrap() * right)
                    .sum::<f64>()
            })
            .sum()
    }
}
