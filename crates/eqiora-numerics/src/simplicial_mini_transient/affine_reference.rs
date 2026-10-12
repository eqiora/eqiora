//! Independent affine projection used only by local-operator tests.
use super::*;
use crate::continuum_kinematics::symmetric_gradient_bilinear_entry;

/// Congruence scales for a dimensionless affine projection.
///
/// Velocity and pressure are trial/test field scales. `power` is the common
/// action-rate normalization. Applying these factors inside each quadrature
/// accumulation preserves the exact algebra selected by the realization; the
/// projection never rescales an already integrated operator.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MiniAffineScales {
    pub(super) velocity: f64,
    pub(super) pressure: f64,
    pub(super) power: f64,
}

impl MiniAffineScales {
    pub(crate) fn new(velocity: f64, pressure: f64, power: f64) -> Result<Self, Diagnostic> {
        if [velocity, pressure, power]
            .into_iter()
            .any(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(invalid(
                "MINI affine projection scales must be finite and positive",
            ));
        }
        Ok(Self {
            velocity,
            pressure,
            power,
        })
    }
}

/// Linear transient MINI/P1 cell projected directly to scaled `(A, b)` form.
///
/// This independent test reference compares the residual/JVP kernel with a
/// directly assembled affine system. Direct assembly
/// retains quadrature addition order and never recovers `b` from `A x - R`.
pub(crate) struct MiniScaledAffineCell<'a, const D: usize> {
    pub(crate) geometry: &'a AffineGeometryMap,
    pub(crate) density: f64,
    pub(crate) viscosity: f64,
    pub(crate) time_step: f64,
    pub(crate) previous_velocity: &'a [[f64; D]],
    pub(crate) scales: MiniAffineScales,
}

/// Dense element-local scaled affine projection in MINI/P1 ordering.
#[derive(Debug)]
pub(crate) struct MiniScaledAffineProjection {
    local_size: usize,
    matrix: Vec<f64>,
    rhs: Vec<f64>,
}

impl MiniScaledAffineProjection {
    pub(crate) fn into_parts(self) -> (usize, Vec<f64>, Vec<f64>) {
        (self.local_size, self.matrix, self.rhs)
    }
}

impl<const D: usize> MiniScaledAffineCell<'_, D> {
    /// Integrate the disabled-transport relation directly as scaled `(A, b)`.
    pub(crate) fn project(
        &self,
        quadrature: &QuadratureRule,
    ) -> Result<MiniScaledAffineProjection, Diagnostic> {
        self.validate(quadrature)?;

        let p1_basis_count = D + 1;
        let velocity_basis_count = D + 2;
        let local_size = velocity_basis_count * D + p1_basis_count;
        let inverse = self.geometry.inverse_jacobian()?;
        let velocity_space = DiscreteSpace::new(
            eqiora_realization::Space::simplex_p1_bubble(),
            eqiora_meshing::ReferenceCell::simplex(D)?,
        )?;
        let pressure_space = DiscreteSpace::new(
            eqiora_realization::Space::continuous_lagrange(std::num::NonZeroU16::MIN),
            eqiora_meshing::ReferenceCell::simplex(D)?,
        )?;
        let mut matrix = vec![0.0; local_size * local_size];
        let mut rhs = vec![0.0; local_size];

        for point in quadrature.points() {
            let velocity = velocity_space.tabulate(&point.coordinates)?;
            let pressure = pressure_space.tabulate(&point.coordinates)?;
            let gradients = (0..velocity_basis_count)
                .map(|basis| {
                    physical_gradient(
                        velocity.gradient(basis).expect("accepted MINI basis index"),
                        &inverse,
                        D,
                    )
                })
                .collect::<Vec<_>>();
            let measure = point.weight * self.geometry.measure_scale();
            let (previous_value, _) =
                evaluate_velocity(self.previous_velocity, velocity.values(), &gradients);

            for row_basis in 0..velocity_basis_count {
                for (row_component, previous_component) in previous_value.iter().enumerate() {
                    let row = local_velocity::<D>(row_basis, row_component);
                    rhs[row] += measure * self.density / self.time_step
                        * velocity.values()[row_basis]
                        * previous_component
                        * self.scales.velocity
                        / self.scales.power;
                    for column_basis in 0..velocity_basis_count {
                        for column_component in 0..D {
                            let column = local_velocity::<D>(column_basis, column_component);
                            let mass = if row_component == column_component {
                                self.density / self.time_step
                                    * velocity.values()[row_basis]
                                    * velocity.values()[column_basis]
                            } else {
                                0.0
                            };
                            matrix[row * local_size + column] += measure
                                * (mass
                                    + self.viscosity
                                        * symmetric_gradient_bilinear_entry(
                                            &gradients[row_basis],
                                            row_component,
                                            &gradients[column_basis],
                                            column_component,
                                        ))
                                * self.scales.velocity
                                * self.scales.velocity
                                / self.scales.power;
                        }
                    }
                    for pressure_basis in 0..p1_basis_count {
                        let pressure_dof = velocity_basis_count * D + pressure_basis;
                        let coupling = -measure
                            * pressure.values()[pressure_basis]
                            * gradients[row_basis][row_component]
                            * self.scales.velocity
                            * self.scales.pressure
                            / self.scales.power;
                        matrix[row * local_size + pressure_dof] += coupling;
                        matrix[pressure_dof * local_size + row] += coupling;
                    }
                }
            }
        }

        if matrix.iter().chain(&rhs).any(|value| !value.is_finite()) {
            return Err(invalid(
                "MINI scaled affine projection produced a non-finite coefficient",
            ));
        }
        Ok(MiniScaledAffineProjection {
            local_size,
            matrix,
            rhs,
        })
    }

    fn validate(&self, quadrature: &QuadratureRule) -> Result<(), Diagnostic> {
        if !matches!(D, 2 | 3) {
            return Err(invalid(
                "MINI scaled affine projection admits dimensions two and three",
            ));
        }
        let velocity_basis_count = D + 2;
        if self.previous_velocity.len() != velocity_basis_count {
            return Err(invalid(format!(
                "{D}D MINI affine projection requires {velocity_basis_count} previous velocity coefficients",
            )));
        }
        if !self.density.is_finite()
            || self.density <= 0.0
            || !self.viscosity.is_finite()
            || self.viscosity <= 0.0
            || !self.time_step.is_finite()
            || self.time_step <= 0.0
            || self
                .previous_velocity
                .iter()
                .flatten()
                .any(|value| !value.is_finite())
        {
            return Err(invalid(
                "MINI affine projection requires finite physical data",
            ));
        }
        if self.geometry.reference_cell().dimension() != D
            || self.geometry.physical_dimension() != D
            || quadrature.reference_cell() != self.geometry.reference_cell()
        {
            return Err(invalid(format!(
                "MINI affine projection requires one affine {D}D simplex and matching quadrature",
            )));
        }
        let required_exactness = MiniTransport::<D>::Disabled.required_quadrature_exactness();
        if quadrature.polynomial_exactness().unwrap_or(0) < required_exactness {
            return Err(invalid(format!(
                "{D}D MINI affine projection requires quadrature exactness at least {required_exactness}, received {}",
                quadrature.polynomial_exactness().unwrap_or(0),
            )));
        }
        Ok(())
    }
}
