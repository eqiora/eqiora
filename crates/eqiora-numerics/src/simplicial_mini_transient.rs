//! Shared element-local MINI/P1 transient incompressible-fluid relation.
//!
//! This private kernel owns one dimension-parametric weak relation over an
//! affine simplex.  Mesh ownership, global numbering, assembly, nonlinear
//! policy, and FSI coupling remain with their respective realizations.  State
//! storage is slice based because stable Rust cannot yet express the required
//! `D + 1` and `D + 2` array lengths as generic constants.

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_meshing::{
    AffineGeometryLinearization, AffineGeometryMap, FixedTopologyCellGeometryAction, GeometryMap,
    QuadratureRule,
};

use crate::affine_fem::physical_gradient;
use crate::continuum_kinematics::symmetric_gradient_bilinear_entry;
use crate::discrete_space::{DiscreteSpace, SimplexP1BubbleSpace, SimplexP1Space};

/// Transport identity carried by the transient fluid relation.
///
/// `Disabled` is the linear transient relation and contains no convective or
/// ALE datum to inspect or accidentally apply. `SkewStationary` is the
/// fixed-domain nonlinear Navier--Stokes specialization and therefore carries
/// no fictitious mesh-motion history. `SkewRelativeGcl` carries the single
/// sealed geometry action from which relative transport and metric correction
/// are derived.
#[derive(Debug, Clone, Copy)]
pub(crate) enum MiniTransport<'a, const D: usize> {
    Disabled,
    SkewStationary,
    SkewRelativeGcl(&'a FixedTopologyCellGeometryAction<D>),
}

impl<const D: usize> MiniTransport<'_, D> {
    pub(crate) const fn required_quadrature_exactness(self) -> usize {
        match self {
            Self::Disabled => 2 * (D + 1),
            Self::SkewStationary | Self::SkewRelativeGcl(_) => 3 * D + 2,
        }
    }

    fn at_primal_point<'a>(
        self,
        reference: &[f64],
        velocity: &'a [f64; D],
        velocity_gradient: &'a [[f64; D]; D],
    ) -> Result<PrimalConvectionPoint<'a, D>, Diagnostic> {
        match self {
            Self::Disabled => Ok(PrimalConvectionPoint::Disabled),
            Self::SkewStationary => Ok(PrimalConvectionPoint::Stationary {
                velocity,
                velocity_gradient,
            }),
            Self::SkewRelativeGcl(action) => {
                let mesh_velocity = action.mesh_velocity(reference)?;
                Ok(PrimalConvectionPoint::Ale {
                    relative_velocity: std::array::from_fn(|axis| {
                        velocity[axis] - mesh_velocity[axis]
                    }),
                    velocity,
                    velocity_gradient,
                    mesh_divergence: action.current_velocity_divergence(),
                })
            }
        }
    }
}

/// Geometry direction at the current affine endpoint.
#[derive(Debug, Clone, Copy)]
pub(crate) enum MiniGeometryDirection<'a> {
    #[cfg(test)]
    Zero,
    Endpoint(&'a AffineGeometryLinearization),
}

/// Primal coefficients for one affine MINI/P1 fluid cell.
pub(crate) struct MiniTransientCell<'a, const D: usize> {
    pub(crate) geometry: &'a AffineGeometryMap,
    pub(crate) transport: MiniTransport<'a, D>,
    pub(crate) density: f64,
    pub(crate) viscosity: f64,
    pub(crate) time_step: f64,
    pub(crate) previous_velocity: &'a [[f64; D]],
    pub(crate) current_velocity: &'a [[f64; D]],
    pub(crate) current_pressure: &'a [f64],
}

/// Exact direction through current state and affine geometry.
pub(crate) struct MiniTransientDirection<'a, const D: usize> {
    pub(crate) current_velocity: &'a [[f64; D]],
    pub(crate) current_pressure: &'a [f64],
    pub(crate) current_geometry: MiniGeometryDirection<'a>,
}

/// Residual and analytic JVP evaluated at one identical primal point.
#[derive(Debug)]
pub(crate) struct MiniTransientEvaluation {
    residual: Vec<f64>,
    jvp: Vec<f64>,
}

impl MiniTransientEvaluation {
    pub(crate) fn into_parts(self) -> (Vec<f64>, Vec<f64>) {
        (self.residual, self.jvp)
    }
}

/// Dense state linearization with geometry and spatial source held fixed.
///
/// The projection differentiates only current MINI velocity and pressure. Its
/// body-force callback is sampled once per physical quadrature point for the
/// primal residual and is deliberately not interpreted as a geometry-tangent
/// callback. A moving-geometry source requires a future explicit source JVP
/// contract rather than an implicit derivative through this projection.
#[derive(Debug)]
pub(crate) struct MiniFixedGeometryStateLinearization {
    jacobian: Vec<f64>,
    residual: Vec<f64>,
}

/// Geometry- and quadrature-owned data for repeated state projections on one
/// immutable affine cell.
#[derive(Debug, Clone)]
pub(crate) struct MiniFixedGeometryQuadrature<const D: usize> {
    geometry: AffineGeometryMap,
    points: Vec<MiniFixedGeometryQuadraturePoint<D>>,
    velocity_basis: Vec<f64>,
    pressure_basis: Vec<f64>,
    gradients: Vec<f64>,
}

#[derive(Debug, Clone)]
struct MiniFixedGeometryQuadraturePoint<const D: usize> {
    physical_coordinates: [f64; D],
    measure: f64,
}

impl<const D: usize> MiniFixedGeometryQuadrature<D> {
    pub(crate) fn prepare(
        geometry: &AffineGeometryMap,
        quadrature: &QuadratureRule,
    ) -> Result<Self, Diagnostic> {
        if !matches!(D, 2 | 3)
            || geometry.reference_cell().dimension() != D
            || geometry.physical_dimension() != D
            || quadrature.reference_cell() != geometry.reference_cell()
        {
            return Err(invalid(format!(
                "fixed-geometry MINI quadrature requires one affine {D}D simplex and matching quadrature",
            )));
        }
        let required_exactness = MiniTransport::<D>::SkewStationary.required_quadrature_exactness();
        if quadrature.polynomial_exactness().unwrap_or(0) < required_exactness {
            return Err(invalid(format!(
                "{D}D MINI transient fluid transport requires quadrature exactness at least {required_exactness}, received {}",
                quadrature.polynomial_exactness().unwrap_or(0),
            )));
        }
        let inverse = geometry.inverse_jacobian()?;
        let velocity_space = SimplexP1BubbleSpace::new(D)?;
        let pressure_space = SimplexP1Space::new(D)?;
        let mut points = Vec::with_capacity(quadrature.points().len());
        let mut velocity_basis = Vec::with_capacity(quadrature.points().len() * (D + 2));
        let mut pressure_basis = Vec::with_capacity(quadrature.points().len() * (D + 1));
        let mut gradients = Vec::with_capacity(quadrature.points().len() * (D + 2) * D);
        for point in quadrature.points() {
            let velocity = velocity_space.tabulate(&point.coordinates)?;
            let pressure = pressure_space.tabulate(&point.coordinates)?;
            velocity_basis.extend_from_slice(velocity.values());
            pressure_basis.extend_from_slice(pressure.values());
            for basis in 0..D + 2 {
                gradients.extend(physical_gradient(
                    velocity.gradient(basis).expect("accepted MINI basis index"),
                    &inverse,
                    D,
                ));
            }
            let mut physical_coordinates = [0.0; D];
            geometry.map_point(&point.coordinates, &mut physical_coordinates)?;
            points.push(MiniFixedGeometryQuadraturePoint {
                physical_coordinates,
                measure: point.weight * geometry.measure_scale(),
            });
        }
        Ok(Self {
            geometry: geometry.clone(),
            points,
            velocity_basis,
            pressure_basis,
            gradients,
        })
    }
}

impl MiniFixedGeometryStateLinearization {
    pub(crate) fn into_parts(self) -> (Vec<f64>, Vec<f64>) {
        (self.jacobian, self.residual)
    }
}

/// Congruence scales for a dimensionless affine projection.
///
/// Velocity and pressure are trial/test field scales. `power` is the common
/// action-rate normalization. Applying these factors inside each quadrature
/// accumulation preserves the exact algebra selected by the realization; the
/// projection never rescales an already integrated operator.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MiniAffineScales {
    velocity: f64,
    pressure: f64,
    power: f64,
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
/// This projection exists alongside the residual/JVP projection because
/// fixed-reference monolithic assembly owns an affine solve. Direct assembly
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
        let velocity_space = SimplexP1BubbleSpace::new(D)?;
        let pressure_space = SimplexP1Space::new(D)?;
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

impl<const D: usize> MiniTransientCell<'_, D> {
    pub(crate) fn residual_prepared_fixed_geometry_state<F>(
        &self,
        body_force: &F,
        prepared: &MiniFixedGeometryQuadrature<D>,
    ) -> Result<Vec<f64>, Diagnostic>
    where
        F: Fn([f64; D]) -> Result<[f64; D], Diagnostic> + Sync,
    {
        self.project_prepared_fixed_geometry_state(body_force, prepared, None)
    }

    /// Evaluate only the primal transient relation for stationary or ALE transport.
    pub(crate) fn residual(&self, quadrature: &QuadratureRule) -> Result<Vec<f64>, Diagnostic> {
        self.validate_primal(quadrature)?;

        let p1_basis_count = D + 1;
        let velocity_basis_count = D + 2;
        let pressure_offset = velocity_basis_count * D;
        let local_dof_count = pressure_offset + p1_basis_count;
        let inverse = self.geometry.inverse_jacobian()?;
        let velocity_space = SimplexP1BubbleSpace::new(D)?;
        let pressure_space = SimplexP1Space::new(D)?;
        let mut residual = vec![0.0; local_dof_count];

        for point in quadrature.points() {
            let velocity_basis = velocity_space.tabulate(&point.coordinates)?;
            let pressure_basis = pressure_space.tabulate(&point.coordinates)?;
            let gradients = (0..velocity_basis_count)
                .map(|basis| {
                    physical_gradient(
                        velocity_basis
                            .gradient(basis)
                            .expect("accepted MINI basis index"),
                        &inverse,
                        D,
                    )
                })
                .collect::<Vec<_>>();
            let (velocity, velocity_gradient) =
                evaluate_velocity(self.current_velocity, velocity_basis.values(), &gradients);
            let (previous_velocity, _) =
                evaluate_velocity(self.previous_velocity, velocity_basis.values(), &gradients);
            let pressure = dot(self.current_pressure, pressure_basis.values());
            let measure = point.weight * self.geometry.measure_scale();
            let primal = MiniPrimalPoint {
                density: self.density,
                viscosity: self.viscosity,
                time_step: self.time_step,
                velocity: &velocity,
                previous_velocity: &previous_velocity,
                velocity_gradient: &velocity_gradient,
                pressure,
            };

            for pressure_test in 0..p1_basis_count {
                let row = pressure_offset + pressure_test;
                residual[row] +=
                    measure * primal.continuity_action(pressure_basis.values()[pressure_test]);
            }

            let convection = self.transport.at_primal_point(
                &point.coordinates,
                &velocity,
                &velocity_gradient,
            )?;
            for (row_basis, test_gradient) in gradients.iter().enumerate() {
                let test = velocity_basis.values()[row_basis];
                for row_component in 0..D {
                    let row = local_velocity::<D>(row_basis, row_component);
                    let convective =
                        convection.action(self.density, test, test_gradient, row_component);
                    residual[row] += measure
                        * primal.momentum_action(test, test_gradient, row_component, convective);
                }
            }
        }

        if residual.iter().any(|value| !value.is_finite()) {
            return Err(invalid("MINI transient fluid residual is non-finite"));
        }
        Ok(residual)
    }

    /// Project fixed-domain skew transport to its dense state Jacobian using
    /// the exact affine map and quadrature data prepared by the owning run.
    pub(crate) fn linearize_prepared_fixed_geometry_state<F>(
        &self,
        body_force: &F,
        prepared: &MiniFixedGeometryQuadrature<D>,
    ) -> Result<MiniFixedGeometryStateLinearization, Diagnostic>
    where
        F: Fn([f64; D]) -> Result<[f64; D], Diagnostic> + Sync,
    {
        let local_dof_count = (D + 2) * D + D + 1;
        let mut jacobian = vec![0.0; local_dof_count * local_dof_count];
        let residual =
            self.project_prepared_fixed_geometry_state(body_force, prepared, Some(&mut jacobian))?;
        Ok(MiniFixedGeometryStateLinearization { jacobian, residual })
    }

    fn project_prepared_fixed_geometry_state<F>(
        &self,
        body_force: &F,
        prepared: &MiniFixedGeometryQuadrature<D>,
        mut jacobian: Option<&mut [f64]>,
    ) -> Result<Vec<f64>, Diagnostic>
    where
        F: Fn([f64; D]) -> Result<[f64; D], Diagnostic> + Sync,
    {
        if !matches!(self.transport, MiniTransport::SkewStationary) {
            return Err(invalid(
                "fixed-geometry MINI state projection requires stationary skew transport",
            ));
        }
        self.validate_primal_state()?;
        if prepared.geometry != *self.geometry {
            return Err(invalid(
                "fixed-geometry MINI state projection requires the exact prepared affine map",
            ));
        }

        let p1_basis_count = D + 1;
        let velocity_basis_count = D + 2;
        let pressure_offset = velocity_basis_count * D;
        let local_dof_count = pressure_offset + p1_basis_count;
        if jacobian
            .as_ref()
            .is_some_and(|entries| entries.len() != local_dof_count * local_dof_count)
        {
            return Err(invalid(
                "fixed-geometry MINI state Jacobian sink has the wrong shape",
            ));
        }
        let mut residual = vec![0.0; local_dof_count];

        for (point_index, point) in prepared.points.iter().enumerate() {
            let velocity_start = point_index * velocity_basis_count;
            let pressure_start = point_index * p1_basis_count;
            let gradient_start = point_index * velocity_basis_count * D;
            let velocity_basis =
                &prepared.velocity_basis[velocity_start..velocity_start + velocity_basis_count];
            let pressure_basis =
                &prepared.pressure_basis[pressure_start..pressure_start + p1_basis_count];
            let gradients =
                &prepared.gradients[gradient_start..gradient_start + velocity_basis_count * D];
            let (velocity, velocity_gradient) =
                evaluate_velocity_flat(self.current_velocity, velocity_basis, gradients);
            let (previous_velocity, _) =
                evaluate_velocity_flat(self.previous_velocity, velocity_basis, gradients);
            let pressure = self
                .current_pressure
                .iter()
                .zip(pressure_basis)
                .map(|(coefficient, basis)| coefficient * basis)
                .sum::<f64>();
            let force = body_force(point.physical_coordinates)?;
            if force.iter().any(|value| !value.is_finite()) {
                return Err(invalid("MINI Navier--Stokes body force is non-finite"));
            }
            let scale = point.measure;

            let divergence = (0..D)
                .map(|axis| velocity_gradient[axis][axis])
                .sum::<f64>();
            for pressure_test in 0..p1_basis_count {
                residual[pressure_offset + pressure_test] -=
                    scale * pressure_basis[pressure_test] * divergence;
            }

            for row_basis in 0..velocity_basis_count {
                let row_gradient = &gradients[row_basis * D..(row_basis + 1) * D];
                let velocity_dot_row_gradient = dot(&velocity, row_gradient);
                for row_component in 0..D {
                    let row = local_velocity::<D>(row_basis, row_component);
                    let test = velocity_basis[row_basis];
                    let time_residual = self.density / self.time_step
                        * test
                        * (velocity[row_component] - previous_velocity[row_component]);
                    let viscous_residual = self.viscosity
                        * projected_symmetric_gradient_test(
                            &velocity_gradient,
                            row_gradient,
                            row_component,
                        );
                    let pressure_residual = -pressure * row_gradient[row_component];
                    let convective_residual = 0.5
                        * self.density
                        * (dot(&velocity, &velocity_gradient[row_component]) * test
                            - velocity_dot_row_gradient * velocity[row_component]);
                    residual[row] += scale
                        * (time_residual
                            + viscous_residual
                            + pressure_residual
                            + convective_residual
                            - force[row_component] * test);

                    if let Some(jacobian) = jacobian.as_deref_mut() {
                        for column_basis in 0..velocity_basis_count {
                            for column_component in 0..D {
                                let column = local_velocity::<D>(column_basis, column_component);
                                let trial = velocity_basis[column_basis];
                                let mass = if row_component == column_component {
                                    self.density / self.time_step * test * trial
                                } else {
                                    0.0
                                };
                                let viscous = self.viscosity
                                    * symmetric_gradient_bilinear_entry(
                                        row_gradient,
                                        row_component,
                                        &gradients[column_basis * D..(column_basis + 1) * D],
                                        column_component,
                                    );
                                let convective = ProjectedConvectiveLinearization {
                                    density: self.density,
                                    velocity: &velocity,
                                    velocity_gradient: &velocity_gradient,
                                    basis: velocity_basis,
                                    gradients,
                                }
                                .entry(
                                    row_basis,
                                    row_component,
                                    column_basis,
                                    column_component,
                                );
                                jacobian[row * local_dof_count + column] +=
                                    scale * (mass + viscous + convective);
                            }
                        }
                        for (pressure_basis_index, pressure_basis_value) in
                            pressure_basis.iter().copied().enumerate()
                        {
                            let column = pressure_offset + pressure_basis_index;
                            let coupling =
                                -scale * pressure_basis_value * row_gradient[row_component];
                            jacobian[row * local_dof_count + column] += coupling;
                            jacobian[column * local_dof_count + row] += coupling;
                        }
                    }
                }
            }
        }

        if residual.iter().any(|value| !value.is_finite())
            || jacobian
                .as_deref()
                .is_some_and(|entries| entries.iter().any(|value| !value.is_finite()))
        {
            return Err(invalid(
                "fixed-geometry MINI state projection produced a non-finite value",
            ));
        }
        Ok(residual)
    }

    /// Evaluate the local weak residual and exact directional action.
    pub(crate) fn evaluate(
        &self,
        direction: MiniTransientDirection<'_, D>,
        quadrature: &QuadratureRule,
    ) -> Result<MiniTransientEvaluation, Diagnostic> {
        self.validate(&direction, quadrature)?;

        let p1_basis_count = D + 1;
        let velocity_basis_count = D + 2;
        let pressure_offset = velocity_basis_count * D;
        let local_dof_count = pressure_offset + p1_basis_count;
        let inverse = self.geometry.inverse_jacobian()?;
        let geometry_tangent = GeometryTangent::new(self.geometry, direction.current_geometry)?;
        let velocity_space = SimplexP1BubbleSpace::new(D)?;
        let pressure_space = SimplexP1Space::new(D)?;
        let transport_tangent =
            TransportTangent::new(self.transport, &inverse, &geometry_tangent, self.time_step)?;
        let mut residual = vec![0.0; local_dof_count];
        let mut jvp = vec![0.0; local_dof_count];

        for point in quadrature.points() {
            let velocity_basis = velocity_space.tabulate(&point.coordinates)?;
            let pressure_basis = pressure_space.tabulate(&point.coordinates)?;
            let gradients = (0..velocity_basis_count)
                .map(|basis| {
                    physical_gradient(
                        velocity_basis
                            .gradient(basis)
                            .expect("accepted MINI basis index"),
                        &inverse,
                        D,
                    )
                })
                .collect::<Vec<_>>();
            let gradient_tangents = (0..velocity_basis_count)
                .map(|basis| {
                    physical_gradient(
                        velocity_basis
                            .gradient(basis)
                            .expect("accepted MINI basis index"),
                        &geometry_tangent.inverse_jacobian,
                        D,
                    )
                })
                .collect::<Vec<_>>();
            let (velocity, velocity_gradient) =
                evaluate_velocity(self.current_velocity, velocity_basis.values(), &gradients);
            let (velocity_tangent, velocity_gradient_tangent) = evaluate_velocity_tangent(
                self.current_velocity,
                direction.current_velocity,
                velocity_basis.values(),
                &gradients,
                &gradient_tangents,
            );
            let (previous_velocity, _) =
                evaluate_velocity(self.previous_velocity, velocity_basis.values(), &gradients);
            let pressure = dot(self.current_pressure, pressure_basis.values());
            let pressure_tangent = dot(direction.current_pressure, pressure_basis.values());
            let divergence_tangent = trace(&velocity_gradient_tangent);
            let measure = point.weight * self.geometry.measure_scale();
            let measure_tangent = point.weight * geometry_tangent.measure_scale;
            let primal = MiniPrimalPoint {
                density: self.density,
                viscosity: self.viscosity,
                time_step: self.time_step,
                velocity: &velocity,
                previous_velocity: &previous_velocity,
                velocity_gradient: &velocity_gradient,
                pressure,
            };

            for pressure_test in 0..p1_basis_count {
                let row = pressure_offset + pressure_test;
                let integrand = primal.continuity_action(pressure_basis.values()[pressure_test]);
                let integrand_tangent =
                    -pressure_basis.values()[pressure_test] * divergence_tangent;
                accumulate(
                    &mut residual[row],
                    &mut jvp[row],
                    measure,
                    measure_tangent,
                    integrand,
                    integrand_tangent,
                );
            }

            let point_state = PointState {
                velocity: &velocity,
                velocity_tangent: &velocity_tangent,
                velocity_gradient: &velocity_gradient,
                velocity_gradient_tangent: &velocity_gradient_tangent,
            };
            let convection = transport_tangent.at_point(
                &point.coordinates,
                point_state,
                &geometry_tangent,
                self.time_step,
            )?;
            for row_basis in 0..velocity_basis_count {
                let test = velocity_basis.values()[row_basis];
                let test_gradient = &gradients[row_basis];
                let test_gradient_tangent = &gradient_tangents[row_basis];

                for row_component in 0..D {
                    let row = local_velocity::<D>(row_basis, row_component);
                    let time_tangent =
                        self.density / self.time_step * test * velocity_tangent[row_component];
                    let viscous_tangent = self.viscosity
                        * symmetric_gradient_test_tangent(
                            &velocity_gradient,
                            &velocity_gradient_tangent,
                            test_gradient,
                            test_gradient_tangent,
                            row_component,
                        );
                    let pressure_action_tangent = -pressure_tangent * test_gradient[row_component]
                        - pressure * test_gradient_tangent[row_component];
                    let (convective, convective_tangent) = convection.action(
                        self.density,
                        test,
                        test_gradient,
                        test_gradient_tangent,
                        row_component,
                    );
                    accumulate(
                        &mut residual[row],
                        &mut jvp[row],
                        measure,
                        measure_tangent,
                        primal.momentum_action(test, test_gradient, row_component, convective),
                        time_tangent
                            + viscous_tangent
                            + pressure_action_tangent
                            + convective_tangent,
                    );
                }
            }
        }

        if residual.iter().chain(&jvp).any(|value| !value.is_finite()) {
            return Err(invalid(
                "MINI transient fluid residual or analytic JVP is non-finite",
            ));
        }
        Ok(MiniTransientEvaluation { residual, jvp })
    }

    fn validate(
        &self,
        direction: &MiniTransientDirection<'_, D>,
        quadrature: &QuadratureRule,
    ) -> Result<(), Diagnostic> {
        self.validate_primal(quadrature)?;
        let p1_basis_count = D + 1;
        let velocity_basis_count = D + 2;
        if direction.current_velocity.len() != velocity_basis_count
            || direction.current_pressure.len() != p1_basis_count
        {
            return Err(invalid(format!(
                "{D}D MINI transient fluid direction requires {velocity_basis_count} velocity and {p1_basis_count} pressure coefficients",
            )));
        }
        if direction
            .current_velocity
            .iter()
            .flatten()
            .chain(direction.current_pressure)
            .any(|value| !value.is_finite())
        {
            return Err(invalid(
                "MINI transient fluid relation requires finite direction data",
            ));
        }
        if matches!(
            direction.current_geometry,
            MiniGeometryDirection::Endpoint(linearization) if linearization.map() != self.geometry
        ) {
            return Err(invalid(
                "MINI geometry direction must linearize the exact current affine map",
            ));
        }
        Ok(())
    }

    fn validate_primal(&self, quadrature: &QuadratureRule) -> Result<(), Diagnostic> {
        self.validate_primal_state()?;
        if self.geometry.reference_cell().dimension() != D
            || self.geometry.physical_dimension() != D
            || quadrature.reference_cell() != self.geometry.reference_cell()
        {
            return Err(invalid(format!(
                "MINI transient fluid relation requires one affine {D}D simplex and matching quadrature",
            )));
        }
        if matches!(
            self.transport,
            MiniTransport::SkewRelativeGcl(action) if action.current_map() != self.geometry
        ) {
            return Err(invalid(
                "ALE MINI transport requires the exact current geometry carried by its sealed action",
            ));
        }
        let required_exactness = self.transport.required_quadrature_exactness();
        if quadrature.polynomial_exactness().unwrap_or(0) < required_exactness {
            return Err(invalid(format!(
                "{D}D MINI transient fluid transport requires quadrature exactness at least {required_exactness}, received {}",
                quadrature.polynomial_exactness().unwrap_or(0),
            )));
        }
        Ok(())
    }

    fn validate_primal_state(&self) -> Result<(), Diagnostic> {
        if !matches!(D, 2 | 3) {
            return Err(invalid(
                "MINI transient fluid relation admits dimensions two and three",
            ));
        }
        let p1_basis_count = D + 1;
        let velocity_basis_count = D + 2;
        if self.previous_velocity.len() != velocity_basis_count
            || self.current_velocity.len() != velocity_basis_count
            || self.current_pressure.len() != p1_basis_count
        {
            return Err(invalid(format!(
                "{D}D MINI transient fluid state requires {velocity_basis_count} velocity and {p1_basis_count} pressure coefficients",
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
                .chain(self.current_velocity)
                .flatten()
                .chain(self.current_pressure)
                .any(|value| !value.is_finite())
        {
            return Err(invalid(
                "MINI transient fluid relation requires finite physical state data",
            ));
        }
        Ok(())
    }
}

struct MiniPrimalPoint<'a, const D: usize> {
    density: f64,
    viscosity: f64,
    time_step: f64,
    velocity: &'a [f64; D],
    previous_velocity: &'a [f64; D],
    velocity_gradient: &'a [[f64; D]; D],
    pressure: f64,
}

impl<const D: usize> MiniPrimalPoint<'_, D> {
    fn continuity_action(&self, pressure_test: f64) -> f64 {
        -pressure_test * trace(self.velocity_gradient)
    }

    fn momentum_action(
        &self,
        test: f64,
        test_gradient: &[f64],
        component: usize,
        convection: f64,
    ) -> f64 {
        let time = self.density / self.time_step
            * test
            * (self.velocity[component] - self.previous_velocity[component]);
        let viscous = self.viscosity
            * symmetric_gradient_test(self.velocity_gradient, test_gradient, component);
        let pressure = -self.pressure * test_gradient[component];
        time + viscous + pressure + convection
    }
}

struct GeometryTangent {
    inverse_jacobian: Vec<f64>,
    origin: Vec<f64>,
    jacobian: Vec<f64>,
    measure_scale: f64,
}

impl GeometryTangent {
    fn new(
        _geometry: &AffineGeometryMap,
        direction: MiniGeometryDirection<'_>,
    ) -> Result<Self, Diagnostic> {
        match direction {
            #[cfg(test)]
            MiniGeometryDirection::Zero => Ok(Self {
                inverse_jacobian: vec![0.0; _geometry.jacobian().len()],
                origin: vec![0.0; _geometry.physical_dimension()],
                jacobian: vec![0.0; _geometry.jacobian().len()],
                measure_scale: 0.0,
            }),
            MiniGeometryDirection::Endpoint(linearization) => Ok(Self {
                inverse_jacobian: linearization.inverse_jacobian_tangent()?,
                origin: linearization.origin_tangent().to_vec(),
                jacobian: linearization.jacobian_tangent().to_vec(),
                measure_scale: linearization.measure_scale_tangent(),
            }),
        }
    }

    fn map_point<const D: usize>(&self, reference: &[f64]) -> [f64; D] {
        std::array::from_fn(|row| {
            self.origin[row]
                + reference
                    .iter()
                    .enumerate()
                    .map(|(column, coordinate)| self.jacobian[row * D + column] * coordinate)
                    .sum::<f64>()
        })
    }
}

enum PrimalConvectionPoint<'a, const D: usize> {
    Disabled,
    Stationary {
        velocity: &'a [f64; D],
        velocity_gradient: &'a [[f64; D]; D],
    },
    Ale {
        relative_velocity: [f64; D],
        velocity: &'a [f64; D],
        velocity_gradient: &'a [[f64; D]; D],
        mesh_divergence: f64,
    },
}

impl<const D: usize> PrimalConvectionPoint<'_, D> {
    fn action(&self, density: f64, test: f64, test_gradient: &[f64], component: usize) -> f64 {
        match self {
            Self::Disabled => 0.0,
            Self::Stationary {
                velocity,
                velocity_gradient,
            } => stationary_convection_action(
                density,
                velocity,
                velocity_gradient,
                test,
                test_gradient,
                component,
            ),
            Self::Ale {
                relative_velocity,
                velocity,
                velocity_gradient,
                mesh_divergence,
            } => ale_convection_action(
                density,
                relative_velocity,
                velocity,
                velocity_gradient,
                *mesh_divergence,
                test,
                test_gradient,
                component,
            ),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn stationary_convection_action<const D: usize>(
    density: f64,
    velocity: &[f64; D],
    velocity_gradient: &[[f64; D]; D],
    test: f64,
    test_gradient: &[f64],
    component: usize,
) -> f64 {
    0.5 * density
        * (dot(velocity, &velocity_gradient[component]) * test
            - dot(velocity, test_gradient) * velocity[component])
}

#[allow(clippy::too_many_arguments)]
fn ale_convection_action<const D: usize>(
    density: f64,
    relative_velocity: &[f64; D],
    velocity: &[f64; D],
    velocity_gradient: &[[f64; D]; D],
    mesh_divergence: f64,
    test: f64,
    test_gradient: &[f64],
    component: usize,
) -> f64 {
    0.5 * density
        * (dot(relative_velocity, &velocity_gradient[component]) * test
            - dot(relative_velocity, test_gradient) * velocity[component]
            + mesh_divergence * velocity[component] * test)
}

enum TransportTangent<'a, const D: usize> {
    Disabled,
    SkewStationary,
    SkewRelativeGcl {
        action: &'a FixedTopologyCellGeometryAction<D>,
        mesh_divergence_tangent: f64,
    },
}

impl<'a, const D: usize> TransportTangent<'a, D> {
    fn new(
        transport: MiniTransport<'a, D>,
        inverse: &[f64],
        geometry_tangent: &GeometryTangent,
        time_step: f64,
    ) -> Result<Self, Diagnostic> {
        match transport {
            MiniTransport::Disabled => Ok(Self::Disabled),
            MiniTransport::SkewStationary => Ok(Self::SkewStationary),
            MiniTransport::SkewRelativeGcl(action) => {
                let reference_tangent = geometry_tangent
                    .jacobian
                    .iter()
                    .map(|value| value / time_step)
                    .collect::<Vec<_>>();
                let gradient_tangent = multiply_linearization::<D>(
                    &reference_tangent,
                    inverse,
                    action.reference_velocity_gradient(),
                    &geometry_tangent.inverse_jacobian,
                )?;
                Ok(Self::SkewRelativeGcl {
                    action,
                    mesh_divergence_tangent: trace_flat::<D>(&gradient_tangent),
                })
            }
        }
    }

    fn at_point(
        &'a self,
        reference: &[f64],
        state: PointState<'a, D>,
        geometry_tangent: &GeometryTangent,
        time_step: f64,
    ) -> Result<ConvectionPoint<'a, D>, Diagnostic> {
        match self {
            Self::Disabled => Ok(ConvectionPoint::Disabled),
            Self::SkewStationary => Ok(ConvectionPoint::Stationary {
                velocity: state.velocity,
                velocity_tangent: state.velocity_tangent,
                velocity_gradient: state.velocity_gradient,
                velocity_gradient_tangent: state.velocity_gradient_tangent,
            }),
            Self::SkewRelativeGcl {
                action,
                mesh_divergence_tangent,
            } => {
                let mesh_velocity = action.mesh_velocity(reference)?;
                let mesh_velocity_tangent = geometry_tangent
                    .map_point::<D>(reference)
                    .map(|value| value / time_step);
                Ok(ConvectionPoint::Ale {
                    relative_velocity: std::array::from_fn(|axis| {
                        state.velocity[axis] - mesh_velocity[axis]
                    }),
                    relative_velocity_tangent: std::array::from_fn(|axis| {
                        state.velocity_tangent[axis] - mesh_velocity_tangent[axis]
                    }),
                    velocity: state.velocity,
                    velocity_tangent: state.velocity_tangent,
                    velocity_gradient: state.velocity_gradient,
                    velocity_gradient_tangent: state.velocity_gradient_tangent,
                    mesh_divergence: action.current_velocity_divergence(),
                    mesh_divergence_tangent: *mesh_divergence_tangent,
                })
            }
        }
    }
}

struct PointState<'a, const D: usize> {
    velocity: &'a [f64; D],
    velocity_tangent: &'a [f64; D],
    velocity_gradient: &'a [[f64; D]; D],
    velocity_gradient_tangent: &'a [[f64; D]; D],
}

enum ConvectionPoint<'a, const D: usize> {
    Disabled,
    Stationary {
        velocity: &'a [f64; D],
        velocity_tangent: &'a [f64; D],
        velocity_gradient: &'a [[f64; D]; D],
        velocity_gradient_tangent: &'a [[f64; D]; D],
    },
    Ale {
        relative_velocity: [f64; D],
        relative_velocity_tangent: [f64; D],
        velocity: &'a [f64; D],
        velocity_tangent: &'a [f64; D],
        velocity_gradient: &'a [[f64; D]; D],
        velocity_gradient_tangent: &'a [[f64; D]; D],
        mesh_divergence: f64,
        mesh_divergence_tangent: f64,
    },
}

impl<const D: usize> ConvectionPoint<'_, D> {
    fn action(
        &self,
        density: f64,
        test: f64,
        test_gradient: &[f64],
        test_gradient_tangent: &[f64],
        component: usize,
    ) -> (f64, f64) {
        match self {
            Self::Disabled => (0.0, 0.0),
            Self::Stationary {
                velocity,
                velocity_tangent,
                velocity_gradient,
                velocity_gradient_tangent,
            } => {
                let velocity_dot_test_gradient = dot(*velocity, test_gradient);
                let velocity_dot_test_gradient_tangent =
                    dot(*velocity_tangent, test_gradient) + dot(*velocity, test_gradient_tangent);
                let velocity_dot_velocity_gradient_tangent =
                    dot(*velocity_tangent, &velocity_gradient[component])
                        + dot(*velocity, &velocity_gradient_tangent[component]);
                (
                    stationary_convection_action(
                        density,
                        velocity,
                        velocity_gradient,
                        test,
                        test_gradient,
                        component,
                    ),
                    0.5 * density
                        * (velocity_dot_velocity_gradient_tangent * test
                            - velocity_dot_test_gradient_tangent * velocity[component]
                            - velocity_dot_test_gradient * velocity_tangent[component]),
                )
            }
            Self::Ale {
                relative_velocity,
                relative_velocity_tangent,
                velocity,
                velocity_tangent,
                velocity_gradient,
                velocity_gradient_tangent,
                mesh_divergence,
                mesh_divergence_tangent,
            } => {
                let relative_dot_test_gradient = dot(relative_velocity, test_gradient);
                let relative_dot_test_gradient_tangent =
                    dot(relative_velocity_tangent, test_gradient)
                        + dot(relative_velocity, test_gradient_tangent);
                let relative_dot_velocity_gradient_tangent =
                    dot(relative_velocity_tangent, &velocity_gradient[component])
                        + dot(relative_velocity, &velocity_gradient_tangent[component]);
                (
                    ale_convection_action(
                        density,
                        relative_velocity,
                        velocity,
                        velocity_gradient,
                        *mesh_divergence,
                        test,
                        test_gradient,
                        component,
                    ),
                    0.5 * density
                        * (relative_dot_velocity_gradient_tangent * test
                            - relative_dot_test_gradient_tangent * velocity[component]
                            - relative_dot_test_gradient * velocity_tangent[component]
                            + mesh_divergence_tangent * velocity[component] * test
                            + mesh_divergence * velocity_tangent[component] * test),
                )
            }
        }
    }
}

fn evaluate_velocity<const D: usize>(
    coefficients: &[[f64; D]],
    basis: &[f64],
    gradients: &[Vec<f64>],
) -> ([f64; D], [[f64; D]; D]) {
    let mut value = [0.0; D];
    let mut gradient = [[0.0; D]; D];
    for local in 0..coefficients.len() {
        for component in 0..D {
            value[component] += coefficients[local][component] * basis[local];
            for axis in 0..D {
                gradient[component][axis] +=
                    coefficients[local][component] * gradients[local][axis];
            }
        }
    }
    (value, gradient)
}

fn evaluate_velocity_flat<const D: usize>(
    coefficients: &[[f64; D]],
    basis: &[f64],
    gradients: &[f64],
) -> ([f64; D], [[f64; D]; D]) {
    let mut value = [0.0; D];
    let mut gradient = [[0.0; D]; D];
    for local in 0..coefficients.len() {
        for component in 0..D {
            value[component] += coefficients[local][component] * basis[local];
            for axis in 0..D {
                gradient[component][axis] +=
                    coefficients[local][component] * gradients[local * D + axis];
            }
        }
    }
    (value, gradient)
}

fn evaluate_velocity_tangent<const D: usize>(
    coefficients: &[[f64; D]],
    coefficient_tangents: &[[f64; D]],
    basis: &[f64],
    gradients: &[Vec<f64>],
    gradient_tangents: &[Vec<f64>],
) -> ([f64; D], [[f64; D]; D]) {
    let mut value = [0.0; D];
    let mut gradient = [[0.0; D]; D];
    for local in 0..coefficients.len() {
        for component in 0..D {
            value[component] += coefficient_tangents[local][component] * basis[local];
            for axis in 0..D {
                gradient[component][axis] += coefficient_tangents[local][component]
                    * gradients[local][axis]
                    + coefficients[local][component] * gradient_tangents[local][axis];
            }
        }
    }
    (value, gradient)
}

fn multiply_linearization<const D: usize>(
    left_tangent: &[f64],
    right: &[f64],
    left: &[f64],
    right_tangent: &[f64],
) -> Result<Vec<f64>, Diagnostic> {
    let matrix_entries = D
        .checked_mul(D)
        .ok_or_else(|| invalid("MINI transport matrix shape overflows usize"))?;
    if left_tangent.len() != matrix_entries
        || right.len() != matrix_entries
        || left.len() != matrix_entries
        || right_tangent.len() != matrix_entries
    {
        return Err(invalid(
            "MINI transport linearization requires four square matrices",
        ));
    }
    Ok((0..matrix_entries)
        .map(|entry| {
            let row = entry / D;
            let column = entry % D;
            (0..D)
                .map(|axis| {
                    left_tangent[row * D + axis] * right[axis * D + column]
                        + left[row * D + axis] * right_tangent[axis * D + column]
                })
                .sum()
        })
        .collect())
}

fn symmetric_gradient_test<const D: usize>(
    gradient: &[[f64; D]; D],
    test_gradient: &[f64],
    test_component: usize,
) -> f64 {
    (0..D)
        .map(|axis| {
            (gradient[test_component][axis] + gradient[axis][test_component]) * test_gradient[axis]
        })
        .sum()
}

fn symmetric_gradient_test_tangent<const D: usize>(
    gradient: &[[f64; D]; D],
    gradient_tangent: &[[f64; D]; D],
    test_gradient: &[f64],
    test_gradient_tangent: &[f64],
    test_component: usize,
) -> f64 {
    (0..D)
        .map(|axis| {
            (gradient_tangent[test_component][axis] + gradient_tangent[axis][test_component])
                * test_gradient[axis]
                + (gradient[test_component][axis] + gradient[axis][test_component])
                    * test_gradient_tangent[axis]
        })
        .sum()
}

/// State-projection form with the established per-axis operation order.
fn projected_symmetric_gradient_test<const D: usize>(
    gradient: &[[f64; D]; D],
    test_gradient: &[f64],
    test_component: usize,
) -> f64 {
    (0..D)
        .map(|axis| {
            gradient[test_component][axis] * test_gradient[axis]
                + gradient[axis][test_component] * test_gradient[axis]
        })
        .sum()
}

struct ProjectedConvectiveLinearization<'a, const D: usize> {
    density: f64,
    velocity: &'a [f64; D],
    velocity_gradient: &'a [[f64; D]; D],
    basis: &'a [f64],
    gradients: &'a [f64],
}

impl<const D: usize> ProjectedConvectiveLinearization<'_, D> {
    fn entry(
        &self,
        row_basis: usize,
        row_component: usize,
        column_basis: usize,
        column_component: usize,
    ) -> f64 {
        let row_value = self.basis[row_basis];
        let column_value = self.basis[column_basis];
        let row_gradient = &self.gradients[row_basis * D..(row_basis + 1) * D];
        let column_gradient = &self.gradients[column_basis * D..(column_basis + 1) * D];
        let diagonal = usize::from(row_component == column_component) as f64;
        0.5 * self.density
            * (column_value * self.velocity_gradient[row_component][column_component] * row_value
                + diagonal * dot(self.velocity, column_gradient) * row_value
                - column_value * row_gradient[column_component] * self.velocity[row_component]
                - diagonal * dot(self.velocity, row_gradient) * column_value)
    }
}

fn accumulate(
    residual: &mut f64,
    jvp: &mut f64,
    measure: f64,
    measure_tangent: f64,
    integrand: f64,
    integrand_tangent: f64,
) {
    *residual += measure * integrand;
    *jvp += measure_tangent * integrand + measure * integrand_tangent;
}

fn local_velocity<const D: usize>(basis: usize, component: usize) -> usize {
    basis * D + component
}

fn trace<const D: usize>(matrix: &[[f64; D]; D]) -> f64 {
    let mut value = matrix[0][0];
    for (axis, row) in matrix.iter().enumerate().skip(1) {
        value += row[axis];
    }
    value
}

fn trace_flat<const D: usize>(matrix: &[f64]) -> f64 {
    let mut value = matrix[0];
    for axis in 1..D {
        value += matrix[axis * D + axis];
    }
    value
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_DISCRETIZATION, message)
}

#[cfg(test)]
mod tests;
