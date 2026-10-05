use std::sync::Arc;

use eqiora_assembly::{AssemblyReport, LinearSystem};
use eqiora_core::Diagnostic;
use eqiora_meshing::{
    GeometryMap, MeshEntity, MeshGeometry, MeshTopology, QuadratureRule, SimplicialMesh,
};
use eqiora_solver::{CanonicalCsrSystemView, SolveReport};

use super::acceptance::require_error_quadrature;
use super::element::{MiniSpaces, evaluate_fields, physical_gradients};
use super::{COMPONENTS, DIMENSION, invalid};
use crate::simplicial_elliptic::SimplicialP1Field;

/// Continuous P1-plus-cell-bubble velocity on an affine triangular mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct SimplicialMiniVelocityField2d {
    pub(super) mesh: SimplicialMesh,
    pub(super) vertex_values: Vec<[f64; COMPONENTS]>,
    pub(super) cell_bubble_values: Vec<[f64; COMPONENTS]>,
}

impl SimplicialMiniVelocityField2d {
    /// Construct one shaped MINI velocity field from canonical coefficients.
    ///
    /// # Errors
    /// Returns `EQ0801` unless the vertex and cell-bubble inventories match
    /// the exact two-dimensional simplicial mesh and every coefficient is
    /// finite.
    pub fn new(
        mesh: SimplicialMesh,
        vertex_values: Vec<[f64; COMPONENTS]>,
        cell_bubble_values: Vec<[f64; COMPONENTS]>,
    ) -> Result<Self, Diagnostic> {
        let cell_count = mesh
            .entity_count(DIMENSION)
            .expect("2D simplex mesh owns cells");
        if vertex_values.len() != mesh.vertices().len()
            || cell_bubble_values.len() != cell_count
            || vertex_values
                .iter()
                .chain(&cell_bubble_values)
                .flatten()
                .any(|value| !value.is_finite())
        {
            return Err(invalid(
                "MINI velocity coefficients must be finite and match the mesh layout",
            ));
        }
        Ok(Self {
            mesh,
            vertex_values,
            cell_bubble_values,
        })
    }

    /// Accepted mesh carrying the velocity space.
    #[must_use]
    pub const fn mesh(&self) -> &SimplicialMesh {
        &self.mesh
    }

    /// Continuous P1 coefficients in canonical mesh-vertex order.
    #[must_use]
    pub fn vertex_values(&self) -> &[[f64; COMPONENTS]] {
        &self.vertex_values
    }

    /// Interior bubble coefficient for each cell and velocity component.
    #[must_use]
    pub fn cell_bubble_values(&self) -> &[[f64; COMPONENTS]] {
        &self.cell_bubble_values
    }
}

/// Continuous error evidence for one accepted MINI Stokes solution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimplicialMiniStokesErrorNorms2d {
    velocity_l2: f64,
    velocity_h1_seminorm: f64,
    pressure_l2: f64,
    divergence_l2: f64,
}

impl SimplicialMiniStokesErrorNorms2d {
    /// Velocity vector L2 norm.
    #[must_use]
    pub const fn velocity_l2(self) -> f64 {
        self.velocity_l2
    }

    /// Frobenius H1 seminorm of the velocity error.
    #[must_use]
    pub const fn velocity_h1_seminorm(self) -> f64 {
        self.velocity_h1_seminorm
    }

    /// Pressure scalar L2 norm in the selected pressure reference.
    #[must_use]
    pub const fn pressure_l2(self) -> f64 {
        self.pressure_l2
    }

    /// L2 norm of the discrete velocity divergence.
    #[must_use]
    pub const fn divergence_l2(self) -> f64 {
        self.divergence_l2
    }
}

/// Accepted resolution of the constant-pressure mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimplicialMiniStokesPressureReference2d {
    /// One global multiplier enforces an exactly zero pressure integral.
    ZeroIntegral {
        /// Accepted multiplier associated with the zero-integral constraint.
        multiplier: f64,
    },
    /// A prescribed full-vector traction on positive boundary measure fixes
    /// the constant pressure mode without adding an algebraic gauge.
    BoundaryTraction,
}

impl SimplicialMiniStokesPressureReference2d {
    /// Gauge multiplier when and only when a zero-integral constraint exists.
    #[must_use]
    pub const fn gauge_multiplier(self) -> Option<f64> {
        match self {
            Self::ZeroIntegral { multiplier } => Some(multiplier),
            Self::BoundaryTraction => None,
        }
    }
}

/// Accepted algebra, fields, conservation evidence, and execution reports.
#[derive(Debug, Clone, PartialEq)]
pub struct SimplicialMiniStokesSolution2d {
    pub(super) velocity: SimplicialMiniVelocityField2d,
    pub(super) pressure: SimplicialP1Field,
    pub(super) pressure_reference: SimplicialMiniStokesPressureReference2d,
    pub(super) algebraic_values: Vec<f64>,
    pub(super) canonical_system: Arc<CanonicalCsrSystemView>,
    pub(super) full_system: LinearSystem<f64>,
    pub(super) volume_only_system: LinearSystem<f64>,
    pub(super) boundary_reaction: [f64; COMPONENTS],
    pub(super) named_boundary_reactions: Vec<(String, [f64; COMPONENTS])>,
    pub(super) integrated_body_force: [f64; COMPONENTS],
    pub(super) integrated_boundary_traction: [f64; COMPONENTS],
    pub(super) pressure_integral: f64,
    pub(super) continuity_residual_norm: f64,
    pub(super) assembly_report: AssemblyReport,
    pub(super) solve_report: SolveReport,
}

impl SimplicialMiniStokesSolution2d {
    /// MINI velocity field.
    #[must_use]
    pub const fn velocity(&self) -> &SimplicialMiniVelocityField2d {
        &self.velocity
    }

    /// Continuous P1 pressure in the selected pressure reference.
    #[must_use]
    pub const fn pressure(&self) -> &SimplicialP1Field {
        &self.pressure
    }

    /// Exact pressure-nullspace resolution selected by the realization.
    #[must_use]
    pub const fn pressure_reference(&self) -> SimplicialMiniStokesPressureReference2d {
        self.pressure_reference
    }

    /// Gauge multiplier when a zero-integral pressure constraint exists.
    #[must_use]
    pub const fn gauge_multiplier(&self) -> Option<f64> {
        self.pressure_reference.gauge_multiplier()
    }

    /// Reduced mixed algebra in canonical block order.
    #[must_use]
    pub fn algebraic_values(&self) -> &[f64] {
        &self.algebraic_values
    }

    /// Captured reduced saddle-point system submitted to the selected solver.
    #[must_use]
    pub fn linear_system(&self) -> &CanonicalCsrSystemView {
        self.canonical_system.as_ref()
    }

    /// Unconstrained loaded system retained for reactions and acceptance.
    #[must_use]
    pub const fn full_system(&self) -> &LinearSystem<f64> {
        &self.full_system
    }

    /// Unconstrained volume-and-constraint system before traction packets.
    ///
    /// Its matrix is exactly the loaded full-system matrix. The right-hand-side
    /// difference is the independently assembled prescribed boundary action.
    #[must_use]
    pub const fn volume_only_system(&self) -> &LinearSystem<f64> {
        &self.volume_only_system
    }

    /// Sum of residual forces at eliminated essential-velocity vertices.
    #[must_use]
    pub const fn boundary_reaction(&self) -> [f64; COMPONENTS] {
        self.boundary_reaction
    }

    /// Reaction over the constrained vertices attributed to one named surface.
    ///
    /// Returns `None` when the accepted boundary did not name that surface.
    #[must_use]
    pub fn named_boundary_reaction(&self, name: &str) -> Option<[f64; COMPONENTS]> {
        self.named_boundary_reactions
            .iter()
            .find_map(|(candidate, reaction)| (candidate == name).then_some(*reaction))
    }

    pub(crate) fn named_boundary_reactions(&self) -> &[(String, [f64; COMPONENTS])] {
        &self.named_boundary_reactions
    }

    /// Independently integrated body-force resultant.
    #[must_use]
    pub const fn integrated_body_force(&self) -> [f64; COMPONENTS] {
        self.integrated_body_force
    }

    /// Independently integrated prescribed boundary traction.
    #[must_use]
    pub const fn integrated_boundary_traction(&self) -> [f64; COMPONENTS] {
        self.integrated_boundary_traction
    }

    /// Independently integrated pressure over the complete domain.
    #[must_use]
    pub const fn pressure_integral(&self) -> f64 {
        self.pressure_integral
    }

    /// Euclidean norm of the weak incompressibility equations `B u = 0`.
    ///
    /// This excludes the pressure-gauge multiplier contribution, so it is
    /// distinct from both the mixed-system residual and the strong L2 norm of
    /// `div(u)`.
    #[must_use]
    pub const fn continuity_residual_norm(&self) -> f64 {
        self.continuity_residual_norm
    }

    /// Accepted assembly placement and packet shape.
    #[must_use]
    pub const fn assembly_report(&self) -> &AssemblyReport {
        &self.assembly_report
    }

    /// Accepted solver and independently recomputed true-residual evidence.
    #[must_use]
    pub const fn solve_report(&self) -> &SolveReport {
        &self.solve_report
    }

    /// Integrate velocity, velocity-gradient, pressure, and divergence errors.
    ///
    /// # Errors
    /// Returns `EQ0801` for an incompatible rule or non-finite oracle value.
    pub fn error_norms<V, G, P>(
        &self,
        quadrature: &QuadratureRule,
        exact_velocity: V,
        exact_velocity_gradient: G,
        exact_pressure: P,
    ) -> Result<SimplicialMiniStokesErrorNorms2d, Diagnostic>
    where
        V: Fn([f64; DIMENSION]) -> [f64; COMPONENTS],
        G: Fn([f64; DIMENSION]) -> [[f64; DIMENSION]; COMPONENTS],
        P: Fn([f64; DIMENSION]) -> f64,
    {
        require_error_quadrature(quadrature)?;
        let mut velocity_l2 = 0.0;
        let mut velocity_h1 = 0.0;
        let mut pressure_l2 = 0.0;
        let mut divergence_l2 = 0.0;
        let spaces = MiniSpaces::new()?;
        let cell_count = self
            .velocity
            .mesh
            .entity_count(DIMENSION)
            .expect("2D simplex mesh owns cells");
        for cell_index in 0..cell_count {
            let cell = MeshEntity::new(DIMENSION, cell_index);
            let geometry = self
                .velocity
                .mesh
                .geometry_map(cell)
                .expect("accepted simplex cell owns geometry");
            let vertices = self
                .velocity
                .mesh
                .entity_vertices(cell)
                .expect("accepted simplex cell owns vertices");
            let inverse = geometry.inverse_jacobian()?;
            for point in quadrature.points() {
                let basis = spaces.tabulate(&point.coordinates)?;
                let gradients = physical_gradients(&basis, &inverse);
                let mut coordinates = [0.0; DIMENSION];
                geometry.map_point(&point.coordinates, &mut coordinates)?;
                let (velocity, velocity_gradient, pressure) = evaluate_fields(
                    &self.velocity,
                    &self.pressure,
                    cell_index,
                    &vertices,
                    &basis.values,
                    &basis.pressure_values,
                    &gradients,
                );
                let expected_velocity = exact_velocity(coordinates);
                let expected_gradient = exact_velocity_gradient(coordinates);
                let expected_pressure = exact_pressure(coordinates);
                let scale = point.weight * geometry.measure_scale();
                for component in 0..COMPONENTS {
                    velocity_l2 +=
                        scale * (velocity[component] - expected_velocity[component]).powi(2);
                    for axis in 0..DIMENSION {
                        velocity_h1 += scale
                            * (velocity_gradient[component][axis]
                                - expected_gradient[component][axis])
                                .powi(2);
                    }
                }
                pressure_l2 += scale * (pressure - expected_pressure).powi(2);
                let divergence = velocity_gradient[0][0] + velocity_gradient[1][1];
                divergence_l2 += scale * divergence.powi(2);
            }
        }
        let values = [velocity_l2, velocity_h1, pressure_l2, divergence_l2];
        if values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(invalid("MINI Stokes error integration is non-finite"));
        }
        Ok(SimplicialMiniStokesErrorNorms2d {
            velocity_l2: velocity_l2.sqrt(),
            velocity_h1_seminorm: velocity_h1.sqrt(),
            pressure_l2: pressure_l2.sqrt(),
            divergence_l2: divergence_l2.sqrt(),
        })
    }
}
