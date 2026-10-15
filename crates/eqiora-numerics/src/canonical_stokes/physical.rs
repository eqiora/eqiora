use eqiora_assembly::AssemblyReport;
use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id};
use eqiora_meshing::SimplicialMesh;
use eqiora_realization::VectorLayoutKind;
use eqiora_solver::{
    CanonicalCsrSystemView, LinearOperatorProperties, LinearProblem, LinearSolution, SolverPlan,
};

use super::IncompressibleFlowScaleProfile2d;
use crate::finalized_spatial::FinalizedSimplicialMiniStokes2dProblem;
use crate::simplicial_elliptic::SimplicialP1Field;
use crate::simplicial_stokes::{
    SimplicialMiniStokesPressureReference2d, SimplicialMiniStokesSolution2d,
    SimplicialMiniVelocityField2d,
};

/// Finalized dimensionless MINI algebra plus opaque coherent-SI reconstruction.
///
/// The solver sees only the congruence-scaled canonical CSR system. Semantic
/// Field identity, the physical mesh, and every dimensional reconstruction
/// scale remain sealed until an accepted solution to that exact system returns.
#[derive(Debug, Clone, PartialEq)]
pub struct FinalizedSteadyStokesMini2dProblem {
    inner: FinalizedSimplicialMiniStokes2dProblem,
    physical_mesh: SimplicialMesh,
    velocity_field: Id<kinds::Field>,
    pressure_field: Id<kinds::Field>,
    force_potential_field: Id<kinds::Field>,
    scales: IncompressibleFlowScaleProfile2d,
}

impl FinalizedSteadyStokesMini2dProblem {
    pub(super) const fn new(
        inner: FinalizedSimplicialMiniStokes2dProblem,
        physical_mesh: SimplicialMesh,
        velocity_field: Id<kinds::Field>,
        pressure_field: Id<kinds::Field>,
        force_potential_field: Id<kinds::Field>,
        scales: IncompressibleFlowScaleProfile2d,
    ) -> Self {
        Self {
            inner,
            physical_mesh,
            velocity_field,
            pressure_field,
            force_potential_field,
            scales,
        }
    }

    /// Mathematical property preserved by the symmetric congruence.
    #[must_use]
    pub fn operator_properties(&self) -> LinearOperatorProperties {
        self.inner.operator_properties()
    }

    /// Exact solver policy selected by the field-wise Realization.
    #[must_use]
    pub const fn solver_plan(&self) -> SolverPlan {
        self.inner.solver_plan()
    }

    /// Exact algebraic vector layout admitted by the Realization.
    #[must_use]
    pub const fn vector_layout(&self) -> VectorLayoutKind {
        self.inner.vector_layout()
    }

    /// Borrow the sole dimensionless CSR system submitted to execution.
    #[must_use]
    pub fn canonical_csr_system_view(&self) -> &CanonicalCsrSystemView {
        self.inner.canonical_csr_system_view()
    }

    /// Accepted assembly placement and packet-shape evidence.
    #[must_use]
    pub const fn assembly_report(&self) -> &AssemblyReport {
        self.inner.assembly_report()
    }

    /// Borrow the dimensionless problem through the common solver boundary.
    ///
    /// # Errors
    /// Returns a structured diagnostic only if captured CSR invariants were
    /// contradicted after construction.
    pub fn linear_problem(&self) -> Result<LinearProblem<'_>, Diagnostic> {
        self.inner.linear_problem()
    }

    pub(crate) fn boundary_reaction(
        &self,
        vertices: &[[f64; 2]],
        bubbles: &[[f64; 2]],
        pressure: &[f64],
        name: &str,
    ) -> Result<Option<[f64; 2]>, Diagnostic> {
        let normalize_velocity = |values: &[[f64; 2]]| {
            values
                .iter()
                .map(|value| value.map(|component| component / self.scales.velocity_value()))
                .collect()
        };
        let force = self.inner.boundary_reaction(
            normalize_velocity(vertices),
            normalize_velocity(bubbles),
            pressure
                .iter()
                .map(|v| v / self.scales.pressure_value())
                .collect(),
            name,
        )?;
        scale_reaction(force, self.scales)
    }

    /// Reaccept one exact dimensionless solution and reconstruct coherent-SI fields.
    ///
    /// # Errors
    /// Preserves finalized residual/topology diagnostics and rejects any
    /// non-finite physical reconstruction.
    pub fn finish(
        self,
        solution: LinearSolution,
    ) -> Result<SteadyStokesMiniSolution2d, Diagnostic> {
        let Self {
            inner,
            physical_mesh,
            velocity_field,
            pressure_field,
            force_potential_field,
            scales,
        } = self;
        let dimensionless = inner.finish(solution)?;
        SteadyStokesMiniSolution2d::reconstruct(
            dimensionless,
            physical_mesh,
            velocity_field,
            pressure_field,
            force_potential_field,
            scales,
        )
    }
}

/// Physical origin of the accepted pressure representative.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SteadyStokesPressureReference2d {
    /// One global zero-integral constraint selects the pressure representative.
    ZeroIntegral {
        /// Physical constraint multiplier in inverse seconds.
        multiplier: f64,
    },
    /// Positive-measure prescribed traction determines absolute pressure.
    BoundaryTraction,
}

impl SteadyStokesPressureReference2d {
    /// Physical gauge multiplier exactly when a zero-integral constraint exists.
    #[must_use]
    pub const fn gauge_multiplier(self) -> Option<f64> {
        match self {
            Self::ZeroIntegral { multiplier } => Some(multiplier),
            Self::BoundaryTraction => None,
        }
    }
}

/// Accepted coherent-SI fields and balance evidence for one canonical Stokes solve.
///
/// Physical fields are bound to exact Semantic Field identities. The complete
/// dimensionless solution remains available for solver, assembly, and
/// congruence evidence; its values must not be interpreted as SI quantities.
/// Absence of a pressure gauge is retained explicitly: a traction-determined
/// pressure reference never fabricates a zero multiplier.
#[derive(Debug, Clone, PartialEq)]
pub struct SteadyStokesMiniSolution2d {
    velocity_field: Id<kinds::Field>,
    pressure_field: Id<kinds::Field>,
    force_potential_field: Id<kinds::Field>,
    velocity: SimplicialMiniVelocityField2d,
    pressure: SimplicialP1Field,
    pressure_reference: SteadyStokesPressureReference2d,
    boundary_reaction: [f64; 2],
    integrated_body_force: [f64; 2],
    integrated_boundary_traction: [f64; 2],
    pressure_integral: f64,
    scales: IncompressibleFlowScaleProfile2d,
    dimensionless: SimplicialMiniStokesSolution2d,
}

impl SteadyStokesMiniSolution2d {
    fn reconstruct(
        dimensionless: SimplicialMiniStokesSolution2d,
        physical_mesh: SimplicialMesh,
        velocity_field: Id<kinds::Field>,
        pressure_field: Id<kinds::Field>,
        force_potential_field: Id<kinds::Field>,
        scales: IncompressibleFlowScaleProfile2d,
    ) -> Result<Self, Diagnostic> {
        let velocity_scale = scales.velocity_value();
        let pressure_scale = scales.pressure_value();
        let gauge_scale = scales.gauge_value();
        let force_scale = pressure_scale * scales.length_value();
        let pressure_integral_scale = pressure_scale * scales.length_value().powi(2);

        let vertex_values = dimensionless
            .velocity()
            .vertex_values()
            .iter()
            .map(|value| value.map(|component| component * velocity_scale))
            .collect();
        let cell_bubble_values = dimensionless
            .velocity()
            .cell_bubble_values()
            .iter()
            .map(|value| value.map(|component| component * velocity_scale))
            .collect();
        let velocity = SimplicialMiniVelocityField2d::new(
            physical_mesh.clone(),
            vertex_values,
            cell_bubble_values,
        )?;
        let pressure = SimplicialP1Field::new(
            physical_mesh,
            dimensionless
                .pressure()
                .vertex_values()
                .iter()
                .map(|value| value * pressure_scale)
                .collect(),
        )?;
        let pressure_reference = match dimensionless.pressure_reference() {
            SimplicialMiniStokesPressureReference2d::ZeroIntegral { multiplier } => {
                SteadyStokesPressureReference2d::ZeroIntegral {
                    multiplier: multiplier * gauge_scale,
                }
            }
            SimplicialMiniStokesPressureReference2d::BoundaryTraction => {
                SteadyStokesPressureReference2d::BoundaryTraction
            }
        };
        let boundary_reaction = dimensionless
            .boundary_reaction()
            .map(|value| value * force_scale);
        let integrated_body_force = dimensionless
            .integrated_body_force()
            .map(|value| value * force_scale);
        let integrated_boundary_traction = dimensionless
            .integrated_boundary_traction()
            .map(|value| value * force_scale);
        let pressure_integral = dimensionless.pressure_integral() * pressure_integral_scale;
        if pressure_reference
            .gauge_multiplier()
            .into_iter()
            .chain([pressure_integral])
            .chain(boundary_reaction)
            .chain(integrated_body_force)
            .chain(integrated_boundary_traction)
            .any(|value| !value.is_finite())
        {
            return Err(invalid_realization(
                "coherent-SI Stokes reconstruction produced non-finite evidence",
            ));
        }
        Ok(Self {
            velocity_field,
            pressure_field,
            force_potential_field,
            velocity,
            pressure,
            pressure_reference,
            boundary_reaction,
            integrated_body_force,
            integrated_boundary_traction,
            pressure_integral,
            scales,
            dimensionless,
        })
    }

    /// Exact Semantic velocity Field represented by [`Self::velocity`].
    #[must_use]
    pub const fn velocity_field(&self) -> Id<kinds::Field> {
        self.velocity_field
    }

    /// Exact Semantic pressure Field represented by [`Self::pressure`].
    #[must_use]
    pub const fn pressure_field(&self) -> Id<kinds::Field> {
        self.pressure_field
    }

    /// Exact retained conservative-force-potential Field.
    #[must_use]
    pub const fn force_potential_field(&self) -> Id<kinds::Field> {
        self.force_potential_field
    }

    /// Physical velocity in metres per second on the original mesh.
    #[must_use]
    pub const fn velocity(&self) -> &SimplicialMiniVelocityField2d {
        &self.velocity
    }

    /// Physical pressure in pascals on the original mesh.
    #[must_use]
    pub const fn pressure(&self) -> &SimplicialP1Field {
        &self.pressure
    }

    /// Exact physical pressure-reference evidence.
    #[must_use]
    pub const fn pressure_reference(&self) -> SteadyStokesPressureReference2d {
        self.pressure_reference
    }

    /// Physical gauge multiplier only when a zero-integral constraint exists.
    #[must_use]
    pub const fn gauge_multiplier(&self) -> Option<f64> {
        self.pressure_reference.gauge_multiplier()
    }

    /// Physical boundary reaction per unit out-of-plane thickness, in N/m.
    #[must_use]
    pub const fn boundary_reaction(&self) -> [f64; 2] {
        self.boundary_reaction
    }

    /// Physical integrated body force per unit out-of-plane thickness, in N/m.
    #[must_use]
    pub const fn integrated_body_force(&self) -> [f64; 2] {
        self.integrated_body_force
    }

    /// Applied boundary traction per unit out-of-plane thickness, in N/m.
    #[must_use]
    pub const fn integrated_boundary_traction(&self) -> [f64; 2] {
        self.integrated_boundary_traction
    }

    /// Physical pressure integral over the intrinsic 2D Domain, in Pa m².
    #[must_use]
    pub const fn pressure_integral(&self) -> f64 {
        self.pressure_integral
    }

    /// Exact characteristic scales selected by the Realization.
    #[must_use]
    pub const fn scales(&self) -> IncompressibleFlowScaleProfile2d {
        self.scales
    }

    /// Accepted dimensionless algebra, assembly, and solver evidence.
    #[must_use]
    pub const fn dimensionless_solution(&self) -> &SimplicialMiniStokesSolution2d {
        &self.dimensionless
    }
}

fn invalid_realization(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}

fn scale_reaction(
    force: Option<[f64; 2]>,
    scales: IncompressibleFlowScaleProfile2d,
) -> Result<Option<[f64; 2]>, Diagnostic> {
    force
        .map(|force| {
            let force =
                force.map(|value| value * (scales.pressure_value() * scales.length_value()));
            if force.iter().any(|v| !v.is_finite()) {
                return Err(invalid_realization(
                    "physical Stokes boundary force is non-finite",
                ));
            }
            Ok(force)
        })
        .transpose()
}
