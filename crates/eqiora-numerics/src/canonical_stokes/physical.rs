use eqiora_assembly::AssemblyReport;
use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id};
use eqiora_meshing::SimplicialMesh;
use eqiora_realization::VectorLayoutKind;
use eqiora_solver::{
    CanonicalCsrSystemView, LinearOperatorProperties, LinearProblem, LinearSolution, SolverPlan,
};

use super::realization::{dimensionless_body_force, force_potential_id, pressure_id, velocity_id};
use super::{IncompressibleFlowScaleProfile2d, SteadyIncompressibleStokesModel2d};
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
    model: SteadyIncompressibleStokesModel2d,
    scales: IncompressibleFlowScaleProfile2d,
}

impl FinalizedSteadyStokesMini2dProblem {
    pub(super) const fn new(
        inner: FinalizedSimplicialMiniStokes2dProblem,
        physical_mesh: SimplicialMesh,
        model: SteadyIncompressibleStokesModel2d,
        scales: IncompressibleFlowScaleProfile2d,
    ) -> Self {
        Self {
            inner,
            physical_mesh,
            model,
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
        let (vertices, bubbles, pressure) = self.normalized_fields(vertices, bubbles, pressure);
        self.inner
            .reconstruction()
            .boundary_reaction(vertices, bubbles, pressure, name)?
            .map(|force| physical_force(force, self.scales))
            .transpose()
    }

    pub(crate) fn force_balance(
        &self,
        vertices: &[[f64; 2]],
        bubbles: &[[f64; 2]],
        pressure: &[f64],
    ) -> Result<[[f64; 2]; 3], Diagnostic> {
        let (vertices, bubbles, pressure) = self.normalized_fields(vertices, bubbles, pressure);
        let context = self.inner.reconstruction();
        let body_force = dimensionless_body_force(&self.model, self.scales);
        Ok([
            physical_force(
                context.global_boundary_reaction(vertices, bubbles, pressure)?,
                self.scales,
            )?,
            physical_force(context.integrated_body_force(&body_force)?, self.scales)?,
            physical_force(context.integrated_boundary_traction()?, self.scales)?,
        ])
    }

    #[allow(clippy::type_complexity)]
    fn normalized_fields(
        &self,
        vertices: &[[f64; 2]],
        bubbles: &[[f64; 2]],
        pressure: &[f64],
    ) -> (Vec<[f64; 2]>, Vec<[f64; 2]>, Vec<f64>) {
        let velocity = |values: &[[f64; 2]]| {
            values
                .iter()
                .map(|v| v.map(|c| c / self.scales.velocity_value()))
                .collect()
        };
        (
            velocity(vertices),
            velocity(bubbles),
            pressure
                .iter()
                .map(|v| v / self.scales.pressure_value())
                .collect(),
        )
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
            model,
            scales,
        } = self;
        let dimensionless = inner.finish(solution)?;
        SteadyStokesMiniSolution2d::reconstruct(dimensionless, physical_mesh, model, scales)
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
    model: SteadyIncompressibleStokesModel2d,
    velocity: SimplicialMiniVelocityField2d,
    pressure: SimplicialP1Field,
    pressure_reference: SteadyStokesPressureReference2d,
    scales: IncompressibleFlowScaleProfile2d,
    dimensionless: SimplicialMiniStokesSolution2d,
}

impl SteadyStokesMiniSolution2d {
    fn reconstruct(
        dimensionless: SimplicialMiniStokesSolution2d,
        physical_mesh: SimplicialMesh,
        model: SteadyIncompressibleStokesModel2d,
        scales: IncompressibleFlowScaleProfile2d,
    ) -> Result<Self, Diagnostic> {
        let velocity_scale = scales.velocity_value();
        let pressure_scale = scales.pressure_value();
        let gauge_scale = scales.gauge_value();

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
        if pressure_reference
            .gauge_multiplier()
            .into_iter()
            .any(|value| !value.is_finite())
        {
            return Err(invalid_realization(
                "coherent-SI Stokes reconstruction produced non-finite evidence",
            ));
        }
        Ok(Self {
            model,
            velocity,
            pressure,
            pressure_reference,
            scales,
            dimensionless,
        })
    }

    /// Exact Semantic velocity Field represented by [`Self::velocity`].
    #[must_use]
    pub fn velocity_field(&self) -> Id<kinds::Field> {
        velocity_id(&self.model)
    }

    /// Exact Semantic pressure Field represented by [`Self::pressure`].
    #[must_use]
    pub fn pressure_field(&self) -> Id<kinds::Field> {
        pressure_id(&self.model)
    }

    /// Exact retained conservative-force-potential Field.
    #[must_use]
    pub fn force_potential_field(&self) -> Option<Id<kinds::Field>> {
        force_potential_id(&self.model)
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
    /// # Errors
    /// Rejects non-finite physical force reconstruction.
    pub fn boundary_reaction(&self) -> Result<[f64; 2], Diagnostic> {
        physical_force(self.dimensionless.boundary_reaction()?, self.scales)
    }

    /// Physical integrated body force per unit out-of-plane thickness, in N/m.
    /// # Errors
    /// Rejects non-finite physical force reconstruction.
    pub fn integrated_body_force(&self) -> Result<[f64; 2], Diagnostic> {
        let body_force = dimensionless_body_force(&self.model, self.scales);
        physical_force(
            self.dimensionless.integrated_body_force(&body_force)?,
            self.scales,
        )
    }

    /// Applied boundary traction per unit out-of-plane thickness, in N/m.
    /// # Errors
    /// Rejects non-finite physical force reconstruction.
    pub fn integrated_boundary_traction(&self) -> Result<[f64; 2], Diagnostic> {
        physical_force(
            self.dimensionless.integrated_boundary_traction()?,
            self.scales,
        )
    }

    /// Physical pressure integral over the intrinsic 2D Domain, in Pa m².
    /// # Errors
    /// Rejects non-finite physical pressure integration.
    pub fn pressure_integral(&self) -> Result<f64, Diagnostic> {
        let value = self.dimensionless.pressure_integral()?
            * (self.scales.pressure_value() * self.scales.length_value().powi(2));
        if !value.is_finite() {
            return Err(invalid_realization(
                "physical Stokes pressure integral is non-finite",
            ));
        }
        Ok(value)
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

fn physical_force(
    force: [f64; 2],
    scales: IncompressibleFlowScaleProfile2d,
) -> Result<[f64; 2], Diagnostic> {
    let force = force.map(|v| v * (scales.pressure_value() * scales.length_value()));
    if force.iter().any(|v| !v.is_finite()) {
        return Err(invalid_realization("physical Stokes force is non-finite"));
    }
    Ok(force)
}
