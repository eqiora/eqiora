use std::sync::Arc;

use eqiora_assembly::{AssemblyReport, LinearSystem};
use eqiora_core::Diagnostic;
use eqiora_meshing::{QuadratureRule, SimplicialMesh};
use eqiora_solver::{CanonicalCsrSystemView, LinearSolution};

use super::acceptance::{
    integrate_pressure, require_weak_incompressibility, require_zero_gauge_multiplier,
};
use super::api::{
    SimplicialMiniStokesPressureReference2d, SimplicialMiniStokesSolution2d,
    SimplicialMiniVelocityField2d,
};
use super::layout::MixedLayout;
use super::{COMPONENTS, invalid};
use crate::simplicial_elliptic::SimplicialP1Field;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FinalizedMiniStokesAssembly {
    pub(super) mesh: SimplicialMesh,
    pub(super) layout: MixedLayout,
    pub(super) fixed_velocity: Vec<Option<[f64; COMPONENTS]>>,
    pub(super) named_reaction_vertices: Vec<(String, Vec<usize>)>,
    pub(super) linear_system: LinearSystem<f64>,
    pub(super) full_system: LinearSystem<f64>,
    pub(super) volume_only_system: LinearSystem<f64>,
    pub(super) integrated_body_force: [f64; COMPONENTS],
    pub(super) integrated_boundary_traction: [f64; COMPONENTS],
    pub(super) quadrature: QuadratureRule,
    pub(super) assembly_report: AssemblyReport,
}

impl FinalizedMiniStokesAssembly {
    pub(crate) fn into_canonical(
        self,
    ) -> Result<(Arc<CanonicalCsrSystemView>, FinalizedMiniStokesState), Diagnostic> {
        let Self {
            mesh,
            layout,
            fixed_velocity,
            named_reaction_vertices,
            linear_system,
            full_system,
            volume_only_system,
            integrated_body_force,
            integrated_boundary_traction,
            quadrature,
            assembly_report,
        } = self;
        let canonical_system = Arc::new(CanonicalCsrSystemView::new(
            &linear_system,
            eqiora_solver::LinearOperatorProperties::SymmetricIndefinite,
        )?);
        Ok((
            canonical_system,
            FinalizedMiniStokesState {
                mesh,
                layout,
                fixed_velocity,
                named_reaction_vertices,
                full_system,
                volume_only_system,
                integrated_body_force,
                integrated_boundary_traction,
                quadrature,
                assembly_report,
            },
        ))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FinalizedMiniStokesState {
    mesh: SimplicialMesh,
    layout: MixedLayout,
    fixed_velocity: Vec<Option<[f64; COMPONENTS]>>,
    named_reaction_vertices: Vec<(String, Vec<usize>)>,
    full_system: LinearSystem<f64>,
    volume_only_system: LinearSystem<f64>,
    integrated_body_force: [f64; COMPONENTS],
    integrated_boundary_traction: [f64; COMPONENTS],
    quadrature: QuadratureRule,
    assembly_report: AssemblyReport,
}

impl FinalizedMiniStokesState {
    pub(crate) const fn assembly_report(&self) -> &AssemblyReport {
        &self.assembly_report
    }

    pub(crate) fn finish(
        self,
        solved: LinearSolution,
        canonical_system: Arc<CanonicalCsrSystemView>,
    ) -> Result<SimplicialMiniStokesSolution2d, Diagnostic> {
        if solved.values().len() != canonical_system.rows() {
            return Err(invalid(
                "MINI Stokes solution shape differs from its finalized system",
            ));
        }
        let residual_target = solved.report().residual_target();
        let (algebraic_values, solve_report) = solved.into_parts();
        let (vertex_values, cell_bubble_values, pressure_values, gauge_multiplier) = self
            .layout
            .reconstruct(&algebraic_values, &self.fixed_velocity)?;
        let mut full_values = vec![0.0; self.layout.full_size];
        self.layout.fill_full_values(
            &mut full_values,
            &vertex_values,
            &cell_bubble_values,
            &pressure_values,
            gauge_multiplier,
        );
        let mut residual = self.full_system.matrix().multiply(&full_values)?;
        for (value, rhs) in residual.iter_mut().zip(self.full_system.rhs()) {
            *value -= rhs;
        }
        let mut boundary_reaction = [0.0; COMPONENTS];
        for (vertex, value) in self.fixed_velocity.iter().enumerate() {
            if value.is_some() {
                for component in 0..COMPONENTS {
                    boundary_reaction[component] +=
                        residual[self.layout.full_vertex_velocity(vertex, component)];
                }
            }
        }
        let named_boundary_reactions = self
            .named_reaction_vertices
            .iter()
            .map(|(name, vertices)| {
                let mut reaction = [0.0; COMPONENTS];
                for vertex in vertices {
                    for (component, value) in reaction.iter_mut().enumerate() {
                        *value += residual[self.layout.full_vertex_velocity(*vertex, component)];
                    }
                }
                (name.clone(), reaction)
            })
            .collect::<Vec<_>>();
        let pressure_integral = integrate_pressure(&self.mesh, &self.quadrature, &pressure_values)?;
        let continuity_residual_norm = require_weak_incompressibility(
            &self.full_system,
            &residual,
            &self.layout,
            gauge_multiplier,
            residual_target,
        )?;
        let pressure_reference = match gauge_multiplier {
            Some(multiplier) => {
                require_zero_gauge_multiplier(
                    &self.mesh,
                    &self.quadrature,
                    multiplier,
                    residual_target,
                )?;
                SimplicialMiniStokesPressureReference2d::ZeroIntegral { multiplier }
            }
            None => SimplicialMiniStokesPressureReference2d::BoundaryTraction,
        };
        if boundary_reaction
            .iter()
            .chain(
                named_boundary_reactions
                    .iter()
                    .flat_map(|(_, reaction)| reaction),
            )
            .chain(&self.integrated_body_force)
            .chain(&self.integrated_boundary_traction)
            .chain([&pressure_integral, &continuity_residual_norm])
            .chain(gauge_multiplier.as_ref())
            .any(|value| !value.is_finite())
        {
            return Err(invalid("MINI Stokes evidence is non-finite"));
        }
        if self.full_system.matrix() != self.volume_only_system.matrix() {
            return Err(invalid(
                "MINI Stokes traction packets changed the volume operator matrix",
            ));
        }
        Ok(SimplicialMiniStokesSolution2d {
            velocity: SimplicialMiniVelocityField2d {
                mesh: self.mesh.clone(),
                vertex_values,
                cell_bubble_values,
            },
            pressure: SimplicialP1Field::new(self.mesh, pressure_values)?,
            pressure_reference,
            algebraic_values,
            canonical_system,
            full_system: self.full_system,
            volume_only_system: self.volume_only_system,
            boundary_reaction,
            named_boundary_reactions,
            integrated_body_force: self.integrated_body_force,
            integrated_boundary_traction: self.integrated_boundary_traction,
            pressure_integral,
            continuity_residual_norm,
            assembly_report: self.assembly_report,
            solve_report,
        })
    }
}
