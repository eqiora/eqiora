//! Method-neutral recognition of conservative transient-flow FSI meaning.

use eqiora_core::{Diagnostic, OntologyId};
use eqiora_schema::Model;
use eqiora_sem::KernelProgram;

use crate::canonical_elasticity::{
    IsotropicElastodynamicsCartesianModel, lower_isotropic_elastodynamics_subdomain,
};
use crate::canonical_stokes::{
    TransientIncompressibleNavierStokesCartesianModel,
    lower_transient_incompressible_navier_stokes_subdomain,
};

use super::{
    FsiInterface, FsiInterfaceSide, cartesian_boxes, model_lowering_error,
    reject_uninterpreted_live_relation_sets, require_closed_fsi_model_parts,
    require_coincident_bounds, require_exact_interface, unique_live_side,
};

/// Exact conservative transient-fluid/dynamic-solid semantic network.
///
/// ALE maps, mesh motion, configuration selection, time integration,
/// nonlinear solution, and execution remain Realization choices. This type
/// retains only the ordinary physical Relations and conserving Connection
/// needed by those choices.
#[derive(Debug, Clone, PartialEq)]
pub struct AleFsiCartesianModel<const D: usize> {
    pub(super) program: KernelProgram,
    model: OntologyId<Model>,
    semantic_revision: u64,
    fluid: TransientIncompressibleNavierStokesCartesianModel<D>,
    solid: IsotropicElastodynamicsCartesianModel<D>,
    interface: FsiInterface,
}

impl<const D: usize> AleFsiCartesianModel<D> {
    /// Exact Semantic Model identity from which this closed projection was lowered.
    #[must_use]
    pub const fn model(&self) -> OntologyId<Model> {
        self.model
    }

    /// Exact Semantic Model revision used during closed lowering.
    #[must_use]
    pub const fn semantic_revision(&self) -> u64 {
        self.semantic_revision
    }

    /// Conservative transient incompressible-fluid meaning.
    #[must_use]
    pub const fn fluid(&self) -> &TransientIncompressibleNavierStokesCartesianModel<D> {
        &self.fluid
    }

    /// Dynamic small-strain solid meaning.
    #[must_use]
    pub const fn solid(&self) -> &IsotropicElastodynamicsCartesianModel<D> {
        &self.solid
    }

    /// Exact ordinary conserving mechanical interface.
    #[must_use]
    pub const fn interface(&self) -> FsiInterface {
        self.interface
    }
}

/// Lower one complete 2D conservative transient-flow FSI network.
///
/// The function reuses the same transient Navier--Stokes and dynamic-solid
/// recognizers used by their standalone projections. It adds no ALE semantic
/// node and accepts no mesh or numerical policy.
///
/// # Errors
/// Returns `EQ0703` when the typed subdomain assignment, live interface, or
/// whole-model closure is not unique and exact.
pub fn lower_ale_fsi_cartesian_2d(
    program: &KernelProgram,
) -> Result<AleFsiCartesianModel<2>, Diagnostic> {
    lower_ale_fsi_cartesian::<2>(program)
}

/// Lower one complete 3D conservative transient-flow FSI network.
///
/// # Errors
/// Returns `EQ0703` when the three-dimensional typed subdomain assignment,
/// six-side boundary closure, interface, or whole-model closure is not exact.
pub fn lower_ale_fsi_cartesian_3d(
    program: &KernelProgram,
) -> Result<AleFsiCartesianModel<3>, Diagnostic> {
    lower_ale_fsi_cartesian::<3>(program)
}

fn lower_ale_fsi_cartesian<const D: usize>(
    program: &KernelProgram,
) -> Result<AleFsiCartesianModel<D>, Diagnostic> {
    let boxes = cartesian_boxes::<D>(program)?;
    if boxes.len() != 2 {
        return Err(model_lowering_error(
            program,
            format!(
                "ALE FSI requires exactly two Cartesian boxes, found {}",
                boxes.len()
            ),
        ));
    }

    let mut candidates = Vec::new();
    for fluid_index in 0..2 {
        let solid_index = 1 - fluid_index;
        let fluid = lower_transient_incompressible_navier_stokes_subdomain::<D>(
            program,
            boxes[fluid_index].0,
            boxes[fluid_index].1,
        );
        let solid = lower_isotropic_elastodynamics_subdomain::<D>(
            program,
            boxes[solid_index].0,
            boxes[solid_index].1,
            None,
        );
        if let (Ok(fluid), Ok(solid)) = (fluid, solid) {
            candidates.push((fluid, solid));
        }
    }
    if candidates.len() != 1 {
        return Err(model_lowering_error(
            program,
            format!(
                "ALE FSI requires one unique conservative transient-fluid/dynamic-solid Domain assignment, found {}",
                candidates.len()
            ),
        ));
    }
    let (fluid, solid) = candidates
        .pop()
        .expect("one unique typed ALE FSI assignment was established");

    reject_uninterpreted_live_relation_sets(&fluid.boundary, &solid)?;
    let fluid_side = unique_live_side(fluid.model.boundary_inventory(), "fluid")?;
    let solid_side = unique_live_side(solid.model.continuum().boundary_inventory(), "solid")?;
    require_exact_interface(program, fluid_side, solid_side)?;
    require_coincident_bounds(
        fluid.model.bounds(),
        solid.model.continuum().bounds(),
        fluid_side,
        solid_side,
    )?;
    require_closed_fsi_model_parts(
        program,
        fluid.model.domain(),
        [
            fluid.model.velocity(),
            fluid.model.pressure(),
            fluid.model.force_potential(),
        ],
        fluid.representation,
        &fluid.volume_relations,
        &fluid.boundary,
        &solid,
        "ALE FSI",
    )?;

    let mut endpoints = [
        FsiInterfaceSide {
            domain: fluid.model.domain(),
            field: fluid.model.velocity(),
            boundary: fluid_side.boundary,
            port: fluid_side.port,
            side: fluid_side.side,
        },
        FsiInterfaceSide {
            domain: solid.model.continuum().domain(),
            field: solid.model.velocity(),
            boundary: solid_side.boundary,
            port: solid_side.port,
            side: solid_side.side,
        },
    ];
    endpoints.sort_by_key(|endpoint| endpoint.domain);
    Ok(AleFsiCartesianModel {
        program: program.clone(),
        model: program.model(),
        semantic_revision: program.revision().0,
        fluid: fluid.model,
        solid: solid.model,
        interface: FsiInterface {
            connection: fluid_side.connection,
            axis: fluid_side.axis,
            endpoints,
        },
    })
}
