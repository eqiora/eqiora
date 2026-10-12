//! Reconstruct physical step diagnostics from physical checkpoints, without a solve.
use super::*;
use crate::form_compiler::region::balance::RegionStepBalance;
use crate::region_assembly::mapping::{RegionDofMap, RegionSolveInput};
use eqiora_meshing::{AffineGeometryMap, MeshEntity, MeshGeometry};

/// Reconstructed physical work over one fixed-geometry backward-Euler step.
/// A small work defect alone does not verify the interface constitutive law.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonSpatialStepBalance {
    pub(crate) cells: Vec<(usize, RegionStepBalance)>,
    pub(crate) boundary: Vec<crate::region_assembly::mapping::SpatialBoundaryWork>,
}

impl CommonSpatialStepBalance {
    /// Directed volume terms, with exact Mesh cell index and Domain identity.
    /// Reciprocal cross terms remain separate contributions.
    pub fn volume_terms(
        &self,
    ) -> impl Iterator<Item = (usize, eqiora_core::RawId, &crate::SpatialBalanceTerm)> {
        self.cells.iter().flat_map(|(index, cell)| {
            cell.contributions
                .iter()
                .map(move |term| (*index, cell.domain, term))
        })
    }
    /// Natural, prescribed, and interface contributions in physical units.
    pub fn boundary_work(&self) -> &[crate::SpatialBoundaryWork] {
        &self.boundary
    }

    /// Signed residual work per Domain and physical dimension, without a positivity claim.
    pub fn defects(&self) -> Result<Vec<(eqiora_core::RawId, DynQuantity)>, Diagnostic> {
        let mut totals = BTreeMap::new();
        for (_, cell) in &self.cells {
            for (dimension, work) in cell.volume_work()? {
                *totals
                    .entry((cell.domain, dimension.exponents()))
                    .or_insert(0.) += work;
            }
        }
        for boundary in &self.boundary {
            *totals
                .entry((boundary.domain, boundary.work.dim().exponents()))
                .or_insert(0.) -= boundary.work.value();
        }
        totals
            .into_iter()
            .map(|((domain, dimension), value)| {
                if !value.is_finite() {
                    return Err(invalid("step work accumulation is nonfinite"));
                }
                Ok((
                    domain,
                    DynQuantity::new(
                        value,
                        DimExponents::from_rationals(dimension).expect("retained units"),
                    ),
                ))
            })
            .collect()
    }
}

fn evaluate_step<'mesh>(
    program: &eqiora_sem::KernelProgram,
    mapping: &RegionDofMap<f64>,
    input: &RegionSolveInput<f64>,
    current: &BTreeMap<
        eqiora_core::RawId,
        crate::region_assembly::mapping::RecoveredRegionField<f64>,
    >,
    mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
) -> Result<CommonSpatialStepBalance, Diagnostic> {
    let cells = evaluate_cells(program, mapping, input, current, mesh)?;
    let boundary = mapping.boundary_step_work(mesh, input, current)?;
    let balance = CommonSpatialStepBalance { cells, boundary };
    balance.defects()?;
    Ok(balance)
}

impl CommonSpatialPlan {
    pub(crate) fn spatial_step_balance(
        &self,
        previous: &CommonState,
        current: &CommonState,
    ) -> Result<CommonSpatialStepBalance, Diagnostic> {
        if current.state_space_identity() != self.identity() {
            return Err(invalid(
                "volume balance requires the exact current Plan checkpoint",
            ));
        }
        let (mapping, input) = self.scalar_step_assembly(previous, current.time_s())?;
        if input.geometry_action.is_some() {
            return Err(invalid("volume balance requires fixed geometry"));
        }
        let values = current
            .spatial_values()
            .ok_or_else(|| invalid("volume balance requires linear physical coefficients"))?;
        let current = self.history_fields(&mapping, values)?;
        let evaluate =
            |mesh| evaluate_step(self.admission.program(), &mapping, &input, &current, mesh);
        match self.admission.resources() {
            NativeMeshResources::Cartesian { mesh, .. } => evaluate_step(
                self.admission.program(),
                &mapping,
                &input,
                &current,
                mesh.mesh(),
            ),
            NativeMeshResources::GmshSimplicial { mesh, .. }
            | NativeMeshResources::AdjacentPartitionSimplicial { mesh, .. } => {
                evaluate(mesh.mesh())
            }
            _ => Err(invalid("volume balance requires the exact affine mesh")),
        }
    }
}

fn evaluate_cells<'mesh>(
    program: &eqiora_sem::KernelProgram,
    mapping: &RegionDofMap<f64>,
    input: &RegionSolveInput<f64>,
    current: &BTreeMap<
        eqiora_core::RawId,
        crate::region_assembly::mapping::RecoveredRegionField<f64>,
    >,
    mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
) -> Result<Vec<(usize, RegionStepBalance)>, Diagnostic> {
    let previous = input
        .previous
        .as_ref()
        .ok_or_else(|| invalid("volume balance requires previous physical coefficients"))?;
    let forms = input
        .forms
        .iter()
        .map(|(form, rule)| (form.domain(), (form, rule)))
        .collect::<BTreeMap<_, _>>();
    mapping
        .cell_domains()
        .iter()
        .enumerate()
        .map(|(index, domain)| {
            let (form, quadrature) = forms
                .get(domain)
                .ok_or_else(|| invalid("volume balance is missing an owned region form"))?;
            let mut layouts = form
                .fields()
                .iter()
                .map(|field| (field.field, field.field))
                .collect::<BTreeMap<_, _>>();
            if let Some(time) = form.time_binding() {
                for state in &time.states {
                    layouts.insert(state.pair().state().erase(), state.pair().rate().erase());
                }
            }
            let gather = |fields| {
                layouts
                    .iter()
                    .map(|(&field, &layout)| {
                        mapping
                            .physical_cell_coefficients(index, field, layout, fields)
                            .map(|values| (field, values))
                    })
                    .collect::<Result<BTreeMap<_, _>, Diagnostic>>()
            };
            let geometry = mesh
                .geometry_map(MeshEntity::new(mesh.topological_dimension(), index))
                .ok_or_else(|| invalid("volume balance is missing its exact cell geometry"))?;
            let balance = form.volume_step_balance(
                program,
                &geometry,
                quadrature,
                &gather(previous)?,
                &gather(current)?,
            )?;
            // Validate finite accumulation separately for every physical dimension.
            balance.volume_work()?;
            Ok((index, balance))
        })
        .collect()
}
