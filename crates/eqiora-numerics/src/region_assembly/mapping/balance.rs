//! Physical work of authenticated natural, essential, and interface boundaries.
use super::*;
use crate::region_assembly::InterfaceReactions;
use eqiora_meshing::{AffineGeometryMap, MeshGeometry};

/// Source of reconstructed boundary work on the retained spatial mesh.
#[derive(Debug, Clone, PartialEq)]
pub enum SpatialBoundarySource {
    /// Natural boundary packets aggregated for a cell and Field.
    Natural { cell: usize, field: RawId },
    /// Reaction at an essential boundary coefficient.
    Prescribed {
        field: RawId,
        entity: MeshEntity,
        slot: usize,
        component: usize,
    },
    /// One Domain endpoint of a conserving Connection.
    Interface {
        connection: RawId,
        field: RawId,
        entity: MeshEntity,
        slot: usize,
        component: usize,
    },
}

/// Signed boundary work and, where available, its physical weak action.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialBoundaryWork {
    pub(crate) domain: RawId,
    pub(crate) source: SpatialBoundarySource,
    pub(crate) work: DynQuantity,
    /// Physical weak dual at a prescribed or interface coordinate, even at zero trace.
    /// Natural work is aggregated per cell/Field and has no single coordinate action.
    pub(crate) action: Option<DynQuantity>,
}

impl SpatialBoundaryWork {
    /// Domain receiving this signed boundary work.
    pub fn domain(&self) -> RawId {
        self.domain
    }
    /// Exact source and coefficient association on the retained Mesh.
    pub fn source(&self) -> &SpatialBoundarySource {
        &self.source
    }
    /// Time-integrated physical boundary work, with its physical dimension.
    pub fn work(&self) -> DynQuantity {
        self.work
    }
    /// Physical weak action at a prescribed or interface coefficient.
    /// Natural contributions are aggregated per cell and Field and return `None`.
    pub fn action(&self) -> Option<DynQuantity> {
        self.action
    }
}

impl RegionDofMap<f64> {
    pub(crate) fn boundary_step_work<'mesh>(
        &self,
        mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
        input: &RegionSolveInput<f64>,
        current: &BTreeMap<RawId, RecoveredRegionField<f64>>,
    ) -> Result<Vec<SpatialBoundaryWork>, Diagnostic> {
        self.validate_physical(current)?;
        if input.geometry_action.is_some() {
            return Err(invalid("boundary step work requires fixed geometry"));
        }
        let dt = input
            .forms
            .first()
            .and_then(|(form, _)| form.time_step())
            .ok_or_else(|| invalid("boundary step work requires BackwardEuler"))?;
        if input
            .forms
            .iter()
            .any(|(form, _)| form.time_step() != Some(dt))
        {
            return Err(invalid("boundary work forms have different time steps"));
        }
        let forms = input
            .forms
            .iter()
            .map(|(form, _)| (form.domain(), form))
            .collect::<BTreeMap<_, _>>();
        let physical_action = |field: RawId, normalized: f64| {
            let (domain, _) = self
                .field_layout(field)
                .ok_or_else(|| invalid("boundary action has a foreign Field"))?;
            let dual = forms
                .get(&domain)
                .ok_or_else(|| invalid("boundary action lacks its exact form"))?
                .physical_dual_scale(field)?;
            let value = dual.value() * normalized;
            if !value.is_finite() {
                return Err(invalid("boundary action is nonfinite"));
            }
            Ok(DynQuantity::new(value, dual.dim()))
        };
        let quantity = |field: RawId, normalized_work: f64| {
            let (domain, layout) = self
                .field_layout(field)
                .ok_or_else(|| invalid("boundary work has a foreign Field"))?;
            let dual = forms
                .get(&domain)
                .ok_or_else(|| invalid("boundary work lacks its exact form"))?
                .physical_dual_scale(field)?;
            let dim = layout
                .space
                .coefficient_dimension(layout.value_type.dimension())
                .and_then(|dim| dim.mul(dual.dim()))
                .and_then(|dim| {
                    dim.mul(
                        eqiora_core::DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0])
                            .expect("time"),
                    )
                })
                .ok_or_else(|| invalid("boundary work dimension overflow"))?;
            let value = dt * dual.value() * normalized_work;
            if !value.is_finite() {
                return Err(invalid("boundary work is nonfinite"));
            }
            Ok(DynQuantity::new(value, dim))
        };
        let mut result = Vec::new();
        // Natural packets are kept separately from volume loads. Their sign in
        // A*u-b is opposite to the work supplied to the volume equations.
        for (index, facet) in &input.natural {
            let local = facet.affine();
            let domain = *self
                .cell_domains
                .get(*index)
                .ok_or_else(|| invalid("natural work has a foreign cell"))?;
            let form = forms
                .get(&domain)
                .ok_or_else(|| invalid("natural work lacks its exact region"))?;
            let mut physical = vec![0.; local.columns()];
            let mut normalized = vec![0.; local.columns()];
            for layout in form.fields() {
                let coefficients =
                    self.physical_cell_coefficients(*index, layout.field, layout.field, current)?;
                if layout.range.end > physical.len() || coefficients.len() != layout.range.len() {
                    return Err(invalid(
                        "natural work differs from the exact local Field layout",
                    ));
                }
                for (i, value) in layout.range.clone().zip(coefficients) {
                    physical[i] = value;
                    normalized[i] = value / layout.scale;
                }
            }
            if local.rows() != physical.len() {
                return Err(invalid(
                    "natural work requires matching test and trial layouts",
                ));
            }
            for layout in form.fields() {
                let work = layout
                    .range
                    .clone()
                    .map(|row| {
                        let product = local.matrix()
                            [row * local.columns()..(row + 1) * local.columns()]
                            .iter()
                            .zip(&normalized)
                            .map(|(a, u)| a * u)
                            .sum::<f64>();
                        physical[row] * (local.rhs()[row] - product)
                    })
                    .sum();
                result.push(SpatialBoundaryWork {
                    domain,
                    source: SpatialBoundarySource::Natural {
                        cell: *index,
                        field: layout.field,
                    },
                    work: quantity(layout.field, work)?,
                    action: None,
                });
            }
        }
        let assembly = self.prepare_assembly(
            eqiora_assembly::AssemblyPacketSetIdentityV1::Unbound,
            mesh,
            input.forms.clone(),
            input.natural.clone(),
            input.previous.as_ref(),
            None,
        )?;
        let reactions = InterfaceReactions::prepare(
            &assembly.work,
            assembly.plan.target_id(1).expect("full target"),
            self,
            &assembly.packet_domains,
        )?;
        let mut full = vec![0.; self.full_count()];
        for key in self.keys() {
            full[self.global_dof(key).expect("mapped key")] =
                current[&key.field].coefficients[&key] / self.field_scale(key.field)?;
        }
        let recovered = reactions.recover(&full)?;
        for (key, action) in &recovered.constrained_actions {
            result.push(SpatialBoundaryWork {
                domain: current[&key.field].domain,
                source: SpatialBoundarySource::Prescribed {
                    field: key.field,
                    entity: key.entity,
                    slot: key.slot,
                    component: key.component,
                },
                action: Some(physical_action(key.field, *action)?),
                work: quantity(key.field, action * current[&key.field].coefficients[key])?,
            });
        }
        for (quotient, keys) in self.traces() {
            let connection = quotient.source().owner();
            for key in keys.iter().filter(|key| self.free_dof(**key).is_some()) {
                let action = recovered.action(connection, *key)?;
                result.push(SpatialBoundaryWork {
                    domain: current[&key.field].domain,
                    source: SpatialBoundarySource::Interface {
                        connection,
                        field: key.field,
                        entity: key.entity,
                        slot: key.slot,
                        component: key.component,
                    },
                    work: quantity(key.field, action * current[&key.field].coefficients[key])?,
                    action: Some(physical_action(key.field, action)?),
                });
            }
        }
        Ok(result)
    }
}
