//! Bind the existing FSI projection from already authenticated Model/Plan support.

use eqiora_meshing::{MeshTopology, ReferenceCell};
use eqiora_realization::CoupledFieldwiseRealizationPlan;
use eqiora_sem::KernelProgram;
use eqiora_solver::AlgebraicBlock;

use super::*;
use crate::region_assembly::mapping::{TraceBinding, TraceFacet, field_layouts};

impl<const D: usize> FsiLayout<D> {
    pub(crate) fn bind(
        program: &KernelProgram,
        plan: &CoupledFieldwiseRealizationPlan,
        mesh: &SimplicialMesh,
        partition: &FixedReferenceFsiPartition<D>,
        boundary: &FixedReferenceFsiBoundary<D>,
        fields: [RawId; 3],
    ) -> Result<Self, Diagnostic> {
        let scales = plan
            .scaling()
            .block_scales()
            .iter()
            .filter_map(|scale| match scale.block() {
                AlgebraicBlock::Field(field) => Some((field.erase(), scale.scale().quantity())),
                AlgebraicBlock::ConstraintMultiplier { .. } => None,
            })
            .collect();
        let reference = ReferenceCell::simplex(D)?;
        let layouts = field_layouts(program, plan.spatial().domains(), reference, &scales)?;
        let domain = |field| {
            plan.spatial()
                .domains()
                .iter()
                .find(|domain| {
                    domain
                        .field_spaces()
                        .iter()
                        .any(|binding| binding.field().erase() == field)
                })
                .map(|domain| domain.domain().erase())
                .ok_or_else(|| invalid("FSI role has no exact Plan Domain"))
        };
        let mut domains = vec![None; partition.cell_count()];
        for (domain, cells) in [
            (domain(fields[0])?, partition.fluid_cells()),
            (domain(fields[2])?, partition.solid_cells()),
        ] {
            for cell in cells {
                domains[cell.index()] = Some(domain);
            }
        }
        let domains = domains
            .into_iter()
            .map(|domain| domain.ok_or_else(|| invalid("FSI partition omits a Model Domain cell")))
            .collect::<Result<Vec<_>, _>>()?;
        let traces = plan
            .spatial()
            .trace_quotients()
            .iter()
            .map(|&quotient| {
                let facets = partition
                    .interface_facets()
                    .iter()
                    .map(|facet| {
                        let facet = MeshEntity::new(D - 1, facet.index());
                        let incidence = mesh
                            .incidence(facet, D)
                            .ok_or_else(|| invalid("interface has no exact mesh incidence"))?;
                        let sides = quotient
                            .endpoints()
                            .iter()
                            .map(|endpoint| {
                                incidence
                                    .iter()
                                    .copied()
                                    .find(|side| {
                                        domains[side.entity.index()] == endpoint.domain().erase()
                                    })
                                    .ok_or_else(|| {
                                        invalid("interface has no exact Domain incidence")
                                    })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        Ok(TraceFacet {
                            facet,
                            sides: sides.try_into().expect("two quotient endpoints"),
                        })
                    })
                    .collect::<Result<Vec<_>, Diagnostic>>()?;
                Ok(TraceBinding { quotient, facets })
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let mapping = RegionDofMap::new(
            mesh,
            &layouts,
            reference,
            &domains,
            &traces,
            &BTreeMap::new(),
        )?;
        Self::new(mesh, partition, boundary, &mapping, fields)
    }
}
