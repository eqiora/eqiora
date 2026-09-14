//! Field-major Cartesian assembly of checked linear scalar equations.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use eqiora_assembly::{
    AssemblyBackend, AssemblyPacket, AssemblyPacketSetIdentityV1, AssemblyPlan, AssemblyReport,
    AssemblyTarget, LinearSystem, TargetAssemblyMap,
};
use eqiora_core::{Diagnostic, RawId, ValueType};
use eqiora_meshing::{CartesianMesh, MeshEntity, MeshGeometry, MeshTopology, QuadratureRule};
use eqiora_schema::kernel::BoundarySide;
use eqiora_solver::{CanonicalCsrSystemView, LinearOperatorProperties};

use super::CartesianBoundaryValue;
use crate::canonical_boundary::PhysicalBoundaryQuantity;
use crate::constrained_dofs::ConstrainedDofLayout;
use crate::form_compiler::linear::CompiledLinearBlockForm;
use crate::region_assembly::{PreparedRegionAssembly, RegionAssemblyCell};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CartesianLinearAssembly {
    pub(crate) fields: Vec<(RawId, ValueType)>,
    pub(crate) mesh: CartesianMesh,
    pub(crate) constraints: ConstrainedDofLayout,
    pub(crate) system: LinearSystem,
    pub(crate) full_system: LinearSystem,
    pub(crate) report: AssemblyReport,
    // Physical steady source only: transient RHS also contains history.
    source_integrals: Option<Vec<f64>>,
}

#[cfg(test)]
mod tests;

impl CartesianLinearAssembly {
    pub(crate) fn into_single_field_canonical(
        self,
    ) -> Result<
        (
            Arc<CanonicalCsrSystemView>,
            super::FinalizedCartesianFemState,
        ),
        Diagnostic,
    > {
        if self.source_integrals.is_none() {
            return Err(super::invalid(
                "transient assembly cannot produce stationary flux diagnostics",
            ));
        }
        if self.fields.len() != 1 {
            return Err(super::invalid(
                "scalar differentiation requires exactly one Field",
            ));
        }
        let canonical = Arc::new(CanonicalCsrSystemView::new(
            &self.system,
            LinearOperatorProperties::General,
        )?);
        Ok((
            canonical,
            super::FinalizedCartesianFemState {
                mesh: self.mesh,
                constrained_dofs: self.constraints,
                full_system: self.full_system,
                integrated_source: self.source_integrals.expect("checked stationary source")[0],
                assembly_report: self.report,
            },
        ))
    }

    pub(crate) fn assemble(
        form: &CompiledLinearBlockForm,
        mesh: &CartesianMesh,
        quadrature: &QuadratureRule,
        backend: &dyn AssemblyBackend,
        boundaries: &BTreeMap<(usize, BoundarySide), RawId>,
    ) -> Result<Self, Diagnostic> {
        Self::assemble_inner(form, mesh, quadrature, backend, boundaries, None)
    }

    pub(crate) fn assemble_backward_euler(
        form: &CompiledLinearBlockForm,
        mesh: &CartesianMesh,
        quadrature: &QuadratureRule,
        backend: &dyn AssemblyBackend,
        boundaries: &BTreeMap<(usize, BoundarySide), RawId>,
        previous: &[f64],
    ) -> Result<Self, Diagnostic> {
        if !form.is_transient() {
            return Err(super::invalid(
                "Backward Euler assembly requires scalar storage",
            ));
        }
        Self::assemble_inner(form, mesh, quadrature, backend, boundaries, Some(previous))
    }

    fn assemble_inner(
        form: &CompiledLinearBlockForm,
        mesh: &CartesianMesh,
        quadrature: &QuadratureRule,
        backend: &dyn AssemblyBackend,
        boundaries: &BTreeMap<(usize, BoundarySide), RawId>,
        previous: Option<&[f64]>,
    ) -> Result<Self, Diagnostic> {
        let volume = form.volume()?;
        if form.is_transient() != previous.is_some() {
            return Err(super::invalid(
                "scalar storage requires explicit previous state",
            ));
        }
        super::validate_problem(mesh, quadrature)?;
        let dimension = mesh.topological_dimension();
        if form.dimension() != dimension {
            return Err(super::invalid("linear block and Mesh dimensions differ"));
        }
        let domains = boundaries.values().copied().collect::<BTreeSet<_>>();
        if domains.len() != 2 * dimension
            || boundaries.len() != domains.len()
            || (0..dimension).any(|axis| {
                [BoundarySide::Lower, BoundarySide::Upper]
                    .iter()
                    .any(|side| !boundaries.contains_key(&(axis, *side)))
            })
            || form
                .boundary_laws()
                .values()
                .any(|laws| laws.keys().copied().collect::<BTreeSet<_>>() != domains)
        {
            return Err(super::invalid(
                "linear boundary laws differ from the exact Cartesian support mapping",
            ));
        }
        let vertices = mesh.entity_count(0).expect("Cartesian vertices");
        let count = form
            .fields()
            .len()
            .checked_mul(vertices)
            .ok_or_else(|| super::invalid("linear block DOF count overflows usize"))?;
        if previous.is_some_and(|values| {
            values.len() != count || values.iter().any(|value| !value.is_finite())
        }) {
            return Err(super::invalid(
                "scalar previous state requires exact full Field-major finite coefficients",
            ));
        }
        let mut fixed = Vec::new();
        fixed
            .try_reserve_exact(count)
            .map_err(|_| super::invalid("linear block DOF allocation exceeds capacity"))?;
        let mut natural = Vec::new();
        let mut natural_integrals = vec![0.0; form.fields().len()];
        for (index, (field, _)) in form.fields().iter().enumerate() {
            let boundary = |axis, side, coordinates: &[f64]| {
                let law = &form.boundary_laws()[field][&boundaries[&(axis, side)]];
                let value = law
                    .evaluate(coordinates, &[])
                    .map(|value| value[0])
                    .unwrap_or(f64::NAN);
                match law.quantity {
                    PhysicalBoundaryQuantity::Trace => CartesianBoundaryValue::Essential(value),
                    PhysicalBoundaryQuantity::Flux => CartesianBoundaryValue::Natural(value),
                }
            };
            fixed.extend(super::essential_fem_values(mesh, &boundary)?);
            let facet_rule = super::scalar_facet_quadrature(dimension)?;
            for facet_index in 0..mesh.entity_count(dimension - 1).expect("Cartesian facets") {
                let facet = MeshEntity::new(dimension - 1, facet_index);
                let Some((axis, side)) = super::cartesian_boundary_facet_side(mesh, facet)? else {
                    continue;
                };
                let facet_geometry = mesh.geometry_map(facet).expect("Cartesian facet geometry");
                if !matches!(
                    boundary(axis, side, facet_geometry.origin()),
                    CartesianBoundaryValue::Natural(_)
                ) {
                    continue;
                }
                let incident = mesh
                    .incidence(facet, dimension)
                    .expect("Cartesian facet incidence");
                let [cell] = incident.as_slice() else {
                    return Err(super::invalid(
                        "natural exterior facet requires one exact parent cell",
                    ));
                };
                let cell_geometry = mesh
                    .geometry_map(cell.entity)
                    .expect("Cartesian cell geometry");
                let cell_vertices = mesh
                    .entity_vertices(cell.entity)
                    .expect("Cartesian cell vertices");
                let facet_vertices = mesh
                    .entity_vertices(facet)
                    .expect("Cartesian facet vertices");
                let parent_vertices = facet_vertices
                    .iter()
                    .map(|vertex| {
                        cell_vertices
                            .iter()
                            .position(|parent| parent == vertex)
                            .ok_or_else(|| {
                                super::invalid("natural facet vertex is absent from its parent")
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let local = volume.evaluate_natural_facet(
                    *field,
                    &cell_geometry,
                    (&facet_geometry, *cell, &parent_vertices),
                    &facet_rule,
                    |point, _normal| match boundary(axis, side, point) {
                        CartesianBoundaryValue::Natural(value) => Ok(vec![value]),
                        CartesianBoundaryValue::Essential(_) => {
                            Err(super::invalid("natural facet changed boundary disposition"))
                        }
                    },
                )?;
                natural_integrals[index] += local.rhs().iter().sum::<f64>();
                let globals = (0..form.fields().len())
                    .flat_map(|field| {
                        cell_vertices
                            .iter()
                            .map(move |vertex| field * vertices + vertex.index())
                    })
                    .collect::<Vec<_>>();
                natural.push((local, globals));
            }
        }
        let constraints = ConstrainedDofLayout::new(fixed)?;
        if constraints.free_count() == 0 {
            return Err(super::invalid("linear block requires unconstrained DOFs"));
        }
        let cells = mesh.entity_count(dimension).expect("Cartesian cells");
        let plan = AssemblyPlan::new(vec![
            AssemblyTarget::new(constraints.free_count())?,
            AssemblyTarget::new(count)?,
        ])?;
        let maps = |globals: &[usize]| -> Result<Vec<TargetAssemblyMap>, Diagnostic> {
            Ok(vec![
                TargetAssemblyMap::new(
                    plan.target_id(0).expect("reduced target"),
                    constraints.reduced_map(globals)?,
                ),
                TargetAssemblyMap::new(
                    plan.target_id(1).expect("full target"),
                    constraints.full_map(globals)?,
                ),
            ])
        };
        let mut region_cells = Vec::new();
        region_cells
            .try_reserve_exact(cells)
            .map_err(|_| super::invalid("linear cell packet allocation exceeds capacity"))?;
        for index in 0..cells {
            let cell = MeshEntity::new(dimension, index);
            let geometry = mesh.geometry_map(cell).expect("Cartesian cell geometry");
            let cell_vertices = mesh.entity_vertices(cell).expect("Cartesian cell vertices");
            let local_count = form
                .fields()
                .len()
                .checked_mul(cell_vertices.len())
                .ok_or_else(|| super::invalid("linear block local DOF count overflows usize"))?;
            let mut globals = Vec::new();
            globals.try_reserve_exact(local_count).map_err(|_| {
                super::invalid("linear block local DOF allocation exceeds capacity")
            })?;
            for field in 0..form.fields().len() {
                globals.extend(
                    cell_vertices
                        .iter()
                        .map(|vertex| field * vertices + vertex.index()),
                );
            }
            region_cells.push(RegionAssemblyCell {
                index,
                geometry,
                mappings: maps(&globals)?,
                previous: previous
                    .map(|values| {
                        form.fields()
                            .iter()
                            .enumerate()
                            .map(|(field_index, (field, _))| {
                                (
                                    *field,
                                    cell_vertices
                                        .iter()
                                        .map(|vertex| {
                                            values[field_index * vertices + vertex.index()]
                                        })
                                        .collect(),
                                )
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
            });
        }
        let boundary_packets = natural
            .into_iter()
            .map(|(local, globals)| AssemblyPacket::new(local, maps(&globals)?))
            .collect::<Result<Vec<_>, _>>()?;
        let work = PreparedRegionAssembly::new(
            AssemblyPacketSetIdentityV1::Unbound,
            &plan,
            vec![(volume.clone(), quadrature.clone())],
            &vec![volume.domain(); cells],
            region_cells,
            boundary_packets,
        )?;
        let (systems, report) = backend.assemble(&plan, &work)?.into_parts();
        let mut systems = systems.into_iter();
        let system = systems.next().expect("validated reduced assembly target");
        let full_system = systems.next().expect("validated full assembly target");
        debug_assert!(systems.next().is_none());
        let source_integrals = full_system
            .rhs()
            .chunks_exact(vertices)
            .zip(natural_integrals)
            .map(|(rhs, natural)| rhs.iter().sum::<f64>() - natural)
            .collect::<Vec<_>>();
        if source_integrals.iter().any(|value| !value.is_finite()) {
            return Err(super::invalid("linear source integral is non-finite"));
        }
        Ok(Self {
            fields: form.fields().to_vec(),
            mesh: mesh.clone(),
            constraints,
            system,
            full_system,
            report,
            source_integrals: (!form.is_transient()).then_some(source_integrals),
        })
    }
}
