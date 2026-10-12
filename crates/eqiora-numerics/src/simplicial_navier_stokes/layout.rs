//! Method-native Field views over the common exact entity and constraint map.
use std::collections::BTreeMap;

use eqiora_assembly::{AssemblyMap, DofId};
use eqiora_core::{Diagnostic, RawId};
use eqiora_meshing::{MeshEntity, MeshTopology, SimplicialMesh};
use eqiora_solver::AlgebraicConstraint;

use super::form::StepForm;
use super::{COMPONENTS, DIMENSION, invalid};
use crate::region_assembly::mapping::{FieldDof, RegionDofMap};

pub(super) struct FlowLayout {
    mapping: RegionDofMap<f64>,
    vector: RawId,
    scalar: RawId,
    gauge: Option<AlgebraicConstraint>,
    pub(super) vertex_count: usize,
    cell_count: usize,
    pressure_vertices: BTreeMap<usize, usize>,
    velocity_rows: Vec<DofId>,
}

fn key(field: RawId, entity: MeshEntity, component: usize) -> FieldDof {
    FieldDof {
        field,
        entity,
        slot: 0,
        component,
    }
}

impl FlowLayout {
    pub(super) fn new(
        mesh: &SimplicialMesh,
        fixed_velocity: &[Option<[f64; COMPONENTS]>],
        with_gauge: bool,
        form: &StepForm,
    ) -> Result<Self, Diagnostic> {
        let vertex_count = mesh.vertices().len();
        if fixed_velocity.len() != vertex_count {
            return Err(invalid(
                "essential velocity inventory differs from the exact mesh",
            ));
        }
        let cell_count = mesh
            .entity_count(DIMENSION)
            .ok_or_else(|| invalid("flow layout requires its exact cell inventory"))?;
        let vector_scale = form
            .form
            .fields()
            .iter()
            .find(|layout| layout.field == form.vector)
            .ok_or_else(|| invalid("flow layout lacks its exact vector Field"))?
            .scale;
        let prescribed = fixed_velocity
            .iter()
            .enumerate()
            .flat_map(|(vertex, value)| {
                value.iter().flat_map(move |value| {
                    value.iter().enumerate().map(move |(component, value)| {
                        (
                            key(form.vector, MeshEntity::new(0, vertex), component),
                            value * vector_scale,
                        )
                    })
                })
            })
            .collect();
        let gauge = with_gauge.then_some(AlgebraicConstraint::ZeroIntegral {
            field: form
                .scalar
                .downcast()
                .ok_or_else(|| invalid("constraint requires a Field"))?,
        });
        let mapping = RegionDofMap::new(
            mesh,
            &BTreeMap::from([(form.form.domain(), form.form.fields().to_vec())]),
            form.form.reference_cell(),
            &vec![form.form.domain(); cell_count],
            &[],
            &prescribed,
        )?
        .with_algebraic_constraints(&gauge.into_iter().collect::<Vec<_>>())?;
        let pressure_vertices = (0..vertex_count)
            .map(|vertex| {
                mapping
                    .global_dof(key(form.scalar, MeshEntity::new(0, vertex), 0))
                    .map(|index| (index, vertex))
                    .ok_or_else(|| invalid("flow layout omits a pressure coefficient"))
            })
            .collect::<Result<_, _>>()?;
        let velocity_rows = mapping.field_free_dofs(form.vector)?;
        Ok(Self {
            mapping,
            vector: form.vector,
            scalar: form.scalar,
            gauge,
            vertex_count,
            cell_count,
            pressure_vertices,
            velocity_rows,
        })
    }

    pub(super) fn reduced_size(&self) -> usize {
        self.mapping.free_count()
    }
    pub(super) fn full_size(&self) -> usize {
        self.mapping.full_count()
    }
    pub(super) fn velocity_rows(&self) -> &[DofId] {
        &self.velocity_rows
    }
    pub(super) fn pressure_vertex(&self, full: usize) -> Option<usize> {
        self.pressure_vertices.get(&full).copied()
    }
    pub(super) fn full_vertex_velocity(&self, vertex: usize, component: usize) -> usize {
        self.mapping
            .global_dof(key(self.vector, MeshEntity::new(0, vertex), component))
            .expect("exact velocity vertex coordinate")
    }
    pub(super) fn pressure_rows(&self) -> Vec<usize> {
        (0..self.vertex_count)
            .map(|vertex| {
                self.mapping
                    .global_dof(key(self.scalar, MeshEntity::new(0, vertex), 0))
                    .expect("exact pressure vertex coordinate")
            })
            .collect()
    }

    pub(super) fn full_gauge(&self) -> Option<usize> {
        self.gauge
            .and_then(|gauge| self.mapping.constraint_global_dof(gauge))
    }
    pub(super) fn cell_map(
        &self,
        cell: usize,
        reduced: bool,
    ) -> Result<AssemblyMap<f64>, Diagnostic> {
        // The common map owns global numbering; this adapter only retains the
        // existing local action's vector-then-scalar basis order.
        let mut keys = self.mapping.cell_field_keys(cell, self.vector)?;
        keys.extend(self.mapping.cell_field_keys(cell, self.scalar)?);
        self.mapping.map_dofs(&keys, reduced)
    }
    pub(super) fn constraint_map(
        &self,
        cell: usize,
        reduced: bool,
    ) -> Result<AssemblyMap<f64>, Diagnostic> {
        self.mapping.constraint_map(
            cell,
            self.gauge
                .ok_or_else(|| invalid("ungauged flow has no constraint packet"))?,
            reduced,
        )
    }

    pub(super) fn initial_point(
        &self,
        fixed_velocity: &[Option<[f64; COMPONENTS]>],
        vertices: &[[f64; COMPONENTS]],
        bubbles: &[[f64; COMPONENTS]],
        pressure: &[f64],
        multiplier: Option<f64>,
    ) -> Result<Vec<f64>, Diagnostic> {
        if fixed_velocity.len() != self.vertex_count
            || vertices.len() != self.vertex_count
            || bubbles.len() != self.cell_count
            || pressure.len() != self.vertex_count
            || multiplier.is_some() != self.gauge.is_some()
        {
            return Err(invalid(
                "initial Fields differ from their exact flow layout",
            ));
        }
        let mut point = vec![0.0; self.reduced_size()];
        for (vertex, values) in vertices.iter().enumerate() {
            for (component, &value) in values.iter().enumerate() {
                if let Some(dof) =
                    self.mapping
                        .free_dof(key(self.vector, MeshEntity::new(0, vertex), component))
                {
                    point[dof.index()] = value;
                } else {
                    let prescribed = fixed_velocity[vertex]
                        .ok_or_else(|| invalid("fixed coordinate lacks essential velocity"))?
                        [component];
                    let tolerance = 4096.0 * f64::EPSILON * (1.0 + prescribed.abs());
                    if (value - prescribed).abs() > tolerance {
                        return Err(invalid(
                            "MINI initial velocity violates an essential boundary value",
                        ));
                    }
                }
            }
        }
        for (cell, values) in bubbles.iter().enumerate() {
            for (component, &value) in values.iter().enumerate() {
                point[self
                    .mapping
                    .free_dof(key(
                        self.vector,
                        MeshEntity::new(DIMENSION, cell),
                        component,
                    ))
                    .ok_or_else(|| invalid("cell coefficient lacks its exact free coordinate"))?
                    .index()] = value;
            }
        }
        for (vertex, &value) in pressure.iter().enumerate() {
            point[self
                .mapping
                .free_dof(key(self.scalar, MeshEntity::new(0, vertex), 0))
                .ok_or_else(|| invalid("pressure lacks its exact free coordinate"))?
                .index()] = value;
        }
        if let (Some(gauge), Some(value)) = (self.gauge, multiplier) {
            point[self
                .mapping
                .constraint_free_dof(gauge)
                .expect("bound gauge")
                .index()] = value;
        }
        Ok(point)
    }

    pub(super) fn reconstruct(&self, point: &[f64]) -> Result<ReconstructedFields, Diagnostic> {
        let full = self.mapping.lift(point, false)?;
        let value = |field, entity, component| {
            full[self
                .mapping
                .global_dof(key(field, entity, component))
                .expect("bound Field coordinate")]
        };
        let vertices = (0..self.vertex_count)
            .map(|vertex| {
                std::array::from_fn(|component| {
                    value(self.vector, MeshEntity::new(0, vertex), component)
                })
            })
            .collect();
        let bubbles = (0..self.cell_count)
            .map(|cell| {
                std::array::from_fn(|component| {
                    value(self.vector, MeshEntity::new(DIMENSION, cell), component)
                })
            })
            .collect();
        let pressure = (0..self.vertex_count)
            .map(|vertex| value(self.scalar, MeshEntity::new(0, vertex), 0))
            .collect();
        Ok((
            vertices,
            bubbles,
            pressure,
            self.full_gauge().map(|index| full[index]),
        ))
    }
}

type ReconstructedFields = (
    Vec<[f64; COMPONENTS]>,
    Vec<[f64; COMPONENTS]>,
    Vec<f64>,
    Option<f64>,
);

#[cfg(test)]
mod tests;
