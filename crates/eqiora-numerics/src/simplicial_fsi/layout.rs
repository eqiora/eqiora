//! FSI result projection over the common exact Field/entity map.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use eqiora_assembly::{AssemblyMap, DofId};
use eqiora_core::{Diagnostic, RawId};
use eqiora_meshing::{MeshEntity, SimplicialMesh, VertexId};

use super::contract::FixedReferenceFsiBoundary;
use super::invalid;
use super::partition::FixedReferenceFsiPartition;
use crate::region_assembly::mapping::{FieldDof, RegionDofMap};

mod binding;
mod roles;
use roles::FsiRoles;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FsiLayout<const D: usize = 2> {
    reference: Arc<SimplicialMesh>,
    partition: Arc<FixedReferenceFsiPartition<D>>,
    boundary: Arc<FixedReferenceFsiBoundary<D>>,
    mapping: RegionDofMap,
    roles: FsiRoles,
    vertex_keys: Vec<[FieldDof; D]>,
    bubble_keys: Vec<[FieldDof; D]>,
    pressure_keys: Vec<FieldDof>,
    pressure_vertices: Vec<VertexId>,
}

type ReconstructedFsiFields<const D: usize> = (Vec<[f64; D]>, Vec<[f64; D]>, Vec<f64>);

fn key(field: RawId, entity: MeshEntity, component: usize) -> FieldDof {
    FieldDof {
        field,
        entity,
        slot: 0,
        component,
    }
}

impl<const D: usize> FsiLayout<D> {
    pub(crate) fn free_field_dof(&self, key: FieldDof) -> Option<eqiora_assembly::DofId> {
        self.mapping.free_dof(key)
    }
    pub(crate) fn partition(&self) -> &FixedReferenceFsiPartition<D> {
        &self.partition
    }
    pub(crate) fn boundary(&self) -> &FixedReferenceFsiBoundary<D> {
        &self.boundary
    }

    pub(crate) fn require_reference(
        &self,
        mesh: &SimplicialMesh,
        partition: &FixedReferenceFsiPartition<D>,
    ) -> Result<(), Diagnostic> {
        if self.reference.as_ref() != mesh || self.partition.as_ref() != partition {
            return Err(invalid(
                "FSI Field map differs from the exact reference mesh or Region partition",
            ));
        }
        Ok(())
    }

    pub(crate) fn require_boundary(
        &self,
        boundary: &FixedReferenceFsiBoundary<D>,
    ) -> Result<(), Diagnostic> {
        if self.with_boundary(boundary)? != *self {
            return Err(invalid(
                "FSI Field map constraints differ from the action boundary",
            ));
        }
        Ok(())
    }

    pub(crate) fn require_scale(
        &self,
        scale: super::FixedReferenceFsiScale<D>,
    ) -> Result<(), Diagnostic> {
        for (field, expected) in [
            self.roles.fluid_velocity,
            self.roles.pressure,
            self.roles.solid_velocity,
        ]
        .into_iter()
        .zip([scale.velocity(), scale.pressure(), scale.velocity()])
        {
            if self.mapping.field_scale(field)? != expected {
                return Err(invalid(
                    "FSI action scale differs from the exact Model/Plan Field map",
                ));
            }
        }
        Ok(())
    }

    /// Project authenticated equation/Plan roles through the common exact map.
    fn new(
        mesh: &SimplicialMesh,
        partition: &FixedReferenceFsiPartition<D>,
        boundary: &FixedReferenceFsiBoundary<D>,
        mapping: &RegionDofMap,
        roles: FsiRoles,
    ) -> Result<Self, Diagnostic> {
        let FsiRoles {
            fluid_velocity,
            pressure,
            solid_velocity,
            ..
        } = roles.clone();
        for (&field, &(domain, space)) in &roles.bindings {
            if mapping
                .field_layout(field)
                .is_none_or(|(owner, layout)| owner != domain || layout.space != space)
            {
                return Err(invalid(
                    "FSI map differs from exact typed role Domain/space binding",
                ));
            }
        }
        let mut expected = BTreeSet::new();
        for (field, vertices) in [
            (fluid_velocity, partition.fluid_vertices()),
            (solid_velocity, partition.solid_vertices()),
        ] {
            expected.extend(vertices.iter().flat_map(|vertex| {
                (0..D)
                    .map(move |component| key(field, MeshEntity::new(0, vertex.index()), component))
            }));
        }
        let bubble_keys = partition
            .fluid_cells()
            .iter()
            .map(|cell| {
                std::array::from_fn(|component| {
                    key(fluid_velocity, MeshEntity::new(D, cell.index()), component)
                })
            })
            .collect::<Vec<_>>();
        expected.extend(bubble_keys.iter().flatten().copied());
        let pressure_vertices = partition.fluid_vertices().to_vec();
        let pressure_keys = pressure_vertices
            .iter()
            .map(|vertex| key(pressure, MeshEntity::new(0, vertex.index()), 0))
            .collect::<Vec<_>>();
        expected.extend(pressure_keys.iter().copied());
        if mapping.keys().collect::<BTreeSet<_>>() != expected {
            return Err(invalid(
                "FSI projection differs from the exact Model Field/space/entity inventory",
            ));
        }
        for vertex in partition.interface_vertices() {
            for component in 0..D {
                let entity = MeshEntity::new(0, vertex.index());
                if mapping.global_dof(key(fluid_velocity, entity, component))
                    != mapping.global_dof(key(solid_velocity, entity, component))
                {
                    return Err(invalid(
                        "FSI interface does not share its admitted trace quotient",
                    ));
                }
            }
        }
        let fluid_vertices = partition
            .fluid_vertices()
            .iter()
            .map(|vertex| vertex.index())
            .collect::<BTreeSet<_>>();
        let vertex_keys = (0..mesh.vertices().len())
            .map(|vertex| {
                let field = if fluid_vertices.contains(&vertex) {
                    fluid_velocity
                } else {
                    solid_velocity
                };
                std::array::from_fn(|component| key(field, MeshEntity::new(0, vertex), component))
            })
            .collect::<Vec<_>>();
        if vertex_keys
            .iter()
            .flatten()
            .any(|key| mapping.global_dof(*key).is_none())
        {
            return Err(invalid(
                "FSI projection contains an unsupported mesh vertex",
            ));
        }
        Self {
            reference: Arc::new(mesh.clone()),
            partition: Arc::new(partition.clone()),
            boundary: Arc::new(boundary.clone()),
            mapping: mapping.clone(),
            roles,
            vertex_keys,
            bubble_keys,
            pressure_keys,
            pressure_vertices,
        }
        .with_boundary(boundary)
    }

    pub(crate) fn with_boundary(
        &self,
        boundary: &FixedReferenceFsiBoundary<D>,
    ) -> Result<Self, Diagnostic> {
        let mapping = &self.mapping;
        let vertex_keys = &self.vertex_keys;
        let mut prescribed = BTreeMap::new();
        if let Some(values) = boundary.prepared_current_quotient() {
            if values.len() != vertex_keys.len() {
                return Err(invalid("prescribed FSI vertices differ from mapped mesh"));
            }
            for (keys, values) in vertex_keys.iter().zip(values) {
                for (key, value) in keys.iter().zip(values) {
                    if let Some(value) = value {
                        prescribed.insert(*key, *value * mapping.field_scale(key.field)?);
                    }
                }
            }
        } else {
            for vertex in boundary.fixed_zero_velocity_vertices() {
                let keys = vertex_keys
                    .get(vertex.index())
                    .ok_or_else(|| invalid("fixed FSI vertex is absent from mapped mesh"))?;
                for key in keys {
                    if prescribed.insert(*key, 0.0).is_some() {
                        return Err(invalid("duplicate fixed FSI vertex"));
                    }
                }
            }
        }
        let mut result = self.clone();
        result.mapping = mapping.with_prescribed(&prescribed)?;
        result.boundary = Arc::new(boundary.clone());
        Ok(result)
    }

    pub(crate) fn cell_map(&self, cell: usize, reduced: bool) -> Result<AssemblyMap, Diagnostic> {
        self.mapping.cell_map(cell, reduced)
    }

    pub(crate) fn fluid_map(
        &self,
        fluid_position: usize,
        vertices: &[MeshEntity],
        reduced: bool,
    ) -> Result<AssemblyMap, Diagnostic> {
        let cell = self
            .partition
            .fluid_cells()
            .get(fluid_position)
            .ok_or_else(|| invalid("fluid position has no exact Region cell"))?;
        if self
            .reference
            .entity_vertices(MeshEntity::new(D, cell.index()))
            .as_deref()
            != Some(vertices)
        {
            return Err(invalid(
                "fluid local vertices differ from the bubble-owning cell closure",
            ));
        }
        let bubbles = self
            .bubble_keys
            .get(fluid_position)
            .ok_or_else(|| invalid("fluid cell has no exact bubble ownership"))?;
        let mut keys = vertices
            .iter()
            .flat_map(|vertex| {
                (0..D).map(move |component| key(self.roles.fluid_velocity, *vertex, component))
            })
            .collect::<Vec<_>>();
        keys.extend_from_slice(bubbles);
        keys.extend(
            vertices
                .iter()
                .map(|vertex| key(self.roles.pressure, *vertex, 0)),
        );
        self.mapping.map_dofs(&keys, reduced)
    }

    pub(crate) fn solid_map(
        &self,
        cell: usize,
        vertices: &[MeshEntity],
        reduced: bool,
    ) -> Result<AssemblyMap, Diagnostic> {
        if cell >= self.partition.cell_count()
            || self.partition.material(cell) != super::partition::CellMaterial::Solid
            || self
                .reference
                .entity_vertices(MeshEntity::new(D, cell))
                .as_deref()
                != Some(vertices)
        {
            return Err(invalid(
                "solid local vertices differ from the exact Region cell closure",
            ));
        }
        let keys = vertices
            .iter()
            .flat_map(|vertex| {
                (0..D).map(move |component| key(self.roles.solid_velocity, *vertex, component))
            })
            .collect::<Vec<_>>();
        self.mapping.map_dofs(&keys, reduced)
    }

    pub(crate) fn full_vertex_velocity(&self, vertex: usize, component: usize) -> usize {
        self.mapping
            .global_dof(self.vertex_keys[vertex][component])
            .expect("validated Field ownership")
    }
    pub(crate) fn reduced_vertex_velocity(&self, vertex: usize, component: usize) -> Option<DofId> {
        self.vertex_keys
            .get(vertex)
            .and_then(|keys| keys.get(component))
            .and_then(|key| self.mapping.free_dof(*key))
    }
    pub(crate) fn reduced_size(&self) -> usize {
        self.mapping.free_count()
    }
    pub(crate) fn full_size(&self) -> usize {
        self.mapping.full_count()
    }
    pub(crate) fn pressure_vertices(&self) -> &[VertexId] {
        &self.pressure_vertices
    }
    pub(crate) fn reduced_pressure_dofs(&self) -> Vec<usize> {
        self.mapping
            .field_free_dofs(self.roles.pressure)
            .expect("validated pressure Field")
            .into_iter()
            .map(DofId::index)
            .collect()
    }
    pub(crate) fn full_pressure_dofs(&self) -> Vec<usize> {
        self.pressure_keys
            .iter()
            .map(|key| {
                self.mapping
                    .global_dof(*key)
                    .expect("validated pressure Field")
            })
            .collect()
    }
    pub(crate) fn fixed_velocity(&self, vertex: usize) -> bool {
        self.vertex_keys[vertex]
            .iter()
            .any(|key| self.mapping.free_dof(*key).is_none())
    }
    pub(crate) fn reconstruct_primal(
        &self,
        values: &[f64],
        fluid_cell_count: usize,
    ) -> Result<ReconstructedFsiFields<D>, Diagnostic> {
        self.split_fields(&self.mapping.lift(values, false)?, fluid_cell_count)
    }
    pub(crate) fn reconstruct_physical(
        &self,
        reduced: &[f64],
    ) -> Result<ReconstructedFsiFields<D>, Diagnostic> {
        let recovered = self.mapping.recover(
            reduced,
            &self.roles.bindings.keys().copied().collect::<Vec<_>>(),
        )?;
        let value = |key: FieldDof| recovered[&key.field].coefficients[&key];
        Ok((
            self.vertex_keys
                .iter()
                .map(|keys| keys.map(value))
                .collect(),
            self.bubble_keys
                .iter()
                .map(|keys| keys.map(value))
                .collect(),
            self.pressure_keys.iter().copied().map(value).collect(),
        ))
    }
    pub(crate) fn reconstruct_direction(
        &self,
        values: &[f64],
        fluid_cell_count: usize,
    ) -> Result<ReconstructedFsiFields<D>, Diagnostic> {
        self.split_fields(&self.mapping.lift(values, true)?, fluid_cell_count)
    }
    fn split_fields(
        &self,
        values: &[f64],
        fluid_cell_count: usize,
    ) -> Result<ReconstructedFsiFields<D>, Diagnostic> {
        if fluid_cell_count != self.bubble_keys.len() {
            return Err(invalid(
                "FSI history differs from exact fluid cell inventory",
            ));
        }
        let value = |key| {
            values[self
                .mapping
                .global_dof(key)
                .expect("validated exact Field DOF")]
        };
        Ok((
            self.vertex_keys
                .iter()
                .map(|keys| keys.map(value))
                .collect(),
            self.bubble_keys
                .iter()
                .map(|keys| keys.map(value))
                .collect(),
            self.pressure_keys.iter().copied().map(value).collect(),
        ))
    }
    pub(crate) fn reduce(
        &self,
        velocity: &[[f64; D]],
        bubbles: &[[f64; D]],
        pressure: &[f64],
    ) -> Result<Vec<f64>, Diagnostic> {
        if velocity.len() != self.vertex_keys.len()
            || bubbles.len() != self.bubble_keys.len()
            || pressure.len() != self.pressure_keys.len()
        {
            return Err(invalid(
                "FSI values differ from exact Field/entity recovery inventory",
            ));
        }
        self.mapping
            .restrict(&self.fill_full(velocity, bubbles, pressure))
    }
    pub(crate) fn fill_full(
        &self,
        velocity: &[[f64; D]],
        bubbles: &[[f64; D]],
        pressure: &[f64],
    ) -> Vec<f64> {
        let mut full = vec![0.0; self.mapping.full_count()];
        for (keys, values) in self
            .vertex_keys
            .iter()
            .zip(velocity)
            .chain(self.bubble_keys.iter().zip(bubbles))
        {
            for (&key, &value) in keys.iter().zip(values) {
                full[self.mapping.global_dof(key).expect("validated Field DOF")] = value;
            }
        }
        for (&key, &value) in self.pressure_keys.iter().zip(pressure) {
            full[self
                .mapping
                .global_dof(key)
                .expect("validated pressure DOF")] = value;
        }
        full
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_assembly::LocalUnknown;
    use eqiora_meshing::{CellId, FacetId, MeshQualityGate, MeshTopology};

    #[test]
    fn shared_constraints_preserve_nonzero_component_values_and_exact_maps() {
        let mesh = SimplicialMesh::new(
            2,
            vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 1.0],
            ],
            vec![vec![0, 1, 2], vec![1, 3, 2]],
            MeshQualityGate::new(0.1).unwrap(),
        )
        .unwrap();
        let interface = (0..mesh.entity_count(1).unwrap())
            .find(|&index| {
                mesh.entity_vertices(MeshEntity::new(1, index))
                    .unwrap()
                    .iter()
                    .all(|vertex| [1, 2].contains(&vertex.index()))
            })
            .unwrap();
        let partition = FixedReferenceFsiPartition::<2>::new(
            &mesh,
            vec![CellId::new(0)],
            vec![CellId::new(1)],
            vec![FacetId::new(interface)],
        )
        .unwrap();
        use crate::simplicial_fsi::{
            FixedReferenceFsiLoad, FixedReferenceFsiMaterial, FixedReferenceFsiScale,
            FixedReferenceFsiStepConfig,
        };
        use eqiora_geometry::{NamedEntitySet, PlanarFace, PlanarRegion};
        let region = PlanarRegion::new(
            vec![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
            vec![
                PlanarFace::new(vec![0, 2, 1], vec![]),
                PlanarFace::new(vec![1, 2, 3], vec![]),
            ],
            vec![
                NamedEntitySet::new("fluid", 2, vec![0]),
                NamedEntitySet::new("solid", 2, vec![1]),
                NamedEntitySet::new("fluid_outer", 1, vec![0, 2]),
                NamedEntitySet::new("solid_outer", 1, vec![4, 5]),
                NamedEntitySet::new("fluid_contact", 1, vec![1]),
                NamedEntitySet::new("solid_contact", 1, vec![3]),
            ],
            1e-12,
        )
        .unwrap();
        let config = FixedReferenceFsiStepConfig::new(
            0.1,
            FixedReferenceFsiMaterial::new(2.0, 0.5, 3.0, 4.0, 2.0).unwrap(),
            FixedReferenceFsiScale::new(1.0, 1.0, 1.0).unwrap(),
            FixedReferenceFsiLoad::Zero,
        )
        .unwrap();
        let solver = eqiora_solver::SolverPlan::new(
            eqiora_solver::LinearSolver::MinimumResidual,
            1e-10,
            1e-12,
            std::num::NonZeroUsize::new(100).unwrap(),
        )
        .unwrap();
        let mut layout = crate::simplicial_fsi::test_model::planar_layout(
            &region,
            &mesh,
            &partition,
            &FixedReferenceFsiBoundary::homogeneous_exterior(&mesh).unwrap(),
            config,
            solver,
            false,
        );
        // This focused constraint-map check supplies physical values on exact
        // authored Field/entity keys; it is not a Model boundary-policy Run.
        layout.mapping = layout
            .mapping
            .with_prescribed(&BTreeMap::from([
                (layout.vertex_keys[0][0], 1.25),
                (layout.vertex_keys[3][1], -2.5),
            ]))
            .unwrap();
        let fluid = [0, 1, 2].map(|index| MeshEntity::new(0, index));
        let solid = [1, 3, 2].map(|index| MeshEntity::new(0, index));
        let fluid_map = layout.fluid_map(0, &fluid, true).unwrap();
        assert_eq!(fluid_map.equations().len(), 11);
        assert_eq!(fluid_map.equations()[0], None);
        assert_eq!(fluid_map.unknowns()[0], LocalUnknown::Fixed(1.25));
        let solid_map = layout.solid_map(1, &solid, true).unwrap();
        assert_eq!(solid_map.equations().len(), 6);
        assert_eq!(solid_map.equations()[3], None);
        assert_eq!(&fluid_map.equations()[2..4], &solid_map.equations()[0..2]);
        assert_eq!(&fluid_map.equations()[4..6], &solid_map.equations()[4..6]);
        assert_eq!(solid_map.unknowns()[3], LocalUnknown::Fixed(-2.5));
        let values = (0..layout.reduced_size())
            .map(|index| index as f64 + 10.0)
            .collect::<Vec<_>>();
        let (velocity, bubbles, pressure) = layout.reconstruct_primal(&values, 1).unwrap();
        assert_eq!((velocity[0][0], velocity[3][1]), (1.25, -2.5));
        assert_eq!(
            layout.reduce(&velocity, &bubbles, &pressure).unwrap(),
            values
        );
        let direction = layout.reconstruct_direction(&values, 1).unwrap().0;
        assert_eq!((direction[0][0], direction[3][1]), (0.0, 0.0));
        assert!(layout.fluid_map(1, &fluid, true).is_err());
        assert!(layout.fluid_map(0, &solid, true).is_err());
        assert!(layout.solid_map(0, &fluid, true).is_err());
        assert!(layout.solid_map(2, &solid, true).is_err());
        layout.require_reference(&mesh, &partition).unwrap();
        let changed = SimplicialMesh::new(
            2,
            mesh.vertices().to_vec(),
            vec![vec![0, 1, 3], vec![0, 3, 2]],
            MeshQualityGate::new(0.1).unwrap(),
        )
        .unwrap();
        assert_eq!(changed.vertices().len(), mesh.vertices().len());
        assert_eq!(changed.cells().len(), mesh.cells().len());
        assert!(layout.require_reference(&changed, &partition).is_err());
        assert!(
            layout
                .require_boundary(&FixedReferenceFsiBoundary::homogeneous_exterior(&mesh).unwrap())
                .is_err()
        );
        layout.require_scale(config.scale()).unwrap();
        assert!(
            layout
                .require_scale(FixedReferenceFsiScale::new(1.0, 1.0, 2.0).unwrap())
                .is_err()
        );
        assert!(
            layout
                .solid_map(1, &[MeshEntity::new(0, 4)], false)
                .is_err()
        );
    }
}
