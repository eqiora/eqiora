//! Projection through the sole exact Field/entity and Region topology map.
use super::contract::FixedReferenceFsiBoundary;
use super::invalid;
use super::partition::FixedReferenceFsiPartition;
use crate::region_assembly::mapping::{FieldDof, RegionDofMap};
use eqiora_assembly::{AssemblyMap, DofId};
use eqiora_core::{Diagnostic, RawId};
use eqiora_meshing::{CellId, MeshEntity, SimplicialMesh};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
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
    time_step: eqiora_realization::BackwardEulerStep,
}

fn key(field: RawId, entity: MeshEntity, component: usize) -> FieldDof {
    FieldDof {
        field,
        entity,
        slot: 0,
        component,
    }
}

impl<const D: usize> FsiLayout<D> {
    pub(crate) fn cell_domain(&self, cell: usize) -> Result<RawId, Diagnostic> {
        self.mapping
            .cell_domains()
            .get(cell)
            .copied()
            .ok_or_else(|| invalid("cell is outside the exact Domain inventory"))
    }
    pub(crate) fn velocity_field(&self, domain: RawId) -> Result<RawId, Diagnostic> {
        self.roles
            .velocities
            .get(&domain)
            .copied()
            .ok_or_else(|| invalid("Domain has no exact velocity execution witness"))
    }
    pub(crate) fn pressure_field(&self, domain: RawId) -> Option<RawId> {
        self.roles
            .constraints
            .keys()
            .copied()
            .find(|field| self.roles.bindings[field].0 == domain)
    }
    pub(crate) fn state_field(&self, domain: RawId) -> Option<RawId> {
        self.time_step
            .eliminated_states()
            .iter()
            .find(|state| self.roles.bindings[&state.pair().rate().erase()].0 == domain)
            .map(|state| state.pair().state().erase())
    }
    pub(crate) fn state_bindings(&self) -> &[eqiora_realization::BackwardEulerStateBinding] {
        self.time_step.eliminated_states()
    }
    pub(crate) fn time_step(&self) -> &eqiora_realization::BackwardEulerStep {
        &self.time_step
    }
    pub(crate) fn state_rate(&self, state: RawId) -> Result<RawId, Diagnostic> {
        self.state_bindings()
            .iter()
            .find(|binding| binding.pair().state().erase() == state)
            .map(|binding| binding.pair().rate().erase())
            .ok_or_else(|| invalid("state has no exact Relation-bound rate"))
    }
    pub(crate) fn require_material(
        &self,
        config: &super::FixedReferenceFsiStepConfig<D>,
    ) -> Result<(), Diagnostic> {
        let material = config.material();
        let densities = self
            .roles
            .velocities
            .iter()
            .map(|(&domain, &field)| (field, domain))
            .collect::<BTreeMap<_, _>>();
        let viscosities = self
            .roles
            .constraints
            .values()
            .map(|&field| (field, self.roles.bindings[&field].0))
            .collect::<BTreeMap<_, _>>();
        let elasticities = self
            .state_bindings()
            .iter()
            .map(|binding| {
                (
                    binding.pair().state().erase(),
                    self.roles.bindings[&binding.pair().rate().erase()].0,
                )
            })
            .collect::<BTreeMap<_, _>>();
        if material
            .densities
            .iter()
            .map(|(&field, &(domain, _))| (field, domain))
            .collect::<BTreeMap<_, _>>()
            != densities
            || material
                .viscosities
                .iter()
                .map(|(&field, &(domain, _))| (field, domain))
                .collect::<BTreeMap<_, _>>()
                != viscosities
            || material
                .elasticities
                .iter()
                .map(|(&field, &(domain, _))| (field, domain))
                .collect::<BTreeMap<_, _>>()
                != elasticities
            || config.time_step() != self.time_step.duration().value()
        {
            return Err(invalid(
                "coefficient or time inventory differs from exact Domain/Field/Relation execution roles",
            ));
        }
        Ok(())
    }
    pub(crate) fn mapping(&self) -> &RegionDofMap {
        &self.mapping
    }
    pub(crate) fn reactions(
        &self,
        work: &dyn eqiora_assembly::AssemblyWork,
        target: eqiora_assembly::AssemblyTargetId,
    ) -> Result<crate::region_assembly::InterfaceReactions, Diagnostic> {
        crate::region_assembly::InterfaceReactions::prepare(
            work,
            target,
            &self.mapping,
            self.mapping.cell_domains(),
        )
    }
    pub(crate) fn free_field_dof(&self, key: FieldDof) -> Option<DofId> {
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
                "Field map differs from exact reference mesh or Region partition",
            ));
        }
        Ok(())
    }
    pub(crate) fn require_boundary(
        &self,
        boundary: &FixedReferenceFsiBoundary<D>,
    ) -> Result<(), Diagnostic> {
        if self.with_boundary(boundary)? != *self {
            return Err(invalid("Field map constraints differ from action boundary"));
        }
        Ok(())
    }
    pub(crate) fn require_scale(
        &self,
        scale: super::FixedReferenceFsiScale<D>,
    ) -> Result<(), Diagnostic> {
        for &field in self.roles.bindings.keys() {
            let expected = if self.roles.constraints.contains_key(&field) {
                scale.pressure()
            } else {
                scale.velocity()
            };
            if self.mapping.field_scale(field)? != expected {
                return Err(invalid("action scale differs from exact Plan Field map"));
            }
        }
        Ok(())
    }
    fn new(
        mesh: &SimplicialMesh,
        partition: &FixedReferenceFsiPartition<D>,
        boundary: &FixedReferenceFsiBoundary<D>,
        mapping: &RegionDofMap,
        roles: FsiRoles,
        time_step: eqiora_realization::BackwardEulerStep,
    ) -> Result<Self, Diagnostic> {
        if mapping.cell_domains() != partition.cell_domains()
            || (roles.quotients.len() != partition.quotients().count()
                || roles
                    .quotients
                    .iter()
                    .any(|quotient| !partition.quotients().any(|actual| actual == *quotient)))
        {
            return Err(invalid(
                "Field map differs from exact partition Domain/Connection inventory",
            ));
        }
        for (&field, &(domain, space)) in &roles.bindings {
            if mapping
                .field_layout(field)
                .is_none_or(|(owner, layout)| owner != domain || layout.space != space)
            {
                return Err(invalid(
                    "Field map differs from exact typed Domain/space binding",
                ));
            }
        }
        if mapping.keys().map(|key| key.field).collect::<BTreeSet<_>>()
            != roles.bindings.keys().copied().collect()
        {
            return Err(invalid("Field map omits or adds algebraic Fields"));
        }
        Self {
            reference: Arc::new(mesh.clone()),
            partition: Arc::new(partition.clone()),
            boundary: Arc::new(boundary.clone()),
            mapping: mapping.clone(),
            roles,
            time_step,
        }
        .with_boundary(boundary)
    }
    pub(crate) fn with_boundary(
        &self,
        boundary: &FixedReferenceFsiBoundary<D>,
    ) -> Result<Self, Diagnostic> {
        let mut prescribed = BTreeMap::new();
        let velocity_keys = self
            .mapping
            .keys()
            .filter(|key| {
                key.entity.dimension() == 0
                    && self
                        .roles
                        .velocities
                        .values()
                        .any(|field| *field == key.field)
            })
            .collect::<Vec<_>>();
        if let Some(values) = boundary.prepared_current_quotient() {
            if values.len() != self.reference.vertices().len() {
                return Err(invalid("prescribed vertices differ from exact mesh"));
            }
            for key in velocity_keys {
                if let Some(value) = values[key.entity.index()][key.component] {
                    prescribed.insert(key, value * self.mapping.field_scale(key.field)?);
                }
            }
        } else {
            let fixed = boundary
                .fixed_zero_velocity_vertices()
                .iter()
                .map(|vertex| vertex.index())
                .collect::<BTreeSet<_>>();
            if fixed.len() != boundary.fixed_zero_velocity_vertices().len()
                || fixed
                    .iter()
                    .any(|&vertex| vertex >= self.reference.vertices().len())
            {
                return Err(invalid(
                    "fixed velocity vertex inventory is repeated or stale",
                ));
            }
            for key in velocity_keys {
                if fixed.contains(&key.entity.index()) {
                    prescribed.insert(key, 0.0);
                }
            }
        }
        let mut result = self.clone();
        result.mapping = self.mapping.with_prescribed(&prescribed)?;
        result.boundary = Arc::new(boundary.clone());
        Ok(result)
    }
    pub(crate) fn cell_map(&self, cell: usize, reduced: bool) -> Result<AssemblyMap, Diagnostic> {
        self.mapping.cell_map(cell, reduced)
    }
    pub(crate) fn fluid_map(
        &self,
        cell: CellId,
        vertices: &[MeshEntity],
        reduced: bool,
    ) -> Result<AssemblyMap, Diagnostic> {
        let domain = self.cell_domain(cell.index())?;
        let velocity = self.velocity_field(domain)?;
        let pressure = self
            .pressure_field(domain)
            .ok_or_else(|| invalid("cell has no exact constraint multiplier"))?;
        self.require_cell_vertices(cell.index(), vertices)?;
        let mut keys = vertices
            .iter()
            .flat_map(|vertex| (0..D).map(move |component| key(velocity, *vertex, component)))
            .collect::<Vec<_>>();
        keys.extend(
            (0..D).map(|component| key(velocity, MeshEntity::new(D, cell.index()), component)),
        );
        keys.extend(vertices.iter().map(|vertex| key(pressure, *vertex, 0)));
        self.mapping.map_dofs(&keys, reduced)
    }
    pub(crate) fn solid_map(
        &self,
        cell: usize,
        vertices: &[MeshEntity],
        reduced: bool,
    ) -> Result<AssemblyMap, Diagnostic> {
        let domain = self.cell_domain(cell)?;
        if self.state_field(domain).is_none() {
            return Err(invalid("cell has no exact eliminated-state witness"));
        }
        let velocity = self.velocity_field(domain)?;
        self.require_cell_vertices(cell, vertices)?;
        self.mapping.map_dofs(
            &vertices
                .iter()
                .flat_map(|vertex| (0..D).map(move |component| key(velocity, *vertex, component)))
                .collect::<Vec<_>>(),
            reduced,
        )
    }
    fn require_cell_vertices(
        &self,
        cell: usize,
        vertices: &[MeshEntity],
    ) -> Result<(), Diagnostic> {
        if self
            .reference
            .entity_vertices(MeshEntity::new(D, cell))
            .as_deref()
            != Some(vertices)
        {
            return Err(invalid(
                "local vertices differ from exact owning cell closure",
            ));
        }
        Ok(())
    }
    pub(crate) fn full_vertex_velocity(
        &self,
        field: RawId,
        vertex: usize,
        component: usize,
    ) -> usize {
        self.mapping
            .global_dof(key(field, MeshEntity::new(0, vertex), component))
            .expect("exact Field ownership")
    }
    pub(crate) fn reduced_vertex_velocity(
        &self,
        field: RawId,
        vertex: usize,
        component: usize,
    ) -> Option<DofId> {
        self.mapping
            .free_dof(key(field, MeshEntity::new(0, vertex), component))
    }
    pub(crate) fn reduced_size(&self) -> usize {
        self.mapping.free_count()
    }
    pub(crate) fn full_size(&self) -> usize {
        self.mapping.full_count()
    }
    pub(crate) fn reduced_pressure_dofs(&self) -> Vec<usize> {
        self.roles
            .constraints
            .keys()
            .flat_map(|field| {
                self.mapping
                    .field_free_dofs(*field)
                    .expect("exact pressure Field")
            })
            .map(DofId::index)
            .collect()
    }
    pub(crate) fn full_pressure_dofs(&self) -> Vec<usize> {
        self.mapping
            .keys()
            .filter(|key| self.roles.constraints.contains_key(&key.field))
            .map(|key| self.mapping.global_dof(key).expect("exact pressure Field"))
            .collect()
    }
    pub(crate) fn fixed_velocity(&self, field: RawId, vertex: usize) -> bool {
        (0..D).any(|component| {
            self.mapping
                .free_dof(key(field, MeshEntity::new(0, vertex), component))
                .is_none()
        })
    }
    pub(crate) fn reconstruct_primal(
        &self,
        values: &[f64],
    ) -> Result<BTreeMap<FieldDof, f64>, Diagnostic> {
        self.split_fields(&self.mapping.lift(values, false)?)
    }
    pub(crate) fn reconstruct_direction(
        &self,
        values: &[f64],
    ) -> Result<BTreeMap<FieldDof, f64>, Diagnostic> {
        self.split_fields(&self.mapping.lift(values, true)?)
    }
    fn split_fields(&self, values: &[f64]) -> Result<BTreeMap<FieldDof, f64>, Diagnostic> {
        Ok(self
            .mapping
            .keys()
            .map(|key| {
                (
                    key,
                    values[self.mapping.global_dof(key).expect("exact Field DOF")],
                )
            })
            .collect())
    }
    pub(crate) fn reduce(&self, values: &BTreeMap<FieldDof, f64>) -> Result<Vec<f64>, Diagnostic> {
        self.mapping.restrict(&self.fill_full(values)?)
    }
    pub(crate) fn fill_full(
        &self,
        values: &BTreeMap<FieldDof, f64>,
    ) -> Result<Vec<f64>, Diagnostic> {
        if !values.keys().copied().eq(self.mapping.keys()) {
            return Err(invalid("values differ from complete exact Field inventory"));
        }
        let mut full = vec![None; self.mapping.full_count()];
        for (&key, &value) in values {
            let index = self.mapping.global_dof(key).expect("exact Field DOF");
            if !value.is_finite() || full[index].is_some_and(|old| old != value) {
                return Err(invalid(
                    "Field values are nonfinite or disagree on an exact quotient",
                ));
            }
            full[index] = Some(value);
        }
        full.into_iter()
            .map(|value| value.ok_or_else(|| invalid("values omit an exact global coordinate")))
            .collect()
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
        let fluid_map = layout
            .fluid_map(eqiora_meshing::CellId::new(0), &fluid, true)
            .unwrap();
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
        let (velocity, bubbles, pressure) = layout.reconstruct_primal(&values).unwrap();
        assert_eq!((velocity[0][0], velocity[3][1]), (1.25, -2.5));
        assert_eq!(
            layout.reduce(&velocity, &bubbles, &pressure).unwrap(),
            values
        );
        assert_eq!(layout.cell_domain(0).unwrap(), layout.fluid_domain());
        assert_eq!(layout.cell_domain(1).unwrap(), layout.solid_domain());
        assert!(layout.cell_domain(2).is_err());
        let mut wrong_bubbles = bubbles.clone();
        let value = wrong_bubbles.remove(&CellId::new(0)).unwrap();
        wrong_bubbles.insert(CellId::new(1), value);
        assert!(layout.reduce(&velocity, &wrong_bubbles, &pressure).is_err());
        assert!(
            layout
                .reduce(&velocity, &BTreeMap::new(), &pressure)
                .is_err()
        );
        let direction = layout.reconstruct_direction(&values).unwrap().0;
        assert_eq!((direction[0][0], direction[3][1]), (0.0, 0.0));
        assert!(
            layout
                .fluid_map(eqiora_meshing::CellId::new(1), &fluid, true)
                .is_err()
        );
        assert!(
            layout
                .fluid_map(eqiora_meshing::CellId::new(0), &solid, true)
                .is_err()
        );
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
