//! Compare package-neutral operators using independently named physical coordinates.

use super::SpatialContext;
use eqiora::meshing::MeshEntity;
use eqiora::solver::CanonicalCsrSystemView;
use eqiora_numerics::fsi::{
    FinalizedResolvedFixedReferenceFsiStep2d, ResolvedFixedReferenceFsiSolution2d,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum FieldRole {
    FluidVelocity,
    FluidPressure,
    SolidVelocity,
}
type Coordinate = (FieldRole, MeshEntity, usize, usize);

pub(crate) struct PhysicalOperator {
    system: CanonicalCsrSystemView,
    coordinates: BTreeMap<Coordinate, usize>,
    scales: BTreeMap<FieldRole, f64>,
}

impl PhysicalOperator {
    pub(crate) fn capture(
        finalized: &FinalizedResolvedFixedReferenceFsiStep2d,
        spatial: &SpatialContext,
    ) -> Self {
        let fields = finalized.fields();
        let mut coordinates = BTreeMap::new();
        let scales = [
            (FieldRole::FluidVelocity, fields.fluid_velocity()),
            (FieldRole::FluidPressure, fields.fluid_pressure()),
            (FieldRole::SolidVelocity, fields.solid_velocity()),
        ]
        .into_iter()
        .map(|(role, field)| {
            let scale = finalized
                .realization_plan()
                .scaling()
                .block_scales()
                .iter()
                .find(|binding| binding.block() == eqiora::solver::AlgebraicBlock::Field(field))
                .expect("exact Plan Field scale")
                .scale()
                .quantity()
                .value();
            (role, scale)
        })
        .collect();
        // Fixture-authored supports, independent of numbering and matrix values.
        for (role, field, vertices, components) in [
            (
                FieldRole::FluidVelocity,
                fields.fluid_velocity(),
                spatial.partition.fluid_vertices(),
                2,
            ),
            (
                FieldRole::FluidPressure,
                fields.fluid_pressure(),
                spatial.partition.fluid_vertices(),
                1,
            ),
            (
                FieldRole::SolidVelocity,
                fields.solid_velocity(),
                spatial.partition.solid_vertices(),
                2,
            ),
        ] {
            for vertex in vertices {
                let entity = MeshEntity::new(0, vertex.index());
                for component in 0..components {
                    if let Some(dof) = finalized.free_field_dof(field, entity, 0, component) {
                        coordinates.insert((role, entity, 0, component), dof.index());
                    }
                }
            }
        }
        for cell in spatial.partition.fluid_cells() {
            let entity = MeshEntity::new(2, cell.index());
            assert!(
                finalized
                    .free_field_dof(fields.fluid_pressure(), entity, 0, 0)
                    .is_none()
            );
            assert!(
                finalized
                    .free_field_dof(fields.solid_velocity(), entity, 0, 0)
                    .is_none()
            );
            assert!(
                finalized
                    .free_field_dof(fields.fluid_velocity(), entity, 0, 2)
                    .is_none()
            );
            for component in 0..2 {
                let dof = finalized
                    .free_field_dof(fields.fluid_velocity(), entity, 0, component)
                    .expect("each MINI bubble is an unconstrained fluid coordinate");
                coordinates.insert(
                    (FieldRole::FluidVelocity, entity, 0, component),
                    dof.index(),
                );
            }
        }
        let system = finalized.linear_system().clone();
        assert!(
            finalized
                .free_field_dof(
                    fields.fluid_velocity(),
                    MeshEntity::new(0, spatial.mesh.vertices().len()),
                    0,
                    0
                )
                .is_none()
        );
        assert_eq!(
            coordinates.values().copied().collect::<BTreeSet<_>>(),
            (0..system.rows()).collect()
        );
        Self {
            system,
            coordinates,
            scales,
        }
    }

    fn permutation(&self, other: &Self) -> Option<Vec<usize>> {
        if self.coordinates.keys().ne(other.coordinates.keys())
            || self.system.rows() != other.system.rows()
        {
            return None;
        }
        let mut permutation = vec![None; self.system.rows()];
        for (coordinate, &direct) in &self.coordinates {
            let packaged = other.coordinates[coordinate];
            if permutation[direct].is_some_and(|old| old != packaged) {
                return None;
            }
            permutation[direct] = Some(packaged);
        }
        let permutation = permutation.into_iter().collect::<Option<Vec<_>>>()?;
        (permutation.iter().copied().collect::<BTreeSet<_>>() == (0..other.system.rows()).collect())
            .then_some(permutation)
    }

    pub(crate) fn agrees(&self, other: &Self) -> bool {
        let Some(permutation) = self.permutation(other) else {
            return false;
        };
        for (row, &other_row) in permutation.iter().enumerate() {
            if self.system.right_hand_side()[row] != other.system.right_hand_side()[other_row] {
                return false;
            }
            for (column, &other_column) in permutation.iter().enumerate() {
                if entry(&self.system, row, column) != entry(&other.system, other_row, other_column)
                {
                    return false;
                }
            }
        }
        true
    }

    pub(crate) fn rejects_wrong_coordinates(&self, other: &Self) {
        // Swap named physical components without inspecting operator values.
        let first = other
            .coordinates
            .iter()
            .find(|((role, entity, _, component), _)| {
                *role == FieldRole::FluidVelocity && entity.dimension() == 2 && *component == 0
            })
            .unwrap();
        let second_key = (first.0.0, first.0.1, first.0.2, 1);
        let second = other.coordinates[&second_key];
        let mut wrong = other.coordinates.clone();
        wrong.insert(*first.0, second);
        wrong.insert(second_key, *first.1);
        assert!(!self.agrees(&Self {
            system: other.system.clone(),
            coordinates: wrong,
            scales: other.scales.clone()
        }));
        let mut wrong = other.coordinates.clone();
        let value = wrong.remove(first.0).unwrap();
        wrong.insert(
            (FieldRole::SolidVelocity, first.0.1, first.0.2, first.0.3),
            value,
        );
        assert!(
            self.permutation(&Self {
                system: other.system.clone(),
                coordinates: wrong,
                scales: other.scales.clone()
            })
            .is_none()
        );
    }

    pub(crate) fn assert_residual_compatible(
        &self,
        other: &Self,
        direct: &ResolvedFixedReferenceFsiSolution2d,
        packaged: &ResolvedFixedReferenceFsiSolution2d,
    ) {
        assert!(self.agrees(other));
        let mut difference = vec![None; self.system.rows()];
        for (&coordinate, &index) in &self.coordinates {
            let delta = physical_value(direct, coordinate) / self.scales[&coordinate.0]
                - physical_value(packaged, coordinate) / other.scales[&coordinate.0];
            if let Some(alias) = difference[index] {
                assert_eq!(alias, delta);
            }
            difference[index] = Some(delta);
        }
        let difference = difference.into_iter().collect::<Option<Vec<_>>>().unwrap();
        let norm = (0..self.system.rows())
            .map(|row| {
                let action = (self.system.row_offsets()[row]..self.system.row_offsets()[row + 1])
                    .map(|entry| {
                        self.system.values()[entry]
                            * difference[self.system.column_indices()[entry]]
                    })
                    .sum::<f64>();
                action * action
            })
            .sum::<f64>()
            .sqrt();
        // A(x_d - P^T x_p) = r_p - r_d. This is residual consistency,
        // not a forward-error claim under an unbounded condition number.
        let bound = direct.numerical_evidence().solve_report().residual_target()
            + packaged
                .numerical_evidence()
                .solve_report()
                .residual_target();
        assert!(
            norm <= bound,
            "physical recovery residual difference {norm} exceeds {bound}"
        );
    }
}

fn physical_value(solution: &ResolvedFixedReferenceFsiSolution2d, coordinate: Coordinate) -> f64 {
    let (role, entity, slot, component) = coordinate;
    assert_eq!(slot, 0);
    let vertex = eqiora::meshing::VertexId::new(entity.index());
    match role {
        FieldRole::FluidVelocity if entity.dimension() == 2 => solution
            .fluid_velocity_bubble_coefficients()[&eqiora::meshing::CellId::new(entity.index())]
            [component],
        FieldRole::FluidVelocity => solution.fluid_velocity_coefficient(vertex).unwrap()[component],
        FieldRole::SolidVelocity => solution.solid_velocity_coefficient(vertex).unwrap()[component],
        FieldRole::FluidPressure => solution.fluid_pressure_coefficient(vertex).unwrap(),
    }
}

fn entry(system: &CanonicalCsrSystemView, row: usize, column: usize) -> f64 {
    let range = system.row_offsets()[row]..system.row_offsets()[row + 1];
    let columns = &system.column_indices()[range.clone()];
    columns
        .binary_search(&column)
        .map_or(0.0, |index| system.values()[range.start + index])
}
