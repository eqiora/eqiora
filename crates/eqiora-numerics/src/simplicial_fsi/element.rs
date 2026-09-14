//! Dimensionless MINI-fluid and P1-solid local operators.

use eqiora_assembly::LocalContribution;
use eqiora_core::Diagnostic;
use eqiora_meshing::{AffineGeometryMap, MeshEntity, QuadratureRule};

use super::contract::{
    FixedReferenceFsiState, FixedReferenceFsiStepConfig, require_local_geometry_dimension,
};
use super::{fluid_local_size, p1_count};
use crate::simplicial_mini_transient::{MiniAffineScales, MiniScaledAffineCell};
use crate::simplicial_solid_element::p1_solid_backward_euler_velocity;

pub(crate) fn fluid_local<const D: usize>(
    geometry: &AffineGeometryMap,
    quadrature: &QuadratureRule,
    config: FixedReferenceFsiStepConfig<D>,
    vertices: &[MeshEntity],
    previous: &FixedReferenceFsiState<D>,
    cell: eqiora_meshing::CellId,
) -> Result<LocalContribution, Diagnostic> {
    require_geometry::<D>(geometry, quadrature)?;
    let p1_count = p1_count::<D>();
    let mut previous_velocity = vertices
        .iter()
        .take(p1_count)
        .map(|vertex| previous.vertex_velocity()[vertex.index()])
        .collect::<Vec<_>>();
    previous_velocity.push(previous.fluid_cell_bubble_velocity()[&cell]);
    let material = config.material();
    let (local_size, matrix, rhs) = MiniScaledAffineCell::<D> {
        geometry,
        density: material.fluid_density(),
        viscosity: material.fluid_dynamic_viscosity(),
        time_step: config.time_step(),
        previous_velocity: &previous_velocity,
        scales: MiniAffineScales::new(
            config.scale().velocity(),
            config.scale().pressure(),
            config.scale().power(),
        )?,
    }
    .project(quadrature)?
    .into_parts();
    debug_assert_eq!(local_size, fluid_local_size::<D>());
    LocalContribution::new(local_size, local_size, matrix, rhs)
}

pub(crate) fn solid_local<const D: usize>(
    geometry: &AffineGeometryMap,
    quadrature: &QuadratureRule,
    config: FixedReferenceFsiStepConfig<D>,
    vertices: &[MeshEntity],
    previous: &FixedReferenceFsiState<D>,
) -> Result<LocalContribution, Diagnostic> {
    require_geometry::<D>(geometry, quadrature)?;
    let p1_count = p1_count::<D>();
    let material = config.material();
    let velocity_scale = config.scale().velocity();
    let power_scale = config.scale().power();
    let previous_vertex_velocity = vertices
        .iter()
        .take(p1_count)
        .map(|vertex| previous.vertex_velocity()[vertex.index()])
        .collect::<Vec<_>>();
    let previous_vertex_displacement = vertices
        .iter()
        .take(p1_count)
        .map(|vertex| previous.solid_displacement()[vertex.index()])
        .collect::<Vec<_>>();
    p1_solid_backward_euler_velocity::<D>(
        geometry,
        quadrature,
        material.solid_density(),
        material.solid_material(),
        config.time_step(),
        &previous_vertex_velocity,
        &previous_vertex_displacement,
        velocity_scale,
        power_scale,
    )
}

pub(crate) fn dot(left: &[f64], right: &[f64]) -> f64 {
    crate::affine_fem::dot(left, right)
}

fn require_geometry<const D: usize>(
    geometry: &AffineGeometryMap,
    quadrature: &QuadratureRule,
) -> Result<(), Diagnostic> {
    require_local_geometry_dimension::<D>(geometry, quadrature)
}

#[cfg(test)]
mod tests {
    use eqiora_meshing::{
        CellId, FacetId, MeshEntity, MeshGeometry, MeshQualityGate, MeshTopology, SimplicialMesh,
        simplex_duffy_gauss_legendre,
    };

    use super::{fluid_local, solid_local};
    use crate::simplicial_fsi::contract::{
        FixedReferenceFsiBoundary, FixedReferenceFsiLoad, FixedReferenceFsiMaterial,
        FixedReferenceFsiScale, FixedReferenceFsiState, FixedReferenceFsiStepConfig,
    };
    use crate::simplicial_fsi::partition::FixedReferenceFsiPartition;

    #[test]
    fn tetrahedral_mini_and_p1_actions_share_one_finite_symmetric_kernel() {
        let fixture = fixture();
        let fluid = local_geometry(&fixture.mesh, 0);
        let solid = local_geometry(&fixture.mesh, 1);
        let fluid_local = fluid_local(
            &fluid.0,
            &fixture.quadrature,
            fixture.config,
            &fluid.1,
            &fixture.previous,
            eqiora_meshing::CellId::new(0),
        )
        .unwrap();
        let solid_local = solid_local(
            &solid.0,
            &fixture.quadrature,
            fixture.config,
            &solid.1,
            &fixture.previous,
        )
        .unwrap();
        let layout = crate::simplicial_fsi::test_model::polyhedra::polyhedral_layout(
            &crate::simplicial_fsi::test_model::polyhedra::z_tetrahedral_geometry(),
            &fixture.mesh,
            &fixture.partition,
            &fixture.boundary,
            fixture.config,
            eqiora_solver::SolverPlan::new(
                eqiora_solver::LinearSolver::MinimumResidual,
                1e-10,
                1e-12,
                std::num::NonZeroUsize::new(100).unwrap(),
            )
            .unwrap(),
            false,
        );
        let fluid_map = layout
            .fluid_map(eqiora_meshing::CellId::new(0), &fluid.1, false)
            .unwrap();
        let solid_map = layout.solid_map(1, &solid.1, false).unwrap();

        // tetrahedral MINI velocity: (P1 four vertices + one bubble) * 3,
        // followed by four P1 pressure coefficients.
        assert_eq!((fluid_local.rows(), fluid_local.columns()), (19, 19));
        assert_eq!((solid_local.rows(), solid_local.columns()), (12, 12));
        assert_eq!(
            (fluid_map.equations().len(), fluid_map.unknowns().len()),
            (19, 19)
        );
        assert_eq!(
            (solid_map.equations().len(), solid_map.unknowns().len()),
            (12, 12)
        );
        for local in [&fluid_local, &solid_local] {
            assert!(
                local
                    .matrix()
                    .iter()
                    .chain(local.rhs())
                    .all(|value| value.is_finite())
            );
            for row in 0..local.rows() {
                for column in 0..local.columns() {
                    assert!(
                        (local.matrix()[row * local.columns() + column]
                            - local.matrix()[column * local.columns() + row])
                            .abs()
                            < 2.0e-13
                    );
                }
            }
        }
    }

    #[test]
    fn tetrahedral_local_action_uses_area_scaled_power_normalization() {
        let fixture = fixture();
        let fluid = local_geometry(&fixture.mesh, 0);
        let solid = local_geometry(&fixture.mesh, 1);
        let wider_scale = FixedReferenceFsiScale::<3>::new(4.0, 5.0, 3.0).unwrap();
        let wider = FixedReferenceFsiStepConfig::<3>::new(
            fixture.config.time_step(),
            fixture.config.material(),
            wider_scale,
            FixedReferenceFsiLoad::Zero,
        )
        .unwrap();

        let fluid_reference = fluid_local(
            &fluid.0,
            &fixture.quadrature,
            fixture.config,
            &fluid.1,
            &fixture.previous,
            eqiora_meshing::CellId::new(0),
        )
        .unwrap();
        let fluid_wider = fluid_local(
            &fluid.0,
            &fixture.quadrature,
            wider,
            &fluid.1,
            &fixture.previous,
            eqiora_meshing::CellId::new(0),
        )
        .unwrap();
        let solid_reference = solid_local(
            &solid.0,
            &fixture.quadrature,
            fixture.config,
            &solid.1,
            &fixture.previous,
        )
        .unwrap();
        let solid_wider = solid_local(
            &solid.0,
            &fixture.quadrature,
            wider,
            &solid.1,
            &fixture.previous,
        )
        .unwrap();

        // With U and P fixed, doubling L in 3D multiplies the power scale by
        // four. Every nonzero dimensionless local coefficient therefore falls
        // by exactly four; a linear-length rule would fail this assertion.
        for (reference, wider) in [
            (fluid_reference.matrix()[0], fluid_wider.matrix()[0]),
            (solid_reference.matrix()[0], solid_wider.matrix()[0]),
        ] {
            assert!(reference != 0.0);
            assert!((reference - 4.0 * wider).abs() < 2.0e-13 * reference.abs());
        }
    }

    struct Fixture {
        mesh: SimplicialMesh,
        partition: FixedReferenceFsiPartition<3>,
        boundary: FixedReferenceFsiBoundary<3>,
        previous: FixedReferenceFsiState<3>,
        config: FixedReferenceFsiStepConfig<3>,
        quadrature: eqiora_meshing::QuadratureRule,
    }

    fn fixture() -> Fixture {
        let mesh = SimplicialMesh::new(
            3,
            vec![
                vec![0.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![0.0, 0.0, -1.0],
            ],
            vec![vec![0, 1, 2, 3], vec![0, 2, 1, 4]],
            MeshQualityGate::new(0.05).unwrap(),
        )
        .unwrap();
        let interface = (0..mesh.entity_count(2).unwrap())
            .find(|&facet| {
                mesh.entity_vertices(MeshEntity::new(2, facet))
                    .unwrap()
                    .iter()
                    .map(|vertex| vertex.index())
                    .collect::<Vec<_>>()
                    == [0, 1, 2]
            })
            .map(FacetId::new)
            .unwrap();
        let partition = FixedReferenceFsiPartition::<3>::new(
            &mesh,
            vec![CellId::new(0)],
            vec![CellId::new(1)],
            vec![interface],
        )
        .unwrap();
        let previous = FixedReferenceFsiState::<3>::new(
            &mesh,
            &partition,
            vec![[0.0; 3]; mesh.vertices().len()],
            partition
                .fluid_cells()
                .iter()
                .copied()
                .map(|cell| (cell, [0.0; 3]))
                .collect(),
            vec![[0.0; 3]; mesh.vertices().len()],
        )
        .unwrap();
        let boundary = FixedReferenceFsiBoundary::<3>::homogeneous_exterior(&mesh).unwrap();
        let material = FixedReferenceFsiMaterial::<3>::new(1.0, 0.1, 2.0, 3.0, 1.0).unwrap();
        let scale = FixedReferenceFsiScale::<3>::new(2.0, 5.0, 3.0).unwrap();
        let config = FixedReferenceFsiStepConfig::<3>::new(
            0.25,
            material,
            scale,
            FixedReferenceFsiLoad::Zero,
        )
        .unwrap();
        Fixture {
            mesh,
            partition,
            boundary,
            previous,
            config,
            quadrature: simplex_duffy_gauss_legendre(3, 6).unwrap(),
        }
    }

    fn local_geometry(
        mesh: &SimplicialMesh,
        cell: usize,
    ) -> (eqiora_meshing::AffineGeometryMap, Vec<MeshEntity>) {
        let entity = MeshEntity::new(3, cell);
        (
            mesh.geometry_map(entity).unwrap(),
            mesh.entity_vertices(entity).unwrap(),
        )
    }
}
