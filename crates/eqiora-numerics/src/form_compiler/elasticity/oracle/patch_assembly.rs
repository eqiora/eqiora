use eqiora_assembly::{
    AssemblyMap, CooAssembler, DofId, LinearSystem, LocalContribution, LocalUnknown,
};
use eqiora_ir::LocalLinearActionIr;
use eqiora_meshing::{CartesianMesh, MeshEntity, MeshGeometry, MeshTopology, QuadratureRule};
use eqiora_sem::KernelProgram;

use super::super::derive_cartesian_q1_elasticity_form_2d;

pub(super) fn assemble_body_free_action(
    mesh: &CartesianMesh,
    action: &LocalLinearActionIr,
) -> LinearSystem<f64> {
    let vertex_count = mesh.entity_count(0).unwrap();
    let mut assembler = CooAssembler::new(2 * vertex_count).unwrap();
    let entries_per_cell = action.rows() * action.columns();
    for cell_index in 0..action.entity_count() {
        let offset = cell_index * entries_per_cell;
        let local = LocalContribution::new(
            action.rows(),
            action.columns(),
            action.coefficients()[offset..offset + entries_per_cell].to_vec(),
            vec![0.0; action.rows()],
        )
        .unwrap();
        assembler
            .scatter(&full_patch_map(mesh, cell_index), &local)
            .unwrap();
    }
    assembler.finish().unwrap()
}

pub(super) fn assemble_loaded_derived_form(
    program: &KernelProgram,
    model: &crate::canonical_elasticity::IsotropicElasticityContinuum<2>,
    mesh: &CartesianMesh,
    quadrature: &QuadratureRule,
) -> LinearSystem<f64> {
    let form = derive_cartesian_q1_elasticity_form_2d(program).unwrap();
    let admitted = form.admit_quadrature(quadrature).unwrap();
    let vertex_count = mesh.entity_count(0).unwrap();
    let mut assembler = CooAssembler::new(2 * vertex_count).unwrap();
    for cell_index in 0..mesh.entity_count(2).unwrap() {
        let geometry = mesh.geometry_map(MeshEntity::new(2, cell_index)).unwrap();
        let local = admitted
            .evaluate(
                &geometry,
                quadrature,
                model.shear_modulus(),
                model.first_lame_parameter(),
                Some(model.load_potential_expression()),
            )
            .unwrap();
        assembler
            .scatter(&full_patch_map(mesh, cell_index), &local)
            .unwrap();
    }
    assembler.finish().unwrap()
}

fn full_patch_map(mesh: &CartesianMesh, cell_index: usize) -> AssemblyMap<f64> {
    let vertices = mesh
        .entity_vertices(MeshEntity::new(2, cell_index))
        .unwrap();
    let global = vertices
        .iter()
        .flat_map(|vertex| [2 * vertex.index(), 2 * vertex.index() + 1])
        .collect::<Vec<_>>();
    let rows = global
        .iter()
        .map(|index| Some(DofId::new(*index)))
        .collect();
    let columns = global
        .iter()
        .map(|index| LocalUnknown::Free(DofId::new(*index)))
        .collect();
    AssemblyMap::new(rows, columns).unwrap()
}

pub(super) fn center_vertex(mesh: &CartesianMesh) -> usize {
    (0..mesh.entity_count(0).unwrap())
        .find(|vertex| {
            mesh.vertex_coordinates(MeshEntity::new(0, *vertex))
                .unwrap()
                == [0.5, 0.5]
        })
        .unwrap()
}
