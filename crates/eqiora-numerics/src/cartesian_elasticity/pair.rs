//! Conforming monolithic Q1 realization of two elasticity subdomains.
//!
//! The two meshes remain independent resources. This module records the exact
//! topological vertex quotient selected by the Realization, then scatters both
//! subdomain operators into the same global interface rows. Consequently,
//! displacement continuity is unknown identity and weak traction balance is
//! ordinary assembled equilibrium; neither condition is approximated by a
//! penalty or represented by an additional multiplier.

use std::sync::Arc;

use eqiora_assembly::{
    AssemblyBackend, AssemblyMap, AssemblyPacket, AssemblyPlan, AssemblyReport, AssemblyTarget,
    DofId, IndexedAssemblyWork, LocalUnknown, TargetAssemblyMap,
};
use eqiora_core::Diagnostic;
use eqiora_meshing::{MeshEntity, MeshGeometry, MeshTopology, QuadratureRule};
use eqiora_solver::{
    CanonicalCsrSystemView, LinearOperatorProperties, LinearSolution, SolveReport,
};

use super::{
    COMPONENTS, CartesianEssentialSides2d, CartesianQ1VectorField2d, DIMENSION, global_dof,
    invalid, require_cell_rule, require_two_dimensional_mesh,
};
use crate::constrained_dofs::ConstrainedDofLayout;
use crate::form_compiler::compile_cartesian_q1_elasticity_form_2d;
use crate::interleaved_dofs::InterleavedDofValues;
use crate::linear_elasticity::IsotropicElasticityMaterial;
use crate::spatial_expression::ScalarSpatialExpression;
use eqiora_meshing::CartesianMesh;

/// Exact topological quotient between two conforming Cartesian interface meshes.
///
/// Local vertex identities remain available for field reconstruction. Only the
/// Realization-owned map identifies paired interface vertices with one global
/// algebraic vertex; the Semantic Model's two Domain identities are untouched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConformingCartesianInterfaceMap2d {
    axis: usize,
    local_to_global: [Vec<usize>; 2],
    interface_vertices: Vec<[usize; 2]>,
    global_vertex_count: usize,
}

impl ConformingCartesianInterfaceMap2d {
    /// Cartesian axis normal to the interface.
    #[must_use]
    pub const fn axis(&self) -> usize {
        self.axis
    }

    /// Local-to-quotient vertex map for one geometrically ordered subdomain.
    #[must_use]
    pub fn local_to_global(&self, subdomain: usize) -> Option<&[usize]> {
        self.local_to_global.get(subdomain).map(Vec::as_slice)
    }

    /// Paired local vertices in increasing tangential topological order.
    ///
    /// Each pair is `[negative_subdomain, positive_subdomain]`.
    #[must_use]
    pub fn interface_vertices(&self) -> &[[usize; 2]] {
        &self.interface_vertices
    }

    /// Number of vertices after identifying the coincident interface once.
    #[must_use]
    pub const fn global_vertex_count(&self) -> usize {
        self.global_vertex_count
    }
}

/// Body-local weak actions on both sides of a conforming interface.
///
/// Values are the interface rows of each cut subdomain residual `K_i u_i-f_i`
/// in the same tangential order as
/// [`ConformingCartesianInterfaceMap2d::interface_vertices`]. Their pairwise
/// sum is the finite-dimensional interface-equilibrium residual.
#[derive(Debug, Clone, PartialEq)]
pub struct ConformingElasticityInterfaceAction2d {
    negative: Vec<[f64; COMPONENTS]>,
    positive: Vec<[f64; COMPONENTS]>,
    free: Vec<bool>,
}

impl ConformingElasticityInterfaceAction2d {
    /// Weak actions exerted on the lower-coordinate subdomain.
    #[must_use]
    pub fn negative(&self) -> &[[f64; COMPONENTS]] {
        &self.negative
    }

    /// Weak actions exerted on the upper-coordinate subdomain.
    #[must_use]
    pub fn positive(&self) -> &[[f64; COMPONENTS]] {
        &self.positive
    }

    /// Whether each paired interface vertex is a free quotient row.
    ///
    /// At a constrained interface endpoint, a cut residual also contains the
    /// external support reaction and is not uniquely attributable to coupling.
    #[must_use]
    pub fn free_mask(&self) -> &[bool] {
        &self.free
    }

    /// Sum the lower-coordinate body's actions over free interface rows.
    #[must_use]
    pub fn negative_free_resultant(&self) -> [f64; COMPONENTS] {
        free_resultant(&self.negative, &self.free)
    }

    /// Sum the upper-coordinate body's actions over free interface rows.
    #[must_use]
    pub fn positive_free_resultant(&self) -> [f64; COMPONENTS] {
        free_resultant(&self.positive, &self.free)
    }

    /// Sum the two cut-subdomain actions at every free interface row.
    ///
    /// Constrained endpoints return `None` because their cut residual includes
    /// an inseparable external support reaction.
    #[must_use]
    pub fn free_equilibrium_residual(&self) -> Vec<Option<[f64; COMPONENTS]>> {
        self.negative
            .iter()
            .zip(&self.positive)
            .zip(&self.free)
            .map(|((negative, positive), free)| {
                free.then(|| {
                    std::array::from_fn(|component| negative[component] + positive[component])
                })
            })
            .collect()
    }
}

fn free_resultant(values: &[[f64; COMPONENTS]], free: &[bool]) -> [f64; COMPONENTS] {
    let mut result = [0.0; COMPONENTS];
    for value in values
        .iter()
        .zip(free)
        .filter_map(|(value, free)| free.then_some(value))
    {
        for component in 0..COMPONENTS {
            result[component] += value[component];
        }
    }
    result
}

/// Accepted monolithic Q1 solution over two conforming elasticity subdomains.
#[derive(Debug, Clone, PartialEq)]
pub struct ConformingCartesianLinearElasticityPair2dSolution {
    displacement: [CartesianQ1VectorField2d; 2],
    algebraic_values: Vec<f64>,
    interface_map: ConformingCartesianInterfaceMap2d,
    interface_action: ConformingElasticityInterfaceAction2d,
    boundary_reaction: [f64; COMPONENTS],
    integrated_body_force: [[f64; COMPONENTS]; 2],
    assembly_report: AssemblyReport,
    solve_report: SolveReport,
}

impl ConformingCartesianLinearElasticityPair2dSolution {
    /// Reconstructed body-local displacement fields in geometric order.
    #[must_use]
    pub const fn displacement(&self) -> &[CartesianQ1VectorField2d; 2] {
        &self.displacement
    }

    /// Free quotient-space values in assembled equation order.
    #[must_use]
    pub fn algebraic_values(&self) -> &[f64] {
        &self.algebraic_values
    }

    /// Exact Realization-owned interface vertex quotient.
    #[must_use]
    pub const fn interface_map(&self) -> &ConformingCartesianInterfaceMap2d {
        &self.interface_map
    }

    /// Cut-subdomain weak interface actions and equilibrium evidence.
    #[must_use]
    pub const fn interface_action(&self) -> &ConformingElasticityInterfaceAction2d {
        &self.interface_action
    }

    /// Sum of full quotient-system residuals on external constrained vertices.
    #[must_use]
    pub const fn boundary_reaction(&self) -> [f64; COMPONENTS] {
        self.boundary_reaction
    }

    /// Integrated conservative body force for each subdomain.
    #[must_use]
    pub const fn integrated_body_force(&self) -> [[f64; COMPONENTS]; 2] {
        self.integrated_body_force
    }

    /// Complete assembly placement and packet-shape evidence.
    #[must_use]
    pub const fn assembly_report(&self) -> &AssemblyReport {
        &self.assembly_report
    }

    /// Complete solver/backend/execution evidence.
    #[must_use]
    pub const fn solve_report(&self) -> &SolveReport {
        &self.solve_report
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FinalizedConformingCartesianElasticityPair2dAssembly {
    meshes: [CartesianMesh; 2],
    interface_map: ConformingCartesianInterfaceMap2d,
    constrained_dofs: ConstrainedDofLayout<f64>,
    linear_system: eqiora_assembly::LinearSystem<f64>,
    full_system: eqiora_assembly::LinearSystem<f64>,
    subdomain_systems: [eqiora_assembly::LinearSystem<f64>; 2],
    integrated_body_force: [[f64; COMPONENTS]; 2],
    assembly_report: AssemblyReport,
}

impl FinalizedConformingCartesianElasticityPair2dAssembly {
    pub(crate) fn into_canonical(
        self,
    ) -> Result<
        (
            Arc<CanonicalCsrSystemView>,
            FinalizedConformingCartesianElasticityPair2dState,
        ),
        Diagnostic,
    > {
        let Self {
            meshes,
            interface_map,
            constrained_dofs,
            linear_system,
            full_system,
            subdomain_systems,
            integrated_body_force,
            assembly_report,
        } = self;
        let canonical_system = Arc::new(CanonicalCsrSystemView::new(
            &linear_system,
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )?);
        Ok((
            canonical_system,
            FinalizedConformingCartesianElasticityPair2dState {
                meshes,
                interface_map,
                constrained_dofs,
                full_system,
                subdomain_systems,
                integrated_body_force,
                assembly_report,
            },
        ))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FinalizedConformingCartesianElasticityPair2dState {
    meshes: [CartesianMesh; 2],
    interface_map: ConformingCartesianInterfaceMap2d,
    constrained_dofs: ConstrainedDofLayout<f64>,
    full_system: eqiora_assembly::LinearSystem<f64>,
    subdomain_systems: [eqiora_assembly::LinearSystem<f64>; 2],
    integrated_body_force: [[f64; COMPONENTS]; 2],
    assembly_report: AssemblyReport,
}

impl FinalizedConformingCartesianElasticityPair2dState {
    pub(crate) const fn assembly_report(&self) -> &AssemblyReport {
        &self.assembly_report
    }

    pub(crate) fn finish(
        self,
        solved: LinearSolution,
        canonical_system: Arc<CanonicalCsrSystemView>,
    ) -> Result<ConformingCartesianLinearElasticityPair2dSolution, Diagnostic> {
        if solved.values().len() != canonical_system.rows() {
            return Err(invalid(
                "conforming elasticity solution shape differs from its finalized system",
            ));
        }
        let (algebraic_values, solve_report) = solved.into_parts();
        let global_values = self.constrained_dofs.lift(&algebraic_values)?;
        let quotient_residual = self
            .constrained_dofs
            .full_residual(&self.full_system, &global_values)?;
        let boundary_reaction = self
            .constrained_dofs
            .reaction_sum::<COMPONENTS>(&quotient_residual)?;

        let local_values: [Vec<f64>; 2] = std::array::from_fn(|subdomain| {
            self.interface_map.local_to_global[subdomain]
                .iter()
                .flat_map(|global_vertex| {
                    let base = global_vertex * COMPONENTS;
                    global_values[base..base + COMPONENTS].iter().copied()
                })
                .collect()
        });
        let mut local_residuals = Vec::with_capacity(2);
        for (system, values) in self.subdomain_systems.iter().zip(&local_values) {
            let mut residual = system.matrix().multiply(values)?;
            for (value, rhs) in residual.iter_mut().zip(system.rhs()) {
                *value -= rhs;
            }
            local_residuals.push(residual);
        }
        let mut negative = Vec::with_capacity(self.interface_map.interface_vertices.len());
        let mut positive = Vec::with_capacity(self.interface_map.interface_vertices.len());
        let mut free = Vec::with_capacity(self.interface_map.interface_vertices.len());
        for [negative_vertex, positive_vertex] in &self.interface_map.interface_vertices {
            negative.push(std::array::from_fn(|component| {
                local_residuals[0][negative_vertex * COMPONENTS + component]
            }));
            positive.push(std::array::from_fn(|component| {
                local_residuals[1][positive_vertex * COMPONENTS + component]
            }));
            let quotient_vertex = self.interface_map.local_to_global[0][*negative_vertex];
            let mut vertex_is_free = true;
            for component in 0..COMPONENTS {
                vertex_is_free &= self
                    .constrained_dofs
                    .is_free(quotient_vertex * COMPONENTS + component)?;
            }
            free.push(vertex_is_free);
        }
        if boundary_reaction
            .iter()
            .chain(self.integrated_body_force.iter().flatten())
            .chain(negative.iter().flatten())
            .chain(positive.iter().flatten())
            .any(|value| !value.is_finite())
        {
            return Err(invalid(
                "conforming elasticity balance evidence is non-finite",
            ));
        }

        let [negative_values, positive_values] = local_values;
        Ok(ConformingCartesianLinearElasticityPair2dSolution {
            displacement: [
                CartesianQ1VectorField2d::new(self.meshes[0].clone(), negative_values)?,
                CartesianQ1VectorField2d::new(self.meshes[1].clone(), positive_values)?,
            ],
            algebraic_values,
            interface_map: self.interface_map,
            interface_action: ConformingElasticityInterfaceAction2d {
                negative,
                positive,
                free,
            },
            boundary_reaction,
            integrated_body_force: self.integrated_body_force,
            assembly_report: self.assembly_report,
            solve_report,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn finalize_conforming_cartesian_q1_linear_elasticity_pair_2d(
    meshes: [CartesianMesh; 2],
    materials: [IsotropicElasticityMaterial<DIMENSION>; 2],
    body_force_potentials: [&ScalarSpatialExpression<f64>; 2],
    quadrature: &QuadratureRule,
    interface_axis: usize,
    essential_sides: [CartesianEssentialSides2d; 2],
    assembly: &dyn AssemblyBackend<f64>,
) -> Result<FinalizedConformingCartesianElasticityPair2dAssembly, Diagnostic> {
    for subdomain in 0..2 {
        require_two_dimensional_mesh(&meshes[subdomain])?;
        require_cell_rule(&meshes[subdomain], quadrature)?;
        if body_force_potentials[subdomain].coordinate_dimension() != DIMENSION {
            return Err(invalid(format!(
                "elasticity body-force potential for subdomain {subdomain} expects {} coordinates, not {DIMENSION}",
                body_force_potentials[subdomain].coordinate_dimension(),
            )));
        }
    }
    let interface_map = build_interface_map(&meshes, interface_axis)?;
    let global_width = interface_map
        .global_vertex_count
        .checked_mul(COMPONENTS)
        .ok_or_else(|| invalid("conforming elasticity global DOF count overflows usize"))?;
    let mut constrained = vec![false; interface_map.global_vertex_count];
    for subdomain in 0..2 {
        for vertex_index in 0..interface_map.local_to_global[subdomain].len() {
            if essential_sides[subdomain]
                .constrains_vertex(&meshes[subdomain], MeshEntity::new(0, vertex_index))
            {
                constrained[interface_map.local_to_global[subdomain][vertex_index]] = true;
            }
        }
    }
    if !constrained.iter().copied().any(|value| value) {
        return Err(invalid(
            "conforming Cartesian elasticity pair requires at least one external homogeneous essential side to remove global rigid modes",
        ));
    }
    let mut fixed_values = vec![Some(0.0); global_width];
    for (global_vertex, constrained) in constrained.into_iter().enumerate() {
        if !constrained {
            for component in 0..COMPONENTS {
                fixed_values[global_dof(global_vertex, component)?] = None;
            }
        }
    }
    let constrained_dofs = ConstrainedDofLayout::new(fixed_values)?;
    if constrained_dofs.free_count() == 0 {
        return Err(invalid(
            "conforming Cartesian elasticity pair requires at least one unconstrained vertex",
        ));
    }

    let local_widths = [
        meshes[0]
            .entity_count(0)
            .expect("a Cartesian mesh owns its vertex stratum")
            .checked_mul(COMPONENTS)
            .ok_or_else(|| invalid("negative elasticity local DOF count overflows usize"))?,
        meshes[1]
            .entity_count(0)
            .expect("a Cartesian mesh owns its vertex stratum")
            .checked_mul(COMPONENTS)
            .ok_or_else(|| invalid("positive elasticity local DOF count overflows usize"))?,
    ];
    let plan = AssemblyPlan::new(vec![
        AssemblyTarget::new(constrained_dofs.free_count())?,
        AssemblyTarget::new(global_width)?,
        AssemblyTarget::new(local_widths[0])?,
        AssemblyTarget::new(local_widths[1])?,
    ])?;
    let targets = [
        plan.target_id(0).expect("pair plan owns reduced target"),
        plan.target_id(1)
            .expect("pair plan owns quotient full target"),
        plan.target_id(2)
            .expect("pair plan owns negative full target"),
        plan.target_id(3)
            .expect("pair plan owns positive full target"),
    ];
    let cell_counts = meshes.each_ref().map(|mesh| {
        mesh.entity_count(DIMENSION)
            .expect("a two-dimensional mesh owns its cell stratum")
    });
    let packet_count = cell_counts[0]
        .checked_add(cell_counts[1])
        .ok_or_else(|| invalid("conforming elasticity cell count overflows usize"))?;
    let operator = compile_cartesian_q1_elasticity_form_2d(quadrature)?;
    let work = IndexedAssemblyWork::new(packet_count, |packet| {
        let (subdomain, cell_index) = if packet < cell_counts[0] {
            (0, packet)
        } else {
            (1, packet - cell_counts[0])
        };
        let mesh = &meshes[subdomain];
        let cell = MeshEntity::new(DIMENSION, cell_index);
        let geometry = mesh
            .geometry_map(cell)
            .expect("a Cartesian cell owns affine geometry");
        let local = operator.evaluate(
            &geometry,
            quadrature,
            materials[subdomain].shear_modulus(),
            materials[subdomain].first_lame_parameter(),
            Some(body_force_potentials[subdomain]),
        )?;
        let vertices = mesh
            .entity_vertices(cell)
            .expect("a Cartesian cell owns its vertex closure");
        let local_capacity = vertices
            .len()
            .checked_mul(COMPONENTS)
            .ok_or_else(|| invalid("conforming elasticity local map width overflows usize"))?;
        let mut quotient_dofs = Vec::with_capacity(local_capacity);
        let mut local_dofs = Vec::with_capacity(local_capacity);
        for vertex in vertices {
            let quotient_vertex = interface_map.local_to_global[subdomain][vertex.index()];
            for component in 0..COMPONENTS {
                quotient_dofs.push(global_dof(quotient_vertex, component)?);
                local_dofs.push(global_dof(vertex.index(), component)?);
            }
        }
        let reduced = constrained_dofs.reduced_map(&quotient_dofs)?;
        let quotient_full = identity_map(&quotient_dofs)?;
        let local_full = identity_map(&local_dofs)?;
        AssemblyPacket::new(
            local,
            vec![
                TargetAssemblyMap::new(targets[0], reduced),
                TargetAssemblyMap::new(targets[1], quotient_full),
                TargetAssemblyMap::new(targets[2 + subdomain], local_full),
            ],
        )
    });
    let (systems, assembly_report) = assembly.assemble(&plan, &work)?.into_parts();
    let [linear_system, full_system, negative_system, positive_system]: [eqiora_assembly::LinearSystem<f64>; 4] =
        systems.try_into().map_err(|systems: Vec<_>| {
            invalid(format!(
                "conforming elasticity assembly returned {} targets instead of four",
                systems.len()
            ))
        })?;
    let subdomain_systems = [negative_system, positive_system];
    let integrated_body_force = [
        InterleavedDofValues::<COMPONENTS>::new(subdomain_systems[0].rhs())?.sum(),
        InterleavedDofValues::<COMPONENTS>::new(subdomain_systems[1].rhs())?.sum(),
    ];
    if integrated_body_force
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(invalid(
            "integrated conforming elasticity body force is non-finite",
        ));
    }

    Ok(FinalizedConformingCartesianElasticityPair2dAssembly {
        meshes,
        interface_map,
        constrained_dofs,
        linear_system,
        full_system,
        subdomain_systems,
        integrated_body_force,
        assembly_report,
    })
}

fn build_interface_map(
    meshes: &[CartesianMesh; 2],
    axis: usize,
) -> Result<ConformingCartesianInterfaceMap2d, Diagnostic> {
    if axis >= DIMENSION {
        return Err(invalid(format!(
            "conforming elasticity interface axis {axis} lies outside dimension {DIMENSION}",
        )));
    }
    let tangent = 1 - axis;
    let negative_normal = meshes[0]
        .axis_coordinates(axis)
        .expect("a two-dimensional mesh owns both axes");
    let positive_normal = meshes[1]
        .axis_coordinates(axis)
        .expect("a two-dimensional mesh owns both axes");
    if negative_normal.last() != positive_normal.first() {
        return Err(invalid(
            "conforming elasticity meshes do not share one exact interface coordinate",
        ));
    }
    let negative_tangent = meshes[0]
        .axis_coordinates(tangent)
        .expect("a two-dimensional mesh owns both axes");
    let positive_tangent = meshes[1]
        .axis_coordinates(tangent)
        .expect("a two-dimensional mesh owns both axes");
    if negative_tangent != positive_tangent {
        return Err(invalid(
            "conforming elasticity meshes require identical tangential vertex coordinates",
        ));
    }

    let vertex_counts = meshes.each_ref().map(|mesh| {
        mesh.entity_count(0)
            .expect("a Cartesian mesh owns its vertex stratum")
    });
    let mut negative_interface = vec![None; negative_tangent.len()];
    let mut positive_interface = vec![None; positive_tangent.len()];
    for (subdomain, interface) in [(0, &mut negative_interface), (1, &mut positive_interface)] {
        let normal_index = if subdomain == 0 {
            meshes[subdomain]
                .axis_coordinates(axis)
                .expect("axis exists")
                .len()
                - 1
        } else {
            0
        };
        for vertex in 0..vertex_counts[subdomain] {
            let index = meshes[subdomain]
                .vertex_multi_index(MeshEntity::new(0, vertex))
                .expect("a Cartesian vertex owns one index per axis");
            if index[axis] == normal_index {
                interface[index[tangent]] = Some(vertex);
            }
        }
    }
    let interface_vertices = negative_interface
        .into_iter()
        .zip(positive_interface)
        .map(|(negative, positive)| {
            Ok([
                negative.ok_or_else(|| invalid("negative interface vertex is missing"))?,
                positive.ok_or_else(|| invalid("positive interface vertex is missing"))?,
            ])
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    let negative_map = (0..vertex_counts[0]).collect::<Vec<_>>();
    let mut positive_map = vec![usize::MAX; vertex_counts[1]];
    for [negative, positive] in &interface_vertices {
        positive_map[*positive] = negative_map[*negative];
    }
    let mut next_global = vertex_counts[0];
    for global in &mut positive_map {
        if *global == usize::MAX {
            *global = next_global;
            next_global = next_global
                .checked_add(1)
                .ok_or_else(|| invalid("conforming elasticity vertex quotient overflows usize"))?;
        }
    }
    let expected = vertex_counts[0]
        .checked_add(vertex_counts[1])
        .and_then(|value| value.checked_sub(interface_vertices.len()))
        .ok_or_else(|| invalid("conforming elasticity vertex quotient shape overflows usize"))?;
    if next_global != expected {
        return Err(invalid(
            "conforming elasticity interface quotient did not identify every interface vertex exactly once",
        ));
    }
    Ok(ConformingCartesianInterfaceMap2d {
        axis,
        local_to_global: [negative_map, positive_map],
        interface_vertices,
        global_vertex_count: next_global,
    })
}

fn identity_map(dofs: &[usize]) -> Result<AssemblyMap<f64>, Diagnostic> {
    AssemblyMap::new(
        dofs.iter().map(|dof| Some(DofId::new(*dof))).collect(),
        dofs.iter()
            .map(|dof| LocalUnknown::Free(DofId::new(*dof)))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interface_map_identifies_each_topological_trace_vertex_once() {
        let meshes = [
            CartesianMesh::uniform(&[[0.0, 0.5], [0.0, 1.0]], &[2, 3]).unwrap(),
            CartesianMesh::uniform(&[[0.5, 1.0], [0.0, 1.0]], &[2, 3]).unwrap(),
        ];
        let map = build_interface_map(&meshes, 0).unwrap();

        assert_eq!(map.axis(), 0);
        assert_eq!(map.interface_vertices().len(), 4);
        assert_eq!(map.global_vertex_count(), 20);
        for [negative, positive] in map.interface_vertices() {
            assert_eq!(
                map.local_to_global(0).unwrap()[*negative],
                map.local_to_global(1).unwrap()[*positive]
            );
            assert_eq!(
                meshes[0]
                    .vertex_coordinates(MeshEntity::new(0, *negative))
                    .unwrap(),
                meshes[1]
                    .vertex_coordinates(MeshEntity::new(0, *positive))
                    .unwrap()
            );
        }
    }

    #[test]
    fn interface_map_rejects_nonmatching_tangential_vertices() {
        let meshes = [
            CartesianMesh::uniform(&[[0.0, 0.5], [0.0, 1.0]], &[2, 2]).unwrap(),
            CartesianMesh::uniform(&[[0.5, 1.0], [0.0, 1.0]], &[2, 3]).unwrap(),
        ];
        let diagnostic = build_interface_map(&meshes, 0).unwrap_err();

        assert!(
            diagnostic
                .message()
                .contains("identical tangential vertex coordinates")
        );
    }

    #[test]
    fn interface_map_is_axis_symmetric() {
        let meshes = [
            CartesianMesh::uniform(&[[0.0, 1.0], [0.0, 0.5]], &[3, 2]).unwrap(),
            CartesianMesh::uniform(&[[0.0, 1.0], [0.5, 1.0]], &[3, 2]).unwrap(),
        ];
        let map = build_interface_map(&meshes, 1).unwrap();

        assert_eq!(map.axis(), 1);
        assert_eq!(map.interface_vertices().len(), 4);
        assert_eq!(map.global_vertex_count(), 20);
        for [negative, positive] in map.interface_vertices() {
            assert_eq!(
                map.local_to_global(0).unwrap()[*negative],
                map.local_to_global(1).unwrap()[*positive]
            );
        }
    }
}
