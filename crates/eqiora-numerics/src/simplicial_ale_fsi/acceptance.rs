//! Independent acceptance of one converged fixed-topology ALE FSI step.
//!
//! Acceptance reassembles the final nonlinear point, recovers the separate
//! fluid and solid interface actions, and checks residual, kinematic,
//! interface, geometry, and provenance evidence. No fixed-domain energy
//! identity is asserted for the moving control volume.

use eqiora_assembly::AssemblyBackend;
use eqiora_core::Diagnostic;
use eqiora_meshing::FixedTopologyGeometryAction;
use eqiora_meshing::{AffineGeometryLinearization, QuadratureRule, SimplicialMesh};

use super::api::{AleFsiInterfaceAction, AleFsiStepEvidence, AleFsiStepEvidenceInput};
use super::assembly::{
    PreparedAleFsiAction, PreparedAleFsiStructure, StepAssembly,
    assemble_step_linearization_with_structure,
};
use super::contract::{AleFsiState, AleFsiStepPlan};
use super::element::{AleMiniFluidCell, AleMiniFluidDirection};
use super::{P1HarmonicMeshMotionAction, invalid};
use crate::discrete_space::{DiscreteSpace, SimplexP1BubbleSpace};
use crate::simplicial_fsi::FixedReferenceFsiPartition;

pub(super) struct NewtonEvidence {
    pub(super) iterations: usize,
    pub(super) initial_residual_norm: f64,
    pub(super) linear_solves: Vec<eqiora_solver::SolveReport>,
}

#[allow(clippy::too_many_arguments)]
fn accept_independent<const D: usize>(
    partition: &FixedReferenceFsiPartition<D>,
    previous: &AleFsiState<D>,
    plan: AleFsiStepPlan<D>,
    quadrature: &QuadratureRule,
    converged: StepAssembly<D>,
    independent: StepAssembly<D>,
    newton: NewtonEvidence,
) -> Result<(AleFsiState<D>, AleFsiStepEvidence<D>), Diagnostic> {
    require_same_accepted_point(&converged, &independent)?;

    let final_residual_norm = independent.residual_norm()?;
    let nonlinear = plan.nonlinear();
    let residual_target = nonlinear
        .absolute_tolerance()
        .max(nonlinear.relative_tolerance() * newton.initial_residual_norm);
    if !residual_target.is_finite() || final_residual_norm > residual_target {
        return Err(invalid(format!(
            "independently reassembled ALE FSI residual {final_residual_norm:e} exceeds target {residual_target:e}"
        )));
    }

    let continuity_residual_norm = finite_norm(
        &independent
            .layout
            .full_pressure_dofs()
            .into_iter()
            .map(|dof| independent.full_fluid_residual()[dof])
            .collect::<Vec<_>>(),
        "ALE FSI weak incompressibility residual",
    )?;
    let algebraic_scale = finite_norm(independent.algebraic_values(), "ALE FSI accepted point")?;
    let dimensionless_tolerance =
        residual_target + 8_192.0 * f64::EPSILON * (1.0 + algebraic_scale + final_residual_norm);
    if continuity_residual_norm > dimensionless_tolerance {
        return Err(invalid(format!(
            "ALE FSI incompressibility residual {continuity_residual_norm:e} exceeds {dimensionless_tolerance:e}"
        )));
    }

    let current = independent.current_state();
    let kinematic_residual_norm = solid_kinematic_residual_norm(partition, previous, current, plan);
    let kinematic_tolerance = 8_192.0
        * f64::EPSILON
        * plan.scale().length()
        * (1.0 + partition.solid_vertices().len() as f64).sqrt();
    if !kinematic_residual_norm.is_finite() || kinematic_residual_norm > kinematic_tolerance {
        return Err(invalid(format!(
            "ALE FSI solid kinematic residual {kinematic_residual_norm:e} exceeds {kinematic_tolerance:e}"
        )));
    }

    // The conforming quotient owns one velocity coefficient at each shared
    // vertex. There is no copied side value from which a nonzero jump could be
    // hidden; this structural zero is still carried in public evidence.
    let interface_velocity_jump_norm = 0.0;
    let action_scale = plan.scale().action();
    let power_scale = plan.scale().power();
    let interface_actions = recover_interface_actions(partition, &independent, action_scale)?;
    let interface_action_imbalance_norm = interface_actions
        .iter()
        .flat_map(|action| action.imbalance())
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let interface_power_imbalance = interface_actions
        .iter()
        .try_fold(0.0, |sum, action| -> Result<f64, Diagnostic> {
            let velocity = current
                .vertex_velocity()
                .get(action.vertex().index())
                .copied()
                .ok_or_else(|| invalid("ALE FSI interface action vertex is outside the state"))?;
            let next = sum + action.power_imbalance(velocity)?;
            if next.is_finite() {
                Ok(next)
            } else {
                Err(invalid("ALE FSI interface power accumulation overflowed"))
            }
        })?
        .abs();
    let maximum_interface_velocity = interface_actions
        .iter()
        .flat_map(|action| current.vertex_velocity()[action.vertex().index()])
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    let interface_action_tolerance = residual_target * action_scale
        + 16_384.0 * f64::EPSILON * action_scale * (1.0 + interface_actions.len() as f64).sqrt();
    let interface_power_tolerance = interface_action_tolerance * maximum_interface_velocity
        + 16_384.0 * f64::EPSILON * power_scale;
    if interface_actions.is_empty()
        || !interface_action_imbalance_norm.is_finite()
        || interface_action_imbalance_norm > interface_action_tolerance
    {
        return Err(invalid(format!(
            "ALE FSI interface action imbalance {interface_action_imbalance_norm:e} exceeds {interface_action_tolerance:e}"
        )));
    }
    if !interface_power_imbalance.is_finite()
        || interface_power_imbalance > interface_power_tolerance
    {
        return Err(invalid(format!(
            "ALE FSI interface power imbalance {interface_power_imbalance:e} exceeds {interface_power_tolerance:e}"
        )));
    }

    let free_stream = compatible_constant_free_stream_probe(
        partition,
        independent.geometry_action(),
        plan,
        quadrature,
    )?;

    let evidence = AleFsiStepEvidence::<D>::new(
        plan,
        independent.geometry_action(),
        current,
        AleFsiStepEvidenceInput::<D> {
            nonlinear_iterations: newton.iterations,
            initial_residual_norm: newton.initial_residual_norm,
            final_residual_norm,
            continuity_residual_norm,
            kinematic_residual_norm,
            interface_velocity_jump_norm,
            interface_actions,
            probed_moving_fluid_cell_count: free_stream.probed_moving_fluid_cell_count,
            gcl_active_moving_fluid_cell_count: free_stream.gcl_active_moving_fluid_cell_count,
            compatible_constant_free_stream_residual_norm: free_stream.residual_norm,
            omitted_gcl_witness_norm: free_stream.omitted_gcl_witness_norm,
            assembly_report: *independent.assembly_report(),
            nonlinear_linear_solves: newton.linear_solves,
        },
    )?;
    Ok((independent.current, evidence))
}

/// Reassemble acceptance evidence through the run-scoped immutable structure.
#[allow(clippy::too_many_arguments)]
pub(super) fn accept_step_prepared<const D: usize>(
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    structure: &PreparedAleFsiStructure<D>,
    action: &PreparedAleFsiAction<D>,
    motion: &P1HarmonicMeshMotionAction<D>,
    previous: &AleFsiState<D>,
    plan: AleFsiStepPlan<D>,
    quadrature: &QuadratureRule,
    assembly_backend: &dyn AssemblyBackend,
    converged: StepAssembly<D>,
    newton: NewtonEvidence,
) -> Result<(AleFsiState<D>, AleFsiStepEvidence<D>), Diagnostic> {
    let independent = assemble_step_linearization_with_structure(
        reference,
        partition,
        structure,
        action,
        motion,
        previous,
        converged.algebraic_values(),
        plan,
        quadrature,
        assembly_backend,
    )?;
    accept_independent(
        partition,
        previous,
        plan,
        quadrature,
        converged,
        independent,
        newton,
    )
}

struct ConstantFreeStreamProbe {
    probed_moving_fluid_cell_count: usize,
    gcl_active_moving_fluid_cell_count: usize,
    residual_norm: f64,
    omitted_gcl_witness_norm: f64,
}

/// Probe the exact moving-cell action with a constant trial velocity.
///
/// Only the MINI bubble momentum rows are observed: their test function has
/// exact zero trace on each cell, so the local integration-by-parts identity
/// is compatible without pretending that a nonzero free stream satisfies the
/// problem's homogeneous exterior boundary. Pointwise continuity rows are
/// included as a separate zero check. The omitted-GCL witness is evaluated
/// independently from the explicit `0.5 div(w) u v` term.
fn compatible_constant_free_stream_probe<const D: usize>(
    partition: &FixedReferenceFsiPartition<D>,
    geometry: &FixedTopologyGeometryAction<D>,
    plan: AleFsiStepPlan<D>,
    quadrature: &QuadratureRule,
) -> Result<ConstantFreeStreamProbe, Diagnostic> {
    let velocity_scale = plan.scale().velocity();
    let pressure_scale = plan.scale().pressure();
    let power_scale = plan.scale().power();
    let momentum_row_scale = velocity_scale / power_scale;
    let continuity_row_scale = pressure_scale / power_scale;
    let constant = std::array::from_fn(|axis| velocity_scale * (-0.5_f64).powi(axis as i32));
    let mut coefficients = vec![constant; D + 2];
    coefficients[D + 1] = [0.0; D];
    let zero_velocity = vec![[0.0; D]; D + 2];
    let zero_pressure = vec![0.0; D + 1];
    let bubble_space = SimplexP1BubbleSpace::new(D)?;
    let mut probed_moving_fluid_cell_count = 0_usize;
    let mut gcl_active_moving_fluid_cell_count = 0_usize;
    let mut residual_squared = 0.0;
    let mut omitted_squared = 0.0;

    for cell in partition.fluid_cells() {
        let cell_geometry = geometry.cell(cell.index()).ok_or_else(|| {
            invalid("constant-free-stream probe cannot find one admitted fluid-cell geometry")
        })?;
        if cell_geometry.previous_map() == cell_geometry.current_map() {
            continue;
        }
        probed_moving_fluid_cell_count += 1;
        let stationary =
            AffineGeometryLinearization::stationary(cell_geometry.current_map().clone())?;
        let evaluated = AleMiniFluidCell::<D> {
            geometry: cell_geometry,
            density: plan.material().fluid_density(),
            viscosity: plan.material().fluid_dynamic_viscosity(),
            time_step: plan.time_step(),
            previous_velocity: &coefficients,
            current_velocity: &coefficients,
            current_pressure: &zero_pressure,
        }
        .evaluate(
            AleMiniFluidDirection::<D> {
                current_velocity: &zero_velocity,
                current_pressure: &zero_pressure,
                current_geometry: &stationary,
            },
            quadrature,
        )?;
        for component in 0..D {
            let value =
                momentum_row_scale * evaluated.residual()[local_velocity::<D>(D + 1, component)];
            residual_squared += value * value;
        }
        let pressure_offset = local_pressure_offset::<D>();
        if evaluated.residual().len() != pressure_offset + D + 1 {
            return Err(invalid(format!(
                "{D}D constant-free-stream probe received a malformed MINI residual"
            )));
        }
        for value in &evaluated.residual()[pressure_offset..] {
            let value = continuity_row_scale * value;
            residual_squared += value * value;
        }

        let divergence = cell_geometry.current_velocity_divergence();
        if divergence != 0.0 {
            gcl_active_moving_fluid_cell_count += 1;
            let bubble_integral = quadrature
                .points()
                .iter()
                .map(|point| -> Result<f64, Diagnostic> {
                    Ok(point.weight
                        * cell_geometry.current_map().measure_scale()
                        * bubble_space.tabulate(&point.coordinates)?.values()[D + 1])
                })
                .sum::<Result<f64, _>>()?;
            for component in constant {
                let omitted = momentum_row_scale
                    * 0.5
                    * plan.material().fluid_density()
                    * divergence
                    * component
                    * bubble_integral;
                omitted_squared += omitted * omitted;
            }
        }
    }
    let residual_norm = residual_squared.sqrt();
    let omitted_gcl_witness_norm = omitted_squared.sqrt();
    if !residual_norm.is_finite() || !omitted_gcl_witness_norm.is_finite() {
        return Err(invalid(
            "constant-free-stream probe produced non-finite residual or omitted-GCL evidence",
        ));
    }
    Ok(ConstantFreeStreamProbe {
        probed_moving_fluid_cell_count,
        gcl_active_moving_fluid_cell_count,
        residual_norm,
        omitted_gcl_witness_norm,
    })
}

fn require_same_accepted_point<const D: usize>(
    converged: &StepAssembly<D>,
    independent: &StepAssembly<D>,
) -> Result<(), Diagnostic> {
    if converged.current_state() != independent.current_state()
        || converged.geometry_action() != independent.geometry_action()
        || converged.algebraic_values() != independent.algebraic_values()
    {
        return Err(invalid(
            "independent ALE FSI reassembly changed its accepted state or geometry action",
        ));
    }
    if converged.residual().len() != independent.residual().len() {
        return Err(invalid(
            "independent ALE FSI reassembly changed residual shape",
        ));
    }
    let difference = converged
        .residual()
        .iter()
        .zip(independent.residual())
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        .sqrt();
    let scale = finite_norm(converged.residual(), "converged ALE FSI residual")?.max(finite_norm(
        independent.residual(),
        "independently reassembled ALE FSI residual",
    )?);
    let tolerance = 4_096.0 * f64::EPSILON * (1.0 + scale);
    if !difference.is_finite() || difference > tolerance {
        return Err(invalid(format!(
            "independent ALE FSI reassembly residual difference {difference:e} exceeds {tolerance:e}"
        )));
    }
    Ok(())
}

fn recover_interface_actions<const D: usize>(
    partition: &FixedReferenceFsiPartition<D>,
    assembly: &StepAssembly<D>,
    action_scale: f64,
) -> Result<Vec<AleFsiInterfaceAction<D>>, Diagnostic> {
    partition
        .interface_vertices()
        .iter()
        .copied()
        .filter(|vertex| !assembly.layout.fixed_velocity(vertex.index()))
        .map(|vertex| {
            let fluid = std::array::from_fn(|component| {
                action_scale
                    * assembly.full_fluid_residual()[assembly
                        .layout
                        .full_vertex_velocity(vertex.index(), component)]
            });
            let solid = std::array::from_fn(|component| {
                action_scale
                    * assembly.full_solid_residual()[assembly
                        .layout
                        .full_vertex_velocity(vertex.index(), component)]
            });
            AleFsiInterfaceAction::<D>::new(vertex, fluid, solid)
        })
        .collect()
}

fn solid_kinematic_residual_norm<const D: usize>(
    partition: &FixedReferenceFsiPartition<D>,
    previous: &AleFsiState<D>,
    current: &AleFsiState<D>,
    plan: AleFsiStepPlan<D>,
) -> f64 {
    partition
        .solid_vertices()
        .iter()
        .flat_map(|vertex| {
            (0..D).map(move |component| {
                current.solid_displacement()[vertex.index()][component]
                    - previous.solid_displacement()[vertex.index()][component]
                    - plan.time_step() * current.vertex_velocity()[vertex.index()][component]
            })
        })
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
}

const fn local_velocity<const D: usize>(basis: usize, component: usize) -> usize {
    basis * D + component
}

const fn local_pressure_offset<const D: usize>() -> usize {
    (D + 2) * D
}

fn finite_norm(values: &[f64], name: &'static str) -> Result<f64, Diagnostic> {
    let value = values.iter().map(|value| value * value).sum::<f64>().sqrt();
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid(format!("{name} must be finite")))
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use eqiora_assembly::{LocalUnknown, REFERENCE_ASSEMBLY_BACKEND};
    use eqiora_meshing::{
        CellId, FacetId, MeshEntity, MeshQualityGate, MeshTopology, VertexId,
        simplex_duffy_gauss_legendre,
    };
    use eqiora_realization::{NonlinearSolvePlan, Target};
    use eqiora_solver::{
        LinearSolveRequest, LinearSolver, PreconditionerPolicy, REFERENCE_LINEAR_SOLVER,
        ReductionPolicy, SolverPlan,
    };

    use super::*;
    use crate::simplicial_ale_fsi::assembly::{assemble_step_linearization, initial_point};
    use crate::simplicial_ale_fsi::{
        AleFsiBoundary, AleFsiState, AleFsiStepPlan, P1HarmonicMeshMotionAction,
    };
    use crate::simplicial_fsi::{
        FixedReferenceFsiLoad, FixedReferenceFsiMaterial, FixedReferenceFsiPartition,
        FixedReferenceFsiScale,
    };

    const INTERFACE_INTERIOR: VertexId = VertexId::new(5);

    struct Fixture3d {
        mesh: SimplicialMesh,
        partition: FixedReferenceFsiPartition<3>,
        layout: crate::simplicial_fsi::layout::FsiLayout<3>,
        boundary: AleFsiBoundary<3>,
        motion: P1HarmonicMeshMotionAction<3>,
        previous: AleFsiState<3>,
        plan: AleFsiStepPlan<3>,
    }

    #[test]
    fn tetrahedral_acceptance_probes_every_component_and_fail_closed() {
        let fixture = fixture_3d();
        let degree_nine = simplex_duffy_gauss_legendre(3, 6).unwrap();
        let quadrature = simplex_duffy_gauss_legendre(3, 7).unwrap();
        assert_eq!(quadrature.polynomial_exactness(), Some(11));

        let mut point = initial_point(
            &fixture.mesh,
            &fixture.boundary,
            &fixture.motion,
            &fixture.previous,
            fixture.plan,
            &quadrature,
            &fixture.layout,
        )
        .unwrap();
        let initial = assemble(&fixture, &point, &quadrature);
        let (fluid_cell, vertices, basis) = interface_fluid_cell(&fixture);
        let map = initial
            .layout
            .fluid_map(fluid_cell, &vertices, true)
            .unwrap();
        let dimensionless_velocity = [0.02, -0.01, 0.015];
        for (component, value) in dimensionless_velocity.iter().copied().enumerate() {
            let local = local_velocity::<3>(basis, component);
            let LocalUnknown::Free(dof) = map.unknowns()[local] else {
                panic!("the interface-interior velocity must be a free quotient unknown");
            };
            point[dof.index()] = value;
        }

        let mut assembled = assemble(&fixture, &point, &quadrature);
        let current = assembled.current_state();
        assert_eq!(
            current.vertex_velocity()[INTERFACE_INTERIOR.index()][2],
            fixture.plan.scale().velocity() * dimensionless_velocity[2]
        );
        let kinematic = solid_kinematic_residual_norm(
            &fixture.partition,
            &fixture.previous,
            current,
            fixture.plan,
        );
        assert_eq!(kinematic, 0.0);

        let mut defective_displacement = current.solid_displacement().to_vec();
        defective_displacement[INTERFACE_INTERIOR.index()][2] += 1.0e-4;
        let defective = AleFsiState::<3>::new(
            current.time(),
            &fixture.mesh,
            &fixture.partition,
            &fixture.motion,
            current.vertex_velocity().to_vec(),
            current.fluid_cell_bubble_velocity().clone(),
            current.fluid_pressure().to_vec(),
            defective_displacement,
        )
        .unwrap();
        let defective_norm = solid_kinematic_residual_norm(
            &fixture.partition,
            &fixture.previous,
            &defective,
            fixture.plan,
        );
        let kinematic_tolerance = 8_192.0
            * f64::EPSILON
            * fixture.plan.scale().length()
            * (1.0 + fixture.partition.solid_vertices().len() as f64).sqrt();
        assert!(defective_norm > kinematic_tolerance);
        assert!((defective_norm - 1.0e-4).abs() < 8.0 * f64::EPSILON);

        let probe = compatible_constant_free_stream_probe(
            &fixture.partition,
            assembled.geometry_action(),
            fixture.plan,
            &quadrature,
        )
        .unwrap();
        assert!(probe.probed_moving_fluid_cell_count > 0);
        assert!(probe.gcl_active_moving_fluid_cell_count > 0);
        assert!(probe.omitted_gcl_witness_norm > 0.0);
        assert_eq!(
            probe.omitted_gcl_witness_norm,
            expected_omitted_gcl_witness(
                &fixture.partition,
                assembled.geometry_action(),
                fixture.plan,
                &quadrature,
            )
        );
        assert!(
            probe.residual_norm <= 65_536.0 * f64::EPSILON * (1.0 + probe.omitted_gcl_witness_norm)
        );
        assert!(
            compatible_constant_free_stream_probe(
                &fixture.partition,
                assembled.geometry_action(),
                fixture.plan,
                &degree_nine,
            )
            .is_err()
        );

        let actions = recover_interface_actions(
            &fixture.partition,
            &assembled,
            fixture.plan.scale().action(),
        )
        .unwrap();
        let action = actions
            .iter()
            .find(|action| action.vertex() == INTERFACE_INTERIOR)
            .copied()
            .unwrap();
        for component in 0..3 {
            let full = assembled
                .layout
                .full_vertex_velocity(INTERFACE_INTERIOR.index(), component);
            assert_eq!(
                action.fluid()[component],
                fixture.plan.scale().action() * assembled.full_fluid_residual()[full]
            );
            assert_eq!(
                action.solid()[component],
                fixture.plan.scale().action() * assembled.full_solid_residual()[full]
            );
        }
        assert!(action.fluid()[2] != 0.0 || action.solid()[2] != 0.0);

        let third = assembled
            .layout
            .full_vertex_velocity(INTERFACE_INTERIOR.index(), 2);
        assembled.full_fluid_residual[third] = f64::INFINITY;
        assert!(
            recover_interface_actions(
                &fixture.partition,
                &assembled,
                fixture.plan.scale().action(),
            )
            .is_err()
        );
    }

    fn assemble(
        fixture: &Fixture3d,
        point: &[f64],
        quadrature: &QuadratureRule,
    ) -> StepAssembly<3> {
        assemble_step_linearization(
            &fixture.mesh,
            &fixture.partition,
            &fixture.boundary,
            &fixture.motion,
            &fixture.previous,
            point,
            fixture.plan,
            quadrature,
            &REFERENCE_ASSEMBLY_BACKEND,
            &fixture.layout,
        )
        .unwrap()
    }

    fn interface_fluid_cell(
        fixture: &Fixture3d,
    ) -> (eqiora_meshing::CellId, Vec<MeshEntity>, usize) {
        fixture
            .partition
            .fluid_cells()
            .iter()
            .find_map(|cell| {
                let vertices = fixture
                    .mesh
                    .entity_vertices(MeshEntity::new(3, cell.index()))?
                    .to_vec();
                let basis = vertices
                    .iter()
                    .position(|vertex| vertex.index() == INTERFACE_INTERIOR.index())?;
                Some((*cell, vertices, basis))
            })
            .expect("the bounded tetrahedral fixture has one fluid interface cell")
    }

    fn expected_omitted_gcl_witness(
        partition: &FixedReferenceFsiPartition<3>,
        geometry: &FixedTopologyGeometryAction<3>,
        plan: AleFsiStepPlan<3>,
        quadrature: &QuadratureRule,
    ) -> f64 {
        let bubble_space = SimplexP1BubbleSpace::new(3).unwrap();
        let momentum_row_scale = plan.scale().velocity() / plan.scale().power();
        let constant: [f64; 3] =
            std::array::from_fn(|axis| plan.scale().velocity() * (-0.5_f64).powi(axis as i32));
        let mut squared = 0.0;
        for cell in partition.fluid_cells() {
            let cell = geometry.cell(cell.index()).unwrap();
            let divergence = cell.current_velocity_divergence();
            if divergence == 0.0 {
                continue;
            }
            let bubble_integral = quadrature
                .points()
                .iter()
                .map(|point| {
                    point.weight
                        * cell.current_map().measure_scale()
                        * bubble_space.tabulate(&point.coordinates).unwrap().values()[4]
                })
                .sum::<f64>();
            for component in constant {
                let omitted = momentum_row_scale
                    * 0.5
                    * plan.material().fluid_density()
                    * divergence
                    * component
                    * bubble_integral;
                squared += omitted * omitted;
            }
        }
        squared.sqrt()
    }

    fn fixture_3d() -> Fixture3d {
        let (mesh, fluid, solid, interface) = tetrahedral_problem();
        let partition =
            FixedReferenceFsiPartition::<3>::new(&mesh, fluid, solid, interface).unwrap();
        let boundary = AleFsiBoundary::<3>::homogeneous_exterior(&mesh).unwrap();
        let motion =
            P1HarmonicMeshMotionAction::<3>::new(&mesh, &partition, harmonic_solver()).unwrap();
        let previous = AleFsiState::<3>::new(
            0.0,
            &mesh,
            &partition,
            &motion,
            vec![[0.0; 3]; mesh.vertices().len()],
            partition
                .fluid_cells()
                .iter()
                .copied()
                .map(|cell| (cell, [0.0; 3]))
                .collect(),
            vec![0.0; partition.fluid_vertices().len()],
            vec![[0.0; 3]; mesh.vertices().len()],
        )
        .unwrap();
        let plan = step_plan_3d();
        let layout = crate::simplicial_fsi::test_model::polyhedra::polyhedral_layout(
            &crate::simplicial_fsi::test_model::polyhedra::tetrahedral_geometry(),
            &mesh,
            &partition,
            &crate::simplicial_fsi::FixedReferenceFsiBoundary::homogeneous_exterior(&mesh).unwrap(),
            plan.fixed_reference_config(),
            plan.linear_solver(),
            true,
        );
        Fixture3d {
            layout,
            mesh,
            partition,
            boundary,
            motion,
            previous,
            plan: step_plan_3d(),
        }
    }

    fn tetrahedral_problem() -> (SimplicialMesh, Vec<CellId>, Vec<CellId>, Vec<FacetId>) {
        let vertices = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![-1.0, 0.0, 0.0],
            vec![-0.25, 0.25, 0.25],
            vec![0.0, 1.0 / 3.0, 1.0 / 3.0],
            vec![1.0, 0.0, 0.0],
        ];
        let mut cells = vec![
            vec![4, 5, 0, 1],
            vec![4, 5, 1, 2],
            vec![4, 5, 2, 0],
            vec![4, 3, 1, 2],
            vec![4, 3, 2, 0],
            vec![4, 3, 0, 1],
            vec![6, 5, 0, 1],
            vec![6, 5, 1, 2],
            vec![6, 5, 2, 0],
        ];
        for cell in &mut cells {
            if signed_tetrahedron_measure(&vertices, cell) < 0.0 {
                cell.swap(1, 2);
            }
        }
        let fluid = (0..6).map(CellId::new).collect();
        let solid = (6..9).map(CellId::new).collect();
        let mesh =
            SimplicialMesh::new(3, vertices, cells, MeshQualityGate::new(0.005).unwrap()).unwrap();
        let interface = (0..mesh.entity_count(2).unwrap())
            .filter(|&facet| {
                mesh.entity_vertices(MeshEntity::new(2, facet))
                    .unwrap()
                    .iter()
                    .all(|vertex| mesh.vertices()[vertex.index()][0] == 0.0)
            })
            .map(FacetId::new)
            .collect();
        (mesh, fluid, solid, interface)
    }

    fn signed_tetrahedron_measure(vertices: &[Vec<f64>], cell: &[usize]) -> f64 {
        let origin = &vertices[cell[0]];
        let column = |vertex: usize, axis: usize| vertices[cell[vertex]][axis] - origin[axis];
        column(1, 0) * (column(2, 1) * column(3, 2) - column(3, 1) * column(2, 2))
            - column(2, 0) * (column(1, 1) * column(3, 2) - column(3, 1) * column(1, 2))
            + column(3, 0) * (column(1, 1) * column(2, 2) - column(2, 1) * column(1, 2))
    }

    fn step_plan_3d() -> AleFsiStepPlan<3> {
        let nonlinear =
            NonlinearSolvePlan::new(1.0e-9, 1.0e-12, NonZeroUsize::new(20).unwrap(), 12).unwrap();
        let linear = SolverPlan::new(
            LinearSolver::BiConjugateGradientStabilized,
            1.0e-10,
            1.0e-12,
            NonZeroUsize::new(500).unwrap(),
        )
        .unwrap()
        .with_preconditioner(PreconditionerPolicy::Identity)
        .with_reduction(ReductionPolicy::Fast);
        AleFsiStepPlan::<3>::new(
            0.05,
            FixedReferenceFsiMaterial::<3>::new(1.0, 0.1, 1.0, 2.0, 1.0).unwrap(),
            FixedReferenceFsiScale::<3>::new(2.0, 5.0, 3.0).unwrap(),
            FixedReferenceFsiLoad::Zero,
            nonlinear,
            linear,
            Target::HostCpu {
                threads: NonZeroUsize::MIN,
            },
        )
        .unwrap()
    }

    fn harmonic_solver() -> LinearSolveRequest<'static> {
        let plan = SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-12,
            1.0e-14,
            NonZeroUsize::new(500).unwrap(),
        )
        .unwrap();
        LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan)
    }
}
