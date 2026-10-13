//! Independent acceptance of one converged fixed-topology ALE FSI step.
//!
//! Acceptance reassembles the final nonlinear point, recovers the separate
//! fluid and solid interface actions, and checks residual, kinematic,
//! interface, geometry, and provenance evidence. No fixed-domain energy
//! identity is asserted for the moving control volume.

use eqiora_assembly::AssemblyBackend;
use eqiora_core::Diagnostic;
use eqiora_meshing::FixedTopologyGeometryAction;
use eqiora_meshing::{QuadratureRule, SimplicialMesh};

use super::api::{AleFsiInterfaceAction, AleFsiStepEvidence, AleFsiStepEvidenceInput};
use super::assembly::{
    PreparedAleFsiAction, PreparedAleFsiStructure, StepAssembly,
    assemble_step_linearization_with_structure,
};
use super::contract::{AleFsiState, AleFsiStepPlan};
use super::{P1HarmonicMeshMotionAction, invalid};
use crate::discrete_space::DiscreteSpace;
use crate::simplicial_fsi::FixedReferenceFsiPartition;

use crate::nonlinear_execution::NewtonEvidence;

#[allow(clippy::too_many_arguments)]
fn accept_independent<const D: usize>(
    partition: &FixedReferenceFsiPartition<D>,
    motion: &P1HarmonicMeshMotionAction<D>,
    previous: &AleFsiState<D>,
    plan: &AleFsiStepPlan<D>,
    quadrature: &QuadratureRule,
    converged: StepAssembly<D>,
    independent: StepAssembly<D>,
    newton: NewtonEvidence,
) -> Result<(AleFsiState<D>, AleFsiStepEvidence<D>), Diagnostic> {
    let [harmonic_domain] = motion.policy().harmonic_domains() else {
        return Err(invalid(
            "this ALE acceptance profile requires one transported Domain",
        ));
    };
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

    let continuity_residual = independent.domain_residual(harmonic_domain.erase())?;
    let continuity_residual_norm = finite_norm(
        &independent
            .layout
            .full_pressure_dofs()
            .into_iter()
            .map(|dof| continuity_residual[dof])
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
    let kinematic_residual_norm = solid_kinematic_residual_norm(
        partition,
        motion,
        &independent.layout,
        previous,
        current,
        plan,
    )?;
    let kinematic_tolerance = 8_192.0
        * f64::EPSILON
        * plan.scale().length()
        * (1.0
            + motion
                .policy()
                .drivers()
                .iter()
                .map(|(domain, _)| {
                    partition
                        .domain_vertices(*domain)
                        .expect("authenticated motion Domain")
                        .len()
                })
                .sum::<usize>() as f64)
            .sqrt();
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
    let interface_actions =
        recover_interface_actions(partition, motion, &independent, action_scale)?;
    let fluid_velocity = independent
        .layout
        .velocity_field(harmonic_domain.erase())?
        .downcast()
        .expect("exact velocity Field");
    let current_velocity = current.physical_state().vector_vertices(fluid_velocity)?;
    let interface_action_imbalance_norm = interface_actions
        .iter()
        .flat_map(|action| action.imbalance())
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let interface_power_imbalance = interface_actions
        .iter()
        .try_fold(0.0, |sum, action| -> Result<f64, Diagnostic> {
            let velocity = current_velocity
                .get(&action.vertex())
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
        .flat_map(|action| current_velocity[&action.vertex()])
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
        motion,
        &independent.layout,
        independent.geometry_action(),
        plan,
        quadrature,
    )?;

    let evidence = AleFsiStepEvidence::<D>::new(
        plan.clone(),
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
            assembly_report: independent.assembly_report().clone(),
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
    plan: &AleFsiStepPlan<D>,
    quadrature: &QuadratureRule,
    assembly_backend: &dyn AssemblyBackend<f64>,
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
        motion,
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
    motion: &P1HarmonicMeshMotionAction<D>,
    layout: &crate::simplicial_fsi::layout::FsiLayout<D>,
    geometry: &FixedTopologyGeometryAction<D>,
    plan: &AleFsiStepPlan<D>,
    quadrature: &QuadratureRule,
) -> Result<ConstantFreeStreamProbe, Diagnostic> {
    let [harmonic_domain] = motion.policy().harmonic_domains() else {
        return Err(invalid(
            "this ALE acceptance profile requires one transported Domain",
        ));
    };
    let velocity_scale = plan.scale().velocity();
    let power_scale = plan.scale().power();
    let momentum_row_scale = velocity_scale / power_scale;
    let constant = std::array::from_fn(|axis| velocity_scale * (-0.5_f64).powi(axis as i32));
    let mut coefficients = vec![constant; D + 2];
    coefficients[D + 1] = [0.0; D];
    let bubble_space = DiscreteSpace::new(
        eqiora_realization::Space::simplex_p1_bubble(),
        eqiora_meshing::ReferenceCell::simplex(D)?,
    )?;
    let mut probed_moving_fluid_cell_count = 0_usize;
    let mut gcl_active_moving_fluid_cell_count = 0_usize;
    let mut residual_squared = 0.0;
    let mut omitted_squared = 0.0;

    let fluid_velocity = layout
        .velocity_field(harmonic_domain.erase())?
        .downcast()
        .expect("exact velocity Field");
    for cell in partition
        .domain_cells(*harmonic_domain)
        .expect("authenticated motion Domain")
    {
        let cell_geometry = geometry.cell(cell.index()).ok_or_else(|| {
            invalid("constant-free-stream probe cannot find one admitted fluid-cell geometry")
        })?;
        if cell_geometry.previous_map() == cell_geometry.current_map() {
            continue;
        }
        probed_moving_fluid_cell_count += 1;
        let form = layout.region_form(harmonic_domain.erase())?;
        let velocity = form
            .fields()
            .iter()
            .find(|field| field.field == fluid_velocity.erase())
            .ok_or_else(|| invalid("free-stream probe lacks the transported Field"))?;
        let pressure = layout
            .pressure_field(form.domain())
            .ok_or_else(|| invalid("free-stream probe lacks its constraint Field"))?;
        let pressure = form
            .fields()
            .iter()
            .find(|field| field.field == pressure)
            .ok_or_else(|| invalid("free-stream probe lacks its constraint layout"))?;
        let mut point = vec![0.; form.fields().iter().map(|field| field.range.len()).sum()];
        for (value, physical) in point[velocity.range.clone()]
            .iter_mut()
            .zip(coefficients.iter().flatten())
        {
            *value = physical / velocity.scale;
        }
        let history = std::collections::BTreeMap::from([(
            fluid_velocity.erase(),
            coefficients.iter().flatten().copied().collect(),
        )]);
        let (action, _) = form.prepare_gcl_cell(
            &[fluid_velocity.erase()],
            geometry,
            cell.index(),
            None,
            quadrature,
        )?;
        let residual = action.residual(&history, &point)?;
        for component in 0..D {
            let value = residual[velocity.range.start + (D + 1) * D + component];
            residual_squared += value * value;
        }
        for value in &residual[pressure.range.clone()] {
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
                    * plan
                        .material()
                        .density(fluid_velocity)
                        .expect("authenticated ALE fluid density")
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
    motion: &P1HarmonicMeshMotionAction<D>,
    assembly: &StepAssembly<D>,
    action_scale: f64,
) -> Result<Vec<AleFsiInterfaceAction<D>>, Diagnostic> {
    let ([harmonic_domain], [interface]) = (
        motion.policy().harmonic_domains(),
        motion.policy().interfaces(),
    ) else {
        return Err(invalid(
            "this ALE action acceptance profile requires one interface",
        ));
    };
    let quotient = partition
        .quotients()
        .find(|quotient| {
            quotient.source()
                == eqiora_realization::ConformingTraceSource::ConservingConnection(*interface)
        })
        .expect("authenticated motion Connection");
    let [fluid_endpoint, solid_endpoint] = quotient.endpoints();
    let (fluid_endpoint, solid_endpoint) = if fluid_endpoint.domain() == *harmonic_domain {
        (fluid_endpoint, solid_endpoint)
    } else {
        (solid_endpoint, fluid_endpoint)
    };
    let recovered = assembly.interface_reactions()?;
    let (_, keys) = assembly
        .layout
        .mapping()
        .traces()
        .iter()
        .find(|(trace, _)| *trace == quotient)
        .ok_or_else(|| invalid("reaction recovery has no exact mapped Connection"))?;
    let vertices = keys
        .iter()
        .filter(|key| key.field == fluid_endpoint.field().erase())
        .map(|key| eqiora_meshing::VertexId::new(key.entity.index()))
        .collect::<std::collections::BTreeSet<_>>();
    vertices
        .into_iter()
        .filter(|vertex| {
            !assembly
                .layout
                .fixed_velocity(fluid_endpoint.field().erase(), vertex.index())
        })
        .map(|vertex| {
            let action = |field: eqiora_core::Id<eqiora_core::entity::kinds::Field>| {
                let mut values = [0.; D];
                for (component, value) in values.iter_mut().enumerate() {
                    let key = crate::region_assembly::mapping::FieldDof {
                        field: field.erase(),
                        entity: eqiora_meshing::MeshEntity::new(0, vertex.index()),
                        slot: 0,
                        component,
                    };
                    *value = action_scale * recovered.action(interface.erase(), key)?;
                }
                Ok::<_, Diagnostic>(values)
            };
            let fluid = action(fluid_endpoint.field())?;
            let solid = action(solid_endpoint.field())?;
            AleFsiInterfaceAction::<D>::new(vertex, fluid, solid)
        })
        .collect()
}

fn solid_kinematic_residual_norm<const D: usize>(
    partition: &FixedReferenceFsiPartition<D>,
    motion: &P1HarmonicMeshMotionAction<D>,
    layout: &crate::simplicial_fsi::layout::FsiLayout<D>,
    previous: &AleFsiState<D>,
    current: &AleFsiState<D>,
    plan: &AleFsiStepPlan<D>,
) -> Result<f64, Diagnostic> {
    let mut squared = 0.0;
    for &(domain, displacement) in motion.policy().drivers() {
        let velocity = layout
            .state_rate(displacement.erase())?
            .downcast()
            .expect("exact rate Field");
        let previous_displacement = previous.physical_state().vector_vertices(displacement)?;
        let current_displacement = current.physical_state().vector_vertices(displacement)?;
        let current_velocity = current.physical_state().vector_vertices(velocity)?;
        for vertex in partition
            .domain_vertices(domain)
            .expect("authenticated motion Domain")
        {
            for component in 0..D {
                let value = current_displacement[vertex][component]
                    - previous_displacement[vertex][component]
                    - plan.time_step() * current_velocity[vertex][component];
                squared += value * value;
            }
        }
    }
    Ok(squared.sqrt())
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
mod tests;
