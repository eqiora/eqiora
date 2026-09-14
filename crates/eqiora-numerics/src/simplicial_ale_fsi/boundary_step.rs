//! Prepared physical traces for one private ALE FSI step.

use std::collections::{BTreeMap, BTreeSet};

use eqiora_assembly::AssemblyBackend;
use eqiora_core::{Diagnostic, ScalarType};
use eqiora_meshing::{MeshEntity, MeshTopology, QuadratureRule, SimplicialMesh, VertexId};
use eqiora_solver::{LinearOperatorProperties, LinearSolverBackend};

use super::api::AleFsiStepEvidence;
use super::contract::{AleFsiBoundary, AleFsiState, AleFsiStepPlan};
use super::newton::solve_one_step;
use super::{P1HarmonicMeshMotionAction, invalid};
use crate::simplicial_fsi::{
    FixedReferenceFsiBoundary, FixedReferenceFsiPartition, layout::FsiLayout, validate_problem,
};

/// Opaque, exact identity supplied by the private campaign owner.
///
/// The four words retain member, schedule, step, and ramp identity even when
/// two endpoints share a physical time or trace. The prepared-boundary owner
/// compares the words; it never interprets or invents campaign identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AleFsiBoundaryEndpointIdentity {
    words: [u64; 4],
    time_bits: u64,
}

impl AleFsiBoundaryEndpointIdentity {
    pub(crate) fn new(words: [u64; 4], time: f64) -> Result<Self, Diagnostic> {
        if !time.is_finite() || time < 0.0 || (time == 0.0 && time.to_bits() != 0.0_f64.to_bits()) {
            return Err(invalid(
                "ALE FSI boundary endpoint time must be finite and non-negative",
            ));
        }
        Ok(Self {
            words,
            time_bits: time.to_bits(),
        })
    }

    pub(crate) const fn time_bits(self) -> u64 {
        self.time_bits
    }
}

/// Complete classification of one exterior facet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum AleFsiExteriorFacetDisposition {
    EssentialVelocity,
    NaturalOutflow,
}

/// One exact previous/current physical trace and its derived quotient cache.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreparedAleFsiBoundaryStep<const D: usize> {
    previous_endpoint: AleFsiBoundaryEndpointIdentity,
    current_endpoint: AleFsiBoundaryEndpointIdentity,
    boundary: FixedReferenceFsiBoundary<D>,
    exterior_facets: Vec<(MeshEntity, AleFsiExteriorFacetDisposition)>,
}

/// Run-scoped invariant physical-boundary preparation.
///
/// The homogeneous public path classifies exterior facets and prescribed
/// components once. Individual actions bind only their endpoint identities;
/// the already admitted physical and quotient values are reused unchanged.
pub(super) struct PreparedAleFsiBoundaryRun<const D: usize> {
    template: PreparedAleFsiBoundaryStep<D>,
    homogeneous: bool,
    #[cfg(test)]
    normalization_count: usize,
}

// Construction rejects every non-finite word, so derived `PartialEq` is
// reflexive for every admitted prepared step.
impl<const D: usize> Eq for PreparedAleFsiBoundaryStep<D> {}

impl<const D: usize> PreparedAleFsiBoundaryStep<D> {
    pub(super) fn from_boundary(boundary: &FixedReferenceFsiBoundary<D>) -> Option<Self> {
        let (previous_words, previous_time_bits) = boundary.prepared_previous_endpoint()?;
        let (current_words, current_time_bits) = boundary.prepared_current_endpoint()?;
        let exterior_facets = boundary
            .prepared_exterior_facets()?
            .into_iter()
            .map(|(facet, essential)| {
                (
                    facet,
                    if essential {
                        AleFsiExteriorFacetDisposition::EssentialVelocity
                    } else {
                        AleFsiExteriorFacetDisposition::NaturalOutflow
                    },
                )
            })
            .collect();
        Some(Self {
            previous_endpoint: AleFsiBoundaryEndpointIdentity {
                words: previous_words,
                time_bits: previous_time_bits,
            },
            current_endpoint: AleFsiBoundaryEndpointIdentity {
                words: current_words,
                time_bits: current_time_bits,
            },
            boundary: boundary.clone(),
            exterior_facets,
        })
    }

    /// Prepare the private canonical P1 trace at exact `U0 = 2 m/s`.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) fn canonical_p1(
        mesh: &SimplicialMesh,
        previous_endpoint: AleFsiBoundaryEndpointIdentity,
        current_endpoint: AleFsiBoundaryEndpointIdentity,
        previous_values: Vec<(VertexId, [f64; D])>,
        current_values: Vec<(VertexId, [f64; D])>,
        exterior_facets: Vec<(MeshEntity, AleFsiExteriorFacetDisposition)>,
        velocity_scale: f64,
    ) -> Result<Self, Diagnostic> {
        if velocity_scale.to_bits() != 2.0_f64.to_bits() {
            return Err(invalid(
                "canonical ALE FSI inlet preparation requires exact U0 = 2 m/s",
            ));
        }
        let prepared = Self::prepare(
            mesh,
            previous_endpoint,
            current_endpoint,
            previous_values,
            current_values,
            exterior_facets,
            velocity_scale,
            true,
        )?;
        if prepared
            .exterior_facets
            .iter()
            .all(|(_, disposition)| *disposition != AleFsiExteriorFacetDisposition::NaturalOutflow)
        {
            return Err(invalid(
                "canonical ALE FSI inlet preparation requires an explicit natural outflow facet",
            ));
        }
        Ok(prepared)
    }

    /// Preserve the established homogeneous public boundary through the same
    /// prepared-step path without imposing the private FSI3 scale.
    pub(super) fn homogeneous(
        mesh: &SimplicialMesh,
        boundary: &FixedReferenceFsiBoundary<D>,
        previous_time: f64,
        current_time: f64,
        velocity_scale: f64,
    ) -> Result<Self, Diagnostic> {
        let previous_endpoint =
            AleFsiBoundaryEndpointIdentity::new([0, previous_time.to_bits(), 0, 0], previous_time)?;
        let current_endpoint =
            AleFsiBoundaryEndpointIdentity::new([0, current_time.to_bits(), 0, 0], current_time)?;
        let values = boundary
            .fixed_zero_velocity_vertices()
            .iter()
            .copied()
            .map(|vertex| (vertex, [0.0; D]))
            .collect::<Vec<_>>();
        let facets = exterior_facets(mesh)?
            .into_iter()
            .map(|facet| (facet, AleFsiExteriorFacetDisposition::EssentialVelocity))
            .collect();
        Self::prepare(
            mesh,
            previous_endpoint,
            current_endpoint,
            values.clone(),
            values,
            facets,
            velocity_scale,
            false,
        )
    }

    fn homogeneous_at_times(
        &self,
        previous_time: f64,
        current_time: f64,
    ) -> Result<Self, Diagnostic> {
        let previous_endpoint =
            AleFsiBoundaryEndpointIdentity::new([0, previous_time.to_bits(), 0, 0], previous_time)?;
        let current_endpoint =
            AleFsiBoundaryEndpointIdentity::new([0, current_time.to_bits(), 0, 0], current_time)?;
        let boundary = FixedReferenceFsiBoundary::from_prepared_velocity(
            previous_endpoint.words,
            previous_endpoint.time_bits,
            current_endpoint.words,
            current_endpoint.time_bits,
            self.previous_physical().to_vec(),
            self.current_physical().to_vec(),
            self.previous_quotient().to_vec(),
            self.current_quotient().to_vec(),
            self.exterior_facets
                .iter()
                .map(|(facet, disposition)| {
                    (
                        *facet,
                        *disposition == AleFsiExteriorFacetDisposition::EssentialVelocity,
                    )
                })
                .collect(),
            false,
        );
        Ok(Self {
            previous_endpoint,
            current_endpoint,
            boundary,
            exterior_facets: self.exterior_facets.clone(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare(
        mesh: &SimplicialMesh,
        previous_endpoint: AleFsiBoundaryEndpointIdentity,
        current_endpoint: AleFsiBoundaryEndpointIdentity,
        previous_values: Vec<(VertexId, [f64; D])>,
        current_values: Vec<(VertexId, [f64; D])>,
        exterior_facets: Vec<(MeshEntity, AleFsiExteriorFacetDisposition)>,
        velocity_scale: f64,
        canonical_velocity_scale: bool,
    ) -> Result<Self, Diagnostic> {
        if !matches!(D, 2 | 3)
            || mesh.topological_dimension() != D
            || !velocity_scale.is_finite()
            || velocity_scale <= 0.0
            || previous_endpoint == current_endpoint
            || previous_endpoint.time_bits() >= current_endpoint.time_bits()
        {
            return Err(invalid(
                "ALE FSI prepared boundary requires one ordered finite endpoint pair on a matching mesh",
            ));
        }
        let previous_physical = physical_inventory(mesh, previous_values, "previous")?;
        let current_physical = physical_inventory(mesh, current_values, "current")?;
        let previous_fixed = prescribed_vertices(&previous_physical);
        let current_fixed = prescribed_vertices(&current_physical);
        if previous_fixed != current_fixed {
            return Err(invalid(
                "ALE FSI prepared boundary endpoints must prescribe the same vertex/component inventory",
            ));
        }
        let exterior_facets = validate_facets::<D>(mesh, exterior_facets, &current_fixed)?;
        let previous_quotient = quotient_inventory(&previous_physical, velocity_scale)?;
        let current_quotient = quotient_inventory(&current_physical, velocity_scale)?;
        let boundary = FixedReferenceFsiBoundary::from_prepared_velocity(
            previous_endpoint.words,
            previous_endpoint.time_bits,
            current_endpoint.words,
            current_endpoint.time_bits,
            previous_physical,
            current_physical,
            previous_quotient,
            current_quotient,
            exterior_facets
                .iter()
                .map(|(facet, disposition)| {
                    (
                        *facet,
                        *disposition == AleFsiExteriorFacetDisposition::EssentialVelocity,
                    )
                })
                .collect(),
            canonical_velocity_scale,
        );
        Ok(Self {
            previous_endpoint,
            current_endpoint,
            boundary,
            exterior_facets,
        })
    }

    #[allow(dead_code)]
    pub(crate) fn previous_endpoint(&self) -> AleFsiBoundaryEndpointIdentity {
        self.previous_endpoint
    }

    #[allow(dead_code)]
    pub(crate) fn current_endpoint(&self) -> AleFsiBoundaryEndpointIdentity {
        self.current_endpoint
    }

    #[allow(dead_code)]
    pub(crate) fn previous_physical(&self) -> &[[Option<f64>; D]] {
        self.boundary
            .prepared_previous_physical()
            .expect("prepared step owns previous physical words")
    }

    #[allow(dead_code)]
    pub(crate) fn current_physical(&self) -> &[[Option<f64>; D]] {
        self.boundary
            .prepared_current_physical()
            .expect("prepared step owns current physical words")
    }

    #[allow(dead_code)]
    pub(crate) fn previous_quotient(&self) -> &[[Option<f64>; D]] {
        self.boundary
            .prepared_previous_quotient()
            .expect("prepared step owns previous quotient words")
    }

    pub(crate) fn current_quotient(&self) -> &[[Option<f64>; D]] {
        self.boundary
            .prepared_current_quotient()
            .expect("prepared step owns current quotient words")
    }

    #[allow(dead_code)]
    pub(crate) fn exterior_facets(&self) -> &[(MeshEntity, AleFsiExteriorFacetDisposition)] {
        &self.exterior_facets
    }

    pub(super) fn as_boundary(&self) -> FixedReferenceFsiBoundary<D> {
        self.boundary.clone()
    }

    pub(super) fn layout(&self, base_layout: &FsiLayout<D>) -> Result<FsiLayout<D>, Diagnostic> {
        base_layout.with_boundary(&self.boundary)
    }

    pub(super) fn validate_inputs(
        &self,
        reference: &SimplicialMesh,
        partition: &FixedReferenceFsiPartition<D>,
        motion: &P1HarmonicMeshMotionAction<D>,
        previous: &AleFsiState<D>,
        plan: AleFsiStepPlan<D>,
        quadrature: &QuadratureRule,
    ) -> Result<(), Diagnostic> {
        motion.validate_reference(reference, partition)?;
        previous.validate_against(reference, partition, motion)?;
        if previous.time().to_bits() != self.previous_endpoint.time_bits()
            || (previous.time() + plan.time_step()).to_bits() != self.current_endpoint.time_bits()
            || (self.boundary.prepared_uses_canonical_velocity_scale()
                && plan.scale().velocity().to_bits() != 2.0_f64.to_bits())
        {
            return Err(invalid(
                "ALE FSI prepared boundary endpoint or velocity-scale identity is stale",
            ));
        }
        for (vertex, components) in self.previous_physical().iter().enumerate() {
            for (component, physical) in components.iter().copied().enumerate() {
                if physical.is_some_and(|expected| {
                    previous.vertex_velocity()[vertex][component].to_bits() != expected.to_bits()
                }) {
                    return Err(invalid(
                        "ALE FSI previous state differs from its prepared physical trace",
                    ));
                }
            }
        }
        let previous_reference = previous.to_fixed_reference_state(reference, partition)?;
        let boundary = self.as_boundary();
        validate_problem(
            reference,
            partition,
            &boundary,
            &previous_reference,
            plan.fixed_reference_config(),
            quadrature,
        )?;
        let required_exactness = 3 * D + 2;
        if quadrature.polynomial_exactness().unwrap_or(0) < required_exactness {
            return Err(invalid(format!(
                "{D}D ALE FSI fluid action requires simplex quadrature exactness at least {required_exactness}"
            )));
        }
        Ok(())
    }

    pub(super) fn validate_action(
        &self,
        previous: &AleFsiState<D>,
        plan: AleFsiStepPlan<D>,
    ) -> Result<(), Diagnostic> {
        if previous.time().to_bits() != self.previous_endpoint.time_bits()
            || (previous.time() + plan.time_step()).to_bits() != self.current_endpoint.time_bits()
            || (self.boundary.prepared_uses_canonical_velocity_scale()
                && plan.scale().velocity().to_bits() != 2.0_f64.to_bits())
        {
            return Err(invalid(
                "ALE FSI prepared boundary endpoint or velocity-scale identity is stale",
            ));
        }
        for (vertex, components) in self.previous_physical().iter().enumerate() {
            for (component, physical) in components.iter().copied().enumerate() {
                if physical.is_some_and(|expected| {
                    previous.vertex_velocity()[vertex][component].to_bits() != expected.to_bits()
                }) {
                    return Err(invalid(
                        "ALE FSI previous state differs from its prepared physical trace",
                    ));
                }
            }
        }
        Ok(())
    }

    pub(super) fn has_same_structure(&self, other: &Self) -> bool {
        self.previous_physical() == other.previous_physical()
            && self.current_physical() == other.current_physical()
            && self.previous_quotient() == other.previous_quotient()
            && self.current_quotient() == other.current_quotient()
            && self.exterior_facets == other.exterior_facets
            && self.boundary.prepared_uses_canonical_velocity_scale()
                == other.boundary.prepared_uses_canonical_velocity_scale()
    }

    pub(super) fn reduce_initial_point(
        &self,
        previous: &AleFsiState<D>,
        plan: AleFsiStepPlan<D>,
        layout: &FsiLayout<D>,
    ) -> Result<Vec<f64>, Diagnostic> {
        let velocity_scale = plan.scale().velocity();
        let pressure_scale = plan.scale().pressure();
        let mut vertex_velocity = previous
            .vertex_velocity()
            .iter()
            .map(|value| value.map(|component| component / velocity_scale))
            .collect::<Vec<_>>();
        for (vertex, components) in self.current_quotient().iter().enumerate() {
            for (component, quotient) in components.iter().copied().enumerate() {
                if let Some(quotient) = quotient {
                    vertex_velocity[vertex][component] = quotient;
                }
            }
        }
        let bubbles = previous
            .fluid_cell_bubble_velocity()
            .iter()
            .map(|(&cell, value)| (cell, value.map(|component| component / velocity_scale)))
            .collect::<std::collections::BTreeMap<_, _>>();
        let pressure = previous
            .fluid_pressure()
            .iter()
            .map(|value| value / pressure_scale)
            .collect::<Vec<_>>();
        layout.reduce(&vertex_velocity, &bubbles, &pressure)
    }

    pub(super) fn reduce_current_point(
        &self,
        current: &AleFsiState<D>,
        plan: AleFsiStepPlan<D>,
        layout: &FsiLayout<D>,
    ) -> Result<Vec<f64>, Diagnostic> {
        if current.time().to_bits() != self.current_endpoint.time_bits() {
            return Err(invalid(
                "ALE FSI verification state differs from its prepared current endpoint",
            ));
        }
        let velocity_scale = plan.scale().velocity();
        let pressure_scale = plan.scale().pressure();
        let velocity = current
            .vertex_velocity()
            .iter()
            .map(|value| value.map(|component| component / velocity_scale))
            .collect::<Vec<_>>();
        let bubbles = current
            .fluid_cell_bubble_velocity()
            .iter()
            .map(|(&cell, value)| (cell, value.map(|component| component / velocity_scale)))
            .collect::<std::collections::BTreeMap<_, _>>();
        let pressure = current
            .fluid_pressure()
            .iter()
            .map(|value| value / pressure_scale)
            .collect::<Vec<_>>();
        layout.reduce(&velocity, &bubbles, &pressure)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn reconstruct_current_state(
        &self,
        reference: &SimplicialMesh,
        partition: &FixedReferenceFsiPartition<D>,
        motion: &P1HarmonicMeshMotionAction<D>,
        previous: &AleFsiState<D>,
        candidate: &[f64],
        plan: AleFsiStepPlan<D>,
        layout: &FsiLayout<D>,
    ) -> Result<AleFsiState<D>, Diagnostic> {
        let (vertex_hat, bubbles_hat, pressure_hat) = layout.reconstruct_primal(candidate)?;
        let velocity_scale = plan.scale().velocity();
        let pressure_scale = plan.scale().pressure();
        let vertex_velocity = vertex_hat
            .iter()
            .map(|value| value.map(|component| component * velocity_scale))
            .collect::<Vec<_>>();
        for (vertex, components) in self.current_physical().iter().enumerate() {
            for (component, physical) in components.iter().copied().enumerate() {
                if physical.is_some_and(|expected| {
                    vertex_velocity[vertex][component].to_bits() != expected.to_bits()
                }) {
                    return Err(invalid(
                        "ALE FSI reconstructed current state differs from its prepared physical trace",
                    ));
                }
            }
        }
        let bubbles = bubbles_hat
            .iter()
            .map(|(&cell, value)| (cell, value.map(|component| component * velocity_scale)))
            .collect::<std::collections::BTreeMap<_, _>>();
        let pressure = pressure_hat
            .iter()
            .map(|value| value * pressure_scale)
            .collect::<Vec<_>>();
        let mut displacement = previous.solid_displacement().to_vec();
        for vertex in partition.solid_vertices() {
            for component in 0..D {
                displacement[vertex.index()][component] +=
                    plan.time_step() * vertex_velocity[vertex.index()][component];
            }
        }
        AleFsiState::<D>::new(
            f64::from_bits(self.current_endpoint.time_bits()),
            reference,
            partition,
            motion,
            vertex_velocity,
            bubbles,
            pressure,
            displacement,
        )
    }

    pub(super) fn build_directions(
        &self,
        partition: &FixedReferenceFsiPartition<D>,
        motion: &P1HarmonicMeshMotionAction<D>,
        plan: AleFsiStepPlan<D>,
        layout: &FsiLayout<D>,
    ) -> Result<Vec<AlgebraicDirection<D>>, Diagnostic> {
        let dimension = layout.reduced_size();
        let mut directions = Vec::new();
        directions
            .try_reserve_exact(dimension)
            .map_err(|_| invalid("ALE FSI direction inventory allocation failed"))?;
        for column in 0..dimension {
            let mut basis = Vec::new();
            basis
                .try_reserve_exact(dimension)
                .map_err(|_| invalid("ALE FSI reduced basis direction allocation failed"))?;
            basis.resize(dimension, 0.0);
            basis[column] = 1.0;
            let (vertex_hat, bubble_hat, pressure_hat) = layout.reconstruct_direction(&basis)?;
            self.require_zero_eliminated_direction(&vertex_hat)?;
            let vertex_velocity = vertex_hat
                .iter()
                .map(|value| value.map(|component| component * plan.scale().velocity()))
                .collect::<Vec<_>>();
            let fluid_bubbles = bubble_hat
                .iter()
                .map(|(&cell, value)| {
                    (
                        cell,
                        value.map(|component| component * plan.scale().velocity()),
                    )
                })
                .collect::<std::collections::BTreeMap<_, _>>();
            let pressure = pressure_hat
                .iter()
                .map(|value| value * plan.scale().pressure())
                .collect::<Vec<_>>();
            let mut displacement = vec![[0.0; D]; vertex_velocity.len()];
            for vertex in partition.solid_vertices() {
                displacement[vertex.index()] =
                    vertex_velocity[vertex.index()].map(|value| plan.time_step() * value);
            }
            directions.push(AlgebraicDirection {
                vertex_velocity,
                fluid_bubbles,
                pressure,
                coordinate: motion.apply_jvp(&displacement)?,
            });
        }
        Ok(directions)
    }

    fn require_zero_eliminated_direction(&self, direction: &[[f64; D]]) -> Result<(), Diagnostic> {
        if direction.len() != self.current_quotient().len() {
            return Err(invalid(
                "ALE FSI direction differs from its prepared vertex inventory",
            ));
        }
        for (vertex, components) in self.current_quotient().iter().enumerate() {
            for (component, fixed) in components.iter().enumerate() {
                if fixed.is_some() && direction[vertex][component].to_bits() != 0.0_f64.to_bits() {
                    return Err(invalid(
                        "ALE FSI direction is nonzero at an eliminated velocity component",
                    ));
                }
            }
        }
        Ok(())
    }
}

impl<const D: usize> PreparedAleFsiBoundaryRun<D> {
    pub(super) fn new(
        mesh: &SimplicialMesh,
        boundary: &AleFsiBoundary<D>,
        initial: &AleFsiState<D>,
        plan: AleFsiStepPlan<D>,
    ) -> Result<Self, Diagnostic> {
        let (template, homogeneous, _normalization_count) =
            match PreparedAleFsiBoundaryStep::from_boundary(boundary) {
                Some(prepared) => (prepared, false, 0),
                None => (
                    PreparedAleFsiBoundaryStep::homogeneous(
                        mesh,
                        boundary,
                        initial.time(),
                        initial.time() + plan.time_step(),
                        plan.scale().velocity(),
                    )?,
                    true,
                    1,
                ),
            };
        Ok(Self {
            template,
            homogeneous,
            #[cfg(test)]
            normalization_count: _normalization_count,
        })
    }

    pub(super) const fn template(&self) -> &PreparedAleFsiBoundaryStep<D> {
        &self.template
    }

    #[cfg(test)]
    pub(super) const fn normalization_count(&self) -> usize {
        self.normalization_count
    }

    pub(super) fn action(
        &self,
        previous: &AleFsiState<D>,
        plan: AleFsiStepPlan<D>,
    ) -> Result<PreparedAleFsiBoundaryStep<D>, Diagnostic> {
        let current_time = previous.time() + plan.time_step();
        let action = if previous.time().to_bits() == self.template.previous_endpoint.time_bits()
            && current_time.to_bits() == self.template.current_endpoint.time_bits()
        {
            self.template.clone()
        } else if self.homogeneous {
            self.template
                .homogeneous_at_times(previous.time(), current_time)?
        } else {
            return Err(invalid(
                "one prepared ALE FSI physical trace cannot be reused at another endpoint pair",
            ));
        };
        if !self.template.has_same_structure(&action) {
            return Err(invalid(
                "ALE FSI action boundary changed its prepared structural closure",
            ));
        }
        action.validate_action(previous, plan)?;
        Ok(action)
    }
}

/// Advance exactly one crate-private prepared-boundary step.
///
/// The established Newton and acceptance path consumes the prepared physical
/// words and explicit facet roles without a public boundary callback.
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn advance_simplicial_ale_fsi_prepared_step<const D: usize>(
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    prepared: &PreparedAleFsiBoundaryStep<D>,
    expected_previous: AleFsiBoundaryEndpointIdentity,
    expected_current: AleFsiBoundaryEndpointIdentity,
    motion: &P1HarmonicMeshMotionAction<D>,
    previous: &AleFsiState<D>,
    plan: AleFsiStepPlan<D>,
    quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend,
    solver: &dyn LinearSolverBackend,
    base_layout: &FsiLayout<D>,
) -> Result<(AleFsiState<D>, AleFsiStepEvidence<D>), Diagnostic> {
    if prepared.previous_endpoint() != expected_previous
        || prepared.current_endpoint() != expected_current
    {
        return Err(invalid(
            "ALE FSI prepared boundary does not match the requested endpoint identities",
        ));
    }
    solver.capabilities().require_problem(
        plan.linear_solver(),
        ScalarType::F64,
        LinearOperatorProperties::General,
    )?;
    previous.validate_against(reference, partition, motion)?;
    prepared.validate_inputs(reference, partition, motion, previous, plan, quadrature)?;
    let boundary = prepared.as_boundary();
    solve_one_step::<D>(
        reference,
        partition,
        &boundary,
        motion,
        previous,
        plan,
        quadrature,
        assembly,
        solver,
        base_layout,
    )
}

pub(super) struct AlgebraicDirection<const D: usize> {
    pub(super) vertex_velocity: Vec<[f64; D]>,
    pub(super) fluid_bubbles: std::collections::BTreeMap<eqiora_meshing::CellId, [f64; D]>,
    pub(super) pressure: Vec<f64>,
    pub(super) coordinate: Vec<[f64; D]>,
}

fn physical_inventory<const D: usize>(
    mesh: &SimplicialMesh,
    values: Vec<(VertexId, [f64; D])>,
    endpoint: &'static str,
) -> Result<Vec<[Option<f64>; D]>, Diagnostic> {
    let mut result = vec![[None; D]; mesh.vertices().len()];
    for (vertex, components) in values {
        let slots = result.get_mut(vertex.index()).ok_or_else(|| {
            invalid(format!(
                "ALE FSI {endpoint} prescribed vertex lies outside the mesh"
            ))
        })?;
        if slots.iter().any(Option::is_some)
            || components.iter().any(|value| {
                !value.is_finite() || (*value == 0.0 && value.to_bits() != 0.0_f64.to_bits())
            })
        {
            return Err(invalid(format!(
                "ALE FSI {endpoint} prescribed inventory is duplicate or contains an invalid word"
            )));
        }
        *slots = components.map(Some);
    }
    Ok(result)
}

fn quotient_inventory<const D: usize>(
    physical: &[[Option<f64>; D]],
    velocity_scale: f64,
) -> Result<Vec<[Option<f64>; D]>, Diagnostic> {
    let mut quotient = vec![[None; D]; physical.len()];
    for (vertex, components) in physical.iter().enumerate() {
        for (component, physical) in components.iter().copied().enumerate() {
            let Some(physical) = physical else {
                continue;
            };
            let value = physical / velocity_scale;
            if !value.is_finite()
                || (value * velocity_scale).to_bits() != physical.to_bits()
                || (physical == 0.0 && physical.to_bits() != 0.0_f64.to_bits())
                || (value == 0.0 && value.to_bits() != 0.0_f64.to_bits())
            {
                return Err(invalid(format!(
                    "ALE FSI prescribed velocity at vertex {vertex} component {component} does not round-trip through its exact quotient"
                )));
            }
            quotient[vertex][component] = Some(value);
        }
    }
    Ok(quotient)
}

fn prescribed_vertices<const D: usize>(values: &[[Option<f64>; D]]) -> BTreeSet<usize> {
    values
        .iter()
        .enumerate()
        .filter_map(|(vertex, components)| components.iter().any(Option::is_some).then_some(vertex))
        .collect()
}

fn validate_facets<const D: usize>(
    mesh: &SimplicialMesh,
    facets: Vec<(MeshEntity, AleFsiExteriorFacetDisposition)>,
    prescribed: &BTreeSet<usize>,
) -> Result<Vec<(MeshEntity, AleFsiExteriorFacetDisposition)>, Diagnostic> {
    let expected = exterior_facets(mesh)?;
    let mut observed = BTreeMap::new();
    let mut essential_vertices = BTreeSet::new();
    for (facet, disposition) in facets {
        if facet.dimension() != D - 1
            || mesh.is_boundary_entity(facet) != Some(true)
            || observed.insert(facet.index(), disposition).is_some()
        {
            return Err(invalid(
                "ALE FSI exterior-facet disposition is duplicate, conflicting, or not a boundary facet",
            ));
        }
        if disposition == AleFsiExteriorFacetDisposition::EssentialVelocity {
            let vertices = mesh
                .entity_vertices(facet)
                .ok_or_else(|| invalid("ALE FSI essential exterior facet has no vertex closure"))?;
            essential_vertices.extend(vertices.into_iter().map(MeshEntity::index));
        }
    }
    if observed.keys().copied().collect::<BTreeSet<_>>()
        != expected.iter().map(|facet| facet.index()).collect()
        || essential_vertices != *prescribed
    {
        return Err(invalid(
            "ALE FSI boundary step must classify every exterior facet exactly once and bind every essential vertex",
        ));
    }
    Ok(expected
        .into_iter()
        .map(|facet| {
            (
                facet,
                *observed
                    .get(&facet.index())
                    .expect("complete disposition owns every exterior facet"),
            )
        })
        .collect())
}

fn exterior_facets(mesh: &SimplicialMesh) -> Result<Vec<MeshEntity>, Diagnostic> {
    let dimension = mesh.topological_dimension();
    if !matches!(dimension, 2 | 3) {
        return Err(invalid(
            "ALE FSI boundary preparation requires a two- or three-dimensional mesh",
        ));
    }
    let count = mesh
        .entity_count(dimension - 1)
        .ok_or_else(|| invalid("ALE FSI mesh omits its exterior-facet stratum"))?;
    Ok((0..count)
        .map(|index| MeshEntity::new(dimension - 1, index))
        .filter(|facet| mesh.is_boundary_entity(*facet) == Some(true))
        .collect())
}

#[cfg(test)]
mod tests;
