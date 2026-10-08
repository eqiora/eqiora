//! Private Model-owned prescribed velocity traces for steady Stokes.
//!
//! The complete-vector lowering and Model-owned essential replay are consumed
//! only by their precommitted `cfg(test)` evidence until the accepted
//! successor product path starts. Under `cfg(test)` the unused-item lint
//! stays denied.
#![cfg_attr(not(test), allow(dead_code))]

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::{Diagnostic, DimExponents, RawId};
use eqiora_schema::kernel::{ExprNode, KernelNode};
use eqiora_sem::KernelProgram;

use crate::spatial_expression::ScalarSpatialExpression;

use super::support::{is_field, relation_expression, relations_on, unique_root};

/// Exact sealed source label owning the no-slip essential datum.
const BODY_SOURCE_LABEL: &str = "body_no_slip";

/// Exact sealed source labels owning the complete outer essential datum.
const OUTER_SOURCE_LABELS: [&str; 4] = [
    "outer_x_minus",
    "outer_x_plus",
    "outer_y_minus",
    "outer_y_plus",
];

const SPEED_DIMENSION: DimExponents =
    DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension");

/// The complete prescribed-velocity meaning retained from one exact Model.
///
/// `Normal` preserves the established scalar-normal subset.  The complete
/// variant is deliberately restricted to the gradient of one retained scalar
/// affine potential; it is not a general vector expression or callback law.
/// Both variants retain the exact Boundary and boundary Relation identity that
/// carried them, so an equal-valued role permutation is rejected by identity.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SteadyStokesPrescribedVelocityTrace2d {
    Normal {
        boundary: RawId,
        relation: RawId,
        coefficient_field: RawId,
        definition_relation: RawId,
        expression: ScalarSpatialExpression<f64>,
    },
    CompleteAffinePotential {
        boundary: RawId,
        relation: RawId,
        potential_field: RawId,
        definition_relation: RawId,
        speed_parameter: RawId,
        expression: ScalarSpatialExpression<f64>,
    },
}

impl SteadyStokesPrescribedVelocityTrace2d {
    pub(super) fn normal(
        boundary: RawId,
        relation: RawId,
        coefficient_field: RawId,
        definition_relation: RawId,
        expression: ScalarSpatialExpression<f64>,
    ) -> Self {
        Self::Normal {
            boundary,
            relation,
            coefficient_field,
            definition_relation,
            expression,
        }
    }

    pub(super) fn complete_affine_potential(
        boundary: RawId,
        relation: RawId,
        potential_field: RawId,
        definition_relation: RawId,
        speed_parameter: RawId,
        expression: ScalarSpatialExpression<f64>,
    ) -> Result<Self, Diagnostic> {
        let gradient = exact_complete_gradient(&expression)?;
        let speed = expression.parameter_values()[0];
        if gradient != [speed, 0.0] || !speed.is_finite() || speed <= 0.0 {
            return Err(invalid(
                "complete Stokes trace requires the exact positive affine gradient `U e_x`",
            ));
        }
        Ok(Self::CompleteAffinePotential {
            boundary,
            relation,
            potential_field,
            definition_relation,
            speed_parameter,
            expression,
        })
    }

    /// Exact Boundary identity this trace was lowered from.
    pub(super) const fn boundary(&self) -> RawId {
        match self {
            Self::Normal { boundary, .. } | Self::CompleteAffinePotential { boundary, .. } => {
                *boundary
            }
        }
    }

    /// Exact boundary Relation identity carrying this trace law.
    pub(super) const fn relation(&self) -> RawId {
        match self {
            Self::Normal { relation, .. } | Self::CompleteAffinePotential { relation, .. } => {
                *relation
            }
        }
    }

    pub(super) const fn coefficient_field(&self) -> RawId {
        match self {
            Self::Normal {
                coefficient_field, ..
            } => *coefficient_field,
            Self::CompleteAffinePotential {
                potential_field, ..
            } => *potential_field,
        }
    }

    pub(super) const fn definition_relation(&self) -> RawId {
        match self {
            Self::Normal {
                definition_relation,
                ..
            }
            | Self::CompleteAffinePotential {
                definition_relation,
                ..
            } => *definition_relation,
        }
    }

    pub(super) const fn expression(&self) -> &ScalarSpatialExpression<f64> {
        match self {
            Self::Normal { expression, .. } | Self::CompleteAffinePotential { expression, .. } => {
                expression
            }
        }
    }

    pub(super) const fn speed_parameter(&self) -> Option<RawId> {
        match self {
            Self::Normal { .. } => None,
            Self::CompleteAffinePotential {
                speed_parameter, ..
            } => Some(*speed_parameter),
        }
    }

    pub(super) const fn is_complete_affine_potential(&self) -> bool {
        matches!(self, Self::CompleteAffinePotential { .. })
    }

    /// The volume law identity every outer side must share, without its
    /// side-local Boundary and Relation identity.
    pub(super) fn law_identity(
        &self,
    ) -> (RawId, RawId, Option<RawId>, &ScalarSpatialExpression<f64>) {
        (
            self.coefficient_field(),
            self.definition_relation(),
            self.speed_parameter(),
            self.expression(),
        )
    }

    /// Replay the complete Model-owned vector at one exact coordinate.
    pub(super) fn evaluate(&self, coordinates: [f64; 2]) -> Result<[f64; 2], Diagnostic> {
        self.value(None, &coordinates)
    }

    /// Replay the complete vector from retained Model meaning.
    pub(super) fn value(
        &self,
        outward_normal: Option<[f64; 2]>,
        coordinates: &[f64],
    ) -> Result<[f64; 2], Diagnostic> {
        match self {
            Self::Normal { expression, .. } => {
                let outward = outward_normal.ok_or_else(|| {
                    invalid("normal Stokes trace requires one parent-outward normal")
                })?;
                let speed = expression.evaluate(coordinates)?;
                Ok(outward.map(|component| component * speed))
            }
            Self::CompleteAffinePotential { expression, .. } => exact_complete_gradient(expression),
        }
    }
}

/// Lower one exact boundary Relation into its retained Stokes trace meaning.
///
/// `Ok(None)` is the exact body no-slip `trace(velocity) = 0` law, which owns
/// no prescribed value. Every other admitted boundary law returns the single
/// retained trace owner above; nothing else is accepted.
pub(super) fn lower_prescribed_velocity_trace_2d(
    program: &KernelProgram,
    domain: RawId,
    velocity: RawId,
    boundary: RawId,
    relation: RawId,
) -> Result<Option<SteadyStokesPrescribedVelocityTrace2d>, Diagnostic> {
    let expression = &relation_expression(program, relation)?;
    let root = unique_root(expression, relation)?;
    if matches!(expression.node(root), Some(ExprNode::Trace(value)) if is_field(expression, *value, velocity))
    {
        return Ok(None);
    }
    super::boundary::prescribed_velocity_trace_2d(program, relation, velocity, domain, boundary)
        .map(Some)
}

/// One complete Model-owned essential velocity replay over owned vertices.
///
/// The replay is derived only from retained Model meaning. A transport
/// callback is compared against it and never supplies an independent value.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct ModelOwnedEssentialVelocityReplay2d {
    vertices: Vec<usize>,
    values: Vec<[f64; 2]>,
    roles: BTreeMap<String, Vec<usize>>,
}

impl ModelOwnedEssentialVelocityReplay2d {
    /// Replayed Model values, in ascending owned-vertex order.
    pub(super) fn values(&self) -> &[[f64; 2]] {
        &self.values
    }

    /// Replayed Model value of one exact vertex owned by one exact role.
    pub(super) fn value_for_role(&self, role: &str, vertex: usize) -> Option<[f64; 2]> {
        self.roles
            .get(role)
            .filter(|owned| owned.contains(&vertex))
            .and_then(|_| self.vertices.binary_search(&vertex).ok())
            .map(|position| self.values[position])
    }

    /// Admit a transport callback only when it reproduces every replayed value.
    pub(super) fn validate_transport(
        &self,
        transport: impl Fn(usize) -> [f64; 2],
    ) -> Result<(), Diagnostic> {
        for (&vertex, expected) in self.vertices.iter().zip(&self.values) {
            let supplied = transport(vertex);
            if supplied != *expected {
                return Err(invalid(format!(
                    "essential transport at vertex {vertex} differs from the replayed Model value"
                )));
            }
        }
        Ok(())
    }
}

/// Admit the complete Model-owned essential velocity datum before assembly.
///
/// The Model gate runs first: exact body no-slip recognition, four complete
/// outer laws with one shared `chi`/definition/`U` identity, and exact
/// Boundary/Relation retention. Only then is every owned essential vertex
/// enumerated and replayed. No caller value enters the result.
pub(super) fn admit_model_owned_essential_velocity_2d(
    program: &KernelProgram,
    velocity: RawId,
    role_boundaries: &BTreeMap<String, RawId>,
    body_relation: RawId,
    outer_traces: &BTreeMap<String, SteadyStokesPrescribedVelocityTrace2d>,
    owners: &BTreeMap<String, Vec<(usize, [f64; 2])>>,
) -> Result<ModelOwnedEssentialVelocityReplay2d, Diagnostic> {
    let boundaries = require_exact_roles(role_boundaries)?;
    let body_boundary = boundaries[BODY_SOURCE_LABEL];
    require_body_no_slip(program, velocity, body_boundary, body_relation)?;
    require_complete_outer_laws(program, &boundaries, body_relation, outer_traces)?;
    replay_owned_vertices(&boundaries, outer_traces, owners)
}

fn require_exact_roles(
    role_boundaries: &BTreeMap<String, RawId>,
) -> Result<BTreeMap<&str, RawId>, Diagnostic> {
    let mut boundaries = BTreeMap::new();
    for label in std::iter::once(BODY_SOURCE_LABEL).chain(OUTER_SOURCE_LABELS) {
        let boundary = role_boundaries.get(label).ok_or_else(|| {
            invalid(format!(
                "essential admission omits the exact role `{label}`"
            ))
        })?;
        boundaries.insert(label, *boundary);
    }
    if role_boundaries.len() != boundaries.len()
        || boundaries.values().copied().collect::<BTreeSet<_>>().len() != boundaries.len()
    {
        return Err(invalid(
            "essential admission requires exactly five distinct exact Boundary identities",
        ));
    }
    Ok(boundaries)
}

fn require_body_no_slip(
    program: &KernelProgram,
    velocity: RawId,
    body_boundary: RawId,
    body_relation: RawId,
) -> Result<(), Diagnostic> {
    if relations_on(program, body_boundary) != vec![body_relation] {
        return Err(invalid(
            "body role does not own exactly the supplied exact no-slip Relation",
        ));
    }
    let expression = &relation_expression(program, body_relation)?;
    let root = unique_root(expression, body_relation)?;
    if !matches!(expression.node(root), Some(ExprNode::Trace(value)) if is_field(expression, *value, velocity))
    {
        return Err(invalid(
            "body role Relation is not the exact `trace(velocity) = 0` no-slip law",
        ));
    }
    Ok(())
}

fn require_complete_outer_laws(
    program: &KernelProgram,
    boundaries: &BTreeMap<&str, RawId>,
    body_relation: RawId,
    outer_traces: &BTreeMap<String, SteadyStokesPrescribedVelocityTrace2d>,
) -> Result<(), Diagnostic> {
    if outer_traces.len() != OUTER_SOURCE_LABELS.len() {
        return Err(invalid(
            "essential admission requires exactly four complete outer traces",
        ));
    }
    let mut relations = BTreeSet::from([body_relation]);
    let mut shared: Option<&SteadyStokesPrescribedVelocityTrace2d> = None;
    for label in OUTER_SOURCE_LABELS {
        let trace = outer_traces.get(label).ok_or_else(|| {
            invalid(format!(
                "essential admission omits the outer role `{label}`"
            ))
        })?;
        if !trace.is_complete_affine_potential() {
            return Err(invalid(format!(
                "outer role `{label}` is not the exact complete affine-potential trace"
            )));
        }
        if trace.boundary() != boundaries[label] {
            return Err(invalid(format!(
                "outer role `{label}` retains another exact Boundary identity"
            )));
        }
        if relations_on(program, trace.boundary()) != vec![trace.relation()]
            || !relations.insert(trace.relation())
        {
            return Err(invalid(format!(
                "outer role `{label}` does not own exactly one distinct exact Relation"
            )));
        }
        match shared {
            Some(accepted) if accepted.law_identity() != trace.law_identity() => {
                return Err(invalid(format!(
                    "outer role `{label}` drifted from the one exact chi/definition/U law"
                )));
            }
            Some(_) => {}
            None => shared = Some(trace),
        }
    }
    let shared = shared.expect("four exact outer roles retain one law");
    let speed = shared
        .speed_parameter()
        .expect("complete trace owns one speed Parameter");
    let Some(KernelNode::Parameter(definition)) = program.node(speed) else {
        return Err(invalid("complete trace speed identity is not a Parameter"));
    };
    let initial = definition
        .real_scalar_value()
        .ok_or_else(|| invalid("trace speed requires a real scalar Parameter"))?;
    let value = program.value(speed).unwrap_or(initial);
    if value.dim() != SPEED_DIMENSION || !value.value().is_finite() || value.value() <= 0.0 {
        return Err(invalid(
            "complete trace speed must be one finite positive velocity Parameter",
        ));
    }
    Ok(())
}

fn replay_owned_vertices(
    boundaries: &BTreeMap<&str, RawId>,
    outer_traces: &BTreeMap<String, SteadyStokesPrescribedVelocityTrace2d>,
    owners: &BTreeMap<String, Vec<(usize, [f64; 2])>>,
) -> Result<ModelOwnedEssentialVelocityReplay2d, Diagnostic> {
    if owners.len() != boundaries.len() || !boundaries.keys().all(|role| owners.contains_key(*role))
    {
        return Err(invalid(
            "essential ownership does not cover exactly the five exact roles",
        ));
    }
    let mut replayed = BTreeMap::<usize, ([f64; 2], [f64; 2])>::new();
    let mut body_vertices = BTreeSet::new();
    let mut roles = BTreeMap::<String, Vec<usize>>::new();
    for (role, owned) in owners {
        let mut role_vertices = BTreeSet::new();
        for &(vertex, coordinates) in owned {
            if !role_vertices.insert(vertex) {
                return Err(invalid(format!(
                    "role `{role}` owns essential vertex {vertex} more than once"
                )));
            }
            let value = if role == BODY_SOURCE_LABEL {
                body_vertices.insert(vertex);
                [0.0, 0.0]
            } else {
                outer_traces[role.as_str()].evaluate(coordinates)?
            };
            match replayed.get(&vertex) {
                Some((accepted, accepted_coordinates))
                    if *accepted != value || *accepted_coordinates != coordinates =>
                {
                    return Err(invalid(format!(
                        "essential vertex {vertex} is replayed with two different Model data"
                    )));
                }
                Some(_) => {}
                None => {
                    replayed.insert(vertex, (value, coordinates));
                }
            }
        }
        roles.insert(role.clone(), role_vertices.into_iter().collect());
    }
    for role in OUTER_SOURCE_LABELS {
        if roles[role]
            .iter()
            .any(|vertex| body_vertices.contains(vertex))
        {
            return Err(invalid(format!(
                "role `{role}` shares an essential vertex with the exact body role"
            )));
        }
    }
    let (vertices, values) = replayed
        .into_iter()
        .map(|(vertex, (value, _))| (vertex, value))
        .collect::<(Vec<_>, Vec<_>)>();
    Ok(ModelOwnedEssentialVelocityReplay2d {
        vertices,
        values,
        roles,
    })
}

fn exact_complete_gradient(
    expression: &ScalarSpatialExpression<f64>,
) -> Result<[f64; 2], Diagnostic> {
    if expression.coordinate_dimension() != 2
        || expression.parameter_fields().len() != 1
        || expression.parameter_values().len() != 1
        || expression.evaluate(&[0.0, 0.0])? != 0.0
    {
        return Err(invalid(
            "complete Stokes trace requires one zero-intercept two-dimensional affine potential",
        ));
    }
    expression
        .affine_gradient()
        .and_then(|gradient| gradient.try_into().ok())
        .ok_or_else(|| invalid("complete Stokes trace potential is not exactly affine in 2D"))
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(
        eqiora_core::diagnostic::codes::INVALID_DISCRETIZATION,
        message,
    )
}
