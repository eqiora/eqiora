//! Method-neutral lowering of conservative scalar transport meaning.

use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id, RawId};
use eqiora_graph::EdgeKind;
use eqiora_schema::kernel::{
    BoundarySide, ConnectionSemantics, ExprDag, ExprId, ExprNode, KernelNode, SymbolRef,
};
use eqiora_sem::KernelProgram;

use crate::canonical::{
    boundary_parent, continuum_fields_on, is_field, lower_flux_coefficient, lowering_error,
    relation_expression, relations_on, unique_cartesian_box, unique_relation_on, unique_root,
};
use crate::spatial_expression::{self, ScalarSpatialExpression};

/// Exact physical boundary law of one conservative scalar transport model.
///
/// The enum records canonical trace or diffusive-flux meaning. Inflow,
/// outflow, and impermeable-wall roles are derived later from the parent-
/// outward advecting velocity at each face; they are not source labels.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarTransportCartesianBoundary {
    /// Prescribed transported-state trace.
    PrescribedTrace(ScalarSpatialExpression<f64>),
    /// Prescribed parent-outward diffusive flux `normal(k grad(c))`.
    PrescribedDiffusiveFlux(ScalarSpatialExpression<f64>),
    /// Exact live identification with the opposite side of one Cartesian axis.
    SpatialPeriodic {
        /// Spatial-periodic Connection owning the identification.
        connection: RawId,
        /// Exact field-valued Port owned by this side's boundary Relation.
        port: RawId,
    },
}

impl ScalarTransportCartesianBoundary {
    /// Canonical prescribed data in full physical coordinates, when present.
    #[must_use]
    pub const fn value(&self) -> Option<&ScalarSpatialExpression<f64>> {
        match self {
            Self::PrescribedTrace(value) | Self::PrescribedDiffusiveFlux(value) => Some(value),
            Self::SpatialPeriodic { .. } => None,
        }
    }

    /// Exact spatial-periodic Connection and this side's Port, when present.
    #[must_use]
    pub const fn spatial_periodic_binding(&self) -> Option<(RawId, RawId)> {
        match self {
            Self::SpatialPeriodic { connection, port } => Some((*connection, *port)),
            _ => None,
        }
    }
}

/// Method-neutral conservative scalar transport model on a two-dimensional
/// Cartesian box.
///
/// This projection retains exact Semantic identities and immutable spatial
/// expression tapes. Mesh density, reconstruction, time step, linear solver,
/// placement, and execution schedule remain Realization choices.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarTransportCartesianModel2d {
    domain: RawId,
    state: RawId,
    potential: RawId,
    transport_relation: RawId,
    potential_definition: RawId,
    bounds: [[f64; 2]; 2],
    diffusivity: ScalarSpatialExpression<f64>,
    potential_expression: ScalarSpatialExpression<f64>,
    boundaries: BTreeMap<(usize, BoundarySide), ScalarTransportCartesianBoundary>,
    spatial_periodic_connections: std::collections::BTreeSet<RawId>,
}

impl ScalarTransportCartesianModel2d {
    /// Exact parent Domain.
    #[must_use]
    pub fn domain(&self) -> Id<kinds::Domain> {
        self.domain
            .downcast()
            .expect("transport lowering stores a Domain identifier")
    }

    /// Exact transported scalar Field.
    #[must_use]
    pub fn state(&self) -> Id<kinds::Field> {
        self.state
            .downcast()
            .expect("transport lowering stores a Field identifier")
    }

    /// Exact scalar potential whose physical gradient is the advector.
    #[must_use]
    pub fn potential(&self) -> Id<kinds::Field> {
        self.potential
            .downcast()
            .expect("transport lowering stores a Field identifier")
    }

    /// Exact conservative transient Relation.
    #[must_use]
    pub fn transport_relation(&self) -> Id<kinds::Relation> {
        self.transport_relation
            .downcast()
            .expect("transport lowering stores a Relation identifier")
    }

    /// Exact Relation defining the advecting potential.
    #[must_use]
    pub fn potential_definition(&self) -> Id<kinds::Relation> {
        self.potential_definition
            .downcast()
            .expect("transport lowering stores a Relation identifier")
    }

    /// Physical Cartesian bounds in coherent SI coordinates.
    #[must_use]
    pub const fn bounds(&self) -> &[[f64; 2]; 2] {
        &self.bounds
    }

    /// Positive constant scalar diffusivity in coherent SI units.
    #[must_use]
    pub fn diffusivity(&self) -> f64 {
        self.diffusivity
            .constant_value()
            .expect("transport lowerer admits a constant diffusivity")
    }

    /// Exact lowered diffusivity expression.
    #[must_use]
    pub const fn diffusivity_expression(&self) -> &ScalarSpatialExpression<f64> {
        &self.diffusivity
    }

    /// Exact lowered advecting-potential expression.
    #[must_use]
    pub const fn potential_expression(&self) -> &ScalarSpatialExpression<f64> {
        &self.potential_expression
    }

    /// Evaluate the canonical advecting velocity `grad(potential)`.
    ///
    /// # Errors
    /// Preserves immutable expression-tape shape and finite-value diagnostics.
    pub fn advecting_velocity(&self, coordinates: &[f64; 2]) -> Result<[f64; 2], Diagnostic> {
        let (_, gradient, _) = self.potential_expression.evaluate_vjp(coordinates, 1.0)?;
        Ok([gradient[0], gradient[1]])
    }

    /// Exact boundary law on one Cartesian axis side.
    #[must_use]
    pub fn boundary(
        &self,
        axis: usize,
        side: BoundarySide,
    ) -> Option<&ScalarTransportCartesianBoundary> {
        self.boundaries.get(&(axis, side))
    }

    /// Exact spatial-periodic Connections required by this Model.
    pub fn spatial_periodic_connections(
        &self,
    ) -> impl ExactSizeIterator<Item = Id<kinds::Connection>> + '_ {
        self.spatial_periodic_connections.iter().map(|connection| {
            connection
                .downcast()
                .expect("transport lowering stores Connection identifiers")
        })
    }
}

/// Lower the accepted conservative scalar transport subset.
///
/// The admitted volume meaning is exactly
/// `derivative(c) + div(c * grad(psi)) - div(k * grad(c)) = 0`, where `k`
/// is finite, positive, and spatially constant. A second Relation defines
/// `psi` as a closed scalar spatial expression. Every exterior side must
/// carry exactly one prescribed trace or prescribed outward diffusive-flux
/// Relation.
///
/// # Errors
/// Returns `EQ0703` for ambiguity, incomplete closure, or any expression
/// outside this deliberately narrow canonical subset.
pub fn lower_scalar_transport_cartesian_2d(
    program: &KernelProgram,
) -> Result<ScalarTransportCartesianModel2d, Diagnostic> {
    let (domain, bounds) = unique_cartesian_box(program)?;
    let bounds: [[f64; 2]; 2] = bounds.try_into().map_err(|bounds: Vec<_>| {
        lowering_error(
            domain,
            format!(
                "scalar transport Cartesian lowering requires dimension two, received {}",
                bounds.len()
            ),
        )
    })?;
    let fields = continuum_fields_on(program, domain);
    if fields.len() != 2 {
        return Err(lowering_error(
            domain,
            format!(
                "scalar transport lowering requires exactly one state and one advecting-potential Field, found {} continuum Fields",
                fields.len()
            ),
        ));
    }
    let volume_relations = relations_on(program, domain).into_iter().filter(|id| {
        matches!(program.node(*id), Some(KernelNode::Relation(relation)) if !relation.is_initial())
    }).collect::<Vec<_>>();
    if volume_relations.len() != 2 {
        return Err(lowering_error(
            domain,
            format!(
                "scalar transport lowering requires exactly one transport and one potential-definition Relation, found {}",
                volume_relations.len()
            ),
        ));
    }

    let transport_relations = volume_relations
        .iter()
        .copied()
        .filter(|relation| relation_derivative_field(program, *relation).is_some())
        .collect::<Vec<_>>();
    if transport_relations.len() != 1 {
        return Err(lowering_error(
            domain,
            format!(
                "scalar transport lowering requires exactly one volume Relation with one state derivative, found {}",
                transport_relations.len()
            ),
        ));
    }
    let transport_relation = transport_relations[0];
    let potential_definition = volume_relations
        .iter()
        .copied()
        .find(|relation| *relation != transport_relation)
        .expect("two distinct volume Relations were validated");

    let state = relation_derivative_field(program, transport_relation)
        .expect("transport Relation was selected by its derivative");
    if !fields.contains(&state) {
        return Err(lowering_error(
            transport_relation,
            "transport derivative does not reference a continuum Field on the exact Domain",
        ));
    }
    let (potential, diffusivity) = lower_transport_relation(program, transport_relation, state)?;
    if potential == state || !fields.contains(&potential) {
        return Err(lowering_error(
            transport_relation,
            "advector potential must be the other continuum Field on the exact Domain",
        ));
    }
    let diffusivity_value = diffusivity
        .constant_value()
        .expect("transport flux lowering admits a spatially constant coefficient");
    if !diffusivity_value.is_finite() || diffusivity_value <= 0.0 {
        return Err(lowering_error(
            transport_relation,
            "transport diffusivity must be finite and positive",
        ));
    }
    let potential_expression =
        lower_potential_definition(program, potential_definition, potential, transport_relation)?;
    if potential_expression.affine_gradient().is_none() {
        return Err(lowering_error(
            potential_definition,
            "closed Cartesian transport requires an affine advecting potential with one constant velocity over the Domain",
        ));
    }

    let mut boundaries = BTreeMap::new();
    for node in program.nodes() {
        let eqiora_schema::kernel::KernelNode::Domain(boundary_domain) = node else {
            continue;
        };
        let eqiora_schema::kernel::DomainKind::CartesianBoundary { axis, side } =
            boundary_domain.kind()
        else {
            continue;
        };
        if boundary_parent(program, boundary_domain.id().erase()) != Some(domain) {
            continue;
        }
        if *axis >= 2 {
            return Err(lowering_error(
                boundary_domain.id().erase(),
                "2D scalar transport boundary lies outside its ambient dimension",
            ));
        }
        let relation = unique_relation_on(program, boundary_domain.id().erase())?;
        let law = lower_boundary_relation(
            program,
            boundary_domain.id().erase(),
            relation,
            state,
            potential,
            &diffusivity,
        )?;
        if boundaries.insert((*axis, *side), law).is_some() {
            return Err(lowering_error(
                boundary_domain.id().erase(),
                "scalar transport Cartesian boundary side is duplicated",
            ));
        }
    }
    for axis in 0..2 {
        for side in [BoundarySide::Lower, BoundarySide::Upper] {
            if !boundaries.contains_key(&(axis, side)) {
                return Err(lowering_error(
                    domain,
                    format!("missing scalar transport boundary on axis {axis} {side:?}"),
                ));
            }
        }
    }
    let spatial_periodic_connections = validate_periodic_boundary_pairs(domain, &boundaries)?;

    Ok(ScalarTransportCartesianModel2d {
        domain,
        state,
        potential,
        transport_relation,
        potential_definition,
        bounds,
        diffusivity,
        potential_expression,
        boundaries,
        spatial_periodic_connections,
    })
}

fn relation_derivative_field(program: &KernelProgram, relation: RawId) -> Option<RawId> {
    let expression = &relation_expression(program, relation).ok()?;
    let derivatives = expression
        .nodes()
        .iter()
        .filter_map(|node| match node {
            ExprNode::Symbol(SymbolRef::Derivative(field, std::num::NonZeroU32::MIN)) => {
                Some(field.erase())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    (derivatives.len() == 1).then(|| derivatives[0])
}

fn lower_transport_relation(
    program: &KernelProgram,
    relation: RawId,
    state: RawId,
) -> Result<(RawId, ScalarSpatialExpression<f64>), Diagnostic> {
    let expression = &relation_expression(program, relation)?;
    let root = unique_root(expression, relation)?;
    let Some(ExprNode::Sub(transient_advection, diffusion_divergence)) = expression.node(root)
    else {
        return Err(lowering_error(
            relation,
            "transport residual must be `(derivative(state) + div(state * grad(potential))) - div(diffusivity * grad(state))`",
        ));
    };
    let (derivative, advection_divergence) = match expression.node(*transient_advection) {
        Some(ExprNode::Add(left, right)) if is_derivative(expression, *left, state) => {
            (*left, *right)
        }
        Some(ExprNode::Add(left, right)) if is_derivative(expression, *right, state) => {
            (*right, *left)
        }
        _ => {
            return Err(lowering_error(
                relation,
                "transport residual must add exactly the state derivative and conservative advective divergence",
            ));
        }
    };
    debug_assert!(is_derivative(expression, derivative, state));
    let potential = lower_advective_divergence(expression, advection_divergence, state, relation)?;
    let diffusion_flux = match expression.node(*diffusion_divergence) {
        Some(ExprNode::Divergence(flux)) => *flux,
        _ => {
            return Err(lowering_error(
                relation,
                "transport diffusion term must be physical divergence of diffusivity times state gradient",
            ));
        }
    };
    let diffusivity =
        lower_flux_coefficient(program, expression, diffusion_flux, state, relation, 2)?;
    Ok((potential, diffusivity))
}

fn lower_advective_divergence(
    expression: &ExprDag,
    value: ExprId,
    state: RawId,
    owner: RawId,
) -> Result<RawId, Diagnostic> {
    let flux = match expression.node(value) {
        Some(ExprNode::Divergence(flux)) => *flux,
        _ => {
            return Err(lowering_error(
                owner,
                "transport advection must be written in conservative divergence form",
            ));
        }
    };
    let Some(ExprNode::Mul(left, right)) = expression.node(flux) else {
        return Err(lowering_error(
            owner,
            "advective flux must multiply the transported state by a scalar-potential gradient",
        ));
    };
    if is_field(expression, *left, state) {
        gradient_field(expression, *right)
            .ok_or_else(|| lowering_error(owner, "advective flux requires `grad(potential)`"))
    } else if is_field(expression, *right, state) {
        gradient_field(expression, *left)
            .ok_or_else(|| lowering_error(owner, "advective flux requires `grad(potential)`"))
    } else {
        Err(lowering_error(
            owner,
            "advective flux must contain the exact transported state Field",
        ))
    }
}

fn gradient_field(expression: &ExprDag, value: ExprId) -> Option<RawId> {
    match expression.node(value) {
        Some(ExprNode::Gradient(argument)) => match expression.node(*argument) {
            Some(ExprNode::Symbol(SymbolRef::Field(field))) => Some(field.erase()),
            _ => None,
        },
        _ => None,
    }
}

fn is_derivative(expression: &ExprDag, value: ExprId, field: RawId) -> bool {
    matches!(
        expression.node(value),
        Some(ExprNode::Symbol(SymbolRef::Derivative(id, std::num::NonZeroU32::MIN))) if id.erase() == field
    )
}

fn lower_potential_definition(
    program: &KernelProgram,
    relation: RawId,
    potential: RawId,
    transport_relation: RawId,
) -> Result<ScalarSpatialExpression<f64>, Diagnostic> {
    let expression = &relation_expression(program, relation)?;
    if expression.nodes().iter().any(|node| {
        matches!(
            node,
            ExprNode::Symbol(SymbolRef::Derivative(_, _) | SymbolRef::Field(_))
        )
    }) && !expression.nodes().iter().any(
        |node| matches!(node, ExprNode::Symbol(SymbolRef::Field(id)) if id.erase() == potential),
    ) {
        return Err(lowering_error(
            relation,
            "potential definition must define the exact advecting-potential Field",
        ));
    }
    let root = unique_root(expression, relation)?;
    let rhs = match expression.node(root) {
        Some(ExprNode::Sub(left, right)) if is_field(expression, *left, potential) => *right,
        _ => {
            return Err(lowering_error(
                relation,
                "advecting potential must be defined as `potential - closed_spatial_expression = 0`",
            ));
        }
    };
    let lowered = spatial_expression::lower(program, expression, rhs, relation, 2)?;
    if lowered.coordinate_dimension() != 2 {
        return Err(lowering_error(
            transport_relation,
            "advecting-potential expression has the wrong coordinate dimension",
        ));
    }
    Ok(lowered)
}

fn lower_boundary_relation(
    program: &KernelProgram,
    boundary: RawId,
    relation: RawId,
    state: RawId,
    potential: RawId,
    diffusivity: &ScalarSpatialExpression<f64>,
) -> Result<ScalarTransportCartesianBoundary, Diagnostic> {
    let expression = &relation_expression(program, relation)?;
    if expression.roots().len() == 2 {
        return lower_spatial_periodic_boundary_relation(
            program,
            boundary,
            relation,
            expression,
            state,
            potential,
            diffusivity,
        );
    }
    let root = unique_root(expression, relation)?;
    let (operator, value) = match expression.node(root) {
        Some(ExprNode::Sub(operator, value)) => (*operator, Some(*value)),
        _ => (root, None),
    };
    let value = match value {
        Some(value) => spatial_expression::lower(program, expression, value, relation, 2)?,
        None => ScalarSpatialExpression::constant(2, 0.0),
    };
    if value.constant_value().is_none() {
        return Err(lowering_error(
            relation,
            "closed Cartesian transport boundary data must be spatially constant",
        ));
    }
    match expression.node(operator) {
        Some(ExprNode::Trace {
            value: argument, ..
        }) if is_field(expression, *argument, state) => {
            Ok(ScalarTransportCartesianBoundary::PrescribedTrace(value))
        }
        Some(ExprNode::NormalComponent { value: flux, .. }) => {
            let coefficient =
                lower_flux_coefficient(program, expression, *flux, state, relation, 2)?;
            if !coefficient.is_same_coefficient_as(diffusivity) {
                return Err(lowering_error(
                    relation,
                    "boundary diffusive-flux coefficient differs from the exact volume constitutive flux",
                ));
            }
            Ok(ScalarTransportCartesianBoundary::PrescribedDiffusiveFlux(
                value,
            ))
        }
        _ => Err(lowering_error(
            relation,
            "transport boundary residual must prescribe the state trace or outward diffusive flux",
        )),
    }
}

fn lower_spatial_periodic_boundary_relation(
    program: &KernelProgram,
    boundary: RawId,
    relation: RawId,
    expression: &ExprDag,
    state: RawId,
    potential: RawId,
    diffusivity: &ScalarSpatialExpression<f64>,
) -> Result<ScalarTransportCartesianBoundary, Diagnostic> {
    let mut trace_port = None;
    let mut flux_port = None;
    for root in expression.roots() {
        let Some(ExprNode::Sub(left, right)) = expression.node(*root) else {
            return Err(lowering_error(
                relation,
                "periodic transport boundary roots must be oriented equality residuals",
            ));
        };
        if matches!(expression.node(*left), Some(ExprNode::Trace { value, .. }) if is_field(expression, *value, state))
        {
            trace_port = port_trace_symbol(expression, *right);
            continue;
        }
        if let Some(ExprNode::NormalComponent {
            value: total_flux, ..
        }) = expression.node(*left)
        {
            flux_port = port_flux_symbol(expression, *right);
            validate_total_transport_flux(
                program,
                expression,
                *total_flux,
                state,
                potential,
                diffusivity,
                relation,
            )?;
            continue;
        }
        return Err(lowering_error(
            relation,
            "periodic transport boundary must bind the state trace and total conservative flux",
        ));
    }
    let Some(port) = trace_port.filter(|port| Some(*port) == flux_port) else {
        return Err(lowering_error(
            relation,
            "periodic transport boundary must bind trace and total flux to one exact Port",
        ));
    };
    let owned_ports = program
        .edges()
        .iter()
        .filter(|edge| edge.kind() == EdgeKind::HasPort && edge.from() == relation)
        .map(|edge| edge.to())
        .collect::<BTreeSet<_>>();
    if owned_ports != BTreeSet::from([port]) {
        return Err(lowering_error(
            relation,
            "periodic transport boundary Relation must own exactly the bound Port",
        ));
    }
    let Some(KernelNode::Port(port_definition)) = program.node(port) else {
        return Err(lowering_error(port, "periodic transport Port is missing"));
    };
    let Some((_, port_boundary)) = port_definition.boundary_physical_contract() else {
        return Err(lowering_error(
            port,
            "periodic transport requires a field-valued boundary Port",
        ));
    };
    if port_boundary.erase() != boundary {
        return Err(lowering_error(
            port,
            "periodic transport Port support differs from its boundary Relation",
        ));
    }
    let connections = program
        .edges()
        .iter()
        .filter(|edge| edge.kind() == EdgeKind::Connects && edge.to() == port)
        .map(|edge| edge.from())
        .collect::<Vec<_>>();
    if connections.len() != 1 {
        return Err(lowering_error(
            port,
            "periodic transport Port requires exactly one Connection",
        ));
    }
    let connection = connections[0];
    let Some(KernelNode::Connection(definition)) = program.node(connection) else {
        return Err(lowering_error(
            connection,
            "periodic transport Connection is missing",
        ));
    };
    if definition.semantics() != ConnectionSemantics::SpatialPeriodic {
        return Err(lowering_error(
            connection,
            "transport trace-plus-total-flux binding requires spatial-periodic Connection semantics",
        ));
    }
    let typed = connection.downcast::<kinds::Connection>().ok_or_else(|| {
        lowering_error(
            connection,
            "periodic transport Connection has the wrong identity kind",
        )
    })?;
    program
        .compose_boundary_physical_junction(typed)
        .map_err(|_| lowering_error(connection, "periodic transport junction is not closed"))?;
    Ok(ScalarTransportCartesianBoundary::SpatialPeriodic { connection, port })
}

fn validate_total_transport_flux(
    program: &KernelProgram,
    expression: &ExprDag,
    total_flux: ExprId,
    state: RawId,
    potential: RawId,
    diffusivity: &ScalarSpatialExpression<f64>,
    relation: RawId,
) -> Result<(), Diagnostic> {
    let Some(ExprNode::Sub(advective, diffusive)) = expression.node(total_flux) else {
        return Err(lowering_error(
            relation,
            "periodic transport flux must be `state * grad(potential) - diffusivity * grad(state)`",
        ));
    };
    let advective_matches = match expression.node(*advective) {
        Some(ExprNode::Mul(left, right)) => {
            (is_field(expression, *left, state) && is_gradient_of(expression, *right, potential))
                || (is_field(expression, *right, state)
                    && is_gradient_of(expression, *left, potential))
        }
        _ => false,
    };
    if !advective_matches {
        return Err(lowering_error(
            relation,
            "periodic transport flux must use the exact state and advecting potential",
        ));
    }
    let coefficient = lower_flux_coefficient(program, expression, *diffusive, state, relation, 2)?;
    if !coefficient.is_same_coefficient_as(diffusivity) {
        return Err(lowering_error(
            relation,
            "periodic transport flux coefficient differs from the volume diffusivity",
        ));
    }
    Ok(())
}

fn is_gradient_of(expression: &ExprDag, value: ExprId, field: RawId) -> bool {
    matches!(
        expression.node(value),
        Some(ExprNode::Gradient(argument)) if is_field(expression, *argument, field)
    )
}

fn port_trace_symbol(expression: &ExprDag, value: ExprId) -> Option<RawId> {
    match expression.node(value) {
        Some(ExprNode::Symbol(SymbolRef::PortTrace(port))) => Some(port.erase()),
        _ => None,
    }
}

fn port_flux_symbol(expression: &ExprDag, value: ExprId) -> Option<RawId> {
    match expression.node(value) {
        Some(ExprNode::Symbol(SymbolRef::PortFlux(port))) => Some(port.erase()),
        _ => None,
    }
}

fn validate_periodic_boundary_pairs(
    domain: RawId,
    boundaries: &BTreeMap<(usize, BoundarySide), ScalarTransportCartesianBoundary>,
) -> Result<std::collections::BTreeSet<RawId>, Diagnostic> {
    let mut connections = BTreeMap::<RawId, Vec<((usize, BoundarySide), RawId)>>::new();
    for (side, boundary) in boundaries {
        if let Some((connection, port)) = boundary.spatial_periodic_binding() {
            connections
                .entry(connection)
                .or_default()
                .push((*side, port));
        }
    }
    for members in connections.values() {
        if members.len() != 2 || members[0].1 == members[1].1 {
            return Err(lowering_error(
                domain,
                "each spatial-periodic transport Connection must close exactly two distinct boundary Ports",
            ));
        }
        let ((left_axis, left_side), _) = members[0];
        let ((right_axis, right_side), _) = members[1];
        if left_axis != right_axis || left_side == right_side {
            return Err(lowering_error(
                domain,
                "spatial-periodic transport sides must be opposite boundaries of one axis",
            ));
        }
    }
    Ok(connections.into_keys().collect())
}
