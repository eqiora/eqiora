//! Private semantic recognition for the bounded one-dimensional ideal-gas Euler system.
//!
//! This module owns no mesh, numerical flux, time policy, or benchmark data. It
//! projects exact Model meaning into the private physics contract that a later
//! conservative-system realization can consume.

use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::ValueFrame;
use eqiora_core::{Diagnostic, DimExponents, RawId, ValueShape};
use eqiora_graph::EdgeKind;
use eqiora_ir::{
    CalculusBuilder, CalculusNode, OperatorDefinitionDigest, PureOperatorApplicationProof,
    PureOperatorDefinition, PureOperatorError, PureValueClass,
};
use eqiora_schema::kernel::typing::TypedResidual;
use eqiora_schema::kernel::{
    ActivationKind, BoundarySide, DomainKind, ExprDag, ExprId, ExprNode, KernelNode,
    RepresentationKind, SymbolRef,
};
use eqiora_sem::KernelProgram;

use crate::additive_residual::AdditiveResidualView;
use crate::canonical::{
    boundary_parent, lowering_error, model_lowering_error, relations_on, unique_cartesian_box,
    unique_root,
};

const DENSITY: DimExponents =
    DimExponents::from_integers([1, -3, 0, 0, 0, 0, 0]).expect("bounded dimension");
const MOMENTUM_DENSITY: DimExponents =
    DimExponents::from_integers([1, -2, -1, 0, 0, 0, 0]).expect("bounded dimension");
const ENERGY_DENSITY: DimExponents =
    DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension");
const VELOCITY: DimExponents =
    DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension");

/// Exact conservative state order admitted by the private Euler descriptor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct EulerConservativeState1d {
    density: f64,
    momentum: f64,
    total_energy: f64,
}

impl EulerConservativeState1d {
    pub(crate) const fn new(density: f64, momentum: f64, total_energy: f64) -> Self {
        Self {
            density,
            momentum,
            total_energy,
        }
    }

    pub(crate) const fn components(self) -> [f64; 3] {
        [self.density, self.momentum, self.total_energy]
    }
}

/// Primitive coordinates paired with one admitted conservative state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct EulerPrimitiveState1d {
    density: f64,
    velocity: f64,
    pressure: f64,
}

impl EulerPrimitiveState1d {
    pub(crate) const fn new(density: f64, velocity: f64, pressure: f64) -> Self {
        Self {
            density,
            velocity,
            pressure,
        }
    }

    pub(crate) const fn components(self) -> [f64; 3] {
        [self.density, self.velocity, self.pressure]
    }
}

/// Package-neutral, method-neutral meaning of one closed 1D ideal-gas Euler Model.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct IdealGasEulerModel1d {
    domain: RawId,
    representation: RawId,
    bounds: [f64; 2],
    boundaries: [RawId; 2],
    conservative_fields: [RawId; 3],
    velocity: RawId,
    pressure: RawId,
    gamma_parameter: RawId,
    gamma: f64,
    balance_relations: [RawId; 3],
    velocity_relation: RawId,
    pressure_relation: RawId,
    flux_operator: OperatorDefinitionDigest,
}

impl IdealGasEulerModel1d {
    pub(crate) const fn domain(&self) -> RawId {
        self.domain
    }

    pub(crate) const fn representation(&self) -> RawId {
        self.representation
    }

    pub(crate) const fn bounds(&self) -> [f64; 2] {
        self.bounds
    }

    pub(crate) const fn boundaries(&self) -> [RawId; 2] {
        self.boundaries
    }

    pub(crate) const fn conservative_fields(&self) -> [RawId; 3] {
        self.conservative_fields
    }

    pub(crate) const fn primitive_fields(&self) -> [RawId; 2] {
        [self.velocity, self.pressure]
    }

    pub(crate) const fn gamma_parameter(&self) -> RawId {
        self.gamma_parameter
    }

    pub(crate) const fn gamma(&self) -> f64 {
        self.gamma
    }

    pub(crate) const fn balance_relations(&self) -> [RawId; 3] {
        self.balance_relations
    }

    pub(crate) const fn closure_relations(&self) -> [RawId; 2] {
        [self.velocity_relation, self.pressure_relation]
    }

    pub(crate) const fn flux_operator(&self) -> OperatorDefinitionDigest {
        self.flux_operator
    }

    pub(crate) fn conservative_to_primitive(
        &self,
        state: EulerConservativeState1d,
    ) -> Result<EulerPrimitiveState1d, Diagnostic> {
        let [density, momentum, total_energy] = state.components();
        if !density.is_finite() || !momentum.is_finite() || !total_energy.is_finite() {
            return Err(self.invalid_state("Euler conservative state must be finite"));
        }
        if density <= 0.0 {
            return Err(self.invalid_state("Euler density must be strictly positive"));
        }
        let velocity = momentum / density;
        let internal_energy = total_energy - 0.5 * momentum * velocity;
        let pressure = (self.gamma - 1.0) * internal_energy;
        if !velocity.is_finite()
            || !internal_energy.is_finite()
            || internal_energy <= 0.0
            || !pressure.is_finite()
            || pressure <= 0.0
        {
            return Err(self.invalid_state(
                "Euler state requires finite positive internal energy and pressure",
            ));
        }
        Ok(EulerPrimitiveState1d::new(density, velocity, pressure))
    }

    pub(crate) fn primitive_to_conservative(
        &self,
        state: EulerPrimitiveState1d,
    ) -> Result<EulerConservativeState1d, Diagnostic> {
        let [density, velocity, pressure] = state.components();
        if !density.is_finite() || density <= 0.0 {
            return Err(self.invalid_state("Euler density must be finite and strictly positive"));
        }
        if !velocity.is_finite() || !pressure.is_finite() || pressure <= 0.0 {
            return Err(self.invalid_state(
                "Euler velocity must be finite and pressure must be finite and positive",
            ));
        }
        let momentum = density * velocity;
        let total_energy = pressure / (self.gamma - 1.0) + 0.5 * density * velocity * velocity;
        let state = EulerConservativeState1d::new(density, momentum, total_energy);
        self.conservative_to_primitive(state)?;
        Ok(state)
    }

    pub(crate) fn physical_flux(
        &self,
        state: EulerConservativeState1d,
    ) -> Result<[f64; 3], Diagnostic> {
        let primitive = self.conservative_to_primitive(state)?;
        let [density, velocity, pressure] = primitive.components();
        let total_energy = state.total_energy;
        let flux = [
            density * velocity,
            density * velocity * velocity + pressure,
            velocity * (total_energy + pressure),
        ];
        if flux.iter().all(|value| value.is_finite()) {
            Ok(flux)
        } else {
            Err(self.invalid_state("Euler physical flux must be finite"))
        }
    }

    pub(crate) fn sound_speed(&self, state: EulerConservativeState1d) -> Result<f64, Diagnostic> {
        let primitive = self.conservative_to_primitive(state)?;
        let sound_speed = (self.gamma * primitive.pressure / primitive.density).sqrt();
        if !sound_speed.is_finite() {
            return Err(self.invalid_state("Euler sound speed must be finite"));
        }
        Ok(sound_speed)
    }

    pub(crate) fn characteristic_speed_bound(
        &self,
        state: EulerConservativeState1d,
    ) -> Result<f64, Diagnostic> {
        let primitive = self.conservative_to_primitive(state)?;
        let bound = primitive.velocity.abs() + self.sound_speed(state)?;
        if bound.is_finite() {
            Ok(bound)
        } else {
            Err(self.invalid_state("Euler characteristic-speed bound must be finite"))
        }
    }

    pub(crate) fn is_admissible(&self, state: EulerConservativeState1d) -> bool {
        self.conservative_to_primitive(state).is_ok()
    }

    fn invalid_state(&self, message: &str) -> Diagnostic {
        lowering_error(self.domain, message)
    }
}

/// Recognize one complete bounded 1D ideal-gas Euler Model.
pub(crate) fn recognize_ideal_gas_euler_1d(
    program: &KernelProgram,
) -> Result<IdealGasEulerModel1d, Diagnostic> {
    let (domain, bounds) = unique_cartesian_box(program)?;
    let [bounds] = bounds.as_slice() else {
        return Err(lowering_error(
            domain,
            format!(
                "ideal-gas Euler recognition requires one spatial dimension, found {}",
                bounds.len()
            ),
        ));
    };
    let boundaries = exact_interval_boundaries(program, domain)?;
    let fields = exact_euler_fields(program, domain)?;
    let representation = shared_continuum_representation(program, &fields.all, domain)?;
    let relations = relations_on(program, domain);
    if relations.len() != 5 {
        return Err(lowering_error(
            domain,
            format!(
                "ideal-gas Euler requires exactly three balances and two closure Relations, found {}",
                relations.len()
            ),
        ));
    }
    for relation in &relations {
        require_continuous_relation(program, *relation)?;
    }
    for boundary in boundaries {
        if !relations_on(program, boundary).is_empty() {
            return Err(lowering_error(
                boundary,
                "ideal-gas Euler semantic recognition does not yet admit boundary physics",
            ));
        }
    }
    let typed = relations
        .iter()
        .map(|relation| Ok((*relation, typed_relation(program, *relation)?)))
        .collect::<Result<BTreeMap<_, _>, Diagnostic>>()?;
    let parameters = dimensionless_parameters(program);
    let mut candidates = Vec::new();
    for pressure in fields.energy_scalars.iter().copied() {
        for total_energy in fields.energy_scalars.iter().copied() {
            if pressure == total_energy {
                continue;
            }
            for (gamma_parameter, gamma) in parameters.iter().copied() {
                let velocity_relations = relations
                    .iter()
                    .copied()
                    .filter(|relation| {
                        matches_velocity_closure(
                            &typed[relation],
                            *relation,
                            fields.momentum,
                            fields.density,
                            fields.velocity,
                        )
                        .unwrap_or(false)
                    })
                    .collect::<Vec<_>>();
                let pressure_relations = relations
                    .iter()
                    .copied()
                    .filter(|relation| {
                        matches_pressure_closure(
                            &typed[relation],
                            *relation,
                            pressure,
                            total_energy,
                            fields.momentum,
                            fields.velocity,
                            gamma_parameter,
                        )
                        .unwrap_or(false)
                    })
                    .collect::<Vec<_>>();
                if velocity_relations.len() != 1 || pressure_relations.len() != 1 {
                    continue;
                }
                let balance_relations =
                    [fields.density, fields.momentum, total_energy].map(|state| {
                        relations
                            .iter()
                            .copied()
                            .filter(|relation| {
                                matches_balance(
                                    &typed[relation],
                                    *relation,
                                    state,
                                    fields.momentum,
                                    fields.velocity,
                                    pressure,
                                    total_energy,
                                )
                                .unwrap_or(false)
                            })
                            .collect::<Vec<_>>()
                    });
                if balance_relations.iter().all(|matches| matches.len() == 1) {
                    candidates.push((
                        pressure,
                        total_energy,
                        gamma_parameter,
                        gamma,
                        velocity_relations[0],
                        pressure_relations[0],
                        balance_relations.map(|matches| matches[0]),
                    ));
                }
            }
        }
    }
    let [candidate] = candidates.as_slice() else {
        return Err(lowering_error(
            domain,
            format!(
                "ideal-gas Euler field, closure, and conservative-balance roles require one unique assignment, found {}",
                candidates.len()
            ),
        ));
    };
    let (
        pressure,
        total_energy,
        gamma_parameter,
        gamma,
        velocity_relation,
        pressure_relation,
        balance_relations,
    ) = *candidate;
    if !gamma.is_finite() || gamma <= 1.0 {
        return Err(lowering_error(
            gamma_parameter,
            "ideal-gas gamma must be finite and greater than one",
        ));
    }
    let model = IdealGasEulerModel1d {
        domain,
        representation,
        bounds: *bounds,
        boundaries,
        conservative_fields: [fields.density, fields.momentum, total_energy],
        velocity: fields.velocity,
        pressure,
        gamma_parameter,
        gamma,
        balance_relations,
        velocity_relation,
        pressure_relation,
        flux_operator: scalar_flux_lift()
            .expect("the closed scalar-flux definition is valid")
            .digest(),
    };
    require_closed_model(program, &model)?;
    Ok(model)
}

struct EulerFields {
    all: Vec<RawId>,
    density: RawId,
    momentum: RawId,
    velocity: RawId,
    energy_scalars: Vec<RawId>,
}

fn exact_euler_fields(program: &KernelProgram, domain: RawId) -> Result<EulerFields, Diagnostic> {
    let fields = program
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Field(field)
                if has_edge(program, field.id().erase(), domain, EdgeKind::DefinedOn) =>
            {
                Some(field)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let scalars = |dimension| {
        fields
            .iter()
            .filter(|field| {
                field.dimension() == dimension
                    && field.shape() == &ValueShape::scalar()
                    && field.frame() == ValueFrame::Invariant
            })
            .map(|field| field.id().erase())
            .collect::<Vec<_>>()
    };
    let density = scalars(DENSITY);
    let momentum = scalars(MOMENTUM_DENSITY);
    let velocity = scalars(VELOCITY);
    let energy_scalars = scalars(ENERGY_DENSITY);
    if fields.len() != 5
        || density.len() != 1
        || momentum.len() != 1
        || velocity.len() != 1
        || energy_scalars.len() != 2
    {
        return Err(lowering_error(
            domain,
            "ideal-gas Euler requires exactly scalar density, momentum, velocity, total-energy and pressure Fields with coherent SI dimensions",
        ));
    }
    Ok(EulerFields {
        all: fields.iter().map(|field| field.id().erase()).collect(),
        density: density[0],
        momentum: momentum[0],
        velocity: velocity[0],
        energy_scalars,
    })
}

fn exact_interval_boundaries(
    program: &KernelProgram,
    domain: RawId,
) -> Result<[RawId; 2], Diagnostic> {
    let mut boundaries = BTreeMap::new();
    for node in program.nodes() {
        let KernelNode::Domain(boundary) = node else {
            continue;
        };
        let DomainKind::CartesianBoundary { axis, side } = boundary.kind() else {
            continue;
        };
        if boundary_parent(program, boundary.id().erase()) != Some(domain) {
            continue;
        }
        if *axis != 0 || boundaries.insert(*side, boundary.id().erase()).is_some() {
            return Err(lowering_error(
                boundary.id().erase(),
                "ideal-gas Euler interval boundary ownership is duplicated or outside axis zero",
            ));
        }
    }
    let Some(lower) = boundaries.get(&BoundarySide::Lower).copied() else {
        return Err(lowering_error(
            domain,
            "ideal-gas Euler interval is missing its lower boundary",
        ));
    };
    let Some(upper) = boundaries.get(&BoundarySide::Upper).copied() else {
        return Err(lowering_error(
            domain,
            "ideal-gas Euler interval is missing its upper boundary",
        ));
    };
    Ok([lower, upper])
}

fn shared_continuum_representation(
    program: &KernelProgram,
    fields: &[RawId],
    owner: RawId,
) -> Result<RawId, Diagnostic> {
    let representations = fields
        .iter()
        .map(|field| {
            let matches = program
                .edges()
                .iter()
                .filter(|edge| edge.from() == *field && edge.kind() == EdgeKind::DefinedOn)
                .filter_map(|edge| match program.node(edge.to()) {
                    Some(KernelNode::Representation(representation))
                        if representation.kind() == RepresentationKind::Continuum =>
                    {
                        Some(edge.to())
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            (matches.len() == 1).then(|| matches[0])
        })
        .collect::<Option<Vec<_>>>();
    let Some(representations) = representations else {
        return Err(lowering_error(
            owner,
            "every ideal-gas Euler Field requires exactly one continuum Representation",
        ));
    };
    if representations
        .iter()
        .all(|value| *value == representations[0])
    {
        Ok(representations[0])
    } else {
        Err(lowering_error(
            owner,
            "ideal-gas Euler Fields must share one continuum Representation",
        ))
    }
}

fn typed_relation(
    program: &KernelProgram,
    relation: RawId,
) -> Result<TypedResidual<RawId>, Diagnostic> {
    let relation_id = relation
        .downcast()
        .ok_or_else(|| lowering_error(relation, "Euler calculus owner is not a Relation"))?;
    program
        .typed_relation_residual(relation_id)
        .map_err(|diagnostics| {
            diagnostics.into_iter().next().unwrap_or_else(|| {
                lowering_error(
                    relation,
                    "Euler calculus typing failed without a diagnostic",
                )
            })
        })
}

fn require_continuous_relation(program: &KernelProgram, relation: RawId) -> Result<(), Diagnostic> {
    let activations = program
        .edges()
        .iter()
        .filter(|edge| edge.kind() == EdgeKind::Activates && edge.to() == relation)
        .filter_map(|edge| match program.node(edge.from()) {
            Some(KernelNode::Activation(activation)) => Some(activation),
            _ => None,
        })
        .collect::<Vec<_>>();
    if activations.len() == 1 && activations[0].kind() == &ActivationKind::Continuous {
        Ok(())
    } else {
        Err(lowering_error(
            relation,
            "ideal-gas Euler Relations require exactly one continuous Activation",
        ))
    }
}

fn dimensionless_parameters(program: &KernelProgram) -> Vec<(RawId, f64)> {
    program
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Parameter(parameter) => parameter
                .real_scalar_value()
                .filter(|value| value.dim() == DimExponents::DIMENSIONLESS)
                .map(|value| (parameter.id().erase(), value.value())),
            _ => None,
        })
        .collect()
}

fn matches_velocity_closure(
    residual: &TypedResidual<RawId>,
    owner: RawId,
    momentum: RawId,
    density: RawId,
    velocity: RawId,
) -> Result<bool, Diagnostic> {
    let Some((field, expression)) = definition_parts(residual.expression(), owner)? else {
        return Ok(false);
    };
    Ok(field == momentum
        && matches_product_of(
            residual.expression(),
            expression,
            |dag, value| is_field(dag, value, density),
            |dag, value| is_field(dag, value, velocity),
        ))
}

fn matches_pressure_closure(
    residual: &TypedResidual<RawId>,
    owner: RawId,
    pressure: RawId,
    total_energy: RawId,
    momentum: RawId,
    velocity: RawId,
    gamma: RawId,
) -> Result<bool, Diagnostic> {
    let expression = residual.expression();
    let Some((field, value)) = definition_parts(expression, owner)? else {
        return Ok(false);
    };
    if field != pressure {
        return Ok(false);
    }
    let Some(ExprNode::Mul(left, right)) = expression.node(value) else {
        return Ok(false);
    };
    for (gamma_factor, internal_energy) in [(*left, *right), (*right, *left)] {
        if !matches_gamma_minus_one(expression, gamma_factor, gamma) {
            continue;
        }
        let Some(ExprNode::Sub(energy, kinetic)) = expression.node(internal_energy) else {
            continue;
        };
        if !is_field(expression, *energy, total_energy) {
            continue;
        }
        let mut factors = Vec::new();
        multiplicative_leaves(expression, *kinetic, 0, &mut factors);
        if factors.len() == 3
            && factors
                .iter()
                .any(|factor| is_dimensionless_constant(expression, *factor, 0.5))
            && factors
                .iter()
                .any(|factor| is_field(expression, *factor, momentum))
            && factors
                .iter()
                .any(|factor| is_field(expression, *factor, velocity))
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn matches_balance(
    residual: &TypedResidual<RawId>,
    owner: RawId,
    state: RawId,
    momentum: RawId,
    velocity: RawId,
    pressure: RawId,
    total_energy: RawId,
) -> Result<bool, Diagnostic> {
    let expression = residual.expression();
    let root = unique_root(expression, owner)?;
    let view = AdditiveResidualView::derive(expression, root, owner)?;
    if view.leaves().len() != 2 || view.leaves()[0].sign() != view.leaves()[1].sign() {
        return Ok(false);
    }
    let mut derivative = None;
    let mut divergence = None;
    for leaf in view.leaves() {
        match expression.node(leaf.value()) {
            Some(ExprNode::Symbol(SymbolRef::Derivative(field, std::num::NonZeroU32::MIN)))
                if field.erase() == state =>
            {
                derivative = Some(leaf.value());
            }
            Some(ExprNode::Divergence(flux)) => divergence = Some(*flux),
            _ => {}
        }
    }
    if derivative.is_none() {
        return Ok(false);
    }
    let Some(flux) = divergence else {
        return Ok(false);
    };
    let Some(argument) = scalar_flux_argument(residual, flux, owner)? else {
        return Ok(false);
    };
    if state == momentum {
        Ok(matches_momentum_flux(
            expression, argument, momentum, velocity, pressure,
        ))
    } else if state == total_energy {
        Ok(matches_energy_flux(
            expression,
            argument,
            velocity,
            total_energy,
            pressure,
        ))
    } else {
        Ok(is_field(expression, argument, momentum))
    }
}

fn definition_parts(
    expression: &ExprDag,
    owner: RawId,
) -> Result<Option<(RawId, ExprId)>, Diagnostic> {
    let root = unique_root(expression, owner)?;
    let view = AdditiveResidualView::derive(expression, root, owner)?;
    if view.leaves().len() != 2 || !view.leaves()[0].sign().is_opposite(view.leaves()[1].sign()) {
        return Ok(None);
    }
    for index in 0..2 {
        let field_leaf = &view.leaves()[index];
        let Some(ExprNode::Symbol(SymbolRef::Field(field))) = expression.node(field_leaf.value())
        else {
            continue;
        };
        return Ok(Some((field.erase(), view.leaves()[1 - index].value())));
    }
    Ok(None)
}

fn scalar_flux_argument(
    residual: &TypedResidual<RawId>,
    flux: ExprId,
    owner: RawId,
) -> Result<Option<ExprId>, Diagnostic> {
    let expected = scalar_flux_lift().map_err(|error| {
        lowering_error(
            owner,
            format!("canonical scalar-flux lift is invalid: {error}"),
        )
    })?;
    let Some(proof) =
        PureOperatorApplicationProof::classify(residual, flux, &expected).map_err(|error| {
            lowering_error(
                owner,
                format!(
                    "scalar-flux pure-operator proof failed at node {}: {error}",
                    flux.index()
                ),
            )
        })?
    else {
        return Ok(None);
    };
    Ok((proof.arguments().len() == 1).then(|| proof.arguments()[0]))
}

fn scalar_flux_lift() -> Result<PureOperatorDefinition, PureOperatorError> {
    let mut builder = CalculusBuilder::new(
        [PureValueClass::invariant_scalar()],
        PureValueClass::spatial_tensor(1)?,
    )?;
    let value = builder.push(CalculusNode::FormalComponent {
        formal: 0,
        axes: Box::default(),
    })?;
    builder.finish(value)
}

fn matches_momentum_flux(
    expression: &ExprDag,
    value: ExprId,
    momentum: RawId,
    velocity: RawId,
    pressure: RawId,
) -> bool {
    let Some(ExprNode::Add(left, right)) = expression.node(value) else {
        return false;
    };
    [(*left, *right), (*right, *left)]
        .into_iter()
        .any(|(dynamic, pressure_value)| {
            is_field(expression, pressure_value, pressure)
                && matches_product_of(
                    expression,
                    dynamic,
                    |dag, node| is_field(dag, node, momentum),
                    |dag, node| is_field(dag, node, velocity),
                )
        })
}

fn matches_energy_flux(
    expression: &ExprDag,
    value: ExprId,
    velocity: RawId,
    total_energy: RawId,
    pressure: RawId,
) -> bool {
    matches_product_of(
        expression,
        value,
        |dag, node| is_field(dag, node, velocity),
        |dag, node| {
            let Some(ExprNode::Add(left, right)) = dag.node(node) else {
                return false;
            };
            (is_field(dag, *left, total_energy) && is_field(dag, *right, pressure))
                || (is_field(dag, *right, total_energy) && is_field(dag, *left, pressure))
        },
    )
}

fn matches_product_of(
    expression: &ExprDag,
    value: ExprId,
    left_matches: impl Fn(&ExprDag, ExprId) -> bool,
    right_matches: impl Fn(&ExprDag, ExprId) -> bool,
) -> bool {
    let Some(ExprNode::Mul(left, right)) = expression.node(value) else {
        return false;
    };
    (left_matches(expression, *left) && right_matches(expression, *right))
        || (left_matches(expression, *right) && right_matches(expression, *left))
}

fn multiplicative_leaves(
    expression: &ExprDag,
    value: ExprId,
    depth: usize,
    leaves: &mut Vec<ExprId>,
) {
    if depth < 4
        && let Some(ExprNode::Mul(left, right)) = expression.node(value)
    {
        multiplicative_leaves(expression, *left, depth + 1, leaves);
        multiplicative_leaves(expression, *right, depth + 1, leaves);
    } else {
        leaves.push(value);
    }
}

fn matches_gamma_minus_one(expression: &ExprDag, value: ExprId, gamma: RawId) -> bool {
    matches!(expression.node(value), Some(ExprNode::Sub(left, right))
        if matches!(expression.node(*left), Some(ExprNode::Symbol(SymbolRef::Parameter(parameter))) if parameter.erase() == gamma)
            && is_dimensionless_constant(expression, *right, 1.0))
}

fn is_dimensionless_constant(expression: &ExprDag, value: ExprId, expected: f64) -> bool {
    matches!(expression.node(value), Some(ExprNode::Constant(value))
        if value.real_scalar_value().is_some_and(|value|
            value.dim() == DimExponents::DIMENSIONLESS && value.value() == expected))
}

fn is_field(expression: &ExprDag, value: ExprId, expected: RawId) -> bool {
    matches!(expression.node(value), Some(ExprNode::Symbol(SymbolRef::Field(field))) if field.erase() == expected)
}

fn has_edge(program: &KernelProgram, from: RawId, to: RawId, kind: EdgeKind) -> bool {
    program
        .edges()
        .iter()
        .any(|edge| edge.from() == from && edge.to() == to && edge.kind() == kind)
}

fn require_closed_model(
    program: &KernelProgram,
    model: &IdealGasEulerModel1d,
) -> Result<(), Diagnostic> {
    let domains = BTreeSet::from([model.domain, model.boundaries[0], model.boundaries[1]]);
    let fields = model
        .conservative_fields
        .into_iter()
        .chain([model.velocity, model.pressure])
        .collect::<BTreeSet<_>>();
    let relations = model
        .balance_relations
        .into_iter()
        .chain([model.velocity_relation, model.pressure_relation])
        .collect::<BTreeSet<_>>();
    let activations = program
        .edges()
        .iter()
        .filter(|edge| edge.kind() == EdgeKind::Activates && relations.contains(&edge.to()))
        .map(|edge| edge.from())
        .collect::<BTreeSet<_>>();
    for node in program.nodes() {
        let admitted = match node {
            KernelNode::Domain(value) => domains.contains(&value.id().erase()),
            KernelNode::Representation(value) => value.id().erase() == model.representation,
            KernelNode::Field(value) => fields.contains(&value.id().erase()),
            KernelNode::Parameter(value) => value.id().erase() == model.gamma_parameter,
            KernelNode::Relation(value) => relations.contains(&value.id().erase()),
            KernelNode::Activation(value) => activations.contains(&value.id().erase()),
            _ => false,
        };
        if !admitted {
            return Err(model_lowering_error(
                program,
                format!(
                    "closed ideal-gas Euler recognition would ignore unexpected {:?} node {}",
                    node.kind(),
                    node.id()
                ),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "canonical_euler/tests.rs"]
mod tests;
