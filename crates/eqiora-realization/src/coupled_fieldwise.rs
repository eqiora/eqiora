use std::cmp::Ordering;
use std::collections::BTreeSet;

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, DynQuantity, Id};
use eqiora_solver::{LinearOperatorProperties, SolverPlan};

use crate::{
    AlgebraicBlock, Discretization, DiscretizationMethod, ExecutionSchedule, FieldSpaceBinding,
    MeshPolicy, PositivePhysicalScale, QuadraturePolicy, SpaceFamily, SymmetricCongruenceScaling,
    Target, invalid_realization,
};

/// One physical Semantic Field represented by a coupled Realization.
///
/// This inventory is independent of whether coefficients are algebraic
/// unknowns or reconstructed from an eliminated time-state relation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RepresentedPhysicalField {
    domain: Id<kinds::Domain>,
    field: Id<kinds::Field>,
}

impl RepresentedPhysicalField {
    const fn new(domain: Id<kinds::Domain>, field: Id<kinds::Field>) -> Self {
        Self { domain, field }
    }

    /// Exact support Domain.
    #[must_use]
    pub const fn domain(self) -> Id<kinds::Domain> {
        self.domain
    }

    /// Exact physical Semantic Field.
    #[must_use]
    pub const fn field(self) -> Id<kinds::Field> {
        self.field
    }
}

/// One exact Domain and all Semantic Fields participating in its realization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DomainFieldInventory {
    domain: Id<kinds::Domain>,
    fields: Vec<Id<kinds::Field>>,
}

impl DomainFieldInventory {
    /// Construct a canonical, nonempty Field inventory.
    ///
    /// # Errors
    /// Returns `EQ0807` for an empty or duplicate Field inventory.
    pub fn new(
        domain: Id<kinds::Domain>,
        fields: impl IntoIterator<Item = Id<kinds::Field>>,
    ) -> Result<Self, Diagnostic> {
        let mut fields = fields.into_iter().collect::<Vec<_>>();
        fields.sort_by_key(Id::ulid);
        if fields.is_empty() {
            return Err(invalid_realization(
                "a coupled field-wise Domain requires at least one participating Semantic Field",
            ));
        }
        if fields.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(invalid_realization(
                "a coupled field-wise Domain contains a duplicate participating Semantic Field",
            ));
        }
        Ok(Self { domain, fields })
    }

    /// Exact Semantic Domain.
    #[must_use]
    pub const fn domain(&self) -> Id<kinds::Domain> {
        self.domain
    }

    /// Canonically ordered exact participating Fields on this Domain.
    #[must_use]
    pub fn fields(&self) -> &[Id<kinds::Field>] {
        &self.fields
    }
}

/// One Domain/Field endpoint participating in an exact trace quotient.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraceFieldEndpoint {
    domain: Id<kinds::Domain>,
    field: Id<kinds::Field>,
}

impl TraceFieldEndpoint {
    /// Select one exact Field trace on one exact Domain.
    #[must_use]
    pub const fn new(domain: Id<kinds::Domain>, field: Id<kinds::Field>) -> Self {
        Self { domain, field }
    }

    /// Selected Domain.
    #[must_use]
    pub const fn domain(self) -> Id<kinds::Domain> {
        self.domain
    }

    /// Selected Field.
    #[must_use]
    pub const fn field(self) -> Id<kinds::Field> {
        self.field
    }
}

/// Equality quotient of two conforming Field traces selected by one Connection.
///
/// This is a numerical identity choice, not a physical interface definition.
/// The semantic lowerer remains responsible for proving conserving Connection
/// semantics and compatible Field shape, support, units, frame, and orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConformingTraceQuotient {
    connection: Id<kinds::Connection>,
    endpoints: [TraceFieldEndpoint; 2],
}

impl ConformingTraceQuotient {
    /// Construct a canonically ordered cross-Domain trace quotient.
    ///
    /// # Errors
    /// Returns `EQ0807` when both endpoints belong to the same Domain.
    pub fn new(
        connection: Id<kinds::Connection>,
        first: TraceFieldEndpoint,
        second: TraceFieldEndpoint,
    ) -> Result<Self, Diagnostic> {
        if first.domain == second.domain {
            return Err(invalid_realization(
                "a conforming trace quotient must join Fields on distinct Domains",
            ));
        }
        let mut endpoints = [first, second];
        endpoints.sort_by(endpoint_order);
        Ok(Self {
            connection,
            endpoints,
        })
    }

    /// Exact conserving Connection selected by the semantic lowerer.
    #[must_use]
    pub const fn connection(self) -> Id<kinds::Connection> {
        self.connection
    }

    /// Canonically ordered trace endpoints.
    #[must_use]
    pub const fn endpoints(self) -> [TraceFieldEndpoint; 2] {
        self.endpoints
    }
}

/// One exact Domain and its algebraic Field-wise spatial choices.
#[derive(Debug, Clone, PartialEq)]
pub struct DomainFieldDiscretization {
    domain: Id<kinds::Domain>,
    field_spaces: Vec<FieldSpaceBinding>,
    constraints: Vec<crate::AlgebraicConstraint>,
}

impl DomainFieldDiscretization {
    /// Construct one canonical Domain-local algebraic Field-space selection.
    ///
    /// # Errors
    /// Returns `EQ0807` for an empty/duplicate binding, duplicate constraint,
    /// or a constraint referring to a Field outside this Domain selection.
    pub fn new(
        domain: Id<kinds::Domain>,
        field_spaces: impl IntoIterator<Item = FieldSpaceBinding>,
        constraints: impl IntoIterator<Item = crate::AlgebraicConstraint>,
    ) -> Result<Self, Diagnostic> {
        let mut field_spaces = field_spaces.into_iter().collect::<Vec<_>>();
        field_spaces.sort_by_key(|binding| binding.field().ulid());
        if field_spaces.is_empty() {
            return Err(invalid_realization(
                "a coupled field-wise Domain requires at least one Field-space binding",
            ));
        }
        if field_spaces
            .windows(2)
            .any(|pair| pair[0].field() == pair[1].field())
        {
            return Err(invalid_realization(
                "a coupled field-wise Domain contains a duplicate Field-space binding",
            ));
        }
        let mut constraints = constraints.into_iter().collect::<Vec<_>>();
        constraints.sort_by_key(|constraint| constraint.field().ulid());
        if constraints
            .windows(2)
            .any(|pair| pair[0].field() == pair[1].field())
        {
            return Err(invalid_realization(
                "a coupled field-wise Domain contains a duplicate Field constraint",
            ));
        }
        if constraints.iter().any(|constraint| {
            !field_spaces
                .iter()
                .any(|binding| binding.field() == constraint.field())
        }) {
            return Err(invalid_realization(
                "a coupled algebraic constraint refers to a Field outside its Domain selection",
            ));
        }
        Ok(Self {
            domain,
            field_spaces,
            constraints,
        })
    }

    /// Exact Semantic Domain.
    #[must_use]
    pub const fn domain(&self) -> Id<kinds::Domain> {
        self.domain
    }

    /// Canonically ordered exact Field-space bindings.
    #[must_use]
    pub fn field_spaces(&self) -> &[FieldSpaceBinding] {
        &self.field_spaces
    }

    /// Canonically ordered Domain-local algebraic constraints.
    #[must_use]
    pub fn constraints(&self) -> &[crate::AlgebraicConstraint] {
        &self.constraints
    }
}

/// Shared-mesh algebraic spatial selection over multiple exact Semantic Domains.
#[derive(Debug, Clone, PartialEq)]
pub struct CoupledFieldwiseSpatialDiscretization {
    coordinate_length_scale: PositivePhysicalScale,
    domains: Vec<DomainFieldDiscretization>,
    trace_quotients: Vec<ConformingTraceQuotient>,
    discretization: Discretization,
}

impl CoupledFieldwiseSpatialDiscretization {
    /// Construct a canonical multi-Domain selection over one imported mesh.
    ///
    /// # Errors
    /// Returns `EQ0807` unless there are at least two distinct Domains, every
    /// algebraic Field is bound exactly once globally, every trace endpoint belongs to
    /// its selected Domain with a matching trace-space signature, the quotient collection
    /// is nonempty without duplicate selections, and the coordinate scale has length dimension.
    pub fn new(
        coordinate_length_scale: PositivePhysicalScale,
        domains: impl IntoIterator<Item = DomainFieldDiscretization>,
        trace_quotients: impl AsRef<[ConformingTraceQuotient]>,
        discretization: Discretization,
    ) -> Result<Self, Diagnostic> {
        if coordinate_length_scale.quantity().dim() != length_dimension() {
            return Err(invalid_realization(
                "coupled field-wise coordinate scale must have physical length dimension",
            ));
        }
        let mut domains = domains.into_iter().collect::<Vec<_>>();
        domains.sort_by_key(|domain| domain.domain.ulid());
        validate_domain_selections(&domains)?;
        let trace_quotients = canonical_trace_quotients(trace_quotients.as_ref().iter().copied())?;
        for &quotient in &trace_quotients {
            validate_trace_selection(&domains, quotient)?;
        }
        Ok(Self {
            coordinate_length_scale,
            domains,
            trace_quotients,
            discretization,
        })
    }

    /// Characteristic coordinate length in coherent physical units.
    #[must_use]
    pub const fn coordinate_length_scale(&self) -> PositivePhysicalScale {
        self.coordinate_length_scale
    }

    /// Canonically ordered exact Domain selections.
    #[must_use]
    pub fn domains(&self) -> &[DomainFieldDiscretization] {
        &self.domains
    }

    /// Canonically ordered exact conforming trace quotients.
    #[must_use]
    pub fn trace_quotients(&self) -> &[ConformingTraceQuotient] {
        &self.trace_quotients
    }

    /// Shared method, imported mesh, and quadrature selection.
    #[must_use]
    pub const fn discretization(&self) -> Discretization {
        self.discretization
    }
}

/// Exact state/rate pair used by one Backward Euler elimination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BackwardEulerStatePair {
    relation: Id<kinds::Relation>,
    state: Id<kinds::Field>,
    rate: Id<kinds::Field>,
}

impl BackwardEulerStatePair {
    /// Select one distinct state Field and its time-derivative Field.
    ///
    /// # Errors
    /// Returns `EQ0807` when the same Field is selected for both roles.
    pub fn new(
        relation: Id<kinds::Relation>,
        state: Id<kinds::Field>,
        rate: Id<kinds::Field>,
    ) -> Result<Self, Diagnostic> {
        if state == rate {
            return Err(invalid_realization(
                "Backward Euler state and rate must be distinct Semantic Fields",
            ));
        }
        Ok(Self {
            relation,
            state,
            rate,
        })
    }

    /// Exact kinematic Relation owning this elimination.
    #[must_use]
    pub const fn relation(self) -> Id<kinds::Relation> {
        self.relation
    }

    /// Eliminated state Field represented after the step.
    #[must_use]
    pub const fn state(self) -> Id<kinds::Field> {
        self.state
    }

    /// Algebraic rate Field retained in the finalized operator.
    #[must_use]
    pub const fn rate(self) -> Id<kinds::Field> {
        self.rate
    }
}

/// Discrete representation of one state eliminated through Backward Euler.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BackwardEulerStateBinding {
    pair: BackwardEulerStatePair,
    state_space: crate::Space,
    state_scale: PositivePhysicalScale,
}

impl BackwardEulerStateBinding {
    /// Bind an eliminated state to its exact rate, space, and physical scale.
    #[must_use]
    pub const fn new(
        pair: BackwardEulerStatePair,
        state_space: crate::Space,
        state_scale: PositivePhysicalScale,
    ) -> Self {
        Self {
            pair,
            state_space,
            state_scale,
        }
    }

    /// Exact state/rate identity pair.
    #[must_use]
    pub const fn pair(self) -> BackwardEulerStatePair {
        self.pair
    }

    /// Discrete space retained for the reconstructed state.
    #[must_use]
    pub const fn state_space(self) -> crate::Space {
        self.state_space
    }

    /// Characteristic physical scale of the reconstructed state.
    #[must_use]
    pub const fn state_scale(self) -> PositivePhysicalScale {
        self.state_scale
    }
}

/// Positive Backward Euler step and its exact eliminated-state inventory.
#[derive(Debug, Clone, PartialEq)]
pub struct BackwardEulerStep {
    duration: DynQuantity,
    eliminated_states: Vec<BackwardEulerStateBinding>,
}

impl BackwardEulerStep {
    /// Validate duration and canonicalize the exact kinematic inventory.
    ///
    /// # Errors
    /// Returns `EQ0807` for invalid time, empty inventory, repeated Relation,
    /// repeated state/rate, or an algebraic rate that is also eliminated.
    pub fn new(
        duration: DynQuantity,
        eliminated_states: impl IntoIterator<Item = BackwardEulerStateBinding>,
    ) -> Result<Self, Diagnostic> {
        if duration.dim() != time_dimension()
            || !duration.value().is_finite()
            || duration.value() <= 0.0
        {
            return Err(invalid_realization(
                "Backward Euler step duration must be finite, strictly positive, and have physical time dimension",
            ));
        }
        let mut eliminated_states = eliminated_states.into_iter().collect::<Vec<_>>();
        eliminated_states.sort_by_key(|state| state.pair().state().ulid());
        canonical_state_pairs(eliminated_states.iter().map(|state| state.pair()))?;
        Ok(Self {
            duration,
            eliminated_states,
        })
    }

    /// Exact step duration in coherent SI base units.
    #[must_use]
    pub const fn duration(&self) -> DynQuantity {
        self.duration
    }

    /// Exact discrete states in canonical state Field identity order.
    #[must_use]
    pub fn eliminated_states(&self) -> &[BackwardEulerStateBinding] {
        &self.eliminated_states
    }
}

pub(crate) fn canonical_state_pairs(
    pairs: impl IntoIterator<Item = BackwardEulerStatePair>,
) -> Result<Vec<BackwardEulerStatePair>, Diagnostic> {
    let mut pairs = pairs.into_iter().collect::<Vec<_>>();
    pairs.sort_by_key(|pair| pair.state().ulid());
    let mut relations = BTreeSet::new();
    let mut states = BTreeSet::new();
    let mut rates = BTreeSet::new();
    if pairs.is_empty()
        || pairs.iter().any(|pair| {
            !relations.insert(pair.relation().ulid())
                || !states.insert(pair.state().ulid())
                || !rates.insert(pair.rate().ulid())
        })
        || !states.is_disjoint(&rates)
    {
        return Err(invalid_realization(
            "Backward Euler requires a nonempty exact inventory with unique Relations, states and rates and no eliminated rate",
        ));
    }
    Ok(pairs)
}

/// Complete physics-neutral multi-Domain Field-wise realization selection.
#[derive(Debug, Clone, PartialEq)]
pub struct CoupledFieldwiseRealizationPlan {
    spatial: CoupledFieldwiseSpatialDiscretization,
    time_step: BackwardEulerStep,
    scaling: SymmetricCongruenceScaling,
    operator_properties: LinearOperatorProperties,
    solver: SolverPlan,
    target: Target,
    schedule: ExecutionSchedule,
}

impl CoupledFieldwiseRealizationPlan {
    /// Construct and cross-validate one complete multi-Domain selection.
    ///
    /// # Errors
    /// Returns `EQ0807` unless the shared spatial family is the admitted
    /// continuous-Galerkin/imported-simplex/Duffy family and congruence
    /// scaling covers every Field and constraint multiplier exactly once.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        spatial: CoupledFieldwiseSpatialDiscretization,
        time_step: BackwardEulerStep,
        scaling: SymmetricCongruenceScaling,
        operator_properties: LinearOperatorProperties,
        solver: SolverPlan,
        target: Target,
        schedule: ExecutionSchedule,
    ) -> Result<Self, Diagnostic> {
        let value = Self {
            spatial,
            time_step,
            scaling,
            operator_properties,
            solver,
            target,
            schedule,
        };
        value.validate()?;
        Ok(value)
    }

    /// Shared multi-Domain spatial selection.
    #[must_use]
    pub const fn spatial(&self) -> &CoupledFieldwiseSpatialDiscretization {
        &self.spatial
    }

    /// Fixed Backward Euler step selection.
    #[must_use]
    pub const fn time_step(&self) -> &BackwardEulerStep {
        &self.time_step
    }

    /// Explicit symmetric congruence scaling.
    #[must_use]
    pub const fn scaling(&self) -> &SymmetricCongruenceScaling {
        &self.scaling
    }

    /// Mathematical property asserted for the complete operator.
    #[must_use]
    pub const fn operator_properties(&self) -> LinearOperatorProperties {
        self.operator_properties
    }

    /// Sole linear solver plan.
    #[must_use]
    pub const fn solver(&self) -> SolverPlan {
        self.solver
    }

    /// Deployment target.
    #[must_use]
    pub const fn target(&self) -> Target {
        self.target
    }

    /// Deployment schedule.
    #[must_use]
    pub const fn schedule(&self) -> ExecutionSchedule {
        self.schedule
    }

    /// Physical Semantic Fields represented by this plan, paired with their
    /// exact support Domains in canonical Field identity order.
    ///
    /// Algebraic Field blocks and a represented-but-eliminated state are both
    /// physical observations. Constraint multipliers are numerical unknowns
    /// and are intentionally absent.
    ///
    /// # Errors
    /// Returns `EQ0807` if the validated eliminated-state rate cannot be
    /// associated with one exact Domain.
    pub fn represented_physical_fields(&self) -> Result<Vec<RepresentedPhysicalField>, Diagnostic> {
        let mut fields = self
            .spatial
            .domains
            .iter()
            .flat_map(|domain| {
                domain.field_spaces.iter().map(move |binding| {
                    RepresentedPhysicalField::new(domain.domain, binding.field())
                })
            })
            .collect::<Vec<_>>();
        for state in self.time_step.eliminated_states() {
            let pair = state.pair();
            let domain = fields
                .iter()
                .find(|entry| entry.field == pair.rate)
                .map(|entry| entry.domain)
                .ok_or_else(|| {
                    invalid_realization(
                        "represented eliminated-state rate has no exact Domain binding",
                    )
                })?;
            fields.push(RepresentedPhysicalField::new(domain, pair.state));
        }
        fields.sort_by_key(|entry| entry.field.ulid());
        Ok(fields)
    }

    pub(crate) fn validate(&self) -> Result<(), Diagnostic> {
        let discretization = self.spatial.discretization;
        if !matches!(
            (
                discretization.method(),
                discretization.mesh(),
                discretization.quadrature(),
            ),
            (
                DiscretizationMethod::ContinuousGalerkin,
                MeshPolicy::ImportedSimplicial { .. },
                QuadraturePolicy::TriangleDuffyGaussLegendre { .. }
                    | QuadraturePolicy::SimplexDuffyGaussLegendre { .. },
            )
        ) || self
            .spatial
            .domains
            .iter()
            .flat_map(|domain| &domain.field_spaces)
            .any(|binding| {
                !matches!(
                    binding.space().family(),
                    SpaceFamily::ContinuousLagrange { .. } | SpaceFamily::SimplexP1Bubble
                )
            })
        {
            return Err(invalid_realization(
                "coupled field-wise v0 requires continuous Galerkin, one imported affine-simplex mesh, explicit simplex Duffy quadrature, and continuous Lagrange or simplex P1-bubble spaces",
            ));
        }
        if matches!(self.schedule, ExecutionSchedule::RealTime { .. })
            && matches!(self.target, Target::CudaGpu { .. })
        {
            return Err(invalid_realization(
                "the coupled field-wise CUDA target has no declared real-time scheduling contract",
            ));
        }
        let mut expected = self
            .spatial
            .domains
            .iter()
            .flat_map(|domain| {
                domain
                    .field_spaces
                    .iter()
                    .map(|binding| AlgebraicBlock::Field(binding.field()))
                    .chain(domain.constraints.iter().map(|constraint| {
                        AlgebraicBlock::ConstraintMultiplier {
                            field: constraint.field(),
                        }
                    }))
            })
            .collect::<Vec<_>>();
        expected.sort_by(algebraic_block_order);
        let actual = self
            .scaling
            .block_scales()
            .iter()
            .map(|entry| entry.block())
            .collect::<Vec<_>>();
        if actual != expected {
            return Err(invalid_realization(
                "coupled congruence scaling must cover every Field and constraint-multiplier block exactly once",
            ));
        }
        for quotient in &self.spatial.trace_quotients {
            let [first, second] = quotient.endpoints;
            let first_scale = field_scale(&self.scaling, first.field).ok_or_else(|| {
                invalid_realization("first trace endpoint Field has no congruence scale")
            })?;
            let second_scale = field_scale(&self.scaling, second.field).ok_or_else(|| {
                invalid_realization("second trace endpoint Field has no congruence scale")
            })?;
            if first_scale != second_scale {
                return Err(invalid_realization(
                    "Fields identified as one conforming trace quotient must have exactly equal congruence scales",
                ));
            }
        }
        for eliminated in self.time_step.eliminated_states() {
            let pair = eliminated.pair;
            if self
                .spatial
                .domains
                .iter()
                .flat_map(|domain| &domain.field_spaces)
                .any(|binding| binding.field() == pair.state)
            {
                return Err(invalid_realization(
                    "a Backward Euler eliminated state must not also be an algebraic Field block",
                ));
            }
            let Some(rate_binding) = self
                .spatial
                .domains
                .iter()
                .flat_map(|domain| &domain.field_spaces)
                .find(|binding| binding.field() == pair.rate)
            else {
                return Err(invalid_realization(
                    "a Backward Euler rate must be an algebraic Field block",
                ));
            };
            if rate_binding.space() != eliminated.state_space {
                return Err(invalid_realization(
                    "Backward Euler coefficient elimination requires identical state and rate spaces",
                ));
            }
            let rate_scale = field_scale(&self.scaling, pair.rate).ok_or_else(|| {
                invalid_realization("Backward Euler rate Field has no congruence scale")
            })?;
            if derivative_dimension(eliminated.state_scale.quantity().dim())
                != Some(rate_scale.quantity().dim())
            {
                return Err(invalid_realization(
                    "Backward Euler state-scale dimension divided by time must equal the rate-scale dimension",
                ));
            }
        }
        Ok(())
    }
}

pub(crate) fn canonical_trace_quotients(
    quotients: impl IntoIterator<Item = ConformingTraceQuotient>,
) -> Result<Vec<ConformingTraceQuotient>, Diagnostic> {
    let mut quotients = quotients.into_iter().collect::<Vec<_>>();
    quotients.sort_by(|left, right| {
        left.connection
            .ulid()
            .cmp(&right.connection.ulid())
            .then_with(|| endpoint_order(&left.endpoints[0], &right.endpoints[0]))
            .then_with(|| endpoint_order(&left.endpoints[1], &right.endpoints[1]))
    });
    if quotients.is_empty() || quotients.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(invalid_realization(
            "conforming trace quotients must be nonempty and contain no duplicate selection",
        ));
    }
    Ok(quotients)
}

fn validate_domain_selections(domains: &[DomainFieldDiscretization]) -> Result<(), Diagnostic> {
    if domains.len() < 2 {
        return Err(invalid_realization(
            "a coupled field-wise selection requires at least two Semantic Domains",
        ));
    }
    if domains
        .windows(2)
        .any(|pair| pair[0].domain == pair[1].domain)
    {
        return Err(invalid_realization(
            "a coupled field-wise selection contains a duplicate Semantic Domain",
        ));
    }
    let mut fields = domains
        .iter()
        .flat_map(|domain| domain.field_spaces.iter().map(|binding| binding.field()))
        .collect::<Vec<_>>();
    fields.sort_by_key(Id::ulid);
    if fields.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(invalid_realization(
            "a Semantic Field cannot be bound in more than one coupled Domain",
        ));
    }
    Ok(())
}

fn validate_trace_selection(
    domains: &[DomainFieldDiscretization],
    trace: ConformingTraceQuotient,
) -> Result<(), Diagnostic> {
    let mut trace_signatures = [0_u16; 2];
    for (index, endpoint) in trace.endpoints.iter().enumerate() {
        let Some(binding) = domains
            .iter()
            .find(|domain| domain.domain == endpoint.domain)
            .and_then(|domain| {
                domain
                    .field_spaces
                    .iter()
                    .find(|binding| binding.field() == endpoint.field)
            })
        else {
            return Err(invalid_realization(
                "each conforming trace endpoint must name a Field bound on its exact Domain",
            ));
        };
        trace_signatures[index] = match binding.space().family() {
            SpaceFamily::ContinuousLagrange { order } => order.get(),
            SpaceFamily::SimplexP1Bubble => 1,
            SpaceFamily::CellConstant => {
                return Err(invalid_realization(
                    "a cell-constant space has no admitted conforming trace signature",
                ));
            }
        };
    }
    if trace_signatures[0] != trace_signatures[1] {
        return Err(invalid_realization(
            "conforming trace quotient endpoints must have identical trace-space signatures",
        ));
    }
    Ok(())
}

fn field_scale(
    scaling: &SymmetricCongruenceScaling,
    field: Id<kinds::Field>,
) -> Option<PositivePhysicalScale> {
    scaling
        .block_scales()
        .iter()
        .find_map(|entry| (entry.block() == AlgebraicBlock::Field(field)).then_some(entry.scale()))
}

fn endpoint_order(left: &TraceFieldEndpoint, right: &TraceFieldEndpoint) -> Ordering {
    left.domain
        .ulid()
        .cmp(&right.domain.ulid())
        .then_with(|| left.field.ulid().cmp(&right.field.ulid()))
}

fn algebraic_block_order(left: &AlgebraicBlock, right: &AlgebraicBlock) -> Ordering {
    let (left_tag, left_field) = match left {
        AlgebraicBlock::Field(field) => (0, field),
        AlgebraicBlock::ConstraintMultiplier { field } => (1, field),
    };
    let (right_tag, right_field) = match right {
        AlgebraicBlock::Field(field) => (0, field),
        AlgebraicBlock::ConstraintMultiplier { field } => (1, field),
    };
    left_tag
        .cmp(&right_tag)
        .then_with(|| left_field.ulid().cmp(&right_field.ulid()))
}

const fn length_dimension() -> DimExponents {
    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension")
}

const fn time_dimension() -> DimExponents {
    DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("bounded dimension")
}

const fn derivative_dimension(dimension: DimExponents) -> Option<DimExponents> {
    dimension.div(time_dimension())
}
