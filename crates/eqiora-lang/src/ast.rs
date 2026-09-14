//! Source-oriented syntax tree. Semantic types are assigned during lowering.

pub(crate) mod comments;
mod compile_time;
mod items;
mod observable;
pub use observable::ObservableDecl;
pub(crate) mod nominal;
mod signature;
pub use items::{ComponentItem, Item};
pub use signature::SignatureItem;
pub(crate) mod document;
mod enumeration;
mod event;
mod record;
pub use enumeration::{CaseArm, EnumDecl};
mod conservation;
pub use record::{RecordDecl, RecordMemberDecl};
pub(crate) mod formulation;
mod name_path;
pub use conservation::ConservationSyntax;
mod relation;
pub use event::EventDecl;
pub use relation::{
    ActivationSyntax, InitialDecl, RelationBody, RelationCondition, RelationDecl,
    RelationFamilyDecl,
};
mod value_type;

pub use value_type::{ValueTypeSyntax, ValueTypeSyntaxKind};

pub use comments::DocComment;
pub use compile_time::{NamedDefinitionDecl, ParameterDecl};
pub use document::{Document, ModelDecl};

use formulation::FormulationDecl;
use std::ops::Range;

pub use crate::cartesian::{BoundarySideSyntax, CartesianCoordinateSyntax};

/// Half-open UTF-8 byte range in one source file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TextRange {
    pub(crate) start: u32,
    pub(crate) end: u32,
}

impl TextRange {
    /// Construct a byte range.
    #[must_use]
    pub const fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }

    /// Inclusive start byte.
    #[must_use]
    pub const fn start(self) -> u32 {
        self.start
    }

    /// Exclusive end byte.
    #[must_use]
    pub const fn end(self) -> u32 {
        self.end
    }
}

/// One nonempty source name, optionally qualified by lexical member selection.
///
/// Segment ranges make qualification structural without splitting or joining dotted strings.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NamePath {
    pub(crate) text: String,
    pub(crate) segments: Vec<Range<usize>>,
    pub(crate) range: TextRange,
}

impl NamePath {
    /// Canonical dotted source spelling.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.text
    }

    /// Identifier segments in lexical order.
    #[must_use]
    pub fn segments(&self) -> impl ExactSizeIterator<Item = &str> {
        self.segments.iter().map(|range| &self.text[range.clone()])
    }

    /// Full qualified-name range.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }
}

mod operator;
pub use operator::{
    ExactIntegerSyntax, PureOperatorDecl, PureOperatorFormal, PureValueClassSyntax,
};

/// A nominal compilation-unit connector family.
#[derive(Debug, Clone, PartialEq)]
pub struct ConnectorDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) visibility: VisibilitySyntax,
    pub(crate) name: String,
    pub(crate) syntax: ConnectorSyntax,
    pub(crate) range: TextRange,
}

impl ConnectorDecl {
    /// Package visibility. Unqualified declarations are private by default.
    #[must_use]
    pub const fn visibility(&self) -> VisibilitySyntax {
        self.visibility
    }

    /// Source name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Closed connector-family contract.
    #[must_use]
    pub const fn syntax(&self) -> &ConnectorSyntax {
        &self.syntax
    }

    /// Full connector declaration range, including a visibility modifier.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }
}

/// Source contract for one nominal connector family.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ConnectorSyntax {
    /// Scalar acausal connector with complete across and through types.
    ScalarPhysical {
        /// Declared member exposing the across quantity.
        across_name: String,
        /// Complete scalar type of the across variable.
        across_type: ValueTypeSyntax,
        /// Declared member exposing the through quantity.
        through_name: String,
        /// Complete scalar type of the through variable.
        through_type: ValueTypeSyntax,
    },
    /// Field-valued trace/flux pair on one oriented boundary support.
    FieldPhysical {
        /// Pointwise trace quantity.
        trace: ConnectorQuantitySyntax,
        /// Pointwise outward-flux quantity dual to `trace`.
        flux: ConnectorQuantitySyntax,
        /// Exact value shape, or one source convenience resolved on lowering.
        shape: ValueShapeSyntax,
        /// Coordinate-frame discipline shared by both quantities.
        frame: FrameSyntax,
        /// Closed boundary duality used by conserving connection sets.
        pairing: BoundaryPairingSyntax,
    },
}

/// One named quantity member of a field-valued physical Connector.
#[derive(Debug, Clone, PartialEq)]
pub struct ConnectorQuantitySyntax {
    pub(crate) name: String,
    pub(crate) dimension: Box<Expr>,
}

impl ConnectorQuantitySyntax {
    /// Source member name. Its declaration identity is assigned on lowering.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Static SI dimension expression.
    #[must_use]
    pub const fn dimension(&self) -> &Expr {
        &self.dimension
    }
}

/// Source spelling of an exact value shape or a context-dependent convenience.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ValueShapeSyntax {
    /// Rank-zero scalar, canonically formatted as `[]`.
    Scalar,
    /// Exact positive extents, canonically spelled `[e0, e1, ...]`.
    Exact(Vec<u32>),
    /// Vector whose extent is the parent support's ambient dimension.
    SpatialVector,
}

/// Source frame discipline for field-valued physical quantities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FrameSyntax {
    /// Frame-independent scalar or tensor components.
    Invariant,
    /// Components in the model-global Cartesian spatial frame.
    Spatial,
}

/// Source boundary pairing for one trace/flux dual pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BoundaryPairingSyntax {
    /// Pointwise Euclidean contraction followed by boundary integration.
    EuclideanBoundaryDuality,
}

/// A reusable typed source definition before deterministic elaboration.
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) visibility: VisibilitySyntax,
    pub(crate) name: String,
    pub(crate) signature: Vec<SignatureItem>,
    pub(crate) items: Vec<ComponentItem>,
    pub(crate) formulations: Vec<FormulationDecl>,
    pub(crate) range: TextRange,
}

impl ComponentDecl {
    /// Package visibility. Unqualified declarations are private by default.
    #[must_use]
    pub const fn visibility(&self) -> VisibilitySyntax {
        self.visibility
    }

    /// Source name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Public requirements and occurrence-owned exposed values.
    #[must_use]
    pub fn signature(&self) -> &[SignatureItem] {
        &self.signature
    }

    /// Component declarations in source order.
    #[must_use]
    pub fn items(&self) -> &[ComponentItem] {
        &self.items
    }
}

/// Source declaration visibility. Absence of a modifier parses as private.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum VisibilitySyntax {
    /// Visible only inside the owning component or package scope.
    #[default]
    Private,
    /// Part of the owning component or package's typed public interface.
    Public,
}

/// Component-local scalar Parameter declaration.
///
/// Visibility belongs to this type, so no Relation or instance can be made
/// public by constructing a generic decorated declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentParameterDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) visibility: VisibilitySyntax,
    pub(crate) name: String,
    pub(crate) value_type: ValueTypeSyntax,
    pub(crate) default: Option<Expr>,
    pub(crate) range: TextRange,
}

impl ComponentParameterDecl {
    /// Private-by-default or explicit public visibility.
    #[must_use]
    pub const fn visibility(&self) -> VisibilitySyntax {
        self.visibility
    }

    /// Source name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Static SI dimension expression.
    #[must_use]
    pub fn dimension(&self) -> Option<&Expr> {
        self.value_type.dimension()
    }

    /// Complete declared mathematical type.
    #[must_use]
    pub const fn value_type(&self) -> &ValueTypeSyntax {
        &self.value_type
    }

    /// Optional compile-time default expression. `None` is a required public
    /// binding; semantic validation rejects an unbound private Parameter.
    #[must_use]
    pub const fn default(&self) -> Option<&Expr> {
        self.default.as_ref()
    }

    /// Full declaration range, including `public` when present.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }
}

/// Component-local Port declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentPortDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) visibility: VisibilitySyntax,
    pub(crate) name: String,
    pub(crate) syntax: PortSyntax,
    pub(crate) range: TextRange,
}

/// One field-physical Port declaration expanded over a complete exterior.
///
/// This is a hierarchy-only family. The binder is not a general collection,
/// array, or runtime loop and no family value survives component elaboration.
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentPortFamilyDecl {
    pub(crate) port: ComponentPortDecl,
    pub(crate) binder: FamilyBinderSyntax,
}

impl ComponentPortFamilyDecl {
    /// Underlying field-physical Port declaration.
    #[must_use]
    pub const fn port(&self) -> &ComponentPortDecl {
        &self.port
    }

    /// Restricted boundary-member binder.
    #[must_use]
    pub const fn binder(&self) -> &FamilyBinderSyntax {
        &self.binder
    }

    /// Full family declaration range.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.port.range
    }
}

impl ComponentPortDecl {
    /// Private-by-default or explicit public visibility.
    #[must_use]
    pub const fn visibility(&self) -> VisibilitySyntax {
        self.visibility
    }

    /// Source name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Causal or conserving interface contract.
    #[must_use]
    pub const fn syntax(&self) -> &PortSyntax {
        &self.syntax
    }

    /// Full declaration range, including `public` when present.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }
}

/// Component-local spatial-support interface.
///
/// A support slot is not a scalar Parameter and never contains mesh or
/// discretization data. An instance binds it to one exact enclosing Domain
/// before deterministic component expansion.
#[derive(Debug, Clone, PartialEq)]
pub struct SupportSlotDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) visibility: VisibilitySyntax,
    pub(crate) name: String,
    pub(crate) syntax: SupportSlotSyntax,
    pub(crate) range: TextRange,
}

impl SupportSlotDecl {
    /// Private-by-default or explicit public visibility.
    #[must_use]
    pub const fn visibility(&self) -> VisibilitySyntax {
        self.visibility
    }

    /// Source name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Closed spatial-support slot contract.
    #[must_use]
    pub const fn syntax(&self) -> &SupportSlotSyntax {
        &self.syntax
    }

    /// Full declaration range, including `public` when present.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }
}

/// Source contract for one component spatial-support slot.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SupportSlotSyntax {
    /// A volume Domain with one exact ambient dimension.
    Volume {
        /// Required number of spatial coordinate axes.
        ambient_dimension: usize,
    },
    /// A boundary whose exact parent is supplied through another slot.
    Boundary {
        /// Name of the parent volume support slot in the same component.
        parent: String,
    },
    /// The complete exterior of an exact bound Cartesian volume.
    ///
    /// Members are supplied by one finite `boundaries(...)` occurrence
    /// binding. The set is hierarchy-only and never becomes a Kernel node.
    CompleteExterior {
        /// Name of the parent volume support slot in the same component.
        parent: String,
    },
}

/// Lexical binder for a member of a finite index or complete exterior support set.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FamilyBinderSyntax {
    pub(crate) member: String,
    pub(crate) set: NamePath,
    pub(crate) range: TextRange,
}

impl FamilyBinderSyntax {
    /// Lexical name of the currently expanded member.
    #[must_use]
    pub fn member(&self) -> &str {
        &self.member
    }

    /// Nominal index set or complete-exterior support slot traversed by this binder.
    #[must_use]
    pub fn set(&self) -> &NamePath {
        &self.set
    }

    /// Full binder range, including its grouping delimiters.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }
}

/// One named compile-time component instance.
#[derive(Debug, Clone, PartialEq)]
pub struct InstanceDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) name: String,
    pub(crate) definition: NamePath,
    pub(crate) family: Option<FamilyBinderSyntax>,
    pub(crate) bindings: Vec<NamedBindingDecl>,
    pub(crate) range: TextRange,
}

impl InstanceDecl {
    /// Optional bounded index-family binder.
    #[must_use]
    pub fn family(&self) -> Option<&FamilyBinderSyntax> {
        self.family.as_ref()
    }
    /// Source occurrence name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Exact lexically resolved definition path.
    #[must_use]
    pub const fn definition(&self) -> &NamePath {
        &self.definition
    }
    /// Explicit named arguments; the target signature determines each kind.
    #[must_use]
    pub fn bindings(&self) -> &[NamedBindingDecl] {
        &self.bindings
    }
    /// Full occurrence range.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }
}

/// One category-free named occurrence binding.
#[derive(Debug, Clone, PartialEq)]
pub struct NamedBindingDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) name: String,
    pub(crate) value: Expr,
    pub(crate) range: TextRange,
}
impl NamedBindingDecl {
    /// Target signature name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Value or nominal reference before target-directed classification.
    #[must_use]
    pub const fn value(&self) -> &Expr {
        &self.value
    }
    /// Full binding range.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }
}

/// Named semantic Domain declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct DomainDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) name: String,
    pub(crate) syntax: DomainSyntax,
    pub(crate) range: TextRange,
}

impl DomainDecl {
    /// Source name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Domain family and its source-level contract.
    #[must_use]
    pub const fn syntax(&self) -> &DomainSyntax {
        &self.syntax
    }

    /// Full declaration range.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }
}

/// Source contract for one canonical Domain.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum DomainSyntax {
    /// Cartesian coordinate sources, one lower/upper pair per axis.
    CartesianBox(Vec<(CartesianCoordinateSyntax, CartesianCoordinateSyntax)>),
    /// One oriented side of a named parent Cartesian box.
    Boundary {
        /// Parent Domain name.
        parent: String,
        /// Zero-based coordinate axis.
        axis: usize,
        /// Lower or upper side.
        side: BoundarySideSyntax,
    },
    /// One nominal scalar conserving domain. The declaration identity, not
    /// dimension coincidence, determines Port compatibility.
    ScalarPhysical {
        /// Declared member exposing the across quantity.
        across_name: String,
        /// Complete scalar type of the across variable.
        across_type: ValueTypeSyntax,
        /// Declared member exposing the through quantity.
        through_name: String,
        /// Complete scalar type of the through variable.
        through_type: ValueTypeSyntax,
    },
}

/// Borrowed exact clock requirement; it does not declare another period or phase.
#[derive(Debug, Clone, PartialEq)]
pub struct ClockRequirementDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) name: String,
    pub(crate) range: TextRange,
}

impl ClockRequirementDecl {
    /// Signature name used by dependent unknown and Relation activation clauses.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Full signature-entry range.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }
}

/// Author-declared mathematical evolution role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldRoleSyntax {
    /// Algebraic unknown; spatial differentiability does not imply state ownership.
    Variable,
    /// Owned evolving state at the declared continuous or clocked activation.
    State,
}

/// A source unknown with independent role, support, and activation.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) name: String,
    pub(crate) domain: Option<String>,
    pub(crate) role: FieldRoleSyntax,
    pub(crate) activation: ActivationSyntax,
    pub(crate) value_type: ValueTypeSyntax,
    pub(crate) range: TextRange,
}

impl FieldDecl {
    /// Source name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Spatial Domain when this is a distributed Field.
    #[must_use]
    pub fn domain(&self) -> Option<&str> {
        self.domain.as_deref()
    }

    /// Author-declared evolution ownership.
    #[must_use]
    pub const fn role(&self) -> FieldRoleSyntax {
        self.role
    }

    /// Activation independent of support and scalar type.
    #[must_use]
    pub const fn activation(&self) -> &ActivationSyntax {
        &self.activation
    }

    /// Complete declared mathematical type.
    #[must_use]
    pub const fn value_type(&self) -> &ValueTypeSyntax {
        &self.value_type
    }

    /// Static SI dimension expression.
    #[must_use]
    pub fn dimension(&self) -> Option<&Expr> {
        self.value_type.dimension()
    }

    /// Full declaration range.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }
}

/// Port declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct PortDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) name: String,
    pub(crate) syntax: PortSyntax,
    pub(crate) range: TextRange,
}

impl PortDecl {
    /// Source name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Causal/conserving contract.
    #[must_use]
    pub const fn syntax(&self) -> &PortSyntax {
        &self.syntax
    }

    /// Full declaration range.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }
}

/// Source-level Port contract.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum PortSyntax {
    /// Causal signal with a complete mathematical value type.
    Signal {
        /// Causal direction.
        direction: SignalDirectionSyntax,
        /// Mathematical scalar domain, dimension and component roles.
        value_type: ValueTypeSyntax,
        /// Exact spatial support when distributed.
        domain: Option<String>,
        /// Continuous or exact declared periodic activation.
        activation: ActivationSyntax,
    },
    /// Executable scalar conserving Port, nominally typed by one Domain.
    /// In the flat source slice, a Relation that reads this Port with
    /// its declared quantity member is its owner: lowering emits both
    /// `DependsOn` and `HasPort`. Cross-component physical observation is not
    /// inferred from an accessor.
    ScalarPhysical {
        /// Owning scalar physical Domain name.
        domain: String,
    },
    /// Component interface typed by a nominal scalar physical Connector.
    ScalarPhysicalConnector {
        /// Connector declaration selected in lexical scope.
        connector: NamePath,
    },
    /// Field-valued conserving interface on one component spatial support.
    FieldPhysical {
        /// Nominal field-physical Connector selected in lexical scope.
        connector: NamePath,
        /// Boundary support slot (in a Component) or Domain (in a Model).
        support: String,
    },
}

/// Source-level signal direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SignalDirectionSyntax {
    /// Value enters the model.
    Input,
    /// Value leaves the model.
    Output,
}

/// Exact periodic clock declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct ClockDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) name: String,
    pub(crate) period: Expr,
    pub(crate) phase: Expr,
    pub(crate) range: TextRange,
}

impl ClockDecl {
    /// Source name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Authored period expression; exact time admission belongs to lowering.
    #[must_use]
    pub const fn period(&self) -> &Expr {
        &self.period
    }

    /// Authored phase expression, or the exact zero-second default.
    #[must_use]
    pub const fn phase(&self) -> &Expr {
        &self.phase
    }

    /// Full declaration range.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }
}

/// Connection declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct ConnectionDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) syntax: ConnectionSyntax,
    pub(crate) binder: Option<FamilyBinderSyntax>,
    pub(crate) ports: Vec<Expr>,
    pub(crate) range: TextRange,
}

impl ConnectionDecl {
    /// Signal or conserving syntax.
    #[must_use]
    pub const fn syntax(&self) -> ConnectionSyntax {
        self.syntax
    }

    /// Optional binder over one exact nominal index set.
    #[must_use]
    pub const fn binder(&self) -> Option<&FamilyBinderSyntax> {
        self.binder.as_ref()
    }

    /// Structurally segmented Port selections in source order.
    #[must_use]
    pub fn port_expressions(&self) -> &[Expr] {
        &self.ports
    }

    /// Full declaration range.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }
}

/// Source connection kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConnectionSyntax {
    /// One causal output followed by one or more inputs.
    Signal,
    /// Acausal connection net.
    Conserving,
    /// Exact spatial identification of two field-valued boundary Ports.
    SpatialPeriodic,
}

/// One boundary Connection containing field-valued Port references.
///
/// A binder denotes pointwise expansion over one complete exterior. Without
/// a binder, at least one Port reference carries an exact boundary selector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundaryConnectionDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) syntax: ConnectionSyntax,
    pub(crate) binder: Option<FamilyBinderSyntax>,
    pub(crate) ports: Vec<BoundaryPortReferenceSyntax>,
    pub(crate) range: TextRange,
}

impl BoundaryConnectionDecl {
    /// Conserving or spatial-periodic boundary semantics.
    #[must_use]
    pub const fn syntax(&self) -> ConnectionSyntax {
        self.syntax
    }

    /// Optional pointwise family binder.
    #[must_use]
    pub const fn binder(&self) -> Option<&FamilyBinderSyntax> {
        self.binder.as_ref()
    }

    /// Conserving Port references in source order.
    #[must_use]
    pub fn ports(&self) -> &[BoundaryPortReferenceSyntax] {
        &self.ports
    }

    /// Full Connection range.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }
}

/// One Port path with an optional exact boundary-member selector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundaryPortReferenceSyntax {
    pub(crate) port: NamePath,
    pub(crate) selector: Option<BoundaryPortSelectorSyntax>,
}

impl BoundaryPortReferenceSyntax {
    /// Structurally segmented Port path.
    #[must_use]
    pub const fn port(&self) -> &NamePath {
        &self.port
    }

    /// Exact boundary-member selector, when present.
    #[must_use]
    pub const fn selector(&self) -> Option<&BoundaryPortSelectorSyntax> {
        self.selector.as_ref()
    }
}

/// Closed `[member = target]` selector on one boundary-family Port.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BoundaryPortSelectorSyntax {
    pub(crate) member: String,
    pub(crate) target: String,
    pub(crate) range: TextRange,
}

impl BoundaryPortSelectorSyntax {
    /// Family binder name declared by the selected Port family.
    #[must_use]
    pub fn member(&self) -> &str {
        &self.member
    }

    /// Exact enclosing boundary Domain or active binder member.
    #[must_use]
    pub fn target(&self) -> &str {
        &self.target
    }

    /// Full `[member = target]` range.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }
}

mod expression;
pub use expression::{BinaryOp, CallArguments, Expr, ExprKind, ReductionOp, UnaryOp};
