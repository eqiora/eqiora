//! **eqiora-lang** — the specification-first Eqiora Language frontend.
//!
//! The lexer retains every byte range, including whitespace and comments.
//! The parser produces a source-oriented AST and structured diagnostics; it
//! never constructs Semantic Kernel nodes directly. Typed transaction
//! lowering belongs to the compiler layer so syntax remains independent of
//! graph storage.

mod ast;
mod ast_property;
mod cartesian;
mod decimal;
mod draft;
mod draft_spatial;
mod factory;
mod formatter;
mod lexer;
mod notation;
mod notation_labels;
mod parser;

pub use ast::{
    ActivationSyntax, BinaryOp, BoundaryConnectionDecl, BoundaryPairingSyntax,
    BoundaryPortReferenceSyntax, BoundaryPortSelectorSyntax, BoundarySideSyntax, CallArguments,
    CaseArm, ClockDecl, ClockRequirementDecl, ComponentDecl, ComponentItem, ComponentParameterDecl,
    ComponentPortDecl, ComponentPortFamilyDecl, ConnectionDecl, ConnectionSyntax, ConnectorDecl,
    ConnectorQuantitySyntax, ConnectorSyntax, ConservationSyntax, DocComment, Document, DomainDecl,
    DomainSyntax, EnumDecl, EventDecl, ExactIntegerSyntax, Expr, ExprKind, FamilyBinderSyntax,
    FieldDecl, FieldRoleSyntax, FrameSyntax, InitialDecl, InstanceDecl, Item, ModelDecl, NamePath,
    NamedBindingDecl, NamedDefinitionDecl, ObservableDecl, ParameterDecl, PortDecl, PortSyntax,
    PureOperatorDecl, PureOperatorFormal, PureValueClassSyntax, RecordDecl, RecordMemberDecl,
    ReductionOp, RelationBody, RelationCondition, RelationDecl, RelationFamilyDecl,
    SignalDirectionSyntax, SignatureItem, SupportSlotDecl, SupportSlotSyntax, TextRange, UnaryOp,
    ValueShapeSyntax, ValueTypeSyntax, ValueTypeSyntaxKind, VisibilitySyntax,
};
pub use cartesian::CartesianCoordinateSyntax;
pub use decimal::DecimalLiteral;
pub use draft::{
    DraftConservingConnection, DraftConservingPort, DraftDeclaration, DraftExpression, DraftField,
    DraftObservable, DraftParameter, DraftPhysicalDomain, DraftRelation, Module,
};
pub use draft_spatial::DraftSpatialDomain;
pub use eqiora_schema::kernel::RelationConditionKind;
pub use factory::{AstConstructionError, SourceAstFactory};
pub use formatter::format;
pub use lexer::{Token, TokenKind, lex};
pub use notation::{
    Notation, NotationAccent, NotationAtom, NotationError, NotationMark, NotationNode,
    NotationStyle,
};
pub use notation_labels::{NotationLabel, NotationProfile};
pub use parser::{ParseResult, parse};

pub use ast_property::{ComponentPropertyDecl, PropertySourceSyntax, PropertyTableSyntax};
