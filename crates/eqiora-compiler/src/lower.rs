mod predicates;
pub(crate) use predicates::comparison_operator;
mod identities;
#[cfg(test)]
use identities::FreshLoweringIdentities;
pub(crate) use identities::LoweringIdentities;
use std::collections::{BTreeMap, BTreeSet, HashMap};
mod equation;
pub(crate) use equation::LoweringEquation;

use std::sync::Arc;

mod binding;
mod connection;
mod declaration;
mod diagnostics;
mod domain;
mod domain_contract;
use diagnostics::{normalize_zero, unresolved};
mod expression;
pub(crate) use expression::partial_result_type;
mod external;
mod integer;
mod native;
pub(crate) use integer::IntegerBuiltin;
mod dependencies;
mod structural;
#[cfg(test)]
mod tests;
mod value_expression;
use crate::units::lower_clock;
use binding::{
    Binding, DomainContract, FieldContract, PortContract, ResolvedPortContract, bind_domain,
    bind_port, insert_binding, resolve_field_contract, resolve_port_contract,
};
use connection::{lower_connection, prepare_flat_physical_connections};
use declaration::lower_port;
pub(crate) use domain_contract::{LoweringDomainContract, LoweringPortContract};
mod relation_body;
pub(crate) use relation_body::LoweringRelationBody;
pub(crate) mod constraints;

use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, DynQuantity, Id, OntologyId, RawId};
use eqiora_graph::{EdgeKind, Op, Transaction};
use eqiora_lang::{
    ActivationSyntax, BinaryOp, BoundarySideSyntax, ConnectionSyntax, DomainSyntax, Expr, ExprKind,
    Module, PortSyntax, SignalDirectionSyntax, TextRange, UnaryOp,
};
use eqiora_schema::kernel::pure_operator::PureOperatorDefinition;
use eqiora_schema::kernel::scalar_connection::{
    ScalarConnectionKind, ScalarConnectionViolation, ScalarPortContract, validate_scalar_connection,
};
use eqiora_schema::kernel::{
    ActivationDef, BoundaryPhysicalConnector, BoundarySide, ClockDomainDef, ConnectionDef,
    ConnectionSemantics, DomainDef, ExprDag, ExprDagBuilder, ExprId, FieldDef, KernelNode,
    ParameterDef, PortDef, RelationDef, RepresentationDef, SignalDirection, SymbolRef,
    UnaryMathFunction,
};
use eqiora_schema::{Model, ModelView};

use crate::connection_sets::{ConnectionFragment, ConnectionSetLimits, normalize_connection_sets};
use crate::diagnostics::source_error;
use crate::dimensions::{dimension_overflow, length_dimension, time_dimension};
use crate::formulation::CompiledAuthoredFormulation;
use crate::projection::PhysicalExposureProjectionMap;
use crate::provenance::ProvenanceMap;

/// Source-name to Semantic Kernel ID map produced with one compiled model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSymbols {
    symbols: BTreeMap<String, RawId>,
}

impl ModelSymbols {
    /// Resolve one source declaration name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<RawId> {
        self.symbols.get(name).copied()
    }

    /// Names and IDs in deterministic lexical order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&str, RawId)> {
        self.symbols.iter().map(|(name, id)| (name.as_str(), *id))
    }

    pub(crate) fn from_map(symbols: BTreeMap<String, RawId>) -> Self {
        Self { symbols }
    }
}

/// One typed model transaction ready for atomic Graph Federation commit.
#[derive(Debug)]
pub struct CompiledModel {
    notation: crate::ModelNotation,
    model: OntologyId<Model>,
    transaction: Transaction,
    symbols: ModelSymbols,
    provenance: Option<ProvenanceMap>,
    physical_exposures: PhysicalExposureProjectionMap,
    authored_formulations: Vec<CompiledAuthoredFormulation>,
}

impl CompiledModel {
    /// Declaration labels resolved once against the complete elaborated Model.
    #[must_use]
    pub const fn notation(&self) -> &crate::ModelNotation {
        &self.notation
    }
    /// Typed Standard Ontology ModelView identifier.
    #[must_use]
    pub const fn model(&self) -> OntologyId<Model> {
        self.model
    }

    /// Resolved declaration IDs.
    #[must_use]
    pub const fn symbols(&self) -> &ModelSymbols {
        &self.symbols
    }

    /// Borrow the typed transaction before commit.
    #[must_use]
    pub const fn transaction(&self) -> &Transaction {
        &self.transaction
    }

    /// Authored source provenance; native declarations have no source locations.
    #[must_use]
    pub const fn provenance(&self) -> Option<&ProvenanceMap> {
        self.provenance.as_ref()
    }

    /// Query projections for public physical Ports eliminated by hierarchy
    /// normalization.
    ///
    /// The catalog is a compiler sidecar. It adds no Kernel node, equation,
    /// alias, or source symbol to the canonical model.
    #[must_use]
    pub const fn physical_exposures(&self) -> &PhysicalExposureProjectionMap {
        &self.physical_exposures
    }

    /// Typed authored mathematics retained only by fresh source compilation.
    #[must_use]
    pub fn authored_formulations(
        &self,
    ) -> impl ExactSizeIterator<Item = &CompiledAuthoredFormulation> {
        self.authored_formulations.iter()
    }

    /// Consume into the transaction, model ID, and source symbol map.
    #[must_use]
    pub fn into_parts(self) -> (Transaction, OntologyId<Model>, ModelSymbols) {
        (self.transaction, self.model, self.symbols)
    }

    pub(crate) fn with_elaboration_metadata(
        mut self,
        symbols: ModelSymbols,
        provenance: ProvenanceMap,
        physical_exposures: PhysicalExposureProjectionMap,
        notation: crate::ModelNotation,
    ) -> Self {
        self.symbols = symbols;
        self.provenance = Some(provenance);
        self.physical_exposures = physical_exposures;
        self.notation = notation;
        self
    }

    pub(crate) fn with_authored_formulations(
        mut self,
        formulations: Vec<CompiledAuthoredFormulation>,
    ) -> Self {
        self.authored_formulations = formulations;
        self
    }
}

/// Type-lower one client-neutral native model draft.
///
/// Native declarations bypass parsing, but intentionally do not bypass name,
/// dimension, activation, expression-DAG, or transaction lowering. Synthetic
/// ranges are removed from diagnostics and replaced with stable declaration
/// paths; they never pretend to be source locations.
///
/// # Errors
/// Returns graph-path diagnostics for invalid native declarations. No partial
/// transaction is returned.
pub fn lower_module(
    module: &Module,
    entry: Option<&str>,
    bindings: &[(&str, crate::StaticBindingValue<'_>)],
) -> Result<CompiledModel, Vec<Diagnostic>> {
    native::lower(module, entry, bindings)
}

/// Compiler-owned declaration form consumed by Kernel lowering.
///
/// It is the sole entry shape for the typed transaction lowerer.
/// Parsed source and hierarchy elaboration both enter lowering through this
/// exact form. In particular, hierarchy-only boundary-physical contracts are
/// represented directly rather than disguised as source-language scalar
/// physical declarations.
#[derive(Debug, Clone)]
pub(crate) struct LoweringModel {
    pub(crate) structural_dependencies: BTreeMap<String, BTreeSet<String>>,
    pub(crate) name: String,
    pub(crate) range: TextRange,
    pub(crate) items: Vec<LoweringItem>,
}

/// Compiler-owned, typed scalar expression consumed by Kernel lowering.
///
/// Source expressions enter through [`Self::from_source`]. Hierarchy
/// elaboration may additionally substitute dimensioned constants and shared
/// Parameter-expression DAGs without fabricating source declarations.
#[derive(Debug, Clone)]
pub(crate) struct LoweringExpression {
    node: Arc<LoweringExpressionNode>,
    range: TextRange,
    structural_parameters: Option<Arc<BTreeSet<String>>>,
}

impl PartialEq for LoweringExpression {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node && self.structural_parameters == other.structural_parameters
    }
}

#[derive(Debug, PartialEq)]
enum LoweringExpressionNode {
    Partial {
        value: LoweringExpression,
        wrt: String,
    },
    Number(eqiora_lang::DecimalLiteral),
    Literal(eqiora_core::ValueLiteral),
    IntegerCall {
        operator: IntegerBuiltin,
        arguments: Vec<LoweringExpression>,
    },
    Name(String),
    Neg(LoweringExpression),
    Not(LoweringExpression),
    Array(Vec<LoweringExpression>),
    Index {
        value: LoweringExpression,
        index: u32,
    },
    Complex {
        real: LoweringExpression,
        imag: LoweringExpression,
    },
    Select {
        condition: LoweringExpression,
        then_value: LoweringExpression,
        else_value: LoweringExpression,
    },
    Case {
        value: LoweringExpression,
        arms: Vec<(eqiora_core::ValueLiteral, LoweringExpression)>,
    },
    Require {
        condition: LoweringExpression,
        value: LoweringExpression,
    },
    Extremum {
        minimum: bool,
        left: LoweringExpression,
        right: LoweringExpression,
    },
    Binary {
        operator: BinaryOp,
        left: LoweringExpression,
        right: LoweringExpression,
    },
    Call {
        callee: String,
        argument: LoweringExpression,
    },
    Sample {
        value: LoweringExpression,
        clock: String,
    },
    Piecewise {
        name: String,
        arguments: Vec<LoweringExpression>,
    },
    Property {
        release: Arc<eqiora_schema::kernel::PropertyRelease>,
        arguments: Vec<LoweringExpression>,
    },
    PureOperator {
        definition: PureOperatorDefinition,
        arguments: Vec<LoweringExpression>,
    },
    UnknownMath(String),
    InvalidValue(&'static str),
    Unsupported,
}

#[derive(Debug, Clone)]
pub(crate) enum LoweringItem {
    RecordInstance {
        id: Id<kinds::RecordInstance>,
        definition: Id<kinds::Record>,
        members: Vec<LoweringExpression>,
    },
    Nominal {
        name: String,
        definition: eqiora_schema::kernel::KernelNode,
    },
    Domain {
        name: String,
        contract: LoweringDomainContract,
        range: TextRange,
    },
    Representation {
        name: String,
        range: TextRange,
    },
    Field {
        name: String,
        domain: Option<String>,
        representation: Option<String>,
        value_type: eqiora_lang::ValueTypeSyntax,
        role: eqiora_lang::FieldRoleSyntax,
        activation: ActivationSyntax,
        range: TextRange,
    },
    Parameter {
        name: String,
        value: eqiora_core::ValueLiteral,
        range: TextRange,
    },
    Port {
        name: String,
        contract: LoweringPortContract,
        range: TextRange,
    },
    Clock {
        name: String,
        period: eqiora_lang::Expr,
        phase: eqiora_lang::Expr,
        range: TextRange,
    },
    Observable {
        name: String,
        value_type: eqiora_lang::ValueTypeSyntax,
        value: LoweringExpression,
        reduction: Option<String>,
        range: TextRange,
    },
    Event {
        name: String,
        guard: LoweringExpression,
        direction: eqiora_schema::kernel::EventDirection,
        range: TextRange,
    },
    Relation {
        name: String,
        activation: ActivationSyntax,
        domain: Option<String>,
        body: LoweringRelationBody,
        initial: bool,
        range: TextRange,
    },
    Connection {
        syntax: ConnectionSyntax,
        ports: Vec<String>,
        range: TextRange,
    },
    Boundary {
        ports: Vec<String>,
        range: TextRange,
    },
}

pub(crate) mod equality;
pub(crate) fn lower_typed_model(
    file: &str,
    model: &LoweringModel,
    identities: &mut impl LoweringIdentities,
) -> Result<CompiledModel, Vec<Diagnostic>> {
    let mut bindings = BTreeMap::new();
    let mut diagnostics = crate::math::model_name_diagnostics(file, &model.name, model.range);

    for item in &model.items {
        match item {
            LoweringItem::Domain {
                name,
                contract,
                range,
            } => {
                let contract = match contract {
                    LoweringDomainContract::Source(syntax) => bind_domain(file, *range, syntax),
                    LoweringDomainContract::ExternalGeometryRegion { dimensions, .. } => {
                        Ok(DomainContract::Spatial {
                            dimensions: Some(*dimensions),
                            parent: None,
                        })
                    }
                    LoweringDomainContract::ExternalGeometryBoundary { parent, .. } => {
                        Ok(DomainContract::Spatial {
                            dimensions: None,
                            parent: Some(parent.clone()),
                        })
                    }
                    LoweringDomainContract::BoundaryPhysical(contract) => {
                        Ok(DomainContract::BoundaryPhysical(contract.clone()))
                    }
                };
                match contract {
                    Ok(contract) => insert_binding(
                        file,
                        &mut bindings,
                        name,
                        Binding::Domain(identities.domain(name), contract),
                        *range,
                        &mut diagnostics,
                    ),
                    Err(diagnostic) => diagnostics.push(diagnostic),
                }
            }
            LoweringItem::Representation { name, range, .. } => insert_binding(
                file,
                &mut bindings,
                name,
                Binding::Representation(identities.representation(name)),
                *range,
                &mut diagnostics,
            ),
            LoweringItem::Field {
                name,
                domain,
                value_type,
                role,
                activation,
                range,
                ..
            } => match crate::value_types::component_dimension(file, value_type) {
                Ok(dimension) => insert_binding(
                    file,
                    &mut bindings,
                    name,
                    Binding::Field(
                        identities.field(name),
                        FieldContract {
                            dimension,
                            value_type: value_type.clone(),
                            domain: domain.clone(),
                            role: *role,
                            activation: activation.clone(),
                        },
                    ),
                    *range,
                    &mut diagnostics,
                ),
                Err(diagnostic) => diagnostics.push(diagnostic),
            },
            LoweringItem::Parameter {
                name, value, range, ..
            } => insert_binding(
                file,
                &mut bindings,
                name,
                Binding::Parameter(identities.parameter(name), value.value_type().clone()),
                *range,
                &mut diagnostics,
            ),
            LoweringItem::Port {
                name,
                contract,
                range,
            } => {
                let contract = match contract {
                    LoweringPortContract::Source(syntax) => bind_port(file, *range, syntax),
                    LoweringPortContract::BoundaryPhysical {
                        connector,
                        boundary,
                    } => Ok(PortContract::BoundaryPhysical {
                        connector: connector.clone(),
                        boundary: boundary.clone(),
                    }),
                };
                match contract {
                    Ok(contract) => insert_binding(
                        file,
                        &mut bindings,
                        name,
                        Binding::Port(identities.port(name), contract),
                        *range,
                        &mut diagnostics,
                    ),
                    Err(diagnostic) => diagnostics.push(diagnostic),
                }
            }
            LoweringItem::Clock {
                name,
                period,
                phase,
                range,
            } => match crate::units::lower_clock(file, period, phase) {
                Ok((period, _)) => insert_binding(
                    file,
                    &mut bindings,
                    name,
                    Binding::Clock(identities.clock(name), period),
                    *range,
                    &mut diagnostics,
                ),
                Err(error) => diagnostics.push(error),
            },
            LoweringItem::Observable { name, range, .. } => insert_binding(
                file,
                &mut bindings,
                name,
                Binding::Observable(identities.observable(name)),
                *range,
                &mut diagnostics,
            ),
            LoweringItem::Event { name, range, .. } => insert_binding(
                file,
                &mut bindings,
                name,
                Binding::Event(identities.activation(name)),
                *range,
                &mut diagnostics,
            ),
            LoweringItem::Relation { name, range, .. } => {
                let (relation, activation) = identities.relation(name);
                insert_binding(
                    file,
                    &mut bindings,
                    name,
                    Binding::Relation {
                        relation,
                        activation,
                    },
                    *range,
                    &mut diagnostics,
                );
            }
            LoweringItem::RecordInstance { .. }
            | LoweringItem::Nominal { .. }
            | LoweringItem::Connection { .. }
            | LoweringItem::Boundary { .. } => {}
        }
    }
    if !diagnostics.is_empty() {
        return Err(diagnostics);
    }

    let model_id = identities.model(&model.name);
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut boundary = BTreeSet::new();
    let mut connected_ports = BTreeSet::new();
    let physical_connections = prepare_flat_physical_connections(file, model, &bindings)?;

    for (item_index, item) in model.items.iter().enumerate() {
        let result = match item {
            LoweringItem::Domain {
                name,
                contract: lowering_contract,
                range,
            } => {
                let Binding::Domain(id, contract) = bindings[name].clone() else {
                    unreachable!("first pass assigns Domain bindings");
                };
                domain::lower_domain(file, *range, id, contract, lowering_contract, &bindings).map(
                    |(definition, parent, dependencies)| {
                        nodes.push(definition.into());
                        if let Some(parent) = parent {
                            edges.push((id.erase(), parent, EdgeKind::BoundaryOf));
                        }
                        for dependency in dependencies {
                            edges.push((id.erase(), dependency, EdgeKind::DependsOn));
                        }
                    },
                )
            }
            LoweringItem::Representation { name, .. } => {
                let Binding::Representation(id) = bindings[name].clone() else {
                    unreachable!("representation binding")
                };
                nodes.push(RepresentationDef::continuum(id).into());
                Ok(())
            }
            LoweringItem::Field {
                name,
                domain,
                representation,
                role,
                activation,
                range,
                ..
            } => {
                let Binding::Field(id, contract) = bindings[name].clone() else {
                    unreachable!("first pass assigns Field bindings");
                };
                resolve_field_contract(file, *range, &contract, &bindings).map(|value_type| FieldDef::new(id, value_type, match role {
                            eqiora_lang::FieldRoleSyntax::Variable => eqiora_schema::kernel::FieldRole::Variable,
                            eqiora_lang::FieldRoleSyntax::State => eqiora_schema::kernel::FieldRole::State,
                        }))
                    .and_then(|definition| {
                        nodes.push(definition.into());
                        if let ActivationSyntax::Named(clock) = activation {
                            let Some(Binding::Clock(clock, _)) = bindings.get(clock) else { return Err(unresolved(file, *range, clock, "Field ClockDomain")); };
                            edges.push((id.erase(), clock.erase(), EdgeKind::ClockedBy));
                        }
                        match (domain.as_deref(), representation.as_deref()) {
                            (None, None) => Ok(()),
                            (Some(domain), Some(representation)) => {
                                let Some(domain_binding) = bindings.get(domain) else {
                                    return Err(unresolved(file, *range, domain, "Field Domain"));
                                };
                                let domain = match domain_binding {
                                    Binding::Domain(id, DomainContract::Spatial { .. }) => *id,
                                    Binding::Domain(
                                        _,
                                        DomainContract::ScalarPhysical { .. },
                                    ) => {
                                        return Err(source_error(
                                            codes::LANGUAGE_TYPE_ERROR,
                                            file,
                                            *range,
                                            "spatial Field cannot be defined on a scalar physical Domain",
                                        ));
                                    }
                                    _ => {
                                        return Err(unresolved(
                                            file,
                                            *range,
                                            domain,
                                            "Field Domain",
                                        ));
                                    }
                                };
                                let Some(Binding::Representation(representation)) =
                                    bindings.get(representation).cloned()
                                else {
                                    return Err(unresolved(
                                        file,
                                        *range,
                                        representation,
                                        "Field Representation",
                                    ));
                                };
                                edges.push((id.erase(), domain.erase(), EdgeKind::DefinedOn));
                                edges.push((
                                    id.erase(),
                                    representation.erase(),
                                    EdgeKind::DefinedOn,
                                ));
                                Ok(())
                            }
                            _ => Err(source_error(
                                codes::LANGUAGE_TYPE_ERROR,
                                file,
                                *range,
                                "spatial Field requires both `on Domain` and `as Representation`",
                            )),
                        }
                    })
            }
            LoweringItem::RecordInstance {
                id,
                definition,
                members,
            } => expression::lower_record(file, members, &bindings).and_then(|expression| {
                nodes.push(
                    eqiora_schema::kernel::RecordInstanceDef::new(*id, *definition, expression)?
                        .into(),
                );
                Ok(())
            }),
            LoweringItem::Nominal { definition, .. } => {
                nodes.push(definition.clone());
                Ok(())
            }
            LoweringItem::Parameter { name, value, .. } => {
                let Binding::Parameter(id, _) = bindings[name].clone() else {
                    unreachable!("first pass assigns Parameter bindings");
                };
                {
                    nodes.push(ParameterDef::new(id, value.clone()).into());
                    Ok(())
                }
            }
            LoweringItem::Port { name, range, .. } => {
                let Binding::Port(id, contract) = bindings[name].clone() else {
                    unreachable!("first pass assigns Port bindings");
                };
                lower_port(file, *range, id, &contract, &bindings).and_then(|port| {
                    if let ResolvedPortContract::Signal { support, clock, .. } =
                        resolve_port_contract(file, *range, &contract, &bindings)?
                    {
                        if let Some(support) = support {
                            edges.push((id.erase(), *support.domain(), EdgeKind::DefinedOn));
                        }
                        if let Some(clock) = clock {
                            edges.push((id.erase(), clock.erase(), EdgeKind::ClockedBy));
                        }
                    }
                    nodes.push(port.into());
                    Ok(())
                })
            }
            LoweringItem::Clock {
                name,
                period,
                phase,
                range,
            } => {
                let Binding::Clock(id, _) = bindings[name].clone() else {
                    unreachable!("first pass assigns Clock bindings");
                };
                lower_clock(file, period, phase)
                    .and_then(|(period, phase)| {
                        ClockDomainDef::periodic(id, period, phase).map_err(|diagnostic| {
                            source_error(
                                codes::LANGUAGE_TYPE_ERROR,
                                file,
                                *range,
                                diagnostic.message(),
                            )
                        })
                    })
                    .map(|definition| nodes.push(definition.into()))
            }
            LoweringItem::Observable {
                name,
                value_type,
                value,
                reduction,
                range,
            } => {
                let Binding::Observable(id) = bindings[name] else {
                    unreachable!("Observable binding")
                };
                expression::lower_observable(
                    file,
                    *range,
                    id,
                    value_type,
                    value,
                    reduction.as_ref(),
                    &bindings,
                )
                .map(|(definition, dependencies)| {
                    for dependency in dependencies {
                        edges.push((id.erase(), dependency, EdgeKind::DependsOn));
                    }
                    if let Some(domain) = definition.reduction().domain() {
                        edges.push((id.erase(), domain.erase(), EdgeKind::AppliesOn));
                    }
                    nodes.push(definition.into());
                })
            }
            LoweringItem::Event {
                name,
                guard,
                direction,
                ..
            } => {
                let Binding::Event(id) = bindings[name] else {
                    unreachable!("event binding")
                };
                expression::lower_event_guard(file, guard, &bindings).and_then(|lowered| {
                    nodes.push(
                        ActivationDef::new(
                            id,
                            eqiora_schema::kernel::ActivationKind::Event {
                                guard: lowered.expression,
                                direction: *direction,
                            },
                        )?
                        .into(),
                    );
                    Ok(())
                })
            }
            LoweringItem::Relation {
                name,
                activation,
                domain,
                body,
                initial,
                range,
            } => body
                .lower(
                    file,
                    *range,
                    activation,
                    domain.as_deref(),
                    *initial,
                    &bindings,
                )
                .and_then(|(lowered, conservation)| {
                    let Binding::Relation {
                        relation,
                        activation: activation_id,
                    } = bindings[name].clone()
                    else {
                        unreachable!("first pass assigns Relation bindings");
                    };
                    nodes.push(
                        if let Some(terms) = conservation {
                            RelationDef::conservation(relation, lowered.expression, terms)
                        } else if *initial {
                            RelationDef::initial(relation, lowered.expression)
                        } else {
                            let LoweringRelationBody::Equations(conditions) = body else {
                                unreachable!("Law lowering retains conservation terms");
                            };
                            RelationDef::with_conditions(
                                relation,
                                lowered.expression,
                                conditions.iter().map(|condition| condition.kind).collect(),
                            )
                        }?
                        .into(),
                    );
                    let activation_definition = match activation {
                        ActivationSyntax::Continuous => {
                            Some(ActivationDef::continuous(activation_id))
                        }
                        ActivationSyntax::Named(name) => match bindings.get(name) {
                            Some(Binding::Event(_)) => None,
                            Some(Binding::Clock(_, _)) => {
                                Some(ActivationDef::periodic(activation_id))
                            }
                            _ => unreachable!("named activation was resolved"),
                        },
                        _ => unreachable!("unsupported Activation was diagnosed"),
                    };
                    if !initial && let Some(definition) = activation_definition {
                        nodes.push(definition.into());
                    }
                    for dependency in lowered.dependencies {
                        edges.push((relation.erase(), dependency, EdgeKind::DependsOn));
                    }
                    structural::connect_relation(
                        file,
                        *range,
                        relation.erase(),
                        body,
                        &bindings,
                        &mut edges,
                    )?;
                    for port in lowered.ports {
                        edges.push((relation.erase(), port, EdgeKind::HasPort));
                    }
                    if !initial {
                        edges.push((activation_id.erase(), relation.erase(), EdgeKind::Activates));
                    }
                    if let Some(domain_name) = domain {
                        let Binding::Domain(domain, _) = bindings[domain_name].clone() else {
                            unreachable!("Relation Domain was resolved while lowering");
                        };
                        edges.push((relation.erase(), domain.erase(), EdgeKind::AppliesOn));
                    }
                    if let ActivationSyntax::Named(clock_name) = activation
                        && let Some(Binding::Clock(clock, _)) = bindings.get(clock_name)
                    {
                        edges.push((activation_id.erase(), clock.erase(), EdgeKind::ClockedBy));
                    }
                    Ok(())
                }),
            LoweringItem::Connection {
                syntax,
                ports,
                range,
            } => {
                if let Some(ports) = physical_connections.emissions.get(&item_index) {
                    let definition = ConnectionDef::new(
                        identities.connection(),
                        ConnectionSemantics::Conserving,
                    );
                    let connection = definition.id().erase();
                    nodes.push(definition.into());
                    for port in ports {
                        edges.push((connection, *port, EdgeKind::Connects));
                    }
                    Ok(())
                } else if physical_connections.consumed.contains(&item_index) {
                    Ok(())
                } else {
                    let connection = identities.connection();
                    lower_connection(
                        file,
                        *range,
                        *syntax,
                        ports,
                        connection,
                        &bindings,
                        &mut connected_ports,
                    )
                    .map(|(definition, ports)| {
                        let connection = definition.id().erase();
                        nodes.push(definition.into());
                        for port in ports {
                            edges.push((connection, port, EdgeKind::Connects));
                        }
                    })
                }
            }
            LoweringItem::Boundary { ports, range } => {
                for name in ports {
                    match bindings.get(name).cloned() {
                        Some(Binding::Port(id, _)) => {
                            boundary.insert(id.erase());
                        }
                        Some(_) => diagnostics.push(source_error(
                            codes::LANGUAGE_TYPE_ERROR,
                            file,
                            *range,
                            format!("boundary name `{name}` is not a Port"),
                        )),
                        None => diagnostics.push(unresolved(file, *range, name, "boundary Port")),
                    }
                }
                Ok(())
            }
        };
        if let Err(diagnostic) = result {
            diagnostics.push(diagnostic);
        }
    }
    if !diagnostics.is_empty() {
        return Err(diagnostics);
    }

    structural::connect_declarations(file, model, &bindings, &mut edges)?;
    let members = nodes.iter().map(KernelNode::id).collect::<BTreeSet<_>>();
    let view = ModelView::new(model_id, members, boundary).map_err(|diagnostic| {
        vec![source_error(
            codes::LANGUAGE_LOWERING_ERROR,
            file,
            model.range,
            diagnostic.message(),
        )]
    })?;
    let mut transaction = Transaction::new(format!("compile model {}", model.name));
    for node in nodes {
        transaction.push(Op::DefineKernelNode { node });
    }
    for (from, to, edge) in edges {
        transaction.push(Op::Connect { from, to, edge });
    }
    transaction.push(Op::DefineOntologyView { view: view.into() });

    // Lowering bindings also own synthesized continuum representations and
    // unnamed initial Relations. Export only authored declaration names; the
    // synthesized nodes remain members of the unchanged Kernel transaction.
    let symbols = model
        .items
        .iter()
        .filter_map(|item| match item {
            LoweringItem::Domain { name, .. }
            | LoweringItem::Field { name, .. }
            | LoweringItem::Parameter { name, .. }
            | LoweringItem::Port { name, .. }
            | LoweringItem::Clock { name, .. }
            | LoweringItem::Observable { name, .. }
            | LoweringItem::Event { name, .. }
            | LoweringItem::Relation {
                name,
                initial: false,
                ..
            } => Some(name),
            LoweringItem::RecordInstance { .. }
            | LoweringItem::Nominal { .. }
            | LoweringItem::Representation { .. }
            | LoweringItem::Relation { initial: true, .. }
            | LoweringItem::Connection { .. }
            | LoweringItem::Boundary { .. } => None,
        })
        .map(|name| (name.clone(), bindings[name].primary_id()))
        .collect();
    Ok(CompiledModel {
        notation: crate::ModelNotation::default(),
        model: model_id,
        transaction,
        symbols: ModelSymbols::from_map(symbols),
        provenance: None,
        physical_exposures: PhysicalExposureProjectionMap::default(),
        authored_formulations: Vec::new(),
    })
}
