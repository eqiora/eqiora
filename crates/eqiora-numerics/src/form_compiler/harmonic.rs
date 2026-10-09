//! Fixed-domain LTI response restriction, retaining its original Model.
use std::collections::{BTreeMap, BTreeSet};

use eqiora_artifact::ModelEnvelope;
use eqiora_compiler::AuthoredFormulationProjection;
use eqiora_core::{Diagnostic, Id, OntologyId, RawId, ScalarDomain, ValueType, entity::kinds};
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_schema::{ModelView, kernel::*};
use eqiora_sem::KernelProgram;
use sha2::{Digest, Sha256};

mod coefficient;
mod compact;
mod expression;
mod reconstruction;
#[cfg(test)]
pub(crate) mod tests;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct HarmonicReduction {
    pub(crate) original: ModelEnvelope,
    pub(crate) reduced: ModelEnvelope,
    pub(crate) angular_frequency: f64,
    pub(crate) amplitudes: Vec<(String, Id<kinds::Field>, Id<kinds::Field>)>,
}

impl HarmonicReduction {
    pub(crate) fn derive(
        program: &KernelProgram,
        form: &AuthoredFormulationProjection,
        geometry: Option<&eqiora_geometry::CanonicalGeometryV1>,
    ) -> Result<Self, Diagnostic> {
        let angular_frequency = form
            .harmonic_angular_frequency()
            .ok_or_else(|| invalid("missing harmonic request"))?;
        let original = ModelEnvelope::from_program(program)?;
        let mut seed = Sha256::new();
        seed.update(b"eqiora.harmonic-reduction/v1\0");
        seed.update(original.digest()?.to_string().as_bytes());
        seed.update(form.canonical_bytes());
        let seed = seed.finalize();
        let fresh = |tag: &[u8], old: &[u8]| {
            let mut hash = Sha256::new();
            hash.update(seed);
            hash.update(tag);
            hash.update(old);
            let digest = hash.finalize();
            ulid::Ulid::from(u128::from_be_bytes(digest[..16].try_into().unwrap()))
        };
        let frequency = coefficient::compile(program, angular_frequency, None)?;
        let frequency_type = ValueType::scalar(
            ScalarDomain::Real,
            eqiora_core::DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap(),
        )
        .unwrap();
        if frequency.value_type != frequency_type {
            return Err(invalid(
                "angular frequency requires a real scalar with dimension 1/time",
            ));
        }
        let values = eqiora_ir::ScalarOperatorIr::lower(&frequency.expression)?
            .evaluate_typed(frequency.expression.roots(), &mut |_| None)?;
        let omega = values[0]
            .real_scalar_value()
            .ok_or_else(|| invalid("angular frequency is not scalar"))?
            .value();
        if !omega.is_finite() || omega <= 0.0 {
            return Err(invalid(
                "harmonic response requires positive finite angular frequency; DC needs a separate decomposition",
            ));
        }
        let mut fields = BTreeMap::new();
        let mut amplitudes = Vec::new();
        for (name, old) in form.harmonic_amplitudes().expect("harmonic request") {
            let old = Id::<kinds::Field>::from_ulid(id(old)?);
            let Some(KernelNode::Field(field)) = program.node(old.erase()) else {
                return Err(invalid(
                    "harmonic mapping does not name a live original Field",
                ));
            };
            if field.value_type().scalar_domain() != ScalarDomain::Real {
                return Err(invalid("harmonic original unknowns must be real"));
            }
            let new = Id::from_ulid(fresh(b"field", &old.ulid().to_bytes()));
            fields.insert(old.erase(), new);
            amplitudes.push((name.clone(), old, new));
        }
        let selected = form
            .harmonic_relations()
            .expect("harmonic request")
            .iter()
            .map(|value| id(value).map(|id| Id::<kinds::Relation>::from_ulid(id).erase()))
            .collect::<Result<BTreeSet<_>, _>>()?;
        let mut inputs = BTreeMap::new();
        for (old, value) in form.harmonic_excitations().expect("harmonic request") {
            let old = Id::<kinds::Port>::from_ulid(id(old)?);
            let Some(KernelNode::Port(port)) = program.node(old.erase()) else {
                return Err(invalid(
                    "harmonic excitation does not name a live original input Port",
                ));
            };
            let Some((SignalDirection::Input, original_type)) = port.signal_contract() else {
                return Err(invalid("harmonic excitation requires a signal input"));
            };
            let support = program
                .edges()
                .iter()
                .find(|edge| edge.from() == old.erase() && edge.kind() == EdgeKind::DefinedOn)
                .and_then(|edge| edge.to().downcast())
                .and_then(|domain| program.spatial_support(domain))
                .cloned();
            let coefficient = coefficient::compile(program, value, support)?;
            if original_type.scalar_domain() != ScalarDomain::Real
                || coefficient.value_type != complex(original_type)?
            {
                return Err(invalid(
                    "harmonic excitation differs from the original real input type",
                ));
            }
            inputs.insert(old.erase(), coefficient.expression);
        }
        let mut transaction = Transaction::new("derive fixed-domain harmonic response");
        let mut dependency_edges = Vec::new();
        let mut mapping = BTreeMap::new();
        let mut members = BTreeSet::new();
        for node in program.nodes() {
            let old = node.id();
            let retained = match node {
                KernelNode::Field(field) => {
                    let new = *fields
                        .get(&old)
                        .ok_or_else(|| invalid("unmapped harmonic unknown"))?;
                    KernelNode::Field(FieldDef::new(
                        new,
                        complex(field.value_type())?,
                        FieldRole::Variable,
                    ))
                }
                KernelNode::Relation(relation) if relation.is_initial() => continue,
                KernelNode::Relation(relation) => {
                    if !selected.contains(&old) {
                        return Err(invalid("harmonic request omits an original Relation"));
                    }
                    if relation.conditions().is_some_and(|conditions| {
                        conditions
                            .iter()
                            .any(|condition| *condition != RelationConditionKind::Equality)
                    }) {
                        return Err(invalid(
                            "harmonic reduction does not admit ordered or complementary conditions",
                        ));
                    }
                    let typed = program
                        .typed_relation_residual(relation.id())
                        .map_err(first)?;
                    let expression = expression::reduce(&typed, &fields, &inputs, omega)?;
                    KernelNode::Relation(RelationDef::new(
                        Id::from_ulid(fresh(b"relation", &relation.id().ulid().to_bytes())),
                        expression,
                    )?)
                }
                KernelNode::Port(port) => {
                    if !matches!(port.signal_contract(), Some((SignalDirection::Input, _)))
                        || !inputs.contains_key(&old)
                    {
                        return Err(invalid(
                            "harmonic reduction requires an explicit excitation for every input and does not admit connected physical or output Ports",
                        ));
                    }
                    continue;
                }
                KernelNode::ClockDomain(clock) if clock.kind() == ClockKind::Continuous => {
                    node.clone()
                }
                KernelNode::Activation(activation)
                    if matches!(activation.kind(), ActivationKind::Continuous) =>
                {
                    node.clone()
                }
                // Passive Observables remain on the original Model, never become amplitude outputs.
                KernelNode::Observable(_) => continue,
                KernelNode::Domain(_)
                | KernelNode::Representation(_)
                | KernelNode::Parameter(_)
                | KernelNode::FiniteSpace(_) => node.clone(),
                _ => {
                    return Err(invalid(
                        "harmonic reduction requires fixed-domain continuous mathematics without events, connections or discrete state",
                    ));
                }
            };
            mapping.insert(old, retained.id());
            members.insert(retained.id());
            if let KernelNode::Relation(relation) = &retained {
                let dependencies = relation
                    .expression()
                    .nodes()
                    .iter()
                    .filter_map(|node| match node {
                        ExprNode::Symbol(SymbolRef::Field(field)) => Some(field.erase()),
                        ExprNode::Symbol(SymbolRef::Parameter(parameter)) => {
                            Some(parameter.erase())
                        }
                        ExprNode::Symbol(SymbolRef::Coordinate { support, .. }) => {
                            Some(support.erase())
                        }
                        _ => None,
                    })
                    .collect::<BTreeSet<_>>();
                for to in dependencies {
                    dependency_edges.push(Op::Connect {
                        from: relation.id().erase(),
                        to,
                        edge: EdgeKind::DependsOn,
                    });
                }
            }
            transaction.push(Op::DefineKernelNode { node: retained });
            if let Some(value) = program.typed_value(old) {
                transaction.push(Op::SetValue {
                    target: old,
                    value: value.clone(),
                });
            }
        }
        if selected.len()
            != program
                .nodes()
                .filter(
                    |node| matches!(node, KernelNode::Relation(relation) if !relation.is_initial()),
                )
                .count()
        {
            return Err(invalid(
                "harmonic request contains a foreign or initial Relation",
            ));
        }
        for edge in dependency_edges {
            transaction.push(edge);
        }
        for edge in program.edges() {
            if edge.kind() == EdgeKind::DependsOn
                && matches!(program.node(edge.from()), Some(KernelNode::Relation(_)))
            {
                continue;
            }
            if let (Some(from), Some(to)) = (mapping.get(&edge.from()), mapping.get(&edge.to())) {
                transaction.push(Op::Connect {
                    from: *from,
                    to: *to,
                    edge: edge.kind(),
                });
            }
        }
        let model = OntologyId::from_ulid(fresh(b"model", &program.model().ulid().to_bytes()));
        transaction.push(Op::DefineOntologyView {
            view: ModelView::new(model, members, [])
                .map_err(|error| invalid(error.to_string()))?
                .into(),
        });
        let store =
            InMemoryGraphStore::restore_snapshot(transaction, program.revision()).map_err(first)?;
        let reduced = match geometry {
            Some(geometry) => {
                KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[geometry])
            }
            None => KernelProgram::from_snapshot(&store.snapshot(), model),
        }
        .map_err(first)?;
        Ok(Self {
            original,
            reduced: ModelEnvelope::from_program(&reduced)?,
            angular_frequency: omega,
            amplitudes,
        })
    }
}

fn complex(value: &ValueType) -> Result<ValueType, Diagnostic> {
    value
        .clone()
        .with_common_scalar_domain(
            &ValueType::scalar(ScalarDomain::Complex, value.dimension()).unwrap(),
        )
        .ok_or_else(|| invalid("harmonic amplitude cannot complexify this original type"))
}
fn id(value: &str) -> Result<ulid::Ulid, Diagnostic> {
    value
        .parse()
        .map_err(|_| invalid("invalid harmonic source identity"))
}
fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(
        eqiora_core::diagnostic::codes::INVALID_DISCRETIZATION,
        message,
    )
}
fn first(errors: Vec<Diagnostic>) -> Diagnostic {
    errors
        .into_iter()
        .next()
        .unwrap_or_else(|| invalid("harmonic Model admission failed"))
}
