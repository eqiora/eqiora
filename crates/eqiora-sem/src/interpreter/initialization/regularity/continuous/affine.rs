//! A shared sufficient regularity proof for first-order linear numeric ODEs.
use super::*;
use eqiora_ir::{ComponentScalarization, ScalarSymbolCoordinate};

pub(super) fn regular_ode(
    program: &KernelProgram,
    fields: &BTreeMap<RawId, std::num::NonZeroU32>,
    relations: &BTreeSet<RawId>,
) -> Result<bool, Diagnostic> {
    if fields.is_empty() || fields.values().any(|order| order.get() != 1) {
        return Ok(false);
    }
    let mut values = Vec::new();
    let mut rates = Vec::new();
    for (&field, &order) in fields {
        let variable = Variable::Field(field);
        values.extend(ScalarSymbolCoordinate::for_value(
            variable.symbol(),
            &variable.value_type(program)?,
        )?);
        let variable = Variable::Derivative(field, order);
        rates.extend(ScalarSymbolCoordinate::for_value(
            variable.symbol(),
            &variable.value_type(program)?,
        )?);
    }
    let n = values.len();
    let time = program
        .execution_symbol_type(SymbolRef::Time)
        .ok_or_else(|| execution_error("physical Time type is unavailable", 0.))?;
    let selected = values
        .into_iter()
        .chain(rates)
        .chain(ScalarSymbolCoordinate::for_value(SymbolRef::Time, &time)?)
        .collect::<Vec<_>>();
    let mut mass = Vec::new();
    let mut rows = 0usize;
    for &owner in relations {
        let Ok(typed) = program.typed_relation_residual(owner.downcast().expect("Relation")) else {
            return Ok(false);
        };
        let Ok(operator) = ComponentScalarization::lower(&typed) else {
            return Ok(false);
        };
        for row in operator.rows() {
            let mut bindings = Vec::new();
            for source in row.symbols() {
                if selected.contains(source) {
                    continue;
                }
                // Only Model Parameters may supply frozen coefficients. Binding a
                // state or Time here would mistake pointwise rank for constant mass.
                let SymbolRef::Parameter(id) = source.symbol() else {
                    return Ok(false);
                };
                let Some(value) = program
                    .typed_value(id.erase())
                    .and_then(|value| components::value_component(source, value))
                else {
                    return Ok(false);
                };
                bindings.push((source.clone(), value));
            }
            let Ok(proof) = row.bind_affine(&selected, &bindings) else {
                return Ok(false);
            };
            mass.extend_from_slice(&proof.coefficients()[n..2 * n]);
            rows += 1;
        }
    }
    if rows != n {
        return Ok(false);
    }
    Ok(ConstantDerivativeMatrixProof::new(n, mass)?.exact_rank() == n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{
        DimExponents, Id, OntologyId, ScalarDomain, ValueLiteral, ValueType, entity::kinds,
    };
    use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
    use eqiora_schema::{
        Model, ModelView,
        kernel::{
            ActivationDef, ExprDagBuilder, FieldDef, FieldRole, RelationDef, UnaryMathFunction,
        },
    };

    #[test]
    fn constant_mass_proof_retains_every_complex_channel_and_rejects_pointwise_shortcuts() {
        for domain in [ScalarDomain::Real, ScalarDomain::Complex] {
            for profile in ["constant", "time-dependent", "state-dependent", "singular"] {
                let z = Id::<kinds::Field>::new();
                let relation = Id::<kinds::Relation>::new();
                let activation = Id::<kinds::Activation>::new();
                let model = OntologyId::<Model>::new();
                let state_type = ValueType::scalar(domain, DimExponents::DIMENSIONLESS)
                    .unwrap()
                    .array(6)
                    .unwrap();
                let mut b = ExprDagBuilder::new();
                let value = b.symbol(SymbolRef::Field(z)).unwrap();
                let rate = b
                    .symbol(SymbolRef::Derivative(z, std::num::NonZeroU32::MIN))
                    .unwrap();
                let frequency = b
                    .constant(DynQuantity::new(
                        1.,
                        DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap(),
                    ))
                    .unwrap();
                let time = b.symbol(SymbolRef::Time).unwrap();
                let dimensionless_time = b.mul(time, frequency).unwrap();
                let mut roots = Vec::new();
                for channel in 0..6 {
                    let value = b.index(value, channel).unwrap();
                    let rate = b.index(rate, channel).unwrap();
                    let left = match profile {
                        "time-dependent" => b.mul(dimensionless_time, rate).unwrap(),
                        "state-dependent" => b.mul(value, rate).unwrap(),
                        "singular" if domain == ScalarDomain::Complex => {
                            let conjugate = b.unary_math(UnaryMathFunction::Conj, rate).unwrap();
                            b.add(rate, conjugate).unwrap()
                        }
                        "singular" => b.sub(rate, rate).unwrap(),
                        _ => rate,
                    };
                    let right = b.mul(frequency, value).unwrap();
                    let right = if domain == ScalarDomain::Complex {
                        let i = b
                            .constant(
                                ValueLiteral::new(
                                    ValueType::scalar(domain, DimExponents::DIMENSIONLESS).unwrap(),
                                    [(0., 1.)],
                                )
                                .unwrap(),
                            )
                            .unwrap();
                        b.mul(i, right).unwrap()
                    } else {
                        right
                    };
                    roots.extend([left, right]);
                }
                let mut tx = Transaction::new("constant mass component proof");
                for node in [
                    FieldDef::new(z, state_type, FieldRole::State).into(),
                    RelationDef::new(relation, b.finish(roots).unwrap())
                        .unwrap()
                        .into(),
                    ActivationDef::continuous(activation).into(),
                ] {
                    tx.push(Op::DefineKernelNode { node });
                }
                tx.push(Op::Connect {
                    from: relation.erase(),
                    to: z.erase(),
                    edge: EdgeKind::DependsOn,
                });
                tx.push(Op::Connect {
                    from: activation.erase(),
                    to: relation.erase(),
                    edge: EdgeKind::Activates,
                });
                tx.push(Op::DefineOntologyView {
                    view: ModelView::new(
                        model,
                        [z.erase(), relation.erase(), activation.erase()],
                        [],
                    )
                    .unwrap()
                    .into(),
                });
                let mut store = InMemoryGraphStore::new();
                store.commit(tx).unwrap();
                let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
                let regular = regular_ode(
                    &program,
                    &BTreeMap::from([(z.erase(), std::num::NonZeroU32::MIN)]),
                    &BTreeSet::from([relation.erase()]),
                )
                .unwrap();
                // Mass is I for constant real channels and 2x2 I per complex
                // channel. Conjugation instead makes diag(2,0), hence singular.
                assert_eq!(regular, profile == "constant", "{domain:?}: {profile}");
            }
        }
    }
}
