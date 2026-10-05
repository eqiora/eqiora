//! Exact typed numerical values grouped by their original semantic unknown.
use super::*;
use eqiora_core::{ScalarDomain, ValueLiteral, ValueType};

impl Variable {
    pub(super) fn symbol(self) -> SymbolRef {
        match self {
            Self::Field(id) => SymbolRef::Field(id.downcast().expect("Field variable")),
            Self::Derivative(id, order) => {
                SymbolRef::Derivative(id.downcast().expect("Field derivative"), order)
            }
            Self::NextField(id) => SymbolRef::Next(id.downcast().expect("next Field")),
            Self::Port(id) => SymbolRef::Port(id.downcast().expect("Port variable")),
            Self::Physical(PhysicalUnknown::Across(id)) => SymbolRef::Across(id),
            Self::Physical(PhysicalUnknown::Through(id)) => SymbolRef::Through(id),
        }
    }

    pub(super) fn value_type(self, program: &KernelProgram) -> Result<ValueType, Diagnostic> {
        program
            .execution_symbol_type(self.symbol())
            .ok_or_else(|| execution_error("numerical unknown has no exact type", 0.))
    }
}

pub(super) fn numeric_width(value_type: &ValueType) -> Result<usize, Diagnostic> {
    let parts = match value_type.scalar_domain() {
        ScalarDomain::Real => 1,
        ScalarDomain::Complex => 2,
        _ => {
            return Err(execution_error(
                "discrete unknown cannot enter a numerical solve",
                0.,
            ));
        }
    };
    value_type
        .shape()
        .component_count()
        .and_then(|count| count.checked_mul(parts))
        .ok_or_else(|| execution_error("numerical coordinate cardinality overflows", 0.))
}

pub(super) struct CandidateMaps {
    pub(super) fields: BTreeMap<RawId, f64>,
    pub(super) typed_fields: BTreeMap<RawId, ValueLiteral>,
    pub(super) derivatives: BTreeMap<(RawId, std::num::NonZeroU32), ValueLiteral>,
    pub(super) next_fields: BTreeMap<RawId, f64>,
    pub(super) typed_next: BTreeMap<RawId, ValueLiteral>,
    pub(super) ports: BTreeMap<RawId, f64>,
    pub(super) typed_ports: BTreeMap<RawId, ValueLiteral>,
    pub(super) physical: BTreeMap<PhysicalUnknown, f64>,
}

impl CandidateMaps {
    pub(super) fn initial_context_state(&self, state: &RuntimeState) -> RuntimeState {
        let mut initial = state.clone();
        initial.fields.clone_from(&self.fields);
        initial.typed_fields.extend(self.typed_fields.clone());
        initial.typed_ports.extend(self.typed_ports.clone());
        initial
    }
}

pub(super) fn candidate_maps(
    program: &KernelProgram,
    variables: &[Variable],
    values: &[f64],
    state: &RuntimeState,
) -> Result<CandidateMaps, Diagnostic> {
    let mut candidates = CandidateMaps {
        fields: BTreeMap::new(),
        typed_fields: BTreeMap::new(),
        derivatives: state.derivatives.clone(),
        next_fields: BTreeMap::new(),
        typed_next: BTreeMap::new(),
        ports: BTreeMap::new(),
        typed_ports: BTreeMap::new(),
        physical: BTreeMap::new(),
    };
    let mut cursor = 0usize;
    for &variable in variables {
        let value_type = variable.value_type(program)?;
        let count = numeric_width(&value_type)?;
        let parts = if value_type.scalar_domain() == ScalarDomain::Complex {
            2
        } else {
            1
        };
        let end = cursor
            .checked_add(count)
            .ok_or_else(|| execution_error("numerical coordinate cardinality overflows", 0.))?;
        let coefficients = values
            .get(cursor..end)
            .ok_or_else(|| execution_error("numerical point omits an unknown component", 0.))?;
        let value = ValueLiteral::new(
            value_type,
            coefficients
                .chunks_exact(parts)
                .map(|part| (part[0], if parts == 2 { part[1] } else { 0. })),
        )
        .map_err(|_| execution_error("numerical unknown does not match its exact type", 0.))?;
        cursor = end;
        match variable {
            Variable::Field(id) => insert(
                id,
                value,
                &mut candidates.fields,
                &mut candidates.typed_fields,
            ),
            Variable::Derivative(id, order) => {
                candidates.derivatives.insert((id, order), value);
            }
            Variable::NextField(id) => insert(
                id,
                value,
                &mut candidates.next_fields,
                &mut candidates.typed_next,
            ),
            Variable::Port(id) => insert(
                id,
                value,
                &mut candidates.ports,
                &mut candidates.typed_ports,
            ),
            Variable::Physical(unknown) => {
                candidates
                    .physical
                    .insert(unknown, evaluate::real(&value)?.value());
            }
        }
    }
    if cursor != values.len() {
        return Err(execution_error(
            "numerical point contains foreign coordinates",
            0.,
        ));
    }
    Ok(candidates)
}

fn insert(
    id: RawId,
    value: ValueLiteral,
    scalar: &mut BTreeMap<RawId, f64>,
    typed: &mut BTreeMap<RawId, ValueLiteral>,
) {
    if let Some(value) = value.real_scalar_value() {
        scalar.insert(id, value.value());
    } else {
        typed.insert(id, value);
    }
}

pub(super) fn commit_solution(
    program: &KernelProgram,
    variables: &[Variable],
    values: &[f64],
    state: &mut RuntimeState,
) -> Result<(), Diagnostic> {
    // Validate the complete candidate before changing any accepted storage.
    let candidates = candidate_maps(program, variables, values, state)?;
    for id in candidates
        .typed_fields
        .keys()
        .chain(candidates.typed_next.keys())
    {
        state.fields.remove(id);
    }
    for id in candidates.typed_ports.keys() {
        state.ports.remove(id);
    }
    state.fields.extend(candidates.fields);
    state.fields.extend(candidates.next_fields);
    state.typed_fields.extend(candidates.typed_fields);
    state.typed_fields.extend(candidates.typed_next);
    state.derivatives = candidates.derivatives;
    state.ports.extend(candidates.ports);
    state.typed_ports.extend(candidates.typed_ports);
    state.physical.extend(candidates.physical);
    Ok(())
}

pub(super) fn variable_value(variable: Variable, state: &RuntimeState) -> f64 {
    match variable {
        Variable::Field(id) | Variable::NextField(id) => state.fields[&id],
        Variable::Derivative(id, order) => state
            .derivatives
            .get(&(id, order))
            .map(|value| {
                value
                    .real_scalar_value()
                    .expect("admitted scalar derivative")
                    .value()
            })
            .unwrap_or(0.0),
        Variable::Port(id) => state.ports.get(&id).copied().unwrap_or(0.0),
        Variable::Physical(unknown) => state.physical.get(&unknown).copied().unwrap_or(0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{DimExponents, Id, OntologyId, entity::kinds};
    use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
    use eqiora_schema::{
        Model, ModelView,
        kernel::{ExprDagBuilder, FieldDef, FieldRole, RelationDef},
    };

    #[test]
    fn complete_numeric_points_preserve_components_and_reject_before_commit() {
        let z = Id::<kinds::Field>::new();
        let w = Id::<kinds::Field>::new();
        let model = OntologyId::<Model>::new();
        let unit = DimExponents::DIMENSIONLESS;
        let complex = ValueType::scalar(ScalarDomain::Complex, unit)
            .unwrap()
            .array(6)
            .unwrap();
        let real = ValueType::scalar(ScalarDomain::Real, unit).unwrap();
        let mut transaction = Transaction::new("complete numeric point");
        for (field, ty) in [(z, complex.clone()), (w, real)] {
            transaction.push(Op::DefineKernelNode {
                node: FieldDef::new(field, ty, FieldRole::State).into(),
            });
        }
        let relation = Id::<kinds::Relation>::new();
        let mut builder = ExprDagBuilder::new();
        let lhs = builder.symbol(SymbolRef::Field(z)).unwrap();
        let rhs = builder
            .constant(ValueLiteral::new(complex.clone(), [(0., 0.); 6]).unwrap())
            .unwrap();
        transaction.push(Op::DefineKernelNode {
            node: RelationDef::initial(relation, builder.finish([lhs, rhs]).unwrap())
                .unwrap()
                .into(),
        });
        transaction.push(Op::Connect {
            from: relation.erase(),
            to: z.erase(),
            edge: EdgeKind::DependsOn,
        });
        transaction.push(Op::DefineOntologyView {
            view: ModelView::new(model, [z.erase(), w.erase(), relation.erase()], [])
                .unwrap()
                .into(),
        });
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        let mut state = RuntimeState {
            typed_fields: BTreeMap::new(),
            typed_ports: BTreeMap::new(),
            typed_next: BTreeMap::new(),
            fields: BTreeMap::from([(w.erase(), 17.)]),
            derivatives: BTreeMap::new(),
            ports: BTreeMap::new(),
            physical: BTreeMap::new(),
        };
        let order = std::num::NonZeroU32::MIN;
        let variables = [
            Variable::Field(z.erase()),
            Variable::Derivative(z.erase(), order),
            Variable::Field(w.erase()),
        ];
        let values = (0..6)
            .flat_map(|i| [i as f64 + 1., 2. * i as f64])
            .chain((0..6).flat_map(|i| [-2. * i as f64, i as f64 + 1.]))
            .chain([3.])
            .collect::<Vec<_>>();
        for invalid in [
            values[..24].to_vec(),
            [values.as_slice(), &[0.]].concat(),
            {
                let mut invalid = values.clone();
                invalid[23] = f64::INFINITY;
                invalid
            },
        ] {
            assert!(commit_solution(&program, &variables, &invalid, &mut state).is_err());
            assert_eq!(state.fields[&w.erase()], 17.);
            assert!(state.typed_fields.is_empty() && state.derivatives.is_empty());
        }
        let candidate = candidate_maps(&program, &variables, &values, &state).unwrap();
        assert_eq!(
            candidate.initial_context_state(&state).typed_fields[&z.erase()].value_type(),
            &complex
        );
        commit_solution(&program, &variables, &values, &mut state).unwrap();
        assert_eq!(state.fields[&w.erase()], 3.);
        let derivative = &state.derivatives[&(z.erase(), order)];
        assert_eq!(
            derivative.value_type().dimension(),
            DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap()
        );
        for i in 0..6 {
            assert_eq!(
                state.typed_fields[&z.erase()].component(i),
                Some((i as f64 + 1., 2. * i as f64))
            );
            assert_eq!(
                derivative.component(i),
                Some((-2. * i as f64, i as f64 + 1.))
            );
        }
    }
}
