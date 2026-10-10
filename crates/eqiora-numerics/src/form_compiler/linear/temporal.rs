//! Exact scalar storage and support-aware initialization.
use super::*;
use eqiora_schema::kernel::FieldRole;

pub(super) fn initial_values<S: Coefficient>(
    program: &KernelProgram,
    domain: RawId,
    dimension: usize,
    storage: &BTreeMap<RawId, Data<S>>,
    coefficients: &BTreeMap<RawId, Data<S>>,
    initial_fields: &BTreeSet<RawId>,
    time_s: Option<f64>,
) -> Result<BTreeMap<RawId, PrescribedDatum<S>>, Diagnostic> {
    let mut values = BTreeMap::new();
    if initial_fields.is_empty() {
        return Ok(values);
    }
    for (field, capacity) in storage {
        if capacity.spatial() || {
            let value = capacity.evaluate(&vec![0.0; dimension])?;
            value.im() != 0.0 || value.re() <= 0.0
        } {
            return Err(invalid(
                "scalar storage requires a finite strictly positive constant capacity",
            ));
        }
        if !matches!(program.node(*field), Some(KernelNode::Field(value)) if value.role() == FieldRole::State)
        {
            return Err(invalid("scalar storage requires an exact State Field"));
        }
    }
    let initials = program
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Relation(relation) if relation.is_initial() => Some(relation),
            _ => None,
        })
        .collect::<Vec<_>>();
    for relation in initials {
        let typed = program
            .typed_relation_residual(relation.id())
            .map_err(|errors| errors.into_iter().next().expect("typing diagnostic"))?;
        let expression = program.numerical_residuals(relation.id().erase())?;
        for root in expression.roots() {
            let root_type = typed
                .node_type(*root)
                .ok_or_else(|| invalid("missing typed initial condition"))?;
            if root_type.support.as_ref().map(|support| *support.domain()) != Some(domain) {
                continue;
            }
            let field = |id| match expression.node(id) {
                Some(ExprNode::Symbol(SymbolRef::Field(field)))
                    if initial_fields.contains(&field.erase()) =>
                {
                    Some(field.erase())
                }
                _ => None,
            };
            let (target, rhs) = match expression.node(*root) {
                Some(ExprNode::Sub(a, b)) if field(*a).is_some() => (field(*a).unwrap(), Some(*b)),
                Some(ExprNode::Sub(a, b)) if field(*b).is_some() => (field(*b).unwrap(), Some(*a)),
                _ if field(*root).is_some() => (field(*root).unwrap(), None),
                _ => {
                    return Err(invalid(
                        "scalar initial condition must equate the exact stored Field to prescribed data",
                    ));
                }
            };
            let Some(KernelNode::Field(definition)) = program.node(target) else {
                unreachable!()
            };
            if &root_type.value_type != definition.value_type()
                || root_type.support.as_ref().map(|support| *support.domain()) != Some(domain)
            {
                return Err(invalid(
                    "scalar initial condition type or support differs from its exact stored Field",
                ));
            }
            let context = Context {
                time_s,
                program,
                dag: &expression,
                owner: relation.id().erase(),
                dimension,
                coefficients,
            };
            let data = PrescribedDatum::derive(&context, &typed, definition.value_type(), rhs)?;
            if matches!(data, PrescribedDatum::NormalMultiple(_)) {
                return Err(invalid(
                    "volume initial data cannot depend on a boundary normal",
                ));
            }
            if values.insert(target, data).is_some() {
                return Err(invalid(
                    "scalar initial condition must be unique on its support",
                ));
            }
        }
    }
    Ok(values)
}

impl<S: Coefficient> CompiledLinearBlockForm<S> {
    pub(crate) fn is_transient(&self) -> bool {
        !self.storage.is_empty() || !self.kinematics.is_empty()
    }
    pub(crate) fn initial_values_at(
        &self,
        point: &[f64],
    ) -> Result<BTreeMap<RawId, Vec<S>>, Diagnostic> {
        self.initial
            .iter()
            .map(|(field, data)| {
                if point.len() != self.dimension {
                    return Err(invalid(
                        "scalar initial point differs from its exact spatial dimension",
                    ));
                }
                Ok((*field, data.evaluate(point, &[])?))
            })
            .collect()
    }
    pub(crate) fn bind_backward_euler(&self, step: DynQuantity) -> Result<Self, Diagnostic> {
        if !self.is_transient() {
            return Err(invalid("Backward Euler requires exact scalar storage"));
        }
        let mut bound = self.clone();
        bound.step = Some(step);
        bound.volume()?;
        Ok(bound)
    }
    pub(super) fn validate_storage(&self) -> Result<(), Diagnostic> {
        for capacity in self.storage.values() {
            let value = capacity.evaluate(&vec![0.0; self.dimension])?;
            if capacity.spatial() || !value.is_finite() || value.im() != 0.0 || value.re() <= 0.0 {
                return Err(invalid(
                    "scalar storage requires finite strictly positive constant capacity",
                ));
            }
        }
        Ok(())
    }
}

pub(in crate::form_compiler) fn require_closed_law(
    dag: &eqiora_schema::kernel::ExprDag,
    law: eqiora_schema::kernel::ConservationTerms,
) -> Result<(), Diagnostic> {
    let mut pending = dag.roots().to_vec();
    if let Some((stored, accumulation)) = law.storage() {
        pending.extend([stored, accumulation]);
    }
    pending.extend([law.flux(), law.source()]);
    let mut reached = BTreeSet::new();
    while let Some(id) = pending.pop() {
        if !reached.insert(id) {
            continue;
        }
        let node = dag
            .node(id)
            .ok_or_else(|| invalid("retained scalar Law term is absent"))?;
        if let ExprNode::PureOperatorApplication(application) = node {
            pending.extend(application.arguments());
        } else {
            super::super::scalar::push_operands(node, &mut pending);
        }
    }
    for (index, node) in dag.nodes().iter().enumerate() {
        if !reached.contains(&dag.node_id(index as u32).expect("node index"))
            && !matches!(node,ExprNode::Constant(value) if value.is_zero())
        {
            return Err(invalid(
                "scalar Law contains an unconsumed node outside its exact physical terms and accumulation",
            ));
        }
    }
    Ok(())
}
