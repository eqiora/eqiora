//! Exact scalar storage and constant-on-support initialization.
use super::*;
use eqiora_schema::kernel::FieldRole;

pub(super) fn initial_values(
    program: &KernelProgram,
    domain: RawId,
    dimension: usize,
    storage: &BTreeMap<RawId, Data>,
    coefficients: &BTreeMap<RawId, Data>,
    transient: bool,
) -> Result<BTreeMap<RawId, Data>, Diagnostic> {
    let mut values = BTreeMap::new();
    if !transient {
        return Ok(values);
    }
    for (field, capacity) in storage {
        if capacity.spatial() || capacity.evaluate(&vec![0.0; dimension])? <= 0.0 {
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
    if initials.len() != storage.len() {
        return Err(invalid(
            "scalar storage requires exactly one initial equation for each stored Field",
        ));
    }
    for relation in initials {
        if relation.equation_sides().len() != 1 {
            return Err(invalid("scalar initial condition requires one equation"));
        }
        let typed = program
            .typed_relation_residual(relation.id())
            .map_err(|errors| errors.into_iter().next().expect("typing diagnostic"))?;
        let expression = program.numerical_residuals(relation.id().erase())?;
        let [root] = expression.roots() else {
            return Err(invalid("scalar initial condition requires one residual"));
        };
        let field = |id| match expression.node(id) {
            Some(ExprNode::Symbol(SymbolRef::Field(field)))
                if storage.contains_key(&field.erase()) =>
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
                    "scalar initial condition must equate the exact stored Field to a constant",
                ));
            }
        };
        let Some(KernelNode::Field(definition)) = program.node(target) else {
            unreachable!()
        };
        let root_type = typed
            .node_type(*root)
            .ok_or_else(|| invalid("missing typed initial condition"))?;
        if &root_type.value_type != definition.value_type()
            || root_type.support.as_ref().map(|support| *support.domain()) != Some(domain)
        {
            return Err(invalid(
                "scalar initial condition type or support differs from its exact stored Field",
            ));
        }
        let context = Context {
            program,
            dag: &expression,
            owner: relation.id().erase(),
            dimension,
            coefficients,
        };
        let data = rhs
            .map(|rhs| context.data(rhs, 0))
            .transpose()?
            .unwrap_or_else(|| Data::constant(dimension, 0.0));
        if data.spatial()
            || !data.evaluate(&vec![0.0; dimension])?.is_finite()
            || values.insert(target, data).is_some()
        {
            return Err(invalid(
                "scalar initial condition must be finite, constant and unique on its support",
            ));
        }
    }
    Ok(values)
}

impl CompiledLinearBlockForm {
    pub(crate) fn is_transient(&self) -> bool {
        !self.storage.is_empty()
    }
    pub(crate) fn initial_values(&self) -> Result<BTreeMap<RawId, f64>, Diagnostic> {
        self.initial
            .iter()
            .map(|(field, data)| Ok((*field, data.evaluate(&vec![0.0; self.dimension])?)))
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
            if capacity.spatial() || !value.is_finite() || value <= 0.0 {
                return Err(invalid(
                    "scalar storage requires finite strictly positive constant capacity",
                ));
            }
        }
        self.initial_values()?;
        Ok(())
    }
}

pub(super) fn require_closed_law(
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
