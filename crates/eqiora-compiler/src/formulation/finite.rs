//! Typed global finite coordinates and original equality ownership.
use super::*;
use eqiora_lang::FormulationBinding;
use eqiora_schema::kernel::{FieldRole, RelationConditionKind};
use std::collections::BTreeSet;

pub(super) const ASSUMPTIONS: &[&str] = &[
    "finite-real-scalar-coordinates",
    "original-equality-operands",
];

#[allow(clippy::too_many_arguments)]
pub(super) fn compile(
    file: &str,
    component: &impl FormulationSource,
    form: (&str, &[String], &[(Expr, Expr)], TextRange),
    binding: &FormulationBinding,
    source_identity: AuthoredFormSourceIdentity,
    symbols: &ModelSymbols,
    index: &KernelIndex<'_>,
) -> Result<CompiledAuthoredFormulation, Diagnostic> {
    let (name, relation_names, equations, range) = form;
    let FormulationBinding::Finite {
        name: space,
        trials: names,
    } = binding
    else {
        unreachable!()
    };
    let reject = |message| error(file, range, message);
    if symbols.get(space).is_some()
        || names.is_empty()
        || names.len() != equations.len()
        || equations.len() != relation_names.len()
    {
        return Err(reject(
            "finite form requires a distinct binder and matching ordered Field/equation inventories",
        ));
    }
    let mut trials = Vec::new();
    let mut dimension = None;
    for trial in names {
        let raw = resolve_symbol(file, range, trial, symbols)?;
        let Some(KernelNode::Field(field)) = index.nodes.get(&raw).copied() else {
            return Err(reject("finite trial is not a Field"));
        };
        if trial == space
            || trials.contains(&field.id())
            || index.defined_on.contains_key(&raw)
            || field.role() != FieldRole::Variable
            || !field.shape().is_scalar()
            || field.value_type().scalar_domain() != eqiora_core::ScalarDomain::Real
            || field.value_type().frame() != eqiora_core::ValueFrame::Invariant
        {
            return Err(reject(
                "finite form requires unique global invariant real scalar Fields",
            ));
        }
        if dimension.is_some_and(|value| value != field.dimension()) {
            return Err(reject(
                "finite constant-shift coordinates must have identical dimensions",
            ));
        }
        dimension = Some(field.dimension());
        trials.push(field.id());
    }
    let mut context = ExpressionContext {
        file,
        symbols,
        index,
        ambient_dimension: 0,
        topological_dimension: 0,
        relation_domain: None,
        tests: BTreeMap::new(),
        integration_domain: None,
        used_tests: BTreeSet::new(),
    };
    let mut relations = Vec::new();
    let mut compiled = Vec::new();
    for (relation_name, equality) in relation_names.iter().zip(equations) {
        let raw = resolve_symbol(file, range, relation_name, symbols)?;
        let Some(KernelNode::Relation(relation)) = index.nodes.get(&raw).copied() else {
            return Err(reject("finite form must name an original Relation"));
        };
        if index.applies_on.contains_key(&raw)
            || relation.is_initial()
            || relation.conditions() != Some(&[RelationConditionKind::Equality][..])
            || relations.contains(&relation.id())
        {
            return Err(reject(
                "finite form requires distinct global static equality Relations",
            ));
        }
        let projected = equality_projection(&mut context, equality)?;
        let [(left, right)] = relation.equation_sides().collect::<Vec<_>>()[..] else {
            return Err(reject(
                "finite Relation requires exactly one original equality",
            ));
        };
        let original = (
            AuthoredFormExpressionV1::from_expression(relation.expression(), left)?,
            AuthoredFormExpressionV1::from_expression(relation.expression(), right)?,
        );
        if original.0.as_ref() != Some(&projected.0) || original.1.as_ref() != Some(&projected.1) {
            return Err(reject(
                "finite form equality differs from the original Relation operands",
            ));
        }
        relations.push(relation.id());
        compiled.push((raw.ulid().to_string(), projected.0, projected.1));
    }
    let gauge = component
        .formulation_gauge(name)
        .map(|(binder, conditions)| {
            if binder != space {
                return Err(reject("finite gauge must name its exact coordinate binder"));
            }
            Ok(wire::WireGauge {
                field_ulids: trials.iter().map(|id| id.ulid().to_string()).collect(),
                reference: equality_projection(&mut context, &conditions[0])?,
                compatibility: equality_projection(&mut context, &conditions[1])?,
            })
        })
        .transpose()?;
    let projection = AuthoredFormulationProjection::encode_finite(
        source_identity.to_string(),
        name.into(),
        space.clone(),
        trials.iter().map(|id| id.ulid().to_string()).collect(),
        compiled,
        gauge,
    )?;
    for (_, left, right) in projection.equations() {
        validate_coordinates(left, projection.trial_ulids())?;
        validate_coordinates(right, projection.trial_ulids())?;
    }
    for condition in [
        projection.gauge_reference(),
        projection.gauge_compatibility(),
    ]
    .into_iter()
    .flatten()
    {
        validate_coordinates(condition.0, projection.trial_ulids())?;
        validate_coordinates(condition.1, projection.trial_ulids())?;
    }
    Ok(CompiledAuthoredFormulation {
        relations,
        domain: None,
        trials,
        projection,
        file: file.into(),
        range,
    })
}

fn equality_projection(
    context: &mut ExpressionContext<'_>,
    (left, right): &(Expr, Expr),
) -> Result<(AuthoredFormExpressionV1, AuthoredFormExpressionV1), Diagnostic> {
    let range = left.range();
    let left = context.compile_root(left)?;
    let right = context.compile_root(right)?;
    let zero = |value: &AuthoredFormExpression| {
        matches!(value.kind, AuthoredFormExpressionKind::Number(0.0))
    };
    if left.value_type.dimension() != right.value_type.dimension() && !zero(&left) && !zero(&right)
    {
        return Err(error(
            context.file,
            range,
            "finite equality sides have different physical dimensions",
        ));
    }
    Ok((wire::expression(&left), wire::expression(&right)))
}

// Check ownership and the finite vocabulary only; numerical lowering remains
// with the existing typed Operator IR owner.
fn validate_coordinates(
    value: &AuthoredFormExpressionV1,
    trials: &[String],
) -> Result<(), Diagnostic> {
    use AuthoredFormExpressionV1 as E;
    match value {
        E::Number { .. } | E::Parameter { .. } => Ok(()),
        E::Field { ulid } if trials.contains(ulid) => Ok(()),
        E::Neg { value } | E::Pow { base: value, .. } | E::Sin { value } => {
            validate_coordinates(value, trials)
        }
        E::Add { left, right }
        | E::Sub { left, right }
        | E::Mul { left, right }
        | E::Div { left, right } => {
            validate_coordinates(left, trials)?;
            validate_coordinates(right, trials)
        }
        _ => Err(wire::rejection(
            "finite expression contains a foreign coordinate or spatial operator",
        )),
    }
}
