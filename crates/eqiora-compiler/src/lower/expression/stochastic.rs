//! Lower process and evolution coefficients through the shared pure expression DAG.
use super::*;
use eqiora_schema::kernel::RelationMeaning;

pub(in crate::lower) fn lower_stochastic(
    file: &str,
    range: TextRange,
    body: &LoweringRelationBody,
    bindings: &BTreeMap<String, Binding>,
) -> Result<(LoweredRelation, RelationMeaning), Diagnostic> {
    let activation = ActivationSyntax::Continuous;
    let mut lowerer = ExpressionLowerer {
        file,
        bindings,
        support: None,
        builder: ExprDagBuilder::new(),
        dependencies: BTreeSet::new(),
        ports: BTreeSet::new(),
        cache: HashMap::new(),
        sampling: false,
        allow_discrete_symbols: false,
        allow_observables: false,
        activation: &activation,
        initial: false,
    };
    let scalar_domain = match body {
        LoweringRelationBody::Noise { .. } => eqiora_core::ScalarDomain::Real,
        LoweringRelationBody::Stochastic { target, .. } => {
            let Some(Binding::Field(_, contract)) = bindings.get(target) else {
                return Err(unresolved(file, range, target, "stochastic State"));
            };
            crate::value_types::lower_value_type::<RawId>(file, &contract.value_type, None)?
                .scalar_domain()
        }
        _ => unreachable!("stochastic lowering dispatch"),
    };
    // Keep normalized owners alive for the pointer-keyed lowering cache, and
    // resolve exact integer jump literals through the same context as equations.
    let expressions = body
        .expressions()
        .map(|expression| contextual::typed_value(file, expression, bindings, None, scalar_domain))
        .collect::<Result<Vec<_>, _>>()?;
    let roots = expressions
        .iter()
        .map(|expression| lowerer.lower(expression).map(|value| value.id))
        .collect::<Result<Vec<_>, _>>()?;
    let meaning = match body {
        LoweringRelationBody::Noise {
            process,
            independent,
            ..
        } => RelationMeaning::Noise {
            process: *process,
            parameter: roots[0],
            independent: *independent,
        },
        LoweringRelationBody::Stochastic {
            target,
            calculus,
            drift,
            diffusions,
            jumps,
        } => {
            let Some(Binding::Field(state, _)) = bindings.get(target) else {
                return Err(unresolved(file, range, target, "stochastic State"));
            };
            lowerer.dependencies.insert(state.erase());
            let mut roots = roots.iter().copied();
            let drift = drift.as_ref().map(|_| roots.next().expect("drift root"));
            let mut bind = |values: &[(String, LoweringExpression)]| {
                values
                    .iter()
                    .map(|(name, _)| {
                        let Some(Binding::Relation { relation, .. }) = bindings.get(name) else {
                            return Err(unresolved(file, range, name, "noise group"));
                        };
                        lowerer.dependencies.insert(relation.erase());
                        Ok((*relation, roots.next().expect("coefficient root")))
                    })
                    .collect::<Result<Vec<_>, Diagnostic>>()
            };
            let diffusions = bind(diffusions)?;
            let jumps = bind(jumps)?;
            RelationMeaning::Stochastic {
                target: *state,
                calculus: *calculus,
                drift,
                diffusions,
                jumps,
            }
        }
        _ => unreachable!("stochastic lowering dispatch"),
    };
    Ok((
        LoweredRelation {
            expression: lowerer.builder.finish(roots)?,
            dependencies: lowerer.dependencies,
            ports: lowerer.ports,
        },
        meaning,
    ))
}
