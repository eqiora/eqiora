//! Lower retained law coefficients through the shared pure expression DAG.
use super::*;
use eqiora_schema::kernel::RelationMeaning;

pub(in crate::lower) fn lower_coefficients(
    file: &str,
    range: TextRange,
    body: &LoweringRelationBody,
    bindings: &BTreeMap<String, Binding>,
) -> Result<(LoweredRelation, RelationMeaning), Diagnostic> {
    let support = match body {
        LoweringRelationBody::ProbabilityDensity { domain, .. } => {
            Some(relation_support(file, range, domain, bindings)?)
        }
        _ => None,
    };
    let activation = ActivationSyntax::Continuous;
    let mut lowerer = ExpressionLowerer {
        file,
        bindings,
        support: support.clone(),
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
        LoweringRelationBody::Noise { .. }
        | LoweringRelationBody::ProbabilityMass { .. }
        | LoweringRelationBody::ProbabilityDensity { .. } => eqiora_core::ScalarDomain::Real,
        LoweringRelationBody::Stochastic { target, .. } => {
            let Some(Binding::Field(_, contract)) = bindings.get(target) else {
                return Err(unresolved(file, range, target, "stochastic State"));
            };
            crate::value_types::lower_value_type::<RawId>(file, &contract.value_type, None)?
                .scalar_domain()
        }
        _ => unreachable!("coefficient law lowering dispatch"),
    };
    // Keep normalized owners alive for the pointer-keyed lowering cache, and
    // resolve exact integer jump literals through the same context as equations.
    let expressions = body
        .expressions()
        .map(|expression| {
            contextual::typed_value(file, expression, bindings, support.as_ref(), scalar_domain)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let roots = expressions
        .iter()
        .map(|expression| lowerer.lower(expression).map(|value| value.id))
        .collect::<Result<Vec<_>, _>>()?;
    let meaning = match body {
        LoweringRelationBody::ProbabilityMass { outcomes, .. } => {
            let outcomes = outcomes.enum_definition().expect("bound outcome Enum");
            lowerer.dependencies.insert(outcomes.erase());
            RelationMeaning::ProbabilityMass {
                outcomes,
                weights: roots[0],
            }
        }
        LoweringRelationBody::ProbabilityDensity {
            domain, measure, ..
        } => {
            let Some(Binding::Domain(domain, _)) = bindings.get(domain) else {
                return Err(unresolved(
                    file,
                    range,
                    domain,
                    "probability support Domain",
                ));
            };
            lowerer.dependencies.insert(domain.erase());
            RelationMeaning::ProbabilityDensity {
                domain: *domain,
                measure: measure.unwrap_or_else(|| {
                    if matches!(
                        support,
                        Some(
                            SpatialSupport::Boundary { .. }
                                | SpatialSupport::PhysicalInterface { .. }
                        )
                    ) {
                        eqiora_schema::kernel::ObservableMeasure::Boundary
                    } else {
                        eqiora_schema::kernel::ObservableMeasure::Volume
                    }
                }),
                density: roots[0],
            }
        }
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
        _ => unreachable!("coefficient law lowering dispatch"),
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
