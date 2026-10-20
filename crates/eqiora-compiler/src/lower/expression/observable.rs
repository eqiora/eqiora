//! All Observable roots share expression lowering and the Kernel reduction contract.
use super::*;
use crate::observation::Reduction;
use eqiora_schema::kernel::{ObservableDef, ObservableMeasure, ObservableReduction};

fn law(
    file: &str,
    range: TextRange,
    name: &str,
    bindings: &BTreeMap<String, Binding>,
) -> Result<(Id<kinds::Relation>, Option<SpatialSupport<RawId>>), Diagnostic> {
    match bindings.get(name) {
        Some(Binding::Relation {
            relation,
            body: LoweringRelationBody::ProbabilityMass { .. },
            ..
        }) => Ok((*relation, None)),
        Some(Binding::Relation {
            relation,
            body: LoweringRelationBody::ProbabilityDensity { domain, .. },
            ..
        }) => Ok((
            *relation,
            Some(relation_support(file, range, domain, bindings)?),
        )),
        _ => Err(unresolved(file, range, name, "probability law")),
    }
}

pub(super) fn outcome(
    file: &str,
    argument: &LoweringExpression,
    bindings: &BTreeMap<String, Binding>,
) -> Result<(Id<kinds::Relation>, ValueType), Diagnostic> {
    let invalid = || {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            argument.range(),
            "outcome requires one exact finite probability law",
        )
    };
    let LoweringExpressionNode::Name(name) = argument.node.as_ref() else {
        return Err(invalid());
    };
    let Some(Binding::Relation {
        relation,
        body: LoweringRelationBody::ProbabilityMass { outcomes, .. },
        ..
    }) = bindings.get(name)
    else {
        return Err(invalid());
    };
    Ok((*relation, outcomes.clone()))
}

fn moment_distribution(
    file: &str,
    values: &[LoweringExpression],
    bindings: &BTreeMap<String, Binding>,
) -> Result<String, Diagnostic> {
    let mut distribution = None;
    for value in values {
        let invalid = || {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                value.range(),
                "probability moments require random variables bound to the same exact law",
            )
        };
        let LoweringExpressionNode::Name(name) = value.node.as_ref() else {
            return Err(invalid());
        };
        let Some(Binding::Observable(
            _,
            _,
            _,
            Some(Reduction::RandomVariable { distribution: name }),
        )) = bindings.get(name)
        else {
            return Err(invalid());
        };
        if distribution
            .replace(name)
            .is_some_and(|previous| previous != name)
        {
            return Err(invalid());
        }
    }
    Ok(distribution.expect("moment has inputs").clone())
}

pub(in crate::lower) struct LoweredObservable {
    pub(in crate::lower) definition: ObservableDef,
    pub(in crate::lower) dependencies: BTreeSet<RawId>,
    pub(in crate::lower) output: Option<Id<kinds::Domain>>,
}

pub(in crate::lower) fn lower_observable(
    file: &str,
    id: Id<kinds::Observable>,
    value_type: &eqiora_lang::ValueTypeSyntax,
    values: &[LoweringExpression],
    reduction: Option<&LoweringReduction>,
    domain: Option<&str>,
    bindings: &BTreeMap<String, Binding>,
) -> Result<LoweredObservable, Diagnostic> {
    let range = values[0].range;
    let probability = match reduction {
        Some(Reduction::RandomVariable { distribution }) => {
            Some(law(file, range, distribution, bindings)?)
        }
        Some(Reduction::ProbabilityMoment { .. }) => Some(law(
            file,
            range,
            &moment_distribution(file, values, bindings)?,
            bindings,
        )?),
        _ => None,
    };
    let support = reduction
        .and_then(|reduction| reduction.domain())
        .map(|domain| relation_support(file, range, domain, bindings))
        .transpose()?
        .or_else(|| {
            probability
                .as_ref()
                .and_then(|(_, support)| support.clone())
        });
    let output = domain
        .map(|name| relation_support(file, range, name, bindings))
        .transpose()?
        .or_else(|| {
            matches!(reduction, Some(Reduction::RandomVariable { .. }))
                .then(|| support.clone())
                .flatten()
        });
    let value_type = crate::value_types::lower_signature_type(
        file,
        value_type,
        output.as_ref().or(support.as_ref()),
    )?;
    let values = values
        .iter()
        .map(|value| {
            contextual::typed_value(
                file,
                value,
                bindings,
                support.as_ref().or(output.as_ref()),
                value_type.scalar_domain(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let inferred = values
        .iter()
        .map(|value| expression_type(file, value, bindings, support.as_ref().or(output.as_ref())))
        .collect::<Result<Vec<_>, _>>()?;
    let integral = matches!(reduction, Some(Reduction::SpatialIntegral { .. }));
    let input = integral
        .then(|| inferred[0].support.as_ref().or(support.as_ref()).cloned())
        .flatten();
    let typed_limits = reduction
        .and_then(|reduction| reduction.limits())
        .map(|limits| {
            limits
                .iter()
                .map(|limit| {
                    let value = contextual::typed_value(
                        file,
                        limit,
                        bindings,
                        support.as_ref(),
                        eqiora_core::ScalarDomain::Real,
                    )?;
                    let ty = expression_type(file, &value, bindings, support.as_ref())?;
                    Ok((value, ty))
                })
                .collect::<Result<Vec<_>, Diagnostic>>()
        })
        .transpose()?;
    let mut reduction = match reduction {
        None => ObservableReduction::Value,
        Some(Reduction::RandomVariable { .. }) => ObservableReduction::RandomVariable {
            distribution: probability.as_ref().expect("bound law").0,
        },
        Some(Reduction::ProbabilityMoment { moment }) => ObservableReduction::ProbabilityMoment {
            distribution: probability.as_ref().expect("bound law").0,
            moment: *moment,
        },
        Some(Reduction::SpatialIntegral {
            domain: name,
            measure,
            ..
        }) => {
            let Some(Binding::Domain(domain, _)) = bindings.get(name) else {
                return Err(unresolved(file, range, name, "Observable Domain"));
            };
            ObservableReduction::SpatialIntegral {
                input: input
                    .as_ref()
                    .expect("integral input")
                    .domain()
                    .downcast()
                    .expect("admitted Domain"),
                domain: *domain,
                limits: None,
                measure: measure.unwrap_or(
                    if matches!(
                        support.as_ref(),
                        Some(
                            SpatialSupport::Boundary { .. }
                                | SpatialSupport::PhysicalInterface { .. }
                        )
                    ) {
                        ObservableMeasure::Boundary
                    } else {
                        ObservableMeasure::Volume
                    },
                ),
            }
        }
    };
    let mut lowerer = ExpressionLowerer {
        file,
        bindings,
        support: input.clone().or(output.clone()),
        builder: ExprDagBuilder::new(),
        dependencies: BTreeSet::new(),
        ports: BTreeSet::new(),
        cache: HashMap::new(),
        sampling: false,
        allow_discrete_symbols: false,
        allow_observables: true,
        activation: &ActivationSyntax::Continuous,
        initial: false,
    };
    let roots = values
        .iter()
        .map(|value| lowerer.lower(value).map(|value| value.id))
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(values) = typed_limits.as_ref() {
        let lower = lowerer.lower(&values[0].0)?.id;
        let upper = lowerer.lower(&values[1].0)?.id;
        let ObservableReduction::SpatialIntegral { limits, .. } = &mut reduction else {
            unreachable!()
        };
        *limits = Some([lower, upper]);
    }
    if let Some((distribution, _)) = probability {
        lowerer.dependencies.insert(distribution.erase());
    }
    let expression = lowerer
        .builder
        .finish(roots)
        .map_err(|error| source_error(codes::LANGUAGE_TYPE_ERROR, file, range, error.message()))?;
    let definition = ObservableDef::new(id, value_type, expression, reduction)?;
    definition
        .validate_type(
            &inferred.iter().collect::<Vec<_>>(),
            typed_limits
                .as_ref()
                .map(|values| [&values[0].1, &values[1].1]),
            input.as_ref(),
            integral.then_some(support.as_ref()).flatten(),
            output.as_ref(),
        )
        .map_err(|error| source_error(codes::LANGUAGE_TYPE_ERROR, file, range, error.message()))?;
    let output = output.map(|support| support.domain().downcast().expect("admitted output Domain"));
    Ok(LoweredObservable {
        definition,
        dependencies: lowerer.dependencies,
        output,
    })
}

/// References read the declared result type and retain a random variable's law support.
pub(super) fn reference_type(
    file: &str,
    range: TextRange,
    value_type: &eqiora_lang::ValueTypeSyntax,
    domain: Option<&str>,
    reduction: Option<&LoweringReduction>,
    bindings: &BTreeMap<String, Binding>,
) -> Result<ExpressionType<RawId>, Diagnostic> {
    let measure_support = match reduction {
        Some(Reduction::SpatialIntegral { domain, .. }) => {
            Some(relation_support(file, range, domain, bindings)?)
        }
        Some(Reduction::RandomVariable { distribution }) => {
            law(file, range, distribution, bindings)?.1
        }
        _ => None,
    };
    let support = domain
        .map(|name| relation_support(file, range, name, bindings))
        .transpose()?
        .or_else(|| {
            matches!(reduction, Some(Reduction::RandomVariable { .. }))
                .then(|| measure_support.clone())
                .flatten()
        });
    let value_type = crate::value_types::lower_signature_type(
        file,
        value_type,
        support.as_ref().or(measure_support.as_ref()),
    )?;
    Ok(ExpressionType::new(value_type, support))
}
