//! Stochastic Law admission through the ordinary typed expression and Model owners.
use super::*;
use eqiora_core::{ScalarDomain, ValueFrame, ValueType};
use eqiora_schema::kernel::{FieldRole, NoiseProcess, RelationMeaning};

pub(super) fn validate_types(
    owner: RawId,
    meaning: &RelationMeaning,
    typed: &TypedResidual<RawId>,
    environment: TypingEnvironment<'_>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let result = match meaning {
        RelationMeaning::Noise {
            process, parameter, ..
        } => noise(owner, *process, *parameter, typed, environment).map(|_| ()),
        RelationMeaning::Stochastic { .. } => evolution(owner, meaning, typed, environment),
        _ => return,
    };
    if let Err(error) = result {
        diagnostics.push(error);
    }
}

fn noise(
    owner: RawId,
    process: NoiseProcess,
    parameter: eqiora_schema::kernel::ExprId,
    typed: &TypedResidual<RawId>,
    environment: TypingEnvironment<'_>,
) -> Result<Option<u32>, Diagnostic> {
    let value = typed
        .node_type(parameter)
        .expect("validated parameter root");
    let ty = &value.value_type;
    if value.support.is_some()
        || ty.scalar_domain() != ScalarDomain::Real
        || ty.frame() != ValueFrame::Invariant
        || ty.finite_bases().next().is_some()
        || !edge_targets(environment.edges, owner, EdgeKind::AppliesOn).is_empty()
    {
        return Err(kernel_error(
            owner,
            "noise parameters require global real channel values",
        ));
    }
    let time = DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("time dimension");
    let dimension = match process {
        NoiseProcess::Wiener => DimExponents::DIMENSIONLESS,
        NoiseProcess::Poisson => time.pow(-1, 1).expect("inverse time"),
    };
    if ty.dimension() != dimension {
        return Err(relation_dimension_error(
            owner,
            "Wiener covariance is dimensionless; counting intensity has inverse-time units",
        ));
    }
    let channels = match (process, ty.shape().extents()) {
        (_, []) => None,
        (NoiseProcess::Wiener, [rows, columns]) if rows == columns && ty.array_rank() == 2 => {
            Some(rows.get())
        }
        (NoiseProcess::Poisson, [channels]) if ty.array_rank() == 1 => Some(channels.get()),
        _ => {
            return Err(kernel_error(
                owner,
                "noise parameter shape does not match its finite channel law",
            ));
        }
    };
    let constant = typed.expression().nodes().iter().all(|node| {
        !matches!(node,
        ExprNode::Symbol(symbol) if !matches!(symbol, SymbolRef::Parameter(_)))
    });
    if process == NoiseProcess::Wiener || constant {
        let values =
            crate::evaluate::evaluate_expression(owner, typed.expression(), &mut |symbol| {
                let SymbolRef::Parameter(id) = symbol else {
                    return None;
                };
                environment.values.get(&id.erase()).cloned()
            })?;
        let [value] = values.as_slice() else {
            unreachable!("one parameter root")
        };
        match process {
            NoiseProcess::Wiener => {
                NoiseProcess::validate_wiener_covariance(value)?;
            }
            NoiseProcess::Poisson
                if value
                    .components()
                    .expect("real intensity")
                    .any(|(rate, _)| rate < 0.) =>
            {
                return Err(kernel_error(
                    owner,
                    "counting process intensity must be nonnegative",
                ));
            }
            NoiseProcess::Poisson => {}
        }
    }
    Ok(channels)
}

fn evolution(
    owner: RawId,
    meaning: &RelationMeaning,
    typed: &TypedResidual<RawId>,
    environment: TypingEnvironment<'_>,
) -> Result<(), Diagnostic> {
    let RelationMeaning::Stochastic {
        target,
        drift,
        diffusions,
        jumps,
        ..
    } = meaning
    else {
        unreachable!("stochastic evolution dispatch")
    };
    let state = target.erase();
    let Some(KernelNode::Field(field)) = environment.nodes.get(&state) else {
        return Err(kernel_error(
            owner,
            "stochastic Law names a missing or foreign State",
        ));
    };
    if field.role() != FieldRole::State
        || !edge_targets(environment.edges, state, EdgeKind::DefinedOn).is_empty()
        || !edge_targets(environment.edges, state, EdgeKind::ClockedBy).is_empty()
        || !edge_targets(environment.edges, owner, EdgeKind::AppliesOn).is_empty()
    {
        return Err(kernel_error(
            owner,
            "stochastic Law requires its own global physical-time State",
        ));
    }
    let ty = field.value_type();
    if !matches!(
        ty.scalar_domain(),
        ScalarDomain::Real | ScalarDomain::Integer
    ) || (!diffusions.is_empty() && ty.scalar_domain() != ScalarDomain::Real)
    {
        return Err(kernel_error(
            owner,
            "Wiener evolution requires real State values; counting jumps admit real or exact integer States",
        ));
    }
    let time = DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("time dimension");
    let coefficient_type = |duration: DimExponents| -> Result<ValueType, Diagnostic> {
        let dimension = ty.dimension().div(duration).ok_or_else(|| {
            relation_dimension_error(owner, "stochastic coefficient dimension overflows")
        })?;
        ty.clone()
            .with_dimension(dimension)
            .map_err(|error| kernel_error(owner, error.to_string()))
    };
    if let Some(drift) = *drift {
        check_coefficient(owner, typed, drift, &coefficient_type(time)?)?;
    }
    for (process, bindings) in [
        (NoiseProcess::Wiener, diffusions),
        (NoiseProcess::Poisson, jumps),
    ] {
        for (group, coefficient) in bindings {
            let Some(KernelNode::Relation(declaration)) = environment.nodes.get(&group.erase())
            else {
                return Err(kernel_error(
                    owner,
                    "stochastic coefficient names a missing or foreign noise group",
                ));
            };
            let RelationMeaning::Noise {
                process: noise_process,
                parameter: noise_parameter,
                ..
            } = declaration.meaning()
            else {
                return Err(kernel_error(
                    owner,
                    "stochastic coefficient must reference a declared noise Law",
                ));
            };
            if *noise_process != process {
                return Err(kernel_error(
                    owner,
                    "Wiener diffusion and counting jump channels cannot be interchanged",
                ));
            }
            let noise_typed = TypedResidual::infer(
                declaration.expression().clone(),
                None,
                |on| environment.spatial_supports.get(&on.erase()).cloned(),
                RootContract::ValueRoots,
                |symbol| {
                    symbol_type(
                        symbol,
                        environment.nodes,
                        environment.edges,
                        environment.spatial_supports,
                    )
                },
            )
            .map_err(|_| {
                kernel_error(owner, "referenced noise Law has invalid typed parameters")
            })?;
            let channels = noise(
                group.erase(),
                *noise_process,
                *noise_parameter,
                &noise_typed,
                environment,
            )?;
            let mut expected = match process {
                NoiseProcess::Wiener => {
                    coefficient_type(time.pow(1, 2).expect("square root of time"))?
                }
                NoiseProcess::Poisson => ty.clone(),
            };
            if let Some(channels) = channels {
                expected = expected
                    .array(channels)
                    .map_err(|error| kernel_error(owner, error.to_string()))?;
            }
            check_coefficient(owner, typed, *coefficient, &expected)?;
        }
    }
    Ok(())
}

fn check_coefficient(
    owner: RawId,
    typed: &TypedResidual<RawId>,
    root: eqiora_schema::kernel::ExprId,
    expected: &ValueType,
) -> Result<(), Diagnostic> {
    let value = typed.node_type(root).expect("validated coefficient root");
    if &value.value_type != expected || value.support.is_some() {
        return Err(relation_dimension_error(
            owner,
            "stochastic coefficient differs from its exact State type, channel shape or time exponent",
        ));
    }
    Ok(())
}

pub(super) fn validate_ownership(
    nodes: &BTreeMap<RawId, KernelNode>,
    edges: &[Edge],
    diagnostics: &mut Vec<Diagnostic>,
) {
    let mut owners = BTreeMap::new();
    let mut groups = Vec::new();
    for (&id, node) in nodes {
        let KernelNode::Relation(relation) = node else {
            continue;
        };
        match relation.meaning() {
            RelationMeaning::Noise { independent, .. } => groups.push((id, *independent)),
            RelationMeaning::Stochastic { target, .. }
                if owners.insert(target.erase(), id).is_some() =>
            {
                diagnostics.push(kernel_error(
                    id,
                    "a State cannot own conflicting stochastic Laws",
                ));
            }
            _ => {}
        }
    }
    if groups.len() > 1 {
        for (id, independent) in groups {
            if !independent {
                diagnostics.push(kernel_error(id, "distinct noise groups require explicit independence; correlated channels belong to one covariance group"));
            }
        }
    }
    super::probability::validate_sample_dependencies(
        nodes,
        edges,
        owners.keys().copied(),
        diagnostics,
    );
    for (&id, node) in nodes {
        let expression = match node {
            KernelNode::Relation(relation) => relation.expression(),
            KernelNode::Observable(observable) => observable.expression(),
            KernelNode::RecordInstance(instance) => instance.expression(),
            KernelNode::Activation(activation) => match activation.kind() {
                ActivationKind::Event { guard, .. } | ActivationKind::Guard { guard } => guard,
                _ => continue,
            },
            _ => continue,
        };
        if expression.nodes().iter().any(|node| matches!(node,
            ExprNode::Symbol(SymbolRef::Derivative(field, _)) if owners.contains_key(&field.erase()))) {
            diagnostics.push(kernel_error(id, "stochastic State has no ordinary physical-time derivative; coefficient AD is not a path derivative"));
        }
    }
}
