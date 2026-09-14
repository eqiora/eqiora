//! Replay the divergence theorem directly against retained live Law terms.
//! This checker neither calls the authored compiler nor rebuilds its projection.
use super::super::*;
use eqiora_schema::kernel::{ExprDag, ExprId, ExprNode, RelationMeaning, SymbolRef};

fn reject() -> Diagnostic {
    Diagnostic::error(
        codes::INVALID_REALIZATION,
        "interval correspondence differs from the live Law, endpoint scope, orientation or assumptions",
    )
}

pub(super) struct Statement<'a> {
    pub relation: &'a str,
    pub domain: &'a str,
    pub trial: &'a str,
    pub binder: Option<(&'a str, &'a str, &'a str)>,
    pub implication: &'a str,
    pub assumptions: &'a [String],
    pub left: &'a AuthoredFormExpressionV1,
    pub right: &'a AuthoredFormExpressionV1,
}
impl<'a> From<&'a AuthoredFormulationProjection> for Statement<'a> {
    fn from(form: &'a AuthoredFormulationProjection) -> Self {
        Self {
            relation: form.relation_ulid(),
            domain: form.domain_ulid(),
            trial: form.trial_ulid(),
            binder: form.interval(),
            implication: form.implication(),
            assumptions: form.assumptions(),
            left: form.left(),
            right: form.right(),
        }
    }
}

pub(super) fn check(
    form: Statement<'_>,
    index: &KernelIndex<'_>,
    geometry: &eqiora_geometry::CanonicalGeometryV1,
) -> Result<(), Diagnostic> {
    let Some((interval, lower, upper)) = form.binder else {
        return Err(reject());
    };
    if form.implication != "strong-implies-interval-conservation"
        || !form
            .assumptions
            .iter()
            .map(String::as_str)
            .eq(super::ASSUMPTIONS.iter().copied())
    {
        return Err(reject());
    }
    let relation = index
        .nodes
        .iter()
        .find_map(|(id, node)| (id.ulid().to_string() == form.relation).then_some((*id, *node)))
        .ok_or_else(reject)?;
    let KernelNode::Relation(law) = relation.1 else {
        return Err(reject());
    };
    let RelationMeaning::Conservation(terms) = law.meaning() else {
        return Err(reject());
    };
    terms.validate_balance(law.expression())?;
    if terms.storage().is_some() {
        return Err(reject());
    }
    let domain = index.applies_on.get(&relation.0).ok_or_else(reject)?;
    if domain.ulid().to_string() != form.domain {
        return Err(reject());
    }
    let Some(KernelNode::Domain(definition)) = index.nodes.get(domain).copied() else {
        return Err(reject());
    };
    let eqiora_schema::kernel::DomainKind::GeometryRegion {
        geometry: digest,
        entity_set,
    } = definition.kind()
    else {
        return Err(reject());
    };
    if digest.bytes() != geometry.digest_bytes()
        || geometry.ambient_dimension() != 1
        || geometry.entity_set_dimension(entity_set) != Some(1)
    {
        return Err(reject());
    }
    let trial = index
        .nodes
        .iter()
        .find_map(|(id, node)| (id.ulid().to_string() == form.trial).then_some((*id, *node)))
        .ok_or_else(reject)?;
    let KernelNode::Field(field) = trial.1 else {
        return Err(reject());
    };
    if !field.shape().is_scalar()
        || field.value_type().scalar_domain() != eqiora_core::ScalarDomain::Real
        || index.defined_on.get(&trial.0) != Some(domain)
    {
        return Err(reject());
    }
    for node in law.expression().nodes() {
        if let ExprNode::Symbol(SymbolRef::Field(id)) = node
            && id.erase() != trial.0
        {
            return Err(reject());
        }
    }
    // Recheck physical types from live definitions, including replay of a decoded
    // projection. An unvalidated Transaction cannot assert that its Law is typed.
    use eqiora_schema::kernel::typing::{
        ExpressionType, RootContract, SpatialSupport, TypedResidual,
    };
    let support = SpatialSupport::Volume {
        domain: *domain,
        dimensions: 1,
    };
    TypedResidual::infer(
        law.expression().clone(),
        Some(support.clone()),
        RootContract::EquationSides,
        |symbol| match symbol {
            SymbolRef::Field(id) if id.erase() == trial.0 => Ok(ExpressionType::new(
                field.value_type().clone(),
                Some(support.clone()),
            )),
            SymbolRef::Parameter(id) => match index.nodes.get(&id.erase()).copied() {
                Some(KernelNode::Parameter(value)) if value.real_scalar_value().is_some() => {
                    Ok(ExpressionType::new(value.value_type().clone(), None))
                }
                _ => Err(()),
            },
            _ => Err(()),
        },
    )
    .map_err(|_| reject())?;
    let AuthoredFormExpressionV1::Add { left, right } = form.left else {
        return Err(reject());
    };
    let mut endpoints = std::collections::BTreeSet::new();
    for term in [left.as_ref(), right.as_ref()] {
        let AuthoredFormExpressionV1::EndpointFlux {
            interval: owner,
            endpoint,
            normal,
            flux,
        } = term
        else {
            return Err(reject());
        };
        let expected = if endpoint == lower {
            -1
        } else if endpoint == upper {
            1
        } else {
            return Err(reject());
        };
        if owner != interval
            || *normal != expected
            || !endpoints.insert(endpoint)
            || !matches_source(flux, law.expression(), terms.flux(), 0)
        {
            return Err(reject());
        }
    }
    let AuthoredFormExpressionV1::IntervalIntegral {
        interval: owner,
        integrand,
    } = form.right
    else {
        return Err(reject());
    };
    if owner != interval || !matches_source(integrand, law.expression(), terms.source(), 0) {
        return Err(reject());
    }
    Ok(())
}

fn matches_source(
    form: &AuthoredFormExpressionV1,
    dag: &ExprDag,
    id: ExprId,
    depth: usize,
) -> bool {
    if depth > 128 {
        return false;
    }
    use AuthoredFormExpressionV1 as F;
    let recurse = |form, id| matches_source(form, dag, id, depth + 1);
    match (form, dag.node(id)) {
        (F::Number { value }, Some(ExprNode::Constant(source))) => source
            .real_scalar_value()
            .is_some_and(|source| source.value().to_bits() == value.to_bits()),
        (F::Field { ulid }, Some(ExprNode::Symbol(SymbolRef::Field(id)))) => {
            *ulid == id.ulid().to_string()
        }
        (F::Parameter { ulid }, Some(ExprNode::Symbol(SymbolRef::Parameter(id)))) => {
            *ulid == id.ulid().to_string()
        }
        (F::Coordinate { axis }, Some(ExprNode::SpatialCoordinate(source))) => axis == source,
        (F::Neg { value }, Some(ExprNode::Neg(source)))
        | (F::Gradient { value }, Some(ExprNode::Gradient(source))) => recurse(value, *source),
        (F::Add { left, right }, Some(ExprNode::Add(a, b)))
        | (F::Sub { left, right }, Some(ExprNode::Sub(a, b)))
        | (F::Mul { left, right }, Some(ExprNode::Mul(a, b)))
        | (F::Div { left, right }, Some(ExprNode::Div(a, b))) => {
            recurse(left, *a) && recurse(right, *b)
        }
        (
            F::Sin { value },
            Some(ExprNode::UnaryMath(eqiora_schema::kernel::UnaryMathFunction::Sin, source)),
        ) => recurse(value, *source),
        (F::Pow { base, exponent }, Some(ExprNode::PowI(source, power))) => {
            exponent == power && recurse(base, *source)
        }
        _ => false,
    }
}
