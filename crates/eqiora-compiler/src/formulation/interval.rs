//! Scalar mathematical interval binding and typed authored balance generation.
use super::*;
use eqiora_lang::FormulationBinding;
use eqiora_schema::kernel::{ExprNode, RelationMeaning, SymbolRef};
use std::collections::BTreeSet;

mod check;
pub(super) const ASSUMPTIONS: &[&str] = &[
    "fixed-one-dimensional-domain",
    "classical-divergence-and-boundary-trace",
    "every-ordered-subinterval-of-parent",
];

pub(super) fn compile(
    file: &str,
    form: (&str, &str, &Expr, &Expr, TextRange),
    binding: &FormulationBinding,
    source_identity: AuthoredFormSourceIdentity,
    symbols: &ModelSymbols,
    index: &KernelIndex<'_>,
    geometry: &eqiora_geometry::CanonicalGeometryV1,
) -> Result<CompiledAuthoredFormulation, Diagnostic> {
    let (form_name, law_name, left, right, range) = form;
    let FormulationBinding::Interval {
        name,
        lower,
        upper,
        domain,
    } = binding
    else {
        unreachable!()
    };
    if (
        geometry.ambient_dimension(),
        geometry.topological_dimension(),
    ) != (1, 1)
    {
        return Err(error(
            file,
            range,
            "interval forms require a one-dimensional physical support",
        ));
    }
    let mut names = BTreeSet::new();
    for name in [name, lower, upper] {
        if !names.insert(name) || symbols.get(name).is_some() {
            return Err(error(
                file,
                range,
                "interval binders must be distinct and cannot shadow Model declarations",
            ));
        }
    }
    let relation = resolve_symbol(file, range, law_name, symbols)?;
    let parent = resolve_symbol(file, range, domain, symbols)?;
    if index.applies_on.get(&relation) != Some(&parent) {
        return Err(error(
            file,
            range,
            "interval parent differs from exact Law support",
        ));
    }
    let Some(KernelNode::Relation(law)) = index.nodes.get(&relation).copied() else {
        return Err(error(
            file,
            range,
            "interval form requires a conservation Law",
        ));
    };
    let RelationMeaning::Conservation(terms) = law.meaning() else {
        return Err(error(
            file,
            range,
            "interval form requires retained physical conservation terms",
        ));
    };
    if terms.storage().is_some() {
        return Err(error(
            file,
            range,
            "interval conservation currently requires a steady Law without storage",
        ));
    }
    let mut fields = BTreeSet::new();
    for node in law.expression().nodes() {
        if let ExprNode::Symbol(SymbolRef::Field(field)) = node {
            fields.insert(field.erase());
        }
    }
    let fields = fields.into_iter().collect::<Vec<_>>();
    let [trial] = fields.as_slice() else {
        return Err(error(
            file,
            range,
            "interval conservation requires exactly one physical scalar Field",
        ));
    };
    let trial = *trial;
    let mut context = ExpressionContext {
        file,
        symbols,
        index,
        ambient_dimension: 1,
        topological_dimension: 1,
        relation_domain: parent
            .downcast()
            .ok_or_else(|| error(file, range, "interval parent is not a Domain"))?,
        tests: BTreeMap::new(),
        used_tests: BTreeSet::new(),
    };
    let mut compile_term =
        |expression: &Expr| -> Result<(AuthoredFormExpressionV1, DimExponents), Diagnostic> {
            let ExprKind::Call {
                callee,
                arguments: eqiora_lang::CallArguments::Positional(args),
            } = expression.kind()
            else {
                return Err(error(
                    file,
                    expression.range(),
                    "interval balance needs explicit endpoint flux and interval integral terms",
                ));
            };
            let owned = |expression: &Expr, expected: &str| matches!(expression.kind(),ExprKind::Name(actual) if actual == expected);
            match (callee.as_str(), args.as_slice()) {
                ("outward_flux", [interval, endpoint, flux]) if owned(interval, name) => {
                    let (endpoint, normal) = if owned(endpoint, lower) {
                        (lower, -1)
                    } else if owned(endpoint, upper) {
                        (upper, 1)
                    } else {
                        return Err(error(
                            file,
                            range,
                            "endpoint is outside the bound mathematical interval",
                        ));
                    };
                    let flux = context.compile(flux)?;
                    if flux.shape != ValueShape::new([1]).expect("one vector component") {
                        return Err(error(
                            file,
                            range,
                            "interval physical flux must be a one-dimensional vector",
                        ));
                    }
                    Ok((
                        AuthoredFormExpressionV1::EndpointFlux {
                            interval: name.clone(),
                            endpoint: endpoint.clone(),
                            normal,
                            flux: Box::new(wire::expression(&flux)),
                        },
                        flux.dimension,
                    ))
                }
                ("integrate", [interval, source]) if owned(interval, name) => {
                    let source = context.compile(source)?;
                    require_scalar(file, range, &source)?;
                    let dimension = source
                        .dimension
                        .mul(length_dimension())
                        .ok_or_else(|| error(file, range, "interval measure dimension overflow"))?;
                    Ok((
                        AuthoredFormExpressionV1::IntervalIntegral {
                            interval: name.clone(),
                            integrand: Box::new(wire::expression(&source)),
                        },
                        dimension,
                    ))
                }
                _ => Err(error(
                    file,
                    range,
                    "interval term has a foreign binder or unsupported mathematical operator",
                )),
            }
        };
    let ExprKind::Binary {
        op: BinaryOp::Add,
        left: first,
        right: second,
    } = left.kind()
    else {
        return Err(error(
            file,
            range,
            "interval boundary balance requires both outward endpoint fluxes",
        ));
    };
    let (first, d1) = compile_term(first)?;
    let (second, d2) = compile_term(second)?;
    let (right, d3) = compile_term(right)?;
    if d1 != d2 || d1 != d3 {
        return Err(error(
            file,
            range,
            "interval conservation terms have different physical dimensions",
        ));
    }
    let projection = AuthoredFormulationProjection::encode_interval(
        source_identity.to_string(),
        relation,
        parent,
        trial,
        form_name.to_owned(),
        (name.clone(), lower.clone(), upper.clone()),
        (
            AuthoredFormExpressionV1::Add {
                left: Box::new(first),
                right: Box::new(second),
            },
            right,
        ),
    )?;
    check::check((&projection).into(), index, geometry)?;
    Ok(CompiledAuthoredFormulation {
        relations: vec![relation.downcast().expect("Law")],
        domain: parent.downcast().expect("Domain"),
        trials: vec![trial.downcast().expect("Field")],
        projection,
        file: file.to_owned(),
        range,
    })
}

impl AuthoredFormulationProjection {
    /// Independently check this interval implication against a complete Model snapshot.
    /// # Errors
    /// Rejects foreign source terms, support, endpoint orientation, scope or hypotheses.
    pub fn check_interval(
        &self,
        transaction: &Transaction,
        geometry: &eqiora_geometry::CanonicalGeometryV1,
    ) -> Result<(), Diagnostic> {
        let index = snapshot_index(transaction)?;
        check::check(self.into(), &index, geometry)
    }
}

fn snapshot_index(transaction: &Transaction) -> Result<KernelIndex<'_>, Diagnostic> {
    let index = KernelIndex::new(transaction);
    let mut values = BTreeSet::new();
    let mut nodes = BTreeSet::new();
    let mut edges = BTreeSet::new();
    for op in transaction.ops() {
        let unique = match op {
            Op::DefineKernelNode { node } => nodes.insert(node.id()),
            Op::SetValue { target, value } => {
                nodes.contains(target)
                    && values.insert(*target)
                    && matches!(index.nodes.get(target).copied(),Some(KernelNode::Parameter(parameter)) if parameter.value_type()==value.value_type())
            }
            Op::Connect { from, to, edge }
                if matches!(edge, EdgeKind::AppliesOn | EdgeKind::BoundaryOf)
                    || (*edge == EdgeKind::DefinedOn
                        && to.downcast::<kinds::Domain>().is_some()) =>
            {
                edges.insert((*from, *edge))
            }
            Op::AddNode { .. } | Op::DefineOntologyView { .. } | Op::Connect { .. } => true,
            _ => false,
        };
        if !unique {
            return Err(error(
                "",
                TextRange::new(0, 0),
                "interval checker requires unique complete definitions, typed Parameter bindings and support edges",
            ));
        }
    }
    Ok(index)
}

/// Derive and independently check the canonical interval content of one retained Law.
/// No authored source identity or provenance is invented for derived mathematics.
/// Returns `None` when the expression exceeds the closed projection inventory.
/// # Errors
/// Rejects an unsupported Law, support, type or interval transformation.
pub fn check_derived_interval_conservation(
    transaction: &Transaction,
    geometry: &eqiora_geometry::CanonicalGeometryV1,
    relation: Id<kinds::Relation>,
) -> Result<Option<()>, Diagnostic> {
    use AuthoredFormExpressionV1 as F;
    let index = snapshot_index(transaction)?;
    let reject = || {
        error(
            "",
            TextRange::new(0, 0),
            "derived interval requires one retained steady scalar Law",
        )
    };
    let Some(KernelNode::Relation(law)) = index.nodes.get(&relation.erase()).copied() else {
        return Err(reject());
    };
    let RelationMeaning::Conservation(terms) = law.meaning() else {
        return Err(reject());
    };
    let domain = index
        .applies_on
        .get(&relation.erase())
        .ok_or_else(reject)?
        .ulid()
        .to_string();
    let fields = law
        .expression()
        .nodes()
        .iter()
        .filter_map(|node| match node {
            ExprNode::Symbol(SymbolRef::Field(id)) => Some(id.erase()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    let [trial] = fields.into_iter().collect::<Vec<_>>()[..] else {
        return Err(reject());
    };
    let Some(flux) = F::from_expression(law.expression(), terms.flux())? else {
        return Ok(None);
    };
    let Some(source) = F::from_expression(law.expression(), terms.source())? else {
        return Ok(None);
    };
    let endpoint = |name: &str, normal| F::EndpointFlux {
        interval: "interval".into(),
        endpoint: name.into(),
        normal,
        flux: Box::new(flux.clone()),
    };
    let left = F::Add {
        left: Box::new(endpoint("a", -1)),
        right: Box::new(endpoint("b", 1)),
    };
    let right = F::IntervalIntegral {
        interval: "interval".into(),
        integrand: Box::new(source),
    };
    check::check(
        check::Statement {
            relation: &relation.ulid().to_string(),
            domain: &domain,
            trial: &trial.ulid().to_string(),
            binder: Some(("interval", "a", "b")),
            implication: "strong-implies-interval-conservation",
            assumptions: &ASSUMPTIONS.iter().map(|s| (*s).into()).collect::<Vec<_>>(),
            left: &left,
            right: &right,
        },
        &index,
        geometry,
    )
    .map(Some)
}
