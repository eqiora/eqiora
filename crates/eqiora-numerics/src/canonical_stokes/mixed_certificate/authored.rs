//! Produced-term half of the existing live Stokes correspondence checker.
use super::*;
use eqiora_compiler::{AuthoredFormExpressionV1 as F, AuthoredFormulationProjection};
use eqiora_schema::kernel::{ExprDag, SymbolRef};

/// Check one authored real steady Stokes system without selecting a numerical method.
///
/// Only complete homogeneous velocity trace and an unrestricted pressure test are admitted.
/// This is directional mathematical inspection, not authored mixed execution or stability.
/// # Errors
/// Rejects any unsupported source, stale or incomplete inventory, or unmatched authored term.
pub fn check_authored_mixed_formulation(
    program: &KernelProgram,
    form: &AuthoredFormulationProjection,
) -> Result<(), Diagnostic> {
    let model =
        super::super::recognize::recognize_steady_incompressible_stokes_geometry_mathematics(
            program,
        )?;
    super::check_model(program, &model, &model.correspondence.entries, Some(form))
}

pub(super) fn check_terms(
    program: &KernelProgram,
    source: &SteadyStokesCertificateSource<'_>,
    entries: &[MixedCertificateEntry],
    form: &AuthoredFormulationProjection,
) -> Result<(), Diagnostic> {
    let reject = || {
        lowering_error(
            source.domain,
            "authored mixed form has unmatched equation, term, test/trial, boundary, or assumption",
        )
    };
    if form.domain_ulid() != source.domain.ulid().to_string()
        || form.interval().is_some()
        || form.equations().len() != 2
        || form.trial_ulids().len() != 2
        || form.test_restrictions().len() != 2
        || form.implication() != "strong-implies-weak"
        || !form.assumptions().iter().map(String::as_str).eq(
            AuthoredFormulationProjection::mixed_assumptions()
                .iter()
                .copied(),
        )
    {
        return Err(reject());
    }
    let mut boundary_ids = source
        .boundaries
        .iter()
        .map(|b| b.boundary().ulid().to_string())
        .collect::<Vec<_>>();
    boundary_ids.sort();
    if entries
        .iter()
        .filter(|e| e.role == MixedTermRole::BoundaryLaw)
        .any(|e| e.boundary_disposition != MixedBoundaryDisposition::EssentialTrace)
    {
        return Err(reject());
    }
    for trial in [source.velocity, source.pressure] {
        let id = trial.ulid().to_string();
        let Some((_, _, bounds)) = form
            .test_restrictions()
            .iter()
            .find(|(_, field, _)| field == &id)
        else {
            return Err(reject());
        };
        if (trial == source.velocity && bounds != &boundary_ids)
            || (trial == source.pressure && !bounds.is_empty())
        {
            return Err(reject());
        }
    }
    let mut actual = Vec::new();
    for (relation, left, right) in form.equations() {
        if relation != &source.momentum_relation.ulid().to_string()
            && relation != &source.incompressibility_relation.ulid().to_string()
        {
            return Err(reject());
        }
        flatten(left, relation, false, false, source.domain, &mut actual, 0)?;
        flatten(right, relation, true, false, source.domain, &mut actual, 0)?;
    }
    let mut consumed = BTreeSet::new();
    for entry in entries {
        if !matches!(
            entry.role,
            MixedTermRole::MomentumViscousStress
                | MixedTermRole::MomentumPressureCoupling
                | MixedTermRole::MomentumBodySource
                | MixedTermRole::ContinuityConstraint
        ) {
            continue;
        }
        let dag = typed_relation(program, entry.relation)?;
        let matches = actual
            .iter()
            .enumerate()
            .filter_map(|(i, (relation, negative, term))| {
                (*relation == entry.relation.ulid().to_string()
                    && *negative == (entry.produced_sign == MixedTermSign::Negative)
                    && matches_role(term, entry, dag.expression(), source))
                .then_some(i)
            })
            .collect::<Vec<_>>();
        let [index] = matches.as_slice() else {
            return Err(reject());
        };
        if !consumed.insert(*index) {
            return Err(reject());
        }
    }
    if consumed.len() != actual.len() {
        return Err(reject());
    }
    Ok(())
}

fn flatten<'a>(
    value: &'a F,
    relation: &'a str,
    negative: bool,
    inside: bool,
    domain: RawId,
    out: &mut Vec<(&'a str, bool, &'a F)>,
    depth: usize,
) -> Result<(), Diagnostic> {
    if depth > 128 || out.len() > 4096 {
        return Err(lowering_error(
            domain,
            "authored mixed inventory exceeds checker bounds",
        ));
    }
    match value {
        F::Add { left, right } => {
            flatten(left, relation, negative, inside, domain, out, depth + 1)?;
            flatten(right, relation, negative, inside, domain, out, depth + 1)?;
        }
        F::Sub { left, right } => {
            flatten(left, relation, negative, inside, domain, out, depth + 1)?;
            flatten(right, relation, !negative, inside, domain, out, depth + 1)?;
        }
        F::Neg { value } => flatten(value, relation, !negative, inside, domain, out, depth + 1)?,
        F::Integrate {
            domain_ulid,
            integrand,
        } if !inside && domain_ulid == &domain.ulid().to_string() => {
            flatten(integrand, relation, negative, true, domain, out, depth + 1)?
        }
        F::Number { value } if *value == 0.0 => {}
        _ if inside => out.push((relation, negative, value)),
        _ => {
            return Err(lowering_error(
                domain,
                "authored mixed term escaped its exact integration support",
            ));
        }
    }
    Ok(())
}

fn test(value: &F, field: RawId) -> bool {
    matches!(value,F::Test{field_ulid} if field_ulid==&field.ulid().to_string())
}
fn field(value: &F, id: RawId) -> bool {
    matches!(value,F::Field{ulid} if ulid==&id.ulid().to_string())
}
fn matches_role(
    value: &F,
    entry: &MixedCertificateEntry,
    dag: &ExprDag,
    source: &SteadyStokesCertificateSource<'_>,
) -> bool {
    match (entry.role, value) {
        (MixedTermRole::MomentumViscousStress, F::Frobenius { left, right }) => {
            matches!(left.as_ref(),F::Gradient{value} if test(value,source.velocity))
                && matches_live(right, dag, entry.source_node, 0)
        }
        (MixedTermRole::MomentumPressureCoupling, F::Mul { left, right }) => {
            let pair = |p: &F, v: &F| {
                field(p, source.pressure)
                    && matches!(v,F::Divergence{value} if test(value,source.velocity))
            };
            pair(left, right) || pair(right, left)
        }
        (MixedTermRole::MomentumBodySource, F::Dot { left, right }) => {
            test(left, source.velocity) && matches_live(right, dag, entry.source_node, 0)
        }
        (MixedTermRole::ContinuityConstraint, F::Mul { left, right }) => {
            let pair = |q: &F, u: &F| {
                test(q, source.pressure) && matches_live(u, dag, entry.source_node, 0)
            };
            pair(left, right) || pair(right, left)
        }
        _ => false,
    }
}

// Exact source-DAG comparison, independent of the source-expression generator.
fn matches_live(value: &F, dag: &ExprDag, id: ExprId, depth: usize) -> bool {
    if depth > 128 {
        return false;
    }
    let next = |v, id| matches_live(v, dag, id, depth + 1);
    match (value, dag.node(id)) {
        (F::Number { value }, Some(ExprNode::Constant(c))) => {
            c.real_scalar_value().is_some_and(|v| v.value() == *value)
        }
        (F::Field { ulid }, Some(ExprNode::Symbol(SymbolRef::Field(id)))) => {
            ulid == &id.ulid().to_string()
        }
        (F::Parameter { ulid }, Some(ExprNode::Symbol(SymbolRef::Parameter(id)))) => {
            ulid == &id.ulid().to_string()
        }
        (F::Coordinate { axis }, Some(ExprNode::SpatialCoordinate(a))) => axis == a,
        (F::Neg { value }, Some(ExprNode::Neg(v)))
        | (F::Gradient { value }, Some(ExprNode::Gradient(v)))
        | (F::Divergence { value }, Some(ExprNode::Divergence(v)))
        | (F::SymmetricPart { value }, Some(ExprNode::SymmetricPart(v))) => next(value, *v),
        (F::Add { left, right }, Some(ExprNode::Add(a, b)))
        | (F::Sub { left, right }, Some(ExprNode::Sub(a, b)))
        | (F::Mul { left, right }, Some(ExprNode::Mul(a, b)))
        | (F::Div { left, right }, Some(ExprNode::Div(a, b))) => next(left, *a) && next(right, *b),
        (F::Pow { base, exponent }, Some(ExprNode::PowI(v, n))) => exponent == n && next(base, *v),
        _ => false,
    }
}
