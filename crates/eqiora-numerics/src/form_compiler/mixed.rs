//! Authored equation/test rows checked against the shared spatial equation owner.
use super::{
    authored_polynomial::{self, weak},
    scalar::{authored::dimensions, typed_relation},
    spatial::CompiledSpatialForm,
};
use crate::canonical_boundary::PhysicalBoundaryQuantity;
use crate::spatial_expression::Coefficient;
use eqiora_compiler::{AuthoredFormExpressionV1 as E, AuthoredFormulationProjection};
use eqiora_core::{Diagnostic, Id, ScalarDomain, entity::kinds};
use eqiora_sem::KernelProgram;
use std::collections::BTreeSet;

pub(crate) fn check(
    program: &KernelProgram,
    form: &AuthoredFormulationProjection,
) -> Result<(), Diagnostic> {
    resolve(program, form).map(|_| ())
}

pub(crate) fn resolve(
    program: &KernelProgram,
    form: &AuthoredFormulationProjection,
) -> Result<AuthoredFormulationProjection, Diagnostic> {
    super::check_authored_dependence(form, program)?;
    let complex = form.trial_ulids().iter().any(|id| {
        id.parse().ok().and_then(|id| program.node(Id::<kinds::Field>::from_ulid(id).erase()))
            .is_some_and(|node| matches!(node, eqiora_schema::kernel::KernelNode::Field(field) if field.value_type().scalar_domain() == ScalarDomain::Complex))
    });
    if complex {
        resolve_typed::<num_complex::Complex64>(program, form)
    } else {
        resolve_typed::<f64>(program, form)
    }
}

fn resolve_typed<S: Coefficient>(
    program: &KernelProgram,
    form: &AuthoredFormulationProjection,
) -> Result<AuthoredFormulationProjection, Diagnostic> {
    if form.interval().is_some()
        || form.gauge_field_ulids().is_some()
        || form.implication() != "strong-implies-weak"
        || !form.assumptions().iter().map(String::as_str).eq(
            AuthoredFormulationProjection::required_assumptions()
                .iter()
                .copied(),
        )
    {
        return Err(reject(
            "spatial row correspondence requires the common weak-form hypotheses",
        ));
    }
    let domain = Id::<kinds::Domain>::from_ulid(
        form.domain_ulid()
            .ok_or_else(|| reject("missing spatial support"))?
            .parse()
            .map_err(|_| reject("invalid spatial support"))?,
    );
    let dimension = program
        .spatial_support(domain)
        .and_then(|support| support.ambient_dimensions())
        .ok_or_else(|| reject("spatial support is not live"))?;
    let tests = form.equation_test_fields()?;
    let spatial = CompiledSpatialForm::<S>::derive_with_tests(
        program,
        domain.erase(),
        dimension,
        &BTreeSet::new(),
        None,
        &tests,
    )?;
    if spatial.motion().is_some() {
        return Err(reject(
            "fixed-domain weak correspondence does not certify moving-domain transport",
        ));
    }
    let trials = spatial
        .fields()
        .iter()
        .map(|(field, _)| field.ulid().to_string())
        .collect::<BTreeSet<_>>();
    if trials != form.trial_ulids().iter().cloned().collect() {
        return Err(reject(
            "authored trials differ from the complete spatial unknown inventory",
        ));
    }
    let mut derived = Vec::new();
    for (relation, field) in tests {
        let field_name = field.ulid().to_string();
        let relation_name = relation.ulid().to_string();
        let (_, _, zero_on, _, regularity) = form
            .test_restrictions()
            .iter()
            .find(|(_, id, _, _, _)| id == &field_name)
            .ok_or_else(|| reject("missing test restriction"))?;
        let (_, value_type) = spatial
            .fields()
            .iter()
            .find(|(id, _)| *id == field)
            .ok_or_else(|| reject("missing tested Field"))?;
        let laws = spatial
            .boundary_laws()
            .get(&field)
            .ok_or_else(|| reject("missing boundary inventory"))?;
        let essential = laws
            .values()
            .filter(|law| law.quantity == PhysicalBoundaryQuantity::Trace)
            .map(|law| law.binding.boundary().ulid().to_string())
            .collect::<BTreeSet<_>>();
        if zero_on.iter().cloned().collect::<BTreeSet<_>>() != essential
            || (regularity.as_deref() != Some("h1")
                && !(laws.is_empty() && regularity.as_deref() == Some("l2")))
        {
            return Err(reject(
                "test regularity or zero trace differs from the spatial row boundary inventory",
            ));
        }
        let typed = typed_relation(program, relation)?;
        if typed.expression().roots().len() != 1
            || typed
                .node_type(typed.expression().roots()[0])
                .is_none_or(|ty| ty.shape() != value_type.shape())
        {
            return Err(reject("equation value shape differs from its test Field"));
        }
        let test = E::Test {
            field_ulid: field_name.clone(),
        };
        let conjugated = value_type.scalar_domain() == ScalarDomain::Complex;
        let integrand = weak::volume(&typed, test.clone(), conjugated).ok_or_else(|| reject("source expression requires an unsupported Green identity or coefficient product rule"))?;
        let mut expected = E::Integrate {
            domain_ulid: domain.ulid().to_string(),
            integrand: Box::new(integrand),
        };
        for law in laws
            .values()
            .filter(|law| law.quantity == PhysicalBoundaryQuantity::Flux)
        {
            let Some(datum) = law.datum_expression else {
                continue;
            };
            let typed = typed_relation(program, law.binding.relation())?;
            let mut datum = E::from_expression(typed.expression(), datum)?
                .ok_or_else(|| reject("boundary datum exceeds the shared expression projection"))?;
            if law.datum_negative {
                datum = E::Neg {
                    value: Box::new(datum),
                };
            }
            let boundary = law.binding.boundary().ulid().to_string();
            let integrand = weak::pair(
                E::Trace {
                    value: Box::new(test.clone()),
                    on_ulid: boundary.clone(),
                },
                datum,
                value_type.shape().rank(),
                conjugated,
            )
            .ok_or_else(|| reject("boundary pairing has unsupported value shape"))?;
            expected = E::Sub {
                left: Box::new(expected),
                right: Box::new(E::Integrate {
                    domain_ulid: boundary,
                    integrand: Box::new(integrand),
                }),
            };
        }
        derived.push((
            relation_name,
            field_name,
            typed,
            laws.keys().copied().collect::<Vec<_>>(),
            expected,
        ));
    }
    let effective = if form.requests_weak_derivation() && form.equations().is_empty() {
        form.with_derived_equations(
            derived
                .iter()
                .map(|(relation, _, _, _, expression)| {
                    (
                        relation.clone(),
                        expression.clone(),
                        E::Number { value: 0. },
                    )
                })
                .collect(),
        )?
    } else {
        form.clone()
    };
    super::check_authored_dependence(&effective, program)?;
    for (relation_name, field_name, typed, boundaries, expected) in derived {
        dimensions::check_row(
            &effective,
            program,
            domain.erase(),
            dimension,
            &boundaries,
            &typed,
            (&relation_name, &field_name),
        )
        .ok_or_else(|| {
            reject("authored row units or support differ from the strong-law pairing")
        })?;
        if !authored_polynomial::matches_weak_row(
            &effective,
            program,
            dimension,
            (&relation_name, &field_name),
            &expected,
            &E::Number { value: 0. },
        ) {
            return Err(reject(
                "authored row differs from the original equation and its boundary laws",
            ));
        }
    }
    Ok(effective)
}

fn reject(message: &str) -> Diagnostic {
    Diagnostic::error(
        eqiora_core::diagnostic::codes::INVALID_DISCRETIZATION,
        format!("authored spatial row correspondence: {message}"),
    )
}
