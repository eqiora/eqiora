//! Exact test pairing of an admitted physical-interface continuity equality.
use super::{PrimalFormDescription, authored_polynomial, check_authored_dependence, scalar};
use crate::scalar_conservation::ScalarMaterialInterface;
use eqiora_compiler::{AuthoredFormExpressionV1 as E, AuthoredFormulationProjection};
use eqiora_core::{Diagnostic, Id, ScalarDomain, diagnostic::codes, entity::kinds};
use eqiora_graph::EdgeKind;
use eqiora_realization::ConformingTraceSource;
use eqiora_schema::kernel::{KernelNode, typing::SpatialSupport};
use eqiora_sem::KernelProgram;

pub(crate) fn admit(
    form: &AuthoredFormulationProjection,
    program: &KernelProgram,
    interfaces: &[ScalarMaterialInterface],
) -> Result<PrimalFormDescription, Diagnostic> {
    let reject = |reason: &str| {
        Diagnostic::error(
            codes::INVALID_DISCRETIZATION,
            format!(
                "authored interface form must pair the exact continuity equality with one unrestricted H1 test on an adjacent parent: {reason}"
            ),
        )
    };
    check_authored_dependence(form, program)?;
    let [(owner, _, _)] = form.equations() else {
        return Err(reject("invalid continuity owner or test support"));
    };
    let interface = interfaces
        .iter()
        .find(|interface| {
            interface.physical_support().is_some_and(|domain| {
                form.domain_ulid() == Some(domain.ulid().to_string().as_str())
            }) && matches!(interface.source(), ConformingTraceSource::RelationEquality { relation, .. }
                if owner == &relation.ulid().to_string())
        })
        .ok_or_else(|| reject("missing exact interface authority"))?;
    let domain = interface
        .physical_support()
        .ok_or_else(|| reject("missing exact interface authority"))?;
    let Some(SpatialSupport::PhysicalInterface {
        parents,
        dimensions,
        ..
    }) = program.spatial_support(
        domain
            .downcast()
            .ok_or_else(|| reject("missing exact interface authority"))?,
    )
    else {
        return Err(reject("invalid continuity owner or test support"));
    };
    let ConformingTraceSource::RelationEquality {
        relation,
        root_index,
    } = interface.source()
    else {
        return Err(reject("invalid continuity owner or test support"));
    };
    let [(_, field, restrictions, _, regularity)] = form.test_restrictions() else {
        return Err(reject("invalid continuity owner or test support"));
    };
    if owner != &relation.ulid().to_string()
        || !restrictions.is_empty()
        || regularity.as_deref() != Some("h1")
    {
        return Err(reject("invalid continuity owner or test support"));
    }
    let field_id = Id::<kinds::Field>::from_ulid(
        field
            .parse()
            .map_err(|_| reject("invalid continuity owner or test support"))?,
    );
    let Some(KernelNode::Field(definition)) = program.node(field_id.erase()) else {
        return Err(reject("invalid continuity owner or test support"));
    };
    if !definition.value_type().shape().is_scalar()
        || definition.value_type().scalar_domain() != ScalarDomain::Real
        || !program.edges().iter().any(|edge| {
            edge.kind() == EdgeKind::DefinedOn
                && edge.from() == field_id.erase()
                && parents.contains(&edge.to())
        })
    {
        return Err(reject("invalid continuity owner or test support"));
    }
    let Some(KernelNode::Relation(source)) = program.node(relation.erase()) else {
        return Err(reject("invalid continuity owner or test support"));
    };
    let typed = scalar::typed_relation(program, relation.erase())?;
    // The retained authority names the numerical residual, not the two authored operands.
    let [root] = typed.expression().roots() else {
        return Err(reject("continuity Relation must have one root"));
    };
    if root.index() != root_index {
        return Err(reject("continuity root identity changed"));
    }
    let [(left, right)] = source.equation_sides().collect::<Vec<_>>()[..] else {
        return Err(reject("invalid continuity owner or test support"));
    };
    let left = E::from_expression(source.expression(), left)?
        .ok_or_else(|| reject("unsupported left continuity operand"))?;
    let right = E::from_expression(source.expression(), right)?
        .ok_or_else(|| reject("unsupported right continuity operand"))?;
    scalar::authored::dimensions::check_pairing(
        form,
        program,
        domain,
        dimensions
            .checked_sub(1)
            .ok_or_else(|| reject("missing exact interface authority"))?,
        &[],
        &typed,
    )
    .ok_or_else(|| reject("weak pairing dimensions differ"))?;
    let on_ulid = domain.ulid().to_string();
    let expected = E::Integrate {
        domain_ulid: on_ulid.clone(),
        integrand: Box::new(E::Mul {
            left: Box::new(E::Trace {
                on_ulid,
                value: Box::new(E::Test {
                    field_ulid: field.clone(),
                }),
            }),
            right: Box::new(E::Sub {
                left: Box::new(left),
                right: Box::new(right),
            }),
        }),
    };
    if !authored_polynomial::matches_weak_residual(
        form,
        program,
        *dimensions,
        &expected,
        &E::Number { value: 0.0 },
    ) {
        return Err(reject("weak residual differs from the exact test pairing"));
    }
    Ok((
        super::vocabulary::FormulationKind::PrimalGalerkin,
        "explicit-interface-trace-and-flux-laws",
        vec!["interface.derive.v1.exact-continuity-test-pairing"],
    ))
}
