//! Directional curl Green identity, independent of a numerical vector space.
use super::scalar::{authored::dimensions, recognition::value_degree, typed_relation};
use super::vocabulary::*;
use crate::additive_residual::{AdditiveResidualView, AdditiveSign};
use crate::canonical::{boundary_parent, continuum_fields_on, lowering_error, relations_on};
use eqiora_compiler::{AuthoredFormExpressionV1 as E, AuthoredFormulationProjection};
use eqiora_core::{Diagnostic, RawId, ScalarDomain, ValueFrame};
use eqiora_schema::kernel::{
    BoundarySide, DomainKind, ExprDag, ExprId, ExprNode, KernelNode, SymbolRef,
    pure_operator::PureOperatorDefinition,
};
use eqiora_sem::KernelProgram;
use std::collections::BTreeMap;

/// Check an authored 3D vector curl-curl form without choosing a numerical method.
///
/// This bounded strong-implies-weak check admits one direct curl-curl occurrence,
/// linear algebraic reaction/load terms, and one homogeneous full-trace or
/// tangential-curl law on each Cartesian box face. Complex trials require
/// conjugated test pairing. Only full-trace faces restrict the test to zero.
/// It establishes neither reverse implication nor uniqueness, H(curl) conformity,
/// tangential-only essential conditions, nonzero curl flux, or a numerical Maxwell realization.
///
/// # Errors
/// Rejects unsupported sources, stale identities, incomplete boundary conditions,
/// incompatible units, or a weak residual differing from the curl Green identity.
pub(super) fn check(
    program: &KernelProgram,
    form: &AuthoredFormulationProjection,
) -> Result<(), Diagnostic> {
    let domain = program
        .nodes()
        .find_map(|node| match node {
            KernelNode::Domain(domain)
                if Some(domain.id().ulid().to_string().as_str()) == form.domain_ulid() =>
            {
                Some(domain)
            }
            _ => None,
        })
        .ok_or_else(|| {
            Diagnostic::error(
                eqiora_core::diagnostic::codes::INVALID_DISCRETIZATION,
                "curl correspondence requires a live volume Domain",
            )
        })?;
    let owner = domain.id().erase();
    let reject = |message| lowering_error(owner, message);
    if !matches!(domain.kind(), DomainKind::CartesianBox { .. })
        || program.resolved_cartesian_bounds(domain.id())?.len() != 3
        || form.interval().is_some()
        || form.gauge_field_ulids().is_some()
    {
        return Err(reject(
            "curl correspondence requires a spatial 3D Cartesian box",
        ));
    }
    let [trial] = form.trial_ulids() else {
        return Err(reject("curl correspondence requires one vector trial"));
    };
    let field = continuum_fields_on(program, owner)
        .into_iter()
        .find(|id| &id.ulid().to_string() == trial)
        .ok_or_else(|| reject("curl trial is not a continuum Field on the exact volume"))?;
    let Some(KernelNode::Field(definition)) = program.node(field) else {
        unreachable!()
    };
    let value_type = definition.value_type();
    if value_type.array_rank() != 0
        || value_type.frame() != ValueFrame::SpatialCartesian
        || !value_type
            .shape()
            .extents()
            .iter()
            .map(|extent| extent.get())
            .eq([3])
        || !matches!(
            value_type.scalar_domain(),
            ScalarDomain::Real | ScalarDomain::Complex
        )
    {
        return Err(reject(
            "curl trial requires a real or complex physical 3-vector",
        ));
    }
    let complex = value_type.scalar_domain() == ScalarDomain::Complex;
    let [(relation, _, _)] = form.equations() else {
        return Err(reject("curl correspondence requires one selected equation"));
    };
    let relation = relations_on(program, owner)
        .into_iter()
        .find(|id| &id.ulid().to_string() == relation)
        .ok_or_else(|| reject("curl equation is not a Relation on the exact volume"))?;
    let typed = typed_relation(program, relation)?;
    let dag = typed.expression();
    let [root] = dag.roots() else {
        return Err(reject("curl relation requires one residual root"));
    };
    let view = AdditiveResidualView::derive(dag, *root, relation)?;
    let operators = view
        .leaves()
        .iter()
        .filter(|leaf| curl_curl_field(dag, leaf.value()) == Some(field))
        .collect::<Vec<_>>();
    let [operator] = operators.as_slice() else {
        return Err(reject(
            "curl relation requires exactly one direct shared 3D curl-curl",
        ));
    };
    let mut values = Vec::new();
    for leaf in view.leaves() {
        if leaf.value() == operator.value() {
            continue;
        }
        // Reuse the closed linear-value inventory; derivatives and foreign
        // unknowns cannot be disguised as reaction or load coefficients.
        let degree = value_degree(dag, leaf.value(), field, relation, false, complex)?;
        if degree > 1 {
            return Err(reject(
                "curl value pairing requires linear trial dependence",
            ));
        }
        let trial_dependent = degree == 1;
        values.push(PrimalValueTerm {
            source_node: leaf.value(),
            trial_dependent,
            sign: sign((leaf.sign() == AdditiveSign::Positive) == trial_dependent),
        });
    }
    let boundaries = boundaries(program, owner, field)?;
    let source = PrimalGalerkinSource {
        domain: owner,
        unknown: field,
        volume_relation: relation,
        root: *root,
        divergence: operator.value(),
        diffusion_rule: DiffusionRule::VectorCurlCurl,
        // ∫ v·curl curl u = ∫ curl v·curl u − ∮ (n×v)·curl u.
        // Full zero test trace or the live natural law n×curl(u)=0
        // discharges the surface term. The latter leaves the test unrestricted.
        divergence_sign: sign(operator.sign() == AdditiveSign::Positive),
        values: &values,
        conjugate_test: complex,
        boundaries: &boundaries,
    };
    let certificate = PrimalGalerkinCorrespondence::derive(source);
    certificate.replay(source).map_err(reject)?;
    certificate.replay_authored(form, program, 3)?;
    dimensions::check_pairing(form, program, owner, 3,
        &boundaries.iter().map(|boundary| boundary.domain).collect::<Vec<_>>(), &typed)
        .ok_or_else(|| reject("curl weak residual dimensions or coordinate support differ from the strong-law pairing"))?;
    let test = E::Test {
        field_ulid: trial.clone(),
    };
    let curl = |value| E::Curl {
        value: Box::new(value),
    };
    let pair = |left, right| {
        if complex {
            E::Inner {
                left: Box::new(left),
                right: Box::new(right),
            }
        } else {
            E::Dot {
                left: Box::new(left),
                right: Box::new(right),
            }
        }
    };
    let integrate = |integrand| E::Integrate {
        domain_ulid: owner.ulid().to_string(),
        integrand: Box::new(integrand),
    };
    let mut left = signed(
        integrate(pair(
            curl(test.clone()),
            curl(E::Field {
                ulid: trial.clone(),
            }),
        )),
        source.divergence_sign,
    );
    let mut right = E::Number { value: 0.0 };
    for term in values {
        let value = E::from_expression(dag, term.source_node)?
            .ok_or_else(|| reject("curl source value exceeds the authored expression inventory"))?;
        // Typed zero may have scalar storage after residual projection.
        if matches!(dag.node(term.source_node), Some(ExprNode::Constant(value)) if value.is_zero())
        {
            continue;
        }
        let integral = signed(integrate(pair(test.clone(), value)), term.sign);
        let side = if term.trial_dependent {
            &mut left
        } else {
            &mut right
        };
        *side = E::Add {
            left: Box::new(side.clone()),
            right: Box::new(integral),
        };
    }
    if !super::authored_polynomial::matches_weak_residual(form, program, 3, &left, &right) {
        return Err(reject(
            "authored curl weak residual differs from the strong-law curl Green identity",
        ));
    }
    Ok(())
}

fn sign(positive: bool) -> WeakSign {
    if positive {
        WeakSign::Positive
    } else {
        WeakSign::Negative
    }
}
fn signed(value: E, sign: WeakSign) -> E {
    if sign == WeakSign::Negative {
        E::Neg {
            value: Box::new(value),
        }
    } else {
        value
    }
}

pub(super) fn curl_curl_field(dag: &ExprDag, root: ExprId) -> Option<RawId> {
    let field = curl_operand(dag, curl_operand(dag, root)?)?;
    let ExprNode::Symbol(SymbolRef::Field(field)) = dag.node(field)? else {
        return None;
    };
    Some(field.erase())
}

// Exact shared definitions are required in both the volume and boundary laws.
pub(super) fn curl_operand(dag: &ExprDag, root: ExprId) -> Option<ExprId> {
    let gradient = pure_operand(
        dag,
        root,
        PureOperatorDefinition::curl_from_gradient(3, 1).ok()?,
    )?;
    let ExprNode::Gradient(value) = dag.node(gradient)? else {
        return None;
    };
    Some(*value)
}
pub(super) fn tangential_lift_operand(dag: &ExprDag, root: ExprId) -> Option<ExprId> {
    pure_operand(dag, root, PureOperatorDefinition::tangential_lift(3).ok()?)
}
fn pure_operand(dag: &ExprDag, root: ExprId, expected: PureOperatorDefinition) -> Option<ExprId> {
    let ExprNode::PureOperatorApplication(application) = dag.node(root)? else {
        return None;
    };
    (dag.definition(application.definition())? == &expected).then_some(())?;
    let [argument] = application.arguments() else {
        return None;
    };
    Some(*argument)
}
fn boundary_discharge(dag: &ExprDag, root: ExprId, field: RawId) -> Option<BoundaryDischarge> {
    let (argument, discharge) = match dag.node(root)? {
        ExprNode::Trace {
            value: argument, ..
        } => (*argument, BoundaryDischarge::ZeroTestTrace),
        ExprNode::NormalComponent { value: lift, .. } => {
            // n × curl(u) = 0, not n · curl(u) = 0. The shared lift
            // fixes the cross-product orientation before normal contraction.
            let curl = tangential_lift_operand(dag, *lift)?;
            (curl_operand(dag, curl)?, BoundaryDischarge::ZeroFlux)
        }
        _ => return None,
    };
    matches!(dag.node(argument), Some(ExprNode::Symbol(SymbolRef::Field(id))) if id.erase() == field).then_some(discharge)
}

fn boundaries(
    program: &KernelProgram,
    volume: RawId,
    field: RawId,
) -> Result<Vec<BoundarySource>, Diagnostic> {
    let reject = || {
        lowering_error(
            volume,
            "curl correspondence requires one homogeneous full trace or tangential-curl law on each of six exact box faces",
        )
    };
    let mut sides = BTreeMap::new();
    for node in program.nodes() {
        let KernelNode::Domain(domain) = node else {
            continue;
        };
        let id = domain.id().erase();
        if boundary_parent(program, id) != Some(volume) {
            continue;
        }
        let DomainKind::CartesianBoundary { axis, side } = domain.kind() else {
            return Err(reject());
        };
        if *axis >= 3 {
            return Err(reject());
        }
        let relations = relations_on(program, id);
        let [relation] = relations.as_slice() else {
            return Err(reject());
        };
        let typed = typed_relation(program, *relation)?;
        let dag = typed.expression();
        let [root] = dag.roots() else {
            return Err(reject());
        };
        let view = AdditiveResidualView::derive(dag, *root, *relation)?;
        let nonzero = view.leaves().iter().filter(|leaf|
            !matches!(dag.node(leaf.value()), Some(ExprNode::Constant(value)) if value.is_zero())
        ).collect::<Vec<_>>();
        let [trace] = nonzero.as_slice() else {
            return Err(reject());
        };
        let discharge = boundary_discharge(dag, trace.value(), field).ok_or_else(reject)?;
        if sides
            .insert(
                (*axis, *side),
                BoundarySource {
                    domain: id,
                    relation: *relation,
                    operator_node: trace.value(),
                    discharge,
                },
            )
            .is_some()
        {
            return Err(reject());
        }
    }
    let expected =
        (0..3).flat_map(|axis| [(axis, BoundarySide::Lower), (axis, BoundarySide::Upper)]);
    expected
        .map(|side| sides.remove(&side).ok_or_else(reject))
        .collect()
}
