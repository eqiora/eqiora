//! A regular radial conservation law is recognized by coefficients, never by names.
use super::polynomial::{self, Atom, Polynomial, coefficient};
use super::*;
use eqiora_core::{RawId, ScalarDomain, ValueFrame, ValueType};
use eqiora_graph::EdgeKind;
use eqiora_schema::kernel::{
    ActivationKind, BoundarySide, ExprNode, FieldRole, RelationConditionKind, RelationDef,
    SymbolRef,
};

#[derive(Debug, Clone, PartialEq)]
pub(in crate::numerical_admission) struct Diffusion {
    pub(super) concentration: Id<kinds::Field>,
    pub(super) flux: Id<kinds::Field>,
    pub(super) concentration_type: ValueType,
    pub(super) flux_type: ValueType,
    pub(super) diffusivity: f64,
    pub(super) production: f64,
    pub(super) surface: f64,
}

impl Diffusion {
    pub(super) fn lower(
        program: &KernelProgram,
        grid: &CoordinateGrid,
    ) -> Result<Self, Diagnostic> {
        grid.source.require_program(program)?;
        let [factor] = grid.source.factors.as_slice() else {
            return Err(invalid(
                "radial conservation requires one exact interval factor",
            ));
        };
        let domain = parse_domain(&grid.source.domain)?;
        if factor.domain != grid.source.domain
            || factor.lower != 0.0
            || factor.dimension
                != DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0])
                    .expect("length")
                    .exponents()
        {
            return Err(invalid(
                "radial conservation requires a length factor starting at the center zero",
            ));
        }
        let mut fields = Vec::new();
        let mut balances = Vec::new();
        let mut endpoints = Vec::new();
        for node in program.nodes() {
            match node {
                KernelNode::Field(field) => {
                    let ty = field.value_type();
                    if field.role() != FieldRole::Variable
                        || !ty.shape().is_scalar()
                        || ty.array_rank() != 0
                        || ty.scalar_domain() != ScalarDomain::Real
                        || ty.frame() != ValueFrame::Invariant
                        || supports(program, field.id().erase(), EdgeKind::DefinedOn)
                            != [domain.erase()]
                    {
                        return Err(invalid(
                            "radial conservation requires exact-support real scalar variables",
                        ));
                    }
                    fields.push(field);
                }
                KernelNode::Relation(relation) => {
                    if relation.is_initial()
                        || relation.conditions() != Some(&[RelationConditionKind::Equality][..])
                    {
                        return Err(invalid(
                            "radial conservation requires continuous equalities",
                        ));
                    }
                    let support = supports(program, relation.id().erase(), EdgeKind::AppliesOn);
                    if support == [domain.erase()] {
                        balances.push(relation);
                    } else if support.is_empty() {
                        endpoints.push(relation);
                    } else {
                        return Err(invalid("radial relation has foreign support"));
                    }
                }
                KernelNode::Parameter(_) | KernelNode::Observable(_) => {}
                KernelNode::Domain(definition) if definition.id() == domain => {}
                KernelNode::Representation(value)
                    if value.kind() == eqiora_schema::kernel::RepresentationKind::Continuum => {}
                KernelNode::Activation(value)
                    if matches!(value.kind(), ActivationKind::Continuous) => {}
                _ => {
                    return Err(invalid(
                        "radial conservation rejects unrelated or dynamic semantic owners",
                    ));
                }
            }
        }
        if fields.len() != 2 || balances.len() != 2 || endpoints.len() != 2 {
            return Err(invalid(
                "radial conservation requires two Fields, two balances and two endpoint conditions",
            ));
        }
        let mut center = None;
        let mut surface = None;
        for relation in endpoints {
            let (field, point, side, value) = endpoint(program, domain, relation)?;
            if point == 0.0 && side == BoundarySide::Upper && value == 0.0 && center.is_none() {
                center = Some(field);
            } else if point == factor.upper && side == BoundarySide::Lower && surface.is_none() {
                surface = Some((field, value));
            } else {
                return Err(invalid(
                    "radial endpoint conditions require zero center flux and an inward surface value",
                ));
            }
        }
        let flux =
            center.ok_or_else(|| invalid("radial conservation omits regular center flux"))?;
        let (concentration, surface) =
            surface.ok_or_else(|| invalid("radial conservation omits its surface value"))?;
        if flux == concentration {
            return Err(invalid(
                "radial concentration and flux must be distinct Fields",
            ));
        }
        let field_type = |id| {
            fields
                .iter()
                .find(|field| field.id() == id)
                .map(|field| field.value_type().clone())
                .ok_or_else(|| invalid("radial endpoint names a foreign Field"))
        };
        let concentration_type = field_type(concentration)?;
        let flux_type = field_type(flux)?;
        let rows = balances
            .into_iter()
            .map(|relation| {
                let dag = relation.expression();
                let [left, right] = dag.roots() else {
                    return Err(invalid("radial equality requires two original operands"));
                };
                polynomial::normalize(domain, dag, *left)?
                    .checked_add(
                        &polynomial::normalize(domain, dag, *right)?
                            .checked_neg()
                            .map_err(polynomial::error)?,
                    )
                    .map_err(polynomial::error)
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let mut physical = None;
        for (conservation, constitutive) in [(&rows[0], &rows[1]), (&rows[1], &rows[0])] {
            if let (Some(production), Some(diffusivity)) = (
                conservation_coefficient(program, conservation, flux),
                constitutive_coefficient(program, constitutive, concentration, flux),
            ) {
                physical = Some((production, diffusivity));
            }
        }
        let (production, diffusivity) = physical.ok_or_else(|| invalid("radial relations must retain (r^2*j)'=q*r^2 and j=-D*c' with fixed q and positive D"))?;
        Ok(Self {
            concentration,
            flux,
            concentration_type,
            flux_type,
            diffusivity,
            production,
            surface,
        })
    }
}

fn supports(program: &KernelProgram, owner: RawId, kind: EdgeKind) -> Vec<RawId> {
    program
        .edges()
        .iter()
        .filter(|edge| {
            edge.from() == owner
                && edge.kind() == kind
                && matches!(program.node(edge.to()), Some(KernelNode::Domain(_)))
        })
        .map(|edge| edge.to())
        .collect()
}

fn endpoint(
    program: &KernelProgram,
    domain: Id<kinds::Domain>,
    relation: &RelationDef,
) -> Result<(Id<kinds::Field>, f64, BoundarySide, f64), Diagnostic> {
    let dag = relation.expression();
    let [left, right] = dag.roots() else {
        return Err(invalid("radial endpoint requires one equality"));
    };
    for (value, prescribed) in [(*left, *right), (*right, *left)] {
        let Some(ExprNode::Evaluate {
            value,
            at,
            side: Some(side),
        }) = dag.node(value)
        else {
            continue;
        };
        let Some(ExprNode::Symbol(SymbolRef::Field(field))) = dag.node(*value) else {
            continue;
        };
        let [(coordinate, point)] = at.as_slice() else {
            continue;
        };
        if !matches!(dag.node(*coordinate), Some(ExprNode::Symbol(SymbolRef::Coordinate {support,factor,axis:0})) if *support == domain && *factor == domain)
        {
            continue;
        }
        let point = polynomial::constant(program, &polynomial::normalize(domain, dag, *point)?)
            .ok_or_else(|| invalid("radial endpoint coordinate must be fixed"))?;
        let value = polynomial::constant(program, &polynomial::normalize(domain, dag, prescribed)?)
            .ok_or_else(|| invalid("radial endpoint value must be fixed"))?;
        return Ok((*field, point, *side, value));
    }
    Err(invalid(
        "radial endpoint must explicitly evaluate one exact Field and side",
    ))
}

fn conservation_coefficient(
    program: &KernelProgram,
    value: &Polynomial,
    flux: Id<kinds::Field>,
) -> Option<f64> {
    let x = Atom::Coordinate;
    let f = Atom::Field(flux.erase());
    let df = Atom::Partial(flux.erase());
    let scale = coefficient(value, &[x, x, df]);
    if scale.is_zero() || coefficient(value, &[x, f]) != scale.checked_add(scale).ok()? {
        return None;
    }
    let mut source = None;
    for (atoms, coefficient) in value.terms() {
        if atoms == [x, x, df] || atoms == [x, f] {
            continue;
        }
        let factors = atoms.strip_prefix(&[x, x])?;
        if source.is_some() {
            return None;
        }
        source = Some(polynomial::fixed_coefficient(
            program,
            factors,
            coefficient,
        )?);
    }
    let production = -source.unwrap_or(0.0) / scale.as_f64();
    production.is_finite().then_some(production)
}
fn constitutive_coefficient(
    program: &KernelProgram,
    value: &Polynomial,
    concentration: Id<kinds::Field>,
    flux: Id<kinds::Field>,
) -> Option<f64> {
    let f = Atom::Field(flux.erase());
    let dc = Atom::Partial(concentration.erase());
    let scale = coefficient(value, &[f]);
    if scale.is_zero() {
        return None;
    }
    let mut transport = None;
    for (atoms, coefficient) in value.terms() {
        if atoms == [f] {
            continue;
        }
        let factors = atoms.strip_suffix(&[dc])?;
        if transport.is_some() {
            return None;
        }
        transport = Some(polynomial::fixed_coefficient(
            program,
            factors,
            coefficient,
        )?);
    }
    let diffusivity = transport? / scale.as_f64();
    (diffusivity.is_finite() && diffusivity > 0.0).then_some(diffusivity)
}
