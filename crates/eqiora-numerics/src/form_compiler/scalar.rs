//! Private proof-carrying FEM derivation for the bounded Cartesian Q1 slice.

use std::collections::{BTreeMap, BTreeSet};

use eqiora_assembly::LocalContribution;
use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, RawId};
use eqiora_graph::EdgeKind;
use eqiora_meshing::{AffineGeometryMap, GeometryMap, QuadratureRule, ReferenceCellFamily};
use eqiora_schema::kernel::typing::TypedResidual;
use eqiora_schema::kernel::{
    ActivationKind, BoundarySide, DomainKind, ExprDag, ExprId, ExprNode, KernelNode,
    RepresentationKind, SymbolRef,
};
use eqiora_sem::KernelProgram;

use crate::canonical::{boundary_parent, lowering_error, relations_on};
use crate::discrete_space::{DiscreteSpace, HypercubeQ1Space};
use crate::form_compiler::vocabulary::{
    BoundarySource, FormulationKind, PrimalGalerkinCorrespondence, PrimalGalerkinSource,
};

mod authored;
mod recognition;
#[cfg(test)]
use crate::form_compiler::{
    DIVERGENCE_BY_PARTS, MatrixSlot, SOURCE_PAIRING, TEST_PAIRING, WeakSign, WeakTermSlot,
    ZERO_TEST_TRACE_DISCHARGE,
};
pub(crate) use authored::admit as admit_authored_scalar_primal_form;
use recognition::{recognize_essential_trace, recognize_volume, validate_source_expression};

const MAX_DAG_NODES: usize = 4_096;
const MAX_DERIVATIVE_ORDER: usize = 1;
const MAX_INTEGRAL_TERMS: usize = 8;
const MAX_QUADRATURE_POINTS: usize = 64;
const MAX_LOCAL_DOFS: usize = 32;
const MAX_TEMPORARIES: usize = 256;
const DERIVED_DERIVATIVE_ORDER: usize = 1;
const DERIVED_INTEGRAL_TERMS: usize = 2;
const Q1_TEMPORARIES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BoundaryRole {
    domain: RawId,
    relation: RawId,
    axis: usize,
    side: BoundarySide,
    trace_node: ExprId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DerivedScalarGalerkinForm {
    dimension: usize,
    domain: RawId,
    field: RawId,
    volume_relation: RawId,
    volume_nodes: VolumeNodes,
    boundary_roles: Vec<BoundaryRole>,
    parameters: Vec<RawId>,
    certificate: PrimalGalerkinCorrespondence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VolumeNodes {
    root: ExprId,
    divergence: ExprId,
    bilinear_flux: ExprId,
    divergence_sign: super::vocabulary::WeakSign,
    gradient: ExprId,
    source: ExprId,
}

#[derive(Debug, Clone, Copy)]
struct BoundaryNodes {
    trace: ExprId,
}

#[derive(Debug, Clone)]
struct QuadratureRecord {
    rule: QuadratureRule,
    declared_exactness: Option<usize>,
}

pub(crate) struct AdmittedScalarGalerkinForm<'a> {
    form: Option<&'a DerivedScalarGalerkinForm>,
    dimension: usize,
    quadrature: QuadratureRecord,
}

impl DerivedScalarGalerkinForm {
    #[cfg(test)]
    pub(super) const fn correspondence(&self) -> &PrimalGalerkinCorrespondence {
        &self.certificate
    }

    pub(crate) fn formulation_description(
        &self,
    ) -> (FormulationKind, &'static str, [&'static str; 4]) {
        (
            self.certificate.formulation.kind,
            self.certificate.formulation.boundary_treatment.id(),
            self.certificate
                .formulation
                .rules
                .map(crate::form_compiler::vocabulary::FormulationRule::id),
        )
    }

    pub(crate) fn admit_quadrature(
        &self,
        quadrature: &QuadratureRule,
    ) -> Result<AdmittedScalarGalerkinForm<'_>, Diagnostic> {
        let exactness = quadrature.polynomial_exactness().ok_or_else(|| {
            gate_error(
                self.volume_relation,
                "realization compatibility",
                "compiled Q1 source quadrature requires declared polynomial exactness",
            )
        })?;
        if quadrature.reference_cell().family() != ReferenceCellFamily::Hypercube
            || quadrature.reference_cell().dimension() != self.dimension
            || exactness < 3
            || quadrature.points().len() > MAX_QUADRATURE_POINTS
        {
            return Err(gate_error(
                self.volume_relation,
                "realization compatibility",
                format!(
                    "compiled Q1 requires a {}D hypercube rule with declared exactness >= 3 and at most {MAX_QUADRATURE_POINTS} points",
                    self.dimension
                ),
            ));
        }
        Ok(AdmittedScalarGalerkinForm {
            form: Some(self),
            dimension: self.dimension,
            quadrature: QuadratureRecord {
                rule: quadrature.clone(),
                declared_exactness: Some(exactness),
            },
        })
    }

    fn validate_certificate(&self) -> Result<(), Diagnostic> {
        let boundary_count = 2 * self.dimension;
        if self.boundary_roles.len() != boundary_count {
            return Err(certificate_error(
                self.volume_relation,
                format!(
                    "complete {}D essential-boundary discharge requires {boundary_count} boundaries",
                    self.dimension
                ),
            ));
        }
        let boundary_sources = self
            .boundary_roles
            .iter()
            .map(|boundary| BoundarySource {
                domain: boundary.domain,
                relation: boundary.relation,
                trace_node: boundary.trace_node,
            })
            .collect::<Vec<_>>();
        self.certificate
            .replay(PrimalGalerkinSource {
                domain: self.domain,
                unknown: self.field,
                volume_relation: self.volume_relation,
                root: self.volume_nodes.root,
                divergence: self.volume_nodes.divergence,
                divergence_sign: self.volume_nodes.divergence_sign,
                source: self.volume_nodes.source,
                boundaries: &boundary_sources,
            })
            .map_err(|message| certificate_error(self.volume_relation, message))
    }
}

pub(crate) fn compile_cartesian_q1_form(
    dimension: usize,
    quadrature: &QuadratureRule,
) -> Result<AdmittedScalarGalerkinForm<'static>, Diagnostic> {
    let space = HypercubeQ1Space::new(dimension)?;
    if quadrature.reference_cell().family() != ReferenceCellFamily::Hypercube
        || quadrature.reference_cell() != space.reference_cell()
    {
        return Err(Diagnostic::error(
            codes::INVALID_DISCRETIZATION,
            "compiled Cartesian Q1 form requires a dimension-matched hypercube quadrature rule",
        ));
    }
    Ok(AdmittedScalarGalerkinForm {
        form: None,
        dimension,
        quadrature: QuadratureRecord {
            rule: quadrature.clone(),
            declared_exactness: quadrature.polynomial_exactness(),
        },
    })
}

impl AdmittedScalarGalerkinForm<'_> {
    pub(crate) fn evaluate<K, S>(
        &self,
        geometry: &AffineGeometryMap,
        quadrature: &QuadratureRule,
        coefficient: &K,
        source: &S,
    ) -> Result<LocalContribution, Diagnostic>
    where
        K: Fn(&[f64]) -> f64 + ?Sized,
        S: Fn(&[f64]) -> f64 + ?Sized,
    {
        self.validate_realization(geometry, quadrature)?;
        super::region::integrate_scalar(self.dimension, geometry, quadrature, |point| {
            let diffusion = coefficient(point);
            if !diffusion.is_finite() || diffusion <= 0.0 {
                return Err(self.realization_error(
                    "compiled Q1 coefficient produced a non-positive or non-finite value",
                ));
            }
            let forcing = source(point);
            if !forcing.is_finite() {
                return Err(
                    self.realization_error("compiled Q1 source produced a non-finite value")
                );
            }
            Ok((diffusion, forcing))
        })
    }

    fn validate_realization(
        &self,
        geometry: &AffineGeometryMap,
        quadrature: &QuadratureRule,
    ) -> Result<(), Diagnostic> {
        let space = HypercubeQ1Space::new(self.dimension)?;
        if space.reference_cell() != self.quadrature.rule.reference_cell()
            || geometry.reference_cell() != self.quadrature.rule.reference_cell()
            || geometry.physical_dimension() != self.dimension
            || quadrature != &self.quadrature.rule
            || quadrature.polynomial_exactness() != self.quadrature.declared_exactness
            || self.form.is_some() && space.local_dofs().len() > MAX_LOCAL_DOFS
        {
            return Err(self.realization_error(
                "compiled Q1 space, affine geometry, bit DOF order, or quadrature record drifted",
            ));
        }
        Ok(())
    }

    fn realization_error(&self, message: impl Into<String>) -> Diagnostic {
        let message = message.into();
        if let Some(form) = self.form {
            gate_error(form.volume_relation, "realization compatibility", message)
        } else {
            Diagnostic::error(codes::INVALID_DISCRETIZATION, message)
        }
    }
}

#[cfg(test)]
pub(super) fn derive_candidate(
    program: &KernelProgram,
    domain: RawId,
) -> Result<Option<DerivedScalarGalerkinForm>, Diagnostic> {
    let Some(dimension) = cartesian_q1_dimension(program, domain) else {
        return Ok(None);
    };
    derive_candidate_with_dimension(program, domain, dimension)
}

pub(crate) fn derive_candidate_with_dimension(
    program: &KernelProgram,
    domain: RawId,
    dimension: usize,
) -> Result<Option<DerivedScalarGalerkinForm>, Diagnostic> {
    let Some(KernelNode::Domain(definition)) = program.node(domain) else {
        return Ok(None);
    };
    if !(1..=3).contains(&dimension)
        || (!matches!(definition.kind(), DomainKind::GeometryRegion { .. })
            && cartesian_q1_dimension(program, domain) != Some(dimension))
    {
        return Ok(None);
    }
    let roles = super::equation_roles::EquationRoles::derive(program, [domain])
        .map_err(|error| role_error(domain, error.message()))?;
    let principal = roles
        .relations
        .iter()
        .filter_map(|(id, entry)| match entry.kind {
            super::equation_roles::Role::Residual { tested } => Some((*id, tested)),
            _ => None,
        })
        .collect::<Vec<_>>();
    if principal.len() != 1 {
        return Err(role_error(
            domain,
            format!(
                "compiled Q1 requires exactly one principal scalar equation, found {}",
                principal.len()
            ),
        ));
    }
    let volume_relations = relations_on(program, domain);
    if volume_relations.len() != 1 {
        return Err(role_error(
            domain,
            format!(
                "compiled Q1 requires exactly one volume Relation, found {}",
                volume_relations.len()
            ),
        ));
    }
    let (volume_relation, field) = principal[0];
    let boundaries = boundary_inventory(program, domain, field, dimension)?;
    let Some(boundary_roles) = boundaries else {
        return Ok(None);
    };
    let typed = typed_relation(program, volume_relation)?;
    validate_expression(&typed, volume_relation, field)?;
    let volume = recognize_volume(typed.expression(), volume_relation, field)?;
    validate_source_expression(typed.expression(), volume.source, volume_relation)?;
    validate_static_bounds(typed.expression(), volume_relation, dimension)?;
    let parameters =
        validate_closed_roles(program, domain, field, volume_relation, &boundary_roles)?;
    let certificate = build_certificate(
        domain,
        field,
        volume_relation,
        volume,
        &boundary_roles,
        &parameters,
    )?;
    let form = DerivedScalarGalerkinForm {
        dimension,
        domain,
        field,
        volume_relation,
        volume_nodes: volume,
        boundary_roles,
        parameters,
        certificate,
    };
    form.validate_certificate()?;
    Ok(Some(form))
}

pub(crate) fn typed_relation(
    program: &KernelProgram,
    relation: RawId,
) -> Result<TypedResidual<RawId>, Diagnostic> {
    let relation = relation.downcast::<kinds::Relation>().ok_or_else(|| {
        gate_error(
            relation,
            "static semantics",
            "selected owner is not a Relation",
        )
    })?;
    program
        .typed_relation_residual(relation)
        .map_err(|diagnostics| {
            diagnostics.into_iter().next().unwrap_or_else(|| {
                gate_error(
                    relation.erase(),
                    "static semantics",
                    "typed residual replay failed without a diagnostic",
                )
            })
        })
}

fn cartesian_q1_dimension(program: &KernelProgram, domain: RawId) -> Option<usize> {
    let Some(KernelNode::Domain(definition)) = program.node(domain) else {
        return None;
    };
    if matches!(definition.kind(), DomainKind::GeometryRegion { .. }) {
        return Some(2);
    }
    if matches!(definition.kind(), DomainKind::CartesianBox { .. }) {
        let bounds = program.resolved_cartesian_bounds(definition.id()).ok()?;
        let dimensions = bounds.len();
        return (1..=3).contains(&dimensions).then_some(dimensions);
    }
    None
}

fn boundary_inventory(
    program: &KernelProgram,
    parent: RawId,
    field: RawId,
    dimension: usize,
) -> Result<Option<Vec<BoundaryRole>>, Diagnostic> {
    let mut by_side = BTreeMap::new();
    let geometry_backed = matches!(
        program.node(parent),
        Some(KernelNode::Domain(domain))
            if matches!(domain.kind(), DomainKind::GeometryRegion { .. })
    );
    let mut geometry_side = 0_usize;
    for node in program.nodes() {
        let KernelNode::Domain(domain) = node else {
            continue;
        };
        if boundary_parent(program, domain.id().erase()) != Some(parent) {
            continue;
        }
        let (axis, side) = match domain.kind() {
            DomainKind::CartesianBoundary { axis, side } => (*axis, *side),
            DomainKind::GeometryBoundary { .. } if geometry_backed => {
                let side = (
                    geometry_side / 2,
                    if geometry_side.is_multiple_of(2) {
                        BoundarySide::Lower
                    } else {
                        BoundarySide::Upper
                    },
                );
                geometry_side += 1;
                side
            }
            _ => continue,
        };
        let relations = relations_on(program, domain.id().erase());
        if relations.len() != 1 {
            return Err(role_error(
                domain.id().erase(),
                format!(
                    "compiled Q1 boundary requires exactly one Relation, found {}",
                    relations.len()
                ),
            ));
        }
        let relation = relations[0];
        let typed = typed_relation(program, relation)?;
        let Some(nodes) = recognize_essential_trace(typed.expression(), relation, field)? else {
            return Ok(None);
        };
        validate_expression(&typed, relation, field)?;
        let role = BoundaryRole {
            domain: domain.id().erase(),
            relation,
            axis,
            side,
            trace_node: nodes.trace,
        };
        if by_side.insert((axis, side), role).is_some() {
            return Err(role_error(
                domain.id().erase(),
                "compiled Q1 boundary side is ambiguous",
            ));
        }
    }
    let expected = (0..dimension)
        .flat_map(|axis| [(axis, BoundarySide::Lower), (axis, BoundarySide::Upper)])
        .collect::<Vec<_>>();
    if by_side.len() != expected.len() || expected.iter().any(|side| !by_side.contains_key(side)) {
        return Err(role_error(
            parent,
            format!("compiled Q1 requires one essential Relation on every {dimension}D box side"),
        ));
    }
    Ok(Some(expected.iter().map(|side| by_side[side]).collect()))
}

fn validate_expression(
    typed: &TypedResidual<RawId>,
    owner: RawId,
    field: RawId,
) -> Result<(), Diagnostic> {
    let expression = typed.expression();
    if expression.nodes().len() > MAX_DAG_NODES {
        return Err(bounds_error(
            owner,
            format!(
                "expression has {} nodes, exceeding {MAX_DAG_NODES}",
                expression.nodes().len()
            ),
        ));
    }
    for node in expression.nodes() {
        let admitted = match node {
            ExprNode::Constant(_)
            | ExprNode::Neg(_)
            | ExprNode::Add(_, _)
            | ExprNode::Sub(_, _)
            | ExprNode::Mul(_, _)
            | ExprNode::PowI(_, _)
            | ExprNode::SpatialCoordinate(_)
            | ExprNode::UnaryMath(eqiora_schema::kernel::UnaryMathFunction::Sin, _)
            | ExprNode::Gradient(_)
            | ExprNode::Divergence(_)
            | ExprNode::Trace(_) => true,
            ExprNode::Symbol(SymbolRef::Field(id)) => id.erase() == field,
            ExprNode::Symbol(SymbolRef::Parameter(_)) => true,
            _ => false,
        };
        if !admitted {
            return Err(gate_error(
                owner,
                "role assignment",
                "residual contains a symbol or expression kind outside the closed compiler subset",
            ));
        }
    }
    require_closed_dag(expression, owner)
}

pub(super) fn require_closed_dag(expression: &ExprDag, owner: RawId) -> Result<(), Diagnostic> {
    if expression.roots().len() != 1 {
        return Err(certificate_error(owner, "compiled Q1 requires one root"));
    }
    let mut reached = vec![false; expression.nodes().len()];
    let mut pending = vec![expression.roots()[0]];
    while let Some(value) = pending.pop() {
        let index = usize::try_from(value.index())
            .expect("ExprDag identities are portable platform indices");
        if reached[index] {
            continue;
        }
        reached[index] = true;
        push_operands(
            expression
                .node(value)
                .expect("ExprDag owns every referenced node"),
            &mut pending,
        );
    }
    // The shared numerical projection preserves authored node identities. Its
    // exact typed-zero elision can therefore leave inert RHS storage outside
    // the residual root. This is not an omitted strong-form operation: every
    // nonzero literal, symbol and operator must still be consumed.
    if expression
        .nodes()
        .iter()
        .zip(reached)
        .any(|(node, reached)| {
            !reached
                && !matches!(node, ExprNode::Constant(value)
                if value.is_zero()
                    && matches!(value.value_type().scalar_domain(),
                        eqiora_core::ScalarDomain::Real | eqiora_core::ScalarDomain::Complex))
        })
    {
        return Err(certificate_error(
            owner,
            "expression contains an unconsumed strong-form node",
        ));
    }
    Ok(())
}

pub(super) fn push_operands(node: &ExprNode, pending: &mut Vec<ExprId>) {
    match node {
        ExprNode::Neg(value)
        | ExprNode::PowI(value, _)
        | ExprNode::UnaryMath(_, value)
        | ExprNode::Gradient(value)
        | ExprNode::Divergence(value)
        | ExprNode::SymmetricPart(value)
        | ExprNode::IsotropicLift(value)
        | ExprNode::NormalComponent(value)
        | ExprNode::Trace(value) => pending.push(*value),
        ExprNode::Add(left, right)
        | ExprNode::Sub(left, right)
        | ExprNode::Mul(left, right)
        | ExprNode::Div(left, right) => {
            pending.push(*right);
            pending.push(*left);
        }
        _ => {}
    }
}

fn validate_static_bounds(
    expression: &ExprDag,
    owner: RawId,
    dimension: usize,
) -> Result<(), Diagnostic> {
    if DERIVED_DERIVATIVE_ORDER > MAX_DERIVATIVE_ORDER
        || DERIVED_INTEGRAL_TERMS > MAX_INTEGRAL_TERMS
        || Q1_TEMPORARIES > MAX_TEMPORARIES
        || expression.nodes().len() > MAX_DAG_NODES
    {
        return Err(bounds_error(
            owner,
            "derived Q1 work exceeds one or more frozen compiler bounds",
        ));
    }
    let dofs = HypercubeQ1Space::new(dimension)?.local_dofs().len();
    if dofs > MAX_LOCAL_DOFS {
        return Err(bounds_error(
            owner,
            format!("Q1 local DOFs exceed {MAX_LOCAL_DOFS}"),
        ));
    }
    Ok(())
}

fn validate_closed_roles(
    program: &KernelProgram,
    domain: RawId,
    field: RawId,
    volume_relation: RawId,
    boundaries: &[BoundaryRole],
) -> Result<Vec<RawId>, Diagnostic> {
    let relations = std::iter::once(volume_relation)
        .chain(boundaries.iter().map(|role| role.relation))
        .collect::<BTreeSet<_>>();
    let scoped_fields = program
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Field(value)
                if program.edges().iter().any(|edge| {
                    edge.from() == value.id().erase()
                        && edge.to() == domain
                        && edge.kind() == EdgeKind::DefinedOn
                }) =>
            {
                Some(value.id().erase())
            }
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    if scoped_fields != BTreeSet::from([field]) {
        return Err(role_error(
            domain,
            "compiled Q1 scope must contain exactly the admitted scalar unknown",
        ));
    }
    let representations = representations_of(program, field);
    if representations.len() != 1
        || !matches!(
            program.node(*representations.first().expect("length checked above")),
            Some(KernelNode::Representation(value))
                if value.kind() == RepresentationKind::Continuum
        )
    {
        return Err(role_error(
            field,
            "compiled Q1 unknown requires exactly one continuum Representation",
        ));
    }
    let boundary_domains = program
        .edges()
        .iter()
        .filter(|edge| edge.kind() == EdgeKind::BoundaryOf && edge.to() == domain)
        .map(|edge| edge.from())
        .collect::<BTreeSet<_>>();
    let admitted_boundary_domains = boundaries
        .iter()
        .map(|boundary| boundary.domain)
        .collect::<BTreeSet<_>>();
    if boundary_domains != admitted_boundary_domains {
        return Err(role_error(
            domain,
            "compiled Q1 boundary inventory leaves a reachable Domain unconsumed",
        ));
    }
    let activations = continuous_activations(program, &relations)?;
    let parameters = relation_parameters(program, &relations)?;
    for relation in &relations {
        validate_relation_dependencies(program, *relation)?;
    }
    debug_assert_eq!(activations.len(), relations.len());
    Ok(parameters.into_iter().collect())
}

fn representations_of(program: &KernelProgram, field: RawId) -> BTreeSet<RawId> {
    program
        .edges()
        .iter()
        .filter(|edge| edge.from() == field && edge.kind() == EdgeKind::DefinedOn)
        .filter_map(|edge| match program.node(edge.to()) {
            Some(KernelNode::Representation(_)) => Some(edge.to()),
            _ => None,
        })
        .collect()
}

fn validate_relation_dependencies(
    program: &KernelProgram,
    relation: RawId,
) -> Result<(), Diagnostic> {
    let typed = typed_relation(program, relation)?;
    let expression_symbols = typed
        .expression()
        .nodes()
        .iter()
        .filter_map(|node| match node {
            ExprNode::Symbol(SymbolRef::Field(value)) => Some(value.erase()),
            ExprNode::Symbol(SymbolRef::Parameter(value)) => Some(value.erase()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    let graph_dependencies = program
        .edges()
        .iter()
        .filter(|edge| edge.from() == relation && edge.kind() == EdgeKind::DependsOn)
        .map(|edge| edge.to())
        .collect::<BTreeSet<_>>();
    if expression_symbols != graph_dependencies {
        return Err(role_error(
            relation,
            "Relation dependency inventory does not exactly match its admitted symbols",
        ));
    }
    Ok(())
}

pub(super) fn continuous_activations(
    program: &KernelProgram,
    relations: &BTreeSet<RawId>,
) -> Result<BTreeSet<RawId>, Diagnostic> {
    let mut activations = BTreeSet::new();
    for relation in relations {
        let selected = program
            .edges()
            .iter()
            .filter(|edge| edge.kind() == EdgeKind::Activates && edge.to() == *relation)
            .filter_map(|edge| match program.node(edge.from()) {
                Some(KernelNode::Activation(value)) => Some(value),
                _ => None,
            })
            .collect::<Vec<_>>();
        if selected.len() != 1
            || !matches!(selected[0].kind(), ActivationKind::Continuous)
            || !activations.insert(selected[0].id().erase())
        {
            return Err(role_error(
                *relation,
                "compiled Q1 requires one distinct continuous Activation per Relation",
            ));
        }
    }
    Ok(activations)
}

fn relation_parameters(
    program: &KernelProgram,
    relations: &BTreeSet<RawId>,
) -> Result<BTreeSet<RawId>, Diagnostic> {
    let mut parameters = BTreeSet::new();
    for relation in relations {
        let typed = typed_relation(program, *relation)?;
        parameters.extend(typed.expression().nodes().iter().filter_map(|node| {
            if let ExprNode::Symbol(SymbolRef::Parameter(parameter)) = node {
                Some(parameter.erase())
            } else {
                None
            }
        }));
    }
    Ok(parameters)
}

fn build_certificate(
    domain: RawId,
    field: RawId,
    volume_relation: RawId,
    volume: VolumeNodes,
    boundaries: &[BoundaryRole],
    _parameters: &[RawId],
) -> Result<PrimalGalerkinCorrespondence, Diagnostic> {
    if volume.gradient == volume.source
        || boundaries
            .iter()
            .any(|role| role.relation == volume_relation)
    {
        return Err(certificate_error(
            volume_relation,
            "certificate node or relation identities overlap across weak terms",
        ));
    }
    let boundary_sources = boundaries
        .iter()
        .map(|boundary| BoundarySource {
            domain: boundary.domain,
            relation: boundary.relation,
            trace_node: boundary.trace_node,
        })
        .collect::<Vec<_>>();
    Ok(PrimalGalerkinCorrespondence::derive(PrimalGalerkinSource {
        domain,
        unknown: field,
        volume_relation,
        root: volume.root,
        divergence: volume.divergence,
        divergence_sign: volume.divergence_sign,
        source: volume.source,
        boundaries: &boundary_sources,
    }))
}

fn gate_error(owner: RawId, gate: &str, message: impl Into<String>) -> Diagnostic {
    lowering_error(
        owner,
        format!("FEM form compiler {gate} gate: {}", message.into()),
    )
}

fn role_error(owner: RawId, message: impl Into<String>) -> Diagnostic {
    gate_error(owner, "role assignment", message)
}

fn certificate_error(owner: RawId, message: impl Into<String>) -> Diagnostic {
    gate_error(owner, "derivation certificate", message)
}

fn bounds_error(owner: RawId, message: impl Into<String>) -> Diagnostic {
    gate_error(owner, "bounded compilation", message)
}

#[cfg(test)]
mod tests;
