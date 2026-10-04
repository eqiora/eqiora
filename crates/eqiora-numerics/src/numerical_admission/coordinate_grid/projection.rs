//! Cell-integrated equality with a constant trial Field and polynomial prescribed density.
use super::*;
use eqiora_core::{ScalarDomain, ValueFrame, ValueType};
use eqiora_graph::EdgeKind;
use eqiora_ir::ScalarOperatorIr;
use eqiora_meshing::QuadratureRule;
use eqiora_schema::kernel::{
    ActivationKind, ExprDag, ExprNode, FieldRole, RelationConditionKind, SymbolRef,
};

#[derive(Debug, Clone, PartialEq)]
pub(in crate::numerical_admission) struct CellProjection {
    pub(in crate::numerical_admission) field: Id<kinds::Field>,
    pub(in crate::numerical_admission) value_type: ValueType,
    operator: ScalarOperatorIr,
    density_root: usize,
    inputs: Vec<Input>,
    source: CoordinateSource,
}

#[derive(Debug, Clone, PartialEq)]
enum Input {
    Constant(f64),
    Coordinate(usize),
}

impl CellProjection {
    pub(in crate::numerical_admission) fn lower(
        program: &KernelProgram,
        grid: &CoordinateGrid,
    ) -> Result<Self, Diagnostic> {
        grid.source.require_program(program)?;
        let mut fields = Vec::new();
        let mut relations = Vec::new();
        for node in program.nodes() {
            match node {
                KernelNode::Field(field) => fields.push(field),
                KernelNode::Relation(relation) => relations.push(relation),
                KernelNode::Parameter(_) | KernelNode::Observable(_) => {}
                KernelNode::Representation(representation)
                    if representation.kind()
                        == eqiora_schema::kernel::RepresentationKind::Continuum => {}
                KernelNode::Domain(domain)
                    if matches!(
                        domain.kind(),
                        DomainKind::CoordinateInterval { .. }
                            | DomainKind::CoordinateProduct { .. }
                    ) => {}
                KernelNode::Activation(activation)
                    if matches!(activation.kind(), ActivationKind::Continuous) => {}
                _ => {
                    return Err(invalid(
                        "coordinate cell projection rejects dynamic and physical semantic owners",
                    ));
                }
            }
        }
        let ([field], [relation]) = (fields.as_slice(), relations.as_slice()) else {
            return Err(invalid(
                "coordinate cell projection requires one Field and one Relation",
            ));
        };
        let ty = field.value_type();
        if field.role() != FieldRole::Variable
            || ty.scalar_domain() != ScalarDomain::Real
            || !ty.shape().is_scalar()
            || ty.array_rank() != 0
            || ty.frame() != ValueFrame::Invariant
        {
            return Err(invalid(
                "coordinate cell projection requires an invariant real scalar variable",
            ));
        }
        let domain = parse_domain(&grid.source.domain)?.erase();
        for (owner, kind) in [
            (field.id().erase(), EdgeKind::DefinedOn),
            (relation.id().erase(), EdgeKind::AppliesOn),
        ] {
            let supports = program
                .edges()
                .iter()
                .filter(|edge| {
                    edge.from() == owner
                        && edge.kind() == kind
                        && matches!(program.node(edge.to()), Some(KernelNode::Domain(_)))
                })
                .collect::<Vec<_>>();
            if supports.len() != 1 || supports[0].to() != domain {
                return Err(invalid(
                    "coordinate cell projection requires exact Field and Relation support",
                ));
            }
        }
        if relation.is_initial()
            || relation.conditions() != Some(&[RelationConditionKind::Equality][..])
        {
            return Err(invalid(
                "coordinate cell projection requires one continuous equality",
            ));
        }
        let expression = relation.expression();
        let roots = expression.roots();
        let is_field = |root: usize| {
            matches!(expression.nodes()[roots[root].index() as usize],
            ExprNode::Symbol(SymbolRef::Field(id)) if id == field.id())
        };
        let density_root = match (is_field(0), is_field(1)) {
            (true, false) => 1,
            (false, true) => 0,
            _ => {
                return Err(invalid(
                    "coordinate cell projection requires Field = prescribed polynomial density",
                ));
            }
        };
        require_polynomial(expression, roots[density_root], field.id(), grid)?;
        let operator = ScalarOperatorIr::lower(expression)?;
        let inputs = operator
            .symbols()
            .iter()
            .map(|symbol| match symbol {
                SymbolRef::Field(id) if *id == field.id() => Ok(Input::Constant(0.0)),
                SymbolRef::Parameter(id) => match program.node(id.erase()) {
                    Some(KernelNode::Parameter(parameter)) => parameter
                        .value()
                        .real_scalar_value()
                        .map(|value| Input::Constant(value.value()))
                        .ok_or_else(|| {
                            invalid("coordinate cell projection requires real scalar Parameters")
                        }),
                    _ => Err(invalid(
                        "coordinate projection Parameter is outside its Model",
                    )),
                },
                SymbolRef::Coordinate {
                    factor, axis: 0, ..
                } => grid
                    .source
                    .factors
                    .iter()
                    .position(|candidate| candidate.domain == factor.ulid().to_string())
                    .map(Input::Coordinate)
                    .ok_or_else(|| invalid("coordinate projection references a foreign factor")),
                _ => Err(invalid(
                    "coordinate projection requires exact factor coordinates",
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            field: field.id(),
            value_type: ty.clone(),
            operator,
            density_root,
            inputs,
            source: grid.source.clone(),
        })
    }

    pub(in crate::numerical_admission) fn cell_values(
        &self,
        grid: &CoordinateGrid,
    ) -> Result<Vec<f64>, Diagnostic> {
        if self.source != grid.source {
            return Err(invalid(
                "coordinate cell projection cannot change its exact factor source",
            ));
        }
        let mesh = grid.mesh.mesh();
        let dimension = grid.source.factors.len();
        let rule = QuadratureRule::tensor_product_gauss_legendre(dimension, 2)?;
        let count = mesh
            .entity_count(dimension)
            .ok_or_else(|| invalid("coordinate grid has no cells"))?;
        if count
            .checked_mul(rule.points().len())
            .and_then(|work| work.checked_mul(self.operator.instruction_count()))
            .filter(|work| *work <= 1_048_576)
            .is_none()
        {
            return Err(invalid(
                "coordinate cell projection exceeds 1048576 expression operations",
            ));
        }
        (0..count)
            .map(|cell| {
                let axes = grid.cell_axes(cell)?;
                let mut sum = 0.0;
                let mut correction = 0.0;
                for sample in rule.points() {
                    let inputs = self
                        .inputs
                        .iter()
                        .map(|input| match input {
                            Input::Constant(value) => *value,
                            Input::Coordinate(axis) => {
                                let bounds = axes[*axis].1;
                                let (lower, upper) =
                                    (bounds.lower().value(), bounds.upper().value());
                                lower + (sample.coordinates[*axis] + 1.0) * (upper - lower) * 0.5
                            }
                        })
                        .collect::<Vec<_>>();
                    // The affine cell Jacobian cancels against the cell measure. Each reference
                    // axis contributes 1/2; no Euclidean metric mixes unlike factor dimensions.
                    let weight = sample.weight * 0.5_f64.powi(dimension as i32);
                    let term = weight * self.operator.evaluate(&inputs)?[self.density_root];
                    let corrected = term - correction;
                    let next = sum + corrected;
                    correction = (next - sum) - corrected;
                    sum = next;
                }
                if !sum.is_finite() {
                    return Err(invalid("coordinate cell average must be finite"));
                }
                Ok(sum)
            })
            .collect()
    }
}

fn require_polynomial(
    expression: &ExprDag,
    root: eqiora_schema::kernel::ExprId,
    field: Id<kinds::Field>,
    grid: &CoordinateGrid,
) -> Result<(), Diagnostic> {
    if !expression.properties().is_empty()
        || expression.nodes().len() > 4096
        || grid.source.factors.len() > 3
    {
        return Err(invalid(
            "coordinate cell polynomial requires at most three factors and 4096 unannotated nodes",
        ));
    }
    let n = grid.source.factors.len();
    let mut degrees: Vec<(Vec<u16>, bool)> = Vec::new();
    for node in expression.nodes() {
        let at = |id: &eqiora_schema::kernel::ExprId| &degrees[id.index() as usize];
        let result = match node {
            ExprNode::Constant(value) if value.real_scalar_value().is_some() => (vec![0; n], false),
            ExprNode::Symbol(SymbolRef::Parameter(_)) => (vec![0; n], false),
            ExprNode::Symbol(SymbolRef::Field(id)) if *id == field => (vec![0; n], true),
            ExprNode::Symbol(SymbolRef::Coordinate {
                factor, axis: 0, ..
            }) => {
                let axis = grid
                    .source
                    .factors
                    .iter()
                    .position(|item| item.domain == factor.ulid().to_string())
                    .ok_or_else(|| invalid("coordinate polynomial references a foreign factor"))?;
                let mut degree = vec![0; n];
                degree[axis] = 1;
                (degree, false)
            }
            ExprNode::Neg(value) => at(value).clone(),
            ExprNode::Add(a, b) | ExprNode::Sub(a, b) | ExprNode::Mul(a, b) => {
                let (left, lf) = at(a);
                let (right, rf) = at(b);
                let degree = left
                    .iter()
                    .zip(right)
                    .map(|(a, b)| {
                        if matches!(node, ExprNode::Mul(..)) {
                            a.saturating_add(*b)
                        } else {
                            (*a).max(*b)
                        }
                    })
                    .collect();
                (degree, *lf || *rf)
            }
            ExprNode::Div(a, b) if at(b).0.iter().all(|degree| *degree == 0) && !at(b).1 => {
                at(a).clone()
            }
            ExprNode::PowI(value, power) if *power >= 0 && *power <= 3 => (
                at(value)
                    .0
                    .iter()
                    .map(|degree| degree.saturating_mul(*power as u16))
                    .collect(),
                at(value).1,
            ),
            _ => {
                return Err(invalid(
                    "coordinate cell density requires a polynomial of degree at most three per factor",
                ));
            }
        };
        if result.0.iter().any(|degree| *degree > 3) {
            return Err(invalid(
                "coordinate cell density exceeds degree three in a factor",
            ));
        }
        degrees.push(result);
    }
    if degrees[root.index() as usize].1 {
        return Err(invalid(
            "coordinate prescribed density cannot depend on an unknown Field",
        ));
    }
    Ok(())
}
