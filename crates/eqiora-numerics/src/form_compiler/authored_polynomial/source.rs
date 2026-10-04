//! Project the typed strong-law components into the same exact coefficient ring.
//! Pure tensor operations are expanded from their actual checked definitions.
use super::*;
use eqiora_core::RawId;
use eqiora_schema::kernel::pure_operator::{CalculusNode, PureOperatorDefinition};
use eqiora_schema::kernel::typing::TypedResidual;
use eqiora_schema::kernel::{ExprId, ExprNode, SymbolRef};

impl Context<'_> {
    pub(super) fn source(
        &mut self,
        typed: &TypedResidual<RawId>,
        id: ExprId,
        coordinate: &[usize],
        depth: usize,
    ) -> Option<Polynomial> {
        self.step(depth)?;
        let ty = typed.node_type(id)?;
        if ty.value_type.array_rank() != 0
            || coordinate.len() != ty.shape().rank()
            || coordinate
                .iter()
                .zip(ty.shape().extents())
                .any(|(i, n)| *i >= n.get() as usize)
        {
            return None;
        }
        Some(match typed.expression().node(id)? {
            ExprNode::Constant(value) => {
                let flat = coordinate
                    .iter()
                    .zip(ty.shape().extents())
                    .fold(0, |flat, (i, n)| flat * n.get() as usize + i);
                let (real, imaginary) = value.component(flat)?;
                if imaginary != 0.0 {
                    return None;
                }
                Polynomial::constant(number(real)?)
            }
            ExprNode::Symbol(SymbolRef::Field(field)) => {
                Polynomial::atom(Atom::Field(field.ulid().to_string(), coordinate.to_vec()))
            }
            ExprNode::Symbol(SymbolRef::Parameter(parameter)) => Polynomial::atom(Atom::Parameter(
                parameter.ulid().to_string(),
                coordinate.to_vec(),
            )),
            ExprNode::Gradient(value) => {
                let ExprNode::Symbol(SymbolRef::Field(field)) = typed.expression().node(*value)?
                else {
                    return None;
                };
                Polynomial::atom(Atom::FieldGradient(
                    field.ulid().to_string(),
                    coordinate.to_vec(),
                ))
            }
            ExprNode::Divergence(value) if coordinate.is_empty() => {
                let ExprNode::Symbol(SymbolRef::Field(field)) = typed.expression().node(*value)?
                else {
                    return None;
                };
                let [extent] = typed.node_type(*value)?.shape().extents() else {
                    return None;
                };
                let mut sum = Polynomial::constant(ExactRational::integer(0));
                for i in 0..extent.get() as usize {
                    sum = sum
                        .checked_add(&Polynomial::atom(Atom::FieldGradient(
                            field.ulid().to_string(),
                            vec![i, i],
                        )))
                        .ok()?;
                }
                sum
            }
            ExprNode::Symbol(SymbolRef::Coordinate {
                support,
                factor,
                axis,
            }) => Polynomial::atom(Atom::Coordinate(
                support.ulid().to_string(),
                factor.ulid().to_string(),
                *axis,
            )),
            ExprNode::Neg(value) => self
                .source(typed, *value, coordinate, depth + 1)?
                .checked_neg()
                .ok()?,
            ExprNode::Add(left, right) => self
                .source(typed, *left, coordinate, depth + 1)?
                .checked_add(&self.source(typed, *right, coordinate, depth + 1)?)
                .ok()?,
            ExprNode::Sub(left, right) => self
                .source(typed, *left, coordinate, depth + 1)?
                .checked_add(
                    &self
                        .source(typed, *right, coordinate, depth + 1)?
                        .checked_neg()
                        .ok()?,
                )
                .ok()?,
            ExprNode::Mul(left, right) => {
                let a = if typed.node_type(*left)?.shape().is_scalar() {
                    &[][..]
                } else {
                    coordinate
                };
                let b = if typed.node_type(*right)?.shape().is_scalar() {
                    &[][..]
                } else {
                    coordinate
                };
                self.source(typed, *left, a, depth + 1)?
                    .checked_mul(&self.source(typed, *right, b, depth + 1)?)
                    .ok()?
            }
            ExprNode::SymmetricPart(value) => self.source_pure(
                typed,
                &PureOperatorDefinition::symmetric_part().ok()?,
                &[*value],
                coordinate,
                depth + 1,
            )?,
            ExprNode::IsotropicLift(value) => self.source_pure(
                typed,
                &PureOperatorDefinition::isotropic_lift().ok()?,
                &[*value],
                coordinate,
                depth + 1,
            )?,
            ExprNode::PureOperatorApplication(application) => self.source_pure(
                typed,
                typed.expression().definition(application.definition())?,
                application.arguments(),
                coordinate,
                depth + 1,
            )?,
            _ => return None,
        })
    }

    fn source_pure(
        &mut self,
        typed: &TypedResidual<RawId>,
        definition: &PureOperatorDefinition,
        arguments: &[ExprId],
        coordinate: &[usize],
        depth: usize,
    ) -> Option<Polynomial> {
        let types = arguments
            .iter()
            .map(|id| typed.node_type(*id).cloned())
            .collect::<Option<Vec<_>>>()?;
        definition.instantiate(&types).ok()?;
        self.remaining = self.remaining.checked_sub(definition.nodes().len())?;
        let coordinate = coordinate
            .iter()
            .map(|i| u32::try_from(*i).ok())
            .collect::<Option<Vec<_>>>()?;
        let mut mapped: Vec<Polynomial> = Vec::new();
        for node in definition.nodes() {
            let value = match node {
                CalculusNode::FormalComponent { formal, axes } => {
                    let indices = axes
                        .iter()
                        .map(|axis| axis.resolve(&coordinate).ok().map(|i| i as usize))
                        .collect::<Option<Vec<_>>>()?;
                    self.source(
                        typed,
                        *arguments.get(usize::from(*formal))?,
                        &indices,
                        depth + 1,
                    )?
                }
                CalculusNode::Rational { value, .. } => Polynomial::constant(*value),
                CalculusNode::KroneckerDelta(a, b) => Polynomial::constant(ExactRational::integer(
                    i64::from(a.resolve(&coordinate).ok()? == b.resolve(&coordinate).ok()?),
                )),
                CalculusNode::Neg(a) => mapped[a.index() as usize].checked_neg().ok()?,
                CalculusNode::Add(a, b) => mapped[a.index() as usize]
                    .checked_add(&mapped[b.index() as usize])
                    .ok()?,
                CalculusNode::Mul(a, b) => mapped[a.index() as usize]
                    .checked_mul(&mapped[b.index() as usize])
                    .ok()?,
                _ => return None,
            };
            mapped.push(value);
        }
        mapped.get(definition.root().index() as usize).cloned()
    }
}
