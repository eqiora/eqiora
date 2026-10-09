//! Project the typed strong-law components into the same exact coefficient ring.
//! Pure tensor operations are expanded from their actual checked definitions.
use super::*;
use eqiora_core::RawId;
use eqiora_schema::kernel::pure_operator::PureOperatorDefinition;
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
            ExprNode::NormalComponent(value) => {
                self.source_normal(typed, id, *value, coordinate, depth + 1)?
            }
            ExprNode::Trace(value) => {
                let boundary = self.boundary()?;
                (typed.node_type(id)?.support.as_ref() == self.domains.get(&boundary))
                    .then_some(())?;
                self.source_restricted(typed, *value, coordinate, depth + 1)?
            }
            ExprNode::Constant(value) => {
                let flat = coordinate
                    .iter()
                    .zip(ty.shape().extents())
                    .fold(0, |flat, (i, n)| flat * n.get() as usize + i);
                let (real, imaginary) = value.component(flat)?;
                Polynomial::complex(
                    Polynomial::constant(number(real)?),
                    Polynomial::constant(number(imaginary)?),
                )?
            }
            ExprNode::Symbol(SymbolRef::Field(field)) => {
                self.atom(Atom::Field(field.ulid().to_string(), coordinate.to_vec()))?
            }
            ExprNode::Symbol(SymbolRef::Parameter(parameter)) => self.atom(Atom::Parameter(
                parameter.ulid().to_string(),
                coordinate.to_vec(),
            ))?,
            ExprNode::Gradient(value) => {
                let ExprNode::Symbol(SymbolRef::Field(field)) = typed.expression().node(*value)?
                else {
                    return None;
                };
                self.atom(Atom::FieldGradient(
                    field.ulid().to_string(),
                    coordinate.to_vec(),
                ))?
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
                        .checked_add(
                            &self
                                .atom(Atom::FieldGradient(field.ulid().to_string(), vec![i, i]))?,
                        )
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
            ExprNode::Complex { real, imag } => Polynomial::complex(
                self.source(typed, *real, coordinate, depth + 1)?,
                self.source(typed, *imag, coordinate, depth + 1)?,
            )?,
            ExprNode::UnaryMath(eqiora_schema::kernel::UnaryMathFunction::Conj, value) => self
                .source(typed, *value, coordinate, depth + 1)?
                .conjugate()
                .ok()?,
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
            ExprNode::FiniteBinary(
                eqiora_schema::kernel::FiniteBinaryOperation::Apply,
                left,
                right,
            ) => {
                let [row] = coordinate else {
                    return None;
                };
                let [columns] = typed.node_type(*right)?.shape().extents() else {
                    return None;
                };
                self.remaining = self.remaining.checked_sub(columns.get() as usize)?;
                let mut sum = Polynomial::constant(ExactRational::integer(0));
                for column in 0..columns.get() as usize {
                    let term = self
                        .source(typed, *left, &[*row, column], depth + 1)?
                        .checked_mul(&self.source(typed, *right, &[column], depth + 1)?)
                        .ok()?;
                    sum = sum.checked_add(&term).ok()?;
                }
                sum
            }
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
        self.pure_component(
            definition,
            coordinate,
            depth,
            |context, formal, indices, depth| {
                context.source(typed, *arguments.get(usize::from(formal))?, indices, depth)
            },
        )
    }
}
