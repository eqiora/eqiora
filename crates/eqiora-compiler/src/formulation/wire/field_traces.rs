//! Live Field boundary operands reuse the Model's native trace proof.
use super::{AuthoredFormExpressionV1 as E, AuthoredTestRestriction, rejection};
use eqiora_core::{
    Diagnostic, DimExponents, Id, RawId, ScalarDomain, ValueLiteral, ValueType, entity::kinds,
};
use eqiora_schema::kernel::typing::{self, SpatialSupport};
use eqiora_schema::kernel::{ExprId, ExprNode, SpatialRegularity, SymbolRef, UnaryMathFunction};
mod native;
use native::{NativeTrace, Ty};

type Resolve<'a> = dyn FnMut(RawId) -> Result<(Ty, SpatialRegularity), Diagnostic> + 'a;
type Support<'a> = dyn FnMut(Id<kinds::Domain>) -> Result<SpatialSupport<RawId>, Diagnostic> + 'a;

struct Context<'a, 'm> {
    native: NativeTrace,
    resolve: &'a mut Resolve<'m>,
    support: &'a mut Support<'m>,
    tests: &'a [AuthoredTestRestriction],
    remaining: usize,
}

fn identity(text: &str) -> Result<ulid::Ulid, Diagnostic> {
    text.parse::<ulid::Ulid>()
        .ok()
        .filter(|id| id.to_string() == text)
        .ok_or_else(|| rejection("Field trace requires a canonical live identity"))
}
fn typed<T>(result: Result<T, impl std::fmt::Display>) -> Result<T, Diagnostic> {
    result.map_err(|error| rejection(&error.to_string()))
}

impl Context<'_, '_> {
    fn expression(&mut self, expression: &E, depth: usize) -> Result<ExprId, Diagnostic> {
        if depth > 96 || self.remaining == 0 {
            return Err(rejection(
                "Field trace expression exceeds the structural bound",
            ));
        }
        self.remaining -= 1;
        match expression {
            E::Number { value } => {
                let ty = typed(ValueType::scalar(
                    ScalarDomain::Real,
                    DimExponents::DIMENSIONLESS,
                ))?;
                let literal = typed(ValueLiteral::from_real(ty.clone(), *value))?;
                self.native
                    .operation(ExprNode::Constant(literal), Ty::new(ty, None))
            }
            E::Rational {
                numerator,
                denominator,
                dimension,
            } => {
                typed(
                    eqiora_schema::kernel::pure_operator::ExactRational::from_canonical_parts(
                        *numerator,
                        *denominator,
                    ),
                )?;
                let dimension = DimExponents::from_rationals(*dimension)
                    .ok_or_else(|| rejection("invalid coefficient dimension"))?;
                let ty = typed(ValueType::scalar(ScalarDomain::Real, dimension))?;
                let literal = typed(ValueLiteral::from_real(
                    ty.clone(),
                    *numerator as f64 / *denominator as f64,
                ))?;
                self.native
                    .operation(ExprNode::Constant(literal), Ty::new(ty, None))
            }
            E::Field { ulid } => {
                let id = Id::<kinds::Field>::from_ulid(identity(ulid)?);
                let (ty, assertion) = (self.resolve)(id.erase())?;
                self.native
                    .push(ExprNode::Symbol(SymbolRef::Field(id)), ty, assertion)
            }
            E::Coordinate {
                support_ulid,
                factor_ulid,
                axis,
            } => {
                let support = Id::<kinds::Domain>::from_ulid(identity(support_ulid)?);
                let factor = Id::<kinds::Domain>::from_ulid(identity(factor_ulid)?);
                let spatial = (self.support)(support)?;
                let ty = typed(Ty::coordinate(&factor.erase(), *axis, Some(&spatial)))?;
                self.native.operation(
                    ExprNode::Symbol(SymbolRef::Coordinate {
                        support,
                        factor,
                        axis: *axis,
                    }),
                    ty,
                )
            }
            E::Cross { left, right }
            | E::Dot { left, right }
            | E::Frobenius { left, right }
            | E::Inner { left, right } => {
                let mut left = self.expression(left, depth + 1)?;
                let right = self.expression(right, depth + 1)?;
                if matches!(expression, E::Inner { .. }) {
                    let ty = self.native.ty(left).clone();
                    left = self
                        .native
                        .operation(ExprNode::UnaryMath(UnaryMathFunction::Conj, left), ty)?;
                }
                let types = [self.native.ty(left).clone(), self.native.ty(right).clone()];
                if matches!(expression, E::Inner { .. })
                    && types.iter().all(|ty| ty.shape().is_scalar())
                {
                    let ty = typed(typing::multiply(&types[0], &types[1]))?;
                    return self.native.operation(ExprNode::Mul(left, right), ty);
                }
                let operation = if matches!(expression, E::Cross { .. }) {
                    crate::math::tensor::Operation::Cross
                } else {
                    let rank = u16::try_from(types[0].shape().rank())
                        .map_err(|_| rejection("trace contraction rank overflow"))?;
                    crate::math::tensor::Operation::Contract(
                        (0..rank).map(|axis| (axis, axis)).collect(),
                    )
                };
                let definition = typed(operation.definition(&types))?;
                self.native.pure(&definition, &[left, right])
            }
            E::CoordinatePartial { value, wrt } => {
                if !matches!(wrt.as_ref(), E::Coordinate { .. }) {
                    return Err(rejection(
                        "coordinate partial requires an exact coordinate selector",
                    ));
                }
                let value = self.expression(value, depth + 1)?;
                let wrt = self.expression(wrt, depth + 1)?;
                let ty = typed(
                    self.native
                        .ty(value)
                        .coordinate_partial(self.native.ty(wrt)),
                )?;
                self.native
                    .operation(ExprNode::CoordinatePartial { value, wrt }, ty)
            }
            E::Parameter { ulid } => {
                let id = Id::<kinds::Parameter>::from_ulid(identity(ulid)?);
                let (ty, assertion) = (self.resolve)(id.erase())?;
                self.native
                    .push(ExprNode::Symbol(SymbolRef::Parameter(id)), ty, assertion)
            }
            E::Test { field_ulid } | E::Direction { field_ulid, .. } => {
                let id = Id::<kinds::Field>::from_ulid(identity(field_ulid)?);
                let (mut ty, _) = (self.resolve)(id.erase())?;
                let (_, _, _, dimension, regularity) = self
                    .tests
                    .iter()
                    .find(|(_, trial, ..)| trial == field_ulid)
                    .ok_or_else(|| rejection("trace test has no declared binding"))?;
                let dimension = DimExponents::from_rationals(*dimension)
                    .ok_or_else(|| rejection("invalid test dimension"))?;
                ty.value_type = typed(ty.value_type.with_dimension(dimension))?;
                let assertion = match regularity.as_deref() {
                    Some("h1") => SpatialRegularity::H1,
                    Some("hdiv") => SpatialRegularity::HDiv,
                    Some("hcurl") => SpatialRegularity::HCurl,
                    Some("l2") => SpatialRegularity::L2,
                    _ => return Err(rejection("missing spatial test regularity")),
                };
                // This transient node carries the test's own type and hypothesis.
                // A Field occurrence gets a separate node resolved above from the Model.
                self.native
                    .push(ExprNode::Symbol(SymbolRef::Field(id)), ty, assertion)
            }
            E::Neg { value } | E::Conjugate { value } | E::Sin { value } => {
                let id = self.expression(value, depth + 1)?;
                let ty = self.native.ty(id).clone();
                let node = match expression {
                    E::Neg { .. } => ExprNode::Neg(id),
                    E::Conjugate { .. } => ExprNode::UnaryMath(UnaryMathFunction::Conj, id),
                    _ => ExprNode::UnaryMath(UnaryMathFunction::Sin, id),
                };
                let ty = if matches!(expression, E::Sin { .. }) {
                    typed(typing::unary_math(UnaryMathFunction::Sin, &ty))?
                } else {
                    ty
                };
                self.native.operation(node, ty)
            }
            E::Add { left, right }
            | E::Sub { left, right }
            | E::Mul { left, right }
            | E::Div { left, right }
            | E::Complex {
                real: left,
                imag: right,
            } => {
                let left = self.expression(left, depth + 1)?;
                let right = self.expression(right, depth + 1)?;
                let (a, b) = (self.native.ty(left), self.native.ty(right));
                let (node, ty) = match expression {
                    E::Add { .. } => (ExprNode::Add(left, right), typed(typing::additive(a, b))?),
                    E::Sub { .. } => (ExprNode::Sub(left, right), typed(typing::additive(a, b))?),
                    E::Mul { .. } => (ExprNode::Mul(left, right), typed(typing::multiply(a, b))?),
                    E::Div { .. } => (ExprNode::Div(left, right), typed(typing::divide(a, b))?),
                    _ => (
                        ExprNode::Complex {
                            real: left,
                            imag: right,
                        },
                        typed(a.clone().complex(b.clone()))?,
                    ),
                };
                self.native.operation(node, ty)
            }
            E::Pow { base, exponent } => {
                let id = self.expression(base, depth + 1)?;
                let ty = typed(typing::power(self.native.ty(id), *exponent))?;
                self.native.operation(ExprNode::PowI(id, *exponent), ty)
            }
            E::Curl { value } => {
                let id = self.expression(value, depth + 1)?;
                let (definition, gradient_type) =
                    typed(crate::math::oriented::Operation::Curl.definition(self.native.ty(id)))?;
                let gradient = self
                    .native
                    .operation(ExprNode::Gradient(id), gradient_type)?;
                self.native.pure(&definition, &[gradient])
            }
            E::Component { value, indices } => {
                let id = self.expression(value, depth + 1)?;
                let definition = typed(
                    crate::math::tensor::Operation::Component(indices.clone())
                        .definition(&[self.native.ty(id).clone()]),
                )?;
                self.native.pure(&definition, &[id])
            }
            E::Variation { value, .. } => self.expression(value, depth + 1),
            E::Gradient { value } | E::Divergence { value } | E::SymmetricPart { value } => {
                let id = self.expression(value, depth + 1)?;
                let ty = self.native.ty(id);
                let (node, ty) = match expression {
                    E::Gradient { .. } => (ExprNode::Gradient(id), typed(typing::gradient(ty))?),
                    E::Divergence { .. } => {
                        (ExprNode::Divergence(id), typed(typing::divergence(ty))?)
                    }
                    _ => (
                        ExprNode::SymmetricPart(id),
                        typed(typing::symmetric_part(ty))?,
                    ),
                };
                self.native.operation(node, ty)
            }
            E::Trace { value, on_ulid }
            | E::NormalTrace { value, on_ulid }
            | E::TangentialTrace { value, on_ulid } => {
                let mut id = self.expression(value, depth + 1)?;
                let on = Id::<kinds::Domain>::from_ulid(identity(on_ulid)?);
                let target = (self.support)(on)?;
                if matches!(expression, E::TangentialTrace { .. }) {
                    let (definition, _) = typed(
                        crate::math::oriented::Operation::TangentialTrace
                            .definition(self.native.ty(id)),
                    )?;
                    id = self.native.pure(&definition, &[id])?;
                }
                let (node, ty) = if matches!(expression, E::Trace { .. }) {
                    (
                        ExprNode::Trace { value: id, on },
                        typed(typing::trace(self.native.ty(id), Some(&target)))?,
                    )
                } else {
                    (
                        ExprNode::NormalComponent { value: id, on },
                        typed(typing::normal(self.native.ty(id), Some(&target)))?,
                    )
                };
                self.native.operation(node, ty)
            }
            E::Apply { .. } | E::LinearMap { .. } => Err(rejection(
                "global finite-map expressions cannot supply physical boundary traces",
            )),
            E::Integrate { .. } | E::IntervalIntegral { .. } | E::EndpointFlux { .. } => {
                Err(rejection(
                    "integrated or endpoint-flux expressions cannot be physical boundary operands",
                ))
            }
        }
    }
}

impl super::AuthoredFormulationProjection {
    /// Check Field-dependent boundary traces against the current Model hypotheses.
    ///
    /// Resolve actual Field/Parameter types and supports from the live Model. Test
    /// declarations remain separate. This checks boundary operands, not the validity
    /// of general distribution products or numerical correspondence of a weak form.
    /// # Errors
    /// Rejects missing live identities, unsupported boundary operands, and traces
    /// beyond the original Field's authored regularity.
    pub fn check_field_trace_regularity(
        &self,
        resolve: &mut Resolve<'_>,
        support: &mut Support<'_>,
    ) -> Result<(), Diagnostic> {
        let super::WireBinding::WeakTests { tests } = &self.wire.binding else {
            return Ok(());
        };
        let mut context = Context {
            native: NativeTrace::default(),
            resolve,
            support,
            tests,
            remaining: 65536,
        };
        for (_, left, right) in &self.wire.equations {
            context.scan(left, 0)?;
            context.scan(right, 0)?;
        }
        Ok(())
    }
}

impl Context<'_, '_> {
    fn scan(&mut self, value: &E, depth: usize) -> Result<bool, Diagnostic> {
        if depth > 96 || self.remaining == 0 {
            return Err(rejection(
                "Field trace expression exceeds the structural bound",
            ));
        }
        self.remaining -= 1;
        let mut field = matches!(value, E::Field { .. });
        for child in children(value) {
            field |= self.scan(child, depth + 1)?;
        }
        if field
            && matches!(
                value,
                E::Trace { .. } | E::NormalTrace { .. } | E::TangentialTrace { .. }
            )
        {
            let root = self.expression(value, 0)?;
            std::mem::take(&mut self.native).check(root)?;
        }
        Ok(field)
    }
}

fn children(value: &E) -> Vec<&E> {
    match value {
        E::Complex {
            real: left,
            imag: right,
        }
        | E::Add { left, right }
        | E::Sub { left, right }
        | E::Mul { left, right }
        | E::Div { left, right }
        | E::Apply { left, right }
        | E::Inner { left, right }
        | E::Cross { left, right }
        | E::Dot { left, right }
        | E::Frobenius { left, right }
        | E::CoordinatePartial {
            value: left,
            wrt: right,
        } => vec![left, right],
        E::Neg { value }
        | E::Trace { value, .. }
        | E::NormalTrace { value, .. }
        | E::Gradient { value }
        | E::Curl { value }
        | E::TangentialTrace { value, .. }
        | E::Divergence { value }
        | E::SymmetricPart { value }
        | E::Sin { value }
        | E::Conjugate { value }
        | E::Component { value, .. }
        | E::Variation { value, .. }
        | E::Integrate {
            integrand: value, ..
        }
        | E::IntervalIntegral {
            integrand: value, ..
        }
        | E::EndpointFlux { flux: value, .. }
        | E::Pow { base: value, .. } => vec![value],
        E::Field { .. }
        | E::Parameter { .. }
        | E::Test { .. }
        | E::Direction { .. }
        | E::Number { .. }
        | E::Rational { .. }
        | E::Coordinate { .. }
        | E::LinearMap { .. } => vec![],
    }
}
