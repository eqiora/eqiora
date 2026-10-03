//! Real coordinate lowering keeps complex parts attached to the original symbol.
use super::*;
use eqiora_core::ScalarDomain;
use eqiora_schema::kernel::UnaryMathFunction;

impl<I: Clone + Eq> ComponentDagLowering<'_, I> {
    fn is_complex(&self, value: ExprId) -> bool {
        self.node_types[value.index() as usize]
            .value_type
            .scalar_domain()
            == ScalarDomain::Complex
    }

    pub(super) fn lower_product_parts(
        &mut self,
        left: ExprId,
        left_component: &[u32],
        right: ExprId,
        right_component: &[u32],
        part: ScalarPart,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        use ScalarPart::{Imaginary, Real};
        if !self.is_complex(left) && !self.is_complex(right) {
            let a = self.lower_shaped_part(left, left_component, Real)?;
            let b = self.lower_shaped_part(right, right_component, Real)?;
            return self.builder.mul(a, b);
        }
        let ar = self.lower_shaped_part(left, left_component, Real)?;
        let ai = self.lower_shaped_part(left, left_component, Imaginary)?;
        let br = self.lower_shaped_part(right, right_component, Real)?;
        let bi = self.lower_shaped_part(right, right_component, Imaginary)?;
        if part == Real {
            let real = self.builder.mul(ar, br)?;
            let imaginary = self.builder.mul(ai, bi)?;
            self.builder.sub(real, imaginary)
        } else {
            let first = self.builder.mul(ar, bi)?;
            let second = self.builder.mul(ai, br)?;
            self.builder.add(first, second)
        }
    }

    pub(super) fn lower_complex(
        &mut self,
        node: &ExprNode,
        component: &[u32],
        part: ScalarPart,
    ) -> Result<Option<ScalarInputValueId>, Diagnostic> {
        use ScalarPart::{Imaginary, Real};
        let value = match *node {
            ExprNode::Complex { real, imag } => {
                self.lower_shaped_part(if part == Real { real } else { imag }, component, Real)?
            }
            ExprNode::Mul(left, right) if self.is_complex(left) || self.is_complex(right) => {
                self.lower_product_parts(left, component, right, component, part)?
            }
            ExprNode::UnaryMath(function, operand) => match function {
                UnaryMathFunction::Real | UnaryMathFunction::Imag => self.lower_shaped_part(
                    operand,
                    component,
                    if function == UnaryMathFunction::Real {
                        Real
                    } else {
                        Imaginary
                    },
                )?,
                UnaryMathFunction::Conj => {
                    let value = self.lower_shaped_part(operand, component, part)?;
                    if part == Imaginary {
                        self.builder.neg(value)?
                    } else {
                        value
                    }
                }
                UnaryMathFunction::Sin | UnaryMathFunction::Sqrt if !self.is_complex(operand) => {
                    let value = self.lower_shaped_part(operand, component, Real)?;
                    self.builder.unary_math(function, value)?
                }
                UnaryMathFunction::Abs2 => {
                    let real = self.lower_shaped_part(operand, component, Real)?;
                    let imaginary = self.lower_shaped_part(operand, component, Imaginary)?;
                    let real = self.builder.mul(real, real)?;
                    let imaginary = self.builder.mul(imaginary, imaginary)?;
                    self.builder.add(real, imaginary)?
                }
                _ => {
                    return Err(invalid_component_ir(
                        "component scalarization has no admitted analytic-function or magnitude lowering",
                    ));
                }
            },
            ExprNode::Div(left, right) if self.is_complex(left) || self.is_complex(right) => {
                return Err(invalid_component_ir(
                    "complex division is not admitted by component scalarization",
                ));
            }
            ExprNode::PowI(base, _) if self.is_complex(base) => {
                return Err(invalid_component_ir(
                    "complex powers are not admitted by component scalarization",
                ));
            }
            ExprNode::SymmetricPart(operand) | ExprNode::IsotropicLift(operand)
                if self.is_complex(operand) =>
            {
                return Err(invalid_component_ir(
                    "complex tensor calculus is not admitted by component scalarization",
                ));
            }
            ExprNode::PureOperatorApplication(ref application)
                if application
                    .arguments()
                    .iter()
                    .any(|argument| self.is_complex(*argument)) =>
            {
                return Err(invalid_component_ir(
                    "complex pure calculus is not admitted by component scalarization",
                ));
            }
            _ => return Ok(None),
        };
        Ok(Some(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{DimExponents, Id, ValueLiteral, ValueType, entity::kinds};
    use eqiora_schema::kernel::{ExprDagBuilder, typing::RootContract};

    #[test]
    fn complex_channels_and_real_projections_keep_each_original_coordinate() {
        let field = SymbolRef::Field(Id::<kinds::Field>::new());
        let unit = DimExponents::DIMENSIONLESS;
        let complex = ValueType::scalar(ScalarDomain::Complex, unit).unwrap();
        let channels = complex.clone().array(2).unwrap();
        let mut dag = ExprDagBuilder::new();
        let z = dag.symbol(field).unwrap();
        let factor = dag
            .constant(ValueLiteral::new(complex, [(1., -2.)]).unwrap())
            .unwrap();
        let product = dag.mul(z, factor).unwrap();
        let first = dag.index(z, 0).unwrap();
        let conjugate = dag.unary_math(UnaryMathFunction::Conj, first).unwrap();
        let selected = dag.index(product, 1).unwrap();
        let magnitude = dag.unary_math(UnaryMathFunction::Abs2, selected).unwrap();
        let real = dag.unary_math(UnaryMathFunction::Real, selected).unwrap();
        let imaginary = dag.unary_math(UnaryMathFunction::Imag, selected).unwrap();
        let typed = TypedResidual::infer(
            dag.finish([product, conjugate, magnitude, real, imaginary])
                .unwrap(),
            None,
            RootContract::ComponentwiseResidual,
            |_| Ok::<_, ()>(ExpressionType::<()>::new(channels.clone(), None)),
        )
        .unwrap();
        let lowering = ComponentScalarization::lower(&typed).unwrap();
        let actual = lowering
            .evaluate(|coordinate| {
                assert_eq!(coordinate.symbol(), field);
                match (coordinate.component_index(), coordinate.part) {
                    ([0], ScalarPart::Real) => Some(3.),
                    ([0], ScalarPart::Imaginary) => Some(4.),
                    ([1], ScalarPart::Real) => Some(-2.),
                    ([1], ScalarPart::Imaginary) => Some(1.),
                    _ => None,
                }
            })
            .unwrap();
        // (3+4i)(1-2i)=11-2i; (-2+i)(1-2i)=5i; |5i|²=25.
        assert_eq!(actual, [11., -2., 0., 5., 3., -4., 25., 0., 5.]);
        assert_eq!(lowering.rows()[0].root_index(), 0);
        assert_eq!(lowering.rows()[0].component_index(), [0]);
        assert_eq!(lowering.rows()[1].component_index(), [0]);
        assert_eq!(lowering.rows()[2].component_index(), [1]);
        assert_eq!(lowering.rows()[3].part, ScalarPart::Imaginary);
        assert_eq!(lowering.rows()[6].part, ScalarPart::Real);
        // Deliberately permute the requested columns: y_im, x_re, y_re, x_im.
        let selected = [
            (1, ScalarPart::Imaginary),
            (0, ScalarPart::Real),
            (1, ScalarPart::Real),
            (0, ScalarPart::Imaginary),
        ]
        .map(|(index, part)| {
            lowering
                .rows()
                .iter()
                .flat_map(|row| row.symbols())
                .find(|coordinate| {
                    coordinate.component_index() == [index] && coordinate.part == part
                })
                .unwrap()
                .clone()
        });
        for (row, expected) in lowering.rows()[..4].iter().zip([
            [0., 1., 0., 2.],
            [0., -2., 0., 1.],
            [2., 0., 1., 0.],
            [1., 0., -2., 0.],
        ]) {
            let affine = row.bind_affine(&selected, &[]).unwrap();
            assert_eq!(affine.selected_symbols(), selected);
            assert_eq!(affine.coefficients(), expected);
            assert_eq!(affine.offsets(), [0.]);
        }
        assert!(lowering.rows()[6].bind_affine(&selected, &[]).is_err());
        assert!(lowering.rows()[0].bind_affine(&selected[..1], &[]).is_err());
        assert!(
            lowering.rows()[0]
                .bind_affine(&[selected[1].clone(), selected[1].clone()], &[])
                .is_err()
        );
        assert!(
            lowering.rows()[0]
                .bind_affine(&selected, &[(selected[1].clone(), 3.)])
                .is_err()
        );
        let bindings = [(selected[1].clone(), 3.), (selected[3].clone(), 4.)];
        assert_eq!(
            lowering.rows()[0]
                .bind_affine(&[], &bindings)
                .unwrap()
                .offsets(),
            [11.]
        );
    }

    #[test]
    fn imaginary_projection_does_not_hide_invalid_real_arithmetic() {
        let real = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap();
        let mut dag = ExprDagBuilder::new();
        let one = dag
            .constant(ValueLiteral::from_real(real.clone(), 1.).unwrap())
            .unwrap();
        let zero = dag
            .constant(ValueLiteral::from_real(real, 0.).unwrap())
            .unwrap();
        let invalid = dag.div(one, zero).unwrap();
        let projected = dag.unary_math(UnaryMathFunction::Imag, invalid).unwrap();
        let typed = TypedResidual::infer(
            dag.finish([projected]).unwrap(),
            None,
            RootContract::ComponentwiseResidual,
            |_| -> Result<ExpressionType<()>, ()> { unreachable!() },
        )
        .unwrap();
        let lowered = ComponentScalarization::lower(&typed).unwrap();
        assert!(lowered.evaluate(|_| None).is_err());
    }

    #[test]
    fn complex_equality_requires_both_residuals_and_exact_real_embedding() {
        let field = SymbolRef::Field(Id::<kinds::Field>::new());
        let unit = DimExponents::DIMENSIONLESS;
        let complex = ValueType::scalar(ScalarDomain::Complex, unit).unwrap();
        let mut dag = ExprDagBuilder::new();
        let z = dag.symbol(field).unwrap();
        let real = dag
            .constant(
                ValueLiteral::from_real(ValueType::scalar(ScalarDomain::Real, unit).unwrap(), 3.)
                    .unwrap(),
            )
            .unwrap();
        let residual = dag.sub(z, real).unwrap();
        let typed = TypedResidual::infer(
            dag.finish([residual]).unwrap(),
            None,
            RootContract::ComponentwiseResidual,
            |_| Ok::<_, ()>(ExpressionType::<()>::new(complex.clone(), None)),
        )
        .unwrap();
        let lowering = ComponentScalarization::lower(&typed).unwrap();
        let residual = lowering
            .evaluate(|coordinate| {
                Some(match coordinate.part {
                    ScalarPart::Real => 3.,
                    ScalarPart::Imaginary => 4.,
                })
            })
            .unwrap();
        assert_eq!(residual, [0., 4.]);
    }
}
