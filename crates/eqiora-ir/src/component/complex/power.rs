use super::*;

impl<I: Clone + Eq> ComponentDagLowering<'_, I> {
    pub(super) fn lower_complex_power(
        &mut self,
        base: ExprId,
        exponent: i32,
        component: &[u32],
        part: ScalarPart,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        let zero = self.builder.constant(DynQuantity::new(
            0.,
            eqiora_core::DimExponents::DIMENSIONLESS,
        ))?;
        let one = self.builder.constant(DynQuantity::new(
            1.,
            eqiora_core::DimExponents::DIMENSIONLESS,
        ))?;
        let mut value = [
            self.lower_shaped_part(base, component, ScalarPart::Real)?,
            self.lower_shaped_part(base, component, ScalarPart::Imaginary)?,
        ];
        // Preserve the common eager operand semantics even for the zero power.
        // An invalid base must not become valid merely because its result is 1.
        if exponent == 0 {
            return Ok(if part == ScalarPart::Real { one } else { zero });
        }
        // Invert before exponentiation, preserving representable reciprocals
        // when the positive power would overflow. The common division kernel
        // handles coefficient scaling and rejects zero/nonfinite divisors.
        if exponent < 0 {
            let operands = [one, zero, value[0], value[1]];
            value = [
                self.builder.complex_div(operands, false)?,
                self.builder.complex_div(operands, true)?,
            ];
        }
        let mut count = exponent.unsigned_abs();
        let mut result = [one, zero];
        while count != 0 {
            if count % 2 == 1 {
                result = self.multiply_complex_parts(result, value)?;
            }
            count /= 2;
            if count != 0 {
                value = self.multiply_complex_parts(value, value)?;
            }
        }
        Ok(result[usize::from(part == ScalarPart::Imaginary)])
    }

    fn multiply_complex_parts(
        &mut self,
        [ar, ai]: [ScalarInputValueId; 2],
        [br, bi]: [ScalarInputValueId; 2],
    ) -> Result<[ScalarInputValueId; 2], Diagnostic> {
        let rr = self.builder.mul(ar, br)?;
        let ii = self.builder.mul(ai, bi)?;
        let ri = self.builder.mul(ar, bi)?;
        let ir = self.builder.mul(ai, br)?;
        Ok([self.builder.sub(rr, ii)?, self.builder.add(ri, ir)?])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{DimExponents, Id, ValueType};
    use eqiora_schema::kernel::{ExprDagBuilder, typing::RootContract};

    fn evaluate(exponent: i32, value: [f64; 2]) -> Result<Vec<f64>, Diagnostic> {
        let symbol = SymbolRef::Field(Id::new());
        let ty = ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS).unwrap();
        let mut dag = ExprDagBuilder::new();
        let z = dag.symbol(symbol).unwrap();
        let power = dag.powi(z, exponent).unwrap();
        let typed = TypedResidual::infer(
            dag.finish([power]).unwrap(),
            None,
            RootContract::ComponentwiseResidual,
            |_| Ok::<_, ()>(ExpressionType::<()>::new(ty.clone(), None)),
        )
        .unwrap();
        ComponentScalarization::lower(&typed)?
            .evaluate(|coordinate| Some(value[usize::from(coordinate.is_imaginary())]))
    }

    #[test]
    fn zero_complex_power_does_not_hide_an_invalid_base() {
        let symbol = SymbolRef::Field(Id::new());
        let ty = ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS).unwrap();
        let mut dag = ExprDagBuilder::new();
        let z = dag.symbol(symbol).unwrap();
        let quotient = dag.div(z, z).unwrap();
        let power = dag.powi(quotient, 0).unwrap();
        let typed = TypedResidual::infer(
            dag.finish([power]).unwrap(),
            None,
            RootContract::ComponentwiseResidual,
            |_| Ok::<_, ()>(ExpressionType::<()>::new(ty.clone(), None)),
        )
        .unwrap();
        let lowering = ComponentScalarization::lower(&typed).unwrap();
        assert!(lowering.evaluate(|_| Some(0.)).is_err());
        assert_eq!(
            lowering
                .evaluate(|coordinate| Some(if coordinate.is_imaginary() { 0. } else { 1. }))
                .unwrap(),
            [1., 0.]
        );
    }

    #[test]
    fn complex_integer_powers_preserve_full_exponent_domain_and_reciprocal() {
        for (exponent, expected) in [
            (0, [1., 0.]),
            (1, [1., 2.]),
            (2, [-3., 4.]),
            (3, [-11., -2.]),
            (-1, [0.2, -0.4]),
            (-2, [-0.12, -0.16]),
        ] {
            let actual = evaluate(exponent, [1., 2.]).unwrap();
            for (value, expected) in actual.iter().zip(expected) {
                assert!((value - expected).abs() < 1e-14);
            }
        }
        assert_eq!(evaluate(i32::MIN, [0., 1.]).unwrap(), [1., 0.]);
        assert_eq!(evaluate(i32::MAX, [0., 1.]).unwrap(), [0., -1.]);
        assert!(evaluate(-1, [0., 0.]).is_err());
    }
}
