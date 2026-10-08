//! Finite actions use the same exact real/imaginary coefficient ring as spatial forms.
use super::*;

impl Context<'_> {
    pub(super) fn closed_component(
        &mut self,
        shape: &[u32],
        values: &[(f64, f64)],
        indices: &[usize],
    ) -> Option<Polynomial> {
        if shape.len() != indices.len() || shape.contains(&0) {
            return None;
        }
        let count = shape
            .iter()
            .try_fold(1usize, |n, extent| n.checked_mul(*extent as usize))?;
        if count != values.len() {
            return None;
        }
        let mut flat = 0usize;
        for (index, extent) in indices.iter().zip(shape) {
            if *index >= *extent as usize {
                return None;
            }
            flat = flat.checked_mul(*extent as usize)?.checked_add(*index)?;
        }
        let (real, imag) = *values.get(flat)?;
        Polynomial::complex(
            Polynomial::constant(number(real)?),
            Polynomial::constant(number(imag)?),
        )
    }

    pub(super) fn apply(
        &mut self,
        matrix: &E,
        vector: &E,
        row: usize,
        depth: usize,
    ) -> Option<Polynomial> {
        let shape = self.shape(matrix, depth + 1)?;
        let [rows, columns] = shape.as_slice() else {
            return None;
        };
        if row >= *rows || self.shape(vector, depth + 1)? != [*columns] {
            return None;
        }
        self.remaining = self.remaining.checked_sub(*columns)?;
        let mut sum = Polynomial::constant(ExactRational::integer(0));
        for column in 0..*columns {
            let term = self
                .tensor(matrix, row, column, depth + 1)?
                .checked_mul(&self.vector(vector, column, depth + 1)?)
                .ok()?;
            sum = sum.checked_add(&term).ok()?;
        }
        Some(sum)
    }
}

impl Context<'_> {
    pub(super) fn finite_residual_dimension(
        &mut self,
        left: &E,
        right: &E,
        test: eqiora_core::DimExponents,
    ) -> Option<eqiora_core::DimExponents> {
        self.finite_dimension(
            &E::Sub {
                left: Box::new(left.clone()),
                right: Box::new(right.clone()),
            },
            test,
            0,
        )
    }

    // DimExponents owns unit arithmetic. Numeric polynomial equality alone cannot
    // distinguish a closed coefficient 1 m from 1, so check this projection first.
    fn finite_dimension(
        &mut self,
        value: &E,
        test: eqiora_core::DimExponents,
        depth: usize,
    ) -> Option<eqiora_core::DimExponents> {
        use eqiora_core::DimExponents;
        self.step(depth)?;
        let field = self.field;
        let mut child = |value| self.finite_dimension(value, test, depth + 1);
        match value {
            E::Number { .. } => Some(DimExponents::DIMENSIONLESS),
            E::Rational { dimension, .. } | E::Components { dimension, .. } => {
                DimExponents::from_rationals(*dimension)
                    .filter(|value| value.exponents() == *dimension)
            }
            E::Field { ulid } | E::Parameter { ulid } => Some(self.symbols.get(ulid)?.dimension()),
            E::Test { field_ulid } if field_ulid == field => Some(test),
            E::Conjugate { value } | E::Neg { value } | E::Component { value, .. } => child(value),
            E::Complex {
                real: left,
                imag: right,
            }
            | E::Add { left, right }
            | E::Sub { left, right } => {
                let a = child(left)?;
                let b = child(right)?;
                if matches!(left.as_ref(), E::Number { value: 0. }) {
                    Some(b)
                } else if matches!(right.as_ref(), E::Number { value: 0. }) || a == b {
                    Some(a)
                } else {
                    None
                }
            }
            E::Mul { left, right }
            | E::Apply { left, right }
            | E::Inner { left, right }
            | E::Dot { left, right } => child(left)?.mul(child(right)?),
            E::Div { left, right } => child(left)?.div(child(right)?),
            E::Pow { base, exponent } => child(base)?.pow(*exponent, 1),
            _ => None,
        }
    }
}
