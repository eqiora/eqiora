use super::*;
use num_complex::ComplexFloat;

/// Explicit availability and action of normal, transpose and conjugate transpose.
/// A conjugate transpose uses the Euclidean pairing; no metric adjoint is implied.
pub trait OrientedLinearOperator: LinearOperator {
    /// Whether this exact action is available before numerical work begins.
    fn supports_orientation(&self, orientation: LinearOperatorOrientation) -> bool;

    /// Apply the requested mathematical orientation.
    ///
    /// # Errors
    /// Rejects unavailable orientations, incompatible buffers and nonfinite arithmetic.
    fn apply_oriented(
        &self,
        orientation: LinearOperatorOrientation,
        input: &[Self::Scalar],
        output: &mut [Self::Scalar],
    ) -> Result<(), Diagnostic>;
}

/// Allocation-free view of one explicitly admitted operator orientation.
#[derive(Debug, Clone, Copy)]
pub struct Oriented<'a, O: OrientedLinearOperator + ?Sized> {
    source: &'a O,
    orientation: LinearOperatorOrientation,
}

impl<'a, O: OrientedLinearOperator + ?Sized> Oriented<'a, O> {
    /// Borrow the exact source after checking its orientation capability.
    ///
    /// # Errors
    /// Rejects an unavailable action before a solve can begin.
    pub fn new(source: &'a O, orientation: LinearOperatorOrientation) -> Result<Self, Diagnostic> {
        if !source.supports_orientation(orientation) {
            return Err(Diagnostic::error(
                codes::INVALID_REALIZATION,
                "requested operator orientation is unavailable",
            ));
        }
        Ok(Self {
            source,
            orientation,
        })
    }

    /// Underlying source action.
    #[must_use]
    pub const fn source(self) -> &'a O {
        self.source
    }
}

impl<O: OrientedLinearOperator + ?Sized> LinearOperator for Oriented<'_, O>
where
    O::Scalar: ComplexFloat,
{
    type Scalar = O::Scalar;
    fn rows(&self) -> usize {
        if self.orientation == LinearOperatorOrientation::Normal {
            self.source.rows()
        } else {
            self.source.columns()
        }
    }
    fn columns(&self) -> usize {
        if self.orientation == LinearOperatorOrientation::Normal {
            self.source.columns()
        } else {
            self.source.rows()
        }
    }
    fn apply(&self, input: &[Self::Scalar], output: &mut [Self::Scalar]) -> Result<(), Diagnostic> {
        self.source.apply_oriented(self.orientation, input, output)
    }
    fn orientation(&self) -> LinearOperatorOrientation {
        self.orientation
    }
    fn row_action(&self) -> Option<&dyn RowLinearAction<Scalar = Self::Scalar>> {
        if self.orientation == LinearOperatorOrientation::Normal {
            self.source.row_action()
        } else {
            None
        }
    }
    fn diagonal(&self, output: &mut [Self::Scalar]) -> Result<DiagonalAvailability, Diagnostic> {
        let availability = self.source.diagonal(output)?;
        if availability == DiagonalAvailability::Available
            && self.orientation == LinearOperatorOrientation::ConjugateTransposed
        {
            for value in output {
                *value = value.conj();
            }
        }
        Ok(availability)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn adjoint_does_not_invent_an_unavailable_diagonal() {
        #[derive(Debug)]
        struct NoDiagonal;
        impl LinearOperator for NoDiagonal {
            type Scalar = f64;
            fn rows(&self) -> usize {
                1
            }
            fn columns(&self) -> usize {
                1
            }
            fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
                output.copy_from_slice(input);
                Ok(())
            }
        }
        impl OrientedLinearOperator for NoDiagonal {
            fn supports_orientation(&self, _: LinearOperatorOrientation) -> bool {
                true
            }
            fn apply_oriented(
                &self,
                _: LinearOperatorOrientation,
                input: &[f64],
                output: &mut [f64],
            ) -> Result<(), Diagnostic> {
                self.apply(input, output)
            }
        }
        let mut output = [7.];
        assert_eq!(
            Oriented::new(&NoDiagonal, LinearOperatorOrientation::ConjugateTransposed)
                .unwrap()
                .diagonal(&mut output)
                .unwrap(),
            DiagonalAvailability::Unavailable
        );
        assert_eq!(output, [7.]);
    }

    #[test]
    fn adjoint_diagonal_conjugates_only_an_available_source_diagonal() {
        use crate::{CanonicalCsrSystemView, CsrStorage, LinearOperatorProperties};
        #[derive(Debug)]
        struct Storage;
        impl CsrStorage<Complex64> for Storage {
            fn rows(&self) -> usize {
                1
            }
            fn columns(&self) -> usize {
                1
            }
            fn row_offsets(&self) -> &[usize] {
                &[0, 1]
            }
            fn column_indices(&self) -> &[usize] {
                &[0]
            }
            fn values(&self) -> &[Complex64] {
                const VALUES: [Complex64; 1] = [Complex64::new(2., 3.)];
                &VALUES
            }
        }
        let source = CanonicalCsrSystemView::new(
            &Storage,
            &[Complex64::new(0., 0.)],
            LinearOperatorProperties::General,
        )
        .unwrap();
        let mut diagonal = [Complex64::new(0., 0.)];
        for (orientation, expected) in [
            (LinearOperatorOrientation::Normal, Complex64::new(2., 3.)),
            (
                LinearOperatorOrientation::Transposed,
                Complex64::new(2., 3.),
            ),
            (
                LinearOperatorOrientation::ConjugateTransposed,
                Complex64::new(2., -3.),
            ),
        ] {
            assert_eq!(
                Oriented::new(&source, orientation)
                    .unwrap()
                    .diagonal(&mut diagonal)
                    .unwrap(),
                DiagonalAvailability::Available
            );
            assert_eq!(diagonal, [expected]);
        }
    }
}
