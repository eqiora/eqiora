use super::*;

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

impl<O: OrientedLinearOperator + ?Sized> LinearOperator for Oriented<'_, O> {
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
        if self.orientation == LinearOperatorOrientation::ConjugateTransposed {
            // An arbitrary scalar requires explicit conjugation by its owner.
            Ok(DiagonalAvailability::Unavailable)
        } else {
            self.source.diagonal(output)
        }
    }
}
