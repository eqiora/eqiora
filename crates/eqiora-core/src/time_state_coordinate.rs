//! Source-preserving real coordinates for complete numeric time states.
use crate::{Id, entity::kinds};

/// One real coordinate of a complete Field value or its time derivative.
///
/// `component` is a row-major flat component of the Field's exact Model shape.
/// Real values have one real coordinate per component; complex values have a
/// real and an imaginary coordinate. The Model owns shape, domain, and units;
/// time lowering must validate this coordinate against that original type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TimeStateCoordinate {
    field: Id<kinds::Field>,
    derivative_order: u32,
    component: usize,
    imaginary: bool,
}

impl TimeStateCoordinate {
    /// Describe a source coordinate; the complete lowering proof checks
    /// uniqueness, component completeness, and lower derivative orders.
    #[must_use]
    pub const fn new(
        field: Id<kinds::Field>,
        derivative_order: u32,
        component: usize,
        imaginary: bool,
    ) -> Self {
        Self {
            field,
            derivative_order,
            component,
            imaginary,
        }
    }
    /// Original authored Field identity.
    #[must_use]
    pub const fn field(self) -> Id<kinds::Field> {
        self.field
    }
    /// Zero denotes the Field value, higher orders its physical time derivatives.
    #[must_use]
    pub const fn derivative_order(self) -> u32 {
        self.derivative_order
    }
    /// Row-major flat component within the Model-owned Field shape.
    #[must_use]
    pub const fn component(self) -> usize {
        self.component
    }
    /// Whether this selects the imaginary rather than the real scalar part.
    #[must_use]
    pub const fn is_imaginary(self) -> bool {
        self.imaginary
    }
}
