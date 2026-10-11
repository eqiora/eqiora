use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id, ValueType};

/// Fixed finite ordinal set used by elaboration and nominal index values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexSetDef {
    id: Id<kinds::IndexSet>,
    extent: u32,
}
impl IndexSetDef {
    /// Construct a nonempty zero-based set with an exclusive bound.
    pub fn new(id: Id<kinds::IndexSet>, extent: u32) -> Result<Self, Diagnostic> {
        if extent == 0 {
            return Err(Diagnostic::error(
                codes::INVALID_KERNEL_DEFINITION,
                "index set extent must be positive",
            ));
        }
        Ok(Self { id, extent })
    }
    /// Exact declaration identity.
    #[must_use]
    pub const fn id(&self) -> Id<kinds::IndexSet> {
        self.id
    }
    /// Exclusive upper bound, fixed by elaboration.
    #[must_use]
    pub const fn extent(&self) -> u32 {
        self.extent
    }
    /// Nominal bounded scalar index type.
    #[must_use]
    pub fn value_type(&self) -> ValueType {
        ValueType::index(self.id, self.extent).expect("checked extent")
    }
}
impl From<IndexSetDef> for super::KernelNode {
    fn from(value: IndexSetDef) -> Self {
        Self::IndexSet(value)
    }
}
