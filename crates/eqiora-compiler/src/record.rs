//! Closed record declarations. Record occurrences execute as ordinary typed leaves.
mod binding;
pub(crate) use binding::{BoundRecord, declarations, resolved_declarations};

pub(crate) mod parameters;
