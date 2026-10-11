//! Exact closed record declarations; occurrences execute through ordinary typed leaves.
mod binding;
pub(crate) use binding::{BoundRecord, declarations, resolved_declarations};

pub(crate) mod parameters;
