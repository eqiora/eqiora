//! Declaration lookup produced alongside a compiled Model.
use eqiora_core::RawId;
use std::collections::BTreeMap;

/// Source-name to Semantic Kernel ID map produced with one compiled model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSymbols {
    symbols: BTreeMap<String, RawId>,
}

impl ModelSymbols {
    /// Resolve one source declaration name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<RawId> {
        self.symbols.get(name).copied()
    }

    /// Names and IDs in deterministic lexical order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&str, RawId)> {
        self.symbols.iter().map(|(name, id)| (name.as_str(), *id))
    }

    pub(crate) fn from_map(symbols: BTreeMap<String, RawId>) -> Self {
        Self { symbols }
    }
}
