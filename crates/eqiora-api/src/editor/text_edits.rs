use std::{borrow::Cow, ops::Range};

use super::{EditorPosition, EditorService, EditorSnapshot, codes, positions, stale_version};
use eqiora_core::Diagnostic;

impl EditorService {
    /// Apply ordered text changes atomically and analyze the final source once.
    /// Each pair contains an optional half-open UTF-16 range and replacement
    /// text. A missing range replaces the complete intermediate source.
    /// An empty batch advances the version without changing text. This performs
    /// whole-source analysis, not incremental parsing or compilation.
    ///
    /// # Errors
    /// Returns a precondition diagnostic without changing source or version when
    /// the version is not newer, a range is reversed, a line does not exist, or
    /// an endpoint splits a UTF-16 surrogate pair. Validation uses each edit's
    /// intermediate source, with columns beyond the line clamped to its end.
    pub fn apply_changes(
        &mut self,
        version: u64,
        changes: impl IntoIterator<Item = (Option<Range<EditorPosition>>, String)>,
    ) -> Result<&EditorSnapshot, Diagnostic> {
        if version <= self.current.version {
            return Err(stale_version(version, self.current.version));
        }
        let mut source = Cow::Borrowed(self.current.source.as_str());
        for (range, text) in changes {
            if let Some(Range { start, end }) = range {
                let invalid = || {
                    Diagnostic::error(
                        codes::PRECONDITION_FAILED,
                        "invalid editor text change range",
                    )
                };
                if (start.line(), start.character()) > (end.line(), end.character()) {
                    return Err(invalid());
                }
                let start = positions::edit_offset(&source, start).ok_or_else(invalid)?;
                let end = positions::edit_offset(&source, end).ok_or_else(invalid)?;
                source.to_mut().replace_range(start..end, &text);
            } else {
                source = Cow::Owned(text);
            }
        }
        let source = source.into_owned();
        self.replace(version, source)
    }
}
