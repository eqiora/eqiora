use std::borrow::Cow;

use super::{EditorPosition, EditorService, EditorSnapshot, codes, positions, stale_version};
use eqiora_core::Diagnostic;

/// One whole-source replacement or UTF-16 range edit in an ordered change batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditorTextChange {
    range: Option<(EditorPosition, EditorPosition)>,
    text: String,
}

impl EditorTextChange {
    /// Replace the complete source at this point in the batch.
    #[must_use]
    pub fn replace_all(text: impl Into<String>) -> Self {
        Self {
            range: None,
            text: text.into(),
        }
    }

    /// Replace a half-open range in the source produced by preceding changes.
    /// Columns beyond a line's end clamp to its end; lines must exist and
    /// positions must not split a UTF-16 surrogate pair.
    #[must_use]
    pub fn replace_range(
        start: EditorPosition,
        end: EditorPosition,
        text: impl Into<String>,
    ) -> Self {
        Self {
            range: Some((start, end)),
            text: text.into(),
        }
    }
}

impl EditorService {
    /// Apply ordered text changes atomically and analyze the final source once.
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
        changes: impl IntoIterator<Item = EditorTextChange>,
    ) -> Result<&EditorSnapshot, Diagnostic> {
        if version <= self.current.version {
            return Err(stale_version(version, self.current.version));
        }
        let mut source = Cow::Borrowed(self.current.source.as_str());
        for change in changes {
            if let Some((start, end)) = change.range {
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
                source.to_mut().replace_range(start..end, &change.text);
            } else {
                source = Cow::Owned(change.text);
            }
        }
        let source = source.into_owned();
        self.replace(version, source)
    }
}
