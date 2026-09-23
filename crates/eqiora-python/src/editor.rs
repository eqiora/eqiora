//! Notebook assistance projects the existing standalone editor snapshot.

use eqiora::api::{EditorSnapshot, EditorWorkspaceSnapshot};
use eqiora::language::{NotationLabel, NotationProfile};
use pyo3::prelude::*;

use crate::error::panic_boundary;

#[pyfunction]
pub(super) fn _hover_source_cell(
    py: Python<'_>,
    source: &str,
    cursor: usize,
) -> PyResult<Option<(usize, usize, String)>> {
    panic_boundary(py, || {
        if source.len() > EditorSnapshot::MAX_SOURCE_BYTES {
            return Ok(None);
        }
        let Some(offset) = source
            .char_indices()
            .map(|(offset, _)| offset)
            .chain(std::iter::once(source.len()))
            .nth(cursor)
        else {
            return Ok(None);
        };
        let source = source.to_owned();
        Ok(py.detach(move || {
            let snapshot = EditorWorkspaceSnapshot::analyze_standalone(0, source.clone());
            let file = snapshot.files().next()?;
            let (name, range) = snapshot.document(file)?.name_at(offset as u32)?;
            let symbol = snapshot.assistance(file, offset as u32, &name)?;
            let mut text = symbol.detail().unwrap_or(&name).to_owned();
            // Authored prose already has a plain-text owner. The Markdown
            // projection escapes punctuation for renderers this adapter does not use.
            let documentation = symbol
                .doc_comment()
                .map(|comment| comment.text().to_owned())
                .or_else(|| symbol.documentation());
            if let Some(documentation) = documentation {
                text.push_str("\n\n");
                text.push_str(&documentation);
            }
            if let Some(notation) = symbol.notation() {
                text.push_str("\n\nNotation: ");
                text.push_str(
                    &NotationLabel::from_notation(notation).render(NotationProfile::Plain),
                );
            }
            Some((
                source[..range.start() as usize].chars().count(),
                source[..range.end() as usize].chars().count(),
                text.chars().take(4096).collect(),
            ))
        }))
    })
}

#[pyfunction]
pub(super) fn _complete_source_cell(
    py: Python<'_>,
    source: &str,
    cursor: usize,
    limit: usize,
) -> PyResult<Option<(String, Vec<String>)>> {
    panic_boundary(py, || {
        if source.len() > EditorSnapshot::MAX_SOURCE_BYTES || limit == 0 {
            return Ok(None);
        }
        // IPython offsets count Unicode code points; the editor uses UTF-8 bytes.
        let Some(offset) = source
            .char_indices()
            .map(|(offset, _)| offset)
            .chain(std::iter::once(source.len()))
            .nth(cursor)
        else {
            return Ok(None);
        };
        let source = source.to_owned();
        Ok(py.detach(move || {
            let snapshot = EditorWorkspaceSnapshot::analyze_standalone(0, source.clone());
            let file = snapshot.files().next()?;
            let (range, items) = snapshot.completion(file, offset as u32)?;
            // IPython's matcher replaces only text before the cursor. Avoid
            // duplicating a suffix when the cursor is inside an existing token.
            if range.end() as usize != offset {
                return None;
            }
            Some((
                source[range.start() as usize..offset].to_owned(),
                items
                    .into_iter()
                    .take(limit)
                    .map(|item| item.insert_text().to_owned())
                    .collect(),
            ))
        }))
    })
}
