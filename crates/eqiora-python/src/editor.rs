//! Notebook completion projects the existing standalone editor snapshot.

use eqiora::api::{EditorSnapshot, EditorWorkspaceSnapshot};
use pyo3::prelude::*;

use crate::error::panic_boundary;

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
