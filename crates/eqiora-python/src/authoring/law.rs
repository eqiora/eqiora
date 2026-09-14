//! Typed physical Law construction through the same source Relation declaration.

use eqiora::language::{ComponentItem, SourceAstFactory as Ast, TextRange};
use pyo3::prelude::*;

use super::declaration::{Declaration, PyAstDeclaration};
use super::expression::{PyAstExpression, syntax_error};

pub(super) fn declaration(
    name: String,
    support: String,
    storage: Option<&PyAstExpression>,
    flux: &PyAstExpression,
    source: &PyAstExpression,
    ordinal: u32,
) -> PyResult<PyAstDeclaration> {
    let value = Ast::law(
        name,
        support,
        storage.map(|value| value.value.clone()),
        flux.value.clone(),
        source.value.clone(),
        TextRange::new(ordinal, ordinal.saturating_add(1)),
    )
    .map_err(syntax_error)?;
    Ok(PyAstDeclaration {
        value: Declaration::Item(ComponentItem::Relation(value)),
    })
}
