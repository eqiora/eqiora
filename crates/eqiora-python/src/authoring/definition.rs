//! Complete definitions retain only compiler-owned declaration nodes.

use eqiora::language::{
    ComponentDecl, ComponentItem, Item, ModelDecl, SourceAstFactory as Ast, TextRange,
    VisibilitySyntax,
};
use pyo3::prelude::*;

use super::declaration::{Declaration, PyAstDeclaration};
use super::expression::{PyAstExpression, syntax_error};

#[derive(Clone)]
pub(super) enum Definition {
    Component(ComponentDecl),
    Model(ModelDecl),
}

#[pyclass(
    name = "_AstDefinition",
    module = "eqiora._eqiora",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(super) struct PyAstDefinition {
    pub(super) value: Definition,
}

type FormInput<'py> = (
    String,
    Vec<String>,
    Vec<(
        String,
        String,
        Vec<String>,
        PyRef<'py, crate::modeling::PyValueType>,
    )>,
    Vec<(PyRef<'py, PyAstExpression>, PyRef<'py, PyAstExpression>)>,
    u32,
);

#[pymethods]
impl PyAstDefinition {
    #[new]
    fn new(
        name: String,
        model: bool,
        declarations: Vec<PyRef<'_, PyAstDeclaration>>,
        form: Option<FormInput<'_>>,
        ordinal: u32,
    ) -> PyResult<Self> {
        if declarations.len() > 256 {
            return Err(syntax_error("definition exceeds 256 declarations"));
        }
        let mut signature = Vec::new();
        let mut items = Vec::new();
        for declaration in declarations {
            match &declaration.value {
                Declaration::Signature(value) => signature.push(value.clone()),
                Declaration::Item(value) => items.push(value.clone()),
            }
        }
        let range = TextRange::new(ordinal, ordinal.saturating_add(1));
        let form = form
            .map(|(name, relations, tests, equations, ordinal)| {
                if relations.len() > 8 || tests.len() > 8 || equations.len() > 8 {
                    return Err(syntax_error("weak form exceeds the 8-item inventory limit"));
                }
                let tests = tests
                    .into_iter()
                    .map(|(name, trial, zero_on, kind)| {
                        if !kind.value.shape().is_scalar()
                            || kind.value.scalar_domain() != eqiora::ScalarDomain::Real
                        {
                            return Err(syntax_error("test dimension requires a real scalar type"));
                        }
                        Ok((name, trial, zero_on, super::boundaries::dimension(&kind)?))
                    })
                    .collect::<PyResult<_>>()?;
                let equations = equations
                    .into_iter()
                    .map(|(left, right)| (left.value.clone(), right.value.clone()))
                    .collect();
                Ok((
                    (
                        name,
                        relations,
                        eqiora::language::FormulationBinding::WeakTests { tests },
                    ),
                    (
                        equations,
                        TextRange::new(ordinal, ordinal.saturating_add(1)),
                    ),
                ))
            })
            .transpose()?;
        let value = if model {
            let items = items.into_iter().map(model_item).collect::<PyResult<_>>()?;
            Definition::Model(
                match form {
                    Some((form, equalities)) => Ast::model_with_form(
                        VisibilitySyntax::Public,
                        name,
                        signature,
                        items,
                        form,
                        equalities,
                        range,
                    ),
                    None => Ast::model(VisibilitySyntax::Public, name, signature, items, range),
                }
                .map_err(syntax_error)?,
            )
        } else {
            Definition::Component(
                match form {
                    Some((form, equalities)) => Ast::component_with_form(
                        VisibilitySyntax::Public,
                        name,
                        signature,
                        items,
                        form,
                        equalities,
                        range,
                    ),
                    None => Ast::component(VisibilitySyntax::Public, name, signature, items, range),
                }
                .map_err(syntax_error)?,
            )
        };
        Ok(Self { value })
    }
}

fn model_item(item: ComponentItem) -> PyResult<Item> {
    Ok(match item {
        ComponentItem::Coordinate(value) => Item::Coordinate(value),
        ComponentItem::Let(value) => Item::Let(value),
        ComponentItem::Field(value) => Item::Field(value),
        ComponentItem::Initial(value) => Item::Initial(value),
        ComponentItem::Observable(value) => Item::Observable(value),
        ComponentItem::Event(value) => Item::Event(value),
        ComponentItem::Clock(value) => Item::Clock(value),
        ComponentItem::Relation(value) => Item::Relation(value),
        ComponentItem::RelationFamily(value) => Item::RelationFamily(value),
        ComponentItem::Connection(value) => Item::Connection(value),
        ComponentItem::BoundaryConnection(value) => Item::BoundaryConnection(value),
        ComponentItem::IndexSet(value) => Item::IndexSet(value),
        ComponentItem::Instance(value) => Item::Instance(value),
        ComponentItem::Parameter(_) | ComponentItem::Port(_) | ComponentItem::PortFamily(_) => {
            return Err(syntax_error(
                "component-local endpoint cannot become a Model item",
            ));
        }
        _ => return Err(syntax_error("unsupported Model item")),
    })
}
