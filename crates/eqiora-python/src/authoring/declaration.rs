//! Immutable typed declaration construction over the source AST factory.

use eqiora::kernel::EventDirection;
use eqiora::kernel::RelationConditionKind;
use eqiora::language::{
    ActivationSyntax, ComponentItem, FieldRoleSyntax, SignatureItem, SourceAstFactory as Ast,
    SupportSlotSyntax, TextRange, ValueTypeSyntax, VisibilitySyntax,
};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use super::expression::{PyAstExpression, path, syntax_error};

#[pyclass(name = "_AstType", module = "eqiora._eqiora", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyAstType {
    pub(crate) value: ValueTypeSyntax,
}

#[pymethods]
impl PyAstType {
    #[getter]
    fn source(&self) -> String {
        self.value.to_source()
    }
}

#[derive(Clone)]
pub(super) enum Declaration {
    Signature(SignatureItem),
    Item(ComponentItem),
}

#[pyclass(
    name = "_AstDeclaration",
    module = "eqiora._eqiora",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(super) struct PyAstDeclaration {
    pub(super) value: Declaration,
}

fn activation(clock: Option<String>) -> ActivationSyntax {
    clock.map_or(ActivationSyntax::Continuous, ActivationSyntax::Named)
}

fn range(ordinal: u32) -> TextRange {
    TextRange::new(ordinal, ordinal.saturating_add(1))
}

#[pymethods]
impl PyAstDeclaration {
    #[staticmethod]
    fn complete_exterior(name: String, parent: String, ordinal: u32) -> PyResult<Self> {
        super::boundaries::exterior(name, parent, ordinal)
    }

    #[staticmethod]
    fn field_port(
        name: String,
        connector: &str,
        support: String,
        set: Option<String>,
        ordinal: u32,
    ) -> PyResult<Self> {
        super::boundaries::port(name, connector, support, set, ordinal)
    }

    #[staticmethod]
    fn boundary_connection(
        ports: Vec<super::boundaries::Endpoint>,
        binder: Option<(String, String)>,
        periodic: bool,
        ordinal: u32,
    ) -> PyResult<Self> {
        super::boundaries::connection(ports, binder, periodic, ordinal)
    }

    #[staticmethod]
    fn boundary_relation(
        name: String,
        member: String,
        set: String,
        conditions: Vec<(
            String,
            PyRef<'_, PyAstExpression>,
            PyRef<'_, PyAstExpression>,
        )>,
        ordinal: u32,
    ) -> PyResult<Self> {
        super::boundaries::relation(name, member, set, conditions, ordinal)
    }

    #[staticmethod]
    fn scalar_port(name: String, connector: &str, ordinal: u32) -> PyResult<Self> {
        super::connections::port(name, connector, ordinal)
    }

    #[staticmethod]
    fn conserving_connection(
        ports: Vec<PyRef<'_, PyAstExpression>>,
        ordinal: u32,
    ) -> PyResult<Self> {
        super::connections::connection(ports, ordinal)
    }

    #[staticmethod]
    fn parameter(
        name: String,
        kind: &PyAstType,
        value: Option<&PyAstExpression>,
        ordinal: u32,
    ) -> PyResult<Self> {
        Ok(Self {
            value: Declaration::Signature(SignatureItem::Parameter(
                Ast::component_parameter(
                    VisibilitySyntax::Public,
                    name,
                    kind.value.clone(),
                    value.map(|value| value.value.clone()),
                    range(ordinal),
                )
                .map_err(syntax_error)?,
            )),
        })
    }

    #[staticmethod]
    fn observable(
        name: String,
        kind: &PyAstType,
        value: &PyAstExpression,
        ordinal: u32,
    ) -> PyResult<Self> {
        Ok(Self {
            value: Declaration::Item(ComponentItem::Observable(
                Ast::observable(
                    name,
                    kind.value.clone(),
                    value.value.clone(),
                    range(ordinal),
                )
                .map_err(syntax_error)?,
            )),
        })
    }

    #[staticmethod]
    fn field(
        name: String,
        kind: &PyAstType,
        role: &str,
        support: Option<String>,
        clock: Option<String>,
        signature: &str,
        ordinal: u32,
    ) -> PyResult<Self> {
        let role = match role {
            "state" => FieldRoleSyntax::State,
            "variable" => FieldRoleSyntax::Variable,
            _ => return Err(syntax_error("invalid field role")),
        };
        let value = Ast::field(
            name,
            support,
            role,
            activation(clock),
            kind.value.clone(),
            range(ordinal),
        )
        .map_err(syntax_error)?;
        Ok(Self {
            value: match signature {
                "input" => Declaration::Signature(SignatureItem::Input(value)),
                "output" => Declaration::Signature(SignatureItem::Output(value)),
                "field" => Declaration::Signature(SignatureItem::Field(value)),
                "" => Declaration::Item(ComponentItem::Field(value)),
                _ => return Err(syntax_error("invalid field declaration position")),
            },
        })
    }

    #[staticmethod]
    fn support(
        name: String,
        dimension: Option<u8>,
        parent: Option<String>,
        ordinal: u32,
    ) -> PyResult<Self> {
        let syntax = match (dimension, parent) {
            (Some(ambient_dimension), None) => SupportSlotSyntax::Volume {
                ambient_dimension: usize::from(ambient_dimension),
            },
            (None, Some(parent)) => SupportSlotSyntax::Boundary { parent },
            _ => {
                return Err(syntax_error(
                    "support needs exactly one volume dimension or boundary parent",
                ));
            }
        };
        Ok(Self {
            value: Declaration::Signature(SignatureItem::Support(
                Ast::support_slot(VisibilitySyntax::Public, name, syntax, range(ordinal))
                    .map_err(syntax_error)?,
            )),
        })
    }

    #[staticmethod]
    fn clock(
        name: String,
        period: Option<&PyAstExpression>,
        phase: Option<&PyAstExpression>,
        ordinal: u32,
    ) -> PyResult<Self> {
        let value = match (period, phase) {
            (Some(period), Some(phase)) => Declaration::Item(ComponentItem::Clock(
                Ast::clock(
                    name,
                    period.value.clone(),
                    phase.value.clone(),
                    range(ordinal),
                )
                .map_err(syntax_error)?,
            )),
            (None, None) => Declaration::Signature(SignatureItem::Clock(
                Ast::clock_requirement(name, range(ordinal)).map_err(syntax_error)?,
            )),
            _ => return Err(syntax_error("a clock needs both period and phase")),
        };
        Ok(Self { value })
    }

    #[staticmethod]
    fn index_set(name: String, extent: &PyAstExpression, ordinal: u32) -> PyResult<Self> {
        Ok(Self {
            value: Declaration::Item(ComponentItem::IndexSet(
                Ast::index_set(name, extent.value.clone(), range(ordinal)).map_err(syntax_error)?,
            )),
        })
    }

    #[staticmethod]
    fn alias(
        name: String,
        kind: Option<&PyAstType>,
        support: Option<String>,
        clock: Option<String>,
        value: &PyAstExpression,
        ordinal: u32,
    ) -> PyResult<Self> {
        Ok(Self {
            value: Declaration::Item(ComponentItem::Let(
                Ast::let_alias(
                    name,
                    kind.map(|kind| kind.value.clone()),
                    support,
                    clock,
                    value.value.clone(),
                    range(ordinal),
                )
                .map_err(syntax_error)?,
            )),
        })
    }

    #[staticmethod]
    fn law(
        name: String,
        support: String,
        flux: &PyAstExpression,
        source: &PyAstExpression,
        ordinal: u32,
    ) -> PyResult<Self> {
        super::law::declaration(name, support, flux, source, ordinal)
    }

    #[staticmethod]
    fn relation(
        name: String,
        support: Option<String>,
        clock: Option<String>,
        equations: Vec<(
            String,
            PyRef<'_, PyAstExpression>,
            PyRef<'_, PyAstExpression>,
        )>,
        ordinal: u32,
    ) -> PyResult<Self> {
        if equations.len() > 256 {
            return Err(syntax_error("relation exceeds the 256-condition limit"));
        }
        let range = range(ordinal);
        let equations = equations
            .into_iter()
            .map(|(kind, left, right)| {
                let kind = match kind.as_str() {
                    "equality" => RelationConditionKind::Equality,
                    "inequality" => RelationConditionKind::Inequality,
                    "complementarity" => RelationConditionKind::Complementarity,
                    _ => return Err(syntax_error("unknown mathematical condition kind")),
                };
                Ast::condition(kind, left.value.clone(), right.value.clone(), range)
                    .map_err(syntax_error)
            })
            .collect::<PyResult<_>>()?;
        Ok(Self {
            value: Declaration::Item(ComponentItem::Relation(
                Ast::relation(name, activation(clock), support, equations, range)
                    .map_err(syntax_error)?,
            )),
        })
    }

    #[staticmethod]
    fn property(name: String, contract: &str, ordinal: u32) -> PyResult<Self> {
        Ok(Self {
            value: Declaration::Signature(SignatureItem::Property(
                Ast::property_requirement(name, path(contract)?, range(ordinal))
                    .map_err(syntax_error)?,
            )),
        })
    }

    #[staticmethod]
    fn event(
        name: String,
        guard: &PyAstExpression,
        direction: &str,
        ordinal: u32,
    ) -> PyResult<Self> {
        let direction = match direction {
            "rising" => EventDirection::Rising,
            "falling" => EventDirection::Falling,
            "any" => EventDirection::Any,
            _ => return Err(syntax_error("invalid crossing direction")),
        };
        Ok(Self {
            value: Declaration::Item(ComponentItem::Event(
                Ast::event(name, guard.value.clone(), direction, range(ordinal))
                    .map_err(syntax_error)?,
            )),
        })
    }

    #[staticmethod]
    fn initial(
        equations: Vec<(PyRef<'_, PyAstExpression>, PyRef<'_, PyAstExpression>)>,
        ordinal: u32,
    ) -> PyResult<Self> {
        if equations.len() > 256 {
            return Err(syntax_error(
                "initial equation count exceeds declaration limit",
            ));
        }
        let range = range(ordinal);
        let equations = equations
            .into_iter()
            .map(|(left, right)| {
                Ast::equation(left.value.clone(), right.value.clone(), range).map_err(syntax_error)
            })
            .collect::<PyResult<_>>()?;
        Ok(Self {
            value: Declaration::Item(ComponentItem::Initial(
                Ast::initial(equations, range).map_err(syntax_error)?,
            )),
        })
    }

    #[staticmethod]
    fn instance(
        name: String,
        target: &str,
        bindings: Vec<(String, PyRef<'_, PyAstExpression>)>,
        ordinal: u32,
    ) -> PyResult<Self> {
        if bindings.len() > 256 {
            return Err(syntax_error("binding count exceeds declaration limit"));
        }
        let range = range(ordinal);
        let bindings = bindings
            .into_iter()
            .map(|(name, value)| {
                Ast::named_binding(name, value.value.clone(), range).map_err(syntax_error)
            })
            .collect::<PyResult<_>>()?;
        Ok(Self {
            value: Declaration::Item(ComponentItem::Instance(
                Ast::instance(name, path(target)?, None, bindings, range).map_err(syntax_error)?,
            )),
        })
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAstType>()?;
    module.add_class::<PyAstDeclaration>()?;
    Ok(())
}
