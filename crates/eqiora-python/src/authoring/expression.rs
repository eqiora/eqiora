//! Syntax construction only: all typing and evaluation stays in the compiler.

use eqiora::language::{
    BinaryOp, BoundarySideSyntax, CallArguments, DecimalLiteral, Expr, ExprKind, NamePath,
    ReductionOp, SourceAstFactory as Ast, TextRange, UnaryOp,
};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

pub(super) const RANGE: TextRange = TextRange::new(0, 1);
const MAX_DEPTH: usize = 64;
const MAX_NODES: usize = 4096;

#[pyclass(
    name = "_AstExpression",
    module = "eqiora._eqiora",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyAstExpression {
    pub(super) value: Expr,
    depth: usize,
    nodes: usize,
}

pub(super) fn syntax_error(error: impl std::fmt::Display) -> PyErr {
    super::ModuleError::new_err(error.to_string())
}

pub(super) fn path(name: &str) -> PyResult<NamePath> {
    NamePath::from_segments(name.split('.'), RANGE).map_err(syntax_error)
}

impl PyAstExpression {
    fn build(
        children: &[&Self],
        extra: usize,
        make: impl FnOnce() -> PyResult<ExprKind>,
    ) -> PyResult<Self> {
        let depth = children.iter().map(|child| child.depth).max().unwrap_or(0) + 1;
        let nodes = children
            .iter()
            .try_fold(extra, |sum, child| sum.checked_add(child.nodes))
            .ok_or_else(|| syntax_error("expression node count overflows"))?;
        if depth > MAX_DEPTH || nodes > MAX_NODES {
            return Err(syntax_error(
                "expression exceeds the 64-depth or 4096-node authoring limit",
            ));
        }
        let value = Ast::expression(make()?, RANGE).map_err(syntax_error)?;
        Ok(Self {
            value,
            depth,
            nodes,
        })
    }
}

fn expressions<'py>(values: &Bound<'py, PyAny>) -> PyResult<Vec<PyRef<'py, PyAstExpression>>> {
    if !values.is_instance_of::<PyList>() && !values.is_instance_of::<PyTuple>() {
        return Err(syntax_error("expression children must be a list or tuple"));
    }
    let count = values.len()?;
    if count > MAX_NODES {
        return Err(syntax_error("expression children exceed the node limit"));
    }
    (0..count)
        .map(|index| Ok(values.get_item(index)?.extract()?))
        .collect()
}

#[pymethods]
impl PyAstExpression {
    #[getter]
    fn depth(&self) -> usize {
        self.depth
    }

    #[getter]
    fn node_count(&self) -> usize {
        self.nodes
    }
    #[staticmethod]
    fn number(text: &str) -> PyResult<Self> {
        if text.len() > 1024 {
            return Err(syntax_error("numeric literal exceeds 1024 bytes"));
        }
        if let Some(positive) = text.strip_prefix('-') {
            return Self::number(positive)?.unary("-");
        }
        Self::build(&[], 1, || {
            Ok(ExprKind::Number(
                DecimalLiteral::parse(text).map_err(syntax_error)?,
            ))
        })
    }

    #[staticmethod]
    fn boolean(value: bool) -> PyResult<Self> {
        Self::build(&[], 1, || Ok(ExprKind::Boolean(value)))
    }

    #[staticmethod]
    pub(super) fn name(name: &str) -> PyResult<Self> {
        Self::build(&[], 1, || {
            let path = path(name)?;
            Ok(if path.is_qualified() {
                ExprKind::Path(path)
            } else {
                ExprKind::Name(name.to_owned())
            })
        })
    }

    #[staticmethod]
    fn partial(value: &Self, wrt: &Self, holding: &Bound<'_, PyAny>) -> PyResult<Self> {
        let held = expressions(holding)?;
        let binding = |expression: &Self| match expression.value.kind() {
            ExprKind::Name(name) => path(name),
            ExprKind::Path(path) => Ok(path.clone()),
            _ => Err(syntax_error(
                "partial bindings must be exact declared names",
            )),
        };
        let selected = binding(wrt)?;
        let mut seen = std::collections::BTreeSet::from([selected.as_str().to_owned()]);
        let held = held
            .iter()
            .map(|expression| {
                let path = binding(expression)?;
                if !seen.insert(path.as_str().to_owned()) {
                    return Err(syntax_error(
                        "partial holding requires distinct other bindings",
                    ));
                }
                Ok(path)
            })
            .collect::<PyResult<Vec<_>>>()?;
        Self::build(&[value], held.len() + 2, || {
            Ok(ExprKind::Partial {
                value: Box::new(value.value.clone()),
                wrt: selected,
                holding: held,
            })
        })
    }

    #[staticmethod]
    #[pyo3(signature = (value, coordinates, points, side=None))]
    fn evaluate(
        value: &Self,
        coordinates: &Bound<'_, PyAny>,
        points: &Bound<'_, PyAny>,
        side: Option<&str>,
    ) -> PyResult<Self> {
        let coordinates = expressions(coordinates)?;
        let points = expressions(points)?;
        if coordinates.is_empty() || coordinates.len() > 64 || coordinates.len() != points.len() {
            return Err(syntax_error(
                "evaluate requires 1..64 exact coordinate bindings",
            ));
        }
        let side = match side {
            None => None,
            Some("lower") => Some(BoundarySideSyntax::Lower),
            Some("upper") => Some(BoundarySideSyntax::Upper),
            _ => return Err(syntax_error("evaluate side must be lower, upper or None")),
        };
        let at = coordinates
            .iter()
            .zip(&points)
            .map(|(coordinate, point)| {
                let coordinate = match coordinate.value.kind() {
                    ExprKind::Name(name) => path(name)?,
                    ExprKind::Path(path) => path.clone(),
                    _ => {
                        return Err(syntax_error(
                            "evaluate coordinates must be exact declared names",
                        ));
                    }
                };
                Ok((coordinate, point.value.clone()))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let mut children = vec![value];
        children.extend(points.iter().map(|point| &**point));
        Self::build(&children, at.len() + 1, || {
            Ok(ExprKind::Evaluate {
                value: Box::new(value.value.clone()),
                at,
                side,
            })
        })
    }

    fn boundary_port(&self, member: &str, target: &str) -> PyResult<Self> {
        Self::build(&[self], 1, || {
            let port = match self.value.kind() {
                ExprKind::Name(name) => path(name)?,
                ExprKind::Path(path) => path.clone(),
                _ => {
                    return Err(syntax_error(
                        "boundary selection requires a named port family",
                    ));
                }
            };
            Ok(ExprKind::BoundaryPortSelection {
                port: Box::new(port),
                selector: Box::new(
                    Ast::boundary_port_selector(member, target, RANGE).map_err(syntax_error)?,
                ),
            })
        })
    }

    fn member(&self, member: &str) -> PyResult<Self> {
        Self::build(&[self], 1, || {
            Ok(ExprKind::Member {
                value: Box::new(self.value.clone()),
                member: member.to_owned(),
            })
        })
    }

    fn unary(&self, operator: &str) -> PyResult<Self> {
        let op = match operator {
            "-" => UnaryOp::Neg,
            "not" => UnaryOp::Not,
            _ => return Err(syntax_error("unknown unary operator")),
        };
        Self::build(&[self], 1, || {
            Ok(ExprKind::Unary {
                op,
                value: Box::new(self.value.clone()),
            })
        })
    }

    fn binary(&self, operator: &str, right: &Self) -> PyResult<Self> {
        let op = match operator {
            "+" => BinaryOp::Add,
            "-" => BinaryOp::Sub,
            "*" => BinaryOp::Mul,
            "/" => BinaryOp::Div,
            "^" => BinaryOp::Pow,
            "==" => BinaryOp::Equal,
            "!=" => BinaryOp::NotEqual,
            "<" => BinaryOp::Less,
            "<=" => BinaryOp::LessEqual,
            ">" => BinaryOp::Greater,
            ">=" => BinaryOp::GreaterEqual,
            "and" => BinaryOp::And,
            "or" => BinaryOp::Or,
            _ => return Err(syntax_error("unknown binary operator")),
        };
        Self::build(&[self, right], 1, || {
            Ok(ExprKind::Binary {
                op,
                left: Box::new(self.value.clone()),
                right: Box::new(right.value.clone()),
            })
        })
    }

    #[staticmethod]
    fn array(values: &Bound<'_, PyAny>) -> PyResult<Self> {
        let values = expressions(values)?;
        Self::build(
            &values.iter().map(|value| &**value).collect::<Vec<_>>(),
            1,
            || {
                Ok(ExprKind::Array(
                    values.iter().map(|value| value.value.clone()).collect(),
                ))
            },
        )
    }

    #[staticmethod]
    fn tuple(values: &Bound<'_, PyAny>) -> PyResult<Self> {
        let values = expressions(values)?;
        Self::build(
            &values.iter().map(|value| &**value).collect::<Vec<_>>(),
            1,
            || {
                Ok(ExprKind::Tuple(
                    values.iter().map(|value| value.value.clone()).collect(),
                ))
            },
        )
    }

    #[staticmethod]
    fn mixed_call(
        name: &str,
        positional: &Bound<'_, PyAny>,
        names: Vec<String>,
        named: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let positional = expressions(positional)?;
        let named = expressions(named)?;
        if names.len() != named.len() {
            return Err(syntax_error("named call arity mismatch"));
        }
        let children = positional
            .iter()
            .chain(&named)
            .map(|value| &**value)
            .collect::<Vec<_>>();
        Self::build(&children, 1, || {
            Ok(ExprKind::Call {
                callee: path(name)?,
                arguments: CallArguments::Mixed {
                    positional: positional.iter().map(|value| value.value.clone()).collect(),
                    named: names
                        .into_iter()
                        .zip(&named)
                        .map(|(name, value)| {
                            Ast::named_binding(name, value.value.clone(), RANGE)
                                .map_err(syntax_error)
                        })
                        .collect::<PyResult<_>>()?,
                },
            })
        })
    }

    fn index(&self, index: &Self) -> PyResult<Self> {
        Self::build(&[self, index], 1, || {
            Ok(ExprKind::Index {
                value: Box::new(self.value.clone()),
                index: Box::new(index.value.clone()),
            })
        })
    }

    fn slice(&self, lower: &Self, upper: &Self) -> PyResult<Self> {
        Self::build(&[self, lower, upper], 1, || {
            Ok(ExprKind::Slice {
                value: Box::new(self.value.clone()),
                lower: Box::new(lower.value.clone()),
                upper: Box::new(upper.value.clone()),
            })
        })
    }

    #[staticmethod]
    #[pyo3(signature = (name, values, names=None))]
    fn call(name: &str, values: &Bound<'_, PyAny>, names: Option<Vec<String>>) -> PyResult<Self> {
        let values = expressions(values)?;
        Self::build(
            &values.iter().map(|value| &**value).collect::<Vec<_>>(),
            1,
            || {
                let arguments = if let Some(names) = names {
                    if names.len() != values.len() {
                        return Err(syntax_error("named call arity mismatch"));
                    }
                    CallArguments::Named(
                        names
                            .into_iter()
                            .zip(&values)
                            .map(|(name, value)| {
                                Ast::named_binding(name, value.value.clone(), RANGE)
                                    .map_err(syntax_error)
                            })
                            .collect::<PyResult<_>>()?,
                    )
                } else {
                    CallArguments::Positional(
                        values.iter().map(|value| value.value.clone()).collect(),
                    )
                };
                Ok(ExprKind::Call {
                    callee: path(name)?,
                    arguments,
                })
            },
        )
    }

    #[staticmethod]
    fn select(condition: &Self, then_value: &Self, else_value: &Self) -> PyResult<Self> {
        Self::build(&[condition, then_value, else_value], 1, || {
            Ok(ExprKind::Select {
                condition: Box::new(condition.value.clone()),
                then_value: Box::new(then_value.value.clone()),
                else_value: Box::new(else_value.value.clone()),
            })
        })
    }

    fn case(&self, patterns: Vec<String>, values: &Bound<'_, PyAny>) -> PyResult<Self> {
        let values = expressions(values)?;
        let mut children = vec![self];
        children.extend(values.iter().map(|value| &**value));
        Self::build(&children, patterns.len() + 1, || {
            if patterns.len() != values.len() {
                return Err(syntax_error("case arm count mismatch"));
            }
            let arms = patterns
                .iter()
                .zip(&values)
                .map(|(pattern, value)| {
                    Ast::case_arm(path(pattern)?, value.value.clone(), RANGE).map_err(syntax_error)
                })
                .collect::<PyResult<_>>()?;
            Ok(ExprKind::Case {
                value: Box::new(self.value.clone()),
                arms,
            })
        })
    }

    fn reduction(&self, operator: &str, member: &str, set: &str) -> PyResult<Self> {
        let op = match operator {
            "sum" => ReductionOp::Sum,
            "product" => ReductionOp::Product,
            "min" => ReductionOp::Min,
            "max" => ReductionOp::Max,
            _ => return Err(syntax_error("unknown reduction")),
        };
        Self::build(&[self], 2, || {
            Ok(ExprKind::Reduction {
                operation: op,
                binder: Ast::index_family_binder(member, path(set)?, RANGE)
                    .map_err(syntax_error)?,
                value: Box::new(self.value.clone()),
            })
        })
    }

    #[staticmethod]
    fn quantity(text: &str, unit: &Self) -> PyResult<Self> {
        if text.len() > 1024 {
            return Err(syntax_error("numeric literal exceeds 1024 bytes"));
        }
        if let Some(positive) = text.strip_prefix('-') {
            return Self::quantity(positive, unit)?.unary("-");
        }
        Self::build(&[unit], 1, || {
            Ok(ExprKind::Quantity {
                value: DecimalLiteral::parse(text).map_err(syntax_error)?,
                unit: Box::new(unit.value.clone()),
            })
        })
    }

    #[getter]
    fn source(&self) -> String {
        self.value.to_source()
    }
}
