//! Coordinate binders retain exact names in the shared source AST.
use super::*;
use eqiora::kernel::CoordinateMapFactor;

impl PyAstExpression {
    pub(super) fn mapping(
        value: Option<&Self>,
        factor: Option<CoordinateMapFactor>,
        source: &Bound<'_, PyAny>,
        coordinates: &Bound<'_, PyAny>,
        points: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let source = expressions(source)?;
        let coordinates = expressions(coordinates)?;
        let points = expressions(points)?;
        if source.is_empty() || coordinates.is_empty() || coordinates.len() != points.len() {
            return Err(syntax_error(
                "coordinate map requires nonempty source and target bindings",
            ));
        }
        let binding = |expression: &Self| match expression.value.kind() {
            ExprKind::Name(name) => path(name),
            ExprKind::Path(path) => Ok(path.clone()),
            _ => Err(syntax_error(
                "coordinate map selectors must be exact declared names",
            )),
        };
        let source = source
            .iter()
            .map(|value| binding(value))
            .collect::<PyResult<Vec<_>>>()?;
        let at = coordinates
            .iter()
            .zip(&points)
            .map(|(coordinate, point)| Ok((binding(coordinate)?, point.value.clone())))
            .collect::<PyResult<Vec<_>>>()?;
        let mut children = value.into_iter().collect::<Vec<_>>();
        children.extend(points.iter().map(|point| &**point));
        Self::build(&children, source.len() + at.len() + 1, || {
            Ok(match (value, factor) {
                (Some(value), None) => ExprKind::Pullback {
                    value: Box::new(value.value.clone()),
                    source,
                    at,
                },
                (None, Some(factor)) => ExprKind::CoordinateMapFactor { factor, source, at },
                _ => return Err(syntax_error("invalid coordinate map operation")),
            })
        })
    }
}
