//! Type a chosen reference and solvability condition without changing the Model.
use super::*;

pub(super) fn compile(
    file: &str,
    range: TextRange,
    declaration: (&str, &[(Expr, Expr); 2]),
    form: &CompiledAuthoredFormulation,
    symbols: &ModelSymbols,
    index: &KernelIndex<'_>,
    geometry: &eqiora_geometry::CanonicalGeometryV1,
) -> Result<wire::WireGauge, Diagnostic> {
    let (name, conditions) = declaration;
    let field = resolve_symbol(file, range, name, symbols)?;
    if form
        .trials
        .as_slice()
        .iter()
        .map(|id| id.erase())
        .collect::<Vec<_>>()
        != [field]
    {
        return Err(error(
            file,
            range,
            "gauge must name the exact scalar interval trial Field",
        ));
    }
    let mut context = ExpressionContext {
        file,
        symbols,
        index,
        ambient_dimension: geometry.ambient_dimension(),
        topological_dimension: geometry.topological_dimension(),
        relation_domain: form.domain,
        tests: BTreeMap::new(),
        integration_domain: None,
        used_tests: std::collections::BTreeSet::new(),
    };
    let mut compile_equality = |(left, right): &(Expr, Expr)| {
        let left = context.compile_root(left)?;
        let right = context.compile_root(right)?;
        let zero = |value: &AuthoredFormExpression| {
            matches!(value.kind, AuthoredFormExpressionKind::Number(0.0))
        };
        if left.value_type.dimension() != right.value_type.dimension()
            && !zero(&left)
            && !zero(&right)
        {
            return Err(error(
                file,
                range,
                "gauge equality sides must have identical physical dimensions",
            ));
        }
        Ok((wire::expression(&left), wire::expression(&right)))
    };
    Ok(wire::WireGauge {
        field_ulids: vec![field.ulid().to_string()],
        reference: compile_equality(&conditions[0])?,
        compatibility: compile_equality(&conditions[1])?,
    })
}
