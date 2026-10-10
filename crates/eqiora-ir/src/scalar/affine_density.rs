//! Prove spatial constancy before specializing map factors for density admission.
use super::*;
use eqiora_core::{DynQuantity, ValueLiteral};
use eqiora_schema::kernel::{ExprNode, typing::TypedResidual};

impl ScalarOperatorIr {
    /// Project a density for regularity admission at fixed nonspatial inputs.
    /// Coordinate-map factors must have structurally affine mapped coordinates.
    /// Their spatially constant Jacobians use the shared LU/conditioning owner.
    /// This projection is not a parameter derivative: retained expressions remain
    /// authoritative for value evaluation and differentiation.
    /// # Errors
    /// Rejects non-affine maps, spatial coefficients, unavailable fixed inputs,
    /// invalid numerical factors and the ordinary scalar projection limits.
    pub fn lower_affine_map_density<I: Clone + Eq>(
        typed: &TypedResidual<I>,
        resolve: &mut impl FnMut(SymbolRef) -> Option<ValueLiteral>,
    ) -> Result<Self, Diagnostic> {
        let mut factors = HashMap::new();
        let expression = typed.expression();
        let mut work = 0usize;
        for (index, node) in expression.nodes().iter().enumerate() {
            let ExprNode::CoordinateMapFactor { factor, source, at } = node else {
                continue;
            };
            work = source
                .len()
                .checked_mul(at.len())
                .and_then(|n| n.checked_mul(expression.nodes().len()))
                .and_then(|n| work.checked_add(n))
                .filter(|n| *n <= 1_000_000)
                .ok_or_else(|| ir_builder_error("affine density map exceeds scalar work budget"))?;
            let id = expression
                .node_id(index as u32)
                .expect("retained map factor");
            let rows = Self::bind_affine_coordinate_map(typed, id, resolve)?;
            let entries = rows
                .iter()
                .flat_map(|row| row.coefficients().iter().copied())
                .collect::<Vec<_>>();
            let value = Self::coordinate_map_factor(&entries, source.len(), *factor)?;
            factors.insert(
                id,
                DynQuantity::new(
                    value,
                    typed.node_type(id).expect("typed factor").dimension(),
                ),
            );
        }
        Self::project_typed_operator(typed, Self::lower_bound_factors(expression, &factors)?)
    }

    /// Prove and bind the rows of one exact retained coordinate map.
    /// Rows follow its target-binding order; columns retain the ordered source
    /// coordinate identities. Fixed nonspatial inputs include physical Time.
    /// The same scalar SSA affine proof supplies values and Jacobian entries;
    /// no point sampling or numerical differentiation is used. Invertibility,
    /// orientation and a consecutive motion path remain with their consumers.
    ///
    /// # Errors
    /// Rejects a foreign or non-map node, non-affine mapped rows, spatial
    /// coefficients, unavailable or mistyped fixed inputs, and resource excess.
    pub fn bind_affine_coordinate_map<I: Clone + Eq>(
        typed: &TypedResidual<I>,
        id: eqiora_schema::kernel::ExprId,
        resolve: &mut impl FnMut(SymbolRef) -> Option<ValueLiteral>,
    ) -> Result<Vec<BoundAffineScalarIr<ScalarSymbolCoordinate>>, Diagnostic> {
        let expression = typed.expression();
        let Some(ExprNode::CoordinateMapFactor { source, at, .. }) = expression.node(id) else {
            return Err(ir_builder_error(
                "affine coordinate map requires its exact retained factor",
            ));
        };
        source
            .len()
            .checked_mul(at.len())
            .and_then(|n| n.checked_mul(expression.nodes().len()))
            .filter(|n| *n <= 1_000_000)
            .ok_or_else(|| ir_builder_error("affine density map exceeds scalar work budget"))?;
        let selected = source
            .iter()
            .map(|id| {
                let Some(ExprNode::Symbol(symbol @ SymbolRef::Coordinate { .. })) =
                    expression.node(*id)
                else {
                    return Err(ir_builder_error(
                        "affine density map source is not an exact coordinate",
                    ));
                };
                let coordinates = ScalarSymbolCoordinate::for_value(
                    *symbol,
                    &typed.node_type(*id).expect("typed source").value_type,
                )?;
                if coordinates.len() != 1 {
                    return Err(ir_builder_error(
                        "affine density source must be real scalar",
                    ));
                }
                Ok(coordinates[0].clone())
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let roots = at.iter().map(|(_, mapped)| *mapped).collect::<Vec<_>>();
        let rows = crate::ComponentScalarization::lower_selected(typed, &roots)?;
        if rows.rows().len() != source.len() {
            return Err(ir_builder_error("affine volume map must be square"));
        }
        let mut result = Vec::new();
        for row in rows.rows() {
            let mut fixed = Vec::new();
            for coordinate in row.symbols() {
                if selected.contains(coordinate) {
                    continue;
                }
                let symbol = coordinate.symbol();
                let ty = expression
                    .nodes()
                    .iter()
                    .zip(typed.node_types())
                    .find_map(|(node, ty)| {
                        matches!(node,ExprNode::Symbol(found) if *found == symbol).then_some(ty)
                    })
                    .ok_or_else(|| {
                        ir_builder_error("affine density coefficient has no retained type")
                    })?;
                if ty.support.is_some()
                    || coordinate.is_imaginary()
                    || !coordinate.component_index().is_empty()
                {
                    return Err(ir_builder_error(
                        "affine density map coefficient must be a nonspatial real scalar",
                    ));
                }
                let value = resolve(symbol).ok_or_else(|| {
                    ir_builder_error("affine density map coefficient is unavailable")
                })?;
                if value.value_type() != &ty.value_type {
                    return Err(ir_builder_error(
                        "affine density map coefficient differs from its retained type",
                    ));
                }
                let value = value.real_scalar_value().ok_or_else(|| {
                    ir_builder_error("affine density map coefficient must be real scalar")
                })?;
                fixed.push((coordinate.clone(), value.value()));
            }
            result.push(row.bind_affine(&selected, &fixed)?);
        }
        Ok(result)
    }
}
