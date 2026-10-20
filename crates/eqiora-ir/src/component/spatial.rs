//! Physical derivative directions remain separate from Field value coordinates.
use super::*;

impl<I: Clone + Eq> ComponentDagLowering<'_, I> {
    pub(super) fn lower_field_gradient(
        &mut self,
        root: ExprId,
        component: &[u32],
        part: ScalarPart,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        if self.pullback_depth != 0 {
            return Err(invalid_component_ir(
                "spatial Field pullback requires an admitted point reconstruction",
            ));
        }
        let dimension = self.node_types[root.index() as usize].dimension();
        let mut operand = root;
        let mut component = component;
        let mut axes = Vec::new();
        loop {
            match self.expression.node(operand) {
                Some(ExprNode::Gradient(value)) => {
                    let (axis, prefix) = component.split_last().ok_or_else(|| {
                        invalid_component_ir("Field gradient lacks its physical direction")
                    })?;
                    axes.push(*axis as usize);
                    component = prefix;
                    operand = *value;
                }
                Some(ExprNode::CoordinatePartial { value, wrt }) => {
                    let Some(ExprNode::Symbol(SymbolRef::Coordinate { axis, .. })) =
                        self.expression.node(*wrt)
                    else {
                        return Err(invalid_component_ir(
                            "Field partial requires an exact coordinate",
                        ));
                    };
                    axes.push(*axis);
                    operand = *value;
                }
                _ => break,
            }
        }
        let ty = &self.node_types[operand.index() as usize];
        validate_component(ty.shape(), component)?;
        axes.reverse();
        if let Some(ExprNode::Symbol(symbol @ SymbolRef::Field(_))) = self.expression.node(operand)
        {
            return self.input(*symbol, ty.shape(), dimension, component, part, &axes);
        }
        let mut dimension = ty.dimension();
        let mut value = self.lower_part(operand, component, part)?;
        for axis in axes {
            dimension = spatial_derivative_dimension(dimension)?;
            value = self.lower_spatial_partial(value, axis, dimension)?;
        }
        Ok(value)
    }

    fn lower_spatial_partial(
        &mut self,
        value: ScalarInputValueId,
        axis: usize,
        dimension: eqiora_core::DimExponents,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        let inputs = self.builder.directional_inputs(value)?;
        let mut seeds = vec![None; self.inputs.len()];
        for slot in inputs {
            let coordinate = slot.source();
            let seed = match coordinate.symbol {
                symbol @ SymbolRef::Field(_) => {
                    let shape = self.symbol_shapes[&symbol].clone();
                    let axes = coordinate
                        .spatial_derivative_axes
                        .iter()
                        .copied()
                        .chain([axis])
                        .collect::<Vec<_>>();
                    Some(self.input(
                        symbol,
                        &shape,
                        spatial_derivative_dimension(slot.dimension())?,
                        &coordinate.component_index,
                        coordinate.part,
                        &axes,
                    )?)
                }
                SymbolRef::Coordinate {
                    axis: source_axis, ..
                } if source_axis == axis => Some(self.builder.constant(DynQuantity::new(
                    1.,
                    eqiora_core::DimExponents::DIMENSIONLESS,
                ))?),
                SymbolRef::Parameter(_) | SymbolRef::Coordinate { .. } | SymbolRef::Time => None,
                _ => {
                    return Err(invalid_component_ir(
                        "spatial derivative requires a reconstructed Field, coordinate, or Parameter",
                    ));
                }
            };
            seeds[slot.ordinal() as usize] = seed;
        }
        match self.builder.directional(value, &seeds, dimension, |work| {
            super::finite::charge_products(self.finite_products, work)
        })? {
            Some(value) => Ok(value),
            None => self.builder.constant(DynQuantity::new(0., dimension)),
        }
    }

    pub(super) fn lower_field_divergence(
        &mut self,
        operand: ExprId,
        component: &[u32],
        part: ScalarPart,
        dimension: eqiora_core::DimExponents,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        if self.pullback_depth != 0 {
            return Err(invalid_component_ir(
                "spatial Field pullback requires an admitted point reconstruction",
            ));
        }
        let shape = self.node_types[operand.index() as usize].shape().clone();
        let extent = shape
            .extents()
            .last()
            .ok_or_else(|| invalid_component_ir("divergence requires a spatial axis"))?
            .get();
        let mut sum = None;
        for axis in 0..extent {
            let indices = component.iter().copied().chain([axis]).collect::<Vec<_>>();
            validate_component(&shape, &indices)?;
            let term = if let Some(ExprNode::Symbol(symbol @ SymbolRef::Field(_))) =
                self.expression.node(operand)
            {
                self.input(*symbol, &shape, dimension, &indices, part, &[axis as usize])?
            } else {
                let value = self.lower_part(operand, &indices, part)?;
                self.lower_spatial_partial(value, axis as usize, dimension)?
            };
            sum = Some(match sum {
                None => term,
                Some(sum) => self.builder.add(sum, term)?,
            });
        }
        sum.ok_or_else(|| invalid_component_ir("divergence has no physical directions"))
    }

    pub(super) fn boundary_normal(&self) -> Result<&[f64], Diagnostic> {
        match self.spatial {
            SpatialContext::Reconstructed {
                normal: Some(normal),
            } if !normal.is_empty()
                && normal.iter().all(|x| x.is_finite())
                && normal.iter().any(|x| *x != 0.) =>
            {
                Ok(normal)
            }
            _ => Err(invalid_component_ir(
                "spatial trace requires an exact oriented boundary normal",
            )),
        }
    }

    pub(super) fn lower_normal_component(
        &mut self,
        operand: ExprId,
        component: &[u32],
        part: ScalarPart,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        let normal = self.boundary_normal()?.to_vec();
        let shape = self.node_types[operand.index() as usize].shape();
        if shape.extents().last().map(|extent| extent.get() as usize) != Some(normal.len()) {
            return Err(invalid_component_ir(
                "normal contraction differs from its physical dimension",
            ));
        }
        let mut sum = None;
        for (axis, weight) in normal.into_iter().enumerate() {
            let indices = component
                .iter()
                .copied()
                .chain([axis as u32])
                .collect::<Vec<_>>();
            let value = self.lower_part(operand, &indices, part)?;
            let weight = self.builder.constant(DynQuantity::new(
                weight,
                eqiora_core::DimExponents::DIMENSIONLESS,
            ))?;
            let term = self.builder.mul(weight, value)?;
            sum = Some(match sum {
                None => term,
                Some(sum) => self.builder.add(sum, term)?,
            });
        }
        sum.ok_or_else(|| invalid_component_ir("normal contraction has no physical directions"))
    }
}

fn spatial_derivative_dimension(
    dimension: eqiora_core::DimExponents,
) -> Result<eqiora_core::DimExponents, Diagnostic> {
    let length =
        eqiora_core::DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("length dimension");
    dimension
        .div(length)
        .ok_or_else(|| invalid_component_ir("spatial derivative dimension overflow"))
}

#[cfg(test)]
mod tests;
