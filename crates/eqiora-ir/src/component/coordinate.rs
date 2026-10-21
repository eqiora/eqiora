//! Typed numerical coordinates shared by finite and spatial consumers.
use super::{invalid_component_ir, row_major_index};
use eqiora_core::Diagnostic;
use eqiora_schema::kernel::SymbolRef;

/// Real coordinate within a mathematical scalar, independent of storage precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum ScalarPart {
    /// Real component (the only coordinate of a real scalar).
    Real,
    /// Imaginary component of a complex scalar.
    Imaginary,
}

impl ScalarPart {
    pub(super) fn for_domain(
        domain: eqiora_core::ScalarDomain,
    ) -> Result<&'static [Self], Diagnostic> {
        match domain {
            eqiora_core::ScalarDomain::Real => Ok(&[Self::Real]),
            eqiora_core::ScalarDomain::Complex => Ok(&[Self::Real, Self::Imaginary]),
            _ => Err(invalid_component_ir(
                "scalar coordinates require a numeric value",
            )),
        }
    }
}

impl<I: Clone + Eq> super::ComponentDagLowering<'_, I> {
    /// Retain every input component in derivative permissions, including components
    /// absent from the output formula. Zero numerical actions do not grant permission.
    pub(super) fn lower_input_coordinates(
        &mut self,
        inputs: &[eqiora_schema::kernel::ExprId],
    ) -> Result<Vec<crate::scalar::ScalarInputValueId>, Diagnostic> {
        let mut values = Vec::new();
        for &input in inputs {
            let index = super::node_index(input, self.node_types.len())?;
            let ty = self.node_types[index].value_type.clone();
            let parts = ScalarPart::for_domain(ty.scalar_domain())?;
            let count = ty
                .shape()
                .component_count()
                .ok_or_else(|| invalid_component_ir("coordinate count overflow"))?;
            let work = count
                .checked_mul(parts.len())
                .ok_or_else(|| invalid_component_ir("coordinate count overflow"))?;
            super::finite::charge_products(self.finite_products, work)?;
            for flat in 0..count {
                let component = row_major_index(ty.shape(), flat)?;
                for &part in parts {
                    values.push(self.lower_part(input, &component, part)?);
                }
            }
        }
        Ok(values)
    }
}

/// One scalar coordinate of a shaped Semantic Model symbol.
///
/// The empty component index denotes a scalar. Non-scalar indices are
/// lexicographic row-major coordinates with the last axis varying fastest.
/// Real and imaginary parts retain that same symbol and channel coordinate.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ScalarSymbolCoordinate {
    pub(super) symbol: SymbolRef,
    pub(super) component_index: Box<[u32]>,
    pub(super) part: ScalarPart,
    pub(super) spatial_derivative_axes: Box<[usize]>,
}

impl ScalarSymbolCoordinate {
    /// Enumerate a numeric value's coordinates in channel-major, real/imaginary order.
    ///
    /// # Errors
    /// Rejects discrete values or unrepresentable component cardinalities.
    pub fn for_value(
        symbol: SymbolRef,
        value_type: &eqiora_core::ValueType,
    ) -> Result<Vec<Self>, Diagnostic> {
        let parts = ScalarPart::for_domain(value_type.scalar_domain())?;
        let count = value_type
            .shape()
            .component_count()
            .ok_or_else(|| invalid_component_ir("coordinate count overflow"))?;
        let mut coordinates = Vec::new();
        coordinates
            .try_reserve_exact(
                count
                    .checked_mul(parts.len())
                    .ok_or_else(|| invalid_component_ir("coordinate count overflow"))?,
            )
            .map_err(|_| invalid_component_ir("cannot allocate scalar coordinates"))?;
        for flat in 0..count {
            let component_index = row_major_index(value_type.shape(), flat)?;
            coordinates.extend(parts.iter().map(|&part| Self {
                symbol,
                component_index: component_index.clone(),
                part,
                spatial_derivative_axes: Box::default(),
            }));
        }
        Ok(coordinates)
    }

    /// Read this numerical coordinate from an already typed symbol value.
    /// The caller owns symbol identity and complete ValueType agreement.
    ///
    /// # Errors
    /// Rejects derivative coordinates, nonnumeric values, an unavailable component,
    /// or an imaginary coordinate of a real value.
    pub fn read_value(&self, value: &eqiora_core::ValueLiteral) -> Result<f64, Diagnostic> {
        let domain = value.value_type().scalar_domain();
        if !self.spatial_derivative_axes.is_empty()
            || !matches!(
                domain,
                eqiora_core::ScalarDomain::Real | eqiora_core::ScalarDomain::Complex
            )
            || (self.is_imaginary() && domain != eqiora_core::ScalarDomain::Complex)
        {
            return Err(invalid_component_ir(
                "coordinate does not select a part of this numeric value",
            ));
        }
        let flat = value
            .value_type()
            .shape()
            .flat_index(self.component_index())
            .ok_or_else(|| {
                invalid_component_ir("numeric component coordinate differs from its exact shape")
            })?;
        let (real, imaginary) = value
            .component(flat)
            .ok_or_else(|| invalid_component_ir("numeric component is unavailable"))?;
        Ok(if self.is_imaginary() { imaginary } else { real })
    }

    /// Semantic symbol before Operator lowering.
    #[must_use]
    pub const fn symbol(&self) -> SymbolRef {
        self.symbol
    }

    /// Rename the source symbol while retaining its exact component and scalar part.
    /// The consumer must establish that the replacement has the same value shape
    /// and scalar domain (for example a Field and its time derivative).
    #[must_use]
    pub fn with_symbol(&self, symbol: SymbolRef) -> Self {
        Self {
            symbol,
            component_index: self.component_index.clone(),
            part: self.part,
            spatial_derivative_axes: self.spatial_derivative_axes.clone(),
        }
    }

    /// Ordered physical derivative directions, independent of value components.
    /// Empty denotes the symbol value. These coordinates are emitted only by
    /// the spatial lowering entry point and require a spatial realization.
    #[must_use]
    pub fn spatial_derivative_axes(&self) -> &[usize] {
        &self.spatial_derivative_axes
    }

    /// Whether this is the imaginary coordinate; false denotes the real coordinate.
    /// This is separate from the channel index.
    #[must_use]
    pub const fn is_imaginary(&self) -> bool {
        matches!(self.part, ScalarPart::Imaginary)
    }

    /// Exact row-major component multi-index; empty for a scalar.
    #[must_use]
    pub const fn component_index(&self) -> &[u32] {
        &self.component_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{DimExponents, Id, ScalarDomain, ValueLiteral, ValueType};

    #[test]
    fn coordinate_reads_preserve_nested_channels_and_complex_phase() {
        let symbol = SymbolRef::Parameter(Id::new());
        let dimension = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
        for domain in [ScalarDomain::Real, ScalarDomain::Complex] {
            let ty = ValueType::scalar(domain, dimension)
                .unwrap()
                .array(2)
                .unwrap()
                .array(3)
                .unwrap();
            let complex = domain == ScalarDomain::Complex;
            let components = (0..6)
                .map(|i| {
                    (
                        (i + 1) as f64,
                        if complex { -((2 * i + 3) as f64) } else { 0. },
                    )
                })
                .collect::<Vec<_>>();
            let value = ValueLiteral::new(ty.clone(), components).unwrap();
            let coordinates = ScalarSymbolCoordinate::for_value(symbol, &ty).unwrap();
            assert_eq!(coordinates.len(), if complex { 12 } else { 6 });
            for (index, coordinate) in coordinates.iter().enumerate() {
                let flat = index / if complex { 2 } else { 1 };
                assert_eq!(coordinate.symbol(), symbol);
                assert_eq!(
                    coordinate.component_index(),
                    &[flat as u32 / 2, flat as u32 % 2]
                );
                let imaginary = complex && index % 2 == 1;
                assert_eq!(coordinate.is_imaginary(), imaginary);
                assert_eq!(
                    coordinate.read_value(&value).unwrap(),
                    if imaginary {
                        -((2 * flat + 3) as f64)
                    } else {
                        (flat + 1) as f64
                    }
                );
            }
        }
    }

    #[test]
    fn coordinate_reads_reject_unavailable_parts_shapes_and_derivatives() {
        let symbol = SymbolRef::Parameter(Id::new());
        let ty = ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS).unwrap();
        let parts = ScalarSymbolCoordinate::for_value(symbol, &ty).unwrap();
        let real = ValueLiteral::from_real(
            ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap(),
            4.,
        )
        .unwrap();
        assert_eq!(parts[0].read_value(&real).unwrap(), 4.);
        assert!(parts[1].read_value(&real).is_err());
        let channels = ty.array(2).unwrap();
        for coordinate in ScalarSymbolCoordinate::for_value(symbol, &channels).unwrap() {
            assert!(coordinate.read_value(&real).is_err());
        }
        let value = ValueLiteral::new(channels, vec![(2., 3.), (5., -7.)]).unwrap();
        assert!(parts[0].read_value(&value).is_err());
        let mut derivative = parts[0].clone();
        derivative.spatial_derivative_axes = vec![0].into_boxed_slice();
        assert!(derivative.read_value(&real).is_err());
        let integer = ValueLiteral::integer(
            ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS).unwrap(),
            [3],
        )
        .unwrap();
        assert!(parts[0].read_value(&integer).is_err());
        assert!(ScalarSymbolCoordinate::for_value(symbol, integer.value_type()).is_err());
    }
}
