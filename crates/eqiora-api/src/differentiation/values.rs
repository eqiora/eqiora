//! Typed finite products with one private real-coordinate layout.

use super::*;
use eqiora_core::ValueLiteral;

/// A complete ordered product of Model-bound numerical values.
///
/// Mathematical type and coefficient support remain attached to each leaf.
/// The storage layout is derived from that ordered signature, never its identity.
#[derive(Debug, Clone, PartialEq)]
pub struct DifferentiableValues {
    signatures: Arc<[DifferentiableValueSignature]>,
    coordinates: Vec<f64>,
}

impl DifferentiableProgram {
    /// Bind private native transport to the canonical Program signature inventory.
    pub(crate) fn point_from_coordinates(
        &self,
        values: &[f64],
    ) -> Result<DifferentiableValues, Diagnostic> {
        DifferentiableValues::from_coordinates(self.identity.inputs.clone(), values.to_vec())
    }
}

impl DifferentiableValues {
    /// Bind finite mathematical values without losing units, shape, basis or frame.
    /// Spatial coefficient collections use `from_components` with their exact support.
    ///
    /// # Errors
    /// Rejects missing leaves, mismatched types, foreign Models and invalid storage.
    pub fn from_literals(
        signatures: &[DifferentiableValueSignature],
        values: &[ValueLiteral],
    ) -> Result<Self, Diagnostic> {
        if signatures.len() != values.len()
            || signatures.iter().zip(values).any(|(signature, value)| {
                signature.space().is_some()
                    || &signature.mathematical_type().value_type != value.value_type()
            })
        {
            return Err(invalid(
                "values differ from their exact mathematical signatures",
            ));
        }
        Self::pack(
            signatures,
            values
                .iter()
                .map(|value| value.components().expect("admitted real/complex value")),
        )
    }

    /// Bind coherent-SI real/imaginary components in each leaf's row-major order.
    ///
    /// A spatial leaf follows its signature's coefficient order, then intrinsic
    /// component order. Each pair is one mathematical component, not two axes.
    ///
    /// # Errors
    /// Rejects wrong cardinality, nonfinite values or an imaginary part in a real leaf.
    pub fn from_components(
        signatures: &[DifferentiableValueSignature],
        values: &[&[(f64, f64)]],
    ) -> Result<Self, Diagnostic> {
        if signatures.len() != values.len()
            || signatures.iter().zip(values).any(|(signature, value)| {
                value.len() != signature.coordinate_count() / signature.width()
            })
        {
            return Err(invalid(
                "component counts differ from the exact value signatures",
            ));
        }
        Self::pack(signatures, values.iter().map(|value| value.iter().copied()))
    }

    fn pack<I: ExactSizeIterator<Item = (f64, f64)>>(
        signatures: &[DifferentiableValueSignature],
        values: impl ExactSizeIterator<Item = I>,
    ) -> Result<Self, Diagnostic> {
        validate_signatures(signatures)?;
        let count = signature::dimension(signatures)?;
        let mut coordinates = Vec::new();
        coordinates
            .try_reserve_exact(count)
            .map_err(|_| invalid("value coordinates cannot be allocated"))?;
        for (signature, components) in signatures.iter().zip(values) {
            for (re, im) in components {
                if !re.is_finite() || !im.is_finite() || (signature.width() == 1 && im != 0.0) {
                    return Err(invalid(
                        "value component differs from its finite real/complex domain",
                    ));
                }
                coordinates.push(re);
                if signature.width() == 2 {
                    coordinates.push(im);
                }
            }
        }
        Self::from_coordinates(signatures.to_vec().into(), coordinates)
    }

    pub(super) fn from_coordinates(
        signatures: Arc<[DifferentiableValueSignature]>,
        coordinates: Vec<f64>,
    ) -> Result<Self, Diagnostic> {
        if signature::dimension(&signatures)? != coordinates.len()
            || coordinates.iter().any(|value| !value.is_finite())
        {
            return Err(invalid(
                "coordinates differ from the complete finite value signatures",
            ));
        }
        Ok(Self {
            signatures,
            coordinates,
        })
    }

    pub(super) fn validate_signatures(
        &self,
        expected: &[DifferentiableValueSignature],
    ) -> Result<(), Diagnostic> {
        if self.signatures() != expected {
            return Err(invalid(
                "value product differs from the exact ordered base-space signatures",
            ));
        }
        Ok(())
    }

    /// Ordered exact semantic leaves, independent of their private storage offsets.
    #[must_use]
    pub fn signatures(&self) -> &[DifferentiableValueSignature] {
        &self.signatures
    }

    /// Real storage projection in signature, component, then scalar-part order.
    #[must_use]
    pub fn coordinates(&self) -> &[f64] {
        &self.coordinates
    }

    pub(super) fn components(
        &self,
        index: usize,
    ) -> Option<impl ExactSizeIterator<Item = (f64, f64)> + '_> {
        let (signature, range) = signature::layout(&self.signatures).nth(index)?;
        Some(components(signature, &self.coordinates[range]))
    }

    /// Borrow a typed leaf by its position in the selected product.
    #[must_use]
    pub fn leaf(&self, index: usize) -> Option<DifferentiableValue<'_>> {
        let (signature, range) = signature::layout(&self.signatures).nth(index)?;
        Some(DifferentiableValue {
            signature,
            coordinates: &self.coordinates[range],
        })
    }

    /// Borrow the exact selected entity; equal shape or units never substitute for identity.
    #[must_use]
    pub fn get(&self, entity: RawId) -> Option<DifferentiableValue<'_>> {
        self.leaf(
            self.signatures
                .iter()
                .position(|value| value.entity() == entity)?,
        )
    }
}

/// A borrowed semantic value, retaining its exact Model, type and coefficient support.
#[derive(Debug, Clone, Copy)]
pub struct DifferentiableValue<'a> {
    signature: &'a DifferentiableValueSignature,
    coordinates: &'a [f64],
}

impl DifferentiableValue<'_> {
    /// Complete semantic identity and native shape of this leaf.
    #[must_use]
    pub const fn signature(&self) -> &DifferentiableValueSignature {
        self.signature
    }

    /// Logical real/imaginary components; real leaves always have zero imaginary part.
    pub fn components(&self) -> impl ExactSizeIterator<Item = (f64, f64)> + '_ {
        components(self.signature, self.coordinates)
    }

    /// Recover one coefficient with its original mathematical type.
    /// Finite values have exactly one coefficient at index zero.
    ///
    /// # Errors
    /// Rejects an out-of-range coefficient or an allocation failure.
    pub fn coefficient(&self, index: usize) -> Result<ValueLiteral, Diagnostic> {
        let count = if self.signature.space().is_some() {
            self.signature.coefficient_entities().len()
        } else {
            1
        };
        if index >= count {
            return Err(invalid("value coefficient index is out of bounds"));
        }
        let components = self
            .signature
            .mathematical_type()
            .value_type
            .shape()
            .component_count()
            .expect("admitted component shape");
        ValueLiteral::new(
            self.signature.mathematical_type().value_type.clone(),
            self.components().skip(index * components).take(components),
        )
        .map_err(|error| invalid(format!("cannot recover a typed coefficient: {error}")))
    }
}

fn validate_signatures(signatures: &[DifferentiableValueSignature]) -> Result<(), Diagnostic> {
    for (index, value) in signatures.iter().enumerate() {
        if value.model() != signatures[0].model()
            || signatures[..index]
                .iter()
                .any(|prior| prior.entity() == value.entity())
        {
            return Err(invalid(
                "value signatures require distinct entities from one exact Model",
            ));
        }
    }
    Ok(())
}

fn components<'a>(
    signature: &DifferentiableValueSignature,
    coordinates: &'a [f64],
) -> impl ExactSizeIterator<Item = (f64, f64)> + 'a {
    coordinates
        .chunks_exact(signature.width())
        .map(|value| (value[0], value.get(1).copied().unwrap_or(0.0)))
}
