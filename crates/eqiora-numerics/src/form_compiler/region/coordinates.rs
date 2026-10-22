//! Select numerical scalar coordinates without changing the physical basis.
use super::{Data, RegionFieldLayout, Term, invalid};
use crate::spatial_expression::Coefficient;
use eqiora_core::{Diagnostic, ScalarDomain, ValueType};

pub(crate) trait Projection<S: Coefficient>: std::fmt::Debug {
    type Scalar: Coefficient;
    fn width(field: &ValueType) -> Result<usize, Diagnostic>;
    fn columns(data: &Data<S>, dimension: usize, width: usize) -> Result<Vec<Data<S>>, Diagnostic>;
    fn component(value: S, coordinate: usize) -> Self::Scalar;
    fn physical_point<'a>(
        fields: &[RegionFieldLayout],
        ranges: &[std::ops::Range<usize>],
        point: &'a [Self::Scalar],
    ) -> std::borrow::Cow<'a, [S]>;
    fn direction(coordinate: usize) -> S;
    fn validate(term: &Term<S>, value: S, point: &[f64]) -> Result<(), Diagnostic>;

    fn ranges(layouts: &[RegionFieldLayout]) -> Result<Vec<std::ops::Range<usize>>, Diagnostic> {
        let mut offset = 0usize;
        layouts
            .iter()
            .map(|layout| {
                let size = layout
                    .range
                    .len()
                    .checked_mul(Self::width(&layout.value_type)?)
                    .ok_or_else(|| invalid("region scalar coordinate count overflow"))?;
                let end = offset
                    .checked_add(size)
                    .ok_or_else(|| invalid("region scalar coordinate count overflow"))?;
                let range = offset..end;
                offset = end;
                Ok(range)
            })
            .collect()
    }
}

pub(super) fn fields<S: Coefficient, P: Projection<S>>(
    layouts: &[RegionFieldLayout],
) -> Result<Vec<super::integration::IntegralField>, Diagnostic> {
    layouts
        .iter()
        .map(|layout| {
            let mut field = super::integration::IntegralField::from(layout);
            field.scalar_coordinates = P::width(&layout.value_type)?;
            Ok(field)
        })
        .collect()
}

/// Encode physical values without inventing imaginary coordinates for real Fields.
pub(crate) fn values<S: Coefficient, P: Projection<S>>(
    field: &ValueType,
    values: &[S],
) -> Result<Vec<P::Scalar>, Diagnostic> {
    let width = P::width(field)?;
    let count = values
        .len()
        .checked_mul(width)
        .ok_or_else(|| invalid("physical value coordinate count overflow"))?;
    let mut coordinates = Vec::with_capacity(count);
    for &value in values {
        if !value.is_finite() || (field.scalar_domain() == ScalarDomain::Real && value.im() != 0.) {
            return Err(invalid(
                "physical values must be finite and preserve the Field scalar domain",
            ));
        }
        coordinates.extend((0..width).map(|coordinate| P::component(value, coordinate)));
    }
    Ok(coordinates)
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Native;
impl<S: Coefficient> Projection<S> for Native {
    type Scalar = S;
    fn width(_: &ValueType) -> Result<usize, Diagnostic> {
        Ok(1)
    }
    fn columns(data: &Data<S>, _: usize, _: usize) -> Result<Vec<Data<S>>, Diagnostic> {
        Ok(vec![data.clone()])
    }
    fn component(value: S, _: usize) -> S {
        value
    }
    fn physical_point<'a>(
        _: &[RegionFieldLayout],
        _: &[std::ops::Range<usize>],
        point: &'a [S],
    ) -> std::borrow::Cow<'a, [S]> {
        std::borrow::Cow::Borrowed(point)
    }
    fn direction(_: usize) -> S {
        <S as From<f64>>::from(1.)
    }
    fn validate(term: &Term<S>, value: S, _: &[f64]) -> Result<(), Diagnostic> {
        term.require_coefficient(value).map(|_| ())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Real;
impl<S: Coefficient> Projection<S> for Real {
    type Scalar = f64;
    fn width(field: &ValueType) -> Result<usize, Diagnostic> {
        match field.scalar_domain() {
            ScalarDomain::Real => Ok(1),
            ScalarDomain::Complex if S::imaginary_unit().is_some() => Ok(2),
            _ => Err(invalid(
                "spatial scalar coordinates require a real or complex Field",
            )),
        }
    }
    fn columns(data: &Data<S>, dimension: usize, width: usize) -> Result<Vec<Data<S>>, Diagnostic> {
        Ok(data
            .coordinate_images(dimension)?
            .into_iter()
            .map(|(image, _)| image)
            .take(width)
            .collect())
    }
    fn component(value: S, coordinate: usize) -> f64 {
        if coordinate == 0 {
            value.re()
        } else {
            value.im()
        }
    }
    fn physical_point<'a>(
        fields: &[RegionFieldLayout],
        ranges: &[std::ops::Range<usize>],
        point: &'a [f64],
    ) -> std::borrow::Cow<'a, [S]> {
        fields
            .iter()
            .zip(ranges)
            .flat_map(|(field, range)| {
                let width = range.len() / field.range.len();
                point[range.clone()].chunks_exact(width).map(|value| {
                    S::from_parts(value[0], value.get(1).copied().unwrap_or(0.))
                        .expect("validated Field scalar domain")
                })
            })
            .collect::<Vec<_>>()
            .into()
    }
    fn direction(coordinate: usize) -> S {
        if coordinate == 0 {
            <S as From<f64>>::from(1.)
        } else {
            S::imaginary_unit().expect("validated imaginary coordinate")
        }
    }
    fn validate(term: &Term<S>, value: S, point: &[f64]) -> Result<(), Diagnostic> {
        // Positivity concerns the original constitutive coefficient, never an
        // individual off-diagonal entry of its real-coordinate matrix.
        let candidate = if term.positive_diffusion {
            term.coefficient.evaluate(point)?
        } else {
            value
        };
        term.require_coefficient(candidate).map(|_| ())
    }
}

impl Real {
    /// Decode complete physical coefficients from their exact Field coordinate words.
    pub(crate) fn decode<S: Coefficient>(
        field: &ValueType,
        words: &[f64],
    ) -> Result<Vec<S>, Diagnostic> {
        let width = <Self as Projection<S>>::width(field)?;
        if !words.len().is_multiple_of(width) || words.iter().any(|word| !word.is_finite()) {
            return Err(invalid(
                "physical Field coordinates must be complete and finite",
            ));
        }
        words
            .chunks_exact(width)
            .map(|parts| {
                S::from_parts(parts[0], parts.get(1).copied().unwrap_or(0.)).ok_or_else(|| {
                    invalid("physical Field coordinates exceed the scalar representation")
                })
            })
            .collect()
    }
}
