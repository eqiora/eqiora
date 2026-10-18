//! Select numerical scalar coordinates without changing the physical basis.
use super::super::RegionFieldLayout;
use super::*;
use num_complex::Complex64;

pub(super) trait Projection<S: Coefficient> {
    type Scalar: Coefficient;
    fn width(field: &RegionFieldLayout) -> Result<usize, Diagnostic>;
    fn columns(
        data: &super::super::Data<S>,
        dimension: usize,
        width: usize,
    ) -> Result<Vec<super::super::Data<S>>, Diagnostic>;
    fn component(value: S, coordinate: usize) -> Self::Scalar;
    fn validate(term: &super::super::Term<S>, value: S, point: &[f64]) -> Result<(), Diagnostic>;
}

pub(super) struct Native;
impl<S: Coefficient> Projection<S> for Native {
    type Scalar = S;
    fn width(_: &RegionFieldLayout) -> Result<usize, Diagnostic> {
        Ok(1)
    }
    fn columns(
        data: &super::super::Data<S>,
        _: usize,
        _: usize,
    ) -> Result<Vec<super::super::Data<S>>, Diagnostic> {
        Ok(vec![data.clone()])
    }
    fn component(value: S, _: usize) -> S {
        value
    }
    fn validate(term: &super::super::Term<S>, value: S, _: &[f64]) -> Result<(), Diagnostic> {
        term.require_coefficient(value).map(|_| ())
    }
}

pub(super) struct Real;
impl Projection<Complex64> for Real {
    type Scalar = f64;
    fn width(field: &RegionFieldLayout) -> Result<usize, Diagnostic> {
        field.real_scalar_coordinates()
    }
    fn columns(
        data: &super::super::Data<Complex64>,
        dimension: usize,
        width: usize,
    ) -> Result<Vec<super::super::Data<Complex64>>, Diagnostic> {
        Ok(data
            .coordinate_images(dimension)?
            .into_iter()
            .take(width)
            .collect())
    }
    fn component(value: Complex64, coordinate: usize) -> f64 {
        if coordinate == 0 { value.re } else { value.im }
    }
    fn validate(
        term: &super::super::Term<Complex64>,
        value: Complex64,
        point: &[f64],
    ) -> Result<(), Diagnostic> {
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
