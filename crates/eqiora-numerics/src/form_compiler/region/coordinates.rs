//! Select numerical scalar coordinates without changing the physical basis.
use super::{Data, Term, invalid};
use crate::spatial_expression::Coefficient;
use eqiora_core::{Diagnostic, ScalarDomain, ValueType};
use num_complex::Complex64;

pub(crate) trait Projection<S: Coefficient> {
    type Scalar: Coefficient;
    fn width(field: &ValueType) -> Result<usize, Diagnostic>;
    fn columns(data: &Data<S>, dimension: usize, width: usize) -> Result<Vec<Data<S>>, Diagnostic>;
    fn component(value: S, coordinate: usize) -> Self::Scalar;
    fn validate(term: &Term<S>, value: S, point: &[f64]) -> Result<(), Diagnostic>;
}

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
    fn validate(term: &Term<S>, value: S, _: &[f64]) -> Result<(), Diagnostic> {
        term.require_coefficient(value).map(|_| ())
    }
}

pub(crate) struct Real;
impl Projection<Complex64> for Real {
    type Scalar = f64;
    fn width(field: &ValueType) -> Result<usize, Diagnostic> {
        match field.scalar_domain() {
            ScalarDomain::Real => Ok(1),
            ScalarDomain::Complex => Ok(2),
            _ => Err(invalid(
                "spatial scalar coordinates require a real or complex Field",
            )),
        }
    }
    fn columns(
        data: &Data<Complex64>,
        dimension: usize,
        width: usize,
    ) -> Result<Vec<Data<Complex64>>, Diagnostic> {
        Ok(data
            .coordinate_images(dimension)?
            .into_iter()
            .take(width)
            .collect())
    }
    fn component(value: Complex64, coordinate: usize) -> f64 {
        if coordinate == 0 { value.re } else { value.im }
    }
    fn validate(term: &Term<Complex64>, value: Complex64, point: &[f64]) -> Result<(), Diagnostic> {
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
