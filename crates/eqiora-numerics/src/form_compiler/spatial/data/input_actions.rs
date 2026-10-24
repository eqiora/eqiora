//! Prescribed scalar atoms whose derivatives are demanded after equation assembly.
use super::*;
use num_complex::Complex;
use num_rational::BigRational;
use num_traits::{One, Zero};

type Scale = Complex<BigRational>;
pub(crate) type InputTerms<S> = Vec<(InputAtom<S>, Scale)>;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct InputAtom<S: Coefficient> {
    data: Data<S>,
    imaginary: bool,
    physical: Vec<f64>,
    coordinate: ScalarSymbolCoordinate,
}

pub(crate) struct InputAction<S: Coefficient> {
    pub(crate) atom: InputAtom<S>,
    pub(crate) coefficients: Vec<BigRational>,
}

impl<S: Coefficient> InputAtom<S> {
    pub(super) fn new(
        data: Data<S>,
        physical: &[f64],
        coordinate: &ScalarSymbolCoordinate,
    ) -> Self {
        Self {
            data,
            imaginary: false,
            physical: physical.to_vec(),
            coordinate: coordinate.clone(),
        }
    }

    pub(crate) fn same_atom(&self, other: &Self) -> bool {
        self.coordinate == other.coordinate
            && self.imaginary == other.imaginary
            && ((!self.data.spatial() && !other.data.spatial()) || self.physical == other.physical)
            && self.data.same_coefficient(&other.data)
    }

    pub(crate) fn partial(&self) -> Result<f64, Diagnostic> {
        let direction = |input: &ScalarSymbolCoordinate| f64::from(input == &self.coordinate);
        self.data
            .derivative(&direction, self.physical.len())?
            .evaluate(&self.physical)
            .map(|value| {
                if self.imaginary {
                    value.im()
                } else {
                    value.re()
                }
            })
    }
}

impl<S: Coefficient> Data<S> {
    /// Keep primal domain checks before decomposition. The exact scalar IR owns
    /// pointwise algebra; compound data retain their existing derivative rules.
    pub(in crate::form_compiler) fn input_atoms(
        &self,
        physical: &[f64],
        coordinate: &ScalarSymbolCoordinate,
    ) -> Result<InputTerms<S>, Diagnostic> {
        self.evaluate(physical)?;
        self.project_input_atoms(physical, coordinate, &mut 65536)
    }

    pub(super) fn provably_real(&self) -> bool {
        if S::imaginary_unit().is_none() {
            return true;
        }
        match self.0.as_ref() {
            Node::Constant { value, .. } => value.im() == 0.,
            // These owners return a real scalar coordinate even with complex storage.
            Node::Pointwise(_) | Node::Input { .. } | Node::Part { .. } => true,
            Node::Add(a, b) | Node::Mul(a, b) | Node::Div(a, b) => {
                a.provably_real() && b.provably_real()
            }
            Node::Derivative(_, tangent) => tangent.provably_real(),
            Node::Chart(data, _) => data.provably_real(),
            Node::Affine(_) => false,
        }
    }

    fn project_input_atoms(
        &self,
        physical: &[f64],
        coordinate: &ScalarSymbolCoordinate,
        budget: &mut usize,
    ) -> Result<InputTerms<S>, Diagnostic> {
        *budget = budget.checked_sub(1).ok_or_else(|| {
            super::super::invalid("coefficient input projection exceeds its work budget")
        })?;
        let direction = |input: &ScalarSymbolCoordinate| f64::from(input == coordinate);
        if self.constant_in_direction(&direction)? {
            return Ok(Vec::new());
        }
        let exact = |value: S| -> Result<Scale, Diagnostic> {
            Ok(Scale::new(
                BigRational::from_float(value.re())
                    .ok_or_else(|| super::super::invalid("non-finite coefficient input scale"))?,
                BigRational::from_float(value.im())
                    .ok_or_else(|| super::super::invalid("non-finite coefficient input scale"))?,
            ))
        };
        let scale = |terms: InputTerms<S>, factor: Scale| -> InputTerms<S> {
            terms
                .into_iter()
                .map(|(atom, coefficient)| (atom, coefficient * &factor))
                .collect()
        };
        match self.0.as_ref() {
            Node::Pointwise(data) if data.expression_view().is_some() => {
                data.input_atoms(physical, coordinate)
            }
            Node::Add(a, b) => {
                let mut terms = a.project_input_atoms(physical, coordinate, budget)?;
                terms.extend(b.project_input_atoms(physical, coordinate, budget)?);
                Ok(terms)
            }
            Node::Mul(a, b) if a.constant_in_direction(&direction)? => Ok(scale(
                b.project_input_atoms(physical, coordinate, budget)?,
                exact(a.evaluate(physical)?)?,
            )),
            Node::Mul(a, b) if b.constant_in_direction(&direction)? => Ok(scale(
                a.project_input_atoms(physical, coordinate, budget)?,
                exact(b.evaluate(physical)?)?,
            )),
            Node::Div(a, b) if b.constant_in_direction(&direction)? => {
                let denominator = exact(b.evaluate(physical)?)?;
                if denominator.is_zero() {
                    return Err(super::super::invalid("zero coefficient input divisor"));
                }
                Ok(scale(
                    a.project_input_atoms(physical, coordinate, budget)?,
                    Scale::one() / denominator,
                ))
            }
            Node::Part { imaginary, data } => Ok(data
                .project_input_atoms(physical, coordinate, budget)?
                .into_iter()
                .map(|(atom, coefficient)| {
                    (
                        atom,
                        Scale::new(
                            if *imaginary {
                                coefficient.im
                            } else {
                                coefficient.re
                            },
                            BigRational::zero(),
                        ),
                    )
                })
                .collect()),
            _ => {
                // Non-polynomial products, charts and already differentiated data
                // remain opaque. Their own evaluator owns every derivative/domain.
                let mut terms = Vec::new();
                for imaginary in [false, true] {
                    if imaginary && self.provably_real() {
                        continue;
                    }
                    let mut atom = InputAtom::new(self.clone(), physical, coordinate);
                    atom.imaginary = imaginary;
                    let coefficient = if imaginary {
                        Scale::new(BigRational::zero(), BigRational::one())
                    } else {
                        Scale::one()
                    };
                    terms.push((atom, coefficient));
                }
                Ok(terms)
            }
        }
    }
}
