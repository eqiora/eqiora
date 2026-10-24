//! Initial coefficient partials use the ordinary local integration and State action.
use super::*;
use crate::form_compiler::spatial::data::{Data, InputAction, InputAtom};
use eqiora_ir::ScalarSymbolCoordinate;
use num_rational::BigRational;
use num_traits::Zero;
use std::cell::RefCell;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct PreparedInputAction<S: Coefficient, P: Projection<S>> {
    atom: InputAtom<S>,
    scale: BigRational,
    action: PreparedRegionCell<S, P>,
}

impl<S: Coefficient> BoundRegionForm<S> {
    pub(super) fn prepare_input_partial<P: Projection<S>>(
        &self,
        geometry: &AffineGeometryMap,
        quadrature: &QuadratureRule,
        coordinate: &ScalarSymbolCoordinate,
    ) -> Result<PreparedRegionCell<S, P>, Diagnostic> {
        let mut primal = self.clone();
        primal.form.coefficient_partial = None;
        // Components project their complete coefficient*expression product.
        // Splitting two Time-dependent factors would use a product rule before
        // proving that either factor is differentiable at the bound point.
        for row in &mut primal.form.rows {
            row.components.clear();
        }
        let samples: RefCell<Vec<(Data<S>, Vec<f64>)>> = RefCell::new(Vec::new());
        let zero = <S as From<f64>>::from(0.);
        let direction = |input: &ScalarSymbolCoordinate| f64::from(input == coordinate);
        let collect = |data: &Data<S>, point: &[f64], tangent: Option<&[f64]>| {
            if tangent.is_some() {
                return Err(invalid("input partial collection requires fixed geometry"));
            }
            if !data.evaluate(point)?.is_finite() {
                return Err(invalid("input partial requires finite primal coefficients"));
            }
            if !data.constant_in_direction(&direction)? {
                let mut samples = samples.borrow_mut();
                if !samples
                    .iter()
                    .any(|(source, physical)| physical == point && source.same_coefficient(data))
                {
                    samples.push((data.clone(), point.to_vec()));
                }
            }
            Ok((zero, zero))
        };
        // The zero-coefficient action retains the exact original inventory,
        // geometry, normalization and quadrature. Boundary partials remain in
        // their ordinary packets. No derivative has been evaluated here.
        let (mut partial, _) = primal.prepare_projected_cell_actions_with::<P>(
            geometry, geometry, None, quadrature, &collect,
        )?;
        for (source, physical) in samples.into_inner() {
            for (atom, coefficient) in source.input_atoms(&physical, coordinate)? {
                for (imaginary, scale) in [(false, coefficient.re), (true, coefficient.im)] {
                    if scale.is_zero() {
                        continue;
                    }
                    let unit = if imaginary {
                        S::imaginary_unit()
                            .ok_or_else(|| invalid("input coefficient exceeds its scalar domain"))?
                    } else {
                        <S as From<f64>>::from(1.)
                    };
                    let evaluate = |data: &Data<S>, point: &[f64], _: Option<&[f64]>| {
                        Ok((
                            if point == physical && source.same_coefficient(data) {
                                unit
                            } else {
                                zero
                            },
                            zero,
                        ))
                    };
                    let (action, _) = primal.prepare_projected_cell_actions_with::<P>(
                        geometry, geometry, None, quadrature, &evaluate,
                    )?;
                    partial.input_actions.push(PreparedInputAction {
                        atom: atom.clone(),
                        scale,
                        action,
                    });
                }
            }
        }
        Ok(partial)
    }
}

impl<S: Coefficient, P: Projection<S, Scalar = f64>> PreparedRegionCell<S, P> {
    pub(crate) fn coefficient_input_actions(
        &self,
        state: &[f64],
        rates: &[f64],
    ) -> Result<Vec<InputAction<S>>, Diagnostic> {
        self.input_actions
            .iter()
            .map(|input| {
                let residual = input
                    .action
                    .implicit_action(state, Some(rates), None)?
                    .residual;
                let coefficients = residual
                    .into_iter()
                    .map(|value| {
                        BigRational::from_float(value)
                            .map(|value| value * &input.scale)
                            .ok_or_else(|| invalid("coefficient input action is non-finite"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(InputAction {
                    atom: input.atom.clone(),
                    coefficients,
                })
            })
            .collect()
    }
}
