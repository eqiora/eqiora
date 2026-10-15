//! Private dense FFI projection onto the accepted map and product owners.

use std::sync::Arc;

use eqiora::Diagnostic;
use eqiora::api::{DifferentiableProgram, EvaluationMapPlan, EvaluationMapProducts};

use super::kernel::{Action, ActionResult, FailureKind, HandlerFailure};

// The same bounded retained-buffer profile as the ordinary Python map default.
// These are separate map/product limits, not a process or peak-memory budget.
pub(super) const NUMERICAL_BYTES_LIMIT: usize = 67_108_864;

pub(super) struct Layout {
    pub parameters: Vec<usize>,
    pub direction: Vec<usize>,
    pub first_output: Vec<usize>,
    pub second_output: Option<Vec<usize>>,
    point_shape: Vec<usize>,
    seed_shape: Vec<usize>,
    point_axes: Vec<usize>,
}

impl Layout {
    pub(super) fn new(
        descriptor: &str,
        action: Action,
        inputs: usize,
        outputs: usize,
    ) -> Result<Self, HandlerFailure> {
        if descriptor.len() > 32 * 22 || inputs == 0 || outputs == 0 {
            return Err(HandlerFailure::invalid("invalid JAX batch signature"));
        }
        let mut grid = Vec::new();
        let mut point_shape = Vec::new();
        let mut seed_shape = Vec::new();
        let mut point_axes = Vec::new();
        if !descriptor.is_empty() {
            for (axis, field) in descriptor.split(',').enumerate() {
                if axis >= 32
                    || field.len() < 2
                    || !field.as_bytes()[1..].iter().all(u8::is_ascii_digit)
                {
                    return Err(HandlerFailure::invalid(
                        "invalid JAX point/seed axis metadata",
                    ));
                }
                let extent = field[1..].parse::<usize>().map_err(|_| {
                    HandlerFailure::invalid("JAX batch dimension is not addressable")
                })?;
                grid.push(extent);
                match field.as_bytes()[0] {
                    b'p' => {
                        point_axes.push(axis);
                        point_shape.push(extent);
                    }
                    b's' if !matches!(action, Action::Primal) => seed_shape.push(extent),
                    _ => return Err(HandlerFailure::invalid("invalid JAX point/seed axis role")),
                }
            }
        }
        let parameters = components(&point_shape, inputs)?;
        let primal = components(&point_shape, outputs)?;
        let direction = components(
            &grid,
            if matches!(action, Action::Vjp) {
                outputs
            } else {
                inputs
            },
        )?;
        let product = components(
            &grid,
            if matches!(action, Action::Vjp) {
                inputs
            } else {
                outputs
            },
        )?;
        Ok(Self {
            parameters,
            direction,
            first_output: if matches!(action, Action::Vjp) {
                product.clone()
            } else {
                primal
            },
            second_output: matches!(action, Action::Jvp).then_some(product),
            point_shape,
            seed_shape,
            point_axes,
        })
    }
}

fn components(shape: &[usize], width: usize) -> Result<Vec<usize>, HandlerFailure> {
    let mut shape = shape.to_vec();
    shape.push(width);
    elements(&shape)?;
    Ok(shape)
}

pub(super) fn elements(shape: &[usize]) -> Result<usize, HandlerFailure> {
    let mut count = 1usize;
    let mut strides = 1usize;
    for &extent in shape {
        count = count.checked_mul(extent).ok_or_else(shape_overflow)?;
        strides = strides
            .checked_mul(extent.max(1))
            .ok_or_else(shape_overflow)?;
    }
    if strides > isize::MAX as usize / size_of::<f64>()
        || count > NUMERICAL_BYTES_LIMIT / size_of::<f64>()
    {
        return Err(shape_overflow());
    }
    Ok(count)
}

fn shape_overflow() -> HandlerFailure {
    HandlerFailure::invalid("JAX shape/stride/byte product exceeds the numerical buffer bound")
}

pub(super) fn compute_action(
    action: Action,
    program: Arc<DifferentiableProgram>,
    layout: &Layout,
    parameters: &[f64],
    direction: Option<&[f64]>,
) -> Result<ActionResult, HandlerFailure> {
    let width = program.identity().input_dimension();
    for (index, value) in parameters.iter().enumerate() {
        if !value.is_finite() {
            return Err(HandlerFailure::invalid(format!(
                "JAX point occurrence {}, coordinate {} must be finite",
                index / width,
                index % width
            )));
        }
    }
    if let Some(values) = direction {
        let width = *layout.direction.last().expect("component-bearing shape");
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(HandlerFailure::invalid(format!(
                "JAX derivative grid occurrence {}, coordinate {} must be finite",
                index / width,
                index % width
            )));
        }
    }
    let points = crate::differentiation::values::points_from_coordinates(
        program.identity().inputs(),
        parameters,
        parameters.len() / width,
        NUMERICAL_BYTES_LIMIT,
    )
    .map_err(|error| diagnostics_failure(&[error]))?;
    let plan = EvaluationMapPlan::new(
        program,
        &points,
        &[],
        eqiora::api::EvaluationMapExecutionPolicy::retained(NUMERICAL_BYTES_LIMIT),
    )
    .map_err(|error| diagnostics_failure(&[error]))?;
    let map = plan.execute().map_err(|report| {
        let mut error = diagnostics_failure(report.diagnostics());
        error.message = format!(
            "JAX point occurrence {} failed: {}",
            report.stopped_index(),
            error.message
        );
        error
    })?;
    let members = map.members().ok_or_else(|| {
        HandlerFailure::internal("dense JAX map requires retained native members")
    })?;
    let primal = || {
        members
            .iter()
            .flat_map(|evaluation| evaluation.primal().output().coordinates().to_vec())
            .collect()
    };
    if matches!(action, Action::Primal) {
        return Ok(ActionResult::Primal(primal()));
    }
    let products = EvaluationMapProducts::new(
        &map,
        &layout.point_shape,
        &layout.seed_shape,
        &layout.point_axes,
        NUMERICAL_BYTES_LIMIT,
    )
    .map_err(|error| diagnostics_failure(&[error]))?;
    let direction =
        direction.ok_or_else(|| HandlerFailure::internal("JAX derivative direction is absent"))?;
    match action {
        Action::Jvp => {
            let product = products
                .jvp(&[], direction)
                .map_err(|error| diagnostics_failure(&[error]))?;
            Ok(ActionResult::Jvp {
                primal: primal(),
                tangent: product
                    .products()
                    .iter()
                    .flat_map(|product| {
                        crate::differentiation::values::coordinate_projection(
                            product.derivative().base_signatures(),
                            |index| {
                                product
                                    .derivative()
                                    .components(index)
                                    .expect("admitted leaf")
                            },
                        )
                    })
                    .collect(),
            })
        }
        Action::Vjp => {
            let product = products
                .vjp(direction)
                .map_err(|error| diagnostics_failure(&[error]))?;
            Ok(ActionResult::Vjp(
                product
                    .mapped_cotangents()
                    .iter()
                    .flat_map(|value| {
                        crate::differentiation::values::coordinate_projection(
                            value.base_signatures(),
                            |index| value.components(index).expect("admitted leaf"),
                        )
                    })
                    .collect(),
            ))
        }
        Action::Primal => unreachable!("primal returned above"),
    }
}

fn diagnostics_failure(diagnostics: &[Diagnostic]) -> HandlerFailure {
    HandlerFailure::new(
        FailureKind::FailedPrecondition,
        diagnostics
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("; "),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axes_preserve_point_seed_roles_and_all_buffer_shapes() {
        let layout = Layout::new("s2,p3,s4,p2", Action::Jvp, 3, 5).unwrap();
        assert_eq!(layout.point_shape, [3, 2]);
        assert_eq!(layout.seed_shape, [2, 4]);
        assert_eq!(layout.point_axes, [1, 3]);
        assert_eq!(layout.parameters, [3, 2, 3]);
        assert_eq!(layout.first_output, [3, 2, 5]);
        assert_eq!(layout.direction, [2, 3, 4, 2, 3]);
        assert_eq!(layout.second_output.unwrap(), [2, 3, 4, 2, 5]);
    }

    #[test]
    fn metadata_empty_singleton_and_overflow_admission_is_fail_closed() {
        assert_eq!(
            Layout::new("", Action::Primal, 2, 3).unwrap().parameters,
            [2]
        );
        assert_eq!(
            Layout::new("p0", Action::Primal, 2, 3)
                .unwrap()
                .first_output,
            [0, 3]
        );
        assert_eq!(
            Layout::new("s0", Action::Jvp, 2, 3).unwrap().first_output,
            [3]
        );
        for descriptor in [
            "s2",
            "p-1",
            "p+1",
            "p",
            "x2",
            "p1,",
            "p18446744073709551616",
        ] {
            assert!(
                Layout::new(descriptor, Action::Primal, 2, 3).is_err(),
                "{descriptor}"
            );
        }
        assert!(Layout::new(&vec!["p1"; 33].join(","), Action::Primal, 2, 3).is_err());
        assert!(Layout::new(&format!("p0,p{}", usize::MAX), Action::Primal, 2, 3).is_err());
        assert!(Layout::new("p8388608", Action::Primal, 2, 3).is_err());
    }
}
