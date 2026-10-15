//! One typed weighted reduction for native accepted-step samples.
use super::*;

pub(super) fn integrate<T>(
    steps: &[T],
    interval: &interval::TimeObservationInterval,
    rule: TimeObservationInterpolation,
    value_type: ValueType,
    bounds: impl Fn(&T) -> [f64; 2],
    mut evaluate: impl FnMut(&T, usize) -> Result<ValueLiteral, Diagnostic>,
) -> Result<ValueLiteral, Diagnostic> {
    let count = value_type
        .shape()
        .component_count()
        .ok_or_else(|| invalid("time Observable shape overflows"))?;
    let mut integral = vec![(0., 0.); count];
    for step in steps {
        let Some(weights) = interval.weights(bounds(step), rule) else {
            continue;
        };
        for (sample, weight) in weights
            .into_iter()
            .enumerate()
            .filter(|(_, weight)| *weight != 0.)
        {
            let value = evaluate(step, sample)?;
            if value.value_type() != &value_type {
                return Err(invalid(
                    "time Observable changes physical type across accepted steps",
                ));
            }
            for (index, sum) in integral.iter_mut().enumerate() {
                let value = value
                    .component(index)
                    .ok_or_else(|| invalid("non-numeric time Observable"))?;
                sum.0 += weight * value.0;
                sum.1 += weight * value.1;
            }
        }
    }
    ValueLiteral::new(time_integral_type(&value_type)?, integral)
        .map_err(|error| invalid(error.to_string()))
}

pub(super) fn time_integral_type(value_type: &ValueType) -> Result<ValueType, Diagnostic> {
    let seconds = DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("exact seconds");
    let dimension = value_type
        .dimension()
        .mul(seconds)
        .ok_or_else(|| invalid("time Observable dimension exceeds its exact representation"))?;
    value_type
        .clone()
        .with_dimension(dimension)
        .map_err(|error| invalid(error.to_string()))
}
