use std::ops::ControlFlow;

/// Advance one ephemeral prepared occurrence through accepted action boundaries.
///
/// `advance` may construct an arbitrary candidate action, but `accept` is the
/// only mutation of the authoritative context. A failed candidate therefore
/// leaves the last accepted context unchanged. The prepared payload is built
/// exactly once and cannot receive a replacement binding during the loop.
pub(crate) fn advance_prepared_actions<C, P, A, B, E>(
    mut context: C,
    maximum_actions: usize,
    prepare: impl FnOnce(&C) -> Result<P, E>,
    mut step_span: impl FnMut(usize, &C) -> tracing::Span,
    mut advance: impl FnMut(&mut P, &C) -> Result<A, E>,
    mut accept: impl FnMut(&mut C, usize, A) -> Result<(), E>,
    mut stop_at_boundary: impl FnMut(usize, &C) -> Option<B>,
) -> Result<ControlFlow<B, C>, E> {
    let mut prepared = {
        let _setup =
            eqiora_execution::telemetry_span!(setup("discretization_preparation")).entered();
        prepare(&context)?
    };
    if let Some(stopped) = stop_at_boundary(0, &context) {
        return Ok(ControlFlow::Break(stopped));
    }
    for accepted_actions in 1..=maximum_actions {
        let _step = step_span(accepted_actions, &context).entered();
        let candidate = advance(&mut prepared, &context)?;
        accept(&mut context, accepted_actions, candidate)?;
        if let Some(stopped) = stop_at_boundary(accepted_actions, &context) {
            return Ok(ControlFlow::Break(stopped));
        }
    }
    Ok(ControlFlow::Continue(context))
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::ops::ControlFlow;

    use super::advance_prepared_actions;

    #[test]
    fn preparation_is_once_and_failed_candidates_are_transactional() {
        let preparations = Cell::new(0);
        let advances = Cell::new(0);
        let accepted = Cell::new(0);
        let result = advance_prepared_actions(
            0_u64,
            10,
            |_| {
                preparations.set(preparations.get() + 1);
                Ok::<_, &'static str>(())
            },
            |step, _| eqiora_execution::telemetry_span!(time_step(step, step as f64, 1.0)),
            |(), state| {
                advances.set(advances.get() + 1);
                if *state == 4 {
                    Err("candidate failed")
                } else {
                    Ok(state + 1)
                }
            },
            |state, _, candidate| {
                *state = candidate;
                accepted.set(accepted.get() + 1);
                Ok(())
            },
            |_, _| None::<()>,
        );
        assert_eq!(result, Err("candidate failed"));
        assert_eq!(preparations.get(), 1);
        assert_eq!(advances.get(), 5);
        assert_eq!(accepted.get(), 4);
    }

    #[test]
    fn cancellation_observes_only_exact_accepted_boundaries() {
        for cancellation_boundary in [0, 4] {
            let boundaries = Cell::new(0);
            let result = advance_prepared_actions(
                0_u64,
                10,
                |_| Ok::<_, ()>(()),
                |step, _| eqiora_execution::telemetry_span!(time_step(step, step as f64, 1.0)),
                |(), state| Ok(*state + 1),
                |state, _, candidate| {
                    *state = candidate;
                    Ok(())
                },
                |accepted_actions, state| {
                    boundaries.set(boundaries.get() + 1);
                    (accepted_actions == cancellation_boundary).then_some(*state)
                },
            )
            .unwrap();
            assert_eq!(result, ControlFlow::Break(cancellation_boundary as u64));
            assert_eq!(boundaries.get(), cancellation_boundary + 1);
        }
    }
}
