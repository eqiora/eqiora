//! Probability integrands use the existing point evaluator and factor quadrature.
use super::*;
use eqiora_schema::kernel::{ObservableMeasure, ProbabilityMoment};

impl Context<'_> {
    pub(super) fn evaluate_continuous_moment(
        &mut self,
        definition: &ObservableDef,
        domain: Id<kinds::Domain>,
        measure: ObservableMeasure,
        moment: ProbabilityMoment,
        point: &Point,
        depth: usize,
    ) -> Result<Evaluation, Diagnostic> {
        let typed = self
            .program
            .typed_observable(definition.id())
            .map_err(|errors| errors.into_iter().next().expect("failed typing"))?;
        let projected =
            ScalarOperatorIr::lower_affine_map_density(&typed, &mut |symbol| self.resolve(symbol))?;
        self.require_regular_density(&projected, depth)?;
        let selected = crate::factor_measure::axes(self.program, domain)?;
        let mut rows = Vec::new();
        let mut weights = Vec::new();
        self.visit_factor_points(
            domain,
            measure,
            &selected,
            point,
            |context, local, weight| {
                let selected = context.evaluation_point(domain, local)?;
                let program = context.program;
                let remaining = context.remaining;
                let row = program.evaluate_observable_integrands_with_points(
                    definition.id(),
                    &selected,
                    remaining,
                    &mut |input, selected| context.resolve_at(input, selected, local, depth),
                )?;
                rows.push(row);
                // Native admission proves a positive constant density on bounded
                // factors. It cancels exactly in normalization; retaining this ratio
                // avoids overflow from an arbitrary proportional density scale.
                weights.push(weight.value());
                Ok(())
            },
        )?;
        let value = moment.evaluate(&rows, &weights)?;
        if value.value_type() != definition.value_type() {
            return Err(invalid("probability moment differs from its admitted type"));
        }
        let evaluation = (value, None);
        self.accepted.insert(definition.id(), evaluation.clone());
        Ok(evaluation)
    }
}
