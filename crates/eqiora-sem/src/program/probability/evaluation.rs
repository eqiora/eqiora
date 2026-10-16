//! Bind exact named outcomes around the canonical expression evaluator.

use super::*;
use crate::{EvaluationInput, EvaluationPoint};
use eqiora_core::ValueLiteral;
use eqiora_schema::kernel::{ObservableDef, ObservableReduction};
use std::cell::Cell;

type Resolver<'a> =
    dyn FnMut(EvaluationInput, Option<&EvaluationPoint>) -> Result<ValueLiteral, Diagnostic> + 'a;

pub(in crate::program) fn evaluate_finite_moment(
    program: &KernelProgram,
    definition: &ObservableDef,
    resolve: &mut Resolver<'_>,
) -> Result<ValueLiteral, Diagnostic> {
    let ObservableReduction::ProbabilityMoment {
        distribution,
        moment,
    } = definition.reduction()
    else {
        unreachable!("probability moment dispatch")
    };
    let Some(KernelNode::Relation(law)) = program.node(distribution.erase()) else {
        return Err(kernel_error(
            definition.id().erase(),
            "probability law is unavailable",
        ));
    };
    let RelationMeaning::ProbabilityMass { outcomes, .. } = law.meaning() else {
        return Err(kernel_error(
            definition.id().erase(),
            "continuous probability moments require the admitted integration owner",
        ));
    };
    let Some(KernelNode::Enum(outcomes)) = program.node(outcomes.erase()) else {
        return Err(kernel_error(
            distribution.erase(),
            "finite probability outcomes are unavailable",
        ));
    };
    let remaining = Cell::new(1_000_000);
    let weights = crate::evaluate::evaluate_with_points(
        program,
        distribution.erase(),
        law.expression(),
        None,
        &remaining,
        &mut |input, point| {
            let EvaluationInput::Value(SymbolRef::Parameter(id)) = input else {
                return Err(kernel_error(
                    distribution.erase(),
                    "mass weights require Parameter inputs",
                ));
            };
            let value = resolve(input, point)?;
            if program
                .typed_value(id.erase())
                .is_none_or(|declared| declared.value_type() != value.value_type())
            {
                return Err(kernel_error(
                    distribution.erase(),
                    "probability Parameter value differs from its admitted type",
                ));
            }
            Ok(value)
        },
    )?;
    let weights = weights[0]
        .components()
        .expect("admitted mass weights")
        .map(|(value, _)| value)
        .collect::<Vec<_>>();
    let mut rows = Vec::with_capacity(outcomes.members().len());
    for tag in 0..outcomes.members().len() {
        let mut context = ProbabilityContext {
            program,
            distribution,
            outcome: Some(outcomes.value(tag as u32)?),
            resolve,
            remaining: &remaining,
            values: BTreeMap::new(),
        };
        rows.push(context.expression(definition.id().erase(), definition.expression(), None, 0)?);
    }
    let result = moment.evaluate(&rows, &weights)?;
    if result.value_type() != definition.value_type() {
        return Err(kernel_error(
            definition.id().erase(),
            "probability moment differs from its admitted result type",
        ));
    }
    Ok(result)
}

pub(in crate::program) fn evaluate_integrands(
    program: &KernelProgram,
    definition: &ObservableDef,
    point: &EvaluationPoint,
    remaining: &Cell<usize>,
    resolve: &mut Resolver<'_>,
) -> Result<Vec<ValueLiteral>, Diagnostic> {
    let mut context = ProbabilityContext {
        program,
        distribution: definition
            .reduction()
            .distribution()
            .expect("bound probability law"),
        outcome: None,
        resolve,
        remaining,
        values: BTreeMap::new(),
    };
    context.expression(
        definition.id().erase(),
        definition.expression(),
        Some(point),
        0,
    )
}

struct ProbabilityContext<'a, 'r> {
    program: &'a KernelProgram,
    distribution: Id<kinds::Relation>,
    outcome: Option<ValueLiteral>,
    resolve: &'r mut Resolver<'r>,
    remaining: &'a Cell<usize>,
    values: BTreeMap<RawId, (Option<EvaluationPoint>, ValueLiteral)>,
}

impl ProbabilityContext<'_, '_> {
    fn expression(
        &mut self,
        owner: RawId,
        expression: &ExprDag,
        point: Option<&EvaluationPoint>,
        depth: usize,
    ) -> Result<Vec<ValueLiteral>, Diagnostic> {
        if depth > 128 {
            return Err(kernel_error(
                owner,
                "probability evaluation exceeds its dependency depth bound",
            ));
        }
        crate::evaluate::evaluate_with_points(
            self.program,
            owner,
            expression,
            point,
            self.remaining,
            &mut |input, point| match input {
                EvaluationInput::Value(SymbolRef::Outcome(distribution)) => {
                    if distribution != self.distribution {
                        return Err(kernel_error(
                            owner,
                            "finite moment contains a foreign outcome binding",
                        ));
                    }
                    self.outcome.clone().ok_or_else(|| {
                        kernel_error(
                            owner,
                            "a continuous probability law has no named finite outcome",
                        )
                    })
                }
                EvaluationInput::Value(SymbolRef::Observable(id)) => {
                    let Some(KernelNode::Observable(variable)) = self.program.node(id.erase())
                    else {
                        return Err(kernel_error(
                            owner,
                            "random variable is outside the exact Model",
                        ));
                    };
                    if let ObservableReduction::RandomVariable { distribution } =
                        variable.reduction()
                    {
                        if distribution != self.distribution {
                            return Err(kernel_error(
                                owner,
                                "paired random variables require the same exact law",
                            ));
                        }
                        // A rebinding or pullback may request this same variable at
                        // another coordinate. Keep one bounded cache entry per variable,
                        // and reuse it only in its complete evaluation context.
                        if let Some((cached_point, value)) = self.values.get(&id.erase())
                            && cached_point.as_ref() == point
                        {
                            return Ok(value.clone());
                        }
                        let expression = variable.expression().clone();
                        let value = self
                            .expression(id.erase(), &expression, point, depth + 1)?
                            .remove(0);
                        if value.value_type() != variable.value_type() {
                            return Err(kernel_error(
                                id.erase(),
                                "random-variable value has the wrong admitted type",
                            ));
                        }
                        self.values
                            .insert(id.erase(), (point.cloned(), value.clone()));
                        Ok(value)
                    } else {
                        (self.resolve)(input, point)
                    }
                }
                _ => (self.resolve)(input, point),
            },
        )
    }
}
