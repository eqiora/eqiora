//! Exhaustive finite-law score identity through typed native AD and shared moments.
use super::*;
use crate::{DifferentiableValues, ModelDocument, differentiation::signature};
use eqiora_core::{DimExponents, Id, ScalarDomain, entity::kinds};
use eqiora_ir::{DifferentiationRole, LinearizedRelation, RelationTangent, ScalarOperatorIr};
use eqiora_schema::kernel::{
    ExprDagBuilder, KernelNode, ObservableReduction, RelationMeaning, SymbolRef, UnaryMathFunction,
};
use eqiora_sem::KernelProgram;
use std::sync::Arc;

impl MomentEstimator<'_> {
    /// Evaluate an exhaustive finite-law expectation and its score gradient.
    ///
    /// Every nominal outcome participates in declaration order. Its strictly
    /// positive mass and real scalar loss come from the exact Model, at one
    /// frozen design point. Native AD computes `dL` and `dlog(w)`; the shared
    /// weighted-moment owner supplies `E[dlog(w)]`, so the normalized score is
    /// `dlog(p) = dlog(w) - E[dlog(w)]`. The result is `E[dL + L*dlog(p)]`.
    /// There is no baseline, sampling, retry, success filtering or IID assumption.
    ///
    /// This bounded reference admits real scalar design Parameters and a loss
    /// expression using Parameters, its own nominal outcome and pure operators.
    /// Other Parameters remain frozen at Model defaults. Native typed AD owns
    /// local regularity and rejects active comparison boundaries. Strictly
    /// positive masses give fixed finite support locally; no infinite-support
    /// interchange, trajectory derivative or higher-order estimator is admitted.
    ///
    /// The byte limit bounds retained and construction binary64 buffers, excluding
    /// Model/signature metadata, nominal tags, allocator overhead, exact-arithmetic
    /// accumulators and transient native-AD scratch. A separate structural work
    /// bound limits outcome/design/expression products before enumeration.
    /// # Errors
    /// Rejects foreign or incomplete types, nonfinite/zero/negative masses,
    /// unsupported loss dependencies, invalid local derivatives and resource limits.
    pub fn enumerated_score(
        document: &ModelDocument,
        variable: &ModelEntityRef<Id<kinds::Observable>>,
        point: &DifferentiableValues,
        numerical_bytes_limit: usize,
    ) -> Result<MomentEstimator<'static, DifferentiableCotangent>, Diagnostic> {
        let model = document.artifact_reference()?;
        let kernel = document.program();
        if variable.model() != &model || point.signatures().is_empty() {
            return Err(invalid(
                "finite score requires an exact Model loss and nonempty design",
            ));
        }
        let parameters = point
            .signatures()
            .iter()
            .map(|input| {
                if input.model() != &model
                    || !input.shape().is_empty()
                    || input.mathematical_type().value_type.scalar_domain() != ScalarDomain::Real
                {
                    return Err(invalid(
                        "finite score design requires exact real scalar Parameters",
                    ));
                }
                input
                    .entity()
                    .downcast::<kinds::Parameter>()
                    .ok_or_else(|| invalid("finite score inputs must be Parameters"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if signature::parameters(kernel, &model, &parameters)? != point.signatures() {
            return Err(invalid(
                "finite score point differs from the exact Model input types",
            ));
        }
        let Some(KernelNode::Observable(loss)) = kernel.node(variable.id().erase()) else {
            return Err(invalid("finite score loss is outside the Model"));
        };
        let ObservableReduction::RandomVariable { distribution } = loss.reduction() else {
            return Err(invalid(
                "finite score requires a declared random-variable loss",
            ));
        };
        let Some(KernelNode::Relation(law)) = kernel.node(distribution.erase()) else {
            return Err(invalid("finite score law is unavailable"));
        };
        let RelationMeaning::ProbabilityMass { outcomes, weights } = law.meaning() else {
            return Err(invalid(
                "finite score requires a declared named-outcome mass law",
            ));
        };
        let Some(KernelNode::Enum(outcomes)) = kernel.node(outcomes.erase()) else {
            return Err(invalid("finite score outcome declaration is unavailable"));
        };
        ProbabilityMoment::Expectation.output_type(&[loss.value_type()])?;
        let count = outcomes.members().len();
        let width = parameters.len();
        resources(
            count,
            width,
            law.expression().nodes().len(),
            loss.expression().nodes().len(),
            numerical_bytes_limit,
        )?;
        let mut builder = ExprDagBuilder::from_dag(law.expression());
        let roots = (0..count)
            .map(|tag| {
                let weight = builder.index(*weights, tag as u32)?;
                builder.unary_math(UnaryMathFunction::Log, weight)
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let mass_ir = ScalarOperatorIr::lower(&builder.finish(roots)?)?;
        let loss_ir = ScalarOperatorIr::lower(loss.expression())?;
        let mut resolve = |symbol| input(kernel, point, distribution, None, symbol).ok();
        let mass = mass_ir.evaluate_typed(&[*weights], &mut resolve)?.remove(0);
        let weights = mass
            .components()
            .ok_or_else(|| invalid("finite masses must be real quantities"))?
            .map(|(weight, _)| weight)
            .collect::<Vec<_>>();
        if weights.len() != count
            || weights
                .iter()
                .any(|weight| !weight.is_finite() || *weight <= 0.0)
        {
            return Err(invalid(
                "score identity requires every declared outcome to have strictly positive finite mass",
            ));
        }
        let (_, log_mass) = actions(&mass_ir, kernel, point, distribution, None)?;
        let mut moments =
            WeightedMoments::new(ProbabilityMoment::Expectation, &[loss.value_type().clone()])?;
        let mut values = Vec::with_capacity(count);
        let mut losses = Vec::with_capacity(count);
        let mut tags = Vec::with_capacity(count);
        for (tag, &weight) in weights.iter().enumerate() {
            let outcome = outcomes.value(tag as u32)?;
            let (value, derivative) =
                actions(&loss_ir, kernel, point, distribution, Some(&outcome))?;
            if value.len() != 1 {
                return Err(invalid("finite score loss must be one real scalar"));
            }
            let value = literal(loss.value_type(), value[0])?;
            moments.push(std::slice::from_ref(&value), weight)?;
            values.push(value);
            losses.push(derivative.into_iter().next().expect("one loss root"));
            tags.push(outcome);
        }
        let mut scores = vec![vec![0.0; width]; count];
        let mut gradient = Vec::with_capacity(width);
        for (column, parameter) in point.signatures().iter().enumerate() {
            let input_unit = parameter.mathematical_type().value_type.dimension();
            let score_type = quotient_type(DimExponents::DIMENSIONLESS, input_unit)?;
            let gradient_type = quotient_type(loss.value_type().dimension(), input_unit)?;
            let mut normalizer = WeightedMoments::new(
                ProbabilityMoment::Expectation,
                std::slice::from_ref(&score_type),
            )?;
            for (row, &weight) in weights.iter().enumerate() {
                normalizer.push(&[literal(&score_type, log_mass[row][column])?], weight)?;
            }
            let normalizer = normalizer
                .value(0)?
                .real_scalar_value()
                .expect("real mean")
                .value();
            let mut derivative = WeightedMoments::new(
                ProbabilityMoment::Expectation,
                std::slice::from_ref(&gradient_type),
            )?;
            for (row, &weight) in weights.iter().enumerate() {
                let score = log_mass[row][column] - normalizer;
                scores[row][column] = score;
                let loss = values[row].real_scalar_value().expect("real loss").value();
                derivative.push(
                    &[literal(&gradient_type, losses[row][column] + loss * score)?],
                    weight,
                )?;
            }
            gradient.push(
                derivative
                    .value(0)?
                    .real_scalar_value()
                    .expect("real gradient")
                    .value(),
            );
        }
        let signatures: Arc<[_]> = point.signatures().to_vec().into();
        let scores = scores
            .into_iter()
            .map(|row| DifferentiableCotangent::from_coordinates(signatures.clone(), row))
            .collect::<Result<Vec<_>, _>>()?;
        let distribution_parameters = parameters
            .into_iter()
            .filter(|id| mass_ir.symbols().contains(&SymbolRef::Parameter(*id)))
            .collect();
        Ok(MomentEstimator {
            source: Source::Finite {
                model,
                point: point.clone(),
                outcomes: tags,
                values,
                scores,
            },
            reduction: MomentReduction::Population(ProbabilityMoment::Expectation),
            outputs: Vec::new(),
            weights: Cow::Owned(weights),
            value: moments.value(0)?,
            moments: Arc::new(moments),
            gradient_estimator: GradientEstimator::EnumeratedScore {
                distribution,
                variable: variable.id(),
                distribution_parameters,
            },
            derivative: DifferentiableCotangent::from_coordinates(signatures, gradient)?,
        })
    }
}

fn input(
    kernel: &KernelProgram,
    point: &DifferentiableValues,
    distribution: Id<kinds::Relation>,
    outcome: Option<&ValueLiteral>,
    symbol: SymbolRef,
) -> Result<ValueLiteral, Diagnostic> {
    match symbol {
        SymbolRef::Parameter(id) => point
            .get(id.erase())
            .map(|leaf| leaf.coefficient(0, None))
            .unwrap_or_else(|| {
                kernel
                    .typed_value(id.erase())
                    .cloned()
                    .ok_or_else(|| invalid("finite score Parameter is unavailable"))
            }),
        SymbolRef::Outcome(law) if law == distribution => outcome
            .cloned()
            .ok_or_else(|| invalid("finite score outcome is unbound")),
        _ => Err(invalid(
            "finite score reference requires Parameters and its exact outcome; State, trajectory and inter-Observable dependencies are unsupported",
        )),
    }
}

fn actions(
    operator: &ScalarOperatorIr,
    kernel: &KernelProgram,
    point: &DifferentiableValues,
    distribution: Id<kinds::Relation>,
    outcome: Option<&ValueLiteral>,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), Diagnostic> {
    let inputs = operator
        .symbols()
        .iter()
        .map(|symbol| input(kernel, point, distribution, outcome, *symbol))
        .collect::<Result<Vec<_>, _>>()?;
    let selected = operator
        .symbols()
        .iter()
        .map(|symbol| match symbol {
            SymbolRef::Parameter(id) => point
                .signatures()
                .iter()
                .position(|input| input.entity() == id.erase()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let roles = selected
        .iter()
        .map(|column| {
            if column.is_some() {
                DifferentiationRole::Parameter
            } else {
                DifferentiationRole::Frozen
            }
        })
        .collect::<Vec<_>>();
    let relation = operator.linearize_typed(&inputs, &roles)?;
    let mut values = vec![0.0; relation.residual_dimension()];
    relation.primal(&mut values)?;
    let mut derivatives = vec![vec![0.0; point.signatures().len()]; values.len()];
    for column in 0..point.signatures().len() {
        let direction = selected
            .iter()
            .flatten()
            .map(|selected| f64::from(*selected == column))
            .collect::<Vec<_>>();
        let mut tangent = vec![0.0; values.len()];
        relation.jvp(RelationTangent::Parameter(&direction), &mut tangent)?;
        for (row, value) in derivatives.iter_mut().zip(tangent) {
            row[column] = value;
        }
    }
    Ok((values, derivatives))
}

fn literal(ty: &ValueType, value: f64) -> Result<ValueLiteral, Diagnostic> {
    ValueLiteral::from_real(ty.clone(), value).map_err(|error| invalid(error.to_string()))
}

fn quotient_type(
    numerator: DimExponents,
    denominator: DimExponents,
) -> Result<ValueType, Diagnostic> {
    ValueType::scalar(
        ScalarDomain::Real,
        numerator
            .div(denominator)
            .ok_or_else(|| invalid("finite score derivative units exceed their bound"))?,
    )
    .map_err(|error| invalid(error.to_string()))
}

fn resources(
    count: usize,
    width: usize,
    mass_nodes: usize,
    loss_nodes: usize,
    limit: usize,
) -> Result<(), Diagnostic> {
    count
        .checked_mul(
            width
                .checked_mul(3)
                .and_then(|n| n.checked_add(4))
                .ok_or_else(|| invalid("finite score size overflow"))?,
        )
        .and_then(|n| n.checked_add(width.checked_mul(4)?))
        .and_then(|n| n.checked_add(1))
        .and_then(|n| n.checked_mul(size_of::<f64>()))
        .filter(|n| *n <= limit && *n <= isize::MAX as usize)
        .ok_or_else(|| invalid("finite score exceeds its numerical byte limit"))?;
    count
        .checked_mul(loss_nodes)
        .and_then(|n| n.checked_add(mass_nodes))
        .and_then(|n| n.checked_add(count.checked_mul(2)?))
        .and_then(|n| n.checked_mul(width.checked_add(1)?))
        .filter(|n| *n <= 1_000_000)
        .ok_or_else(|| invalid("finite score exceeds its structural expression work bound"))?;
    Ok(())
}
