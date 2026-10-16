//! Structural admission of a fixed-law location-scale path; execution stays in mapped AD.
use super::*;
use eqiora_core::{Id, RawId, ScalarDomain, entity::kinds};
use eqiora_schema::kernel::{
    DomainKind, ExprDag, ExprId, ExprNode, KernelNode, ObservableDef, ObservableMeasure,
    ObservableReduction, RelationMeaning, SymbolRef, UnaryMathFunction,
    pure_operator::CalculusNode, typing::SpatialSupport,
};
use eqiora_sem::KernelProgram;
use std::collections::BTreeMap;

impl MomentEstimator<'_> {
    /// Admit a pathwise location-scale expectation-gradient interpretation.
    ///
    /// The caller explicitly assumes an integrable local bound on the loss's
    /// design derivatives and finite required moments under this fixed base law.
    /// These analytic assumptions are not proved by finite samples. The sampler
    /// approximates that continuous law on its declared finite grid; no general
    /// unbiasedness or limit-interchange theorem is asserted.
    ///
    /// Structural admission requires a declared `location + scale * noise`
    /// random variable, positive scale, a fixed uniform interval-product base law,
    /// complete direct coordinate bindings to frozen sampled Parameters, equal
    /// positive mean weights, and smooth real scalar loss dependencies. The two
    /// design Parameters must be exactly the map's shared/fixed inputs. Noise
    /// cannot depend on any Program input before its coordinate realization.
    ///
    /// # Errors
    /// Rejects detached/foreign parameters or losses, changed sampling support,
    /// discrete/conditional paths, trajectory dependencies and other profiles.
    pub fn pathwise_location_scale(
        mut self,
        variable: &ModelEntityRef<Id<kinds::Observable>>,
        location: &ModelEntityRef<Id<kinds::Parameter>>,
        scale: &ModelEntityRef<Id<kinds::Parameter>>,
    ) -> Result<Self, Diagnostic> {
        let map = self
            .map()
            .ok_or_else(|| invalid("pathwise estimation requires a frozen sample map"))?;
        let identity = map.plan().program_identity();
        if variable.model() != identity.model()
            || location.model() != identity.model()
            || scale.model() != identity.model()
            || location.id() == scale.id()
            || !matches!(
                self.reduction,
                MomentReduction::Empirical(ProbabilityMoment::Expectation)
                    | MomentReduction::UnbiasedIid(ProbabilityMoment::Expectation)
            )
            || self
                .weights
                .iter()
                .any(|w| *w <= 0.0 || *w != self.weights[0])
        {
            return Err(invalid(
                "pathwise expectation requires an exact location-scale selection and equal positive mean weights",
            ));
        }
        let design = BTreeSet::from([location.id().erase(), scale.id().erase()]);
        if map
            .plan()
            .shared_inputs()
            .iter()
            .map(|id| id.erase())
            .collect::<BTreeSet<_>>()
            != design
            || identity.inputs().iter().any(|input| {
                !input.shape().is_empty()
                    || input.mathematical_type().value_type.scalar_domain() != ScalarDomain::Real
            })
        {
            return Err(invalid(
                "location-scale design must be exactly the shared real scalar inputs",
            ));
        }
        let kernel = map.plan().program().kernel_program()?;
        let variable_def = observable(&kernel, variable.id())?;
        let ObservableReduction::RandomVariable { distribution } = variable_def.reduction() else {
            return Err(invalid(
                "pathwise variable must retain its declared base probability law",
            ));
        };
        let Some(KernelNode::Relation(law)) = kernel.node(distribution.erase()) else {
            return Err(invalid("pathwise base law is unavailable"));
        };
        let RelationMeaning::ProbabilityDensity {
            domain,
            measure: ObservableMeasure::Volume,
            ..
        } = law.meaning()
        else {
            return Err(invalid(
                "pathwise location-scale requires a continuous volume base law",
            ));
        };
        let Some(SpatialSupport::Coordinates { factors, .. }) = kernel.spatial_support(*domain)
        else {
            return Err(invalid(
                "pathwise base law requires fixed coordinate intervals",
            ));
        };
        let mut bounds = BTreeMap::new();
        for (factor, _, axes) in factors {
            let Some(KernelNode::Domain(definition)) = kernel.node(*factor) else {
                return Err(invalid("pathwise support factor is unavailable"));
            };
            let DomainKind::CoordinateInterval { bounds: interval } = definition.kind() else {
                return Err(invalid(
                    "parameter-dependent or noninterval base support is unsupported",
                ));
            };
            if *axes != 1 {
                return Err(invalid("pathwise base factor must have one coordinate"));
            }
            bounds.insert(
                (*factor, 0),
                [interval.lower().value(), interval.upper().value()],
            );
        }
        let inputs = identity
            .inputs()
            .iter()
            .map(|input| input.entity())
            .collect::<BTreeSet<_>>();
        let mut context = Smooth {
            kernel: &kernel,
            distribution,
            variable: variable.id(),
            domain: *domain,
            coordinates: bounds.keys().copied().collect(),
            inputs,
            remaining: 65_536,
            found: false,
        };
        if law.expression().nodes().iter().any(|node| {
            matches!(node,
            ExprNode::Symbol(symbol) if !matches!(symbol, SymbolRef::Parameter(_)))
        }) {
            return Err(invalid(
                "pathwise base density must be uniform over its fixed support",
            ));
        }
        context.expression(law.expression(), law.expression().roots()[0], true, 0)?;
        let noise = location_scale(variable_def.expression(), location.id(), scale.id())?;
        context.expression(variable_def.expression(), noise, true, 0)?;
        let output = identity.outputs()[self.outputs[0]]
            .entity()
            .downcast()
            .ok_or_else(|| invalid("pathwise loss must be a selected named Observable"))?;
        let loss = observable(&kernel, output)?;
        let ExprNode::Evaluate {
            value,
            at,
            side: None,
        } = loss
            .expression()
            .node(loss.expression().roots()[0])
            .expect("admitted root")
        else {
            return Err(invalid(
                "pathwise loss requires an explicit complete probability realization",
            ));
        };
        if loss.reduction() != ObservableReduction::Value || at.len() != bounds.len() {
            return Err(invalid(
                "pathwise loss must realize every coordinate of its base law",
            ));
        }
        let mut draws = BTreeMap::new();
        for (selector, value) in at {
            let Some(ExprNode::Symbol(SymbolRef::Coordinate {
                support: selected_domain,
                factor,
                axis,
            })) = loss.expression().node(*selector)
            else {
                return Err(invalid(
                    "pathwise selector must be an exact base coordinate",
                ));
            };
            let Some(ExprNode::Symbol(SymbolRef::Parameter(draw))) = loss.expression().node(*value)
            else {
                return Err(invalid(
                    "pathwise coordinates must bind directly to frozen sampled Parameters",
                ));
            };
            let interval = bounds.remove(&(factor.erase(), *axis)).ok_or_else(|| {
                invalid("pathwise coordinate is repeated or outside the base support")
            })?;
            if selected_domain != domain
                || design.contains(&draw.erase())
                || draws.insert(draw.erase(), interval).is_some()
            {
                return Err(invalid(
                    "pathwise sampling must cover distinct base coordinates without design dependence",
                ));
            }
        }
        let sampled = identity
            .inputs()
            .iter()
            .filter(|input| !design.contains(&input.entity()))
            .map(|input| {
                draws
                    .remove(&input.entity())
                    .ok_or_else(|| invalid("Program input is not bound to the base law"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if !bounds.is_empty() || !draws.is_empty() {
            return Err(invalid(
                "pathwise sampling omits a declared base coordinate",
            ));
        }
        let samples = map.plan().samples().ok_or_else(|| {
            invalid("pathwise expectation requires retained frozen sample metadata")
        })?;
        let scale_index = identity
            .inputs()
            .iter()
            .position(|input| input.parameter() == scale.id())
            .expect("shared scale");
        for sample in samples {
            let fixed = crate::evaluation_map::partition::selected_parameters(
                identity,
                sample.fixed_values(),
            )?;
            if fixed
                .into_iter()
                .map(|id| id.erase())
                .collect::<BTreeSet<_>>()
                != design
                || sample.bounds() != sampled
                || sample.point().coordinates()[scale_index] <= 0.0
            {
                return Err(invalid(
                    "pathwise sample design differs from the fixed base law or positive scale",
                ));
            }
        }
        context.expression(loss.expression(), *value, false, 0)?;
        if !context.found {
            return Err(invalid(
                "pathwise loss is detached from the selected random variable",
            ));
        }
        self.gradient_estimator = GradientEstimator::PathwiseLocationScale {
            distribution,
            variable: variable.id(),
            location: location.id(),
            scale: scale.id(),
        };
        Ok(self)
    }
}

fn observable(
    kernel: &KernelProgram,
    id: Id<kinds::Observable>,
) -> Result<&ObservableDef, Diagnostic> {
    match kernel.node(id.erase()) {
        Some(KernelNode::Observable(value)) => Ok(value),
        _ => Err(invalid("pathwise Observable is outside the exact Model")),
    }
}

fn location_scale(
    expression: &ExprDag,
    location: Id<kinds::Parameter>,
    scale: Id<kinds::Parameter>,
) -> Result<ExprId, Diagnostic> {
    let is_parameter = |id, parameter| matches!(expression.node(id), Some(ExprNode::Symbol(SymbolRef::Parameter(p))) if *p == parameter);
    if let Some(ExprNode::Add(a, b)) = expression.node(expression.roots()[0]) {
        for (offset, product) in [(*a, *b), (*b, *a)] {
            if is_parameter(offset, location)
                && let Some(ExprNode::Mul(c, d)) = expression.node(product)
            {
                for (factor, noise) in [(*c, *d), (*d, *c)] {
                    if is_parameter(factor, scale) {
                        return Ok(noise);
                    }
                }
            }
        }
    }
    Err(invalid(
        "random variable must retain the selected location + scale * base noise paths",
    ))
}

struct Smooth<'a> {
    kernel: &'a KernelProgram,
    distribution: Id<kinds::Relation>,
    variable: Id<kinds::Observable>,
    domain: Id<kinds::Domain>,
    coordinates: BTreeSet<(RawId, usize)>,
    inputs: BTreeSet<RawId>,
    remaining: usize,
    found: bool,
}

impl Smooth<'_> {
    fn expression(
        &mut self,
        expression: &ExprDag,
        root: ExprId,
        noise: bool,
        depth: usize,
    ) -> Result<(), Diagnostic> {
        if depth > 32 || !expression.properties().is_empty() {
            return Err(invalid(
                "pathwise dependency depth or property profile is unsupported",
            ));
        }
        let mut pending = vec![root];
        let mut visited = BTreeSet::new();
        while let Some(id) = pending.pop() {
            if !visited.insert(id) {
                continue;
            }
            self.remaining = self
                .remaining
                .checked_sub(1)
                .ok_or_else(|| invalid("pathwise admission exceeds 65536 operations"))?;
            let node = expression.node(id).expect("admitted expression");
            match node {
                ExprNode::Constant(value) if value.real_scalar_value().is_some() => {}
                ExprNode::Symbol(SymbolRef::Parameter(id))
                    if !noise || !self.inputs.contains(&id.erase()) => {}
                ExprNode::Symbol(SymbolRef::Coordinate {
                    support: domain,
                    factor,
                    axis,
                }) if *domain == self.domain
                    && self.coordinates.contains(&(factor.erase(), *axis)) => {}
                ExprNode::Symbol(SymbolRef::Observable(id)) if !noise && *id == self.variable => {
                    self.found = true;
                }
                ExprNode::Symbol(SymbolRef::Observable(id)) => {
                    let value = observable(self.kernel, *id)?;
                    if value.reduction()
                        != (ObservableReduction::RandomVariable {
                            distribution: self.distribution,
                        })
                    {
                        return Err(invalid(
                            "pathwise dependencies must share the exact fixed base law",
                        ));
                    }
                    self.expression(
                        value.expression(),
                        value.expression().roots()[0],
                        noise,
                        depth + 1,
                    )?;
                }
                ExprNode::Neg(_)
                | ExprNode::Add(_, _)
                | ExprNode::Sub(_, _)
                | ExprNode::Mul(_, _)
                | ExprNode::Div(_, _)
                | ExprNode::PowI(_, _) => {}
                ExprNode::UnaryMath(function, _) if smooth_math(*function) => {}
                ExprNode::PureOperatorApplication(application) => {
                    let definition = expression
                        .definition(application.definition())
                        .expect("retained pure definition");
                    self.remaining = self
                        .remaining
                        .checked_sub(definition.nodes().len())
                        .ok_or_else(|| {
                            invalid("pathwise pure expansion exceeds its work budget")
                        })?;
                    if definition.nodes().iter().any(|node| !matches!(node,
                        CalculusNode::Rational { .. } | CalculusNode::FormalComponent { .. }
                        | CalculusNode::BoundInput(_) | CalculusNode::Neg(_) | CalculusNode::Add(_, _) | CalculusNode::Mul(_, _)
                    ) && !matches!(node, CalculusNode::UnaryMath(function, _) if smooth_math(*function))) {
                        return Err(invalid("pathwise pure operator contains an unsupported condition or derivative order"));
                    }
                }
                _ => {
                    return Err(invalid(
                        "pathwise expectation requires smooth real dependencies and a design-independent base law",
                    ));
                }
            }
            node.try_for_each_operand(|operand| {
                pending.push(operand);
                Ok::<_, Diagnostic>(())
            })?;
        }
        Ok(())
    }
}

fn smooth_math(function: UnaryMathFunction) -> bool {
    matches!(
        function,
        UnaryMathFunction::Sin
            | UnaryMathFunction::Cos
            | UnaryMathFunction::Exp
            | UnaryMathFunction::Log
            | UnaryMathFunction::Sqrt
    )
}
