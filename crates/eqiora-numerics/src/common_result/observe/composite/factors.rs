//! Finite coordinate-product quadrature uses the shared scalar Operator IR.
use std::collections::BTreeMap;

use eqiora_core::{DynQuantity, RawId, ScalarDomain};
use eqiora_meshing::ReferenceCell;
use eqiora_schema::kernel::typing::SpatialSupport;
use eqiora_schema::kernel::{AxisBounds, DomainKind};

use super::*;

pub(super) type Point = BTreeMap<(RawId, usize), f64>;

type Axis = ((RawId, usize), AxisBounds);

impl Context<'_> {
    fn output_domain(&self, id: Id<kinds::Observable>) -> Option<Id<kinds::Domain>> {
        self.program
            .observable_output_support(id)
            .ok()
            .flatten()
            .and_then(|support| support.domain().downcast())
    }

    pub(super) fn output_point(
        &self,
        id: Id<kinds::Observable>,
        coordinates: Option<&[DynQuantity]>,
    ) -> Result<Point, Diagnostic> {
        match (self.output_domain(id), coordinates) {
            (None, None) => Ok(Point::new()),
            (Some(domain), Some(coordinates)) => {
                let axes = axes(self.program, domain)?;
                if coordinates.len() != axes.len() {
                    return Err(invalid(
                        "Observable point requires exactly its output support coordinates",
                    ));
                }
                axes.into_iter().zip(coordinates).map(|((axis, bounds), coordinate)| {
                    if coordinate.dim() != bounds.lower().dim()
                        || !coordinate.value().is_finite()
                        || coordinate.value() < bounds.lower().value()
                        || coordinate.value() > bounds.upper().value() {
                        return Err(invalid("Observable output coordinate has wrong units or lies outside its exact support"));
                    }
                    Ok((axis, coordinate.value()))
                }).collect()
            }
            (Some(_), None) => Err(invalid(
                "field-valued Observable requires an explicit output point",
            )),
            (None, Some(_)) => Err(invalid(
                "lumped Observable does not have output coordinates",
            )),
        }
    }

    pub(super) fn factor_supported(&self, definition: &ObservableDef) -> bool {
        definition
            .reduction()
            .input_domain()
            .or_else(|| self.output_domain(definition.id()))
            .and_then(|domain| self.program.spatial_support(domain))
            .is_some_and(|support| matches!(support, SpatialSupport::Coordinates { .. }))
    }

    pub(super) fn evaluate_factors(
        &mut self,
        definition: &ObservableDef,
        point: &Point,
        depth: usize,
    ) -> Result<Evaluation, Diagnostic> {
        if self.tangent.is_some() {
            return Err(invalid(
                "coordinate-factor observation State tangents require an admitted field realization",
            ));
        }
        let ty = definition.value_type();
        if ty.scalar_domain() != ScalarDomain::Real
            || !ty.shape().is_scalar()
            || ty.array_rank() != 0
        {
            return Err(invalid(
                "coordinate-factor quadrature requires a real scalar density",
            ));
        }
        let output = self.output_domain(definition.id());
        if let Some(domain) = output {
            for (axis, bounds) in axes(self.program, domain)? {
                let value = point.get(&axis).ok_or_else(|| {
                    invalid("field-valued Observable requires a point on its exact output support")
                })?;
                if !value.is_finite()
                    || *value < bounds.lower().value()
                    || *value > bounds.upper().value()
                {
                    return Err(invalid(
                        "Observable output point is outside its bounded support",
                    ));
                }
            }
        }
        let typed = self
            .program
            .typed_observable(definition.id())
            .map_err(|errors| errors.into_iter().next().expect("failed typing"))?;
        let operator = ScalarOperatorIr::lower_typed_scalar(&typed)?;
        let value = match definition.reduction() {
            ObservableReduction::Value => self.factor_point(&operator, point, depth)?,
            ObservableReduction::SpatialIntegral {
                domain, measure, ..
            } => {
                if measure != eqiora_schema::kernel::ObservableMeasure::Volume {
                    return Err(invalid(
                        "coordinate product requires its declared Cartesian factor measure",
                    ));
                }
                self.require_regular_density(&operator, depth)?;
                let selected = axes(self.program, domain)?;
                let rule = self.quadratures.get(&domain).ok_or_else(|| {
                    invalid(
                        "factor integral requires explicit quadrature for its exact measure Domain",
                    )
                })?;
                if rule.reference_cell() != ReferenceCell::hypercube(selected.len())? {
                    return Err(invalid(
                        "factor quadrature dimension differs from the selected measure",
                    ));
                }
                self.used_rules.insert(domain);
                let mut sum = 0.0;
                let mut correction = 0.0;
                for sample in rule.points() {
                    let mut local = point.clone();
                    let mut weight = sample.weight;
                    for ((axis, bounds), coordinate) in selected.iter().zip(&sample.coordinates) {
                        let lower = bounds.lower().value();
                        let half_width = (bounds.upper().value() - lower) * 0.5;
                        local.insert(*axis, lower + (coordinate + 1.0) * half_width);
                        weight *= half_width;
                    }
                    let term = weight * self.factor_point(&operator, &local, depth)?;
                    let corrected = term - correction;
                    let next = sum + corrected;
                    correction = (next - sum) - corrected;
                    sum = next;
                }
                sum
            }
        };
        let value = ValueLiteral::from_real(ty.clone(), value)
            .map_err(|_| invalid("factor integral produced a non-finite value"))?;
        let evaluation = (value, None);
        if output.is_none() {
            self.accepted.insert(definition.id(), evaluation.clone());
        }
        Ok(evaluation)
    }

    fn require_regular_density(
        &mut self,
        operator: &ScalarOperatorIr,
        depth: usize,
    ) -> Result<(), Diagnostic> {
        if depth > 256 {
            return Err(invalid(
                "factor density composition exceeds its dependency depth bound",
            ));
        }
        self.remaining = self
            .remaining
            .checked_sub(operator.instruction_count())
            .ok_or_else(|| invalid("factor density admission exceeds its expression work bound"))?;
        operator.require_regular_density(|symbol| match symbol {
            SymbolRef::Coordinate { .. } => true,
            SymbolRef::Observable(id) => self.output_domain(id).is_some(),
            SymbolRef::Field(id) => self.program.edges().iter().any(|edge| {
                edge.from() == id.erase() && edge.kind() == eqiora_graph::EdgeKind::DefinedOn
            }),
            _ => false,
        })?;
        for symbol in operator.symbols() {
            if let SymbolRef::Observable(id) = symbol {
                let typed = self
                    .program
                    .typed_observable(*id)
                    .map_err(|errors| errors.into_iter().next().expect("failed typing"))?;
                let dependency = ScalarOperatorIr::lower_typed_scalar(&typed)?;
                self.require_regular_density(&dependency, depth + 1)?;
            }
        }
        Ok(())
    }

    fn factor_point(
        &mut self,
        operator: &ScalarOperatorIr,
        point: &Point,
        depth: usize,
    ) -> Result<f64, Diagnostic> {
        self.remaining = self
            .remaining
            .checked_sub(operator.instruction_count())
            .ok_or_else(|| invalid("factor quadrature exceeds its expression work bound"))?;
        let values = operator
            .symbols()
            .iter()
            .map(|symbol| match symbol {
                SymbolRef::Coordinate { factor, axis, .. } => {
                    point.get(&(factor.erase(), *axis)).copied().ok_or_else(|| {
                        invalid("coordinate factor is unbound at the integration point")
                    })
                }
                SymbolRef::Observable(id) => self
                    .evaluate(*id, point, depth + 1)?
                    .0
                    .real_scalar_value()
                    .map(|value| value.value())
                    .ok_or_else(|| {
                        invalid("factor density requires real scalar Observable inputs")
                    }),
                _ => self
                    .resolve(*symbol)
                    .and_then(|value| value.real_scalar_value())
                    .map(|value| value.value())
                    .ok_or_else(|| {
                        invalid("factor density Field or Parameter is unavailable in this Result")
                    }),
            })
            .collect::<Result<Vec<_>, _>>()?;
        operator
            .evaluate(&values)?
            .first()
            .copied()
            .ok_or_else(|| invalid("factor density has no scalar root"))
    }
}

fn axes(program: &KernelProgram, domain: Id<kinds::Domain>) -> Result<Vec<Axis>, Diagnostic> {
    let support = program
        .spatial_support(domain)
        .ok_or_else(|| invalid("factor integral support is outside the Model"))?;
    let factors = match support {
        SpatialSupport::Coordinates { factors, .. } => {
            factors.iter().map(|(id, _, _)| *id).collect()
        }
        SpatialSupport::Volume { domain, .. } => vec![*domain],
        _ => {
            return Err(invalid(
                "factor quadrature requires bounded Cartesian factors",
            ));
        }
    };
    let mut axes = Vec::new();
    for factor in factors {
        let Some(KernelNode::Domain(definition)) = program.node(factor) else {
            return Err(invalid("coordinate factor Domain is unavailable"));
        };
        let bounds = match definition.kind() {
            DomainKind::CoordinateInterval { bounds } => std::slice::from_ref(bounds),
            _ => program.resolved_cartesian_bounds(definition.id())?,
        };
        axes.extend(
            bounds
                .iter()
                .enumerate()
                .map(|(axis, bounds)| ((factor, axis), *bounds)),
        );
    }
    Ok(axes)
}
