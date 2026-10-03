//! Result-owned derived values and explicitly requested spatial quadrature.

use eqiora_artifact::ModelEnvelope;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id, ValueLiteral};
use eqiora_meshing::QuadratureRule;
use eqiora_schema::kernel::{KernelNode, ObservableReduction, SymbolRef};
use eqiora_sem::{ExpressionBackend, ReferenceExpressionBackend};

use super::{CommonResult, invalid};

mod spatial;
mod tangent;
pub use tangent::CommonObservableStateTangent;

/// Accepted derived value retaining the exact result and numerical integration rule.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonObservation {
    observable: Id<kinds::Observable>,
    result_identity: String,
    value: ValueLiteral,
    quadrature: Option<QuadratureRule>,
}

impl CommonObservation {
    /// Exact Model Observable.
    #[must_use]
    pub const fn observable(&self) -> Id<kinds::Observable> {
        self.observable
    }
    /// Complete accepted value and physical type.
    #[must_use]
    pub const fn value(&self) -> &ValueLiteral {
        &self.value
    }
    /// Exact accepted Result, including its Plan, State and field lineage.
    #[must_use]
    pub fn result_identity(&self) -> &str {
        &self.result_identity
    }
    /// Effective spatial quadrature; absent for a finite instantaneous value.
    #[must_use]
    pub const fn quadrature(&self) -> Option<&QuadratureRule> {
        self.quadrature.as_ref()
    }
}

impl CommonResult {
    /// Evaluate a typed Model Observable against this exact accepted Result.
    ///
    /// Spatial integrals require explicit quadrature on the measure's reference
    /// cell. The initial profile admits real scalar Q1 fields on Cartesian meshes,
    /// explicit traces and normal gradients; no output cadence controls this operation.
    /// # Errors
    /// Rejects foreign/stale Model meaning, unavailable fields, wrong measures,
    /// unsupported dependence and mismatched quadrature.
    pub fn observe(
        &self,
        model: &ModelEnvelope,
        observable: Id<kinds::Observable>,
        quadrature: Option<&QuadratureRule>,
    ) -> Result<CommonObservation, Diagnostic> {
        if model != self.plan().model_artifact() {
            return Err(invalid(
                "Observable Model differs from the exact accepted Result Model",
            ));
        }
        let program = observation_program(self, model)?;
        let typed = program.typed_observable(observable).map_err(|errors| {
            errors
                .into_iter()
                .next()
                .expect("typing failure has a diagnostic")
        })?;
        let Some(KernelNode::Observable(definition)) = program.node(observable.erase()) else {
            return Err(invalid("Observable is outside the exact Result Model"));
        };
        let value = match definition.reduction() {
            ObservableReduction::Value => {
                if quadrature.is_some() {
                    return Err(invalid(
                        "finite Observable value does not accept a spatial quadrature rule",
                    ));
                }
                let plan = self.plan().as_algebraic().ok_or_else(|| invalid("instantaneous Observable execution requires the admitted finite algebraic Result"))?;
                let values = self
                    .finite_values()
                    .ok_or_else(|| invalid("finite Observable has no accepted algebraic values"))?;
                let fields = plan.field_values(values)?;
                let mut resolve = |symbol| match symbol {
                    SymbolRef::Parameter(id) => program.typed_value(id.erase()).cloned(),
                    SymbolRef::Field(id) => fields
                        .iter()
                        .find(|(field, _)| *field == id)
                        .map(|(_, value)| value.clone()),
                    _ => plan
                        .symbols()
                        .iter()
                        .position(|candidate| *candidate == symbol)
                        .and_then(|index| {
                            let ty = eqiora_core::ValueType::scalar(
                                eqiora_core::ScalarDomain::Real,
                                plan.dimensions()[index],
                            )
                            .ok()?;
                            ValueLiteral::from_real(ty, values[index]).ok()
                        }),
                };
                let values = ReferenceExpressionBackend.evaluate(
                    observable.erase(),
                    typed.expression(),
                    typed.expression().roots(),
                    &mut resolve,
                )?;
                values
                    .into_iter()
                    .next()
                    .ok_or_else(|| invalid("Observable expression has no evaluated root"))?
            }
            ObservableReduction::SpatialIntegral { domain, .. } => {
                let rule = quadrature.ok_or_else(|| {
                    invalid("spatial Observable requires an explicit quadrature rule")
                })?;
                spatial::integrate(self, &program, definition, &typed, domain, rule, None)?
            }
        };
        if value.value_type() != definition.value_type() {
            return Err(invalid(
                "evaluated Observable type differs from its admitted declaration",
            ));
        }
        Ok(CommonObservation {
            observable,
            result_identity: self.identity().to_owned(),
            value,
            quadrature: quadrature.cloned(),
        })
    }
}

fn observation_program(
    result: &CommonResult,
    model: &ModelEnvelope,
) -> Result<eqiora_sem::KernelProgram, Diagnostic> {
    if let Some(plan) = result.plan().as_scalar() {
        Ok(plan.observation_program().clone())
    } else if let Some(plan) = result.plan().as_algebraic() {
        Ok(plan.kernel().clone())
    } else {
        model.to_program().map_err(|errors| {
            errors
                .into_iter()
                .next()
                .expect("failed replay has diagnostic")
        })
    }
}
