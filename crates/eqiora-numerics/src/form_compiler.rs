//! Private proof-carrying FEM derivations.

mod authored_polynomial;
mod bilinear;
mod elasticity;
mod finite;
mod finite_typing;
pub(crate) mod harmonic;
pub(crate) mod integral_constraint;
pub(crate) mod interface;
pub(crate) mod mixed;
mod planar_curl;
mod vector_curl;
pub(crate) use finite::admit_authored_finite_weak_form;
pub(crate) mod equation_roles;
pub(crate) mod region;
#[cfg(test)]
mod regularity_tests;
mod scalar;
pub(crate) mod spatial;
#[cfg(test)]
mod tests;
pub(crate) mod vocabulary;

pub(crate) type PrimalFormDescription =
    (vocabulary::FormulationKind, &'static str, Vec<&'static str>);

pub(crate) use elasticity::{
    compile_cartesian_q1_elasticity_form_2d, derive_elasticity_correspondence,
};
pub(crate) use scalar::{
    AdmittedScalarGalerkinForm, DerivedScalarGalerkinForm, admit_authored_scalar_primal_form,
    compile_cartesian_q1_form, derive_candidate_with_dimension, equivalent_authored_expression,
};
#[cfg(test)]
use vocabulary::{
    DIVERGENCE_BY_PARTS, MatrixSlot, SOURCE_PAIRING, TEST_PAIRING, WeakSign, WeakTermSlot,
    ZERO_TEST_TRACE_DISCHARGE,
};

/// Reauthenticate argument dependence using the live Model's scalar domains.
pub(crate) fn check_authored_dependence(
    projection: &eqiora_compiler::AuthoredFormulationProjection,
    program: &eqiora_sem::KernelProgram,
) -> Result<(), eqiora_core::Diagnostic> {
    use eqiora_schema::kernel::KernelNode;
    check_field_traces(projection, program)?;
    projection.check_complex_dependence(&mut |id| match program.node(id) {
        Some(KernelNode::Field(field)) => Ok(field.value_type().clone()),
        Some(KernelNode::Parameter(parameter)) => Ok(parameter.value_type().clone()),
        _ => Err(eqiora_core::Diagnostic::error(
            eqiora_core::diagnostic::codes::INVALID_DISCRETIZATION,
            "authored form coefficient or argument is not a live Field or Parameter",
        )),
    })
}

/// Resolve a weak derivation request or check authored equations against live strong Laws.
///
/// Explicit equation/test associations select the rows of the shared spatial
/// equations. The checker authenticates volume pairings, complete boundary laws,
/// test restrictions and physical dimensions. Existing vector curl and steady
/// Stokes Green identities remain available for their admitted operator profiles.
/// This inspection selects no numerical method and proves neither stability nor
/// reverse implication.
///
/// # Errors
/// Rejects unsupported profiles, incomplete test/trial or boundary inventories,
/// stale source identities, or authored terms that do not match the strong Laws.
pub fn resolve_authored_spatial_formulation(
    program: &eqiora_sem::KernelProgram,
    form: &eqiora_compiler::AuthoredFormulationProjection,
) -> Result<eqiora_compiler::AuthoredFormulationProjection, eqiora_core::Diagnostic> {
    check_field_traces(form, program)?;
    mixed::resolve(program, form).or_else(|generic_error| {
        if form.requests_weak_derivation() {
            return Err(generic_error);
        }
        let specialized = if form.trial_ulids().len() == 1 {
            vector_curl::check(program, form)
        } else {
            crate::canonical_stokes::check_authored_mixed_formulation(program, form)
        };
        specialized.map(|_| form.clone()).map_err(|error| {
            generic_error.with_context(format!(
                "specialized Green-identity check also rejected the form ({})",
                error.message()
            ))
        })
    })
}

/// Reauthenticate boundary operands after decoding, using the admitted Model.
fn check_field_traces(
    projection: &eqiora_compiler::AuthoredFormulationProjection,
    program: &eqiora_sem::KernelProgram,
) -> Result<(), eqiora_core::Diagnostic> {
    use eqiora_core::{Diagnostic, entity::kinds};
    use eqiora_graph::EdgeKind;
    use eqiora_schema::kernel::{KernelNode, SpatialRegularity, typing::ExpressionType};
    let reject = || {
        Diagnostic::error(
            eqiora_core::diagnostic::codes::INVALID_DISCRETIZATION,
            "Field trace requires a live symbol and its exact admitted support",
        )
    };
    projection.check_field_trace_regularity(
        &mut |id| match program.node(id) {
            Some(KernelNode::Field(field)) => {
                let support = program
                    .edges()
                    .iter()
                    .find(|edge| edge.from() == id && edge.kind() == EdgeKind::DefinedOn)
                    .map(|edge| {
                        edge.to()
                            .downcast::<kinds::Domain>()
                            .and_then(|domain| program.spatial_support(domain))
                            .cloned()
                            .ok_or_else(reject)
                    })
                    .transpose()?;
                Ok((
                    ExpressionType::new(field.value_type().clone(), support),
                    field.spatial_regularity(),
                ))
            }
            Some(KernelNode::Parameter(parameter)) => Ok((
                ExpressionType::new(parameter.value_type().clone(), None),
                SpatialRegularity::Unspecified,
            )),
            _ => Err(reject()),
        },
        &mut |domain| program.spatial_support(domain).cloned().ok_or_else(reject),
    )
}
