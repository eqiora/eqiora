//! Private proof-carrying FEM derivations.

mod authored_polynomial;
mod bilinear;
mod elasticity;
mod finite;
mod finite_typing;
pub(crate) mod harmonic;
pub(crate) use finite::admit_authored_finite_weak_form;
pub(crate) mod equation_roles;
pub(crate) mod linear;
pub(crate) mod region;
mod scalar;
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
    projection.check_complex_dependence(&mut |id| match program.node(id) {
        Some(KernelNode::Field(field)) => Ok(field.value_type().clone()),
        Some(KernelNode::Parameter(parameter)) => Ok(parameter.value_type().clone()),
        _ => Err(eqiora_core::Diagnostic::error(
            eqiora_core::diagnostic::codes::INVALID_DISCRETIZATION,
            "authored form coefficient or argument is not a live Field or Parameter",
        )),
    })
}
