//! Private proof-carrying FEM derivations.

mod bilinear;
mod elasticity;
pub(crate) mod equation_roles;
pub(crate) mod linear;
pub(crate) mod region;
mod scalar;
#[cfg(test)]
mod tests;
pub(crate) mod vocabulary;

pub(crate) use elasticity::compile_cartesian_q1_elasticity_form_2d;
pub(crate) use scalar::{
    AdmittedScalarGalerkinForm, DerivedScalarGalerkinForm, admit_authored_scalar_primal_form,
    compile_cartesian_q1_form, derive_candidate_with_dimension,
};
#[cfg(test)]
use vocabulary::{
    DIVERGENCE_BY_PARTS, MatrixSlot, SOURCE_PAIRING, TEST_PAIRING, WeakSign, WeakTermSlot,
    ZERO_TEST_TRACE_DISCHARGE,
};
