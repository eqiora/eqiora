//! Native CPU targets for the optional JAX typed-FFI adapter.
//!
//! Framework registration remains private to `eqiora.jax`. Static accepted
//! program identity and numerical execution are safe Rust; the allowlisted XLA
//! C ABI is isolated in [`abi`].

#[allow(unsafe_code)]
mod abi;
mod batch;
mod kernel;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

pub(super) const PRIMAL_TARGET: &str = "eqiora_differentiable_primal_v2";
pub(super) const JVP_TARGET: &str = "eqiora_differentiable_jvp_v2";
pub(super) const VJP_TARGET: &str = "eqiora_differentiable_vjp_v2";

#[pyfunction(name = "_jax_ffi_targets")]
fn ffi_targets(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    abi::target_capsules(py)
}

#[pyfunction(name = "_jax_ffi_abi_layout")]
fn ffi_abi_layout(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    abi::layout(py)
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(ffi_targets, module)?)?;
    module.add_function(wrap_pyfunction!(ffi_abi_layout, module)?)?;
    Ok(())
}
