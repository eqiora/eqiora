//! **eqiora-backend-diffsol** — optional Diffsol time adapter.
//!
//! Diffsol types and callbacks remain private to this L3 crate. The adapter
//! consumes only Eqiora's lowered first-order time contract and fails closed
//! for equation/method pairs it cannot represent faithfully.

#[cfg(feature = "diffsol-runtime")]
mod runtime;

#[cfg(feature = "diffsol-runtime")]
pub use runtime::DiffsolTimeBackend;
