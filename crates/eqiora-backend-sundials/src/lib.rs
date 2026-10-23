//! Optional residual-native IDA adapter. Native handles never escape this crate.
#[cfg(feature = "ida-runtime")]
#[allow(unsafe_code)]
mod native;
#[cfg(feature = "ida-runtime")]
mod runtime;
#[cfg(feature = "ida-runtime")]
pub use runtime::IdaTimeBackend;

#[cfg(all(test, feature = "ida-runtime"))]
mod tests;
