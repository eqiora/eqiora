//! Python projection of the single authored continuum-regularity enum.
use eqiora::kernel::SpatialRegularity;
use pyo3::prelude::*;

/// Authored continuum regularity on a Field's own spatial support.
#[pyclass(
    name = "SpatialRegularity",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PySpatialRegularity {
    Unspecified,
    L2,
    H1,
    HCurl,
    HDiv,
    Smooth,
}

impl From<PySpatialRegularity> for SpatialRegularity {
    fn from(value: PySpatialRegularity) -> Self {
        match value {
            PySpatialRegularity::Unspecified => Self::Unspecified,
            PySpatialRegularity::L2 => Self::L2,
            PySpatialRegularity::H1 => Self::H1,
            PySpatialRegularity::HCurl => Self::HCurl,
            PySpatialRegularity::HDiv => Self::HDiv,
            PySpatialRegularity::Smooth => Self::Smooth,
        }
    }
}

impl From<SpatialRegularity> for PySpatialRegularity {
    fn from(value: SpatialRegularity) -> Self {
        match value {
            SpatialRegularity::Unspecified => Self::Unspecified,
            SpatialRegularity::L2 => Self::L2,
            SpatialRegularity::H1 => Self::H1,
            SpatialRegularity::HCurl => Self::HCurl,
            SpatialRegularity::HDiv => Self::HDiv,
            SpatialRegularity::Smooth => Self::Smooth,
        }
    }
}
