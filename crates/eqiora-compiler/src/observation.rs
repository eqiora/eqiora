//! Declaration reductions shared by source checking and flattened expression lowering.
use eqiora_schema::kernel::{ObservableMeasure, ProbabilityMoment};

#[derive(Debug, Clone, Copy)]
pub(crate) enum Reduction<N, E> {
    SpatialIntegral {
        domain: N,
        measure: Option<ObservableMeasure>,
        limits: Option<[E; 2]>,
    },
    RandomVariable {
        distribution: N,
    },
    ProbabilityMoment {
        moment: ProbabilityMoment,
    },
}

impl<N, E> Reduction<N, E> {
    pub(crate) fn domain(&self) -> Option<&N> {
        match self {
            Self::SpatialIntegral { domain, .. } => Some(domain),
            _ => None,
        }
    }
    pub(crate) fn limits(&self) -> Option<&[E; 2]> {
        match self {
            Self::SpatialIntegral { limits, .. } => limits.as_ref(),
            _ => None,
        }
    }
    pub(crate) fn measure(&self) -> Option<ObservableMeasure> {
        match self {
            Self::SpatialIntegral { measure, .. } => *measure,
            _ => None,
        }
    }
}

pub(crate) fn moment(name: &str) -> Option<ProbabilityMoment> {
    match name {
        "expectation" => Some(ProbabilityMoment::Expectation),
        "variance" => Some(ProbabilityMoment::Variance),
        "covariance" => Some(ProbabilityMoment::Covariance),
        _ => None,
    }
}
