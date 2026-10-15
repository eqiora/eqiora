//! Safe framework-neutral derivative actions and XLA failure projection.

use std::sync::Arc;

use eqiora::api::DifferentiableProgram;

use crate::differentiation::registry::{self, RegistryError};

pub(super) fn resolve_program(key: &str) -> Result<Arc<DifferentiableProgram>, HandlerFailure> {
    registry::resolve(key).map_err(|error| {
        let kind = match error {
            RegistryError::Missing => FailureKind::NotFound,
            RegistryError::ForeignProcess => FailureKind::FailedPrecondition,
            RegistryError::Unavailable | RegistryError::IdentityCollision => FailureKind::Internal,
        };
        HandlerFailure::new(kind, error.message())
    })
}

#[derive(Debug, Clone, Copy)]
pub(super) enum Action {
    Primal,
    Jvp,
    Vjp,
}

impl Action {
    pub(super) const fn argument_count(self) -> usize {
        match self {
            Self::Primal => 1,
            Self::Jvp | Self::Vjp => 2,
        }
    }

    pub(super) const fn result_count(self) -> usize {
        match self {
            Self::Primal | Self::Vjp => 1,
            Self::Jvp => 2,
        }
    }
}

pub(super) enum ActionResult {
    Primal(Vec<f64>),
    Jvp { primal: Vec<f64>, tangent: Vec<f64> },
    Vjp(Vec<f64>),
}

#[derive(Debug, Clone, Copy)]
pub(super) enum FailureKind {
    InvalidArgument,
    NotFound,
    FailedPrecondition,
    Internal,
    DataLoss,
}

#[derive(Debug)]
pub(super) struct HandlerFailure {
    pub(super) kind: FailureKind,
    pub(super) message: String,
}

impl HandlerFailure {
    pub(super) fn new(kind: FailureKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub(super) fn invalid(message: impl Into<String>) -> Self {
        Self::new(FailureKind::InvalidArgument, message)
    }

    pub(super) fn internal(message: impl Into<String>) -> Self {
        Self::new(FailureKind::Internal, message)
    }

    pub(super) fn data_loss(message: impl Into<String>) -> Self {
        Self::new(FailureKind::DataLoss, message)
    }
}
