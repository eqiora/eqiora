//! One execution meaning shared by live Programs and solve-free record reopening.
use super::*;
use std::borrow::Cow;
mod artifact;
pub(crate) use artifact::WireContext;

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Context<'a> {
    Stationary {
        plan: Cow<'a, ResolvedCommonPlan>,
        initial: Option<CommonInitialState<'a>>,
    },
    Run(Cow<'a, crate::RunRequest>),
}

impl<'a> Context<'a> {
    pub(crate) fn stationary(
        plan: &'a ResolvedCommonPlan,
        initial: Option<&'a CommonInitialState<'_>>,
    ) -> Self {
        Self::Stationary {
            plan: Cow::Borrowed(plan),
            initial: initial.map(borrow_initial),
        }
    }

    pub(crate) fn plan(&self) -> &ResolvedCommonPlan {
        match self {
            Self::Stationary { plan, .. } => plan,
            Self::Run(request) => request.native().plan(),
        }
    }

    pub(crate) fn initial(&self) -> Option<CommonInitialState<'_>> {
        match self {
            Self::Stationary { initial, .. } => initial.as_ref().map(borrow_initial),
            Self::Run(request) => Some(request.native().state().into()),
        }
    }

    pub(crate) fn run_request(&self) -> Option<&crate::RunRequest> {
        match self {
            Self::Stationary { .. } => None,
            Self::Run(request) => Some(request),
        }
    }
}

fn borrow_initial<'a>(initial: &'a CommonInitialState<'_>) -> CommonInitialState<'a> {
    match initial {
        CommonInitialState::Spatial(state) => state.as_ref().into(),
        CommonInitialState::Algebraic(state) => state.as_ref().into(),
        CommonInitialState::Time(state) => state.as_ref().into(),
    }
}
