//! One initial-State selection for borrowed execution views and retained programs.
use crate::{CommonAlgebraicState, CommonState, CommonTimeState, ResolvedCommonPlan};
use eqiora_core::{Diagnostic, diagnostic::codes};
use std::borrow::Cow;

/// Exact initial State, borrowed from a Result or owned by a retained program.
#[derive(Debug, Clone, PartialEq)]
pub enum CommonInitialState<'a> {
    Spatial(Cow<'a, CommonState>),
    Algebraic(Cow<'a, CommonAlgebraicState>),
    Time(Cow<'a, CommonTimeState>),
}

impl<'a> From<&'a CommonState> for CommonInitialState<'a> {
    fn from(state: &'a CommonState) -> Self {
        Self::Spatial(Cow::Borrowed(state))
    }
}
impl From<CommonState> for CommonInitialState<'static> {
    fn from(state: CommonState) -> Self {
        Self::Spatial(Cow::Owned(state))
    }
}

impl<'a> From<&'a CommonAlgebraicState> for CommonInitialState<'a> {
    fn from(state: &'a CommonAlgebraicState) -> Self {
        Self::Algebraic(Cow::Borrowed(state))
    }
}
impl From<CommonAlgebraicState> for CommonInitialState<'static> {
    fn from(state: CommonAlgebraicState) -> Self {
        Self::Algebraic(Cow::Owned(state))
    }
}

impl<'a> From<&'a CommonTimeState> for CommonInitialState<'a> {
    fn from(state: &'a CommonTimeState) -> Self {
        Self::Time(Cow::Borrowed(state))
    }
}
impl From<CommonTimeState> for CommonInitialState<'static> {
    fn from(state: CommonTimeState) -> Self {
        Self::Time(Cow::Owned(state))
    }
}

impl CommonInitialState<'_> {
    /// Retain the State independently of the source borrow.
    #[must_use]
    pub fn into_owned(self) -> CommonInitialState<'static> {
        match self {
            Self::Spatial(state) => CommonInitialState::Spatial(Cow::Owned(state.into_owned())),
            Self::Algebraic(state) => CommonInitialState::Algebraic(Cow::Owned(state.into_owned())),
            Self::Time(state) => CommonInitialState::Time(Cow::Owned(state.into_owned())),
        }
    }

    /// Identity supplied by the selected State's existing authority.
    #[must_use]
    pub fn identity(&self) -> &str {
        match self {
            Self::Spatial(state) => state.identity(),
            Self::Algebraic(state) => state.identity(),
            Self::Time(state) => state.identity(),
        }
    }

    /// Bound the coordinate and identity payload when retaining an owned copy.
    /// The enclosing value is counted by its consumer; shared Model, Mesh and
    /// time-space owners remain shared. Specialized flow history is not admitted
    /// by this bound and must not silently count as an empty payload.
    pub fn retained_payload_bytes(&self) -> Result<usize, Diagnostic> {
        let invalid = || {
            Diagnostic::error(
                codes::INVALID_REALIZATION,
                "initial State payload has no admitted storage bound",
            )
        };
        let (bytes, owner) = match self {
            Self::Algebraic(state) => (size_of_val(state.values()), state.plan_identity()),
            Self::Spatial(state) => (
                state.spatial_retained_values_bytes().ok_or_else(invalid)?,
                state.state_space_identity(),
            ),
            Self::Time(state) => (size_of_val(state.values()), state.state_space_identity()),
        };
        bytes
            .checked_add(owner.len())
            .and_then(|bytes| bytes.checked_add(self.identity().len()))
            .ok_or_else(invalid)
    }

    /// Use the selected State's canonical codec without adding an envelope.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Diagnostic> {
        match self {
            Self::Spatial(state) => state.to_bytes(),
            Self::Algebraic(state) => state.to_bytes(),
            Self::Time(state) => state.to_bytes(),
        }
    }
}

impl CommonInitialState<'static> {
    /// Reauthenticate bytes through the codec owned by this exact Plan family.
    pub fn from_bytes(bytes: &[u8], plan: &ResolvedCommonPlan) -> Result<Self, Diagnostic> {
        Ok(match plan {
            ResolvedCommonPlan::Algebraic(plan) => {
                CommonAlgebraicState::from_bytes(bytes, plan)?.into()
            }
            ResolvedCommonPlan::Time(plan) => plan.state_from_bytes(bytes)?.into(),
            ResolvedCommonPlan::Spatial(_) | ResolvedCommonPlan::TransientFlow(_) => {
                CommonState::from_bytes(bytes, plan)?.into()
            }
            ResolvedCommonPlan::Eigen(_) | ResolvedCommonPlan::SteadyStokes(_) => {
                return Err(Diagnostic::error(
                    codes::INVALID_REALIZATION,
                    "initial State requires an algebraic, time, or spatial Plan",
                ));
            }
        })
    }
}
