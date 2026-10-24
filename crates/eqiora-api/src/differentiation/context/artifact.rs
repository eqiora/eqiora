//! Collection-owned context payloads reuse Plan, State and RunRequest codecs.
use super::*;
use base64::{Engine, engine::general_purpose::STANDARD};
use eqiora_time::TimeBackendCapabilities;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum WireContext {
    Stationary {
        plan: String,
        initial: Option<String>,
    },
    Run {
        request: String,
    },
}

impl Context<'_> {
    pub(crate) fn wire(&self) -> Result<WireContext, Diagnostic> {
        Ok(match self {
            Self::Stationary { plan, initial } => WireContext::Stationary {
                plan: STANDARD.encode(plan.to_bytes()?),
                initial: initial
                    .as_ref()
                    .map(|state| state.to_bytes().map(|bytes| STANDARD.encode(bytes)))
                    .transpose()?,
            },
            Self::Run(request) => WireContext::Run {
                request: STANDARD.encode(request.to_bytes()?),
            },
        })
    }
}

impl WireContext {
    pub(crate) fn replay(
        self,
        backend: &dyn LinearSolverBackend,
        time_backend: &[TimeBackendCapabilities],
    ) -> Result<Context<'static>, Diagnostic> {
        Ok(match self {
            Self::Stationary { plan, initial } => {
                let plan = ResolvedCommonPlan::from_bytes(&decode(&plan)?, backend, time_backend)?;
                let initial = initial
                    .map(|value| CommonInitialState::from_bytes(&decode(&value)?, &plan))
                    .transpose()?;
                Context::Stationary {
                    plan: Cow::Owned(plan),
                    initial,
                }
            }
            Self::Run { request } => Context::Run(Cow::Owned(crate::RunRequest::from_bytes(
                &decode(&request)?,
                backend,
                time_backend,
            )?)),
        })
    }
}

fn decode(bytes: &str) -> Result<Vec<u8>, Diagnostic> {
    STANDARD
        .decode(bytes)
        .map_err(|error| invalid(format!("invalid Program context encoding: {error}")))
}
