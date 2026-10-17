//! Exact process request, sampler and arithmetic provenance shared by saved values.
use super::*;
use crate::common_time::request_artifact::WireTimeRequest;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RunBinding {
    pub(super) run: WireTimeRequest,
    pub(super) sampler: String,
    pub(super) target_arch: String,
    pub(super) target_os: String,
    pub(super) arithmetic: String,
}

impl RunBinding {
    pub(super) fn new(request: &CommonTimeRunRequest) -> Result<Self, Diagnostic> {
        let (_, method) = request
            .plan()
            .process()
            .ok_or_else(|| invalid("process provenance requires a stochastic Plan"))?;
        Ok(Self {
            run: WireTimeRequest::from_request(request)?,
            sampler: method.generator().to_owned(),
            target_arch: std::env::consts::ARCH.to_owned(),
            target_os: std::env::consts::OS.to_owned(),
            arithmetic: "binary64-native-transcendentals-exact-retained-path/v1".to_owned(),
        })
    }
    pub(super) fn reopen_request(
        &self,
        plan: &crate::CommonTimePlan,
    ) -> Result<CommonTimeRunRequest, Diagnostic> {
        let request = self.run.replay(plan)?;
        if self != &Self::new(&request)? {
            return Err(invalid(
                "process history differs from its sampler or arithmetic environment",
            ));
        }
        Ok(request)
    }
}
