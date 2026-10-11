use std::time::Duration;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

use crate::{Completion, DeviceElement, TransferPlan};

/// Independently measured wall-time phases for one device execution.
///
/// The phases are not required to sum to `total`: asynchronous queues may
/// overlap work. Every phase must nevertheless fit within the observed total.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceExecutionTimings {
    setup: Duration,
    host_to_device: Duration,
    solve: Duration,
    device_to_host: Duration,
    verification: Duration,
    total: Duration,
}

impl DeviceExecutionTimings {
    /// Construct timing evidence with an explicit total observation.
    ///
    /// # Errors
    /// Returns `EQ0807` when a phase exceeds the total wall time.
    pub fn new(
        setup: Duration,
        host_to_device: Duration,
        solve: Duration,
        device_to_host: Duration,
        verification: Duration,
        total: Duration,
    ) -> Result<Self, Diagnostic> {
        if [setup, host_to_device, solve, device_to_host, verification]
            .into_iter()
            .any(|phase| phase > total)
        {
            return Err(invalid_evidence(
                "each device execution phase must fit within total wall time",
            ));
        }
        Ok(Self {
            setup,
            host_to_device,
            solve,
            device_to_host,
            verification,
            total,
        })
    }

    /// Runtime/context/operator setup wall time.
    #[must_use]
    pub const fn setup(self) -> Duration {
        self.setup
    }

    /// Host-to-device transfer wall time.
    #[must_use]
    pub const fn host_to_device(self) -> Duration {
        self.host_to_device
    }

    /// Device numerical execution wall time.
    #[must_use]
    pub const fn solve(self) -> Duration {
        self.solve
    }

    /// Device-to-host transfer wall time.
    #[must_use]
    pub const fn device_to_host(self) -> Duration {
        self.device_to_host
    }

    /// Independent host verification wall time.
    #[must_use]
    pub const fn verification(self) -> Duration {
        self.verification
    }

    /// End-to-end wall time.
    #[must_use]
    pub const fn total(self) -> Duration {
        self.total
    }
}

/// Evidence that one planned transfer was enqueued on the resident device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferEvidence<T: DeviceElement> {
    plan: TransferPlan<T>,
    completion: Completion,
}

impl<T: DeviceElement> TransferEvidence<T> {
    /// Bind a transfer plan to its completion identity.
    ///
    /// # Errors
    /// Returns `EQ0807` when the queue device is not one of the transfer's
    /// device endpoints.
    pub fn new(plan: TransferPlan<T>, completion: Completion) -> Result<Self, Diagnostic> {
        let queue_device = completion.submission().queue().device();
        let source_device = plan.source().device();
        let destination_device = plan.destination().device();
        if source_device != Some(queue_device) && destination_device != Some(queue_device) {
            return Err(invalid_evidence(
                "transfer completion queue must belong to a transfer endpoint device",
            ));
        }
        Ok(Self { plan, completion })
    }

    /// Exact transfer that was submitted.
    #[must_use]
    pub const fn plan(self) -> TransferPlan<T> {
        self.plan
    }

    /// Completion identity for the transfer.
    #[must_use]
    pub const fn completion(self) -> Completion {
        self.completion
    }
}

fn invalid_evidence(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}
