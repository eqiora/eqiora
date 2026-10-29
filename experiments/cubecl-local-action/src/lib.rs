//! Isolated CubeCL experiment for Eqiora's device-neutral local-action IR.
//!
//! This crate is intentionally outside the production workspace. CubeCL
//! types appear only here; the admitted input is [`LocalLinearActionIr`].

use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::{Duration, Instant};

use cubecl::Device;
use cubecl::features::TypeUsage;
use cubecl::prelude::*;
use eqiora_ir::LocalLinearActionIr;

/// CubeCL release evaluated by this experiment.
pub const CUBECL_VERSION: &str = "0.11.3";

/// Eqiora-owned source contract for the two generated kernels.
pub const KERNEL_CONTRACT: &str = "eqiora.local-linear-action/cubecl-v1";

const CUBE_WIDTH: u32 = 128;

/// Numerical expression policy selected before launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalActionPolicy {
    /// One unit per output and ascending local-column accumulation.
    Ordered,
    /// The same ownership map with CubeCL fast floating-point transformations.
    Fast,
}

impl LocalActionPolicy {
    /// Stable evidence spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ordered => "ordered",
            Self::Fast => "fast",
        }
    }
}

/// Explicit observations from one experimental kernel execution.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalActionEvidence {
    cubecl_version: &'static str,
    kernel_contract: &'static str,
    runtime: &'static str,
    device: String,
    policy: LocalActionPolicy,
    output: Vec<f64>,
    reference_output: Vec<f64>,
    maximum_absolute_error: f64,
    allocation_and_upload: Duration,
    submission_and_download: Duration,
    reference_comparison: Duration,
    total: Duration,
}

impl LocalActionEvidence {
    /// Exact CubeCL release used to compile the adapter.
    #[must_use]
    pub const fn cubecl_version(&self) -> &'static str {
        self.cubecl_version
    }

    /// Eqiora-owned generated-kernel source contract.
    #[must_use]
    pub const fn kernel_contract(&self) -> &'static str {
        self.kernel_contract
    }

    /// CubeCL runtime identity reported by the selected adapter.
    #[must_use]
    pub const fn runtime(&self) -> &'static str {
        self.runtime
    }

    /// Debug identity of the runtime device selected by the caller.
    #[must_use]
    pub fn device(&self) -> &str {
        &self.device
    }

    /// Selected expression policy.
    #[must_use]
    pub const fn policy(&self) -> LocalActionPolicy {
        self.policy
    }

    /// Device-produced packed local output.
    #[must_use]
    pub fn output(&self) -> &[f64] {
        &self.output
    }

    /// Independently evaluated Eqiora reference output.
    #[must_use]
    pub fn reference_output(&self) -> &[f64] {
        &self.reference_output
    }

    /// Maximum pointwise absolute difference from the reference action.
    #[must_use]
    pub const fn maximum_absolute_error(&self) -> f64 {
        self.maximum_absolute_error
    }

    /// Host allocation and upload wall time.
    #[must_use]
    pub const fn allocation_and_upload(&self) -> Duration {
        self.allocation_and_upload
    }

    /// Kernel submission through synchronized output download wall time.
    #[must_use]
    pub const fn submission_and_download(&self) -> Duration {
        self.submission_and_download
    }

    /// Independent comparison after output download.
    #[must_use]
    pub const fn reference_comparison(&self) -> Duration {
        self.reference_comparison
    }

    /// Complete adapter call wall time.
    #[must_use]
    pub const fn total(&self) -> Duration {
        self.total
    }
}

/// Typed failure at the experimental adapter boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExperimentError {
    /// The selected runtime cannot represent the required `f64` operations.
    MissingF64Capability,
    /// A shape cannot be represented by CubeCL's current scalar arguments.
    ShapeExceedsU32,
    /// The caller supplied an invalid input or the reference action failed.
    InvalidAction(String),
    /// CubeCL rejected compilation or launch.
    Launch(String),
    /// The returned byte buffer did not contain the requested `f64` output.
    InvalidDownload,
}

impl fmt::Display for ExperimentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingF64Capability => formatter
                .write_str("selected CubeCL runtime lacks f64 buffer or arithmetic capability"),
            Self::ShapeExceedsU32 => {
                formatter.write_str("local-action shape exceeds CubeCL u32 launch arguments")
            }
            Self::InvalidAction(message) => write!(formatter, "invalid local action: {message}"),
            Self::Launch(message) => write!(formatter, "CubeCL launch failed: {message}"),
            Self::InvalidDownload => formatter.write_str("CubeCL returned an invalid f64 buffer"),
        }
    }
}

impl std::error::Error for ExperimentError {}

#[cube]
fn apply_local_action(
    coefficients: &[f64],
    input: &[f64],
    output: &mut [f64],
    rows: usize,
    columns: usize,
) {
    let output_index = ABSOLUTE_POS;
    if output_index < output.len() {
        let entity = output_index / rows;
        let row = output_index % rows;
        let matrix_offset = entity * rows * columns + row * columns;
        let input_offset = entity * columns;
        let mut value = 0.0_f64;
        for column in 0..columns {
            value += coefficients[matrix_offset + column] * input[input_offset + column];
        }
        output[output_index] = value;
    }
}

#[cube(launch)]
fn ordered_local_action(
    coefficients: &[f64],
    input: &[f64],
    output: &mut [f64],
    rows: usize,
    columns: usize,
) {
    apply_local_action(coefficients, input, output, rows, columns);
}

#[cube(launch, fast_math = FastMath::all())]
fn fast_local_action(
    coefficients: &[f64],
    input: &[f64],
    output: &mut [f64],
    rows: usize,
    columns: usize,
) {
    apply_local_action(coefficients, input, output, rows, columns);
}

/// Execute one local-action batch through a selected CubeCL runtime.
///
/// Input validation and acceptance use Eqiora's independent ordered CPU
/// evaluator. The function does not gather from or scatter to a global field.
///
/// # Errors
/// Returns a typed capability, shape, validation, launch, or download error.
#[allow(unsafe_code)]
pub fn execute(
    device: &Device,
    action: &LocalLinearActionIr,
    input: &[f64],
    policy: LocalActionPolicy,
) -> Result<LocalActionEvidence, ExperimentError> {
    let total_start = Instant::now();
    let mut reference_output = vec![0.0; action.output_len()];
    action
        .apply_reference(input, &mut reference_output)
        .map_err(|diagnostic| ExperimentError::InvalidAction(diagnostic.to_string()))?;

    let rows = action.rows();
    let columns = action.columns();
    let output_len =
        u32::try_from(action.output_len()).map_err(|_| ExperimentError::ShapeExceedsU32)?;
    let cube_count = output_len
        .checked_add(CUBE_WIDTH - 1)
        .ok_or(ExperimentError::ShapeExceedsU32)?
        / CUBE_WIDTH;

    let client = device.client();
    let f64_uses = f64::supported_uses(&client);
    if !f64_uses.contains(TypeUsage::Buffer) || !f64_uses.contains(TypeUsage::Arithmetic) {
        return Err(ExperimentError::MissingF64Capability);
    }

    let upload_start = Instant::now();
    let coefficients = client.create_from_slice(f64::as_bytes(action.coefficients()));
    let input = client.create_from_slice(f64::as_bytes(input));
    let output = client.empty(action.output_len() * size_of::<f64>());
    let allocation_and_upload = upload_start.elapsed();

    let launch_start = Instant::now();
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: `action` proves all three exact buffer lengths before the
        // raw launch arguments are constructed. Each generated kernel also
        // guards its only externally selected output index.
        unsafe {
            let coefficients =
                BufferArg::from_raw_parts(coefficients.clone(), action.coefficients().len());
            let input = BufferArg::from_raw_parts(input.clone(), action.input_len());
            let output_arg = BufferArg::from_raw_parts(output.clone(), action.output_len());
            match policy {
                LocalActionPolicy::Ordered => ordered_local_action::launch(
                    &client,
                    CubeCount::Static(cube_count, 1, 1),
                    CubeDim::new_1d(CUBE_WIDTH),
                    coefficients,
                    input,
                    output_arg,
                    rows,
                    columns,
                ),
                LocalActionPolicy::Fast => fast_local_action::launch(
                    &client,
                    CubeCount::Static(cube_count, 1, 1),
                    CubeDim::new_1d(CUBE_WIDTH),
                    coefficients,
                    input,
                    output_arg,
                    rows,
                    columns,
                ),
            }
        }
    }))
    .map_err(|payload| ExperimentError::Launch(panic_payload(payload)))?;
    let bytes = client
        .read_one(output)
        .map_err(|error| ExperimentError::Launch(error.to_string()))?;
    let submission_and_download = launch_start.elapsed();

    let verification_start = Instant::now();
    let output = f64::from_bytes(&bytes).to_vec();
    if output.len() != action.output_len() || output.iter().any(|value| !value.is_finite()) {
        return Err(ExperimentError::InvalidDownload);
    }
    let maximum_absolute_error = output
        .iter()
        .zip(&reference_output)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    let reference_comparison = verification_start.elapsed();

    Ok(LocalActionEvidence {
        cubecl_version: CUBECL_VERSION,
        kernel_contract: KERNEL_CONTRACT,
        runtime: client.name(),
        device: format!("{device:?}"),
        policy,
        output,
        reference_output,
        maximum_absolute_error,
        allocation_and_upload,
        submission_and_download,
        reference_comparison,
        total: total_start.elapsed(),
    })
}

fn panic_payload(payload: Box<dyn std::any::Any + Send>) -> String {
    payload.downcast_ref::<&str>().map_or_else(
        || {
            payload
                .downcast_ref::<String>()
                .cloned()
                .unwrap_or_else(|| "runtime panicked without a string payload".to_owned())
        },
        |message| (*message).to_owned(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_and_kernel_identity_are_stable_evidence() {
        assert_eq!(LocalActionPolicy::Ordered.as_str(), "ordered");
        assert_eq!(LocalActionPolicy::Fast.as_str(), "fast");
        assert_eq!(CUBECL_VERSION, "0.11.3");
        assert_eq!(KERNEL_CONTRACT, "eqiora.local-linear-action/cubecl-v1");
    }
}
