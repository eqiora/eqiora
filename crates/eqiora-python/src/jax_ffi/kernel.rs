//! Safe accepted-program registry and framework-neutral derivative actions.

use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock, RwLock};

use eqiora::api::{DerivativeContract, DifferentiableDevice, DifferentiableProgram};
use eqiora::{Diagnostic, ScalarType, diagnostic::codes};
use sha2::{Digest, Sha256};

const PROGRAM_KEY_DOMAIN: &[u8] = b"eqiora.jax-differentiable-program/v1";

struct RegisteredProgram {
    pid: u32,
    program: Arc<DifferentiableProgram>,
}

static PROGRAMS: OnceLock<RwLock<BTreeMap<String, RegisteredProgram>>> = OnceLock::new();

fn programs() -> &'static RwLock<BTreeMap<String, RegisteredProgram>> {
    PROGRAMS.get_or_init(|| RwLock::new(BTreeMap::new()))
}

pub(super) fn register_program(program: Arc<DifferentiableProgram>) -> Result<String, Diagnostic> {
    let key = program_key(&program);
    let pid = std::process::id();
    let mut entries = programs().write().map_err(|_| {
        Diagnostic::error(
            codes::INTERNAL_FAILURE,
            "the native JAX program registry is unavailable",
        )
    })?;
    if let Some(existing) = entries.get(&key) {
        if existing.pid != pid {
            return Err(Diagnostic::error(
                codes::NOT_IMPLEMENTED,
                "JAX program registration cannot reuse an inherited process registry",
            ));
        }
        if existing.program.identity() != program.identity() {
            return Err(Diagnostic::error(
                codes::INTERNAL_FAILURE,
                "a native JAX program key collision was rejected",
            ));
        }
        return Ok(key);
    }
    entries.insert(key.clone(), RegisteredProgram { pid, program });
    Ok(key)
}

pub(super) fn resolve_program(key: &str) -> Result<Arc<DifferentiableProgram>, HandlerFailure> {
    let entries = programs()
        .read()
        .map_err(|_| HandlerFailure::internal("the native program registry is unavailable"))?;
    let entry = entries.get(key).ok_or_else(|| {
        HandlerFailure::new(
            FailureKind::NotFound,
            "the deterministic Eqiora JAX program identity is not registered",
        )
    })?;
    if entry.pid != std::process::id() {
        return Err(HandlerFailure::new(
            FailureKind::FailedPrecondition,
            "Eqiora JAX programs cannot cross a process boundary",
        ));
    }
    Ok(Arc::clone(&entry.program))
}

fn program_key(program: &DifferentiableProgram) -> String {
    let identity = program.identity();
    let model = identity.model();
    let mut hasher = Sha256::new();
    hasher.update(PROGRAM_KEY_DOMAIN);
    hash_field(
        &mut hasher,
        b"model-artifact",
        model.artifact().to_string().as_bytes(),
    );
    hash_field(
        &mut hasher,
        b"model-id",
        model.model().ulid().to_string().as_bytes(),
    );
    hash_u64(
        &mut hasher,
        b"semantic-revision",
        model.semantic_revision().get(),
    );
    hash_field(&mut hasher, b"plan", identity.plan_identity().as_bytes());
    for input in identity.inputs() {
        hash_field(&mut hasher, b"input", input.ulid().to_string().as_bytes());
    }
    hash_field(
        &mut hasher,
        b"output",
        identity.output().ulid().to_string().as_bytes(),
    );
    hash_usize(&mut hasher, b"input-dimension", identity.input_dimension());
    hash_usize(
        &mut hasher,
        b"output-dimension",
        identity.output_dimension(),
    );
    let scalar = match identity.scalar_type() {
        ScalarType::F32 => b"f32".as_slice(),
        ScalarType::F64 => b"f64".as_slice(),
    };
    let device = match identity.device() {
        DifferentiableDevice::HostCpu => b"host-cpu".as_slice(),
    };
    let derivative = match identity.derivative() {
        DerivativeContract::ImplicitFirstOrder => b"implicit-first-order".as_slice(),
    };
    hash_field(&mut hasher, b"scalar", scalar);
    hash_field(&mut hasher, b"device", device);
    hash_field(&mut hasher, b"derivative", derivative);
    hex(&hasher.finalize())
}

fn hash_field(hasher: &mut Sha256, label: &[u8], value: &[u8]) {
    hasher.update((label.len() as u64).to_be_bytes());
    hasher.update(label);
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value);
}

fn hash_u64(hasher: &mut Sha256, label: &[u8], value: u64) {
    hash_field(hasher, label, &value.to_be_bytes());
}

fn hash_usize(hasher: &mut Sha256, label: &[u8], value: usize) {
    let value = u64::try_from(value).expect("supported Python targets have at most 64-bit usize");
    hash_u64(hasher, label, value);
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
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
