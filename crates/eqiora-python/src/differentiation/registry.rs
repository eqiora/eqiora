//! One process-local owner for immutable programs used by framework adapters.

use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock, RwLock};

use eqiora::ScalarType;
use eqiora::api::DifferentiableProgram;
use pyo3::prelude::*;
use sha2::{Digest, Sha256};

const PROGRAM_KEY_DOMAIN: &[u8] = b"eqiora.differentiable-program/v1";

struct RegisteredProgram {
    pid: u32,
    program: Arc<DifferentiableProgram>,
}

static PROGRAMS: OnceLock<RwLock<BTreeMap<String, RegisteredProgram>>> = OnceLock::new();

fn programs() -> &'static RwLock<BTreeMap<String, RegisteredProgram>> {
    PROGRAMS.get_or_init(|| RwLock::new(BTreeMap::new()))
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum RegistryError {
    Unavailable,
    Missing,
    ForeignProcess,
    IdentityCollision,
}

impl RegistryError {
    pub(crate) const fn message(self) -> &'static str {
        match self {
            Self::Unavailable => "native differentiable program registry is unavailable",
            Self::Missing => "native differentiable program identity is not registered",
            Self::ForeignProcess => "differentiable programs cannot cross a process boundary",
            Self::IdentityCollision => "native differentiable program key collision was rejected",
        }
    }

    pub(super) fn python_error(self) -> PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(self.message())
    }
}

pub(crate) fn register(program: Arc<DifferentiableProgram>) -> Result<String, RegistryError> {
    let key = program_key(&program);
    let pid = std::process::id();
    let mut entries = programs().write().map_err(|_| RegistryError::Unavailable)?;
    if let Some(existing) = entries.get(&key) {
        if existing.pid != pid {
            return Err(RegistryError::ForeignProcess);
        }
        if existing.program.identity() != program.identity() {
            return Err(RegistryError::IdentityCollision);
        }
        return Ok(key);
    }
    entries.insert(key.clone(), RegisteredProgram { pid, program });
    Ok(key)
}

pub(crate) fn resolve(key: &str) -> Result<Arc<DifferentiableProgram>, RegistryError> {
    let entries = programs().read().map_err(|_| RegistryError::Unavailable)?;
    let entry = entries.get(key).ok_or(RegistryError::Missing)?;
    if entry.pid != std::process::id() {
        return Err(RegistryError::ForeignProcess);
    }
    Ok(Arc::clone(&entry.program))
}

#[pyfunction(name = "_differentiable_program")]
fn resolve_python(py: Python<'_>, key: &str) -> PyResult<super::PyDifferentiableProgram> {
    crate::error::panic_boundary(py, || {
        resolve(key)
            .map(|value| super::PyDifferentiableProgram { value })
            .map_err(RegistryError::python_error)
    })
}

pub(super) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(resolve_python, module)?)?;
    Ok(())
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
    hash_field(
        &mut hasher,
        b"initial-state",
        identity
            .initial_state_identity()
            .unwrap_or_default()
            .as_bytes(),
    );
    for input in identity.inputs() {
        hash_field(
            &mut hasher,
            b"input",
            input.entity().ulid().to_string().as_bytes(),
        );
    }
    for output in identity.outputs() {
        hash_field(
            &mut hasher,
            b"output",
            output.entity().ulid().to_string().as_bytes(),
        );
    }
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
    hash_field(&mut hasher, b"scalar", scalar);
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
