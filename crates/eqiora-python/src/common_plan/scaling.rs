//! Typed Python projection of resolver-owned incompressible scaling.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use eqiora_numerics::fluid::IncompressibleFlowScaleProfile2d;
use eqiora_numerics::{
    IncompressibleScalingReceipt2d, IncompressibleScalingRequest2d, ScalingAuthority2d,
    ScalingComponent2d, ScalingComponentRecord2d, ScalingMode2d, ScalingRule2d,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyFloat, PyModule, PyTuple};

use crate::error::validation_error;

#[pyclass(
    name = "IncompressibleScaling",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PyIncompressibleScaling {
    native: IncompressibleScalingRequest2d,
}

impl PyIncompressibleScaling {
    pub(crate) const fn native(&self) -> IncompressibleScalingRequest2d {
        self.native
    }
}

#[pymethods]
impl PyIncompressibleScaling {
    #[new]
    #[pyo3(signature = (*, length_m=None, velocity_m_per_s=None, pressure_pa=None))]
    fn new(
        py: Python<'_>,
        length_m: Option<&Bound<'_, PyAny>>,
        velocity_m_per_s: Option<&Bound<'_, PyAny>>,
        pressure_pa: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        IncompressibleScalingRequest2d::from_si(
            optional_float(length_m, "length_m")?,
            optional_float(velocity_m_per_s, "velocity_m_per_s")?,
            optional_float(pressure_pa, "pressure_pa")?,
        )
        .map(|native| Self { native })
        .map_err(|diagnostic| validation_error(py, &[diagnostic]))
    }

    #[getter]
    const fn length_m(&self) -> Option<f64> {
        self.native.length_m()
    }

    #[getter]
    const fn velocity_m_per_s(&self) -> Option<f64> {
        self.native.velocity_m_per_s()
    }

    #[getter]
    const fn pressure_pa(&self) -> Option<f64> {
        self.native.pressure_pa()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, Self>>()
            .is_ok_and(|other| self.native == other.native)
    }

    fn __hash__(&self) -> isize {
        let mut hasher = DefaultHasher::new();
        for value in [self.length_m(), self.velocity_m_per_s(), self.pressure_pa()] {
            value.map(f64::to_bits).hash(&mut hasher);
        }
        hasher.finish() as isize
    }
}

#[pyclass(
    name = "IncompressibleScales",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct PyIncompressibleScales {
    native: IncompressibleFlowScaleProfile2d,
}

impl PyIncompressibleScales {
    pub(crate) const fn from_native(native: IncompressibleFlowScaleProfile2d) -> Self {
        Self { native }
    }
}

#[pymethods]
impl PyIncompressibleScales {
    #[getter]
    const fn length_m(&self) -> f64 {
        self.native.length().value()
    }
    #[getter]
    const fn velocity_m_per_s(&self) -> f64 {
        self.native.velocity().value()
    }
    #[getter]
    const fn pressure_pa(&self) -> f64 {
        self.native.pressure().value()
    }
    #[getter]
    const fn gauge_per_s(&self) -> f64 {
        self.native.gauge().value()
    }
    #[getter]
    const fn weak_functional_w(&self) -> f64 {
        self.native.weak_functional().value()
    }
}

#[pyclass(
    name = "IncompressibleScalingComponent2d",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PyScalingComponent2d {
    Length,
    Velocity,
    Pressure,
    Gauge,
    WeakFunctional,
}

#[pyclass(
    name = "IncompressibleScalingMode",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PyScalingMode {
    Manual,
    Automatic,
    Derived,
}

#[pyclass(
    name = "IncompressibleScalingRule2d",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PyScalingRule2d {
    ManualOverrideV1,
    ExactChannelHeightV1,
    ExactInletMaximumV1,
    ViscousStokesPressureV1,
    GaugeRateV1,
    WeakFunctionalV1,
    ExactPartitionLengthV1,
    SolidShearWaveVelocityV1,
    FluidDynamicPressureV1,
}

#[pyclass(
    name = "IncompressibleScalingAuthorityKind",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PyScalingAuthorityKind {
    ManualRequest,
    ExactGeometrySpan,
    ModelInletMaximum,
    ModelDynamicViscosity,
    ModelSolidShearWave,
    ModelFluidMassDensity,
}

#[pyclass(
    name = "IncompressibleScalingAuthority2d",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct PyScalingAuthority2d {
    native: ScalingAuthority2d,
}

#[pymethods]
impl PyScalingAuthority2d {
    #[getter]
    const fn kind(&self) -> PyScalingAuthorityKind {
        match self.native {
            ScalingAuthority2d::ManualRequest => PyScalingAuthorityKind::ManualRequest,
            ScalingAuthority2d::ExactGeometrySpan { .. } => {
                PyScalingAuthorityKind::ExactGeometrySpan
            }
            ScalingAuthority2d::ModelInletMaximum { .. } => {
                PyScalingAuthorityKind::ModelInletMaximum
            }
            ScalingAuthority2d::ModelDynamicViscosity { .. } => {
                PyScalingAuthorityKind::ModelDynamicViscosity
            }
            ScalingAuthority2d::ModelSolidShearWave { .. } => {
                PyScalingAuthorityKind::ModelSolidShearWave
            }
            ScalingAuthority2d::ModelFluidMassDensity { .. } => {
                PyScalingAuthorityKind::ModelFluidMassDensity
            }
        }
    }

    #[getter]
    const fn axis(&self) -> Option<usize> {
        match self.native {
            ScalingAuthority2d::ExactGeometrySpan { axis, .. } => Some(axis),
            _ => None,
        }
    }

    #[getter]
    const fn bounds_m(&self) -> Option<(f64, f64)> {
        match self.native {
            ScalingAuthority2d::ExactGeometrySpan {
                lower_m, upper_m, ..
            } => Some((lower_m, upper_m)),
            _ => None,
        }
    }

    #[getter]
    const fn coordinate_m(&self) -> Option<(f64, f64)> {
        match self.native {
            ScalingAuthority2d::ModelInletMaximum { coordinate_m, .. } => {
                Some((coordinate_m[0], coordinate_m[1]))
            }
            _ => None,
        }
    }

    #[getter]
    const fn outward_normal(&self) -> Option<(f64, f64)> {
        match self.native {
            ScalingAuthority2d::ModelInletMaximum { outward_normal, .. } => {
                Some((outward_normal[0], outward_normal[1]))
            }
            _ => None,
        }
    }

    #[getter]
    const fn velocity_m_per_s(&self) -> Option<(f64, f64)> {
        match self.native {
            ScalingAuthority2d::ModelInletMaximum {
                velocity_m_per_s, ..
            } => Some((velocity_m_per_s[0], velocity_m_per_s[1])),
            _ => None,
        }
    }

    #[getter]
    const fn dynamic_viscosity_pa_s(&self) -> Option<f64> {
        match self.native {
            ScalingAuthority2d::ModelDynamicViscosity {
                dynamic_viscosity_pa_s,
            } => Some(dynamic_viscosity_pa_s),
            _ => None,
        }
    }

    #[getter]
    const fn shear_modulus_pa(&self) -> Option<f64> {
        match self.native {
            ScalingAuthority2d::ModelSolidShearWave {
                shear_modulus_pa, ..
            } => Some(shear_modulus_pa),
            _ => None,
        }
    }

    #[getter]
    const fn mass_density_kg_per_m3(&self) -> Option<f64> {
        match self.native {
            ScalingAuthority2d::ModelSolidShearWave {
                mass_density_kg_per_m3,
                ..
            }
            | ScalingAuthority2d::ModelFluidMassDensity {
                mass_density_kg_per_m3,
            } => Some(mass_density_kg_per_m3),
            _ => None,
        }
    }
}

#[pyclass(
    name = "IncompressibleScalingComponentRecord2d",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct PyScalingComponentRecord2d {
    native: ScalingComponentRecord2d,
}

#[pymethods]
impl PyScalingComponentRecord2d {
    #[getter]
    const fn component(&self) -> PyScalingComponent2d {
        project_component(self.native.component())
    }
    #[getter]
    const fn value(&self) -> f64 {
        self.native.value().value()
    }
    #[getter]
    fn dimension(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyTuple>> {
        crate::modeling::dimension::exponents(py, self.native.value().dim())
    }
    #[getter]
    const fn mode(&self) -> PyScalingMode {
        match self.native.mode() {
            ScalingMode2d::Manual => PyScalingMode::Manual,
            ScalingMode2d::Automatic => PyScalingMode::Automatic,
            ScalingMode2d::Derived => PyScalingMode::Derived,
        }
    }
    #[getter]
    const fn rule(&self) -> PyScalingRule2d {
        match self.native.rule() {
            ScalingRule2d::ManualOverrideV1 => PyScalingRule2d::ManualOverrideV1,
            ScalingRule2d::ExactChannelHeightV1 => PyScalingRule2d::ExactChannelHeightV1,
            ScalingRule2d::ExactInletMaximumV1 => PyScalingRule2d::ExactInletMaximumV1,
            ScalingRule2d::ViscousStokesPressureV1 => PyScalingRule2d::ViscousStokesPressureV1,
            ScalingRule2d::GaugeRateV1 => PyScalingRule2d::GaugeRateV1,
            ScalingRule2d::WeakFunctionalV1 => PyScalingRule2d::WeakFunctionalV1,
            ScalingRule2d::ExactPartitionLengthV1 => PyScalingRule2d::ExactPartitionLengthV1,
            ScalingRule2d::SolidShearWaveVelocityV1 => PyScalingRule2d::SolidShearWaveVelocityV1,
            ScalingRule2d::FluidDynamicPressureV1 => PyScalingRule2d::FluidDynamicPressureV1,
        }
    }
    #[getter]
    fn dependencies(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let values = self
            .native
            .dependencies()
            .as_slice()
            .iter()
            .copied()
            .map(project_component)
            .collect::<Vec<_>>();
        PyTuple::new(py, values).map(Bound::unbind)
    }
    #[getter]
    fn authorities(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let values = self
            .native
            .authorities()
            .as_slice()
            .iter()
            .copied()
            .map(|native| PyScalingAuthority2d { native })
            .collect::<Vec<_>>();
        PyTuple::new(py, values).map(Bound::unbind)
    }
}

#[pyclass(
    name = "IncompressibleScalingReceipt2d",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyIncompressibleScalingReceipt2d {
    native: IncompressibleScalingReceipt2d,
}

impl PyIncompressibleScalingReceipt2d {
    pub(crate) fn from_native(native: IncompressibleScalingReceipt2d) -> Self {
        Self { native }
    }

    fn record(&self, component: ScalingComponent2d) -> PyScalingComponentRecord2d {
        PyScalingComponentRecord2d {
            native: self.native.component(component),
        }
    }
}

#[pymethods]
impl PyIncompressibleScalingReceipt2d {
    #[getter]
    fn provenance_digest(&self) -> String {
        self.native.provenance_digest().to_string()
    }
    #[getter]
    fn model_digest(&self) -> &str {
        self.native.model().as_str()
    }
    #[getter]
    fn geometry_digest(&self) -> &str {
        self.native.geometry().as_str()
    }
    #[getter]
    fn correspondence_digest(&self) -> &str {
        self.native.correspondence().as_str()
    }
    #[getter]
    fn mesh_digest(&self) -> &str {
        self.native.mesh().as_str()
    }
    #[getter]
    fn production_digest(&self) -> Option<&str> {
        self.native.production().map(|digest| digest.as_str())
    }
    #[getter]
    fn length(&self) -> PyScalingComponentRecord2d {
        self.record(ScalingComponent2d::Length)
    }
    #[getter]
    fn velocity(&self) -> PyScalingComponentRecord2d {
        self.record(ScalingComponent2d::Velocity)
    }
    #[getter]
    fn pressure(&self) -> PyScalingComponentRecord2d {
        self.record(ScalingComponent2d::Pressure)
    }
    #[getter]
    fn gauge(&self) -> PyScalingComponentRecord2d {
        self.record(ScalingComponent2d::Gauge)
    }
    #[getter]
    fn weak_functional(&self) -> PyScalingComponentRecord2d {
        self.record(ScalingComponent2d::WeakFunctional)
    }
    #[getter]
    fn components(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        PyTuple::new(
            py,
            [
                self.length(),
                self.velocity(),
                self.pressure(),
                self.gauge(),
                self.weak_functional(),
            ],
        )
        .map(Bound::unbind)
    }
}

const fn project_component(value: ScalingComponent2d) -> PyScalingComponent2d {
    match value {
        ScalingComponent2d::Length => PyScalingComponent2d::Length,
        ScalingComponent2d::Velocity => PyScalingComponent2d::Velocity,
        ScalingComponent2d::Pressure => PyScalingComponent2d::Pressure,
        ScalingComponent2d::Gauge => PyScalingComponent2d::Gauge,
        ScalingComponent2d::WeakFunctional => PyScalingComponent2d::WeakFunctional,
    }
}

fn optional_float(value: Option<&Bound<'_, PyAny>>, name: &str) -> PyResult<Option<f64>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    if value.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(format!(
            "{name} must be a float or None, not bool"
        )));
    }
    if !value.is_instance_of::<PyFloat>() {
        return Err(PyTypeError::new_err(format!(
            "{name} must be a float or None"
        )));
    }
    value.extract::<f64>().map(Some).map_err(|_| {
        PyTypeError::new_err(format!("{name} must be a finite positive float or None"))
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyIncompressibleScaling>()?;
    module.add_class::<PyIncompressibleScales>()?;
    module.add_class::<PyScalingComponent2d>()?;
    module.add_class::<PyScalingMode>()?;
    module.add_class::<PyScalingRule2d>()?;
    module.add_class::<PyScalingAuthorityKind>()?;
    module.add_class::<PyScalingAuthority2d>()?;
    module.add_class::<PyScalingComponentRecord2d>()?;
    module.add_class::<PyIncompressibleScalingReceipt2d>()?;
    Ok(())
}
