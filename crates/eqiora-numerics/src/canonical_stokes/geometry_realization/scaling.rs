//! Private automatic scaling owned by the authenticated exact-cylinder binding.

use eqiora_artifact::{ArtifactDigest, ModelEnvelope};
use eqiora_core::{Diagnostic, DimExponents, DynQuantity};
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_sem::KernelProgram;
use sha2::{Digest, Sha256};

use super::{SteadyStokesGeometryBinding2d, invalid};
use crate::canonical_boundary::{PhysicalBoundaryDisposition, PhysicalBoundaryQuantity};
use crate::canonical_stokes::api::StokesBoundaryKey2d;
use crate::canonical_stokes::realization::IncompressibleFlowScaleProfile2d;

const LENGTH: DimExponents =
    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
const VELOCITY: DimExponents =
    DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension");
const PRESSURE: DimExponents =
    DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension");

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IncompressibleScalingRequest2d {
    length: Option<DynQuantity>,
    velocity: Option<DynQuantity>,
    pressure: Option<DynQuantity>,
}

impl IncompressibleScalingRequest2d {
    /// Construct a coherent-SI manual/automatic component request.
    pub fn from_si(
        length_m: Option<f64>,
        velocity_m_per_s: Option<f64>,
        pressure_pa: Option<f64>,
    ) -> Result<Self, Diagnostic> {
        Self::new(
            length_m.map(|value| DynQuantity::new(value, LENGTH)),
            velocity_m_per_s.map(|value| DynQuantity::new(value, VELOCITY)),
            pressure_pa.map(|value| DynQuantity::new(value, PRESSURE)),
        )
    }

    pub(super) fn new(
        length: Option<DynQuantity>,
        velocity: Option<DynQuantity>,
        pressure: Option<DynQuantity>,
    ) -> Result<Self, Diagnostic> {
        validate_component(length, LENGTH, "incompressible length scale L")?;
        validate_component(velocity, VELOCITY, "incompressible velocity scale U")?;
        validate_component(pressure, PRESSURE, "incompressible pressure scale P")?;
        Ok(Self {
            length,
            velocity,
            pressure,
        })
    }

    /// Optional manual characteristic length in metres.
    #[must_use]
    pub const fn length_m(self) -> Option<f64> {
        match self.length {
            Some(value) => Some(value.value()),
            None => None,
        }
    }

    /// Optional manual characteristic velocity in metres per second.
    #[must_use]
    pub const fn velocity_m_per_s(self) -> Option<f64> {
        match self.velocity {
            Some(value) => Some(value.value()),
            None => None,
        }
    }

    /// Optional manual characteristic pressure in pascals.
    #[must_use]
    pub const fn pressure_pa(self) -> Option<f64> {
        match self.pressure {
            Some(value) => Some(value.value()),
            None => None,
        }
    }
}

/// Closed intrinsic-2D incompressible scaling component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalingComponent2d {
    /// Characteristic length `L`.
    Length,
    /// Characteristic velocity `U`.
    Velocity,
    /// Characteristic pressure `P`.
    Pressure,
    /// Derived gauge/gradient scale `G = U/L`.
    Gauge,
    /// Derived intrinsic-2D weak-functional scale `Theta = P U L`.
    WeakFunctional,
}

/// Closed provenance mode for an effective component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalingMode2d {
    /// Supplied by the typed request.
    Manual,
    /// Obtained from an admitted capability-owned observation.
    Automatic,
    /// Computed from already resolved dependencies.
    Derived,
}

/// Versioned closed rule that resolved one intrinsic-2D component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalingRule2d {
    /// Manual override rule v1.
    ManualOverrideV1,
    /// Exact source-Geometry channel-height rule v1.
    ExactChannelHeightV1,
    /// Exact Model-owned inlet maximum rule v1.
    ExactInletMaximumV1,
    /// Viscous steady-Stokes pressure rule v1.
    ViscousStokesPressureV1,
    /// Derived gauge-rate rule v1.
    GaugeRateV1,
    /// Derived intrinsic-2D weak-functional rule v1.
    WeakFunctionalV1,
    /// Exact adjacent-partition streamwise span rule v1.
    ExactPartitionLengthV1,
    /// Solid shear-wave characteristic velocity rule v1.
    SolidShearWaveVelocityV1,
    /// Fluid dynamic-pressure rule v1.
    FluidDynamicPressureV1,
}

/// Fixed ordered dependency shape for one component record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalingDependencies2d {
    /// No component dependencies.
    None,
    /// Exactly one component dependency.
    One([ScalingComponent2d; 1]),
    /// Exactly two ordered component dependencies.
    Two([ScalingComponent2d; 2]),
    /// Exactly three ordered component dependencies.
    Three([ScalingComponent2d; 3]),
}

impl ScalingDependencies2d {
    pub const fn as_slice(&self) -> &[ScalingComponent2d] {
        match self {
            Self::None => &[],
            Self::One(values) => values,
            Self::Two(values) => values,
            Self::Three(values) => values,
        }
    }
}

/// One closed authoritative observation used by scaling resolution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScalingAuthority2d {
    /// The typed manual request.
    ManualRequest,
    /// Exact source-Geometry span on one axis.
    ExactGeometrySpan {
        /// Zero-based Geometry axis.
        axis: usize,
        /// Exact lower coordinate in metres.
        lower_m: f64,
        /// Exact upper coordinate in metres.
        upper_m: f64,
    },
    /// Model-owned prescribed inlet maximum observation.
    ModelInletMaximum {
        /// Exact evaluation coordinate in metres.
        coordinate_m: [f64; 2],
        /// Exact authenticated outward normal.
        outward_normal: [f64; 2],
        /// Evaluated velocity in metres per second.
        velocity_m_per_s: [f64; 2],
    },
    /// Model-owned dynamic viscosity observation.
    ModelDynamicViscosity {
        /// Dynamic viscosity in pascal-seconds.
        dynamic_viscosity_pa_s: f64,
    },
    /// Model-owned solid shear modulus and mass density.
    ModelSolidShearWave {
        shear_modulus_pa: f64,
        mass_density_kg_per_m3: f64,
    },
    /// Model-owned fluid mass density.
    ModelFluidMassDensity { mass_density_kg_per_m3: f64 },
}

/// Fixed closed authority inventory for one component record.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScalingAuthorities2d {
    /// No external observation; dependencies are the authority.
    None,
    /// Exactly one ordered authority.
    One([ScalingAuthority2d; 1]),
    /// Exactly two ordered authorities.
    Two([ScalingAuthority2d; 2]),
}

impl ScalingAuthorities2d {
    pub const fn as_slice(&self) -> &[ScalingAuthority2d] {
        match self {
            Self::None => &[],
            Self::One(values) => values,
            Self::Two(values) => values,
        }
    }
}

/// Immutable resolver-owned record for one effective component.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalingComponentRecord2d {
    component: ScalingComponent2d,
    value: DynQuantity,
    mode: ScalingMode2d,
    rule: ScalingRule2d,
    dependencies: ScalingDependencies2d,
    authorities: ScalingAuthorities2d,
}

impl ScalingComponentRecord2d {
    pub const fn component(self) -> ScalingComponent2d {
        self.component
    }

    pub const fn value(self) -> DynQuantity {
        self.value
    }

    pub const fn mode(self) -> ScalingMode2d {
        self.mode
    }

    pub const fn rule(self) -> ScalingRule2d {
        self.rule
    }

    pub const fn dependencies(self) -> ScalingDependencies2d {
        self.dependencies
    }

    pub const fn authorities(self) -> ScalingAuthorities2d {
        self.authorities
    }
}

/// Fixed resolver-owned intrinsic-2D incompressible scaling receipt.
#[derive(Clone, Debug, PartialEq)]
pub struct IncompressibleScalingReceipt2d {
    model: ArtifactDigest,
    geometry: ArtifactDigest,
    correspondence: ArtifactDigest,
    mesh: ArtifactDigest,
    production: Option<ArtifactDigest>,
    components: [ScalingComponentRecord2d; 5],
}

impl IncompressibleScalingReceipt2d {
    pub const fn model(&self) -> &ArtifactDigest {
        &self.model
    }

    pub const fn geometry(&self) -> &ArtifactDigest {
        &self.geometry
    }

    pub const fn correspondence(&self) -> &ArtifactDigest {
        &self.correspondence
    }

    pub const fn mesh(&self) -> &ArtifactDigest {
        &self.mesh
    }

    pub const fn production(&self) -> Option<&ArtifactDigest> {
        self.production.as_ref()
    }

    pub const fn components(&self) -> &[ScalingComponentRecord2d; 5] {
        &self.components
    }

    pub const fn component(&self, component: ScalingComponent2d) -> ScalingComponentRecord2d {
        self.components[component as usize]
    }

    /// Deterministic identity of the exact request provenance and lineage.
    #[must_use]
    pub fn provenance_digest(&self) -> ArtifactDigest {
        let mut bytes = Vec::new();
        for digest in [
            &self.model,
            &self.geometry,
            &self.correspondence,
            &self.mesh,
        ] {
            push_framed(&mut bytes, digest.as_str().as_bytes());
        }
        if let Some(production) = &self.production {
            push_framed(&mut bytes, production.as_str().as_bytes());
        }
        for record in self.components {
            bytes.push(component_code(record.component));
            bytes.extend_from_slice(&record.value.value().to_bits().to_be_bytes());
            let dimension = record.value.dim();
            for (numerator, denominator) in dimension.exponents() {
                bytes.extend_from_slice(&numerator.to_be_bytes());
                bytes.extend_from_slice(&denominator.to_be_bytes());
            }
            bytes.extend_from_slice(&[mode_code(record.mode), rule_code(record.rule)]);
            bytes.push(record.dependencies.as_slice().len() as u8);
            for dependency in record.dependencies.as_slice() {
                bytes.push(component_code(*dependency));
            }
            bytes.push(record.authorities.as_slice().len() as u8);
            for authority in record.authorities.as_slice() {
                encode_authority(&mut bytes, *authority);
            }
        }
        ArtifactDigest::from_sha256(
            Sha256::digest(
                [
                    b"eqiora.incompressible-scaling-receipt-2d/v2\0".as_slice(),
                    bytes.as_slice(),
                ]
                .concat(),
            )
            .into(),
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedIncompressibleScaling2d {
    scales: IncompressibleFlowScaleProfile2d,
    receipt: IncompressibleScalingReceipt2d,
}

impl ResolvedIncompressibleScaling2d {
    pub(crate) const fn scales(&self) -> IncompressibleFlowScaleProfile2d {
        self.scales
    }

    pub(crate) const fn receipt(&self) -> &IncompressibleScalingReceipt2d {
        &self.receipt
    }
}

impl SteadyStokesGeometryBinding2d {
    /// Resolve the closed exact-cylinder scaling family after this binding has
    /// authenticated Model meaning and the Geometry/correspondence/Mesh chain.
    pub(crate) fn resolve_incompressible_scaling(
        &self,
        model: &ModelEnvelope,
        request: Option<IncompressibleScalingRequest2d>,
    ) -> Result<ResolvedIncompressibleScaling2d, Diagnostic> {
        let replayed = replay_model(model, &self.source)?;
        if replayed != self.program {
            return Err(invalid(
                "automatic scaling Model meaning differs from the authenticated Stokes binding",
            ));
        }
        let request = request.unwrap_or_default();

        let (length, length_mode, length_rule, length_authorities) = match request.length {
            Some(value) => manual(value),
            None => {
                let bounds = self.exact_bounds()?;
                let value = positive(bounds[1][1] - bounds[1][0], "exact channel height")?;
                (
                    DynQuantity::new(value, LENGTH),
                    ScalingMode2d::Automatic,
                    ScalingRule2d::ExactChannelHeightV1,
                    ScalingAuthorities2d::One([ScalingAuthority2d::ExactGeometrySpan {
                        axis: 1,
                        lower_m: bounds[1][0],
                        upper_m: bounds[1][1],
                    }]),
                )
            }
        };

        let (velocity, velocity_mode, velocity_rule, velocity_authorities) = match request.velocity
        {
            Some(value) => manual(value),
            None => {
                let bounds = self.exact_bounds()?;
                let height = positive(bounds[1][1] - bounds[1][0], "exact channel height")?;
                let mut candidates = self.model.boundary_entries.iter().filter_map(
                    |(key, entry)| {
                        if !matches!(entry.disposition, PhysicalBoundaryDisposition::Prescribed(law)
                        if law.quantity() == PhysicalBoundaryQuantity::Trace)
                        {
                            return None;
                        }
                        let StokesBoundaryKey2d::NamedEntitySet(name) = key else {
                            return None;
                        };
                        let normal = self.source.constant_parent_outward_normal(name)?;
                        (normal == [-1.0, 0.0]).then_some((key, normal))
                    },
                );
                let (inlet, outward_normal) = candidates.next().ok_or_else(|| {
                    invalid("automatic U requires one prescribed trace on the exact lower-x side")
                })?;
                if candidates.next().is_some() {
                    return Err(invalid(
                        "automatic U has ambiguous lower-x prescribed traces",
                    ));
                }
                let coordinate_m = [bounds[0][0], bounds[1][0] + 0.5 * height];
                let inlet_velocity = self
                    .model
                    .prescribed_velocity(inlet, Some(outward_normal), &coordinate_m)?
                    .ok_or_else(|| {
                        invalid("automatic U requires the Model-owned prescribed velocity law")
                    })?;
                let value = positive(inlet_velocity[0], "Model inlet maximum")?;
                if inlet_velocity[1] != 0.0 {
                    return Err(invalid(
                        "automatic U requires the inlet maximum to align with the channel axis",
                    ));
                }
                (
                    DynQuantity::new(value, VELOCITY),
                    ScalingMode2d::Automatic,
                    ScalingRule2d::ExactInletMaximumV1,
                    ScalingAuthorities2d::Two([
                        ScalingAuthority2d::ExactGeometrySpan {
                            axis: 1,
                            lower_m: bounds[1][0],
                            upper_m: bounds[1][1],
                        },
                        ScalingAuthority2d::ModelInletMaximum {
                            coordinate_m,
                            outward_normal,
                            velocity_m_per_s: inlet_velocity,
                        },
                    ]),
                )
            }
        };

        let (pressure, pressure_mode, pressure_rule, pressure_dependencies, pressure_authorities) =
            match request.pressure {
                Some(value) => {
                    let (value, mode, rule, authorities) = manual(value);
                    (value, mode, rule, ScalingDependencies2d::None, authorities)
                }
                None => {
                    let viscosity =
                        positive(self.model.dynamic_viscosity(), "Model dynamic viscosity")?;
                    (
                        DynQuantity::new(viscosity * velocity.value() / length.value(), PRESSURE),
                        ScalingMode2d::Derived,
                        ScalingRule2d::ViscousStokesPressureV1,
                        ScalingDependencies2d::Two([
                            ScalingComponent2d::Length,
                            ScalingComponent2d::Velocity,
                        ]),
                        ScalingAuthorities2d::One([ScalingAuthority2d::ModelDynamicViscosity {
                            dynamic_viscosity_pa_s: viscosity,
                        }]),
                    )
                }
            };

        let scales = IncompressibleFlowScaleProfile2d::new(length, velocity, pressure)?;
        let components = [
            record(
                ScalingComponent2d::Length,
                scales.length(),
                length_mode,
                length_rule,
                ScalingDependencies2d::None,
                length_authorities,
            ),
            record(
                ScalingComponent2d::Velocity,
                scales.velocity(),
                velocity_mode,
                velocity_rule,
                ScalingDependencies2d::None,
                velocity_authorities,
            ),
            record(
                ScalingComponent2d::Pressure,
                scales.pressure(),
                pressure_mode,
                pressure_rule,
                pressure_dependencies,
                pressure_authorities,
            ),
            record(
                ScalingComponent2d::Gauge,
                scales.gauge(),
                ScalingMode2d::Derived,
                ScalingRule2d::GaugeRateV1,
                ScalingDependencies2d::Two([
                    ScalingComponent2d::Velocity,
                    ScalingComponent2d::Length,
                ]),
                ScalingAuthorities2d::None,
            ),
            record(
                ScalingComponent2d::WeakFunctional,
                scales.weak_functional(),
                ScalingMode2d::Derived,
                ScalingRule2d::WeakFunctionalV1,
                ScalingDependencies2d::Three([
                    ScalingComponent2d::Pressure,
                    ScalingComponent2d::Velocity,
                    ScalingComponent2d::Length,
                ]),
                ScalingAuthorities2d::None,
            ),
        ];
        Ok(ResolvedIncompressibleScaling2d {
            scales,
            receipt: IncompressibleScalingReceipt2d {
                model: model.digest()?,
                geometry: ArtifactDigest::from_sha256(self.source.digest_bytes()),
                correspondence: self.correspondence.digest()?,
                mesh: self.mesh.digest()?,
                production: None,
                components,
            },
        })
    }

    fn exact_bounds(&self) -> Result<&[[f64; 2]; 2], Diagnostic> {
        let source_bounds = self
            .source
            .circular_hole_bounds()
            .ok_or_else(|| invalid("automatic scaling requires exact circular-hole bounds"))?;
        if self.model.bounds() != source_bounds {
            return Err(invalid(
                "automatic scaling Model bounds differ from authenticated exact Geometry",
            ));
        }
        Ok(source_bounds)
    }
}

pub(crate) fn resolve_complete_manual_incompressible_scaling_2d(
    request: Option<IncompressibleScalingRequest2d>,
    model: ArtifactDigest,
    geometry: ArtifactDigest,
    correspondence: ArtifactDigest,
    mesh: ArtifactDigest,
) -> Result<ResolvedIncompressibleScaling2d, Diagnostic> {
    let request = request.ok_or_else(|| {
        invalid("transient incompressible flow requires complete manual L/U/P scaling")
    })?;
    let (Some(length), Some(velocity), Some(pressure)) =
        (request.length, request.velocity, request.pressure)
    else {
        return Err(invalid(
            "transient incompressible flow requires complete manual L/U/P scaling",
        ));
    };
    let scales = IncompressibleFlowScaleProfile2d::new(length, velocity, pressure)?;
    let manual_authority = ScalingAuthorities2d::One([ScalingAuthority2d::ManualRequest]);
    Ok(ResolvedIncompressibleScaling2d {
        scales,
        receipt: IncompressibleScalingReceipt2d {
            model,
            geometry,
            correspondence,
            mesh,
            production: None,
            components: [
                record(
                    ScalingComponent2d::Length,
                    scales.length(),
                    ScalingMode2d::Manual,
                    ScalingRule2d::ManualOverrideV1,
                    ScalingDependencies2d::None,
                    manual_authority,
                ),
                record(
                    ScalingComponent2d::Velocity,
                    scales.velocity(),
                    ScalingMode2d::Manual,
                    ScalingRule2d::ManualOverrideV1,
                    ScalingDependencies2d::None,
                    manual_authority,
                ),
                record(
                    ScalingComponent2d::Pressure,
                    scales.pressure(),
                    ScalingMode2d::Manual,
                    ScalingRule2d::ManualOverrideV1,
                    ScalingDependencies2d::None,
                    manual_authority,
                ),
                record(
                    ScalingComponent2d::Gauge,
                    scales.gauge(),
                    ScalingMode2d::Derived,
                    ScalingRule2d::GaugeRateV1,
                    ScalingDependencies2d::Two([
                        ScalingComponent2d::Velocity,
                        ScalingComponent2d::Length,
                    ]),
                    ScalingAuthorities2d::None,
                ),
                record(
                    ScalingComponent2d::WeakFunctional,
                    scales.weak_functional(),
                    ScalingMode2d::Derived,
                    ScalingRule2d::WeakFunctionalV1,
                    ScalingDependencies2d::Three([
                        ScalingComponent2d::Pressure,
                        ScalingComponent2d::Velocity,
                        ScalingComponent2d::Length,
                    ]),
                    ScalingAuthorities2d::None,
                ),
            ],
        },
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn resolve_fixed_reference_fsi_scaling_2d(
    request: Option<IncompressibleScalingRequest2d>,
    model: ArtifactDigest,
    geometry: ArtifactDigest,
    correspondence: ArtifactDigest,
    mesh: ArtifactDigest,
    production: ArtifactDigest,
    streamwise_bounds_m: [f64; 2],
    solid_materials: &[(f64, f64)],
    fluid_densities: &[f64],
) -> Result<ResolvedIncompressibleScaling2d, Diagnostic> {
    if request.is_some() {
        let mut resolved = resolve_complete_manual_incompressible_scaling_2d(
            request,
            model,
            geometry,
            correspondence,
            mesh,
        )?;
        resolved.receipt.production = Some(production);
        return Ok(resolved);
    }
    let &(solid_shear_modulus_pa, solid_mass_density_kg_per_m3) = solid_materials
        .first()
        .ok_or_else(|| invalid("automatic FSI scale has no exact elastic material witness"))?;
    let &fluid_mass_density_kg_per_m3 = fluid_densities.first().ok_or_else(|| {
        invalid("automatic FSI scale has no exact incompressible density witness")
    })?;
    if solid_materials
        .iter()
        .any(|value| *value != (solid_shear_modulus_pa, solid_mass_density_kg_per_m3))
        || fluid_densities
            .iter()
            .any(|value| *value != fluid_mass_density_kg_per_m3)
    {
        return Err(invalid(
            "heterogeneous FSI material scales require an explicit complete scaling request",
        ));
    }
    let length = positive(
        streamwise_bounds_m[1] - streamwise_bounds_m[0],
        "FSI exact streamwise span",
    )?;
    let shear = positive(solid_shear_modulus_pa, "FSI solid shear modulus")?;
    let solid_density = positive(solid_mass_density_kg_per_m3, "FSI solid mass density")?;
    let fluid_density = positive(fluid_mass_density_kg_per_m3, "FSI fluid mass density")?;
    let velocity = positive(
        (shear / solid_density).sqrt(),
        "FSI solid shear-wave velocity",
    )?;
    let pressure = positive(
        fluid_density * velocity * velocity,
        "FSI fluid dynamic pressure",
    )?;
    let scales = IncompressibleFlowScaleProfile2d::new(
        DynQuantity::new(length, LENGTH),
        DynQuantity::new(velocity, VELOCITY),
        DynQuantity::new(pressure, PRESSURE),
    )?;
    Ok(ResolvedIncompressibleScaling2d {
        scales,
        receipt: IncompressibleScalingReceipt2d {
            model,
            geometry,
            correspondence,
            mesh,
            production: Some(production),
            components: [
                record(
                    ScalingComponent2d::Length,
                    scales.length(),
                    ScalingMode2d::Automatic,
                    ScalingRule2d::ExactPartitionLengthV1,
                    ScalingDependencies2d::None,
                    ScalingAuthorities2d::One([ScalingAuthority2d::ExactGeometrySpan {
                        axis: 0,
                        lower_m: streamwise_bounds_m[0],
                        upper_m: streamwise_bounds_m[1],
                    }]),
                ),
                record(
                    ScalingComponent2d::Velocity,
                    scales.velocity(),
                    ScalingMode2d::Automatic,
                    ScalingRule2d::SolidShearWaveVelocityV1,
                    ScalingDependencies2d::None,
                    ScalingAuthorities2d::One([ScalingAuthority2d::ModelSolidShearWave {
                        shear_modulus_pa: shear,
                        mass_density_kg_per_m3: solid_density,
                    }]),
                ),
                record(
                    ScalingComponent2d::Pressure,
                    scales.pressure(),
                    ScalingMode2d::Derived,
                    ScalingRule2d::FluidDynamicPressureV1,
                    ScalingDependencies2d::One([ScalingComponent2d::Velocity]),
                    ScalingAuthorities2d::One([ScalingAuthority2d::ModelFluidMassDensity {
                        mass_density_kg_per_m3: fluid_density,
                    }]),
                ),
                record(
                    ScalingComponent2d::Gauge,
                    scales.gauge(),
                    ScalingMode2d::Derived,
                    ScalingRule2d::GaugeRateV1,
                    ScalingDependencies2d::Two([
                        ScalingComponent2d::Velocity,
                        ScalingComponent2d::Length,
                    ]),
                    ScalingAuthorities2d::None,
                ),
                record(
                    ScalingComponent2d::WeakFunctional,
                    scales.weak_functional(),
                    ScalingMode2d::Derived,
                    ScalingRule2d::WeakFunctionalV1,
                    ScalingDependencies2d::Three([
                        ScalingComponent2d::Pressure,
                        ScalingComponent2d::Velocity,
                        ScalingComponent2d::Length,
                    ]),
                    ScalingAuthorities2d::None,
                ),
            ],
        },
    })
}

fn replay_model(
    model: &ModelEnvelope,
    source: &eqiora_geometry::CanonicalGeometryV1,
) -> Result<KernelProgram, Diagnostic> {
    let (transaction, model_id) = model.to_transaction().map_err(first_diagnostic)?;
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).map_err(first_diagnostic)?;
    KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model_id, &[source])
        .map_err(first_diagnostic)
}

fn first_diagnostic(diagnostics: Vec<Diagnostic>) -> Diagnostic {
    diagnostics
        .into_iter()
        .next()
        .unwrap_or_else(|| invalid("automatic scaling Model replay failed without a diagnostic"))
}

fn manual(
    value: DynQuantity,
) -> (
    DynQuantity,
    ScalingMode2d,
    ScalingRule2d,
    ScalingAuthorities2d,
) {
    (
        value,
        ScalingMode2d::Manual,
        ScalingRule2d::ManualOverrideV1,
        ScalingAuthorities2d::One([ScalingAuthority2d::ManualRequest]),
    )
}

fn positive(value: f64, label: &str) -> Result<f64, Diagnostic> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(invalid(format!(
            "{label} must be finite and strictly positive"
        )))
    }
}

fn validate_component(
    value: Option<DynQuantity>,
    dimension: DimExponents,
    label: &str,
) -> Result<(), Diagnostic> {
    let Some(value) = value else {
        return Ok(());
    };
    if value.dim() != dimension {
        return Err(invalid(format!(
            "{label} has incompatible physical dimension"
        )));
    }
    positive(value.value(), label).map(|_| ())
}

fn push_framed(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
    bytes.extend_from_slice(value);
}

const fn component_code(value: ScalingComponent2d) -> u8 {
    match value {
        ScalingComponent2d::Length => 0,
        ScalingComponent2d::Velocity => 1,
        ScalingComponent2d::Pressure => 2,
        ScalingComponent2d::Gauge => 3,
        ScalingComponent2d::WeakFunctional => 4,
    }
}

const fn mode_code(value: ScalingMode2d) -> u8 {
    match value {
        ScalingMode2d::Manual => 0,
        ScalingMode2d::Automatic => 1,
        ScalingMode2d::Derived => 2,
    }
}

const fn rule_code(value: ScalingRule2d) -> u8 {
    match value {
        ScalingRule2d::ManualOverrideV1 => 0,
        ScalingRule2d::ExactChannelHeightV1 => 1,
        ScalingRule2d::ExactInletMaximumV1 => 2,
        ScalingRule2d::ViscousStokesPressureV1 => 3,
        ScalingRule2d::GaugeRateV1 => 4,
        ScalingRule2d::WeakFunctionalV1 => 5,
        ScalingRule2d::ExactPartitionLengthV1 => 6,
        ScalingRule2d::SolidShearWaveVelocityV1 => 7,
        ScalingRule2d::FluidDynamicPressureV1 => 8,
    }
}

fn encode_authority(bytes: &mut Vec<u8>, authority: ScalingAuthority2d) {
    match authority {
        ScalingAuthority2d::ManualRequest => bytes.push(0),
        ScalingAuthority2d::ExactGeometrySpan {
            axis,
            lower_m,
            upper_m,
        } => {
            bytes.push(1);
            bytes.extend_from_slice(&(axis as u64).to_be_bytes());
            bytes.extend_from_slice(&lower_m.to_bits().to_be_bytes());
            bytes.extend_from_slice(&upper_m.to_bits().to_be_bytes());
        }
        ScalingAuthority2d::ModelInletMaximum {
            coordinate_m,
            outward_normal,
            velocity_m_per_s,
        } => {
            bytes.push(2);
            for value in [coordinate_m, outward_normal, velocity_m_per_s]
                .into_iter()
                .flatten()
            {
                bytes.extend_from_slice(&value.to_bits().to_be_bytes());
            }
        }
        ScalingAuthority2d::ModelDynamicViscosity {
            dynamic_viscosity_pa_s,
        } => {
            bytes.push(3);
            bytes.extend_from_slice(&dynamic_viscosity_pa_s.to_bits().to_be_bytes());
        }
        ScalingAuthority2d::ModelSolidShearWave {
            shear_modulus_pa,
            mass_density_kg_per_m3,
        } => {
            bytes.push(4);
            bytes.extend_from_slice(&shear_modulus_pa.to_bits().to_be_bytes());
            bytes.extend_from_slice(&mass_density_kg_per_m3.to_bits().to_be_bytes());
        }
        ScalingAuthority2d::ModelFluidMassDensity {
            mass_density_kg_per_m3,
        } => {
            bytes.push(5);
            bytes.extend_from_slice(&mass_density_kg_per_m3.to_bits().to_be_bytes());
        }
    }
}

const fn record(
    component: ScalingComponent2d,
    value: DynQuantity,
    mode: ScalingMode2d,
    rule: ScalingRule2d,
    dependencies: ScalingDependencies2d,
    authorities: ScalingAuthorities2d,
) -> ScalingComponentRecord2d {
    ScalingComponentRecord2d {
        component,
        value,
        mode,
        rule,
        dependencies,
        authorities,
    }
}
