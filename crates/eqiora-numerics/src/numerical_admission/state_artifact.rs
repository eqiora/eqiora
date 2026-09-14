//! Canonical persistence for one exact common spatial State.

use serde::{Deserialize, Serialize};

use super::*;

const SCHEMA: &str = "eqiora.common-spatial-state/v1";
const ENCODING: &str = "canonical-json-rfc8259-v1";
const MAX_BYTES: usize = 512 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WirePressureReference {
    ZeroIntegral { multiplier: f64 },
    BoundaryTraction,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "kebab-case", deny_unknown_fields)]
enum WireStatePayload {
    MiniP1 {
        velocity_vertex: Vec<[f64; 2]>,
        velocity_cell: Vec<[f64; 2]>,
        pressure_vertex: Vec<f64>,
        pressure_reference: WirePressureReference,
    },
    CellCentered {
        velocity_cell: Vec<[f64; 2]>,
        pressure_cell: Vec<f64>,
        gauge_multiplier: f64,
        previous_face_volume_fluxes: Vec<f64>,
    },
    FixedReferenceFsi {
        vertex_velocity: Vec<[f64; 2]>,
        fluid_velocity_cell: Vec<[f64; 2]>,
        pressure_vertex: Vec<f64>,
        solid_displacement: Vec<[f64; 2]>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCommonSpatialStateV1 {
    schema: String,
    encoding: String,
    state_space_identity: String,
    identity: String,
    time_s: f64,
    payload: WireStatePayload,
    named_boundary_forces_on_domain: Vec<(String, [f64; 2])>,
}

impl CommonState {
    /// Encode this exact restartable spatial State as bounded canonical bytes.
    ///
    /// Accepted-solve evidence cached beside an in-process FSI State belongs to
    /// its Result occurrence and is deliberately not State content.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&WireCommonSpatialStateV1::from_state(self)).map_err(|error| {
            invalid(format!(
                "cannot encode common spatial State artifact: {error}"
            ))
        })
    }

    /// Decode and reauthenticate one exact State against its state-space-owning Plan.
    pub fn from_bytes(bytes: &[u8], plan: &ResolvedCommonPlan) -> Result<Self, Diagnostic> {
        if bytes.len() > MAX_BYTES {
            return Err(invalid(format!(
                "common spatial State has {} bytes, exceeding the {MAX_BYTES} byte limit",
                bytes.len()
            )));
        }
        let wire: WireCommonSpatialStateV1 = serde_json::from_slice(bytes)
            .map_err(|error| invalid(format!("invalid common spatial State JSON: {error}")))?;
        if wire.schema != SCHEMA || wire.encoding != ENCODING {
            return Err(invalid(
                "common spatial State has an unknown schema or encoding",
            ));
        }
        let state = wire.replay(plan)?;
        if state.to_bytes()? != bytes {
            return Err(invalid(
                "common spatial State bytes are not the canonical encoding of their content",
            ));
        }
        Ok(state)
    }
}

impl WireCommonSpatialStateV1 {
    fn from_state(state: &CommonState) -> Self {
        let payload = match &state.kind {
            CommonStateKind::MiniP1(value) => WireStatePayload::MiniP1 {
                velocity_vertex: value.velocity().vertex_values().to_vec(),
                velocity_cell: value.velocity().cell_bubble_values().to_vec(),
                pressure_vertex: value.pressure().vertex_values().to_vec(),
                pressure_reference: match value.pressure_reference() {
                    SteadyStokesPressureReference2d::ZeroIntegral { multiplier } => {
                        WirePressureReference::ZeroIntegral { multiplier }
                    }
                    SteadyStokesPressureReference2d::BoundaryTraction => {
                        WirePressureReference::BoundaryTraction
                    }
                },
            },
            CommonStateKind::CellCentered(value) => WireStatePayload::CellCentered {
                velocity_cell: value.velocity().values().to_vec(),
                pressure_cell: value.pressure().values().to_vec(),
                gauge_multiplier: value.gauge_multiplier(),
                previous_face_volume_fluxes: value.previous_face_volume_fluxes().to_vec(),
            },
            CommonStateKind::Fsi {
                state, pressure, ..
            } => WireStatePayload::FixedReferenceFsi {
                vertex_velocity: state.vertex_velocity().to_vec(),
                fluid_velocity_cell: state
                    .fluid_cell_bubble_velocity()
                    .values()
                    .copied()
                    .collect(),
                pressure_vertex: pressure.to_vec(),
                solid_displacement: state.solid_displacement().to_vec(),
            },
        };
        Self {
            schema: SCHEMA.to_owned(),
            encoding: ENCODING.to_owned(),
            state_space_identity: state.state_space_identity.clone(),
            identity: state.identity.clone(),
            time_s: state.time_s,
            payload,
            named_boundary_forces_on_domain: state.named_boundary_forces_on_domain.clone(),
        }
    }

    fn replay(&self, plan: &ResolvedCommonPlan) -> Result<CommonState, Diagnostic> {
        let state = match (plan, &self.payload) {
            (ResolvedCommonPlan::TransientFlow(plan), WireStatePayload::MiniP1 { .. }) => {
                self.replay_mini(plan)?
            }
            (ResolvedCommonPlan::TransientFlow(plan), WireStatePayload::CellCentered { .. }) => {
                self.replay_cell_centered(plan)?
            }
            (ResolvedCommonPlan::Fsi(plan), WireStatePayload::FixedReferenceFsi { .. }) => {
                self.replay_fsi(plan)?
            }
            (
                ResolvedCommonPlan::Algebraic(_)
                | ResolvedCommonPlan::Ode(_)
                | ResolvedCommonPlan::Scalar(_)
                | ResolvedCommonPlan::Elasticity(_)
                | ResolvedCommonPlan::SteadyStokes(_),
                _,
            ) => {
                return Err(invalid(
                    "common spatial State requires a transient spatial Plan",
                ));
            }
            (ResolvedCommonPlan::TransientFlow(_), WireStatePayload::FixedReferenceFsi { .. })
            | (
                ResolvedCommonPlan::Fsi(_),
                WireStatePayload::MiniP1 { .. } | WireStatePayload::CellCentered { .. },
            ) => {
                return Err(invalid(
                    "common spatial State family crossed an incompatible Plan",
                ));
            }
        };
        if state.state_space_identity() != self.state_space_identity
            || state.identity() != self.identity
        {
            return Err(invalid(
                "common spatial State replay differs from its persisted identities",
            ));
        }
        Ok(state)
    }

    fn replay_mini(&self, plan: &CommonTransientFlowPlan) -> Result<CommonState, Diagnostic> {
        let WireStatePayload::MiniP1 {
            velocity_vertex,
            velocity_cell,
            pressure_vertex,
            pressure_reference,
        } = &self.payload
        else {
            unreachable!()
        };
        let (mesh, model) = match (
            plan.admission.resources(),
            plan.admission.recognized_model(),
        ) {
            (
                NativeMeshResources::AffineTriangleSimplicial { mesh, .. },
                RecognizedNativeModel::Transient(model),
            ) => (mesh, model.common_projection()),
            (
                NativeMeshResources::GmshSimplicial { mesh, .. },
                RecognizedNativeModel::TransientGeometry(binding),
            ) => (mesh, binding.model().clone()),
            _ => return Err(invalid("MINI/P1 State crossed a non-MINI transient Plan")),
        };
        let reference = match pressure_reference {
            WirePressureReference::ZeroIntegral { multiplier } => {
                SteadyStokesPressureReference2d::ZeroIntegral {
                    multiplier: *multiplier,
                }
            }
            WirePressureReference::BoundaryTraction => {
                SteadyStokesPressureReference2d::BoundaryTraction
            }
        };
        let mesh_data = mesh.mesh().clone();
        let native = TransientNavierStokesInitialState2d::new_for_model(
            &model,
            DynQuantity::new(self.time_s, TIME),
            mesh.artifact_reference()?,
            SimplicialMiniVelocityField2d::new(
                mesh_data.clone(),
                velocity_vertex.clone(),
                velocity_cell.clone(),
            )?,
            SimplicialP1Field::new(mesh_data, pressure_vertex.clone())?,
            reference,
        )?;
        CommonState::new_with_boundary_forces(
            plan.state_space_identity(),
            self.time_s,
            Arc::new(plan.admission.model().clone()),
            Arc::new(plan.admission.resources().clone()),
            CommonStateKind::MiniP1(Box::new(native)),
            self.named_boundary_forces_on_domain.clone(),
        )
    }

    fn replay_cell_centered(
        &self,
        plan: &CommonTransientFlowPlan,
    ) -> Result<CommonState, Diagnostic> {
        let WireStatePayload::CellCentered {
            velocity_cell,
            pressure_cell,
            gauge_multiplier,
            previous_face_volume_fluxes,
        } = &self.payload
        else {
            unreachable!()
        };
        let (mesh, model) = match (
            plan.admission.resources(),
            plan.admission.recognized_model(),
        ) {
            (
                NativeMeshResources::Cartesian { mesh, .. },
                RecognizedNativeModel::Transient(model),
            ) => (mesh, model.as_ref()),
            _ => {
                return Err(invalid(
                    "cell-centered State crossed a non-cell-centered transient Plan",
                ));
            }
        };
        let mesh_data = mesh.mesh().clone();
        let native = CellCenteredNavierStokesInitialState2d::new(
            model,
            DynQuantity::new(self.time_s, TIME),
            CellCenteredVelocityField2d::new(mesh_data.clone(), velocity_cell.clone())?,
            CellCenteredPressureField2d::new(mesh_data, pressure_cell.clone())?,
            *gauge_multiplier,
            previous_face_volume_fluxes.clone(),
        )?;
        CommonState::new_with_boundary_forces(
            plan.state_space_identity(),
            self.time_s,
            Arc::new(plan.admission.model().clone()),
            Arc::new(plan.admission.resources().clone()),
            CommonStateKind::CellCentered(Box::new(native)),
            self.named_boundary_forces_on_domain.clone(),
        )
    }

    fn replay_fsi(&self, plan: &CommonFsiPlan) -> Result<CommonState, Diagnostic> {
        let WireStatePayload::FixedReferenceFsi {
            vertex_velocity,
            fluid_velocity_cell,
            pressure_vertex,
            solid_displacement,
        } = &self.payload
        else {
            unreachable!()
        };
        if pressure_vertex.len() != plan.partition.fluid_vertices().len() {
            return Err(invalid(
                "FSI State pressure cardinality differs from its exact fluid support",
            ));
        }
        if fluid_velocity_cell.len() != plan.partition.fluid_cells().len() {
            return Err(invalid(
                "FSI State bubble cardinality differs from its exact fluid support",
            ));
        }
        let native = FixedReferenceFsiState::<2>::new(
            plan.mesh(),
            &plan.partition,
            vertex_velocity.clone(),
            plan.partition
                .fluid_cells()
                .iter()
                .copied()
                .zip(fluid_velocity_cell.iter().copied())
                .collect(),
            solid_displacement.clone(),
        )?;
        CommonState::new_with_boundary_forces(
            plan.state_space_identity(),
            self.time_s,
            Arc::new(plan.model().clone()),
            Arc::new(plan.resources().clone()),
            CommonStateKind::Fsi {
                state: Box::new(native),
                pressure: pressure_vertex.clone().into_boxed_slice(),
                accepted: None,
            },
            self.named_boundary_forces_on_domain.clone(),
        )
    }
}
