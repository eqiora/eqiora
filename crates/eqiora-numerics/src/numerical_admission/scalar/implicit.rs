//! Continuous spatial equations in free real coordinates, including analytic trace rates.
mod midpoint;
mod rate;
mod recovery;
mod replay;
mod run;
use super::*;
use crate::form_compiler::region::FieldPullback;
use crate::form_compiler::spatial::InputPartialScope;
use crate::region_assembly::RegionPoint;
use crate::region_assembly::mapping::{
    MappedRegionAssembly, RegionAlgebraicCoordinates, RegionDofMap, RegionRealCoordinates,
};
use crate::spatial_expression::Coefficient;
use eqiora_time::ImplicitTimeSystem;

type RealRegionAssembly<S> =
    MappedRegionAssembly<S, <RegionDofMap<S> as RegionAlgebraicCoordinates<f64, S>>::Projection>;

pub(in crate::numerical_admission) struct SpatialImplicitSystem<'a, S: Coefficient> {
    plan: &'a CommonSpatialPlan,
    equations: &'a ExecutableSpatialEquations<S>,
    mapping: RegionDofMap<S>,
    dimension: usize,
    algebraic: BTreeSet<usize>,
}

/// One transformed Newton point, retaining original residuals and canonical CSR.
pub(super) struct ImplicitPoint {
    pub(super) unknown: Vec<f64>,
    pub(super) assembly: crate::region_assembly::nonlinear::RegionAssemblyLinearization,
    pub(super) row_scales: Vec<f64>,
}

impl ImplicitPoint {
    fn new(
        unknown: &[f64],
        assembly: crate::region_assembly::nonlinear::RegionAssemblyLinearization,
    ) -> Result<Self, Diagnostic> {
        let matrix = assembly.systems[0].matrix();
        let row_scales = matrix
            .row_offsets()
            .windows(2)
            .map(|offsets| {
                matrix.values()[offsets[0]..offsets[1]]
                    .iter()
                    .fold(0_f64, |scale, value| scale.max(value.abs()))
            })
            .collect::<Vec<_>>();
        if row_scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale <= 0.)
        {
            return Err(invalid(
                "implicit Jacobian requires finite nonzero equation rows",
            ));
        }
        Ok(Self {
            unknown: unknown.to_vec(),
            assembly,
            row_scales,
        })
    }
}

impl<'a, S: Coefficient + Send + Sync> SpatialImplicitSystem<'a, S>
where
    RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
{
    pub(in crate::numerical_admission) fn new(
        plan: &'a CommonSpatialPlan,
        equations: &'a ExecutableSpatialEquations<S>,
        time_s: f64,
    ) -> Result<Self, Diagnostic> {
        let (mapping, input) = plan.scalar_assembly_at(equations, time_s)?;
        let dimension = RegionRealCoordinates::new(&mapping)?.free_count();
        let algebraic = mapping
            .temporal_coordinate_partition(&input.temporal_structure()?)?
            .0;
        Ok(Self {
            plan,
            equations,
            mapping,
            dimension,
            algebraic,
        })
    }

    /// Assemble an exact stage Jacobian through the same local actions as JVP.
    /// This preserves canonical CSR for the Plan's admitted linear provider;
    /// columns are never reconstructed by expanding coordinate basis vectors.
    pub(in crate::numerical_admission) fn linearize(
        &self,
        time_s: f64,
        state: &[f64],
        derivative: &[f64],
        pullback: FieldPullback<'_>,
    ) -> Result<crate::region_assembly::nonlinear::RegionAssemblyLinearization, Diagnostic> {
        self.validate(state)?;
        self.validate(derivative)?;
        let (prepared, rates) = self.prepare(time_s, derivative)?;
        prepared.work.linearize(
            &prepared.plan,
            prepared.plan.target_id(0).expect("reduced target"),
            RegionPoint::Independent {
                state,
                rate_target: prepared.plan.target_id(1).expect("full target"),
                rates: &rates,
                pullback,
            },
            &eqiora_assembly::REFERENCE_ASSEMBLY_BACKEND,
        )
    }

    fn prepare(
        &self,
        time_s: f64,
        derivative: &[f64],
    ) -> Result<(RealRegionAssembly<S>, Vec<f64>), Diagnostic> {
        let (mapping, input) = self.plan.scalar_assembly_at(self.equations, time_s)?;
        if !mapping.has_same_coordinate_layout(&self.mapping) {
            return Err(invalid(
                "Time binding changed the exact spatial coordinate layout",
            ));
        }
        let (partial_mapping, _) = self.plan.scalar_time_partial(
            self.equations,
            time_s,
            InputPartialScope::EssentialTraces,
        )?;
        let fixed = mapping.prescribed_rates(&partial_mapping)?;
        let rates = RegionRealCoordinates::new(&mapping)?.lift_rates(derivative, &fixed)?;
        let prepared = match self.plan.admission.resources() {
            NativeMeshResources::Cartesian { mesh, .. } => {
                mapping.prepare_numeric_assembly(mesh.mesh(), &input)?
            }
            NativeMeshResources::GmshSimplicial { mesh, .. }
            | NativeMeshResources::AffineTriangleSimplicial { mesh, .. } => {
                mapping.prepare_numeric_assembly(mesh.mesh(), &input)?
            }
            _ => {
                return Err(invalid(
                    "spatial time actions require the exact admitted Mesh",
                ));
            }
        };
        Ok((prepared, rates))
    }

    fn validate(&self, values: &[f64]) -> Result<(), Diagnostic> {
        if values.len() != self.dimension || values.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "spatial time actions require exact finite free coordinates",
            ));
        }
        Ok(())
    }
}

impl<S: Coefficient + Send + Sync> ImplicitTimeSystem for SpatialImplicitSystem<'_, S>
where
    RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
{
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn residual(
        &self,
        time: f64,
        state: &[f64],
        derivative: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        self.validate(state)?;
        self.validate(derivative)?;
        if output.len() != self.dimension {
            return Err(invalid(
                "spatial time residual output has a foreign dimension",
            ));
        }
        let (prepared, rates) = self.prepare(time, derivative)?;
        let result = prepared.work.residual(
            &prepared.plan,
            prepared.plan.target_id(0).expect("reduced target"),
            RegionPoint::Independent {
                state,
                rate_target: prepared.plan.target_id(1).expect("full target"),
                rates: &rates,
                pullback: FieldPullback::Uniform {
                    current: None,
                    rate: None,
                },
            },
        )?;
        output.copy_from_slice(&result);
        Ok(())
    }

    fn residual_jvp(
        &self,
        time: f64,
        state: &[f64],
        derivative: &[f64],
        state_direction: &[f64],
        derivative_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        self.validate(state)?;
        self.validate(derivative)?;
        self.validate(state_direction)?;
        self.validate(derivative_direction)?;
        if output.len() != self.dimension {
            return Err(invalid(
                "spatial time tangent output has a foreign dimension",
            ));
        }
        let (prepared, rates) = self.prepare(time, derivative)?;
        let weights = state_direction
            .iter()
            .zip(derivative_direction)
            .map(|(state, rate)| [Some(*state), Some(*rate)])
            .collect::<Vec<_>>();
        let result = prepared.work.jacobian_action(
            &prepared.plan,
            prepared.plan.target_id(0).expect("reduced target"),
            RegionPoint::Independent {
                state,
                rate_target: prepared.plan.target_id(1).expect("full target"),
                rates: &rates,
                pullback: FieldPullback::Coordinates(&weights),
            },
            &vec![1.; self.dimension],
        )?;
        output.copy_from_slice(&result);
        Ok(())
    }
}
