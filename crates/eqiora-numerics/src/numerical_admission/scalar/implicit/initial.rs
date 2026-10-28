//! Complete physical initial rates without changing the accepted initial State.
use super::*;
use eqiora_solver::{FixedOrderInnerProduct, LinearOperator, SERIAL_LINEAR_EXECUTION};
use eqiora_time::ConstantDerivativeMatrixProof;
use num_rational::BigRational;

impl<S: Coefficient + Send + Sync> SpatialImplicitSystem<'_, S>
where
    RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
{
    pub(in crate::numerical_admission) fn initial_rates(
        &self,
        time: f64,
        state: &[f64],
        backend: &dyn LinearSolverBackend,
    ) -> Result<Vec<f64>, Diagnostic> {
        let coordinates = RegionRealCoordinates::new(&self.mapping)?;
        let (prepared, fixed) = self.prepare(time, &vec![0.; self.dimension])?;
        let mass = self.linearize(
            time,
            state,
            &vec![0.; self.dimension],
            FieldPullback::Uniform {
                current: None,
                rate: Some(1.),
            },
        )?;
        let all = (0..self.dimension).collect::<Vec<_>>();
        let proof = ConstantDerivativeMatrixProof::new(
            self.dimension,
            crate::region_assembly::mapping::square_projection(mass.systems[0].matrix(), &all)?,
        )?;
        let full_rate = prepared.consistent_rates(
            &coordinates,
            (state, &fixed),
            &proof,
            &mass.residuals[0]
                .iter()
                .map(|value| -*value)
                .collect::<Vec<_>>(),
            &BTreeSet::new(),
            self.plan.linear(),
        )?;
        let mut rate = coordinates.restrict_action(&full_rate)?;
        if self.algebraic.is_empty() {
            return Ok(rate);
        }
        // Differentiate F(t,u,v)=0. Columns are differential accelerations
        // and algebraic velocities, with differential velocities already fixed.
        let weights = (0..self.dimension)
            .map(|i| {
                if self.algebraic.contains(&i) {
                    [Some(1.), None]
                } else {
                    [None, Some(1.)]
                }
            })
            .collect::<Vec<_>>();
        let mut tangent =
            self.linearize(time, state, &rate, FieldPullback::Coordinates(&weights))?;
        if ConstantDerivativeMatrixProof::new(
            self.dimension,
            crate::region_assembly::mapping::square_projection(tangent.systems[0].matrix(), &all)?,
        )?
        .exact_rank()
            != self.dimension
        {
            return Err(invalid(
                "physical initial rates require a regular index-one State/rate Jacobian",
            ));
        }
        tangent.residuals[0] =
            self.initial_tangent_source(time, state, &full_rate, &proof, &mass, &prepared)?;
        let structure = self.equations.algebraic_structure(None)?;
        let checked = self
            .plan
            .admission
            .linear
            .checked_backend(backend, Some(&structure))?;
        let (unknown, _) = tangent.solve_correction(
            0,
            LinearSolveRequest::new(&checked, self.plan.linear()),
            self.plan.admission.linear.workers,
            None,
        )?;
        for &i in &self.algebraic {
            rate[i] = unknown[i];
        }
        // Algebraic rates must remain absent from the original residual.
        let mut residual = vec![0.; self.dimension];
        self.residual(time, state, &rate, &mut residual)?;
        if FixedOrderInnerProduct::norm(&SERIAL_LINEAR_EXECUTION, &residual)?
            > self
                .plan
                .linear()
                .residual_target(FixedOrderInnerProduct::norm(
                    &SERIAL_LINEAR_EXECUTION,
                    &mass.residuals[0],
                )?)?
        {
            return Err(invalid(
                "physical initial rates violate the original residual",
            ));
        }
        Ok(rate)
    }

    fn initial_tangent_source(
        &self,
        time: f64,
        state: &[f64],
        full_rate: &[f64],
        mass_proof: &ConstantDerivativeMatrixProof,
        mass: &crate::region_assembly::nonlinear::RegionAssemblyLinearization,
        prepared: &RealRegionAssembly<S>,
    ) -> Result<Vec<f64>, Diagnostic> {
        let coordinates = RegionRealCoordinates::new(&self.mapping)?;
        let (_, input) = self.plan.scalar_assembly_at(self.equations, time)?;
        let partial = self.plan.scalar_time_partial(
            self.equations,
            time,
            &[InputPartialScope::CoefficientData],
        )?;
        let (_, partial) = match self.plan.admission.resources() {
            NativeMeshResources::Cartesian { mesh, .. } => {
                self.mapping
                    .prepare_initial_rate_data(mesh.mesh(), &input, partial)?
            }
            NativeMeshResources::GmshSimplicial { mesh, .. }
            | NativeMeshResources::AffineTriangleSimplicial { mesh, .. } => self
                .mapping
                .prepare_initial_rate_data(mesh.mesh(), &input, partial)?,
            _ => {
                return Err(invalid(
                    "physical initial rates require the admitted affine Mesh",
                ));
            }
        };
        let coordinate = eqiora_ir::ScalarSymbolCoordinate::for_value(
            eqiora_schema::kernel::SymbolRef::Time,
            &eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, TIME)
                .expect("real scalar Time"),
        )?
        .remove(0);
        let point = RegionPoint::Independent {
            state,
            rate_target: prepared.plan.target_id(1).expect("full target"),
            rates: full_rate,
            pullback: FieldPullback::Uniform {
                current: None,
                rate: None,
            },
        };
        let mut source = partial.work.residual(
            &partial.plan,
            partial.plan.target_id(0).expect("reduced target"),
            point,
        )?;
        let input_partial = prepared.work.projected_input_partial(
            &prepared.plan,
            prepared.plan.target_id(0).expect("reduced target"),
            point,
            (&partial.work, &partial.plan),
            &coordinate,
            mass_proof,
        )?;
        for (value, partial) in source.iter_mut().zip(input_partial) {
            *value += partial;
        }
        let state_action = self.linearize(
            time,
            state,
            &coordinates.restrict_action(full_rate)?,
            FieldPullback::Uniform {
                current: Some(1.),
                rate: None,
            },
        )?;
        let mut action = vec![0.; coordinates.full_count()];
        state_action.systems[1]
            .matrix()
            .apply(full_rate, &mut action)?;
        for (value, action) in source.iter_mut().zip(coordinates.restrict_action(&action)?) {
            *value += action;
        }
        // A prescribed acceleration in the free mass range only changes the
        // nuisance differential acceleration. Demand second trace derivatives
        // only when that cancellation is not certified by the exact rank owner.
        let matrix = mass.systems[1].matrix();
        let mut needs_acceleration = false;
        for column in 0..coordinates.full_count() {
            if coordinates.free_coordinate(column).is_some() {
                continue;
            }
            let mut direction = vec![0.; coordinates.full_count()];
            direction[column] = 1.;
            matrix.apply(&direction, &mut action)?;
            let column = coordinates
                .restrict_action(&action)?
                .into_iter()
                .map(|value| {
                    BigRational::from_float(value)
                        .ok_or_else(|| invalid("non-finite initial mass action"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if !mass_proof.is_compatible(&[], &column)? {
                needs_acceleration = true;
                break;
            }
        }
        if needs_acceleration {
            let (mapping, _) = self.plan.scalar_time_partial(
                self.equations,
                time,
                &[
                    InputPartialScope::EssentialTraces,
                    InputPartialScope::EssentialTraces,
                ],
            )?;
            let acceleration = self.mapping.prescribed_rates(&mapping)?;
            matrix.apply(&acceleration, &mut action)?;
            for (value, action) in source.iter_mut().zip(coordinates.restrict_action(&action)?) {
                *value += action;
            }
        }
        Ok(source)
    }
}
