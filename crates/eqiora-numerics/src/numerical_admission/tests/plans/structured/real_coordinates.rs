//! The same finalized solve/reacceptance boundary serves both numerical domains.
use super::*;
use crate::region_assembly::mapping::RegionAlgebraicCoordinates;
use num_complex::Complex64 as C;

#[test]
fn finalized_real_storage_preserves_complex_channels_and_reaccepts_original_equations() {
    let geometry = cartesian_interval();
    for conjugate in [false, true] {
        let (extra, forcing) = if conjugate {
            (
                "+math.complex(1,2)*math.conj(potential)",
                "[math.complex(12+8*coordinate(0)/1[m],4-2*coordinate(0)/1[m]),math.complex(4+12*coordinate(0)/1[m],6+8*coordinate(0)/1[m])]",
            )
        } else {
            (
                "",
                "[math.complex(7+7*coordinate(0)/1[m],4-9*coordinate(0)/1[m]),math.complex(5*coordinate(0)/1[m],13+14*coordinate(0)/1[m])]",
            )
        };
        let source = POISSON_INTERVAL
            .replace("variable potential: 1", "variable potential: array<complex<1>,2>")
            .replace("-div(grad(potential)) - source_scale = 0", &format!(
                "-div(math.complex(2,3)*grad(potential))+source_scale*(math.complex(3,-2)*potential{extra}-{forcing})=0"
            ))
            .replace("relation left_value on left { trace(potential) = 0; }", "relation left_value on left {trace(potential)=[math.complex(1,2),math.complex(-2,3)];}")
            .replace("relation right_value on right { trace(potential) = 0; }", "relation right_value on right {trace(potential)=[math.complex(4,1),math.complex(-3,7)];}");
        let model = scalar_box_model(&geometry, &source, "PoissonInterval", &["left", "right"]);
        let request = CommonSpatialRequest::Uniform(CommonSpatialPolicy::Q1);
        let recognized = RecognizedNativeAdmission::recognize(
            &model,
            cartesian_box_resources(&geometry, &[4]),
            &request,
            None,
        )
        .unwrap();
        let RecognizedNativeModel::ComplexSpatial(equations) = &recognized.recognized else {
            panic!("complex physical equations");
        };
        let selected = equations.resolve_discretizations(&request, &model).unwrap();
        let solver = SolverPlan::new(
            LinearSolver::BiConjugateGradientStabilized,
            1e-12,
            1e-14,
            NonZeroUsize::new(100).unwrap(),
        )
        .unwrap();
        let finalize = || {
            equations
                .assemble_real_static(
                    &recognized.resources,
                    &selected,
                    &eqiora_realization::SystemScaling::Dimensional,
                )
                .unwrap()
                .finalize_real(NonZeroUsize::MIN, solver)
        };
        let system = finalize();
        assert_eq!(system.core.canonical_csr_system_view().rows(), 12);
        assert_eq!(
            system
                .core
                .canonical_csr_system_view()
                .complex_coordinate_system()
                .unwrap()
                .is_some(),
            !conjugate,
        );
        let initial_residual = system.core.residual_norm(&[0.; 12]).unwrap();
        system
            .validate_initial_residual(None, initial_residual)
            .unwrap();
        assert!(
            system
                .validate_initial_residual(None, initial_residual.next_up())
                .is_err()
        );
        let output = system
            .solve(
                LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, solver),
                None,
            )
            .unwrap();
        let [(field, ty, values, space)] = output.fields.as_slice() else {
            panic!("complete channel Field");
        };
        assert_eq!(ty.scalar_domain(), eqiora_core::ScalarDomain::Complex);
        assert_eq!(ty.shape().component_count(), Some(2));
        assert_eq!(
            *space,
            Space::continuous_lagrange(std::num::NonZeroU16::MIN)
        );
        assert_eq!(values.len(), 10);
        // Independent continuum solution has zero curvature. Both coupled real
        // parts and both channel slopes are represented exactly by Q1.
        for (pair, x) in values
            .as_chunks::<2>()
            .0
            .iter()
            .zip([0., 0.25, 0.5, 0.75, 1.])
        {
            for (actual, expected) in pair
                .iter()
                .zip([C::new(1. + 3. * x, 2. - x), C::new(-2. - x, 3. + 4. * x)])
            {
                assert!((*actual - expected).norm() < 2e-10);
            }
        }
        let candidate = BTreeMap::from([(field.erase(), values.clone())]);
        let system = finalize();
        system.validate_coefficients(&candidate).unwrap();
        let physical = system
            .mapping
            .physical_fields_from_coefficients(&candidate)
            .unwrap();
        let reduced: Vec<f64> = system.mapping.validated_coordinates(&physical).unwrap();
        system
            .validate_initial_residual(
                Some(&candidate),
                system.core.residual_norm(&reduced).unwrap(),
            )
            .unwrap();
        let mut changed = candidate.clone();
        changed.get_mut(&field.erase()).unwrap()[2].im += 0.1;
        assert!(system.validate_coefficients(&changed).is_err());
        changed.get_mut(&field.erase()).unwrap()[0].re += 0.1;
        assert!(system.validate_coefficients(&changed).is_err());
        let resumed = system
            .solve(
                LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, solver),
                Some(&candidate),
            )
            .unwrap();
        assert_eq!(resumed.fields, output.fields);

        let plan = ResolvedCommonPlan::resolve(
            &model,
            cartesian_box_resources(&geometry, &[4]),
            CommonSpatialPolicy::Q1,
            CommonSolvePolicy::Linear(exact_reference_linear(
                solver.algorithm(),
                solver.relative_tolerance(),
                solver.absolute_tolerance(),
                solver.maximum_iterations(),
            )),
            None,
            None,
            &REFERENCE_LINEAR_SOLVER,
            None,
        )
        .unwrap();
        let plan = replay_plan(plan, &REFERENCE_LINEAR_SOLVER);
        let spatial = plan.as_spatial().unwrap();
        assert_eq!(
            spatial.admission.linear.scalar_domain,
            if conjugate {
                eqiora_core::ScalarDomain::Real
            } else {
                eqiora_core::ScalarDomain::Complex
            }
        );
        let result = spatial.run_result(None, &REFERENCE_LINEAR_SOLVER).unwrap();
        let bytes = result.to_bytes().unwrap();
        let replayed = crate::CommonResult::from_bytes(&bytes, &plan).unwrap();
        assert_eq!(replayed.to_bytes().unwrap(), bytes);
        let (_, actual, shape) = replayed.field_block(0, 0).unwrap();
        assert_eq!(shape, &[5, 2]);
        for (actual, expected) in actual
            .iter()
            .zip(values.iter().flat_map(|value| [value.re, value.im]))
        {
            assert!((*actual - expected).abs() < 2e-10);
        }
    }
}
