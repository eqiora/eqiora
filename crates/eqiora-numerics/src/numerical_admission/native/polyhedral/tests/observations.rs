use super::*;
use eqiora_core::{DimExponents, DynQuantity, entity::kinds};

#[test]
fn tetrahedral_point_observations_reconstruct_complete_real_and_complex_p1_fields() {
    for complex in [false, true] {
        for permuted in [false, true] {
            let (model, program, owner) = fixture_source(
                &[vec![0, 1, 2, 3]],
                false,
                permuted,
                2.,
                complex,
                |source| {
                    let real = "(1+coordinate(0)/1[m])*e0+(-2+2*coordinate(1)/1[m])*e1+(3-3*coordinate(2)/1[m])*e2";
                    let prescribed = if complex {
                        format!(
                            "math.complex({real},(2-coordinate(0)/1[m])*e0+(3+coordinate(1)/1[m])*e1+(-1+2*coordinate(2)/1[m])*e2)"
                        )
                    } else {
                        real.to_owned()
                    };
                    let source = source.replace("variable u:", "parameter e0:vector<1,3>=tensor_value(frame=body,components=[1,0,0]); parameter e1:vector<1,3>=tensor_value(frame=body,components=[0,1,0]); parameter e2:vector<1,3>=tensor_value(frame=body,components=[0,0,1]); variable u:");
                    let source = source
                        .replace(
                            "curl(curl(u)) = 0",
                            &format!("1[m^2]*curl(curl(u))+u={prescribed}"),
                        )
                        .replace(
                            "tangential_trace(-curl(u))",
                            "tangential_trace(-1[m^2]*curl(u))",
                        );
                    let (prefix, _) = source.rsplit_once('}').unwrap();
                    let scalar = if complex { "complex<1>" } else { "1" };
                    format!("{prefix} observable profile:vector<{scalar},3> on body=u; }}")
                },
            );
            let observable = program
                .nodes()
                .find_map(|node| match node {
                    KernelNode::Observable(definition) => Some(definition.id()),
                    _ => None,
                })
                .unwrap();
            let domain = program
                .observable_output_support(observable)
                .unwrap()
                .unwrap()
                .domain()
                .downcast::<kinds::Domain>()
                .unwrap();
            let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
            // Membership is admitted from exact Geometry before any mesh sampling.
            let bind = |values: &[f64], dimension| {
                eqiora_sem::EvaluationPoint::new(
                    &program,
                    domain,
                    values
                        .iter()
                        .enumerate()
                        .map(|(axis, value)| {
                            ((domain.erase(), axis), DynQuantity::new(*value, dimension))
                        })
                        .collect(),
                    None,
                )
            };
            assert!(bind(&[0.25, 0.5, 0.75], length).is_ok());
            assert!(bind(&[0., 0., 4.], length).is_ok());
            let outside = bind(&[2., 3., 4.], length).unwrap_err();
            assert!(
                outside
                    .message()
                    .contains("outside its exact geometry support")
            );
            assert!(bind(&[0., 0., 0.], DimExponents::DIMENSIONLESS).is_err());
            assert!(bind(&[0., 0.], length).is_err());
            assert!(bind(&[0., 0., f64::NAN], length).is_err());
            let solver = SolverPlan::new(
                LinearSolver::BiConjugateGradientStabilized,
                1e-13,
                1e-14,
                NonZeroUsize::new(2000).unwrap(),
            )
            .unwrap();
            let plan = ResolvedCommonPlan::resolve(
                &model,
                owner,
                CommonSpatialPolicy::P1,
                CommonSolvePolicy::Linear(
                    CommonLinearRequest::exact(solver, REFERENCE_LINEAR_SOLVER.provider()).unwrap(),
                ),
                None,
                None,
                &REFERENCE_LINEAR_SOLVER,
                None,
                None,
            )
            .unwrap();
            let result = plan
                .as_spatial()
                .unwrap()
                .run_result(None, &REFERENCE_LINEAR_SOLVER)
                .unwrap();
            let replay =
                crate::CommonResult::from_bytes(&result.to_bytes().unwrap(), &plan).unwrap();
            for result in [&result, &replay] {
                let sample = result
                    .observe_at(
                        &model,
                        observable,
                        &[0.25, 0.5, 0.75].map(|x| DynQuantity::new(x, length)),
                        &Default::default(),
                        None,
                    )
                    .unwrap();
                assert_eq!(sample.value().value_type(), result.field(0).unwrap().1);
                // The physical affine fields above are curl-free. Their mass
                // equations reproduce them exactly in P1, including each phase.
                for ((real, imaginary), (a, b)) in sample.value().components().unwrap().zip([
                    (1.25, 1.75),
                    (-1., 3.5),
                    (0.75, 0.5),
                ]) {
                    assert!((real - a).abs() < 1e-10);
                    assert!((imaginary - if complex { b } else { 0. }).abs() < 1e-10);
                }
                assert!(
                    result
                        .observe_at(
                            &model,
                            observable,
                            &[2., 3., 4.].map(|x| DynQuantity::new(x, length)),
                            &Default::default(),
                            None,
                        )
                        .is_err()
                );
            }
        }
    }
}
