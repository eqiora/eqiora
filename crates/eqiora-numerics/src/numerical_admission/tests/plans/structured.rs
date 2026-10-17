//! Channel axes retain their meaning through ordinary spatial execution.
use super::*;

#[test]
fn channel_fields_retain_each_real_and_complex_component() {
    let geometry = cartesian_interval();
    for complex in [false, true] {
        let (ty, balance, left, right) = if complex {
            (
                "array<complex<1>,2>",
                "-div(math.complex(2,3)*grad(potential)) + math.complex(3,-2)*source_scale*potential - [math.complex(5+4*coordinate(0)/1[m],1-7*coordinate(0)/1[m]),math.complex(7*coordinate(0)/1[m],13+4*coordinate(0)/1[m])]*source_scale = 0",
                "[math.complex(1,1),math.complex(-2,3)]",
                "[math.complex(3,0),math.complex(-1,5)]",
            )
        } else {
            (
                "array<1,2>",
                "-div(2*grad(potential)) + 3*source_scale*potential - [3+6*coordinate(0)/1[m],-6+3*coordinate(0)/1[m]]*source_scale = 0",
                "[1,-2]",
                "[3,-1]",
            )
        };
        let source = POISSON_INTERVAL
            .replace(
                "variable potential: 1",
                &format!("variable potential: {ty}"),
            )
            .replace("-div(grad(potential)) - source_scale = 0", balance)
            .replace(
                "relation left_value on left { trace(potential) = 0; }",
                &format!("relation left_value on left {{ trace(potential) = {left}; }}"),
            )
            .replace(
                "relation right_value on right { trace(potential) = 0; }",
                &format!("relation right_value on right {{ trace(potential) = {right}; }}"),
            );
        let model = scalar_box_model(&geometry, &source, "PoissonInterval", &["left", "right"]);
        for method in [
            CommonSpatialPolicy::Q1,
            CommonSpatialPolicy::CellCenteredTpfa,
        ] {
            let plan = ResolvedCommonPlan::resolve(
                &model,
                cartesian_box_resources(&geometry, &[4]),
                method,
                CommonSolvePolicy::Linear(exact_reference_linear(
                    LinearSolver::BiConjugateGradientStabilized,
                    1e-12,
                    1e-14,
                    NonZeroUsize::new(128).unwrap(),
                )),
                None,
                None,
                &REFERENCE_LINEAR_SOLVER,
                None,
            )
            .unwrap_or_else(|error| panic!("{method:?}, complex={complex}: {error:?}"));
            let plan = replay_plan(plan, &REFERENCE_LINEAR_SOLVER);
            let spatial = plan.as_spatial().unwrap();
            let (_, value_type) = spatial.fields().next().unwrap();
            assert_eq!(value_type.array_rank(), 1);
            assert_eq!(value_type.frame(), eqiora_core::ValueFrame::Invariant);
            let result = spatial.run_result(None, &REFERENCE_LINEAR_SOLVER).unwrap();
            assert_eq!(
                result.field_scalar_domain(0),
                Some(value_type.scalar_domain())
            );
            let (association, values, shape) = result.field_block(0, 0).unwrap();
            let nodal = method == CommonSpatialPolicy::Q1;
            assert_eq!(association, if nodal { "vertex" } else { "cell" });
            assert_eq!(shape, &[if nodal { 5 } else { 4 }, 2]);
            let width = if complex { 4 } else { 2 };
            for (index, value) in values.chunks_exact(width).enumerate() {
                let x = (index as f64 + if nodal { 0. } else { 0.5 }) / 4.;
                // Two independent affine channels; no spatial-vector interpretation.
                let expected = if complex {
                    vec![1. + 2. * x, 1. - x, -2. + x, 3. + 2. * x]
                } else {
                    vec![1. + 2. * x, -2. + x]
                };
                for (actual, expected) in value.iter().zip(expected) {
                    assert!((actual - expected).abs() < 1e-10);
                }
            }
            assert_eq!(
                crate::CommonResult::from_bytes(&result.to_bytes().unwrap(), &plan).unwrap(),
                result
            );
        }
    }
}
