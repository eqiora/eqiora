use super::*;
use eqiora_meshing::CellId;

#[test]
fn refined_state_interpolates_interior_and_lifts_exact_target_trace() {
    let geometry = rectangle();
    let region = geometry.entity_set("region").unwrap();
    let mut supports = vec![("region", region, None)];
    supports.extend(["left", "right", "bottom", "top"].map(|name| {
        (
            name,
            geometry.entity_set(name).unwrap(),
            Some(("region", region)),
        )
    }));
    let mut previous_model_result = None;
    for quadratic in [false, true] {
        let trace = if quadratic {
            "coordinate(0)*coordinate(0)/1[m]"
        } else {
            "2*coordinate(0)-3*coordinate(1)"
        };
        let forcing = if quadratic { "-2[1/m]" } else { "0[1/m]" };
        let source = format!(
            "public component Transfer(
            support region:volume(ambient_dimension=2),
            support left:boundary(parent=region), support right:boundary(parent=region),
            support bottom:boundary(parent=region), support top:boundary(parent=region)) {{
            variable u:m on region in h1;
            relation balance on region {{ -div(grad(u))={forcing}; }}
            relation a on left {{ trace(u)={trace}; }}
            relation b on right {{ trace(u)={trace}; }}
            relation c on bottom {{ trace(u)={trace}; }}
            relation d on top {{ trace(u)={trace}; }}
        }}"
        );
        let model = compile_model(
            "transfer.eqi",
            &source,
            &geometry,
            "Transfer",
            &supports,
            &[],
        );
        let resolve = |mesh| {
            ResolvedCommonPlan::resolve(
                &model,
                mesh,
                CommonSpatialPolicy::P1,
                CommonSolvePolicy::Linear(exact_reference_linear(
                    LinearSolver::BiConjugateGradientStabilized,
                    1e-12,
                    1e-14,
                    NonZeroUsize::new(1000).unwrap(),
                )),
                None,
                None,
                &REFERENCE_LINEAR_SOLVER,
                None,
                None,
            )
            .unwrap()
        };
        let coarse_mesh = affine_resources(&geometry);
        let coarse = resolve(coarse_mesh.clone());
        let result = coarse
            .as_spatial()
            .unwrap()
            .run_result(None, &REFERENCE_LINEAR_SOLVER)
            .unwrap();
        let source_bytes = result.to_bytes().unwrap();
        let count = coarse_mesh.simplicial_mesh().unwrap().mesh().cells().len();
        let first = coarse_mesh
            .refine(&(0..count).map(CellId::new).collect::<Vec<_>>(), 1000)
            .unwrap();
        let count = first.simplicial_mesh().unwrap().mesh().cells().len();
        let target_mesh = first
            .refine(&(0..count).map(CellId::new).collect::<Vec<_>>(), 1000)
            .unwrap();
        let target_mesh =
            AuthenticatedCommonMesh::from_bytes(&target_mesh.to_bytes().unwrap()).unwrap();
        let target = resolve(target_mesh.clone());
        let target = replay_plan(target, &REFERENCE_LINEAR_SOLVER);
        let plan = target.as_spatial().unwrap();
        let result = crate::CommonResult::from_bytes(&source_bytes, &coarse).unwrap();
        if let Some(foreign) = &previous_model_result {
            let error = plan.interpolate_state(foreign).unwrap_err();
            assert!(
                error.message().contains("same stationary Model"),
                "{error:?}"
            );
        }
        previous_model_result = Some(result.clone());
        let state = plan.interpolate_state(&result).unwrap();
        assert_eq!(state.time_s(), None);
        let mut lifted_new_boundary = false;
        let mut interpolated_new_interior = false;
        let coarse_vertices = coarse_mesh
            .simplicial_mesh()
            .unwrap()
            .mesh()
            .vertices()
            .len();
        for (index, (value, point)) in state
            .spatial_values()
            .unwrap()
            .iter()
            .zip(target_mesh.simplicial_mesh().unwrap().mesh().vertices())
            .enumerate()
        {
            let [x, y] = <[f64; 2]>::try_from(point.as_slice()).unwrap();
            let boundary = x == 0. || x == 1. || y == 0. || y == 1.;
            // Uniform source nodal x^2 is piecewise linear in x with slopes
            // 1/2 and 3/2. This independent formula is not a sampled transfer.
            let interpolant = if x <= 0.5 { 0.5 * x } else { 1.5 * x - 0.5 };
            let expected = if !quadratic {
                2. * x - 3. * y
            } else if boundary {
                x * x
            } else {
                interpolant
            };
            assert!(
                (value - expected).abs() < 1e-12,
                "{point:?}: {value} != {expected}"
            );
            if quadratic && index >= coarse_vertices && (interpolant - x * x).abs() > 0.01 {
                lifted_new_boundary |= boundary;
                interpolated_new_interior |= !boundary;
            }
        }
        if quadratic {
            assert!(lifted_new_boundary && interpolated_new_interior);
        }
        let saved = state.to_bytes().unwrap();
        assert_eq!(CommonState::from_bytes(&saved, &target).unwrap(), state);
        assert!(CommonState::from_bytes(&saved, &coarse).is_err());
        let solved = plan
            .run_result(Some(&state), &REFERENCE_LINEAR_SOLVER)
            .unwrap();
        assert_eq!(
            crate::CommonResult::from_bytes(&solved.to_bytes().unwrap(), &target).unwrap(),
            solved
        );
        assert_eq!(result.to_bytes().unwrap(), source_bytes);
        assert!(
            coarse
                .as_spatial()
                .unwrap()
                .interpolate_state(&solved)
                .is_err()
        );
        // A sibling revision has no prefix relationship with this target.
        let sibling = resolve(coarse_mesh.refine(&[CellId::new(0)], 1000).unwrap());
        let sibling_result = sibling
            .as_spatial()
            .unwrap()
            .run_result(None, &REFERENCE_LINEAR_SOLVER)
            .unwrap();
        assert!(plan.interpolate_state(&sibling_result).is_err());
    }
}
