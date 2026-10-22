//! Exact coefficient entities on anisotropic two- and three-dimensional meshes.
use super::*;
use eqiora_meshing::MeshGeometry;

fn rectangle_geometry() -> CanonicalGeometryV1 {
    CanonicalGeometryV1::decode_cartesian_box_v1_canonical(
        br#"{"schema":"eqiora.cartesian-box-envelope/v1","encoding":"eqiora.canonical-json/v1","length_unit":"metre","bounds":[[0.0,1.0],[0.0,1.0]],"entity_sets":[{"name":"x_lower","dimension":1,"members":[0]},{"name":"x_upper","dimension":1,"members":[1]},{"name":"y_lower","dimension":1,"members":[2]},{"name":"y_upper","dimension":1,"members":[3]},{"name":"body","dimension":2,"members":[0]}]}"#,
        eqiora_geometry::CanonicalGeometryLimits::default(),
    )
    .unwrap()
}

#[test]
fn scalar_domains_preserve_cartesian_field_entities_in_two_and_three_dimensions() {
    for (dimension, geometry) in [(2, rectangle_geometry()), (3, cartesian_box_3d())] {
        let boundaries = [
            "x_lower", "x_upper", "y_lower", "y_upper", "z_lower", "z_upper",
        ];
        let boundaries = &boundaries[..2 * dimension];
        let source = if dimension == 2 {
            POISSON_BOX
                .lines()
                .filter(|line| !line.contains("z_lower") && !line.contains("z_upper"))
                .collect::<Vec<_>>()
                .join("\n")
                .replace("ambient_dimension = 3", "ambient_dimension = 2")
        } else {
            POISSON_BOX.to_owned()
        };
        for complex in [false, true] {
            let (balance, trace) = if complex {
                (
                    "-div(math.complex(2, 3)*grad(potential)) + math.complex(3, -2)*source_scale*potential - math.complex(5 + 4*coordinate(0)/1[m], 1 - 7*coordinate(0)/1[m])*source_scale = 0",
                    "trace(potential) = math.complex(1 + 2*coordinate(0)/1[m], 1 - coordinate(0)/1[m])",
                )
            } else {
                (
                    "-div(2*grad(potential)) + 3*source_scale*potential - (3 + 6*coordinate(0)/1[m])*source_scale = 0",
                    "trace(potential) = 1 + 2*coordinate(0)/1[m]",
                )
            };
            let mut source = source
                .replace("-div(grad(potential)) - source_scale = 0", balance)
                .replace("trace(potential) = 0", trace);
            if complex {
                source = source.replace("variable potential: 1", "variable potential: complex<1>");
            }
            let model = scalar_box_model(&geometry, &source, "PoissonBox", boundaries);
            let cells = &[2, 3, 4][..dimension];
            for method in [
                CommonSpatialPolicy::Q1,
                CommonSpatialPolicy::CellCenteredTpfa,
            ] {
                let owner = cartesian_box_resources(&geometry, cells);
                let NativeMeshResources::Cartesian { mesh, .. } = &owner.resources else {
                    unreachable!()
                };
                let mesh = mesh.mesh().clone();
                let plan = ResolvedCommonPlan::resolve(
                    &model,
                    owner,
                    method,
                    CommonSolvePolicy::Linear(exact_reference_linear(
                        LinearSolver::BiConjugateGradientStabilized,
                        1e-12,
                        1e-14,
                        NonZeroUsize::new(512).unwrap(),
                    )),
                    None,
                    None,
                    &REFERENCE_LINEAR_SOLVER,
                    None,
                )
                .unwrap();
                let plan = replay_plan(plan, &REFERENCE_LINEAR_SOLVER);
                let scalar = plan.as_spatial().unwrap();
                let (field, ty) = scalar.fields().next().unwrap();
                let result = scalar.run_result(None, &REFERENCE_LINEAR_SOLVER).unwrap();
                assert_eq!(
                    result.field(0).map(|(_, ty, _)| ty.scalar_domain()),
                    Some(ty.scalar_domain())
                );
                assert_eq!(
                    result
                        .field(0)
                        .and_then(|(_, ty, space)| space.coefficient_dimension(ty.dimension())),
                    Some(DimExponents::DIMENSIONLESS)
                );
                let (association, values, shape) = result.field_block(0, 0).unwrap();
                let nodal = method == CommonSpatialPolicy::Q1;
                assert_eq!(association, if nodal { "vertex" } else { "cell" });
                assert_eq!(
                    shape,
                    cells
                        .iter()
                        .map(|n| n + usize::from(nodal))
                        .collect::<Vec<_>>()
                );
                let entities = scalar.field_coefficient_entities(field).unwrap();
                assert_eq!(entities.len(), shape.iter().product::<usize>());
                for (entity, value) in
                    entities
                        .iter()
                        .zip(values.chunks_exact(if complex { 2 } else { 1 }))
                {
                    assert_eq!(entity.dimension(), if nodal { 0 } else { dimension });
                    let x = if nodal {
                        mesh.vertex_coordinates(*entity).unwrap()[0]
                    } else {
                        mesh.geometry_map(*entity).unwrap().origin()[0]
                    };
                    assert!((value[0] - (1. + 2. * x)).abs() < 1e-10);
                    if complex {
                        assert!((value[1] - (1. - x)).abs() < 1e-10);
                    }
                }
                assert_eq!(
                    crate::CommonResult::from_bytes(&result.to_bytes().unwrap(), &plan).unwrap(),
                    result
                );
            }
        }
    }
}

#[test]
fn transient_tpfa_keeps_affine_time_data_on_anisotropic_cartesian_cells() {
    for (dimension, geometry) in [(2, rectangle_geometry()), (3, cartesian_box_3d())] {
        let boundaries = [
            "x_lower", "x_upper", "y_lower", "y_upper", "z_lower", "z_upper",
        ];
        let boundaries = &boundaries[..2 * dimension];
        let affine = (0..dimension)
            .map(|axis| format!("+{}*coordinate({axis})/1[m]", axis + 1))
            .collect::<String>();
        for complex in [false, true] {
            let (scalar, multiplier) = if complex {
                ("complex<1>", "math.complex(1,2)")
            } else {
                ("1", "1")
            };
            let supports = boundaries
                .iter()
                .map(|name| format!(",support {name}:boundary(parent=body)"))
                .collect::<String>();
            let traces = boundaries
                .iter()
                .map(|name| {
                    format!("relation {name}_value on {name} {{trace(u)={multiplier}*(1{affine}+time()/1[s]);}}")
                })
                .collect::<String>();
            let source = format!(
                "public component Heat(support body:volume(ambient_dimension={dimension}){supports},parameter source_scale:1/m^2) {{
                    state u:{scalar} on body in smooth;
                    law balance on body {{storage 1[s/m^2]*u; flux -grad(u); source {multiplier}*source_scale;}}
                    {traces}
                }}"
            );
            let model = scalar_box_model(&geometry, &source, "Heat", boundaries);
            for nonlinear in [false, true] {
                let owner = cartesian_box_resources(&geometry, &[2, 3, 4][..dimension]);
                let NativeMeshResources::Cartesian { mesh, .. } = &owner.resources else {
                    unreachable!()
                };
                let mesh = mesh.mesh().clone();
                let linear = exact_reference_linear(
                    LinearSolver::BiConjugateGradientStabilized,
                    1e-12,
                    1e-14,
                    NonZeroUsize::new(512).unwrap(),
                );
                let solve = if nonlinear {
                    CommonSolvePolicy::Newton {
                        linear,
                        nonlinear: NonlinearSolvePlan::new(
                            0.,
                            1e-11,
                            NonZeroUsize::new(16).unwrap(),
                            16,
                        )
                        .unwrap(),
                    }
                } else {
                    CommonSolvePolicy::Linear(linear)
                };
                let resolved = ResolvedCommonPlan::resolve(
                    &model,
                    owner,
                    CommonSpatialPolicy::CellCenteredTpfa,
                    solve,
                    None,
                    Some(CommonBackwardEuler::from_seconds(0.25).unwrap()),
                    &REFERENCE_LINEAR_SOLVER,
                    None,
                )
                .unwrap();
                let resolved = replay_plan(resolved, &REFERENCE_LINEAR_SOLVER);
                let plan = resolved.as_spatial().unwrap();
                let (field, ty) = plan.fields().next().unwrap();
                let entities = plan.field_coefficient_entities(field).unwrap();
                let spatial_value = |entity| {
                    1. + mesh
                        .geometry_map(entity)
                        .unwrap()
                        .origin()
                        .iter()
                        .enumerate()
                        .map(|(axis, x)| (axis + 1) as f64 * x)
                        .sum::<f64>()
                };
                let coefficients = entities
                    .iter()
                    .map(|&entity| {
                        let value = spatial_value(entity);
                        C::new(value, if complex { 2. * value } else { 0. })
                    })
                    .collect::<Vec<_>>();
                let assignment = CommonInitialField::new(
                    resolved.model_artifact().digest().unwrap(),
                    field,
                    None,
                    Some(CommonInitialValues::new(ty.shape().clone(), coefficients).unwrap()),
                )
                .unwrap();
                let initial = plan.initial_state(Some(0.), vec![assignment]).unwrap();
                let before = initial.to_bytes().unwrap();
                let mut state = initial.clone();
                for step in 0..=3 {
                    if step != 0 {
                        state = plan
                            .advance_scalar(&state, &REFERENCE_LINEAR_SOLVER)
                            .unwrap()
                            .state()
                            .clone();
                    }
                    let width = if complex { 2 } else { 1 };
                    let values = state.spatial_values().unwrap();
                    assert_eq!(values.len(), width * entities.len());
                    // An affine field has exact opposite face fluxes. Its cell
                    // average equals its centroid value; BE differentiates t exactly.
                    for (entity, value) in entities.iter().zip(values.chunks_exact(width)) {
                        assert_eq!(entity.dimension(), dimension);
                        let expected = spatial_value(*entity) + step as f64 / 4.;
                        assert!((value[0] - expected).abs() < 1e-10);
                        if complex {
                            assert!((value[1] - 2. * expected).abs() < 1e-10);
                        }
                    }
                    state = CommonState::from_bytes(&state.to_bytes().unwrap(), &resolved).unwrap();
                }
                assert_eq!(initial.to_bytes().unwrap(), before);
            }
        }
    }
}
