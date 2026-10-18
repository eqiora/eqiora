//! Channel axes retain their meaning through ordinary spatial execution.
use super::*;
mod anisotropic;
mod differentiation;
mod integrals;
mod observations;
mod real_coordinates;

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
        let source = observations::channel_source(&source, complex);
        let source = integrals::source(&source, complex);
        let body = geometry.entity_set("body").unwrap();
        let scalar = eqiora_core::ValueType::scalar(
            if complex {
                eqiora_core::ScalarDomain::Complex
            } else {
                eqiora_core::ScalarDomain::Real
            },
            DimExponents::DIMENSIONLESS,
        )
        .unwrap();
        let weights = eqiora_core::ValueLiteral::new(
            scalar.array(2).unwrap(),
            if complex {
                vec![(2., 3.), (-1., 2.)]
            } else {
                vec![(2., 0.), (-3., 0.)]
            },
        )
        .unwrap();
        let model = compile_typed_model(
            "observed-channels.eqi",
            &source,
            &geometry,
            "PoissonInterval",
            &[
                ("body", body, None),
                (
                    "left",
                    geometry.entity_set("left").unwrap(),
                    Some(("body", body)),
                ),
                (
                    "right",
                    geometry.entity_set("right").unwrap(),
                    Some(("body", body)),
                ),
            ],
            &[
                (
                    "source_scale",
                    eqiora_core::ValueLiteral::try_from(DynQuantity::new(
                        1.,
                        DimExponents::from_integers([0, -2, 0, 0, 0, 0, 0]).unwrap(),
                    ))
                    .unwrap(),
                ),
                ("weights", weights),
                (
                    "observation_scale",
                    eqiora_core::ValueLiteral::try_from(DynQuantity::new(
                        1.,
                        DimExponents::DIMENSIONLESS,
                    ))
                    .unwrap(),
                ),
            ],
        );
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
            observations::check_channels(&model, &plan, &result, method, complex);
            integrals::check(&model, &plan, &result, method, complex);
            let expected_type = eqiora_core::ValueType::scalar(
                if complex {
                    eqiora_core::ScalarDomain::Complex
                } else {
                    eqiora_core::ScalarDomain::Real
                },
                DimExponents::DIMENSIONLESS,
            )
            .unwrap()
            .array(2)
            .unwrap();
            assert_eq!(result.field(0).unwrap().1, &expected_type);
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
            // Same domain, dimension and shape do not make channels spatial axes.
            // Keep the exact Model/Plan identity and corrupt only the retained type
            // to reach the full-type cross-link check before numerical acceptance.
            let mut substituted = spatial.clone();
            substituted.fields[0].1 = eqiora_core::ValueType::shaped(
                expected_type.scalar_domain(),
                expected_type.dimension(),
                expected_type.shape().clone(),
                eqiora_core::ValueFrame::SpatialCartesian,
            )
            .unwrap();
            let substituted = ResolvedCommonPlan::Spatial(Box::new(substituted));
            let error = crate::CommonResult::from_bytes(&result.to_bytes().unwrap(), &substituted)
                .unwrap_err();
            assert!(
                error
                    .message()
                    .contains("Result Fields differ from the exact Plan")
            );
        }
    }
}

#[test]
fn nonsymmetric_component_map_preserves_vector_field_and_nonunit_geometry() {
    for complex in [false, true] {
        for prescribed_fields in [false, true] {
            for diffusion in [false, true] {
                nonsymmetric_component_map(complex, prescribed_fields, diffusion);
            }
        }
    }
}

fn nonsymmetric_component_map(complex: bool, prescribed_fields: bool, diffusion: bool) {
    use eqiora_meshing::MeshGeometry;
    let geometry = CanonicalGeometryV1::decode_cartesian_box_v1_canonical(
        br#"{"schema":"eqiora.cartesian-box-envelope/v1","encoding":"eqiora.canonical-json/v1","length_unit":"metre","bounds":[[0.0,2.0],[0.0,3.0]],"entity_sets":[{"name":"bottom","dimension":1,"members":[2]},{"name":"left","dimension":1,"members":[0]},{"name":"right","dimension":1,"members":[1]},{"name":"top","dimension":1,"members":[3]},{"name":"body","dimension":2,"members":[0]}]}"#,
        eqiora_geometry::CanonicalGeometryLimits::default(),
    ).unwrap();
    let source = r#"public component CoupledComponents(
        support body:volume(ambient_dimension=2),
        support left:boundary(parent=body), support right:boundary(parent=body),
        support bottom:boundary(parent=body), support top:boundary(parent=body),
        parameter source_scale:1/m^2
    ) {
        coordinate x:m on body from body[0];
        coordinate y:m on body from body[1];
        parameter q:tensor<1,2,2>=tensor_value(frame=body,components=[[2,3],[5,7]]);
        parameter e0:vector<1,2>=tensor_value(frame=body,components=[1,0]);
        parameter e1:vector<1,2>=tensor_value(frame=body,components=[0,1]);
        variable u:vector<1,2> on body in h1;
        relation balance on body {
            -div(grad(u))+source_scale*contract(q,u,axes=((1,0),))=
                source_scale*((-4+7*x/1[m]-7*y/1[m])*e0+(-9+17*x/1[m]-16*y/1[m])*e1);
        }
        relation lower_x on left { trace(u)=(1+2*coordinate(0)/1[m]+coordinate(1)/1[m])*e0+(-2+coordinate(0)/1[m]-3*coordinate(1)/1[m])*e1; }
        relation upper_x on right { trace(u)=(1+2*coordinate(0)/1[m]+coordinate(1)/1[m])*e0+(-2+coordinate(0)/1[m]-3*coordinate(1)/1[m])*e1; }
        relation lower_y on bottom { trace(u)=(1+2*coordinate(0)/1[m]+coordinate(1)/1[m])*e0+(-2+coordinate(0)/1[m]-3*coordinate(1)/1[m])*e1; }
        relation upper_y on top { trace(u)=(1+2*coordinate(0)/1[m]+coordinate(1)/1[m])*e0+(-2+coordinate(0)/1[m]-3*coordinate(1)/1[m])*e1; }
    }"#;
    let source = if complex {
        source
            .replace("q:tensor<1,2,2>", "q:tensor<complex<1>,2,2>")
            .replace("components=[[2,3],[5,7]]", "components=[[math.complex(2,-1),math.complex(3,2)],[math.complex(5,-4),math.complex(7,3)]]")
            .replace("contract(q,u,axes=((1,0),))", "contract(math.conj(math.complex(math.real(q),math.imag(q))),u,axes=((1,0),))")
            .replace("u:vector<1,2>", "u:vector<complex<1>,2>")
            .replace("-div(grad(u))", "-div(math.complex(2,3)*grad(u))")
            .replace("(-4+7*x/1[m]-7*y/1[m])", "math.complex(-4+14*x/1[m]-11*y/1[m],12+7*x/1[m]+8*y/1[m])")
            .replace("(-9+17*x/1[m]-16*y/1[m])", "math.complex(-14+30*x/1[m]-27*y/1[m],27+21*x/1[m]+16*y/1[m])")
            .replace("(1+2*coordinate(0)/1[m]+coordinate(1)/1[m])*e0+(-2+coordinate(0)/1[m]-3*coordinate(1)/1[m])*e1", "math.complex((1+2*coordinate(0)/1[m]+coordinate(1)/1[m])*e0+(-2+coordinate(0)/1[m]-3*coordinate(1)/1[m])*e1,(2-coordinate(0)/1[m]+2*coordinate(1)/1[m])*e0+(1+3*coordinate(0)/1[m]-coordinate(1)/1[m])*e1)")
    } else {
        source.to_owned()
    };
    let source = if prescribed_fields {
        let source = if complex {
            source
        } else {
            // Re(q) + Im(q) is independently [[2,3],[5,7]]. A real solve
            // must retain both coordinates of the prescribed complex Field.
            source
                .replace("q:tensor<1,2,2>", "q:tensor<complex<1>,2,2>")
                .replace("components=[[2,3],[5,7]]", "components=[[math.complex(1,1),math.complex(5,-2)],[math.complex(1,4),math.complex(10,-3)]]")
                .replace("contract(q,u,axes=((1,0),))", "contract(math.real(q)+math.imag(q),u,axes=((1,0),))")
        };
        let datum = source
            .split_once("trace(u)=")
            .unwrap()
            .1
            .split_once(';')
            .unwrap()
            .0;
        let scalar = if complex { "complex<1>" } else { "1" };
        let fields = format!(
            "variable q:tensor<complex<1>,2,2> on body;
            variable q_middle:tensor<complex<1>,2,2> on body;
            relation q_definition on body {{ q=q_middle; }}
            relation q_middle_definition on body {{ q_middle=q_input; }}
            variable g:vector<{scalar},2> on body in h1;
            relation g_definition on body {{ g={datum}; }}
            variable u:"
        );
        source
            .replace(&format!("trace(u)={datum};"), "trace(u)=trace(g);")
            .replace("parameter q:", "parameter q_input:")
            .replace("variable u:", &fields)
    } else {
        source
    };
    let source = if diffusion {
        source
    } else {
        source
            .replace("-div(grad(u))+", "")
            .replace("-div(math.complex(2,3)*grad(u))+", "")
            .lines()
            .filter(|line| !line.contains("relation lower_") && !line.contains("relation upper_"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let model = scalar_box_model(
        &geometry,
        &source,
        "CoupledComponents",
        &["left", "right", "bottom", "top"],
    );
    for method in [
        CommonSpatialPolicy::Q1,
        CommonSpatialPolicy::CellCenteredTpfa,
    ] {
        let resources = cartesian_box_resources(&geometry, &[2, 3]);
        let NativeMeshResources::Cartesian { mesh, .. } = &resources.resources else {
            unreachable!()
        };
        let mesh = mesh.mesh().clone();
        let plan = ResolvedCommonPlan::resolve(
            &model,
            resources,
            method,
            CommonSolvePolicy::Linear(exact_reference_linear(
                LinearSolver::BiConjugateGradientStabilized,
                // For the algebraic Q1 system A=Mx⊗My⊗Q, the exact inverse
                // infinity norms are 5, 24/5 and at most 10 (real/complex Q).
                // ||r||_2 <= 1e-12 therefore bounds nodal error by 2.4e-10,
                // below the unchanged 1e-9 oracle, independently of load size.
                0.0,
                1e-12,
                NonZeroUsize::new(512).unwrap(),
            )),
            None,
            None,
            &REFERENCE_LINEAR_SOLVER,
            None,
        )
        .unwrap_or_else(|error| panic!("{method:?}, complex={complex}: {error:?}"));
        let plan = replay_plan(plan, &REFERENCE_LINEAR_SOLVER);
        let spatial = plan.as_spatial().unwrap();
        let (field, ty) = spatial.fields().next().unwrap();
        assert_eq!(ty.frame(), eqiora_core::ValueFrame::SpatialCartesian);
        assert_eq!(ty.array_rank(), 0);
        let result = spatial.run_result(None, &REFERENCE_LINEAR_SOLVER).unwrap();
        let expected_type = eqiora_core::ValueType::shaped(
            if complex {
                eqiora_core::ScalarDomain::Complex
            } else {
                eqiora_core::ScalarDomain::Real
            },
            DimExponents::DIMENSIONLESS,
            eqiora_core::ValueShape::new([2]).unwrap(),
            eqiora_core::ValueFrame::SpatialCartesian,
        )
        .unwrap();
        assert_eq!(result.field(0).unwrap().1, &expected_type);
        let (association, values, _) = result.field_block(0, 0).unwrap();
        let nodal = method == CommonSpatialPolicy::Q1;
        assert_eq!(association, if nodal { "vertex" } else { "cell" });
        let entities = spatial.field_coefficient_entities(field).unwrap();
        let width = if complex { 4 } else { 2 };
        assert_eq!(values.len(), entities.len() * width);
        for (entity, values) in entities.iter().zip(values.chunks_exact(width)) {
            let point = if nodal {
                mesh.vertex_coordinates(*entity).unwrap().to_vec()
            } else {
                mesh.geometry_map(*entity).unwrap().origin().to_vec()
            };
            let [x, y]: [f64; 2] = point.try_into().unwrap();
            // Both affine components have zero Laplacian; the load is the
            // independently expanded nonsymmetric matrix-vector product q*u.
            let expected = if complex {
                vec![
                    1. + 2. * x + y,
                    2. - x + 2. * y,
                    -2. + x - 3. * y,
                    1. + 3. * x - y,
                ]
            } else {
                vec![1. + 2. * x + y, -2. + x - 3. * y]
            };
            for (actual, expected) in values.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < 1e-9,
                    "{method:?}, complex={complex}, prescribed={prescribed_fields}, diffusion={diffusion}, x={x}, y={y}: actual={actual}, expected={expected}"
                );
            }
        }
        assert_eq!(
            crate::CommonResult::from_bytes(&result.to_bytes().unwrap(), &plan).unwrap(),
            result
        );
    }
    if complex && !diffusion {
        let conjugated = source.replace(",u,axes=((1,0),))", ",math.conj(u),axes=((1,0),))");
        let model = scalar_box_model(
            &geometry,
            &conjugated,
            "CoupledComponents",
            &["left", "right", "bottom", "top"],
        );
        let resources = cartesian_box_resources(&geometry, &[2, 3]);
        let NativeMeshResources::Cartesian { mesh, .. } = &resources.resources else {
            unreachable!()
        };
        let mesh = mesh.mesh().clone();
        let plan = ResolvedCommonPlan::resolve(
            &model,
            resources,
            CommonSpatialPolicy::Q1,
            CommonSolvePolicy::Linear(exact_reference_linear(
                LinearSolver::BiConjugateGradientStabilized,
                0.,
                1e-12,
                NonZeroUsize::new(512).unwrap(),
            )),
            None,
            None,
            &REFERENCE_LINEAR_SOLVER,
            None,
        )
        .unwrap();
        let plan = replay_plan(plan, &REFERENCE_LINEAR_SOLVER);
        let spatial = plan.as_spatial().unwrap();
        let result = spatial.run_result(None, &REFERENCE_LINEAR_SOLVER).unwrap();
        let (field, _) = spatial.fields().next().unwrap();
        let (_, values, _) = result.field_block(0, 0).unwrap();
        // q*conj(u)=q*v with invertible q gives u=conj(v). Conjugation
        // preserves the inverse norm, so the same 1e-12 absolute residual
        // and independently derived 2.4e-10 coefficient bound apply.
        for (entity, values) in spatial
            .field_coefficient_entities(field)
            .unwrap()
            .iter()
            .zip(values.as_chunks::<4>().0.iter())
        {
            let point = mesh.vertex_coordinates(*entity).unwrap();
            let (x, y) = (point[0], point[1]);
            for (actual, expected) in values.iter().zip([
                1. + 2. * x + y,
                -2. + x - 2. * y,
                -2. + x - 3. * y,
                -1. - 3. * x + y,
            ]) {
                assert!((actual - expected).abs() < 1e-9);
            }
        }
        assert_eq!(
            crate::CommonResult::from_bytes(&result.to_bytes().unwrap(), &plan).unwrap(),
            result
        );
    }
}
