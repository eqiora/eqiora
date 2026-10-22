use super::*;

const SOURCE: &str = r#"
public model Throughflow(
    support body:volume(ambient_dimension=2),
    support left:boundary(parent=body),
    support right:boundary(parent=body),
    support bottom:boundary(parent=body),
    support top:boundary(parent=body)
) {
    variable u:vector<m/s,2> on body in smooth;
    variable g:m^2/s on body in smooth;
    variable f:kg/(m*s^2) on body in smooth;
    relation prescribed_velocity on body {g=1[m/s]*coordinate(0)+2[m/s]*coordinate(1);}
    relation force on body {f=0[kg/(m*s^2)];}
    relation momentum on body {-div(2[kg/(m*s)]*symmetric_part(grad(u)))-grad(f)=0;}
    relation inlet on left {trace(u)=trace(grad(g));}
    relation lower on bottom {trace(u)=trace(grad(g));}
    relation upper on top {trace(u)=trace(grad(g));}
    relation outlet on right {normal(2[kg/(m*s)]*symmetric_part(grad(u)))=0;}
    observable profile:vector<m/s,2> on body=u;
}
"#;

#[test]
fn planar_vector_traces_preserve_every_component() {
    let source = SOURCE;
    let geometry = geometry(true);
    let body = geometry.entity_set("body").unwrap();
    let mut supports = vec![("body", body, None)];
    supports.extend(["left", "right", "bottom", "top"].map(|name| {
        (
            name,
            geometry.entity_set(name).unwrap(),
            Some(("body", body)),
        )
    }));
    for source in [
        source.to_owned(),
        source.replace("Throughflow", "RenamedFlow"),
    ] {
        let entry = if source.contains("model RenamedFlow") {
            "RenamedFlow"
        } else {
            "Throughflow"
        };
        let model = compile_model(
            "common-planar-vector.eqi",
            &source,
            &geometry,
            entry,
            &supports,
            &[],
        );
        for permuted in [false, true] {
            let resolved = ResolvedCommonPlan::resolve(
                &model,
                open_resources(&geometry, permuted),
                CommonSpatialPolicy::P1,
                CommonSolvePolicy::Linear(exact_reference_linear(
                    LinearSolver::BiConjugateGradientStabilized,
                    1e-11,
                    1e-13,
                    NonZeroUsize::new(1000).unwrap(),
                )),
                None,
                None,
                &REFERENCE_LINEAR_SOLVER,
                None,
            )
            .unwrap();
            let resolved = replay_plan(resolved, &REFERENCE_LINEAR_SOLVER);
            let plan = resolved
                .as_spatial()
                .expect("vector P1 must use the shared linear owner");
            let result = plan.run(None, &REFERENCE_LINEAR_SOLVER).unwrap();
            let published = plan.run_result(None, &REFERENCE_LINEAR_SOLVER).unwrap();
            let (association, published_values, shape) = published.field_block(0, 0).unwrap();
            assert_eq!(association, "vertex");
            assert_eq!(shape, &[6, 2]);
            assert_eq!(published_values, result.fields[0].2);
            let observable = plan
                .kernel()
                .nodes()
                .find_map(|node| match node {
                    eqiora_schema::kernel::KernelNode::Observable(definition) => {
                        Some(definition.id())
                    }
                    _ => None,
                })
                .unwrap();
            let length = eqiora_core::DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
            let point = [0.375, 0.25].map(|value| DynQuantity::new(value, length));
            let sample = published
                .observe_at(
                    &model,
                    observable,
                    &point,
                    &std::collections::HashMap::new(),
                    None,
                )
                .unwrap();
            let components = sample.value().components().unwrap().collect::<Vec<_>>();
            assert_eq!(components.len(), 2);
            for ((real, imaginary), expected) in components.iter().zip([1.0, 2.0]) {
                assert!((real - expected).abs() < 1e-10);
                assert_eq!(*imaginary, 0.0);
            }
            let bytes = published.to_bytes().unwrap();
            assert_eq!(
                crate::CommonResult::from_bytes(&bytes, &resolved)
                    .unwrap()
                    .to_bytes()
                    .unwrap(),
                bytes
            );
            assert_eq!(result.fields.len(), 1);
            // u=(1,2) has zero symmetric gradient, so both the volume residual
            // and the right-edge traction vanish. P1 reproduces this exact
            // field at all six vertices, including the free outlet midpoint.
            let (_, value_type, values, space) = &result.fields[0];
            assert!(!value_type.shape().is_scalar());
            assert_eq!(
                *space,
                Space::continuous_lagrange(std::num::NonZeroU16::MIN)
            );
            assert_eq!(values.len(), 12);
            let field = plan.fields().next().unwrap().0;
            let scaling = eqiora_realization::SystemScaling::SymmetricCongruence(
                eqiora_realization::SymmetricCongruenceScaling::new(
                    [eqiora_realization::AlgebraicBlockScale::new(
                        eqiora_solver::AlgebraicBlock::Field(field),
                        eqiora_realization::PositivePhysicalScale::new(DynQuantity::new(
                            2.5,
                            value_type.dimension(),
                        ))
                        .unwrap(),
                    )],
                    eqiora_realization::PositivePhysicalScale::new(DynQuantity::new(
                        7.0,
                        eqiora_core::DimExponents::from_integers([1, 1, -3, 0, 0, 0, 0]).unwrap(),
                    ))
                    .unwrap(),
                )
                .unwrap(),
            );
            let eqiora_realization::SystemScaling::SymmetricCongruence(scales) = &scaling else {
                unreachable!()
            };
            let mut wrong_units = scales.block_scales().to_vec();
            wrong_units[0] = eqiora_realization::AlgebraicBlockScale::new(
                eqiora_solver::AlgebraicBlock::Field(field),
                eqiora_realization::PositivePhysicalScale::new(DynQuantity::new(
                    2.5,
                    eqiora_core::DimExponents::DIMENSIONLESS,
                ))
                .unwrap(),
            );
            let mut extra_block = scales.block_scales().to_vec();
            extra_block.push(eqiora_realization::AlgebraicBlockScale::new(
                eqiora_solver::AlgebraicBlock::ConstraintMultiplier { field },
                scales.block_scales()[0].scale(),
            ));
            for (blocks, expected) in [
                (Vec::new(), "no exact congruence scale"),
                (wrong_units, "exact Space coefficient units"),
                (extra_block, "exact equal coverage"),
            ] {
                let invalid_scales = eqiora_realization::SymmetricCongruenceScaling::new(
                    blocks,
                    scales.weak_functional_scale(),
                )
                .unwrap();
                let error = ResolvedCommonPlan::resolve(
                    &model,
                    open_resources(&geometry, permuted),
                    CommonSpatialPolicy::P1,
                    resolved.canonical_solve_request().unwrap(),
                    Some(CommonScalingRequest::SymmetricCongruence(invalid_scales)),
                    None,
                    &REFERENCE_LINEAR_SOLVER,
                    None,
                )
                .unwrap_err();
                assert!(error.message().contains(expected), "{error:?}");
            }
            let request = CommonScalingRequest::SymmetricCongruence(scales.clone());
            let scaled = ResolvedCommonPlan::resolve(
                &model,
                open_resources(&geometry, permuted),
                CommonSpatialPolicy::P1,
                resolved.canonical_solve_request().unwrap(),
                Some(request.clone()),
                None,
                &REFERENCE_LINEAR_SOLVER,
                None,
            )
            .unwrap();
            assert_eq!(scaled.canonical_scaling_request(), Some(request.clone()));
            let scaled = replay_plan(scaled, &REFERENCE_LINEAR_SOLVER);
            assert_eq!(scaled.canonical_scaling_request(), Some(request));
            let scaled = scaled.as_spatial().unwrap();
            scaled.reauthenticate_portable_realization().unwrap();
            assert_eq!(
                scaled.portable_realization().systems()[0].scaling(),
                &scaling
            );
            assert_ne!(scaled.identity(), plan.identity());
            let scaled_result = scaled.run(None, &REFERENCE_LINEAR_SOLVER).unwrap();
            for value in scaled_result.fields[0].2.as_chunks::<2>().0 {
                assert!((value[0] - 1.0).abs() < 1e-10);
                assert!((value[1] - 2.0).abs() < 1e-10);
            }
            for value in values.as_chunks::<2>().0 {
                assert!((value[0] - 1.).abs() < 1e-10);
                assert!((value[1] - 2.).abs() < 1e-10);
            }
        }
    }
}

// A free outlet midpoint exercises natural boundaries as well as the
// constrained corners and the unconstrained interior vertex.
pub(super) fn open_resources(
    geometry: &CanonicalGeometryV1,
    permuted: bool,
) -> AuthenticatedCommonMesh {
    let mut msh = String::from(
        "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n1 6 1 6\n2 1 0 6\n1\n2\n3\n4\n5\n6\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n0.5 0.5 0\n1 0.5 0\n$EndNodes\n$Elements\n1 5 1 5\n2 1 2 5\n",
    );
    for (index, mut cell) in [[1, 2, 5], [2, 6, 5], [6, 3, 5], [3, 4, 5], [4, 1, 5]]
        .into_iter()
        .enumerate()
    {
        if permuted {
            cell.rotate_left(1);
        }
        msh += &format!("{} {} {} {}\n", index + 1, cell[0], cell[1], cell[2]);
    }
    msh += "$EndElements\n";
    AuthenticatedCommonMesh::gmsh_4152(
        geometry.clone(),
        eqiora_artifact::GmshMeshPolicyV1::explicit(1e-12, 0.01, 8, 5.).unwrap(),
        msh.into_bytes(),
    )
    .unwrap()
}

#[test]
fn fieldwise_simplicial_assembly_preserves_each_selected_basis() {
    use eqiora_realization::{DomainFieldDiscretization, FieldSpaceBinding};
    let source = SOURCE.replace("variable u:vector", "variable z:m/s on body in h1; relation scalar_balance on body {-div(grad(z))=1[1/(m*s)];} relation scalar_left on left {trace(z)=0;} relation scalar_right on right {trace(z)=0;} relation scalar_bottom on bottom {trace(z)=0;} relation scalar_top on top {trace(z)=0;} variable u:vector");
    let geometry = geometry(true);
    let body = geometry.entity_set("body").unwrap();
    let mut supports = vec![("body", body, None)];
    supports.extend(["left", "right", "bottom", "top"].map(|name| {
        (
            name,
            geometry.entity_set(name).unwrap(),
            Some(("body", body)),
        )
    }));
    let model = compile_model(
        "fieldwise-spaces.eqi",
        &source,
        &geometry,
        "Throughflow",
        &supports,
        &[],
    );
    let solver = exact_reference_linear(
        LinearSolver::BiConjugateGradientStabilized,
        1e-11,
        1e-13,
        NonZeroUsize::new(1000).unwrap(),
    );
    for permuted in [false, true] {
        let resolved = ResolvedCommonPlan::resolve(
            &model,
            open_resources(&geometry, permuted),
            CommonSpatialPolicy::P1,
            CommonSolvePolicy::Linear(solver),
            None,
            None,
            &REFERENCE_LINEAR_SOLVER,
            None,
        )
        .unwrap();
        let admission = &resolved.as_spatial().unwrap().admission;
        let RecognizedNativeModel::Spatial(equations) = admission.recognized_model() else {
            panic!("linear equations");
        };
        let NativeMeshResources::GmshSimplicial { mesh, .. } = admission.resources() else {
            panic!("simplicial mesh");
        };
        let region = &equations.regions[0];
        let bindings = region
            .form
            .represented_fields()
            .iter()
            .map(|(field, ty)| {
                FieldSpaceBinding::new(
                    field.downcast().unwrap(),
                    if ty.shape().is_scalar() {
                        Space::simplex_p1_bubble()
                    } else {
                        Space::continuous_lagrange(std::num::NonZeroU16::MIN)
                    },
                )
            })
            .collect::<Vec<_>>();
        let selected = vec![
            DomainFieldDiscretization::new(region.domain_id(), bindings.clone(), None).unwrap(),
        ];
        let reference = eqiora_meshing::ReferenceCell::simplex(2).unwrap();
        assert!(
            equations
                .bind_spaces(
                    reference,
                    &[],
                    &eqiora_realization::SystemScaling::Dimensional
                )
                .is_err()
        );
        assert!(
            equations
                .bind_spaces(
                    reference,
                    &[selected[0].clone(), selected[0].clone()],
                    &eqiora_realization::SystemScaling::Dimensional
                )
                .is_err()
        );
        assert!(
            region
                .form
                .bind_field_spaces(
                    reference,
                    &bindings[..1],
                    &eqiora_realization::SystemScaling::Dimensional
                )
                .is_err()
        );
        assert!(
            region
                .form
                .bind_field_spaces(
                    reference,
                    &[bindings[0], bindings[0]],
                    &eqiora_realization::SystemScaling::Dimensional
                )
                .is_err()
        );
        let (mapping, forms, natural) = equations
            .simplicial_assembly(
                mesh,
                &selected,
                &eqiora_realization::SystemScaling::Dimensional,
            )
            .unwrap();
        let output = mapping
            .solve(
                mesh.mesh(),
                crate::region_assembly::mapping::RegionSolveInput {
                    conservative: None,
                    operator_properties: LinearOperatorProperties::General,
                    geometry_action: None,
                    forms,
                    natural,
                    previous: None,
                    prescribed_states: BTreeMap::new(),
                },
                NonZeroUsize::MIN,
                LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, admission.linear.solver),
            )
            .unwrap();
        // For -Delta z=1 with zero trace, the interior P1 row is
        // 4*z_center=1/3. Bubble gradients are orthogonal to affine gradients
        // because the bubble vanishes on every facet. With b=27*l0*l1*l2,
        // integral(b)=9*A/20 and integral(|grad b|^2)=81*A*sum(|grad li|^2)/20.
        // Thus bubble coefficients are 1/(9*sum(|grad li|^2)): 1/72 in
        // the three larger cells and 1/144 in the two half-size right cells.
        let bubbles = [1.0 / 72.0, 1.0 / 144.0, 1.0 / 144.0, 1.0 / 72.0, 1.0 / 72.0];
        assert_eq!(output.fields.len(), 2);
        for recovered in output.fields.values() {
            let scalar = recovered.value_type.shape().is_scalar();
            assert_eq!(
                recovered.space,
                if scalar {
                    Space::simplex_p1_bubble()
                } else {
                    Space::continuous_lagrange(std::num::NonZeroU16::MIN)
                }
            );
            assert_eq!(recovered.coefficients.len(), if scalar { 11 } else { 12 });
            for (key, value) in &recovered.coefficients {
                let expected = if scalar {
                    if key.entity.dimension() == 0 {
                        if key.entity.index() == 4 {
                            1.0 / 12.0
                        } else {
                            0.0
                        }
                    } else {
                        bubbles[key.entity.index()]
                    }
                } else {
                    (key.component + 1) as f64
                };
                assert!((value - expected).abs() < 1e-10);
            }
        }
    }
}
