//! Full tensor flux reaches ordinary plans, physical traces and retained Fields.
use super::*;

#[test]
fn anisotropic_vector_flux_preserves_natural_work_phase_and_complete_results() {
    let geometry = CanonicalGeometryV1::decode_cartesian_box_v1_canonical(
        br#"{"schema":"eqiora.cartesian-box-envelope/v1","encoding":"eqiora.canonical-json/v1","length_unit":"metre","bounds":[[0.0,2.0],[0.0,3.0]],"entity_sets":[{"name":"bottom","dimension":1,"members":[2]},{"name":"left","dimension":1,"members":[0]},{"name":"right","dimension":1,"members":[1]},{"name":"top","dimension":1,"members":[3]},{"name":"body","dimension":2,"members":[0]}]}"#,
        eqiora_geometry::CanonicalGeometryLimits::default()).unwrap();
    let real = "(1+2*coordinate(0)/1[m]+coordinate(1)/1[m])*e0+(-2+coordinate(0)/1[m]-3*coordinate(1)/1[m])*e1";
    let imaginary = "(2-coordinate(0)/1[m]+2*coordinate(1)/1[m])*e0+(1+3*coordinate(0)/1[m]-coordinate(1)/1[m])*e1";
    for complex in [false, true] {
        let scalar = if complex { "complex<1>" } else { "1" };
        let rate = if complex { "complex<1/m>" } else { "1/m" };
        let volume_unit = if complex { "complex<m^2>" } else { "m^2" };
        let gradient_unit = if complex { "complex<m>" } else { "m" };
        let phase = if complex { "math.complex(2,3)" } else { "1" };
        let trace = if complex {
            format!("math.complex({real},{imaginary})")
        } else {
            real.to_owned()
        };
        // K acts on [u0,x,u0,y,u1,x,u1,y]. For the exact affine Field,
        // the two outward-right flux components are (17+i, 9+22i)/m;
        // multiplication by 2+3i gives (31+53i,-48+71i)/m.
        let natural = if complex {
            "(math.complex(31,53)*e0+math.complex(-48,71)*e1)/1[m]"
        } else {
            "(17*e0+9*e1)/1[m]"
        };
        let flux = format!("{phase}*contract(k,grad(u),axes=((2,0),(3,1)))");
        let source = format!("public component Anisotropic(
            support body:volume(ambient_dimension=2),
            support left:boundary(parent=body), support right:boundary(parent=body),
            support bottom:boundary(parent=body), support top:boundary(parent=body)
        ) {{
            parameter k:tensor<1,2,2,2,2>=tensor_value(frame=body,components=[[[[7,1],[2,0]],[[-1,8],[0,1]]],[[[3,0],[9,2]],[[0,-2],[1,10]]]]);
            parameter e0:vector<1,2>=tensor_value(frame=body,components=[1,0]);
            parameter e1:vector<1,2>=tensor_value(frame=body,components=[0,1]);
            variable u:vector<{scalar},2> on body in smooth;
            relation balance on body {{-div({flux})=0;}}
            relation left_value on left {{trace(u)={trace};}}
            relation right_flux on right {{normal({flux})={natural};}}
            relation bottom_value on bottom {{trace(u)={trace};}}
            relation top_value on top {{trace(u)={trace};}}
            coordinate x:m on body from body[0];
            coordinate y:m on body from body[1];
            observable profile:vector<{scalar},2> on body=u;
            observable dx:vector<{rate},2> on body=partial(u,wrt=x);
            observable dy:vector<{rate},2> on body=partial(u,wrt=y);
            observable integrated:vector<{volume_unit},2>=integral(u,measure(body));
            observable integrated_gradient:tensor<{gradient_unit},2,2>=integral(grad(u),measure(body));
            observable integrated_flux:vector<{scalar},2>=integral(normal({flux}),measure(right));
        }}");
        let body = geometry.entity_set("body").unwrap();
        let mut supports = vec![("body", body, None)];
        supports.extend(["left", "right", "bottom", "top"].into_iter().map(|name| {
            (
                name,
                geometry.entity_set(name).unwrap(),
                Some(("body", body)),
            )
        }));
        let model = compile_model(
            "anisotropic.eqi",
            &source,
            &geometry,
            "Anisotropic",
            &supports,
            &[],
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
            None,
        )
        .unwrap();
        let plan = replay_plan(plan, &REFERENCE_LINEAR_SOLVER);
        let spatial = plan.as_spatial().unwrap();
        let (field, ty) = spatial.fields().next().unwrap();
        assert_eq!(ty.frame(), eqiora_core::ValueFrame::SpatialCartesian);
        assert_eq!(ty.array_rank(), 0);
        let result = spatial.run_result(None, &REFERENCE_LINEAR_SOLVER).unwrap();
        super::observations::check_vector(&model, &plan, &result, complex);
        super::integrals::check_vector(&model, &plan, &result, complex);
        assert_eq!(
            result.field(0).map(|(_, ty, _)| ty.scalar_domain()),
            Some(ty.scalar_domain())
        );
        let (association, values, _) = result.field_block(0, 0).unwrap();
        assert_eq!(association, "vertex");
        let entities = spatial.field_coefficient_entities(field).unwrap();
        let width = if complex { 4 } else { 2 };
        assert_eq!(values.len(), entities.len() * width);
        for (entity, actual) in entities.iter().zip(values.chunks_exact(width)) {
            let point = mesh.vertex_coordinates(*entity).unwrap();
            let (x, y) = (point[0], point[1]);
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
            for (a, b) in actual.iter().zip(expected) {
                assert!((a - b).abs() < 1e-10, "complex={complex}: {a} != {b}");
            }
        }
        assert_eq!(
            crate::CommonResult::from_bytes(&result.to_bytes().unwrap(), &plan).unwrap(),
            result
        );
    }
}
