//! Nonlinear constitutive traction is balanced once across a conforming interface.
use super::*;

const SOURCE: &str = r#"
model MaterialTransmission() {
  domain left=box(0,1);
  domain right=box(1,2);
  domain lower=boundary(left,axis=0,side=lower);
  domain left_face=boundary(left,axis=0,side=upper);
  domain right_face=boundary(right,axis=0,side=lower);
  domain upper=boundary(right,axis=0,side=upper);
  domain contact=interface(left_face,right_face);
  parameter kl:1=1;
  parameter kr:1=2;
  variable ul:1 on left in smooth;
  variable ur:1 on right in smooth;
  relation left_balance on left {
    -div(kl*(1+ul^2)*grad(ul))+2*kl*(1+coordinate(0)/1[m])/1[m^2]=0;
  }
  relation right_balance on right {
    -div(kr*(1+ur^2)*grad(ur))+kr*(2+(coordinate(0)/1[m]-1)/2)/2[m^2]=0;
  }
  relation left_value on lower {trace(ul)=1;}
  relation right_value on upper {trace(ur)=2.5;}
  relation transmission on contact {
    trace(ul,on=contact)=trace(ur,on=contact);
    normal(kl*(1+ul^2)*grad(ul),on=contact)=normal(kr*(1+ur^2)*grad(ur),on=contact);
  }
}
"#;

#[test]
fn stationary_material_interfaces_preserve_unequal_flux_laws_and_replay() {
    for domain_case in [None, Some("valid"), Some("left"), Some("right")] {
        for reversed in [false, true] {
            let source = if reversed {
                SOURCE.replace(
                    "interface(left_face,right_face)",
                    "interface(right_face,left_face)",
                )
            } else {
                SOURCE.to_owned()
            };
            let source = if let Some(domain_case) = domain_case {
                guarded_source(&source, domain_case)
            } else {
                source
            };
            let (transaction, model, symbols) =
                eqiora_compiler::compile("material-interface.eqi", &source)
                    .unwrap()
                    .remove(0)
                    .into_parts();
            let mut store = InMemoryGraphStore::new();
            store.commit(transaction).unwrap();
            let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
            let model = ModelEnvelope::from_program(&program).unwrap();
            let graph = GeometryGraph::new();
            let interval = graph.interval([0., 2.]).unwrap();
            let [lower, upper]: [_; 2] = interval.boundaries().try_into().unwrap();
            let geometry = graph
                .build(
                    &interval,
                    &BTreeMap::from([
                        ("body".to_owned(), vec![interval.region().into()]),
                        ("lower".to_owned(), vec![lower.into()]),
                        ("upper".to_owned(), vec![upper.into()]),
                    ]),
                )
                .unwrap();
            let plan = ResolvedCommonPlan::resolve(
                &model,
                cartesian_box_resources(&geometry, &[4]),
                CommonSpatialPolicy::Q1,
                CommonSolvePolicy::Newton {
                    linear: exact_reference_linear(
                        LinearSolver::BiConjugateGradientStabilized,
                        1e-13,
                        1e-14,
                        NonZeroUsize::new(100).unwrap(),
                    ),
                    nonlinear: NonlinearSolvePlan::new(
                        0.,
                        1e-11,
                        NonZeroUsize::new(16).unwrap(),
                        16,
                    )
                    .unwrap(),
                },
                None,
                None,
                &REFERENCE_LINEAR_SOLVER,
                None,
                None,
            )
            .unwrap();
            let plan = replay_plan(plan, &REFERENCE_LINEAR_SOLVER);
            let spatial = plan.as_spatial().unwrap();
            let assignments = spatial
                .fields()
                .map(|(field, ty)| {
                    let count = spatial.field_coefficient_entities(field).unwrap().len();
                    assert_eq!(count, 3);
                    // Preserve the prescribed endpoints and common trace; perturb
                    // only interior values away from the independent exact solution.
                    let values = if field.ulid() == symbols.get("ul").unwrap().ulid() {
                        if matches!(domain_case, Some("left" | "right")) {
                            vec![1., 1.5, 2.]
                        } else {
                            vec![1., 1.75, 2.]
                        }
                    } else if matches!(domain_case, Some("left" | "right")) {
                        vec![2., 2.25, 2.5]
                    } else {
                        vec![2., 2., 2.5]
                    };
                    CommonInitialField::new(
                        model.digest().unwrap(),
                        field,
                        Some(CommonInitialValues::new(ty.shape().clone(), values).unwrap()),
                        None,
                    )
                    .unwrap()
                })
                .collect();
            let initial = spatial.initial_state(None, assignments).unwrap();
            let before = initial.to_bytes().unwrap();
            let result = spatial.run_result(Some(&initial), &REFERENCE_LINEAR_SOLVER);
            assert_eq!(initial.to_bytes().unwrap(), before);
            if matches!(domain_case, Some("left" | "right")) {
                // Both volume residuals vanish at this exact piecewise-affine seed.
                // Last left volume sample: 7/4+1/(4*sqrt(3))<1.99.
                // First right volume sample: 17/8-1/(8*sqrt(3))>2.01.
                // Only the interface value 2 violates the selected side's domain.
                assert_eq!(
                    result.err().unwrap().code(),
                    eqiora_core::diagnostic::codes::EXPRESSION_DOMAIN_VIOLATION
                );
                continue;
            }
            let result = result.unwrap();
            assert!(result.nonlinear_iterations(None).unwrap() > 1);
            // Independently, ul=1+x and ur=2+(x-1)/2 give continuous value 2
            // and +x flux 5 on both sides at x=1, despite unequal coefficients.
            // Their q' are 2*kl*ul and kr*ur/2, exactly the authored volume loads.
            for (name, expected) in [("ul", [1., 1.5, 2.]), ("ur", [2., 2.25, 2.5])] {
                let id = symbols.get(name).unwrap().ulid().to_string();
                let index = (0..result.field_count())
                    .find(|&index| result.field(index).unwrap().0 == id)
                    .unwrap();
                let (association, values, _) = result.field_block(index, 0).unwrap();
                assert_eq!(association, "vertex");
                assert_eq!(values.len(), expected.len());
                for (actual, exact) in values.iter().zip(expected) {
                    assert!(
                        (actual - exact).abs() < 1e-10,
                        "{name}: {actual} != {exact}"
                    );
                }
                let mut changed = values.to_vec();
                changed[1] += 0.25;
                assert!(
                    crate::CommonResult::from_bytes(
                        &result_replay::reseal_values(&result, index, &changed),
                        &plan
                    )
                    .is_err()
                );
            }
            assert_eq!(
                crate::CommonResult::from_bytes(&result.to_bytes().unwrap(), &plan).unwrap(),
                result
            );
        }
    }
}

fn guarded_source(source: &str, domain_case: &str) -> String {
    let left_upper = if domain_case == "left" { "1.99" } else { "4" };
    let right_lower = if domain_case == "right" { "2.01" } else { "0" };
    let connection = if source.contains("interface(right_face,left_face)") {
        "connect right_physics.edge,left_physics.edge;"
    } else {
        "connect left_physics.edge,right_physics.edge;"
    };
    format!(
        "property contract Response(input value:1):1 {{derivatives first_partials;}}
property release LeftMaterial:Response {{
 analytic {{value=1+value^2;source_unit:1=1;}}
 validity value>=0 and value<={left_upper};outside reject;branch single;
 citation org.example.manufactured;license spdx.CC0_1_0;
}}
property release RightMaterial:Response {{
 analytic {{value=1+value^2;source_unit:1=1;}}
 validity value>={right_lower} and value<=4;outside reject;branch single;
 citation org.example.manufactured;license spdx.CC0_1_0;
}}
connector ScalarBoundary {{
 trace value:1;flux outward_flux:1/m;shape [];frame invariant;
 pairing euclidean_boundary_duality;orientation parent_outward;
}}
component MaterialSide(
 support body:volume(ambient_dimension=1),support face:boundary(parent=body),
 variable u:1 on body in smooth,property response:Response,
 parameter k:1,parameter slope:1,parameter offset:1,
 port edge:ScalarBoundary over face
) {{
 relation balance on body {{
  -div(k*response(value=u)*grad(u))
    +2*k*slope^2*(offset+slope*coordinate(0)/1[m])/1[m^2]=0;
 }}
 relation carrier on face {{
  trace(u)=edge.value;
  normal(k*response(value=u)*grad(u))=edge.outward_flux;
 }}
}}
model MaterialTransmission() {{
 domain left=box(0,1);domain right=box(1,2);
 domain lower=boundary(left,axis=0,side=lower);
 domain left_face=boundary(left,axis=0,side=upper);
 domain right_face=boundary(right,axis=0,side=lower);
 domain upper=boundary(right,axis=0,side=upper);
 variable ul:1 on left in smooth;variable ur:1 on right in smooth;
 instance left_physics:MaterialSide(body=left,face=left_face,u=ul,
  response=LeftMaterial,k=1,slope=1,offset=1);
 instance right_physics:MaterialSide(body=right,face=right_face,u=ur,
  response=RightMaterial,k=2,slope=0.5,offset=1.5);
 relation left_value on lower {{trace(ul)=1;}}
 relation right_value on upper {{trace(ur)=2.5;}}
 {connection}
}}"
    )
}
