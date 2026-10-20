//! Boundary witnesses retain the complete nonlinear constitutive expression.
use super::*;

#[test]
fn material_component_identity_cannot_drop_or_change_its_validity_domain() {
    for (limit, boundary_response, admitted) in [
        ("4", "witness(value=u)", true),
        ("5", "witness(value=u)", false),
        ("4", "(1+u^2)", false),
    ] {
        let release = |name, limit| {
            format!(
                "property release {name}:Response {{
 analytic {{value=1+value^2;source_unit:1=1;}}
 validity value>=0 and value<={limit};outside reject;branch single;
 citation org.example.manufactured;license spdx.CC0_1_0;
}}"
            )
        };
        let source = format!(
            "property contract Response(input value:1):1 {{derivatives first_partials;}}
{}
{}
component Equation(support body:volume(ambient_dimension=1),
 property response:Response, property witness:Response) {{
 variable u:1 on body in smooth;
 relation balance on body {{response(value=u)=0;}}
 relation comparison on body {{{boundary_response}=0;}}
}}
model Main() {{
 domain body=box(0,1);
 instance physics:Equation(body=body,response=Material,witness=Witness);
}}",
            release("Material", "4"),
            release("Witness", limit)
        );
        let (transaction, model, symbols) = compile("guarded-flux.eqi", &source)
            .unwrap()
            .remove(0)
            .into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        let coefficients = BTreeMap::new();
        let components = crate::canonical::relations_on(&program, symbols.get("body").unwrap())
            .into_iter()
            .map(|relation| {
                let typed = typed_relation(&program, relation).unwrap();
                let context = Context::<f64> {
                    time_s: None,
                    program: &program,
                    dag: typed.expression(),
                    owner: relation,
                    dimension: 1,
                    coefficients: &coefficients,
                };
                context
                    .field_components(typed.expression().roots()[0], 0)
                    .unwrap()
                    .unwrap()
                    .remove(0)
            })
            .collect::<Vec<_>>();
        assert_eq!(components.len(), 2);
        assert_eq!(
            components[0].same_expression(&components[1]),
            admitted,
            "limit={limit}, {boundary_response}"
        );
    }
}

#[test]
fn nonlinear_flux_witness_retains_fields_parameters_and_physical_coordinates() {
    let exact = "(1+coordinate(0)/1[m])*(1+k*u^2)*grad(u)";
    for (flux, homogeneous, admitted) in [
        (exact.to_owned(), false, true),
        (format!("-({exact})"), true, true),
        (format!("-({exact})"), false, false),
        (exact.replace("k*u", "other*u"), false, false),
        (exact.replace("u^2", "v^2"), false, false),
        (
            exact.replace("coordinate(0)", "coordinate(1)"),
            false,
            false,
        ),
        (exact.replace("1+k", "2+k"), false, false),
        (exact.replace("grad(u)", "grad(v)"), false, false),
    ] {
        let source = format!(
            "model MaterialFlux() {{
 domain body=box(0,1,0,1);
 domain wall=boundary(body,axis=1,side=upper);
 parameter k:1=2; parameter other:1=2;
 variable u:1 on body in smooth; variable v:1 on body in smooth;
 relation balance on body {{-div({exact})=0;}}
 relation auxiliary on body {{v=0;}}
 relation boundary on wall {{normal({flux})=0;}}
}}"
        );
        let (transaction, model, symbols) = compile("material-flux.eqi", &source)
            .unwrap()
            .remove(0)
            .into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        let id = |name| symbols.get(name).unwrap();
        let form = CompiledRegionForm::<f64>::derive(&program, id("body"), 2).unwrap();
        let typed = typed_relation(&program, id("boundary")).unwrap();
        let result = form.require_boundary_flux(
            &program,
            id("wall"),
            id("boundary"),
            id("u"),
            typed.expression().roots()[0],
            homogeneous,
        );
        assert_eq!(result.is_ok(), admitted, "{flux}: {result:?}");
    }
}
