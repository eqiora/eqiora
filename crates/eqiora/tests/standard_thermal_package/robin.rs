use super::*;

// Fixed protocol inputs (L, H, k, h, Tc, Tinf). The external reference owns
// operator, solution and reaction expectations and their predeclared budgets.
const CASES: [(u32, u32, u32, u32, u32, u32); 4] = [
    (2, 1, 3, 0, 280, 320),
    (2, 1, 3, 2, 280, 320),
    (2, 1, 3, 5, 280, 320),
    (3, 2, 2, 4, 310, 280),
];

fn source(k: u32, h: u32, fixed: u32, ambient: u32) -> String {
    format!(
        "{IMPORT}{SIGNATURE}
 variable temperature: K on body in smooth;
 instance conduction: scalar.SteadyHeat2d(body=body, temperature=temperature,
   conductivity={k}[W/(m*K)], heating=0[W/m^3]);
 instance fixed: scalar.FixedTemperature2d(body=body, face=left,
   temperature=temperature, value={fixed}[K]);
 instance convection: scalar.ConvectiveHeatTransfer2d(body=body, face=right,
   temperature=temperature, conductivity={k}[W/(m*K)],
   coefficient={h}[W/(m^2*K)], ambient={ambient}[K]);
 instance bottom_flux: scalar.OutwardHeatFlux2d(body=body, face=bottom,
   temperature=temperature, conductivity={k}[W/(m*K)], flux=0[W/m^2]);
 instance top_flux: scalar.OutwardHeatFlux2d(body=body, face=top,
   temperature=temperature, conductivity={k}[W/(m*K)], flux=0[W/m^2]);
 }}"
    )
}

#[test]
fn imported_convection_compiles_with_exact_caller_supports_and_required_inputs() {
    for (length, height, k, h, fixed, ambient) in CASES {
        let (document, _) = fixture_rectangle(
            &source(k, h, fixed, ambient),
            [f64::from(length), f64::from(height)],
        )
        .unwrap();
        assert!(matches!(document, SourceCompilation::Packaged(_)));
    }
}

#[test]
fn convection_rejects_wrong_units_regularity_parent_and_missing_inputs() {
    let good = source(3, 2, 280, 320);
    for (invalid, message) in [
        (
            good.replace("coefficient=2[W/(m^2*K)]", "coefficient=2[W/(m*K)]"),
            "Parameter binding has dimension",
        ),
        (
            good.replace("ambient=320[K]", "ambient=320"),
            "expression slot `ambient` requires its declared value type and exact bound support",
        ),
        (
            good.replace("in smooth", "in h1"),
            "disagree in authored spatial regularity",
        ),
        (
            good.replace("coefficient=2[W/(m^2*K)],", ""),
            "required Parameter `coefficient` has no instance binding",
        ),
        (
            good.replace("ambient=320[K]", ""),
            "no binding for required expression slot `ambient`",
        ),
        (
            good.replace("conductivity=3[W/(m*K)]", "conductivity=3"),
            "Parameter binding has dimension",
        ),
    ] {
        let error = fixture(&invalid).expect_err("invalid convection binding must reject");
        let diagnostics = error.diagnostics().expect("typed source diagnostics");
        assert!(
            diagnostics.iter().any(|diagnostic| {
                format!("{:?}", diagnostic.code()) == "Code(\"EQ0603\")"
                    && diagnostic.message().contains(message)
            }),
            "expected semantic diagnostic {message}: {error:?}"
        );
    }
}

#[test]
fn convection_rejects_boundary_from_another_parent() {
    let source = source(3, 2, 280, 320)
        .replace(
            SIGNATURE,
            r#"
model Main() {
 domain body = box(0,2,0,1);
 domain left = boundary(body,axis=0,side=lower);
 domain right = boundary(body,axis=0,side=upper);
 domain bottom = boundary(body,axis=1,side=lower);
 domain top = boundary(body,axis=1,side=upper);
 domain foreign = box(0,2,0,1);
 domain other = boundary(foreign,axis=0,side=upper);
"#,
        )
        .replace("face=right", "face=other");
    let error = compile_source(
        "foreign.eqi",
        &source,
        Some("Main"),
        &[],
        SourceImportContext::Bundled,
    )
    .expect_err("foreign face must reject");
    assert!(
        error
            .diagnostics()
            .expect("typed support diagnostics")
            .iter()
            .any(|diagnostic| {
                format!("{:?}", diagnostic.code()) == "Code(\"EQ0603\")"
                    && diagnostic
                        .message()
                        .contains("not BoundaryOf its exact bound parent")
            }),
        "expected exact-parent diagnostic: {error:?}"
    );
}
