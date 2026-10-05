use std::collections::HashMap;

use eqiora::api::ModelDocument;
use eqiora_ir::{ComponentScalarization, ScalarSymbolCoordinate};
use eqiora_schema::kernel::{KernelNode, SymbolRef};

struct SourcePencil {
    rows: ComponentScalarization,
    mode: Vec<ScalarSymbolCoordinate>,
    spectral: ScalarSymbolCoordinate,
    bindings: Vec<(ScalarSymbolCoordinate, f64)>,
}

impl SourcePencil {
    fn compile(source: &str) -> Self {
        let model = ModelDocument::compile("pencil.eqi", source).unwrap();
        let program = model.program();
        let coordinates = |id| {
            let Some(KernelNode::Field(field)) = program.node(id) else {
                panic!("source Field");
            };
            ScalarSymbolCoordinate::for_value(SymbolRef::Field(field.id()), field.value_type())
                .unwrap()
        };
        let mode = coordinates(model.aliases()["u"]);
        let spectral = coordinates(model.aliases()["lambda"]).remove(0);
        let relation = model.aliases()["r"].downcast().unwrap();
        let rows =
            ComponentScalarization::lower(&program.typed_relation_residual(relation).unwrap())
                .unwrap();
        let bindings = rows
            .rows()
            .iter()
            .flat_map(|row| row.symbols())
            .filter_map(|coordinate| {
                let SymbolRef::Parameter(id) = coordinate.symbol() else {
                    return None;
                };
                let value = program.typed_value(id.erase()).unwrap();
                let index = coordinate
                    .component_index()
                    .iter()
                    .zip(value.value_type().shape().extents())
                    .fold(0, |flat, (&index, extent)| {
                        flat * extent.get() as usize + index as usize
                    });
                let (real, imaginary) = value.component(index).unwrap();
                Some((
                    coordinate.clone(),
                    if coordinate.is_imaginary() {
                        imaginary
                    } else {
                        real
                    },
                ))
            })
            .collect::<HashMap<_, _>>()
            .into_iter()
            .collect();
        Self {
            rows,
            mode,
            spectral,
            bindings,
        }
    }

    fn scalar(expression: &str) -> Self {
        Self::compile(&format!(
            "model M(){{parameter k:1=2;variable u:1;variable lambda:1;relation r{{{expression}=0;}}}}"
        ))
    }
}

#[test]
fn pencil_coefficients_follow_algebra_without_operator_subtraction() {
    for (expression, expected_a, expected_c) in [
        ("(1e20-lambda)*u", 1e20, -1.),
        ("(k*u-lambda*u)/k", 1., -0.5),
        ("u*(k-lambda)", 2., -1.),
        ("math.sin(k)*u-lambda*u", 2_f64.sin(), -1.),
    ] {
        let source = SourcePencil::scalar(expression);
        let (a, c) = source.rows.rows()[0]
            .bind_affine_pencil(&source.mode, &source.spectral, &source.bindings)
            .unwrap();
        assert_eq!(a.coefficients(), [expected_a]);
        assert_eq!(c.coefficients(), [expected_c]);
        assert_eq!(a.offsets(), [0.]);
        assert_eq!(c.offsets(), [0.]);
    }
}

#[test]
fn pencil_admission_rejects_false_sampling_linearity_and_inhomogeneous_terms() {
    for expression in [
        "lambda*(lambda-1)*u", // Both lambda=0 and lambda=1 hide the quadratic term.
        "lambda*u*u",          // Binding lambda=0 alone hides nonlinear mode dependence.
        "u/(1+lambda)",
        "math.sin(lambda)*u",
        "u+lambda",
        "u+1",
    ] {
        let source = SourcePencil::scalar(expression);
        assert!(
            source.rows.rows()[0]
                .bind_affine_pencil(&source.mode, &source.spectral, &source.bindings)
                .is_err(),
            "{expression}"
        );
    }
}

#[test]
fn six_complex_mode_components_keep_the_original_coordinate_correspondence() {
    let matrix = (0..6)
        .map(|row| {
            format!(
                "[{}]",
                (0..6)
                    .map(|col| if row == col {
                        (row + 2).to_string()
                    } else {
                        "0".to_string()
                    })
                    .collect::<Vec<_>>()
                    .join(",")
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let source = SourcePencil::compile(&format!(
        r#"
space Modes=orthonormal(a,b,c,d,e,f);
model M() {{
 parameter a:map<1,Modes,Modes>=linear_map(Modes,Modes,[{matrix}]);
 variable u:coordinates<complex<1>,Modes>;
 variable lambda:1;
 relation r{{apply(a,u)-lambda*u=0;}}
}}
"#
    ));
    assert_eq!(source.mode.len(), 12);
    assert_eq!(source.rows.rows().len(), 12);
    for (row_index, row) in source.rows.rows().iter().enumerate() {
        let (a, c) = row
            .bind_affine_pencil(&source.mode, &source.spectral, &source.bindings)
            .unwrap();
        assert_eq!(a.selected_symbols(), source.mode);
        for col in 0..12 {
            assert_eq!(
                a.coefficients()[col],
                if col == row_index {
                    (row_index / 2 + 2) as f64
                } else {
                    0.
                }
            );
            assert_eq!(
                c.coefficients()[col],
                if col == row_index { -1. } else { 0. }
            );
        }
    }
}
