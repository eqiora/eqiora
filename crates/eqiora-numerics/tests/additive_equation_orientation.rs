use eqiora_compiler::compile;
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_numerics::scalar::lower_scalar_elliptic_cartesian;
use eqiora_schema::kernel::BoundarySide;
use eqiora_sem::KernelProgram;

const SOURCE: &str = r#"
model additive_poisson() {
  domain body = box(0, 1, 0, 1);
  domain x_lower = boundary(body, axis = 0, side = lower);
  domain x_upper = boundary(body, axis = 0, side = upper);
  domain y_lower = boundary(body, axis = 1, side = lower);
  domain y_upper = boundary(body, axis = 1, side = upper);

  variable potential: 1 on body in h1;
  parameter diffusion: 1 = 2;
  parameter source: 1 / m ^ 2 = 3;
  parameter value: 1 = 4;
  relation balance on body {
    -div(diffusion * grad(potential)) - source = 0;
  }
  relation x_lower_value on x_lower { trace(potential) - value = 0; }
  relation x_upper_value on x_upper { trace(potential) - value = 0; }
  relation y_lower_value on y_lower { trace(potential) - value = 0; }
  relation y_upper_value on y_upper { trace(potential) - value = 0; }
}
"#;

fn compile_program(source: &str) -> KernelProgram {
    let mut compiled = compile("additive-poisson.eqi", source).expect("source compiles");
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).expect("transaction commits");
    KernelProgram::from_snapshot(&store.snapshot(), model).expect("model validates")
}

#[test]
fn scalar_elliptic_admission_accepts_additive_grouping_and_whole_sign_reversal() {
    let additive = SOURCE
        .replace(
            "-div(diffusion * grad(potential)) - source = 0;",
            "-div(diffusion * grad(potential)) + -source = 0;",
        )
        .replace(
            "trace(potential) - value = 0;",
            "trace(potential) + -value = 0;",
        );
    let additive_model = lower_scalar_elliptic_cartesian(&compile_program(&additive))
        .expect("additively grouped scalar equation lowers");
    assert_eq!(
        additive_model.coefficient_expression().constant_value(),
        Some(2.0)
    );
    assert_eq!(additive_model.source().constant_value(), Some(3.0));

    let reversed = SOURCE
        .replace(
            "-div(diffusion * grad(potential)) - source = 0;",
            "div(diffusion * grad(potential)) + source = 0;",
        )
        .replace(
            "trace(potential) - value = 0;",
            "-trace(potential) + value = 0;",
        );
    let reversed_model = lower_scalar_elliptic_cartesian(&compile_program(&reversed))
        .expect("whole-equation scalar sign reversal lowers");
    assert_eq!(reversed_model.source().constant_value(), Some(3.0));
    assert!(
        reversed_model
            .boundary(0, BoundarySide::Lower)
            .unwrap()
            .is_essential()
    );
}

#[test]
fn scalar_elliptic_admission_rejects_sign_wrong_and_duplicate_roles() {
    let sign_wrong = SOURCE.replace(
        "-div(diffusion * grad(potential)) - source = 0;",
        "-div(diffusion * grad(potential)) + source = 0;",
    );
    let diagnostic = lower_scalar_elliptic_cartesian(&compile_program(&sign_wrong))
        .expect_err("sign-wrong source role rejects");
    assert!(diagnostic.message().contains("unmatched signed leaves"));

    let duplicate = SOURCE.replace(
        "-div(diffusion * grad(potential)) - source = 0;",
        "-div(diffusion * grad(potential)) - source - source = 0;",
    );
    assert!(lower_scalar_elliptic_cartesian(&compile_program(&duplicate)).is_err());

    let excessive = SOURCE.replace(
        "-div(diffusion * grad(potential)) - source = 0;",
        &format!(
            "-div(diffusion * grad(potential)) {} = 0;",
            std::iter::repeat_n("- source", 16)
                .collect::<Vec<_>>()
                .join(" ")
        ),
    );
    let diagnostic = lower_scalar_elliptic_cartesian(&compile_program(&excessive))
        .expect_err("bounded additive leaf inventory rejects");
    assert!(
        diagnostic
            .message()
            .contains("additive residual exceeds maximum"),
        "{}",
        diagnostic.message()
    );
}

#[test]
fn scalar_elliptic_orientation_retains_composite_source_and_boundary_values() {
    let composite = SOURCE
        .replace("- source = 0;", "- (source + source) = 0;")
        .replace(
            "trace(potential) - value = 0;",
            "trace(potential) - (value + value) = 0;",
        );
    let model = lower_scalar_elliptic_cartesian(&compile_program(&composite))
        .expect("existing composite scalar roles remain admitted");
    assert_eq!(model.source().constant_value(), Some(6.0));
    assert_eq!(
        model
            .boundary(0, BoundarySide::Lower)
            .unwrap()
            .value()
            .constant_value(),
        Some(8.0)
    );

    let additive = SOURCE.replace(
        "-div(diffusion * grad(potential)) - source = 0;",
        "-div(diffusion * grad(potential)) + -(source + source) = 0;",
    );
    let model = lower_scalar_elliptic_cartesian(&compile_program(&additive))
        .expect("additive orientation retains one grouped composite source role");
    assert_eq!(model.source().constant_value(), Some(6.0));

    let reversed = SOURCE
        .replace(
            "-div(diffusion * grad(potential)) - source = 0;",
            "div(diffusion * grad(potential)) = -(source + source);",
        )
        .replace(
            "trace(potential) - value = 0;",
            "(value + value) - trace(potential) = 0;",
        );
    let model = lower_scalar_elliptic_cartesian(&compile_program(&reversed))
        .expect("whole reversal retains grouped composite scalar roles");
    assert_eq!(model.source().constant_value(), Some(6.0));
    assert_eq!(
        model
            .boundary(0, BoundarySide::Lower)
            .unwrap()
            .value()
            .constant_value(),
        Some(8.0)
    );
}
