use super::*;
use eqiora::api::ModelDocument;

const SOURCE: &str = r#"
model Reset() {
    state x: 1;
    initial { x=0; }
    parameter threshold: 1=0.4;
    relation flow { derivative(x)=1[1/s]; }
    event hit=crossing(x-threshold,direction=rising);
    relation reset at hit { next(x)=0; }
    observable sample: 1=x;
}
"#;

fn message(py: Python<'_>, error: &PyErr) -> PyResult<String> {
    Ok(error.value(py).getattr("args")?.extract::<(String,)>()?.0)
}

#[test]
fn typed_factories_bind_aliases_and_ids_to_the_exact_model() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let authored =
            PyModel::from_document(py, ModelDocument::compile("selection.eqi", SOURCE).unwrap())?;
        let canonical = PyModel::from_artifact(py, authored.artifact().clone())?;
        assert!(authored.document().is_ok());
        assert!(canonical.document().is_err());
        let digest = authored
            .artifact()
            .artifact_reference()
            .unwrap()
            .artifact()
            .to_string();
        let foreign = PyModel::from_document(
            py,
            ModelDocument::compile(
                "foreign.eqi",
                &SOURCE
                    .replace("hit", "foreign_hit")
                    .replace("sample", "foreign_sample"),
            )
            .unwrap(),
        )?;
        let field_id = authored.artifact_ids(eqiora::EntityKind::Field).unwrap()[0].clone();

        macro_rules! check {
            ($factory:ident, $alias:literal, $foreign_alias:literal, $kind:literal) => {
                let selected = authored.$factory(py, $alias)?;
                let id = selected.id.ulid().to_string();
                let other = foreign.$factory(py, $foreign_alias)?;
                assert_ne!(selected.model_digest, other.model_digest);
                assert_ne!(selected.id, other.id);
                assert_eq!(selected.model_digest, digest);
                for model in [&authored, &canonical] {
                    assert_eq!(model.$factory(py, &id)?, selected);
                    for rejected in [
                        "00000000000000000000000000".to_owned(),
                        field_id.clone(),
                        other.id.ulid().to_string(),
                    ] {
                        let error = model.$factory(py, &rejected).unwrap_err();
                        assert!(error.is_instance_of::<PyKeyError>(py));
                        assert_eq!(
                            message(py, &error)?,
                            concat!($kind, " is outside this exact Model")
                        );
                    }
                    for rejected in ["missing_alias", "not-a-ulid", "0".repeat(27).as_str()] {
                        let error = model.$factory(py, rejected).unwrap_err();
                        assert!(error.is_instance_of::<PyKeyError>(py));
                        assert_eq!(message(py, &error)?, rejected);
                    }
                }
                let error = authored.$factory(py, "x").unwrap_err();
                assert!(error.is_instance_of::<PyValueError>(py));
                assert_eq!(
                    message(py, &error)?,
                    concat!("selection does not identify an ", $kind)
                );
                let error = canonical.$factory(py, $alias).unwrap_err();
                assert!(error.is_instance_of::<PyKeyError>(py));
                assert_eq!(message(py, &error)?, $alias);
            };
        }
        check!(activation, "hit", "foreign_hit", "Activation");
        check!(observable, "sample", "foreign_sample", "Observable");
        Ok(())
    })
}
