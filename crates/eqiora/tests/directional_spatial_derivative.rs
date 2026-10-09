//! A spatial direction contracts the derivative axis, without normalization.
use eqiora::api::ModelDocument;
use eqiora::{DimExponents, DynQuantity};
use eqiora_sem::{EvaluationInput, EvaluationPoint};

const SOURCE: &str = r#"model M() {
    domain body=box(0,8,0,8,0,8);
    coordinate x:m on body from body[0];
    coordinate y:m on body from body[1];
    coordinate z:m on body from body[2];
    parameter ex:vector<1,3>=tensor_value(frame=body,components=[1,0,0]);
    parameter ey:vector<1,3>=tensor_value(frame=body,components=[0,1,0]);
    parameter ez:vector<1,3>=tensor_value(frame=body,components=[0,0,1]);
    let direction=(2*ex-ey+3*ez)*1[m/s];
    let F=ex*y*y*z+ey*z*z*x+ez*x*x*y;
    let dF=contract(grad(F),direction,axes=((1,0),));
    let transposed=contract(grad(F),direction,axes=((0,0),));
    variable anchor:1;
    relation retained { anchor=0; }
    observable scalar:m^3/s on body=contract(grad(x*y*z),direction,axes=((0,0),));
    observable d0:m^3/s on body=component(dF,indices=(0,));
    observable d1:m^3/s on body=component(dF,indices=(1,));
    observable d2:m^3/s on body=component(dF,indices=(2,));
    observable t0:m^3/s on body=component(transposed,indices=(0,));
    observable t1:m^3/s on body=component(transposed,indices=(1,));
    observable t2:m^3/s on body=component(transposed,indices=(2,));
}"#;

#[test]
fn directional_derivatives_contract_the_last_coordinate_axis() {
    let document = ModelDocument::compile("directional.eqi", SOURCE).unwrap();
    let replay = ModelDocument::replay(&document.canonical_json().unwrap()).unwrap();
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let rate = DimExponents::from_integers([0, 3, -1, 0, 0, 0, 0]).unwrap();
    let body = document.aliases()["body"];
    for program in [document.program(), replay.program()] {
        let point = EvaluationPoint::new(
            program,
            body.downcast().unwrap(),
            [2., 3., 5.]
                .into_iter()
                .enumerate()
                .map(|(axis, value)| ((body, axis), DynQuantity::new(value, length)))
                .collect(),
            None,
        )
        .unwrap();
        // At (2,3,5), grad(xyz)=(15,10,6). The rows of grad(F) are
        // (0,30,9), (25,0,20), (12,4,0). Multiply each by (2,-1,3).
        // Contracting the value axis instead gives (11,72,-2), not dF.
        for (name, expected) in [
            ("scalar", 38.),
            ("d0", -3.),
            ("d1", 110.),
            ("d2", 20.),
            ("t0", 11.),
            ("t1", 72.),
            ("t2", -2.),
        ] {
            let value = program
                .evaluate_observable_with_points(
                    document.aliases()[name].downcast().unwrap(),
                    Some(&point),
                    &mut |input, _| match input {
                        EvaluationInput::Value(eqiora::kernel::SymbolRef::Parameter(id)) => {
                            Ok(program.typed_value(id.erase()).unwrap().clone())
                        }
                        _ => panic!("explicit polynomial needs no Field reconstruction"),
                    },
                )
                .unwrap();
            assert_eq!(
                value.real_scalar_value().unwrap(),
                DynQuantity::new(expected, rate)
            );
        }
    }
    assert!(
        ModelDocument::compile(
            "wrong-unit.eqi",
            &SOURCE.replace("scalar:m^3/s", "scalar:m^2/s")
        )
        .is_err()
    );
}
