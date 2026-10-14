use crate::GeometryGraph;
use serde_json::{Value, json};

#[test]
fn both_histories_reject_invalid_shared_primitives() {
    let owner = GeometryGraph::new();
    let base = owner
        .rectangle_extrusion((-2.0, 2.0), (-1.0, 1.0), 0.0, 3.0, 1.0e-9)
        .unwrap();
    let cut = owner
        .circular_through_cut(&base, [0.0, 0.0], 0.25, 1.0e-9)
        .unwrap();
    let decoder = GeometryGraph::new();
    for operation in [base, cut] {
        decoder.decode_solid(operation.canonical_bytes()).unwrap();
        let wire: Value = serde_json::from_slice(operation.canonical_bytes()).unwrap();
        let mut wrong_unit = wire.clone();
        wrong_unit["length_unit"] = json!("millimetre");
        let mut unknown_field = wire.clone();
        unknown_field["profile"]["unknown"] = json!(0);
        let mut wrong_dependency = wire;
        wrong_dependency["face"]["profile"] = json!("foreign-profile");
        for invalid in [wrong_unit, unknown_field, wrong_dependency] {
            let bytes = serde_json::to_vec(&invalid).unwrap();
            assert!(decoder.decode_solid(&bytes).is_err(), "{invalid}");
        }
    }
}
