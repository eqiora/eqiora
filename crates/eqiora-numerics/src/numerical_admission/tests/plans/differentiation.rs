//! Supplied candidates use exact Plan assembly and original residual admission.
mod material;
mod mixed_scalar;
mod tensor_material;

use super::*;
use eqiora_ir::{LinearizedRelation, RelationTangent};

#[test]
fn supplied_spatial_candidates_share_boundary_lifting_and_original_acceptance() {
    let geometry = cartesian_interval();
    for affine_boundary in [false, true] {
        let source = if affine_boundary {
            POISSON_INTERVAL
                .replace("trace(potential) = 0; }", "trace(potential) = 1; }")
                .replace(
                    "relation right_value on right { trace(potential) = 1; }",
                    "relation right_value on right { trace(potential) = 2; }",
                )
        } else {
            POISSON_INTERVAL.to_owned()
        };
        let model = scalar_box_model(&geometry, &source, "PoissonInterval", &["left", "right"]);
        for spatial in [
            CommonSpatialPolicy::Q1,
            CommonSpatialPolicy::CellCenteredTpfa,
        ] {
            let plan =
                resolve_scalar_box(&model, cartesian_box_resources(&geometry, &[4]), spatial);
            let parameter = plan
                .admission
                .program()
                .nodes()
                .find_map(|node| match node {
                    eqiora_schema::kernel::KernelNode::Parameter(value) => Some(value.id()),
                    _ => None,
                })
                .unwrap();
            let outputs = plan
                .fields()
                .map(|(field, _)| field.erase())
                .collect::<Vec<_>>();
            // -u''=2 on [0,1]. Q1 interior nodal values are x(1-x).
            // Four TPFA cells have half-cell boundary resistance; solving their
            // tridiagonal balance gives [1/8,1/4,1/4,1/8]. An affine lift 1+x
            // adds exactly its nodal/cell-centre samples to either discrete system.
            let (mut candidate, locations) = match spatial {
                CommonSpatialPolicy::Q1 => (
                    vec![3. / 16., 1. / 4., 3. / 16.],
                    vec![1. / 4., 1. / 2., 3. / 4.],
                ),
                CommonSpatialPolicy::CellCenteredTpfa => (
                    vec![1. / 8., 1. / 4., 1. / 4., 1. / 8.],
                    vec![1. / 8., 3. / 8., 5. / 8., 7. / 8.],
                ),
                _ => unreachable!(),
            };
            if affine_boundary {
                for (value, x) in candidate.iter_mut().zip(locations) {
                    *value += 1. + x;
                }
            }
            // No execution precedes this independently supplied candidate.
            let accepted = plan
                .validate_differentiated_values(
                    None,
                    &[parameter],
                    Some(&[2.]),
                    &outputs,
                    &std::collections::HashMap::new(),
                    &candidate,
                )
                .unwrap();
            assert_eq!(accepted.relation().accepted_unknowns(), candidate);
            assert_eq!(accepted.relation().design_values(), &[2.]);
            assert_eq!(accepted.initial_residual_norm(), None);
            let expected = if spatial == CommonSpatialPolicy::Q1 {
                let (left, right) = if affine_boundary { (1., 2.) } else { (0., 0.) };
                [vec![left], candidate.clone(), vec![right]].concat()
            } else {
                candidate.clone()
            };
            assert_eq!(accepted.output_values(), expected);
            let mut action = vec![0.; candidate.len()];
            accepted
                .relation()
                .jvp(RelationTangent::Parameter(&[1.]), &mut action)
                .unwrap();
            for value in action {
                assert!((value + 0.25).abs() < 1e-14);
            }
            for bad in [
                vec![],
                vec![f64::NAN; candidate.len()],
                vec![f64::INFINITY; candidate.len()],
                candidate.iter().map(|value| value + 0.25).collect(),
            ] {
                assert!(
                    plan.validate_differentiated_values(
                        None,
                        &[parameter],
                        Some(&[2.]),
                        &outputs,
                        &std::collections::HashMap::new(),
                        &bad
                    )
                    .is_err()
                );
            }
            assert!(
                plan.validate_differentiated_values(
                    None,
                    &[parameter],
                    Some(&[1.]),
                    &outputs,
                    &std::collections::HashMap::new(),
                    &candidate
                )
                .is_err()
            );
            let foreign_mesh =
                resolve_scalar_box(&model, cartesian_box_resources(&geometry, &[8]), spatial);
            assert!(
                foreign_mesh
                    .validate_differentiated_values(
                        None,
                        &[parameter],
                        Some(&[2.]),
                        &outputs,
                        &std::collections::HashMap::new(),
                        &candidate
                    )
                    .is_err()
            );
            // Ordinary execution returns the same mathematical owner. Its exact
            // computed coordinates revalidate identically, without another solve.
            let live = plan
                .differentiate(
                    None,
                    &[parameter],
                    Some(&[2.]),
                    &outputs,
                    &std::collections::HashMap::new(),
                    &REFERENCE_LINEAR_SOLVER,
                )
                .unwrap();
            let replayed = plan
                .validate_differentiated_values(
                    None,
                    &[parameter],
                    Some(&[2.]),
                    &outputs,
                    &std::collections::HashMap::new(),
                    live.linearization().relation().accepted_unknowns(),
                )
                .unwrap();
            assert_eq!(&replayed, live.linearization());
        }
    }
}
