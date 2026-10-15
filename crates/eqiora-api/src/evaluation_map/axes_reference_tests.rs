use eqiora_numerics::CommonSpatialPolicy;
use eqiora_realization::RealizationRevision;

use super::*;
use crate::evaluation_map::tests::{fixture, program_for};

const LIMITS: Limits = Limits {
    rank: 8,
    values: 128,
    bytes: 1024,
};

fn coordinates(mut index: usize, extents: &[usize]) -> Vec<usize> {
    let mut result = vec![0; extents.len()];
    for (coordinate, &extent) in result.iter_mut().zip(extents).rev() {
        *coordinate = index % extent;
        index /= extent;
    }
    result
}

/// Ordinary arithmetic specimen, not another Program evaluator. Shape/budget
/// admission completes before allocating outputs or reaching the function.
fn affine(
    a: (&[f64], &Layout<'_>),
    b: (&[f64], &Layout<'_>),
    output: &[Layout<'_>; 2],
    limits: Limits,
) -> Admission<([Vec<f64>; 2], usize)> {
    for layout in [a.1, b.1, &output[0], &output[1]] {
        if layout.extents != a.1.extents || !layout.value_shape.is_empty() {
            return Err(Error::Signature);
        }
    }
    a.1.admit_buffer(&a.1.shape, a.0.len())?;
    b.1.admit_buffer(&b.1.shape, b.0.len())?;
    output_elements(output, limits)?;
    let mut result = [vec![0.0; output[0].elements], vec![0.0; output[1].elements]];
    let mut calls = 0;
    for index in 0..product(a.1.extents)? {
        let occurrence = coordinates(index, a.1.extents);
        let a = *a.1.select(a.0, &occurrence, &[])?;
        let b = *b.1.select(b.0, &occurrence, &[])?;
        let values = [2.0 * a + b, a - 3.0 * b];
        calls += 1;
        for leaf in 0..2 {
            result[leaf][output[leaf].offset(&occurrence, &[])?] = values[leaf];
        }
    }
    Ok((result, calls))
}

#[test]
fn zipped_shared_duplicates_and_permutation_have_hand_derived_outputs() {
    let extents = infer_extents(&[None], &[(&[3], &[Some(0)])], LIMITS).unwrap();
    let mapped = Layout::new(&[], &extents, &[Some(0)], LIMITS).unwrap();
    let shared = Layout::new(&[], &extents, &[None], LIMITS).unwrap();
    let outputs = [
        Layout::new(&[], &extents, &[Some(0)], LIMITS).unwrap(),
        Layout::new(&[], &extents, &[Some(0)], LIMITS).unwrap(),
    ];
    let a = [2.0, 1.0, 2.0];
    let b = [10.0, 20.0, 10.0];
    let (zipped, calls) = affine((&a, &mapped), (&b, &mapped), &outputs, LIMITS).unwrap();
    assert_eq!(calls, 3);
    assert_eq!(zipped, [vec![14.0, 22.0, 14.0], vec![-28.0, -59.0, -28.0]]);
    let (permuted, _) = affine(
        (&[1.0, 2.0, 2.0], &mapped),
        (&[20.0, 10.0, 10.0], &mapped),
        &outputs,
        LIMITS,
    )
    .unwrap();
    assert_eq!(
        permuted,
        [vec![22.0, 14.0, 14.0], vec![-59.0, -28.0, -28.0]]
    );
    let shared_value = [10.0];
    let (shared_result, _) =
        affine((&a, &mapped), (&shared_value, &shared), &outputs, LIMITS).unwrap();
    assert_eq!(
        shared_result,
        [vec![14.0, 12.0, 14.0], vec![-28.0, -29.0, -28.0]]
    );
    assert_eq!(shared_value, [10.0]);
}

#[test]
fn nested_maps_form_an_explicit_product_with_declared_output_positions() {
    let extents = infer_extents(
        &[None, None],
        &[(&[2], &[Some(0), None]), (&[2], &[None, Some(0)])],
        LIMITS,
    )
    .unwrap();
    let a = Layout::new(&[], &extents, &[Some(0), None], LIMITS).unwrap();
    let b = Layout::new(&[], &extents, &[None, Some(0)], LIMITS).unwrap();
    let outputs = [
        Layout::new(&[], &extents, &[Some(0), Some(1)], LIMITS).unwrap(),
        Layout::new(&[], &extents, &[Some(1), Some(0)], LIMITS).unwrap(),
    ];
    let (result, calls) = affine((&[1.0, 2.0], &a), (&[10.0, 20.0], &b), &outputs, LIMITS).unwrap();
    assert_eq!(calls, 4);
    assert_eq!(result[0], [12.0, 22.0, 14.0, 24.0]);
    // Leaf 1 puts the inner b axis first: transpose of [-29,-59; -28,-58].
    assert_eq!(result[1], [-29.0, -28.0, -59.0, -58.0]);
}

#[test]
fn nonleading_and_nested_axes_preserve_the_unbatched_shape() {
    let value_shape = [2];
    let layout = Layout::new(&value_shape, &[3], &[Some(1)], LIMITS).unwrap();
    assert_eq!(layout.shape, [2, 3]);
    assert_eq!(layout.value_shape, value_shape);
    // Index payloads only: this does not evaluate complex numbers.
    let buffer = [10, 20, 30, 11, 21, 31];
    let member = |position| {
        [
            buffer[layout.offset(&[position], &[0]).unwrap()],
            buffer[layout.offset(&[position], &[1]).unwrap()],
        ]
    };
    assert_eq!(member(0), [10, 11]);
    assert_eq!(member(1), [20, 21]);
    assert_eq!(member(2), [30, 31]);
    assert_eq!(layout.admit_buffer(&[3, 2], 6), Err(Error::Shape));
    let nested = Layout::new(&value_shape, &[3, 2], &[Some(1), Some(0)], LIMITS).unwrap();
    assert_eq!(nested.shape, [2, 3, 2]);
    assert_eq!(nested.offset(&[2, 1], &[1]), Ok(11));
    assert_eq!(nested.offset(&[0, 1], &[0]), Ok(6));
    assert_eq!(nested.offset(&[3, 0], &[0]), Err(Error::Bounds));
    assert_eq!(nested.offset(&[0, 0], &[2]), Err(Error::Bounds));
}

#[test]
fn empty_singleton_and_shared_only_maps_have_static_output_shapes() {
    for (extent, expected_calls) in [(0, 0), (1, 1)] {
        let extents = infer_extents(&[Some(extent)], &[(&[], &[None])], LIMITS).unwrap();
        let shared = Layout::new(&[], &extents, &[None], LIMITS).unwrap();
        let outputs = [
            Layout::new(&[], &extents, &[Some(0)], LIMITS).unwrap(),
            Layout::new(&[], &extents, &[Some(0)], LIMITS).unwrap(),
        ];
        let (values, calls) =
            affine((&[1.0], &shared), (&[10.0], &shared), &outputs, LIMITS).unwrap();
        assert_eq!(calls, expected_calls);
        assert_eq!(outputs[0].shape, [extent]);
        assert!(outputs[0].value_shape.is_empty());
        if extent == 0 {
            assert_eq!(values, [Vec::<f64>::new(), Vec::<f64>::new()]);
            assert_eq!(shared.admit_buffer(&[0], 0), Err(Error::Shape));
        } else {
            assert_eq!(values, [vec![12.0], vec![-29.0]]);
        }
    }
    assert_eq!(
        infer_extents(&[None], &[(&[], &[None])], LIMITS),
        Err(Error::Extent)
    );
    let empty = Layout::new(&[], &[2, 0], &[Some(0), Some(1)], LIMITS).unwrap();
    assert_eq!(empty.shape, [2, 0]);
    assert_eq!(empty.offset(&[0, 0], &[]), Err(Error::Bounds));
}

#[test]
fn shape_axis_extent_and_resource_failures_reject_before_buffer_allocation() {
    assert_eq!(
        infer_extents(&[None], &[(&[2], &[Some(0)]), (&[1], &[Some(0)])], LIMITS),
        Err(Error::Extent)
    );
    assert_eq!(
        infer_extents(&[Some(3)], &[(&[2], &[Some(0)])], LIMITS),
        Err(Error::Extent)
    );
    assert_eq!(
        infer_extents(&[None], &[(&[2], &[Some(1)])], LIMITS),
        Err(Error::Axis)
    );
    assert_eq!(
        infer_extents(&[None, None], &[(&[2], &[Some(0), Some(0)])], LIMITS),
        Err(Error::Axis)
    );
    assert_eq!(infer_extents(&[], &[], LIMITS), Err(Error::Rank));
    assert_eq!(product(&[usize::MAX, 2]), Err(Error::Overflow));
    assert_eq!(product(&[0, usize::MAX, 2]), Err(Error::Overflow));
    assert_eq!(budget(usize::MAX, LIMITS), Err(Error::Overflow));
    assert_eq!(
        Layout::new(&[], &[129], &[Some(0)], LIMITS).unwrap_err(),
        Error::Resource
    );
    assert_eq!(
        Layout::new(&[], &[2], &[Some(1)], LIMITS).unwrap_err(),
        Error::Axis
    );
    assert_eq!(
        Layout::new(&[], &[2], &[], LIMITS).unwrap_err(),
        Error::Rank
    );
    let outputs = [
        Layout::new(&[], &[2], &[Some(0)], LIMITS).unwrap(),
        Layout::new(&[], &[2], &[Some(0)], LIMITS).unwrap(),
    ];
    assert_eq!(outputs[0].admit_buffer(&[2], 1), Err(Error::Shape));
    assert_eq!(outputs[0].admit_buffer(&[1, 2], 2), Err(Error::Shape));
    assert_eq!(
        output_elements(
            &outputs,
            Limits {
                bytes: 31,
                ..LIMITS
            }
        ),
        Err(Error::Resource)
    );
    let omitted = [Layout::new(&[], &[2], &[None], LIMITS).unwrap()];
    assert_eq!(output_elements(&omitted, LIMITS), Err(Error::Axis));
}

#[test]
fn occurrences_sample_ids_points_and_accepted_lineage_remain_distinct() {
    let (document, program) = fixture();
    let foreign = program_for(
        &document,
        CommonSpatialPolicy::Q1,
        RealizationRevision::new(22),
        &["source_scale", "diffusion", "boundary_offset"],
    );
    // Same field and flat output length do not imply the same exact Plan.
    assert_eq!(
        program.identity().output_dimension(),
        foreign.identity().output_dimension()
    );
    assert_eq!(
        require_signature(program.identity(), foreign.identity()),
        Err(Error::Signature)
    );
    assert_eq!(
        require_signature(program.identity(), program.identity()),
        Ok(())
    );
    let accepted = program.evaluate(program.default_point()).unwrap();
    let point = accepted.point();
    // Same numerical point, but three request occurrences; only the final
    // record refers to an already accepted evaluation. A sample ID is not an
    // authority to substitute that acceptance into either earlier request.
    let requests = [
        (Some(41_u64), point, None),
        (Some(42), point, None),
        (Some(41), point, Some(&accepted)),
    ];
    let layout = Layout::new(&[], &[3], &[Some(0)], LIMITS).unwrap();
    let positions = [2, 0, 1];
    let associations = positions
        .into_iter()
        .enumerate()
        .map(|(new_position, old_position)| {
            (
                new_position,
                layout.select(&requests, &[old_position], &[]).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        associations
            .iter()
            .map(|entry| entry.1.0)
            .collect::<Vec<_>>(),
        [Some(41), Some(41), Some(42)]
    );
    for (position, association) in associations.iter().enumerate() {
        assert_eq!(association.0, position);
        assert!(std::ptr::eq(association.1, &requests[positions[position]]));
        assert!(std::ptr::eq(association.1.1, point));
        assert_eq!(association.1.1.signatures(), program.identity().inputs());
    }
    assert!(std::ptr::eq(associations[0].1.2.unwrap(), &accepted));
    assert!(associations[1].1.2.is_none());
    assert!(associations[2].1.2.is_none());
    // Attaching an ID does not produce a new acceptance, point, or random path.
    assert_eq!(point.coordinates(), program.default_point().coordinates());
}
