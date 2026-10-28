use eqiora_assembly::{AssemblyBackend, LinearSystem, REFERENCE_ASSEMBLY_BACKEND};

use super::*;

mod boundary;
mod fixture;
mod implicit;
mod validation;
use fixture::Fixture;

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-11 * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}

fn dense(system: &LinearSystem<f64>) -> Vec<f64> {
    let matrix = system.matrix();
    let n = matrix.rows();
    let mut dense = vec![0.0; n * n];
    for row in 0..n {
        for slot in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
            dense[row * n + matrix.column_indices()[slot]] = matrix.values()[slot];
        }
    }
    dense
}

#[test]
fn matrix_free_tangents_preserve_orientation_and_zero_fixed_directions() {
    for reversed in [false, true] {
        let mut fixture = Fixture::new(3, reversed);
        let n = fixture.rhs.len();
        let sign = |global: usize| if global.is_multiple_of(2) { 1. } else { -1. };
        let full_target = fixture.plan.target_id(1).unwrap();
        for cell in &mut fixture.cells {
            let full = cell
                .mappings
                .iter()
                .find(|map| map.target() == full_target)
                .unwrap();
            for (orientation, unknown) in cell.orientation.iter_mut().zip(full.map().unknowns()) {
                let LocalUnknown::Free(dof) = unknown else {
                    unreachable!()
                };
                *orientation = if sign(dof.index()) > 0. { 1 } else { -1 };
            }
        }
        let free = fixture
            .fixed
            .iter()
            .enumerate()
            .filter_map(|(i, value)| value.is_none().then_some(i))
            .collect::<Vec<_>>();
        let direction = free
            .iter()
            .map(|i| 0.25 + *i as f64 / 7.)
            .collect::<Vec<_>>();
        let prepared = fixture.prepare().unwrap();
        let actual = prepared
            .jacobian_action(
                &fixture.plan,
                fixture.plan.target_id(0).unwrap(),
                RegionPoint::Bound(&vec![2.; free.len()]),
                &direction,
            )
            .unwrap();
        // The fixture's weak-integral oracle is independent of assembly.
        // Reversing functional signs gives D*A*D; fixed values never enter J*v.
        for (row, &global_row) in free.iter().enumerate() {
            let expected = free
                .iter()
                .zip(&direction)
                .map(|(&column, value)| {
                    sign(global_row)
                        * fixture.matrix[global_row * n + column]
                        * sign(column)
                        * value
                })
                .sum::<f64>();
            close(actual[row], expected);
        }
        for invalid in [
            &direction[..direction.len() - 1],
            &vec![f64::NAN; direction.len()],
        ] {
            assert!(
                prepared
                    .jacobian_action(
                        &fixture.plan,
                        fixture.plan.target_id(0).unwrap(),
                        RegionPoint::Bound(&vec![2.; free.len()]),
                        invalid,
                    )
                    .is_err()
            );
        }
    }
}

#[test]
fn domain_reactions_keep_only_selected_rows_from_exact_packet_ownership() {
    let fixture = Fixture::new(3, true);
    let prepared = fixture.prepare().unwrap();
    let n = fixture.rhs.len();
    let rows = [0, 4, n - 1].into_iter().collect::<BTreeSet<_>>();
    let target = fixture.plan.target_id(1).unwrap();
    let reactions = DomainReactions::prepare(
        &prepared,
        target,
        n,
        &fixture.domains,
        fixture.domains.len(),
        &rows,
    )
    .unwrap();
    let values = (0..n)
        .map(|index| 0.25 + index as f64 / 7.0)
        .collect::<Vec<_>>();
    let recovered = reactions.recover(&values).unwrap();
    for (group, domain) in fixture.domains.as_chunks::<2>().0.iter().enumerate() {
        let first = 3 * group * (group + 1) / 2;
        let end = first + 3 * (group + 1);
        let actual = &recovered.values[&domain[0]];
        for (row, actual) in actual.iter().enumerate() {
            let expected = if rows.contains(&row) && (first..end).contains(&row) {
                fixture.matrix[row * n..(row + 1) * n]
                    .iter()
                    .zip(&values)
                    .map(|(coefficient, value)| coefficient * value)
                    .sum::<f64>()
                    - fixture.rhs[row]
            } else {
                0.0
            };
            close(*actual, expected);
        }
        assert!(reactions.recover(&values[..n - 1]).is_err());
    }
    assert!(
        DomainReactions::prepare(
            &prepared,
            target,
            n,
            &fixture.domains[..fixture.domains.len() - 1],
            fixture.domains.len(),
            &rows
        )
        .is_err()
    );
    assert!(
        DomainReactions::prepare(
            &prepared,
            target,
            n,
            &fixture.domains,
            fixture.domains.len(),
            &[n].into_iter().collect()
        )
        .is_err()
    );
    let empty = DomainReactions::prepare(
        &prepared,
        target,
        n,
        &fixture.domains,
        fixture.domains.len(),
        &BTreeSet::new(),
    )
    .unwrap();
    assert!(
        empty
            .recover(&values)
            .unwrap()
            .values
            .values()
            .all(|values| values == &vec![0.0; n])
    );
}

#[test]
fn two_and_three_regions_assemble_variable_fields_through_existing_constraint_maps() {
    for regions in [2, 3] {
        for reversed in [false, true] {
            let fixture = Fixture::new(regions, reversed);
            let prepared = fixture.prepare().unwrap();
            assert_eq!(prepared.packet_count(), 2 * regions);
            assert_eq!(prepared.packet_set_identity(), fixture.packet_set);
            assert_eq!(prepared, prepared.clone());
            let result = REFERENCE_ASSEMBLY_BACKEND
                .assemble(&fixture.plan, &prepared)
                .unwrap();
            let full = result.system(fixture.plan.target_id(1).unwrap()).unwrap();
            for (actual, expected) in dense(full).into_iter().zip(&fixture.matrix) {
                close(actual, *expected);
            }
            for (actual, expected) in full.rhs().iter().zip(&fixture.rhs) {
                close(*actual, *expected);
            }
            let reduced = result.system(fixture.plan.target_id(0).unwrap()).unwrap();
            let free = fixture
                .fixed
                .iter()
                .enumerate()
                .filter_map(|(i, value)| value.is_none().then_some(i))
                .collect::<Vec<_>>();
            let actual = dense(reduced);
            let n = fixture.rhs.len();
            for (row, &global_row) in free.iter().enumerate() {
                let mut rhs = fixture.rhs[global_row];
                for (column, fixed) in fixture.fixed.iter().enumerate() {
                    if let Some(value) = fixed {
                        rhs -= fixture.matrix[global_row * n + column] * value;
                    }
                }
                close(reduced.rhs()[row], rhs);
                for (column, &global_column) in free.iter().enumerate() {
                    close(
                        actual[row * free.len() + column],
                        fixture.matrix[global_row * n + global_column],
                    );
                }
            }
            // Retained packet contributions and projections are the same recovery input.
            for index in 0..prepared.packet_count() {
                let packet = prepared.evaluate(index).unwrap();
                assert_eq!(packet.mappings().len(), 2);
                assert_eq!(packet, prepared.clone().evaluate(index).unwrap());
                packet.project(&fixture.plan, index).unwrap();
            }
            assert!(
                prepared
                    .evaluate(prepared.packet_count())
                    .unwrap_err()
                    .message()
                    .contains("outside the prepared mesh")
            );
        }
    }
}

#[test]
fn region_newton_retains_prepared_multi_target_assembly() {
    use eqiora_realization::NonlinearSolvePlan;
    use eqiora_solver::{LinearSolveRequest, LinearSolver, REFERENCE_LINEAR_SOLVER, SolverPlan};
    use std::num::NonZeroUsize;
    let fixture = Fixture::new(3, true);
    let work = fixture.prepare().unwrap();
    let target = fixture.plan.target_id(0).unwrap();
    let initial = vec![0.; fixture.plan.target(target).unwrap().size()];
    let solution = work
        .solve_newton(
            &fixture.plan,
            target,
            &initial,
            NonlinearSolvePlan::new(0., 1e-10, NonZeroUsize::new(8).unwrap(), 8).unwrap(),
            LinearSolveRequest::new(
                &REFERENCE_LINEAR_SOLVER,
                SolverPlan::new(
                    LinearSolver::BiConjugateGradientStabilized,
                    1e-13,
                    1e-14,
                    NonZeroUsize::new(200).unwrap(),
                )
                .unwrap(),
            ),
            None,
            NonZeroUsize::MIN,
            |_| Ok(()),
        )
        .unwrap();
    assert!(solution.assembly.report.structure_identity().is_some());
    assert_eq!(solution.assembly.report.target_count(), 2);
    assert_eq!(solution.assembly.report.packet_count(), fixture.cells.len());
    let mut free = solution.values.iter();
    let full = fixture
        .fixed
        .iter()
        .map(|fixed| fixed.unwrap_or_else(|| *free.next().unwrap()))
        .collect::<Vec<_>>();
    for (row, prescribed) in fixture.fixed.iter().enumerate() {
        if prescribed.is_none() {
            let expected_rhs = fixture.matrix[row * full.len()..(row + 1) * full.len()]
                .iter()
                .zip(&full)
                .map(|(a, x)| a * x)
                .sum::<f64>();
            close(expected_rhs, fixture.rhs[row]);
        }
    }
}
