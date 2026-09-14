//! Ordinary plural Region/Connection Run and independently predicted recovery.
use super::*;
use eqiora_core::RawId;
use eqiora_meshing::QuadratureRule;

fn chain_source(order: &[usize], names: &[&str]) -> String {
    let count = order.len();
    let mut source = String::from(
        r#"
public connector ScalarBoundary {
  trace value: 1;
  flux outward_flux: 1 / m;
  shape [];
  frame invariant;
  pairing euclidean_boundary_duality;
  orientation parent_outward;
}
public component Interface(
  support body: volume(ambient_dimension = 1),
  support face: boundary(parent = body),
  variable value: 1 on body,
  parameter conductivity: 1,
  port edge: ScalarBoundary over face
) {
  relation carrier on face {
    trace(value) - edge.value = 0;
    normal(conductivity * grad(value)) - edge.outward_flux = 0;
  }
}
model Chain() {
  parameter conductivity: 1 = 2;
"#,
    );
    for &index in order {
        let name = names[index];
        source += &format!(
            r#"
  domain body{index} = box({index}, {end});
  domain lower{index} = boundary(body{index}, axis = 0, side = lower);
  domain upper{index} = boundary(body{index}, axis = 0, side = upper);
  variable {name}: 1 on body{index};
  relation balance{index} on body{index} {{ -div(conductivity * grad({name})) = 0; }}
  observable integral{index}: m = integral({name}, measure(body{index}));
  observable lower_flux{index}: 1/m = integral(normal(conductivity * grad({name})), measure(lower{index}));
  observable upper_flux{index}: 1/m = integral(normal(conductivity * grad({name})), measure(upper{index}));
"#,
            end = index + 1
        );
        for side in ["lower", "upper"] {
            if (index == 0 && side == "lower") || (index + 1 == count && side == "upper") {
                let value = usize::from(side == "upper");
                source += &format!(
                    "relation fixed{index} on {side}{index} {{ trace({name}) = {value}; }}\n"
                );
            } else {
                source += &format!(
                    "instance {side}_carrier{index}: Interface(body = body{index}, face = {side}{index}, value = {name}, conductivity = conductivity);\n"
                );
            }
        }
    }
    for index in 0..count - 1 {
        source += &format!(
            "connect upper_carrier{index}.edge, lower_carrier{}.edge;\n",
            index + 1
        );
    }
    source + "}"
}

fn source_model(source: &str, names: &[&str]) -> (ModelEnvelope, BTreeMap<String, RawId>) {
    let (transaction, model, symbols) = eqiora_compiler::compile("plural-chain.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let mut ids = BTreeMap::new();
    for name in names {
        ids.insert((*name).to_owned(), symbols.get(name).unwrap());
    }
    for index in 0..names.len() {
        for prefix in ["integral", "lower_flux", "upper_flux"] {
            let name = format!("{prefix}{index}");
            ids.insert(name.clone(), symbols.get(&name).unwrap());
        }
    }
    (ModelEnvelope::from_program(&program).unwrap(), ids)
}

fn chain_resources(count: usize) -> AuthenticatedCommonMesh {
    let graph = GeometryGraph::new();
    let interval = graph.interval([0.0, count as f64]).unwrap();
    let [left, right]: [_; 2] = interval.boundaries().try_into().unwrap();
    let geometry = graph
        .build(
            &interval,
            &BTreeMap::from([
                ("body".to_owned(), vec![interval.region().into()]),
                ("left".to_owned(), vec![left.into()]),
                ("right".to_owned(), vec![right.into()]),
            ]),
        )
        .unwrap();
    cartesian_box_resources(&geometry, &[2 * count])
}

fn chain_plan(model: &ModelEnvelope, count: usize) -> Result<ResolvedCommonPlan, Diagnostic> {
    resolve_common_plan(
        model,
        chain_resources(count),
        CommonSpatialPolicy::Q1,
        CommonSolvePolicy::Linear(exact_reference_linear(
            LinearSolver::BiConjugateGradientStabilized,
            1e-10,
            1e-12,
            NonZeroUsize::new(1000).unwrap(),
        )),
        None,
        None,
        &ResolveOnlyBackend,
        None,
    )
}

#[test]
fn plural_chain_ordinary_run_recovers_every_owned_field_and_oriented_interface() {
    for count in [2, 3, 4] {
        let names = ["alpha", "beta", "gamma", "delta"];
        let order = (0..count).collect::<Vec<_>>();
        let (model, ids) = source_model(&chain_source(&order, &names), &names[..count]);
        let resolved = replay_plan(chain_plan(&model, count).unwrap(), &ResolveOnlyBackend);
        let plan = resolved.as_scalar().unwrap();
        assert_eq!(plan.portable_realization().domains().len(), count);
        assert_eq!(
            plan.portable_realization().transformations().len(),
            count - 1
        );
        let result = plan.run_result(&REFERENCE_LINEAR_SOLVER).unwrap();
        assert_eq!(result.field_count(), count);
        let bytes = result.to_bytes().unwrap();
        let recovered = crate::CommonResult::from_bytes(&bytes, &resolved).unwrap();
        assert_eq!(recovered.identity(), result.identity());
        assert_eq!(recovered.to_bytes().unwrap(), bytes);
        for (region, name) in names[..count].iter().enumerate() {
            let field = (0..count)
                .find(|&index| recovered.field(index).unwrap().0 == ids[*name].ulid().to_string())
                .unwrap();
            let (association, values, shape) = recovered.field_block(field, 0).unwrap();
            assert_eq!(association, "vertex");
            assert_eq!(shape, &[3]);
            assert_eq!(
                values.len(),
                3,
                "only this Region's owned vertices are published"
            );
            for (local, value) in values.iter().enumerate() {
                // Independent continuum solution u=x/count is represented exactly by Q1.
                let expected = (region as f64 + local as f64 / 2.0) / count as f64;
                assert!((value - expected).abs() < 1e-9);
            }
            for (prefix, expected, quadrature) in [
                (
                    "integral",
                    (region as f64 + 0.5) / count as f64,
                    QuadratureRule::gauss_legendre(2).unwrap(),
                ),
                ("lower_flux", -2.0 / count as f64, QuadratureRule::point()),
                ("upper_flux", 2.0 / count as f64, QuadratureRule::point()),
            ] {
                let observable = ids[&format!("{prefix}{region}")].downcast().unwrap();
                let actual = recovered
                    .observe(&model, observable, Some(&quadrature))
                    .unwrap()
                    .value()
                    .real_scalar_value()
                    .unwrap()
                    .value();
                assert!(
                    (actual - expected).abs() < 1e-9,
                    "{prefix}{region}: {actual} != {expected}"
                );
            }
        }
    }
}

#[test]
fn plural_chain_reordered_declarations_and_field_names_preserve_support_recovery() {
    let fixtures = [
        ([0, 1, 2], ["alpha", "beta", "gamma"]),
        ([2, 0, 1], ["gamma", "alpha", "beta"]),
    ];
    let mut predictions = Vec::new();
    for (order, names) in fixtures {
        let (model, ids) = source_model(&chain_source(&order, &names), &names);
        let resolved = chain_plan(&model, 3).unwrap();
        let result = resolved
            .as_scalar()
            .unwrap()
            .run_result(&REFERENCE_LINEAR_SOLVER)
            .unwrap();
        let by_support = names
            .iter()
            .map(|name| {
                let field = (0..3)
                    .find(|&index| result.field(index).unwrap().0 == ids[*name].ulid().to_string())
                    .unwrap();
                result.field_block(field, 0).unwrap().1.to_vec()
            })
            .collect::<Vec<_>>();
        predictions.push(by_support);
    }
    for (first, second) in predictions[0]
        .iter()
        .flatten()
        .zip(predictions[1].iter().flatten())
    {
        assert!((first - second).abs() < 1e-9);
    }
}

#[test]
fn plural_chain_rejects_missing_connection_and_inexact_mesh_coverage() {
    let names = ["alpha", "beta", "gamma"];
    let source = chain_source(&[0, 1, 2], &names);
    let (model, _) = source_model(&source, &names);
    for mesh_extent in [2, 4] {
        let error = chain_plan(&model, mesh_extent).unwrap_err();
        assert!(
            error.message().contains("Region bound")
                || error.message().contains("complete Region owner")
        );
    }

    let disconnected = source.replace("connect upper_carrier0.edge, lower_carrier1.edge;", "");
    // The compiler may reject the unconnected port before numerical admission.
    if eqiora_compiler::compile("disconnected.eqi", &disconnected).is_ok() {
        let (model, _) = source_model(&disconnected, &names);
        assert!(chain_plan(&model, 3).is_err());
    }
    for invalid in [
        source.replace("box(1, 2)", "box(1.5, 2)"),
        source.replace("box(1, 2)", "box(0.5, 2)"),
        source.replace("box(2, 3)", "box(2, 4)"),
    ] {
        if let Err(diagnostics) = eqiora_compiler::compile("invalid-coverage.eqi", &invalid) {
            assert!(
                diagnostics
                    .iter()
                    .any(|diagnostic| diagnostic.message().contains("NoncoincidentBoundaries"))
            );
        } else {
            let (model, _) = source_model(&invalid, &names);
            assert!(chain_plan(&model, 3).is_err());
        }
    }
}

#[derive(Debug)]
struct FailingChainBackend;
impl LinearSolverBackend for FailingChainBackend {
    fn provider(&self) -> SolverProvider {
        REFERENCE_LINEAR_SOLVER.provider()
    }
    fn capabilities(&self) -> SolverCapabilities {
        REFERENCE_LINEAR_SOLVER.capabilities()
    }
    fn solve_with_execution(
        &self,
        _problem: &LinearProblem<'_>,
        _plan: SolverPlan,
        _execution: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution, Diagnostic> {
        Err(Diagnostic::error(
            eqiora_core::diagnostic::codes::INVALID_REALIZATION,
            "injected plural solve failure",
        ))
    }
}

#[test]
fn plural_chain_failed_run_publishes_no_result_and_leaves_reusable_plan() {
    let names = ["alpha", "beta", "gamma"];
    let (model, _) = source_model(&chain_source(&[0, 1, 2], &names), &names);
    let resolved = chain_plan(&model, 3).unwrap();
    let before = resolved.to_bytes().unwrap();
    let plan = resolved.as_scalar().unwrap();
    assert!(plan.run_result(&FailingChainBackend).is_err());
    assert_eq!(resolved.to_bytes().unwrap(), before);
    assert_eq!(
        plan.run_result(&REFERENCE_LINEAR_SOLVER)
            .unwrap()
            .field_count(),
        3
    );
}

#[test]
fn plural_chain_rejects_partial_misbound_and_wrong_support_result_inventory() {
    let names = ["alpha", "beta", "gamma"];
    let (model, _) = source_model(&chain_source(&[0, 1, 2], &names), &names);
    let resolved = chain_plan(&model, 3).unwrap();
    let plan = resolved.as_scalar().unwrap();
    let output = plan
        .admission
        .execute_scalar(&REFERENCE_LINEAR_SOLVER)
        .unwrap();
    let mut missing = output.clone();
    missing.fields.pop();
    let mut duplicate = output.clone();
    duplicate.fields[1].0 = duplicate.fields[0].0;
    let mut foreign = output.clone();
    foreign.fields[0].0 = eqiora_core::Id::new();
    let mut whole_mesh = output.clone();
    whole_mesh.fields[0].2.resize(7, 0.0);
    let mut nonfinite = output.clone();
    nonfinite.fields[0].2[0] = f64::NAN;
    for invalid in [missing, duplicate, foreign, whole_mesh, nonfinite] {
        assert!(crate::CommonResult::accept_scalar(plan.clone(), 0.0, invalid).is_err());
    }
    assert!(crate::CommonResult::accept_scalar(plan.clone(), 0.0, output).is_ok());
}

#[test]
fn plural_chain_permutation_retains_exact_field_identity_and_result_recovery() {
    let names = ["alpha", "beta", "gamma"];
    let (model, ids) = source_model(&chain_source(&[0, 1, 2], &names), &names);
    let resolved = chain_plan(&model, 3).unwrap();
    let plan = resolved.as_scalar().unwrap();
    let RecognizedNativeModel::Scalar(equations) = plan.admission.recognized_model() else {
        panic!("scalar inventory");
    };
    let mut permuted = equations.clone();
    permuted.regions.reverse();
    permuted.interfaces.reverse();
    let NativeMeshResources::Cartesian { mesh, .. } = plan.admission.resources() else {
        panic!("Cartesian mesh");
    };
    let output = permuted
        .execute(
            &plan.admission,
            LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan.admission.linear.solver),
            mesh.mesh(),
        )
        .unwrap();
    let result = crate::CommonResult::accept_scalar(plan.clone(), 0.0, output).unwrap();
    for (region, name) in names.iter().enumerate() {
        let index = (0..3)
            .find(|&index| result.field(index).unwrap().0 == ids[*name].ulid().to_string())
            .unwrap();
        let (_, values, shape) = result.field_block(index, 0).unwrap();
        assert_eq!(shape, &[3]);
        for (local, value) in values.iter().enumerate() {
            let expected = (region as f64 + local as f64 / 2.0) / 3.0;
            assert!((value - expected).abs() < 1e-9);
        }
    }
    let replay = crate::CommonResult::from_bytes(&result.to_bytes().unwrap(), &resolved).unwrap();
    assert_eq!(replay.identity(), result.identity());
}

#[test]
fn plural_solver_admits_and_rechecks_exact_fields_for_manual_and_planned_runs() {
    use eqiora_solver::{AlgebraicStructure, HostSerialSolverProfile};
    let names = ["alpha", "beta", "gamma"];
    let (model, ids) = source_model(&chain_source(&[0, 1, 2], &names), &names);
    let expected =
        AlgebraicStructure::new(names.iter().map(|name| ids[*name].downcast().unwrap()), [])
            .unwrap();
    for (planned, request) in [
        (
            false,
            exact_reference_linear(
                LinearSolver::BiConjugateGradientStabilized,
                1e-10,
                1e-12,
                NonZeroUsize::new(1000).unwrap(),
            ),
        ),
        (
            true,
            CommonLinearRequest::program_controlled(
                1e-10,
                1e-12,
                NonZeroUsize::new(1000).unwrap(),
                SolverPlanningObjective::LowMemory,
            )
            .unwrap(),
        ),
    ] {
        let supplied: &dyn LinearSolverBackend = if planned {
            &PlanningFaerBackend
        } else {
            &REFERENCE_LINEAR_SOLVER
        };
        let resolved = resolve_common_plan(
            &model,
            chain_resources(3),
            CommonSpatialPolicy::Q1,
            CommonSolvePolicy::Linear(request),
            None,
            None,
            supplied,
            None,
        )
        .unwrap();
        let resolved = replay_plan(resolved, supplied);
        let plan = resolved.as_scalar().unwrap();
        let profile = plan.admission.linear.planning_profile.as_ref().unwrap();
        profile.require_structure(Some(&expected)).unwrap();
        assert!(profile.require_structure(None).is_err());
        if !planned {
            let accepted = plan.run_result(&REFERENCE_LINEAR_SOLVER).unwrap();
            assert_eq!(accepted.field_count(), 3);
            for index in 0..3 {
                let (id, _, _, _) = accepted.field(index).unwrap();
                let region = names
                    .iter()
                    .position(|name| ids[*name].ulid().to_string() == id)
                    .unwrap();
                let (_, values, _) = accepted.field_block(index, 0).unwrap();
                for (vertex, value) in values.iter().enumerate() {
                    // Independent exact harmonic profile on [0,3], with endpoints 0 and 1.
                    assert!((value - (region as f64 + vertex as f64 / 2.0) / 3.0).abs() < 1e-9);
                }
            }
        }
        let missing = AlgebraicStructure::new([ids["alpha"].downcast().unwrap()], []).unwrap();
        let foreign = AlgebraicStructure::new(
            [eqiora_core::Id::from_ulid(
                "01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap(),
            )],
            [],
        )
        .unwrap();
        for stale in [None, Some(missing), Some(foreign)] {
            let mut changed = plan.clone();
            let replacement = HostSerialSolverProfile::canonical_csr(
                LinearOperatorProperties::General,
                None,
                None,
            );
            changed.admission.linear.planning_profile = Some(match stale {
                Some(structure) => replacement.with_structure(structure).unwrap(),
                None => replacement,
            });
            let error = changed.run_result(supplied).unwrap_err();
            assert!(error.message().contains("structure"), "{error:?}");
        }
    }
}
