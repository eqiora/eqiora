//! Command transport for the existing stationary map and exact collection owners.
use super::*;
use eqiora::api::{
    EvaluationCollection, EvaluationMapExecutionPolicy, EvaluationMapPlan,
    RecordedEvaluationOutcome,
};
use eqiora::artifact::WireValueLiteral;
use eqiora_numerics::ResolvedCommonPlan;
use eqiora_solver::LinearSolverBackend;
#[cfg(not(feature = "faer"))]
use eqiora_solver::REFERENCE_LINEAR_SOLVER;
use std::io::Write;
use std::sync::Arc;

const LIMIT: usize = 67_108_864;

pub(super) fn command() -> clap::Command {
    use clap::{Arg, ArgAction, Command};
    let path = |name| {
        Arg::new(name)
            .required(true)
            .value_parser(clap::builder::OsStringValueParser::new())
    };
    Command::new("study")
        .about("Run, inspect or retry exact stationary Parameter studies")
        .subcommand_required(true)
        .subcommand(
            Command::new("run")
                .arg(path("plan"))
                .arg(
                    Arg::new("parameter")
                        .long("parameter")
                        .required(true)
                        .action(ArgAction::Append),
                )
                .arg(
                    Arg::new("point")
                        .long("point")
                        .required(true)
                        .action(ArgAction::Append),
                )
                .arg(path("output").long("output")),
        )
        .subcommand(Command::new("inspect").arg(path("collection")))
        .subcommand(
            Command::new("retry")
                .arg(path("collection"))
                .arg(
                    Arg::new("group")
                        .long("group")
                        .required(true)
                        .value_parser(clap::value_parser!(usize)),
                )
                .arg(path("output").long("output")),
        )
}

fn backend() -> Arc<dyn LinearSolverBackend + Send + Sync> {
    #[cfg(feature = "faer")]
    {
        Arc::new(eqiora_backend_faer::FaerLinearSolver)
    }
    #[cfg(not(feature = "faer"))]
    {
        Arc::new(REFERENCE_LINEAR_SOLVER)
    }
}

fn read(path: &OsString) -> Result<Vec<u8>, String> {
    if !std::fs::metadata(path)
        .map_err(|error| error.to_string())?
        .is_file()
    {
        return Err("input must be a regular file".into());
    }
    let file = File::open(path).map_err(|error| error.to_string())?;
    if !file
        .metadata()
        .map_err(|error| error.to_string())?
        .is_file()
    {
        return Err("input must be a regular file".into());
    }
    let mut bytes = Vec::new();
    file.take((LIMIT + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() > LIMIT {
        return Err("study input exceeds 67108864 bytes".into());
    }
    Ok(bytes)
}
fn diagnostics(errors: Vec<eqiora::Diagnostic>) -> String {
    errors
        .iter()
        .map(|error| format!("{error:?}"))
        .collect::<Vec<_>>()
        .join("\n")
}
fn one(error: eqiora::Diagnostic) -> String {
    diagnostics(vec![error])
}
fn collection(
    args: &clap::ArgMatches,
    backend: &dyn LinearSolverBackend,
) -> Result<EvaluationCollection, String> {
    EvaluationCollection::from_bytes(
        &read(args.get_one::<OsString>("collection").expect("required"))?,
        backend,
        &[],
    )
    .map_err(diagnostics)
}
fn execute(
    plan: ResolvedCommonPlan,
    parameters: &[eqiora::Id<eqiora_core::entity::kinds::Parameter>],
    points: &[Vec<eqiora::ValueLiteral>],
    backend: Arc<dyn LinearSolverBackend + Send + Sync>,
) -> Result<EvaluationCollection, String> {
    EvaluationMapPlan::from_parameters(
        Arc::new(plan),
        parameters,
        points,
        EvaluationMapExecutionPolicy::retained(LIMIT),
        8_388_608,
        backend,
    )
    .map_err(one)?
    .execute_recorded(|| false)
    .map_err(one)
}
fn perform(matches: &clap::ArgMatches) -> Result<OracleOutcome, String> {
    let backend = backend();
    let (name, args) = matches.subcommand().ok_or("expected study command")?;
    let (saved, execution_complete) = match name {
        "run" => {
            let bytes = read(args.get_one::<OsString>("plan").expect("required"))?;
            let plan =
                ResolvedCommonPlan::from_bytes(&bytes, backend.as_ref(), &[]).map_err(one)?;
            let parameters = args
                .get_many::<String>("parameter")
                .expect("required")
                .map(|id| {
                    id.parse()
                        .map(eqiora::Id::from_ulid)
                        .map_err(|_| "invalid Parameter ID".to_owned())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut charge = 0usize;
            let points = args
                .get_many::<String>("point")
                .expect("required")
                .map(|point| {
                    charge = charge
                        .checked_add(point.len())
                        .ok_or("point inventory size overflow")?;
                    if charge > LIMIT {
                        return Err("point inventory exceeds 67108864 bytes".into());
                    }
                    let values: Vec<WireValueLiteral> =
                        serde_json::from_str(point).map_err(|error| error.to_string())?;
                    values
                        .iter()
                        .map(|value| value.decode().map_err(one))
                        .collect::<Result<Vec<_>, String>>()
                })
                .collect::<Result<Vec<_>, String>>()?;
            let saved = execute(plan, &parameters, &points, backend)?;
            let complete = saved.is_complete();
            (saved, complete)
        }
        "retry" => {
            let original = collection(args, backend.as_ref())?;
            let group = original
                .groups()
                .get(*args.get_one::<usize>("group").expect("required"))
                .ok_or("group index out of range")?;
            let parameters = group
                .parameters()
                .ok_or("retry requires a stationary Parameter group")?;
            let points = group
                .parameter_points()
                .ok_or("retry requires Parameter points")?;
            let unsuccessful = points
                .iter()
                .enumerate()
                .filter(|(index, _)| !group.occurrence(*index).expect("indexed").1.is_accepted())
                .map(|(_, point)| point.clone())
                .collect::<Vec<_>>();
            if unsuccessful.is_empty() {
                return Err("selected group has no unsuccessful occurrences".into());
            }
            let retried = execute(group.plan().clone(), parameters, &unsuccessful, backend)?;
            let complete = retried.is_complete();
            (
                EvaluationCollection::combine(vec![original, retried]).map_err(one)?,
                complete,
            )
        }
        "inspect" => {
            let saved = collection(args, backend.as_ref())?;
            let value = inspect(&saved)?;
            let bytes = serde_json::to_vec_pretty(&value).map_err(|error| error.to_string())?;
            if bytes.len() > LIMIT {
                return Err("study inspection exceeds output bound".into());
            }
            return Ok(OracleOutcome::stdout(0, bytes));
        }
        _ => return Err("unknown study command".into()),
    };
    let bytes = saved.to_bytes().map_err(one)?;
    let path = args.get_one::<OsString>("output").expect("required");
    write_new(path, &bytes)?;
    Ok(OracleOutcome::stdout(
        if execution_complete { 0 } else { 1 },
        format!("{}\n", saved.identity()),
    ))
}

fn write_new(path: &OsString, bytes: &[u8]) -> Result<(), String> {
    if bytes.len() > LIMIT {
        return Err("study output exceeds 67108864 bytes".into());
    }
    // Never replace a historical collection, including when output aliases an input.
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| error.to_string())?;
    if let Err(error) = output.write_all(bytes).and_then(|()| output.sync_all()) {
        drop(output);
        let _ = std::fs::remove_file(path);
        return Err(error.to_string());
    }
    Ok(())
}

pub(super) fn run(matches: &clap::ArgMatches) -> CommandResult {
    Ok(match perform(matches) {
        Ok(outcome) => outcome,
        Err(error) => OracleOutcome::stderr(1, format!("eqiora study: {error}\n")),
    })
}

fn inspect(saved: &EvaluationCollection) -> Result<serde_json::Value, String> {
    use serde_json::json;
    let groups = saved.groups().iter().map(|group| {
        if group.parameters().is_none() { return Err("inspection requires stationary Parameter groups".into()); }
        let members = (0..group.len()).map(|index| {
            let (occurrence, outcome) = group.occurrence(index).expect("indexed");
            let point = group.parameter_points().map(|points| points[index].iter().map(WireValueLiteral::encode).collect::<Result<Vec<_>, _>>()).transpose().map_err(one)?;
            let mut member = json!({"occurrence": occurrence.to_string(), "point": point});
            match outcome {
                RecordedEvaluationOutcome::RunAccepted(result) => {
                    member["status"] = json!("accepted");
                    member["result"] = json!(result.identity());
                    member["plan"] = json!(result.plan().identity());
                    let plan = result.plan();
                    let harmonic = if let Some(plan) = plan.as_algebraic() {
                        plan.harmonic_angular_frequency().map(|omega| (omega,
                            plan.harmonic_amplitudes().map(|(name, original, amplitude)|
                                json!({"name":name,"original":original.ulid().to_string(),"amplitude":amplitude.ulid().to_string()})).collect::<Vec<_>>(),
                            plan.formulation()))
                    } else if let Some(plan) = plan.as_spatial() {
                        plan.harmonic_angular_frequency().map(|omega| (omega,
                            plan.harmonic_amplitudes().map(|(name, original, amplitude)|
                                json!({"name":name,"original":original.ulid().to_string(),"amplitude":amplitude.ulid().to_string()})).collect::<Vec<_>>(),
                            plan.formulation()))
                    } else { None };
                    if let Some((omega, amplitudes, formulation)) = harmonic {
                        member["harmonic"] = json!({"angular_frequency_per_second":omega,
                            "amplitudes":amplitudes,
                            "formulation_rules":formulation.as_ref().map(|form|form.rule_ids())});
                    }
                    let mut fields = Vec::new();
                    if result.plan().as_algebraic().is_some() {
                        for (id, value) in result.finite_field_values().map_err(one)? {
                            fields.push(json!({"field": id.ulid().to_string(), "association":"global", "value": WireValueLiteral::encode(&value).map_err(one)?}));
                        }
                    }
                    for field in 0..result.field_count() {
                        let (id, value_type, space) = result.field(field).expect("indexed");
                        let blocks = (0..result.field_block_count(field)).map(|block| {
                            let (association, coordinates, shape) = result.field_block(field, block).expect("indexed");
                            let width = if value_type.scalar_domain() == eqiora::ScalarDomain::Complex { 2 } else { 1 };
                            let count = eqiora::ValueLiteral::from_real(value_type.clone(), 0.0).map_err(|error| error.to_string())?.component_count();
                            let values = coordinates.chunks_exact(width * count).map(|coefficient| {
                                let components = coefficient.chunks_exact(width).map(|c| (c[0], if width == 2 { c[1] } else { 0.0 }));
                                let literal = eqiora::ValueLiteral::new(value_type.clone(), components).map_err(|error| error.to_string())?;
                                WireValueLiteral::encode(&literal).map_err(one)
                            }).collect::<Result<Vec<_>, String>>()?;
                            Ok(json!({"association":association, "logical_shape":shape, "values":values}))
                        }).collect::<Result<Vec<_>, String>>()?;
                        fields.push(json!({"field":id,"space":format!("{space:?}"),"blocks":blocks}));
                    }
                    member["fields"] = json!(fields);
                }
                RecordedEvaluationOutcome::Accepted(_) => { member["status"] = json!("accepted"); }
                RecordedEvaluationOutcome::Failed(errors) => { member["status"] = json!("failed"); member["diagnostics"] = json!(errors.iter().map(|error| json!({"code":error.code(),"message":error.message()})).collect::<Vec<_>>()); }
                RecordedEvaluationOutcome::Cancelled(_) => member["status"] = json!("cancelled"),
                RecordedEvaluationOutcome::NotStarted => member["status"] = json!("not_started"),
            }
            Ok(member)
        }).collect::<Result<Vec<_>, String>>()?;
        Ok(json!({"execution":group.execution_id(),"complete":group.is_complete(),"parameters":group.parameters().map(|ids| ids.iter().map(|id|id.ulid().to_string()).collect::<Vec<_>>()),"members":members}))
    }).collect::<Result<Vec<_>, String>>()?;
    Ok(json!({"identity":saved.identity(),"complete":saved.is_complete(),"groups":groups}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora::artifact::ModelEnvelope;
    use eqiora_numerics::{CommonAlgebraicPlan, CommonLinearRequest, CommonSolvePolicy};
    use eqiora_solver::{
        LinearSolver, PreconditionerPolicy, REFERENCE_SOLVER_PROVIDER, ReductionPolicy, SolverPlan,
    };
    use std::num::NonZeroUsize;

    #[test]
    fn stationary_commands_preserve_order_typed_values_reopen_and_reject_overwrite() {
        let document = ModelDocument::compile(
            "study.eqi",
            "model M(){parameter p:V=2[V];variable u:V;relation r{u=p;}}",
        )
        .unwrap();
        let parameter = document
            .program()
            .nodes()
            .find_map(|node| match node {
                eqiora_schema::kernel::KernelNode::Parameter(parameter) => Some(parameter.id()),
                _ => None,
            })
            .unwrap();
        let artifact = ModelEnvelope::from_program(document.program()).unwrap();
        let linear_backend = backend();
        let reference = linear_backend.provider() == REFERENCE_SOLVER_PROVIDER;
        let policy = CommonSolvePolicy::Linear(
            CommonLinearRequest::exact(
                SolverPlan::new(
                    if reference {
                        LinearSolver::BiConjugateGradientStabilized
                    } else {
                        LinearSolver::SparseLu
                    },
                    1e-12,
                    1e-14,
                    NonZeroUsize::new(64).unwrap(),
                )
                .unwrap()
                .with_preconditioner(PreconditionerPolicy::Identity)
                .with_reduction(if reference {
                    ReductionPolicy::Reproducible
                } else {
                    ReductionPolicy::Fast
                }),
                linear_backend.provider(),
            )
            .unwrap(),
        );
        let plan = ResolvedCommonPlan::Algebraic(Box::new(
            CommonAlgebraicPlan::resolve(
                &artifact,
                policy,
                None,
                &[],
                None,
                linear_backend.as_ref(),
            )
            .unwrap(),
        ));
        let directory = std::env::temp_dir().join(format!("eqiora-cli-study-{}", parameter.ulid()));
        std::fs::create_dir(&directory).unwrap();
        assert!(read(&directory.clone().into_os_string()).is_err());
        let input = directory.join("input.eqplan");
        let output = directory.join("output.eqcollection");
        std::fs::write(&input, plan.to_bytes().unwrap()).unwrap();
        let value_type = document
            .program()
            .typed_value(parameter.erase())
            .unwrap()
            .value_type()
            .clone();
        let points = [2., 3., 2.].map(|value| {
            serde_json::to_string(&vec![
                WireValueLiteral::encode(
                    &eqiora::ValueLiteral::from_real(value_type.clone(), value).unwrap(),
                )
                .unwrap(),
            ])
            .unwrap()
        });
        let mut arguments = vec![
            OsString::from("study"),
            OsString::from("run"),
            input.into_os_string(),
            "--parameter".into(),
            parameter.ulid().to_string().into(),
            "--output".into(),
            output.clone().into_os_string(),
        ];
        for point in points {
            arguments.extend(["--point".into(), point.into()]);
        }
        let matches = command().try_get_matches_from(arguments).unwrap();
        assert_eq!(perform(&matches).unwrap().0, 0);
        let bytes = std::fs::read(&output).unwrap();
        let reopened = EvaluationCollection::from_bytes(&bytes, backend().as_ref(), &[]).unwrap();
        let group = &reopened.groups()[0];
        assert_eq!(group.len(), 3);
        assert_ne!(
            group.occurrence(0).unwrap().0,
            group.occurrence(2).unwrap().0
        );
        for (index, wanted) in [2., 3., 2.].into_iter().enumerate() {
            let RecordedEvaluationOutcome::RunAccepted(result) = group.occurrence(index).unwrap().1
            else {
                panic!("accepted")
            };
            let values = result.finite_field_values().unwrap();
            assert_eq!(
                values[0].1.components().unwrap().collect::<Vec<_>>(),
                vec![(wanted, 0.)]
            );
        }
        let printed = inspect(&reopened).unwrap();
        assert_eq!(
            printed["groups"][0]["members"][1]["fields"][0]["value"]["components"]["values"],
            serde_json::json!([[3., 0.]])
        );
        assert!(perform(&matches).is_err());
        assert_eq!(std::fs::read(output).unwrap(), bytes);
        let oversized = directory.join("oversized.eqcollection").into_os_string();
        assert!(write_new(&oversized, &vec![0; LIMIT + 1]).is_err());
        assert!(!Path::new(&oversized).exists());
        std::fs::remove_dir_all(directory).unwrap();
    }
}
