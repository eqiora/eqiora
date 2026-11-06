use std::collections::VecDeque;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::*;

mod library_evidence;

struct Fixture {
    root: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            env::temp_dir().join(format!("eqiora-verify-test-{}-{nonce}", std::process::id()));
        fs::create_dir_all(root.join("verify/area/case/models")).unwrap();
        fs::create_dir_all(root.join("verify/area/case/references")).unwrap();
        fs::create_dir_all(root.join("verify/area/case/expected")).unwrap();
        fs::write(root.join("verify/area/case/README.md"), "case\n").unwrap();
        fs::write(root.join("verify/area/case/expected/evidence.csv"), "x,y\n").unwrap();
        fs::create_dir_all(root.join("tools/ci")).unwrap();
        fs::write(
            root.join("tools/ci/python_evidence.py"),
            "print('evidence')\n",
        )
        .unwrap();
        fs::create_dir_all(root.join("eqiora-numerics/src")).unwrap();
        fs::create_dir_all(root.join("eqiora-numerics/tests")).unwrap();
        fs::write(
            root.join("Cargo.toml"),
            "[workspace]\nmembers = [\"eqiora-numerics\"]\nresolver = \"2\"\n",
        )
        .unwrap();
        fs::write(
            root.join("eqiora-numerics/Cargo.toml"),
            "[package]\nname = \"eqiora-numerics\"\nversion = \"0.0.0\"\nedition = \"2024\"\n",
        )
        .unwrap();
        fs::write(root.join("eqiora-numerics/src/lib.rs"), "").unwrap();
        fs::write(
            root.join("eqiora-numerics/tests/evidence_test.rs"),
            "#[test]\nfn evidence_test() {}\n",
        )
        .unwrap();
        Self { root }
    }

    fn write_manifest(&self, source: &str) {
        fs::write(self.root.join("verify/area/case/case.toml"), source).unwrap();
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).unwrap();
    }
}

fn targets() -> BTreeMap<String, BTreeSet<String>> {
    BTreeMap::from([(
        "eqiora-numerics".to_owned(),
        BTreeSet::from(["evidence_test".to_owned()]),
    )])
}

fn valid_manifest() -> &'static str {
    r#"
id = "area.case"
status = "verified"
reference_kind = "analytic"
capabilities = ["convergence", "conservation"]
conformance_kits = ["scalar-convergence-v1"]

[evidence]
package = "eqiora-numerics"
test = "evidence_test"
features = ["evidence-runtime"]
table = "expected/evidence.csv"
"#
}

fn valid_python_manifest() -> &'static str {
    r#"
id = "area.case"
status = "verified"
reference_kind = "framework-contract"
capabilities = ["python-autograd"]

[evidence]
runner = "python-installed-wheel"
script = "tools/ci/python_evidence.py"
"#
}

#[test]
fn repository_contract_rejects_duplicate_index_keys_and_missing_artifact() {
    let fixture = Fixture::new();
    fixture.write_manifest(&valid_manifest().replace(
        "[\"convergence\", \"conservation\"]",
        "[\"convergence\", \"convergence\"]",
    ));
    let error = load_repository_with_targets(&fixture.root, &targets()).unwrap_err();
    assert!(error.contains("repeats capability"));

    fixture.write_manifest(&valid_manifest().replace(
        "[\"scalar-convergence-v1\"]",
        "[\"scalar-convergence-v1\", \"scalar-convergence-v1\"]",
    ));
    let error = load_repository_with_targets(&fixture.root, &targets()).unwrap_err();
    assert!(error.contains("repeats conformance kit"));

    fixture
        .write_manifest(&valid_manifest().replace("expected/evidence.csv", "expected/missing.csv"));
    let error = load_repository_with_targets(&fixture.root, &targets()).unwrap_err();
    assert!(error.contains("missing or inaccessible"));
}

#[test]
fn unknown_status_and_arbitrary_evidence_command_are_rejected() {
    assert!(
        toml::from_str::<CaseManifest>(&valid_manifest().replace("verified", "unknown-status"))
            .is_err()
    );
    assert!(
        toml::from_str::<CaseManifest>(&format!(
            "{}\ncommand = \"sh -c anything\"\n",
            valid_manifest()
        ))
        .is_err()
    );
    assert!(
        toml::from_str::<CaseManifest>(&format!(
            "{}\nargs = [\"--arbitrary\"]\n",
            valid_python_manifest()
        ))
        .is_err()
    );
    assert!(
        toml::from_str::<CaseManifest>(&valid_manifest().replace(
            "features = [\"evidence-runtime\"]",
            "features = [\"evidence-runtime\"]\nenvironment = \"unknown\"",
        ))
        .is_err()
    );
}

#[test]
fn installed_wheel_python_target_is_closed_and_repository_owned() {
    let fixture = Fixture::new();
    fixture.write_manifest(valid_python_manifest());
    let contracts = load_repository_with_targets(&fixture.root, &targets()).unwrap();
    assert!(matches!(
        contracts[0].evidence,
        Some(EvidenceTarget::PythonInstalledWheel(
            PythonInstalledWheelEvidenceTarget {
                runner: PythonEvidenceRunner::PythonInstalledWheel,
                ref script,
                ..
            }
        )) if script == "tools/ci/python_evidence.py"
    ));

    for invalid in [
        "../python_evidence.py",
        "/tmp/python_evidence.py",
        "tools/ci/python_evidence",
        "tools/ci/python_evidence.txt",
    ] {
        fixture.write_manifest(
            &valid_python_manifest().replace("tools/ci/python_evidence.py", invalid),
        );
        let error = load_repository_with_targets(&fixture.root, &targets()).unwrap_err();
        assert!(
            error.contains(
                "Python evidence script must be a normalized repository-relative `.py` path"
            ),
            "{error}"
        );
    }

    fixture.write_manifest(
        &valid_python_manifest().replace("tools/ci/python_evidence.py", "tools/ci/missing.py"),
    );
    let error = load_repository_with_targets(&fixture.root, &targets()).unwrap_err();
    assert!(error.contains("missing or inaccessible"), "{error}");

    assert!(
        toml::from_str::<CaseManifest>(
            &valid_python_manifest()
                .replace("runner = \"python-installed-wheel\"", "runner = \"cargo\"")
        )
        .is_err()
    );
}

#[cfg(unix)]
#[test]
fn installed_wheel_python_target_rejects_a_script_symlink() {
    use std::os::unix::fs::symlink;

    let fixture = Fixture::new();
    symlink(
        fixture.root.join("tools/ci/python_evidence.py"),
        fixture.root.join("tools/ci/python_evidence_link.py"),
    )
    .unwrap();
    fixture.write_manifest(&valid_python_manifest().replace(
        "tools/ci/python_evidence.py",
        "tools/ci/python_evidence_link.py",
    ));
    let error = load_repository_with_targets(&fixture.root, &targets()).unwrap_err();
    assert!(error.contains("not a regular repository file"), "{error}");
}

#[test]
fn target_serialization_preserves_the_existing_cargo_json_shape() {
    assert_eq!(REPORT_SCHEMA, "eqiora.verification-report/v6");
    assert_eq!(
        CAPABILITY_INDEX_SCHEMA,
        "eqiora.capability-evidence-index/v3"
    );
    let target = EvidenceTarget::Cargo(CargoEvidenceTarget {
        package: "eqiora".to_owned(),
        test: "evidence_test".to_owned(),
        features: Vec::new(),
        table: None,
        environment: EvidenceEnvironment::HostCpu,
    });
    assert_eq!(
        serde_json::to_value(target).unwrap(),
        serde_json::json!({
            "package": "eqiora",
            "test": "evidence_test"
        })
    );

    let physical = EvidenceTarget::Cargo(CargoEvidenceTarget {
        package: "eqiora".to_owned(),
        test: "physical_test".to_owned(),
        features: vec!["mpi-cuda".to_owned()],
        table: None,
        environment: EvidenceEnvironment::PhysicalMpiCuda,
    });
    assert_eq!(
        serde_json::to_value(physical).unwrap(),
        serde_json::json!({
            "package": "eqiora",
            "test": "physical_test",
            "features": ["mpi-cuda"],
            "environment": "physical-mpi-cuda"
        })
    );
}

#[test]
fn canonical_repository_is_sorted_by_case_id() {
    let fixture = Fixture::new();
    fixture.write_manifest(valid_manifest());
    for (area, case) in [("zeta", "last"), ("alpha", "first")] {
        let directory = fixture.root.join(format!("verify/{area}/{case}"));
        fs::create_dir_all(directory.join("models")).unwrap();
        fs::create_dir_all(directory.join("references")).unwrap();
        fs::create_dir_all(directory.join("expected")).unwrap();
        fs::write(directory.join("README.md"), "case\n").unwrap();
        fs::write(
            directory.join("case.toml"),
            format!(
                "id = \"{area}.{case}\"\nstatus = \"specified\"\nreference_kind = \"analytic\"\ncapabilities = [\"one\"]\n"
            ),
        )
        .unwrap();
    }
    let contracts = load_repository_with_targets(&fixture.root, &targets()).unwrap();
    let ids = contracts
        .into_iter()
        .map(|contract| contract.id)
        .collect::<Vec<_>>();
    assert_eq!(ids, ["alpha.first", "area.case", "zeta.last"]);
}

struct FakeRunner {
    outputs: Mutex<VecDeque<EvidenceOutput>>,
    target_outputs: Mutex<Vec<(EvidenceTarget, EvidenceOutput)>>,
    targets: Mutex<Vec<EvidenceTarget>>,
    groups: Mutex<Vec<Vec<EvidenceTarget>>>,
    build_failures: Mutex<Vec<(CargoBuildGroup, EvidenceOutput)>>,
    delays: Mutex<Vec<(EvidenceTarget, Duration)>>,
    completions: Mutex<Vec<EvidenceTarget>>,
}

impl FakeRunner {
    fn new(outputs: impl IntoIterator<Item = EvidenceOutput>) -> Self {
        Self {
            outputs: Mutex::new(outputs.into_iter().collect()),
            target_outputs: Mutex::new(Vec::new()),
            targets: Mutex::new(Vec::new()),
            groups: Mutex::new(Vec::new()),
            build_failures: Mutex::new(Vec::new()),
            delays: Mutex::new(Vec::new()),
            completions: Mutex::new(Vec::new()),
        }
    }

    fn for_targets(outputs: impl IntoIterator<Item = (EvidenceTarget, EvidenceOutput)>) -> Self {
        Self {
            outputs: Mutex::new(VecDeque::new()),
            target_outputs: Mutex::new(outputs.into_iter().collect()),
            targets: Mutex::new(Vec::new()),
            groups: Mutex::new(Vec::new()),
            build_failures: Mutex::new(Vec::new()),
            delays: Mutex::new(Vec::new()),
            completions: Mutex::new(Vec::new()),
        }
    }

    fn with_build_failure(mut self, group: CargoBuildGroup, output: EvidenceOutput) -> Self {
        self.build_failures.get_mut().unwrap().push((group, output));
        self
    }

    fn with_delay(mut self, target: EvidenceTarget, delay: Duration) -> Self {
        self.delays.get_mut().unwrap().push((target, delay));
        self
    }

    fn targets(&self) -> Vec<EvidenceTarget> {
        self.targets.lock().unwrap().clone()
    }

    fn groups(&self) -> Vec<Vec<EvidenceTarget>> {
        self.groups.lock().unwrap().clone()
    }

    fn completions(&self) -> Vec<EvidenceTarget> {
        self.completions.lock().unwrap().clone()
    }
}

impl EvidenceRunner for FakeRunner {
    fn build_cargo_group(
        &self,
        _root: &Path,
        targets: &[EvidenceTarget],
    ) -> Option<EvidenceOutput> {
        self.groups.lock().unwrap().push(targets.to_vec());
        let Some(EvidenceTarget::Cargo(target)) = targets.first() else {
            panic!("fake Cargo group is non-empty and Cargo-only");
        };
        let group = CargoBuildGroup::from_target(target);
        self.build_failures
            .lock()
            .unwrap()
            .iter()
            .find(|(failed, _)| failed == &group)
            .map(|(_, output)| output.clone())
    }

    fn run(&self, _root: &Path, target: &EvidenceTarget) -> EvidenceOutput {
        self.targets.lock().unwrap().push(target.clone());
        let delay = self
            .delays
            .lock()
            .unwrap()
            .iter()
            .find(|(delayed, _)| delayed == target)
            .map(|(_, delay)| *delay);
        if let Some(delay) = delay {
            thread::sleep(delay);
        }
        let output = if let Some(output) = self
            .target_outputs
            .lock()
            .unwrap()
            .iter()
            .find(|(mapped, _)| mapped == target)
            .map(|(_, output)| output.clone())
        {
            output
        } else {
            self.outputs.lock().unwrap().pop_front().unwrap()
        };
        self.completions.lock().unwrap().push(target.clone());
        output
    }
}

fn successful_output() -> EvidenceOutput {
    EvidenceOutput {
        duration_ms: Some(7),
        exit_code: Some(0),
        stdout: String::new(),
        stderr: String::new(),
        start_error: None,
    }
}

fn cargo_target() -> EvidenceTarget {
    EvidenceTarget::Cargo(CargoEvidenceTarget {
        package: "eqiora-numerics".to_owned(),
        test: "evidence_test".to_owned(),
        features: vec!["evidence-runtime".to_owned()],
        table: None,
        environment: EvidenceEnvironment::HostCpu,
    })
}

fn python_target() -> EvidenceTarget {
    EvidenceTarget::PythonInstalledWheel(PythonInstalledWheelEvidenceTarget {
        runner: PythonEvidenceRunner::PythonInstalledWheel,
        script: "tools/ci/python_evidence.py".to_owned(),
        environment: EvidenceEnvironment::HostCpu,
    })
}

fn contract_with_target(id: &str, evidence: EvidenceTarget) -> CaseContract {
    CaseContract {
        id: id.to_owned(),
        manifest: format!("verify/area/{id}/case.toml"),
        status: Status::Verified,
        reference_kind: "analytic".to_owned(),
        capabilities: vec!["convergence".to_owned()],
        conformance_kits: vec!["scalar-convergence-v1".to_owned()],
        evidence: Some(evidence),
    }
}

fn contract(id: &str) -> CaseContract {
    contract_with_target(id, cargo_target())
}

fn named_cargo_target(package: &str, test: &str, features: &[&str]) -> EvidenceTarget {
    EvidenceTarget::Cargo(CargoEvidenceTarget {
        package: package.to_owned(),
        test: test.to_owned(),
        features: features
            .iter()
            .map(|feature| (*feature).to_owned())
            .collect(),
        table: None,
        environment: EvidenceEnvironment::HostCpu,
    })
}

fn output(exit_code: i32, label: &str) -> EvidenceOutput {
    EvidenceOutput {
        duration_ms: Some(u64::try_from(label.len()).unwrap()),
        exit_code: Some(exit_code),
        stdout: format!("{label}-out"),
        stderr: format!("{label}-err"),
        start_error: None,
    }
}

fn legacy_execute_selected(
    root: &Path,
    selected: Vec<CaseContract>,
    request: &Request,
    runner: &dyn EvidenceRunner,
) -> Vec<CaseReport> {
    let mut cases = Vec::with_capacity(selected.len());
    let mut completed_targets = Vec::<(ExecutionKey, EvidenceOutput)>::new();
    let mut prior_failure = false;
    for contract in selected {
        let mut case = CaseReport::from_contract(&contract);
        match request.command {
            CommandKind::List => case.outcome = Outcome::Listed,
            CommandKind::Check => case.outcome = Outcome::Checked,
            CommandKind::Run if !contract.status.is_executable() => {
                case.outcome = Outcome::NotRunnable;
                case.message = Some("case status does not declare executable evidence".to_owned());
            }
            CommandKind::Run
                if request.environment.is_some_and(|selected| {
                    contract
                        .evidence
                        .as_ref()
                        .is_some_and(|target| target.environment() != selected)
                }) =>
            {
                let required = contract
                    .evidence
                    .as_ref()
                    .expect("executable contracts were validated with evidence")
                    .environment();
                case.outcome = Outcome::NotSelected;
                case.message = Some(format!(
                    "evidence requires `{}` execution",
                    required.as_str()
                ));
            }
            CommandKind::Run => {
                let target = contract
                    .evidence
                    .as_ref()
                    .expect("executable contracts were validated with evidence");
                let key = ExecutionKey::from_target(target);
                let completed = completed_targets
                    .iter()
                    .position(|(completed, _)| completed == &key);
                if completed.is_none()
                    && prior_failure
                    && request.policy == ExecutionPolicy::FailFast
                {
                    case.outcome = Outcome::Skipped;
                    case.message = Some("not run after fail-fast evidence failure".to_owned());
                } else {
                    let completed = completed.unwrap_or_else(|| {
                        completed_targets.push((key, runner.run(root, target)));
                        completed_targets.len() - 1
                    });
                    if !project_evidence_output(&mut case, &completed_targets[completed].1) {
                        prior_failure = true;
                    }
                }
            }
        }
        cases.push(case);
    }
    cases
}

fn report_for(request: &Request, cases: Vec<CaseReport>) -> VerificationReport {
    VerificationReport {
        schema: REPORT_SCHEMA,
        command: request.command,
        policy: request.policy,
        selected_cases: request.cases.clone(),
        selected_environment: request.environment,
        selected_runner_kind: request.runner_kind,
        success: cases.iter().all(|case| case.outcome != Outcome::Failed),
        cases,
        errors: Vec::new(),
    }
}

fn stderr_without_cargo_progress(stderr: Option<&str>) -> Option<Vec<&str>> {
    stderr.map(|stderr| {
        stderr
            .lines()
            .filter(|line| {
                let line = line.trim_start();
                !["Compiling ", "Finished ", "Running "]
                    .iter()
                    .any(|prefix| line.starts_with(prefix))
            })
            .collect()
    })
}

#[test]
fn environment_selection_is_explicit_and_does_not_weaken_full_execution() {
    let mut physical = contract("physical");
    let Some(EvidenceTarget::Cargo(target)) = physical.evidence.as_mut() else {
        panic!("fixture has Cargo evidence");
    };
    target.environment = EvidenceEnvironment::PhysicalMpiCuda;

    let selected = execute_selected(
        Path::new("."),
        vec![physical.clone(), contract("portable")],
        &Request::new(CommandKind::Run, Vec::new(), ExecutionPolicy::FailFast)
            .for_environment(EvidenceEnvironment::HostCpu),
        &FakeRunner::new([successful_output()]),
    );
    assert_eq!(
        selected.iter().map(|case| case.outcome).collect::<Vec<_>>(),
        [Outcome::NotSelected, Outcome::Passed]
    );
    assert_eq!(
        selected[0].message.as_deref(),
        Some("evidence requires `physical-mpi-cuda` execution")
    );

    let full = execute_selected(
        Path::new("."),
        vec![physical],
        &Request::new(CommandKind::Run, Vec::new(), ExecutionPolicy::FailFast),
        &FakeRunner::new([EvidenceOutput {
            duration_ms: Some(11),
            exit_code: Some(29),
            stdout: String::new(),
            stderr: String::new(),
            start_error: None,
        }]),
    );
    assert_eq!(full[0].outcome, Outcome::Failed);
}

#[test]
fn capability_index_is_manifest_derived_sorted_and_exactly_filterable() {
    let fixture = Fixture::new();
    fixture.write_manifest(valid_manifest());
    let contracts = load_repository_with_targets(&fixture.root, &targets()).unwrap();

    let all = build_capability_evidence_index(contracts.clone(), None);
    assert!(all.success);
    assert_eq!(
        all.entries
            .iter()
            .map(|entry| entry.capability.as_str())
            .collect::<Vec<_>>(),
        ["conservation", "convergence"]
    );
    assert!(all.entries.iter().all(|entry| {
        entry.case == "area.case"
            && entry.reference_kind == "analytic"
            && entry.conformance_kits == ["scalar-convergence-v1"]
            && entry.evidence.as_ref().is_some_and(|evidence| {
                matches!(
                    evidence,
                    EvidenceTarget::Cargo(CargoEvidenceTarget { package, test, .. })
                        if package == "eqiora-numerics" && test == "evidence_test"
                )
            })
    }));

    let selected = build_capability_evidence_index(contracts.clone(), Some("convergence"));
    assert!(selected.success);
    assert_eq!(selected.entries.len(), 1);
    assert_eq!(selected.entries[0].capability, "convergence");

    let unknown = build_capability_evidence_index(contracts, Some("not-declared"));
    assert!(!unknown.success);
    assert!(unknown.entries.is_empty());
    assert_eq!(
        unknown.errors,
        ["unknown verification capability `not-declared`"]
    );
}

#[test]
fn identical_targets_execute_once_and_preserve_distinct_case_reports() {
    let runner = FakeRunner::new([EvidenceOutput {
        duration_ms: Some(37),
        exit_code: Some(0),
        stdout: "shared-out".to_owned(),
        stderr: "shared-err".to_owned(),
        start_error: None,
    }]);
    let reports = execute_selected(
        Path::new("."),
        vec![contract("a"), contract("b")],
        &Request::new(CommandKind::Run, Vec::new(), ExecutionPolicy::KeepGoing),
        &runner,
    );

    assert_eq!(runner.targets(), [cargo_target()]);
    assert_eq!(
        reports
            .iter()
            .map(|case| (case.id.as_str(), case.outcome))
            .collect::<Vec<_>>(),
        [("a", Outcome::Passed), ("b", Outcome::Passed)]
    );
    assert_eq!(reports[0].stdout.as_deref(), Some("shared-out"));
    assert_eq!(reports[1].stdout.as_deref(), Some("shared-out"));
    assert_eq!(reports[0].stderr.as_deref(), Some("shared-err"));
    assert_eq!(reports[1].stderr.as_deref(), Some("shared-err"));
    assert_eq!(reports[0].duration_ms, Some(37));
    assert_eq!(reports[1].duration_ms, Some(37));

    let report = VerificationReport {
        schema: REPORT_SCHEMA,
        command: CommandKind::Run,
        policy: ExecutionPolicy::KeepGoing,
        selected_cases: Vec::new(),
        selected_environment: None,
        selected_runner_kind: None,
        success: true,
        cases: reports,
        errors: Vec::new(),
    };
    assert!(report.render_human().contains("a [verified; 37 ms]"));
    let json = serde_json::to_value(report).unwrap();
    assert_eq!(json["cases"][0]["duration_ms"], 37);
    assert_eq!(json["cases"][1]["duration_ms"], 37);
}

#[test]
fn six_claims_over_two_execution_keys_run_twice_and_build_once() {
    let first = named_cargo_target("package-a", "first", &["feature-a"]);
    let second = named_cargo_target("package-a", "second", &["feature-a"]);
    let mut first_with_table = first.clone();
    let EvidenceTarget::Cargo(first_details) = &mut first_with_table else {
        unreachable!();
    };
    first_details.table = Some("expected/first.csv".to_owned());

    let mut second_reordered_metadata = contract_with_target("f", second.clone());
    second_reordered_metadata.manifest = "verify/other/f/case.toml".to_owned();
    second_reordered_metadata.reference_kind = "independent-numerical".to_owned();
    second_reordered_metadata.capabilities = vec!["other-capability".to_owned()];
    second_reordered_metadata.conformance_kits.clear();

    let runner = FakeRunner::for_targets([
        (first.clone(), output(0, "first")),
        (second.clone(), output(0, "second")),
    ]);
    let reports = execute_selected(
        Path::new("."),
        vec![
            contract_with_target("a", first.clone()),
            contract_with_target("b", first_with_table),
            contract_with_target("c", first),
            contract_with_target("d", second.clone()),
            contract_with_target("e", second),
            second_reordered_metadata,
        ],
        &Request::new(CommandKind::Run, Vec::new(), ExecutionPolicy::KeepGoing),
        &runner,
    );

    assert_eq!(runner.targets().len(), 2);
    assert_eq!(runner.groups().len(), 1);
    assert_eq!(runner.groups()[0].len(), 2);
    assert_eq!(reports.len(), 6);
    assert!(reports.iter().all(|case| case.outcome == Outcome::Passed));
}

#[test]
fn request_selection_is_canonical_and_reports_every_unknown_id_before_execution() {
    let first = contract_with_target("a", cargo_target());
    let second = contract_with_target("b", python_target());
    let requests = [
        Request::new(
            CommandKind::Run,
            vec!["b".to_owned(), "a".to_owned(), "b".to_owned()],
            ExecutionPolicy::KeepGoing,
        ),
        Request::new(
            CommandKind::Run,
            vec!["a".to_owned(), "b".to_owned()],
            ExecutionPolicy::KeepGoing,
        ),
    ];
    let reports = requests.each_ref().map(|request| {
        let selected = select_cases(vec![first.clone(), second.clone()], &request.cases).unwrap();
        report_for(
            request,
            execute_selected(
                Path::new("."),
                selected,
                request,
                &FakeRunner::for_targets([
                    (cargo_target(), output(0, "cargo")),
                    (python_target(), output(0, "python")),
                ]),
            ),
        )
    });
    assert_eq!(requests[0].cases, ["a", "b"]);
    assert_eq!(requests[0], requests[1]);
    assert_eq!(
        serde_json::to_string(&reports[0]).unwrap(),
        serde_json::to_string(&reports[1]).unwrap()
    );
    let json = serde_json::to_value(&reports[0]).unwrap();
    assert_eq!(json["schema"], "eqiora.verification-report/v6");
    assert_eq!(json["selected_cases"], serde_json::json!(["a", "b"]));
    assert!(json.get("selected_case").is_none());

    let fixture = Fixture::new();
    fixture.write_manifest(valid_manifest());
    let runner = FakeRunner::new([]);
    let report = execute(
        &fixture.root,
        &Request::new(
            CommandKind::Run,
            vec!["z.missing".to_owned(), "a.missing".to_owned()],
            ExecutionPolicy::FailFast,
        ),
        &runner,
    );
    assert_eq!(
        report.errors,
        ["unknown verification case ID(s): `a.missing`, `z.missing`"]
    );
    assert!(report.cases.is_empty());
    assert!(runner.groups().is_empty());
    assert!(runner.targets().is_empty());
}

#[test]
fn current_registry_collision_counts_match_the_frozen_execution_contract() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .unwrap();
    let contracts = load_repository(root).unwrap();
    let mut selecting_cases = BTreeMap::<ExecutionKey, Vec<String>>::new();
    for contract in contracts
        .iter()
        .filter(|contract| contract.status.is_executable())
    {
        let target = contract.evidence.as_ref().unwrap();
        selecting_cases
            .entry(ExecutionKey::from_target(target))
            .or_default()
            .push(contract.id.clone());
    }

    let count_for_script = |script: &str| {
        selecting_cases
            .iter()
            .filter_map(|(key, cases)| match key {
                ExecutionKey::PythonInstalledWheel {
                    script: candidate, ..
                } if candidate == script => Some(cases.len()),
                _ => None,
            })
            .sum::<usize>()
    };
    assert_eq!(count_for_script("tools/ci/python_distribution_gate.py"), 4);
    assert_eq!(count_for_script("tools/ci/python_package_gate.py"), 3);

    let mut distribution_cases = selecting_cases
        .iter()
        .filter_map(|(key, cases)| match key {
            ExecutionKey::PythonInstalledWheel { script, .. }
                if script == "tools/ci/python_distribution_gate.py" =>
            {
                Some(cases.iter().map(String::as_str))
            }
            _ => None,
        })
        .flatten()
        .collect::<Vec<_>>();
    distribution_cases.sort_unstable();
    assert_eq!(
        distribution_cases,
        [
            "interfaces.python-distribution-candidate",
            "interfaces.python-jax-differentiation",
            "interfaces.python-mixed-boundary-elasticity-demo",
            "interfaces.python-pytorch-differentiation",
        ]
    );

    let pair = [
        "fsi.fixed-reference-monolithic-step-2d",
        "numerics.physics-neutral-discrete-block-system",
    ];
    assert!(selecting_cases.values().any(|cases| {
        pair.iter()
            .all(|expected| cases.iter().any(|case| case == expected))
    }));
}

#[test]
fn non_executed_outcomes_omit_duration() {
    let runner = FakeRunner::new([]);
    let listed = execute_selected(
        Path::new("."),
        vec![contract("listed")],
        &Request::new(CommandKind::List, Vec::new(), ExecutionPolicy::FailFast),
        &runner,
    );
    let checked = execute_selected(
        Path::new("."),
        vec![contract("checked")],
        &Request::new(CommandKind::Check, Vec::new(), ExecutionPolicy::FailFast),
        &runner,
    );

    let mut not_runnable_contract = contract("not-runnable");
    not_runnable_contract.status = Status::Specified;
    let not_runnable = execute_selected(
        Path::new("."),
        vec![not_runnable_contract],
        &Request::new(CommandKind::Run, Vec::new(), ExecutionPolicy::FailFast),
        &runner,
    );

    let mut physical_contract = contract("not-selected");
    let Some(EvidenceTarget::Cargo(target)) = physical_contract.evidence.as_mut() else {
        panic!("fixture has Cargo evidence");
    };
    target.environment = EvidenceEnvironment::PhysicalMpiCuda;
    let not_selected = execute_selected(
        Path::new("."),
        vec![physical_contract],
        &Request::new(CommandKind::Run, Vec::new(), ExecutionPolicy::FailFast)
            .for_environment(EvidenceEnvironment::HostCpu),
        &runner,
    );
    assert!(runner.targets().is_empty());

    let mut distinct = cargo_target();
    let EvidenceTarget::Cargo(target) = &mut distinct else {
        panic!("fixture has Cargo evidence");
    };
    target.test = "distinct_test".to_owned();
    let fail_fast_runner = FakeRunner::new([EvidenceOutput {
        duration_ms: Some(19),
        exit_code: Some(17),
        stdout: String::new(),
        stderr: String::new(),
        start_error: None,
    }]);
    let fail_fast = execute_selected(
        Path::new("."),
        vec![
            contract("failed"),
            contract_with_target("skipped", distinct),
        ],
        &Request::new(CommandKind::Run, Vec::new(), ExecutionPolicy::FailFast),
        &fail_fast_runner,
    );

    let reports = [
        &listed[0],
        &checked[0],
        &not_runnable[0],
        &not_selected[0],
        &fail_fast[1],
    ];
    assert_eq!(
        reports.map(|case| case.outcome),
        [
            Outcome::Listed,
            Outcome::Checked,
            Outcome::NotRunnable,
            Outcome::NotSelected,
            Outcome::Skipped,
        ]
    );
    for case in reports {
        assert_eq!(case.duration_ms, None);
        let json = serde_json::to_value(case).unwrap();
        assert_eq!(json["duration_ms"], serde_json::Value::Null, "{json}");
    }
}

#[test]
fn execution_identity_excludes_claim_only_fields_and_normalizes_features() {
    let cargo_base = CargoEvidenceTarget {
        package: "package-a".to_owned(),
        test: "test-a".to_owned(),
        features: vec!["feature-a".to_owned(), "feature-b".to_owned()],
        table: None,
        environment: EvidenceEnvironment::HostCpu,
    };
    let mut targets = vec![EvidenceTarget::Cargo(cargo_base.clone())];
    for target in [
        CargoEvidenceTarget {
            package: "package-b".to_owned(),
            ..cargo_base.clone()
        },
        CargoEvidenceTarget {
            test: "test-b".to_owned(),
            ..cargo_base.clone()
        },
        CargoEvidenceTarget {
            features: vec!["feature-b".to_owned(), "feature-a".to_owned()],
            ..cargo_base.clone()
        },
        CargoEvidenceTarget {
            table: Some("expected/evidence.csv".to_owned()),
            ..cargo_base.clone()
        },
        CargoEvidenceTarget {
            environment: EvidenceEnvironment::PhysicalMpiCuda,
            ..cargo_base
        },
    ] {
        targets.push(EvidenceTarget::Cargo(target));
    }

    let python_base = PythonInstalledWheelEvidenceTarget {
        runner: PythonEvidenceRunner::PythonInstalledWheel,
        script: "tools/ci/python_a.py".to_owned(),
        environment: EvidenceEnvironment::HostCpu,
    };
    targets.extend([
        EvidenceTarget::PythonInstalledWheel(python_base.clone()),
        EvidenceTarget::PythonInstalledWheel(PythonInstalledWheelEvidenceTarget {
            script: "tools/ci/python_b.py".to_owned(),
            ..python_base.clone()
        }),
        EvidenceTarget::PythonInstalledWheel(PythonInstalledWheelEvidenceTarget {
            environment: EvidenceEnvironment::PhysicalMpiCuda,
            ..python_base
        }),
    ]);

    let contracts = targets
        .iter()
        .enumerate()
        .map(|(index, target)| contract_with_target(&format!("case-{index}"), target.clone()))
        .collect();
    let unique_targets = [0, 1, 2, 5, 6, 7, 8]
        .map(|index| targets[index].clone())
        .to_vec();
    let runner = FakeRunner::new((0..unique_targets.len()).map(|_| successful_output()));
    let reports = execute_selected(
        Path::new("."),
        contracts,
        &Request::new(CommandKind::Run, Vec::new(), ExecutionPolicy::KeepGoing),
        &runner,
    );

    assert!(reports.iter().all(|case| case.outcome == Outcome::Passed));
    assert_eq!(runner.targets(), unique_targets);

    let keys = targets
        .iter()
        .map(ExecutionKey::from_target)
        .collect::<Vec<_>>();
    assert_eq!(keys[0], keys[3]);
    assert_eq!(keys[0], keys[4]);
    for index in [1, 2, 5, 6, 7, 8] {
        assert_ne!(keys[0], keys[index]);
        assert_ne!(keys[0].label(), keys[index].label());
    }
    assert!(
        keys[0]
            .label()
            .contains("features=2:[9:feature-a,9:feature-b]")
    );
}

#[test]
fn shared_failure_forms_are_projected_without_hiding_affected_cases() {
    for output in [
        EvidenceOutput {
            duration_ms: Some(23),
            exit_code: Some(17),
            stdout: "exit-out".to_owned(),
            stderr: "exit-err".to_owned(),
            start_error: None,
        },
        EvidenceOutput {
            duration_ms: Some(29),
            exit_code: None,
            stdout: "signal-out".to_owned(),
            stderr: "signal-err".to_owned(),
            start_error: None,
        },
        EvidenceOutput {
            duration_ms: None,
            exit_code: None,
            stdout: String::new(),
            stderr: String::new(),
            start_error: Some("cannot start evidence target: unavailable".to_owned()),
        },
    ] {
        let runner = FakeRunner::new([output.clone()]);
        let reports = execute_selected(
            Path::new("."),
            vec![contract("a"), contract("b")],
            &Request::new(CommandKind::Run, Vec::new(), ExecutionPolicy::KeepGoing),
            &runner,
        );

        assert_eq!(runner.targets(), [cargo_target()]);
        assert!(reports.iter().all(|case| case.outcome == Outcome::Failed));
        for report in reports {
            assert_eq!(report.duration_ms, output.duration_ms);
            assert_eq!(report.exit_code, output.exit_code);
            assert_eq!(report.stdout.as_deref(), Some(output.stdout.as_str()));
            assert_eq!(report.stderr.as_deref(), Some(output.stderr.as_str()));
            assert_eq!(
                report.message,
                output.start_error.clone().or_else(|| Some(format!(
                    "evidence target exited with {}",
                    output
                        .exit_code
                        .map_or_else(|| "no exit code".to_owned(), |code| code.to_string())
                )))
            );
            let json = serde_json::to_value(report).unwrap();
            match output.duration_ms {
                Some(duration_ms) => assert_eq!(json["duration_ms"], duration_ms),
                None => assert_eq!(json["duration_ms"], serde_json::Value::Null, "{json}"),
            }
        }
    }
}

#[test]
fn fail_fast_projects_completed_targets_and_skips_only_unseen_targets() {
    let failure = EvidenceOutput {
        duration_ms: Some(31),
        exit_code: Some(17),
        stdout: "child-out".to_owned(),
        stderr: "child-err".to_owned(),
        start_error: None,
    };
    let mut distinct = cargo_target();
    let EvidenceTarget::Cargo(target) = &mut distinct else {
        panic!("fixture has Cargo evidence");
    };
    target.test = "distinct_test".to_owned();
    let cases = vec![
        contract("a"),
        contract("b"),
        contract_with_target("c", distinct.clone()),
        contract("d"),
    ];

    let fail_fast_runner = FakeRunner::new([failure.clone()]);
    let fail_fast = execute_selected(
        Path::new("."),
        cases.clone(),
        &Request::new(CommandKind::Run, Vec::new(), ExecutionPolicy::FailFast),
        &fail_fast_runner,
    );
    assert_eq!(fail_fast_runner.targets(), [cargo_target()]);
    assert_eq!(
        fail_fast
            .iter()
            .map(|case| case.outcome)
            .collect::<Vec<_>>(),
        [
            Outcome::Failed,
            Outcome::Failed,
            Outcome::Skipped,
            Outcome::Failed
        ]
    );
    assert_eq!(
        fail_fast[2].message.as_deref(),
        Some("not run after fail-fast evidence failure")
    );

    let keep_going_runner = FakeRunner::new([failure, successful_output()]);
    let keep_going = execute_selected(
        Path::new("."),
        cases,
        &Request::new(CommandKind::Run, Vec::new(), ExecutionPolicy::KeepGoing),
        &keep_going_runner,
    );
    assert_eq!(keep_going_runner.targets(), [cargo_target(), distinct]);
    assert_eq!(
        keep_going
            .iter()
            .map(|case| case.outcome)
            .collect::<Vec<_>>(),
        [
            Outcome::Failed,
            Outcome::Failed,
            Outcome::Passed,
            Outcome::Failed
        ]
    );
}

#[test]
fn jobs_one_preserves_full_registry_semantics_without_cargo_progress_lines() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .unwrap();
    let contracts = load_repository(root).unwrap();
    let targets = contracts
        .iter()
        .filter(|contract| contract.status.is_executable())
        .filter_map(|contract| contract.evidence.clone())
        .fold(
            (
                BTreeSet::<ExecutionKey>::new(),
                Vec::<EvidenceTarget>::new(),
            ),
            |(mut keys, mut targets), target| {
                if keys.insert(ExecutionKey::from_target(&target)) {
                    targets.push(target);
                }
                (keys, targets)
            },
        )
        .1;
    let failing_target = targets
        .iter()
        .rposition(|target| matches!(target, EvidenceTarget::Cargo(_)))
        .expect("the full registry contains Cargo evidence");
    let mut legacy_outputs = Vec::with_capacity(targets.len());
    let mut direct_outputs = Vec::with_capacity(targets.len());
    for (index, target) in targets.iter().enumerate() {
        let failed = index == failing_target;
        let diagnostics = if failed {
            "thread 'frozen_failure' panicked at 'frozen panic text'\n\
             test result: FAILED. 0 passed; 1 failed; 0 ignored\n\
             error: test failed, to rerun pass `-p frozen-package --test frozen`"
                .to_owned()
        } else {
            format!("target-{index}-diagnostic")
        };
        let direct = EvidenceOutput {
            duration_ms: Some(1_000 + u64::try_from(index).unwrap()),
            exit_code: Some(if failed { 17 } else { 0 }),
            stdout: format!("target-{index}-stdout"),
            stderr: diagnostics.clone(),
            start_error: None,
        };
        let mut legacy = direct.clone();
        legacy.duration_ms = Some(10 + u64::try_from(index).unwrap());
        if matches!(target, EvidenceTarget::Cargo(_)) {
            legacy.stderr = format!(
                "   Compiling frozen-package v0.0.0\n\
                 Finished `test` profile [unoptimized] target(s) in 0.01s\n\
                 Running tests/frozen.rs (target/debug/deps/frozen)\n\
                 {diagnostics}"
            );
        }
        legacy_outputs.push(legacy);
        direct_outputs.push(direct);
    }
    let request =
        Request::new(CommandKind::Run, Vec::new(), ExecutionPolicy::FailFast).with_jobs(1);

    let legacy = report_for(
        &request,
        legacy_execute_selected(
            root,
            contracts.clone(),
            &request,
            &FakeRunner::new(legacy_outputs),
        ),
    );
    let grouped = report_for(
        &request,
        execute_selected(root, contracts, &request, &FakeRunner::new(direct_outputs)),
    );

    assert_eq!(
        grouped
            .cases
            .iter()
            .map(|case| case.id.as_str())
            .collect::<Vec<_>>(),
        legacy
            .cases
            .iter()
            .map(|case| case.id.as_str())
            .collect::<Vec<_>>()
    );
    assert_eq!(grouped.success, legacy.success);
    for (direct, legacy) in grouped.cases.iter().zip(&legacy.cases) {
        assert_eq!(direct.outcome, legacy.outcome, "{}", direct.id);
        assert_eq!(direct.exit_code, legacy.exit_code, "{}", direct.id);
        assert_eq!(direct.message, legacy.message, "{}", direct.id);
        assert_eq!(
            direct.duration_ms.is_some(),
            legacy.duration_ms.is_some(),
            "{}",
            direct.id
        );
        assert_eq!(direct.stdout, legacy.stdout, "{}", direct.id);
        assert_eq!(
            direct
                .stderr
                .as_deref()
                .map(str::lines)
                .map(Iterator::collect::<Vec<_>>),
            stderr_without_cargo_progress(legacy.stderr.as_deref()),
            "{}",
            direct.id
        );
    }
    let failures = grouped
        .cases
        .iter()
        .filter(|case| case.outcome == Outcome::Failed)
        .collect::<Vec<_>>();
    assert!(!failures.is_empty());
    for failure in failures {
        let stderr = failure.stderr.as_deref().unwrap();
        assert!(stderr.contains("frozen panic text"), "{}", failure.id);
        assert!(stderr.contains("test result: FAILED"), "{}", failure.id);
    }
}

#[test]
fn keep_going_jobs_preserve_deterministic_case_semantics() {
    let first = named_cargo_target("package-a", "first", &[]);
    let second = named_cargo_target("package-a", "second", &[]);
    let third = named_cargo_target("package-a", "third", &[]);
    let contracts = vec![
        contract_with_target("a", first.clone()),
        contract_with_target("b", second.clone()),
        contract_with_target("c", third.clone()),
        contract_with_target("d", first.clone()),
    ];
    let mapped = [
        (first.clone(), output(0, "first")),
        (second.clone(), output(17, "second")),
        (third.clone(), output(0, "third")),
    ];
    let serial_request =
        Request::new(CommandKind::Run, Vec::new(), ExecutionPolicy::KeepGoing).with_jobs(1);
    let serial = execute_selected(
        Path::new("."),
        contracts.clone(),
        &serial_request,
        &FakeRunner::for_targets(mapped.clone()),
    );
    for jobs in [1, 2, 3, 8] {
        let concurrent_request =
            Request::new(CommandKind::Run, Vec::new(), ExecutionPolicy::KeepGoing).with_jobs(jobs);
        let concurrent_runner = FakeRunner::for_targets(mapped.clone())
            .with_delay(first.clone(), Duration::from_millis(40))
            .with_delay(second.clone(), Duration::from_millis(20));
        let concurrent = execute_selected(
            Path::new("."),
            contracts.clone(),
            &concurrent_request,
            &concurrent_runner,
        );

        assert_eq!(
            concurrent
                .iter()
                .map(|case| (
                    case.id.as_str(),
                    case.outcome,
                    case.exit_code,
                    case.stdout.as_deref()
                ))
                .collect::<Vec<_>>(),
            serial
                .iter()
                .map(|case| (
                    case.id.as_str(),
                    case.outcome,
                    case.exit_code,
                    case.stdout.as_deref()
                ))
                .collect::<Vec<_>>(),
            "jobs={jobs}"
        );
        assert_eq!(
            concurrent
                .iter()
                .map(|case| case.id.as_str())
                .collect::<Vec<_>>(),
            ["a", "b", "c", "d"]
        );
        assert!(!report_for(&concurrent_request, concurrent).success);
    }
}

#[test]
fn completed_worker_starts_next_target_without_waiting_for_slowest_peer() {
    let long = named_cargo_target("package-a", "long", &[]);
    let first_short = named_cargo_target("package-a", "first-short", &[]);
    let queued_short = named_cargo_target("package-a", "queued-short", &[]);
    let runner = FakeRunner::for_targets([
        (long.clone(), successful_output()),
        (first_short.clone(), successful_output()),
        (queued_short.clone(), successful_output()),
    ])
    .with_delay(long.clone(), Duration::from_millis(200))
    .with_delay(first_short.clone(), Duration::from_millis(10))
    .with_delay(queued_short.clone(), Duration::from_millis(10));
    let reports = execute_selected(
        Path::new("."),
        vec![
            contract_with_target("a", long.clone()),
            contract_with_target("b", first_short),
            contract_with_target("c", queued_short.clone()),
        ],
        &Request::new(CommandKind::Run, Vec::new(), ExecutionPolicy::KeepGoing).with_jobs(2),
        &runner,
    );

    assert!(reports.iter().all(|case| case.outcome == Outcome::Passed));
    let completions = runner.completions();
    let queued_short_completion = completions
        .iter()
        .position(|target| target == &queued_short)
        .unwrap();
    let long_completion = completions
        .iter()
        .position(|target| target == &long)
        .unwrap();
    assert!(queued_short_completion < long_completion);
}

#[test]
fn cargo_groups_preserve_each_exact_declared_feature_set() {
    let feature_a = named_cargo_target("package-a", "test-a", &["feature-a"]);
    let feature_b = named_cargo_target("package-a", "test-b", &["feature-b"]);
    let runner = FakeRunner::for_targets([
        (feature_a.clone(), output(0, "a")),
        (feature_b.clone(), output(0, "b")),
    ]);
    let reports = execute_selected(
        Path::new("."),
        vec![
            contract_with_target("a", feature_a),
            contract_with_target("b", feature_b),
        ],
        &Request::new(CommandKind::Run, Vec::new(), ExecutionPolicy::KeepGoing).with_jobs(2),
        &runner,
    );

    assert!(reports.iter().all(|case| case.outcome == Outcome::Passed));
    let keys = runner
        .groups()
        .iter()
        .map(|group| match &group[0] {
            EvidenceTarget::Cargo(target) => CargoBuildGroup::from_target(target),
            EvidenceTarget::PythonInstalledWheel(_) => panic!("expected Cargo group"),
        })
        .collect::<Vec<_>>();
    assert_eq!(
        keys,
        [
            CargoBuildGroup {
                package: "package-a".to_owned(),
                features: vec!["feature-a".to_owned()],
            },
            CargoBuildGroup {
                package: "package-a".to_owned(),
                features: vec!["feature-b".to_owned()],
            },
        ]
    );
    assert!(runner.groups().iter().all(|group| group.len() == 1));
}

#[test]
fn target_failure_is_attributed_without_poisoning_group_siblings() {
    let failing = named_cargo_target("package-a", "failing", &["feature-a"]);
    let passing = named_cargo_target("package-a", "passing", &["feature-a"]);
    let reports = execute_selected(
        Path::new("."),
        vec![
            contract_with_target("a", failing.clone()),
            contract_with_target("b", passing.clone()),
            contract_with_target("c", failing.clone()),
        ],
        &Request::new(CommandKind::Run, Vec::new(), ExecutionPolicy::KeepGoing).with_jobs(2),
        &FakeRunner::for_targets([
            (failing, output(9, "failed")),
            (passing, output(0, "passed")),
        ]),
    );

    assert_eq!(
        reports.iter().map(|case| case.outcome).collect::<Vec<_>>(),
        [Outcome::Failed, Outcome::Passed, Outcome::Failed]
    );
}

#[test]
fn build_failure_names_its_group_and_does_not_skip_other_groups() {
    let first = named_cargo_target("package-a", "first", &["feature-a"]);
    let mut first_claim_variant = first.clone();
    let EvidenceTarget::Cargo(first_claim) = &mut first_claim_variant else {
        unreachable!();
    };
    first_claim.table = Some("expected/first.csv".to_owned());
    let sibling = named_cargo_target("package-a", "sibling", &["feature-a"]);
    let other = named_cargo_target("package-b", "other", &["feature-b"]);
    let failed_group = match &first {
        EvidenceTarget::Cargo(target) => CargoBuildGroup::from_target(target),
        EvidenceTarget::PythonInstalledWheel(_) => unreachable!(),
    };
    let runner = FakeRunner::for_targets([(other.clone(), output(0, "other"))]).with_build_failure(
        failed_group,
        EvidenceOutput {
            duration_ms: None,
            exit_code: Some(101),
            stdout: "build-out".to_owned(),
            stderr: "build-err".to_owned(),
            start_error: None,
        },
    );
    let reports = execute_selected(
        Path::new("."),
        vec![
            contract_with_target("a", first),
            contract_with_target("b", first_claim_variant),
            contract_with_target("c", sibling),
            contract_with_target("d", other.clone()),
        ],
        &Request::new(CommandKind::Run, Vec::new(), ExecutionPolicy::FailFast).with_jobs(1),
        &runner,
    );

    assert_eq!(
        reports.iter().map(|case| case.outcome).collect::<Vec<_>>(),
        [
            Outcome::Failed,
            Outcome::Failed,
            Outcome::Failed,
            Outcome::Passed
        ]
    );
    assert!(
        reports[0]
            .message
            .as_deref()
            .unwrap()
            .contains("package-a features=[feature-a]")
    );
    assert!(
        reports[1]
            .message
            .as_deref()
            .unwrap()
            .contains("package-a features=[feature-a]")
    );
    assert!(
        reports[2]
            .message
            .as_deref()
            .unwrap()
            .contains("package-a features=[feature-a]")
    );
    assert_eq!(runner.targets(), [other]);
    assert_eq!(runner.groups().len(), 2);
}

#[test]
fn concurrent_fail_fast_reports_only_real_results_and_bounds_speculation() {
    let jobs = 3;
    let targets = (0..8)
        .map(|index| named_cargo_target("package-a", &format!("test-{index}"), &[]))
        .collect::<Vec<_>>();
    let mut runner = FakeRunner::for_targets(
        targets
            .iter()
            .enumerate()
            .map(|(index, target)| {
                (
                    target.clone(),
                    output(
                        match index {
                            0 => 17,
                            1 => 19,
                            _ => 0,
                        },
                        &format!("target-{index}"),
                    ),
                )
            })
            .collect::<Vec<_>>(),
    )
    .with_delay(targets[0].clone(), Duration::from_millis(100));
    for target in targets.iter().skip(1) {
        runner = runner.with_delay(target.clone(), Duration::from_millis(200));
    }
    let reports = execute_selected(
        Path::new("."),
        targets
            .iter()
            .enumerate()
            .map(|(index, target)| contract_with_target(&format!("case-{index}"), target.clone()))
            .collect(),
        &Request::new(CommandKind::Run, Vec::new(), ExecutionPolicy::FailFast).with_jobs(jobs),
        &runner,
    );

    let targets_after_failure = runner.targets().len() - 1;
    assert!(targets_after_failure <= jobs.saturating_sub(1));
    let failures = reports
        .iter()
        .filter(|case| case.outcome == Outcome::Failed)
        .collect::<Vec<_>>();
    assert_eq!(
        failures
            .iter()
            .map(|case| (case.id.as_str(), case.exit_code))
            .collect::<Vec<_>>(),
        [("case-0", Some(17)), ("case-1", Some(19))]
    );
    assert!(
        failures
            .iter()
            .all(|case| case.exit_code.is_some_and(|code| code != 0))
    );
    assert!(
        reports
            .iter()
            .filter(|case| !matches!(case.id.as_str(), "case-0" | "case-1"))
            .all(|case| case.outcome != Outcome::Failed)
    );
    assert_eq!(
        reports
            .iter()
            .filter(|case| case.outcome == Outcome::Passed)
            .count(),
        jobs - failures.len()
    );
    assert_eq!(
        reports
            .iter()
            .filter(|case| case.outcome == Outcome::Skipped)
            .count(),
        targets.len() - jobs
    );
    let request =
        Request::new(CommandKind::Run, Vec::new(), ExecutionPolicy::FailFast).with_jobs(jobs);
    assert!(!report_for(&request, reports).success);
}

#[test]
fn environment_filter_excludes_targets_before_group_building() {
    let portable = named_cargo_target("package-a", "portable", &[]);
    let mut physical = named_cargo_target("package-b", "physical", &["mpi-cuda"]);
    let EvidenceTarget::Cargo(physical_target) = &mut physical else {
        unreachable!();
    };
    physical_target.environment = EvidenceEnvironment::PhysicalMpiCuda;
    let runner = FakeRunner::for_targets([(portable.clone(), output(0, "portable"))]);
    let reports = execute_selected(
        Path::new("."),
        vec![
            contract_with_target("physical", physical),
            contract_with_target("portable", portable.clone()),
        ],
        &Request::new(CommandKind::Run, Vec::new(), ExecutionPolicy::FailFast)
            .for_environment(EvidenceEnvironment::HostCpu)
            .with_jobs(2),
        &runner,
    );

    assert_eq!(
        reports.iter().map(|case| case.outcome).collect::<Vec<_>>(),
        [Outcome::NotSelected, Outcome::Passed]
    );
    assert_eq!(runner.targets(), [portable]);
    let groups = runner.groups();
    assert_eq!(groups.len(), 1);
    assert!(
        groups[0]
            .iter()
            .all(|target| { target.environment() == EvidenceEnvironment::HostCpu })
    );
}

#[test]
fn full_registry_runner_kind_partition_is_exact_visible_and_prebuild() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .unwrap();
    let contracts = load_repository(root).unwrap();
    let targets = contracts
        .iter()
        .filter(|contract| contract.status.is_executable())
        .filter_map(|contract| contract.evidence.clone())
        .fold(Vec::new(), |mut targets, target| {
            if !targets.contains(&target) {
                targets.push(target);
            }
            targets
        });
    let runner_for = || {
        FakeRunner::for_targets(
            targets
                .iter()
                .cloned()
                .map(|target| (target, successful_output())),
        )
    };
    let request = Request::new(CommandKind::Run, Vec::new(), ExecutionPolicy::KeepGoing);
    let unfiltered = execute_selected(root, contracts.clone(), &request, &runner_for());

    let cargo_runner = runner_for();
    let cargo = execute_selected(
        root,
        contracts.clone(),
        &request.clone().for_runner_kind(RunnerKind::Cargo),
        &cargo_runner,
    );
    let python_runner = runner_for();
    let python = execute_selected(
        root,
        contracts.clone(),
        &request
            .clone()
            .for_runner_kind(RunnerKind::PythonInstalledWheel),
        &python_runner,
    );
    let passed = |reports: &[CaseReport]| {
        reports
            .iter()
            .filter(|case| case.outcome == Outcome::Passed)
            .map(|case| case.id.clone())
            .collect::<BTreeSet<_>>()
    };
    let all = passed(&unfiltered);
    let cargo_cases = passed(&cargo);
    let python_cases = passed(&python);
    assert!(cargo_cases.is_disjoint(&python_cases));
    assert_eq!(
        cargo_cases
            .union(&python_cases)
            .cloned()
            .collect::<BTreeSet<_>>(),
        all
    );

    for (reports, selected) in [
        (&cargo, RunnerKind::Cargo),
        (&python, RunnerKind::PythonInstalledWheel),
    ] {
        for case in reports {
            let contract = contracts
                .iter()
                .find(|contract| contract.id == case.id)
                .unwrap();
            if !contract.status.is_executable() {
                continue;
            }
            let required = contract.evidence.as_ref().unwrap().runner_kind();
            if required == selected {
                assert_eq!(case.outcome, Outcome::Passed, "{}", case.id);
            } else {
                assert_eq!(case.outcome, Outcome::NotSelected, "{}", case.id);
                assert_eq!(
                    case.message.as_deref(),
                    Some(format!("evidence requires `{}` runner kind", required.as_str()).as_str()),
                    "{}",
                    case.id
                );
            }
        }
    }
    assert!(python_runner.groups().is_empty());
    assert!(
        cargo_runner
            .targets()
            .iter()
            .all(|target| target.runner_kind() == RunnerKind::Cargo)
    );
    assert!(
        python_runner
            .targets()
            .iter()
            .all(|target| target.runner_kind() == RunnerKind::PythonInstalledWheel)
    );
}

#[test]
fn runner_kind_and_environment_filters_state_distinct_reasons() {
    let mut physical = cargo_target();
    let EvidenceTarget::Cargo(target) = &mut physical else {
        unreachable!();
    };
    target.environment = EvidenceEnvironment::PhysicalMpiCuda;
    let reports = execute_selected(
        Path::new("."),
        vec![
            contract_with_target("physical-cargo", physical),
            contract_with_target("host-python", python_target()),
        ],
        &Request::new(CommandKind::Run, Vec::new(), ExecutionPolicy::KeepGoing)
            .for_environment(EvidenceEnvironment::HostCpu)
            .for_runner_kind(RunnerKind::Cargo),
        &FakeRunner::new([]),
    );

    assert_eq!(
        reports
            .iter()
            .map(|case| (case.outcome, case.message.as_deref()))
            .collect::<Vec<_>>(),
        [
            (
                Outcome::NotSelected,
                Some("evidence requires `physical-mpi-cuda` execution")
            ),
            (
                Outcome::NotSelected,
                Some("evidence requires `python-installed-wheel` runner kind")
            ),
        ]
    );
}

#[test]
fn empty_runner_kind_selection_succeeds_without_build_or_execution() {
    let fixture = Fixture::new();
    fixture.write_manifest(valid_manifest());
    let runner = FakeRunner::new([]);
    let request = Request::new(CommandKind::Run, Vec::new(), ExecutionPolicy::FailFast)
        .for_runner_kind(RunnerKind::PythonInstalledWheel);
    let report = execute(&fixture.root, &request, &runner);

    assert!(report.success);
    assert!(report.errors.is_empty());
    assert_eq!(report.cases.len(), 1);
    assert_eq!(report.cases[0].outcome, Outcome::NotSelected);
    assert_eq!(
        report.cases[0].message.as_deref(),
        Some("evidence requires `cargo` runner kind")
    );
    assert_eq!(
        report.selected_runner_kind,
        Some(RunnerKind::PythonInstalledWheel)
    );
    assert_eq!(
        serde_json::to_value(report).unwrap()["selected_runner_kind"],
        "python-installed-wheel"
    );
    assert!(runner.groups().is_empty());
    assert!(runner.targets().is_empty());
}

#[test]
fn runner_kind_filters_do_not_narrow_repository_validation() {
    for (source, expected_error) in [
        (
            valid_manifest().replace("expected/evidence.csv", "expected/missing.csv"),
            "evidence artifact `expected/missing.csv` is missing or inaccessible",
        ),
        (
            valid_python_manifest().replace("python_evidence.py", "missing.py"),
            "Python evidence script `tools/ci/missing.py` is missing or inaccessible",
        ),
    ] {
        let fixture = Fixture::new();
        fixture.write_manifest(&source);

        let reports = [RunnerKind::Cargo, RunnerKind::PythonInstalledWheel].map(|runner_kind| {
            execute(
                &fixture.root,
                &Request::new(CommandKind::Run, Vec::new(), ExecutionPolicy::FailFast)
                    .for_runner_kind(runner_kind),
                &FakeRunner::new([]),
            )
        });
        for report in &reports {
            assert!(!report.success);
            assert!(report.cases.is_empty());
            assert_eq!(report.errors.len(), 1);
            assert!(report.errors[0].contains(expected_error), "{report:?}");
        }
        assert_eq!(reports[0].errors, reports[1].errors);
    }
}

#[test]
fn unfiltered_request_preserves_complete_pre_split_report() {
    let contracts = vec![
        contract_with_target("cargo", cargo_target()),
        contract_with_target("python", python_target()),
    ];
    let request = Request::new(CommandKind::Run, Vec::new(), ExecutionPolicy::KeepGoing);
    let outputs = [
        (cargo_target(), output(0, "cargo")),
        (python_target(), output(0, "python")),
    ];
    let reference = report_for(
        &request,
        legacy_execute_selected(
            Path::new("."),
            contracts.clone(),
            &request,
            &FakeRunner::for_targets(outputs.clone()),
        ),
    );
    let actual = report_for(
        &request,
        execute_selected(
            Path::new("."),
            contracts,
            &request,
            &FakeRunner::for_targets(outputs),
        ),
    );

    assert_eq!(actual, reference);
    assert_eq!(actual.selected_runner_kind, None);
}
