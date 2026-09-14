const ACCEPTED_BYTES: &[u8] =
    include_bytes!("../../../verify/interfaces/cli-compile-check/models/accepted-secret.eqi");
const REJECTED_BYTES: &[u8] =
    include_bytes!("../../../verify/interfaces/cli-compile-check/models/rejected-secret.eqi");
const ROOT_HELP: &[u8] =
    include_bytes!("../../../verify/interfaces/cli-compile-check/expected/root-help.txt");
const CHECK_HELP: &[u8] =
    include_bytes!("../../../verify/interfaces/cli-compile-check/expected/check-help.txt");

#[path = "support/cli_compile_check_home_path.rs"]
mod cli_compile_check_home_path;
mod cli_compile_check_trailing_bare_delimiter;
#[path = "../src/bin/eqiora/main.rs"]
mod cli_main;
const ACCEPTED_LITERAL: &[u8] = b"// EQIORA_CLI_SECRET_ACCEPTED_c1479c2e\nmodel decay() {\n  state x: 1;\n  initial { x = 1; }\n  parameter rate: 1 / s = 1;\n  relation flow {\n    derivative(x) + rate * x = 0;\n  }\n}\n";
const REJECTED_LITERAL: &[u8] = b"// EQIORA_CLI_SECRET_REJECTED_918bf4ad\n";

mod full {
    use super::*;
    use std::cell::Cell;
    use std::ffi::{OsStr, OsString};
    use std::fs::{self, File};
    use std::io::{Cursor, Read, Write};
    use std::path::{Path, PathBuf};
    use std::process::{Command, Output, Stdio};
    use std::sync::atomic::{AtomicU64, Ordering};

    use eqiora::api::{ModelDocument, SemanticFingerprintGeneration};
    use eqiora::{Code, Diagnostic, GraphPath, Patch, Severity, Span};
    use serde_json::Value;

    const CONTRACT_SOURCE: &str =
        include_str!("../../../verify/interfaces/cli-compile-check/expected/contract.json");
    const ACCEPTED_SENTINEL: &str = "EQIORA_CLI_SECRET_ACCEPTED_c1479c2e";
    const REJECTED_SENTINEL: &str = "EQIORA_CLI_SECRET_REJECTED_918bf4ad";
    const PANIC_SENTINEL: &str = "EQIORA_CLI_PANIC_PAYLOAD_SECRET_524bb795";

    static SCRATCH_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn expected() -> Value {
        serde_json::from_str(CONTRACT_SOURCE).expect("frozen CLI contract JSON")
    }

    fn repository_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("workspace root")
            .to_path_buf()
    }

    #[allow(clippy::option_env_unwrap)]
    fn binary() -> &'static OsStr {
        OsStr::new(
            option_env!("CARGO_BIN_EXE_eqiora")
                .expect("production binary is absent; oracle remains intentionally RED"),
        )
    }

    struct Scratch {
        path: PathBuf,
    }

    impl Scratch {
        fn new(label: &str) -> Self {
            let tmpdir = PathBuf::from(
                std::env::var_os("TMPDIR")
                    .expect("TMPDIR must name an absolute home-backed scratch root"),
            );
            let home = PathBuf::from(std::env::var_os("HOME").expect("HOME is required"));
            cli_compile_check_home_path::require_canonical_home_backed_tmpdir(&tmpdir, &home)
                .expect("TMPDIR must be canonical and home-backed");
            let unique = format!(
                "eqiora-cli-{label}-{}-{}",
                std::process::id(),
                SCRATCH_SEQUENCE.fetch_add(1, Ordering::Relaxed)
            );
            let path = tmpdir.join(unique);
            fs::create_dir(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            if self.path.exists() {
                fs::remove_dir_all(&self.path).expect("remove exact CLI oracle scratch root");
            }
        }
    }

    fn command_output(program: &OsStr, args: &[OsString], cwd: &Path) -> Output {
        Command::new(program)
            .args(args)
            .current_dir(cwd)
            .env("NO_COLOR", "1")
            .env("COLUMNS", "17")
            .env("LC_ALL", "C")
            .env("LANG", "fr_FR.UTF-8")
            .output()
            .expect("run CLI candidate")
    }

    fn run(args: impl IntoIterator<Item = impl Into<OsString>>, cwd: &Path) -> Output {
        let args = args.into_iter().map(Into::into).collect::<Vec<_>>();
        command_output(binary(), &args, cwd)
    }

    fn assert_output(output: &Output, code: i32, stdout: &[u8], stderr: &[u8]) {
        assert_eq!(output.status.code(), Some(code));
        assert_eq!(output.stdout, stdout);
        assert_eq!(output.stderr, stderr);
        assert!(!output.stdout.contains(&0x1b));
        assert!(!output.stderr.contains(&0x1b));
    }

    fn fixed_stderr<'a>(contract: &'a Value, key: &str) -> &'a [u8] {
        contract["stderr"][key]
            .as_str()
            .unwrap_or_else(|| panic!("missing frozen stderr {key}"))
            .as_bytes()
    }

    fn exit(contract: &Value, key: &str) -> i32 {
        contract["exits"][key].as_i64().unwrap() as i32
    }

    #[rustfmt::skip]
    const CLI_PATHS: [&str; 3] = [
        "crates/eqiora/src/bin/eqiora/main.rs",
        "crates/eqiora/src/bin/eqiora/command.rs",
        "crates/eqiora/src/bin/eqiora/terminal.rs",
    ];

    #[rustfmt::skip]
    fn compact_cli_architecture(sources: &[String; 3]) -> bool {
        let joined = sources.join("\n");
        let forbidden = ["compiler::compile", "eqiora::control", "eqiora_control", "control::", "eqiora::mcp", "eqiora_mcp", "mcp::", "eqiora::python", "eqiora_python", "python::", "studio::", "eqiora::artifact", "eqiora_artifact", "artifact::", "serde", "json"];
        sources[0].matches("command::contained_run(").count() == 1
            && joined.matches("ModelDocument::compile").count() == 1
            && forbidden.iter().all(|authority| !joined.contains(authority))
            && sources.iter().all(|source| !source.lines().any(|line| line.trim_start().starts_with("pub ")))
    }

    #[rustfmt::skip]
    #[test]
    fn compact_cli_architecture_facts_and_selftest_are_focused() {
        let _operation: fn(&str, &str) -> Result<ModelDocument, Vec<Diagnostic>> = ModelDocument::compile;
        let root = repository_root();
        let contract = expected();
        let paths = contract["productionPaths"].as_array().unwrap().iter().map(|path| path.as_str().unwrap()).collect::<Vec<_>>();
        assert_eq!(paths, CLI_PATHS);
        let sources: [String; 3] = CLI_PATHS.map(|path| fs::read_to_string(root.join(path)).unwrap());
        let mut files = fs::read_dir(root.join("crates/eqiora/src/bin/eqiora")).unwrap().map(|entry| entry.unwrap().file_name()).collect::<Vec<_>>();
        files.sort();
        assert_eq!(files, ["command.rs", "main.rs", "terminal.rs"].map(OsString::from));
        assert!(compact_cli_architecture(&sources));
        let mut second = sources.clone(); second[1].push_str("\nfn extra(){let _=ModelDocument::compile;}\n"); assert!(!compact_cli_architecture(&second));
        let mut lower = sources.clone(); lower[0] = lower[0].replace("ModelDocument::compile", "eqiora::compiler::compile"); assert!(!compact_cli_architecture(&lower));
        let mut public = sources; public[2].push_str("\npub fn widened() {}\n"); assert!(!compact_cli_architecture(&public));
    }

    #[test]
    fn command_grammar_help_version_and_parse_failures_are_exact() {
        let scratch = Scratch::new("grammar");
        let contract = expected();
        assert_eq!(contract["packageVersion"], "0.1.0-alpha.3");
        for args in [["-h"], ["--help"]] {
            assert_output(
                &run(args, &scratch.path),
                exit(&contract, "accepted"),
                ROOT_HELP,
                b"",
            );
        }
        for args in [["check", "-h"], ["check", "--help"]] {
            assert_output(
                &run(args, &scratch.path),
                exit(&contract, "accepted"),
                CHECK_HELP,
                b"",
            );
        }
        let version = format!("eqiora {}\n", env!("CARGO_PKG_VERSION"));
        for args in [["-V"], ["--version"]] {
            assert_output(
                &run(args, &scratch.path),
                exit(&contract, "accepted"),
                version.as_bytes(),
                b"",
            );
        }
        let invalid = fixed_stderr(&contract, "invalidCommandLine");
        let cases: &[&[&str]] = &[
            &[],
            &["check"],
            &["check", "one", "two"],
            &["check", "--unknown"],
            &["unknown"],
            &["help"],
            &["--help", "extra"],
            &["--version", "extra"],
            &["--help", "--version"],
            &["check", "--help", "extra"],
            &["check", "--", "one", "two"],
            &["--=malformed"],
        ];
        for args in cases {
            assert_output(
                &run(args.iter().copied(), &scratch.path),
                exit(&contract, "usage"),
                b"",
                invalid,
            );
        }

        for columns in ["1", "80", "4096"] {
            let output = Command::new(binary())
                .arg("--help")
                .current_dir(&scratch.path)
                .env("COLUMNS", columns)
                .env("LC_ALL", "tr_TR.UTF-8")
                .env("LANG", "ja_JP.UTF-8")
                .output()
                .unwrap();
            assert_output(&output, 0, ROOT_HELP, b"");
        }
    }

    #[cfg(unix)]
    #[test]
    fn path_argument_unicode_and_literal_hyphen_boundaries_are_exact() {
        use std::os::unix::ffi::OsStringExt;

        let scratch = Scratch::new("paths");
        let contract = expected();
        let unavailable = fixed_stderr(&contract, "unavailable");
        let invalid = fixed_stderr(&contract, "invalidPath");
        let usage = exit(&contract, "usage");

        for admitted in ["a".repeat(4096), "é".repeat(2048)] {
            assert_output(
                &run(
                    [OsString::from("check"), OsString::from(admitted)],
                    &scratch.path,
                ),
                exit(&contract, "unavailable"),
                b"",
                unavailable,
            );
        }
        for rejected in [
            String::new(),
            "a".repeat(4097),
            "é".repeat(2049),
            "bad\npath".to_owned(),
            "bad\tpath".to_owned(),
        ] {
            assert_output(
                &run(
                    [OsString::from("check"), OsString::from(rejected)],
                    &scratch.path,
                ),
                usage,
                b"",
                invalid,
            );
        }
        let non_utf8 = OsString::from_vec(vec![b'b', b'a', b'd', 0xff]);
        let output = command_output(binary(), &["check".into(), non_utf8], &scratch.path);
        assert_output(&output, usage, b"", invalid);
        assert!(!output.stderr.windows(3).any(|bytes| bytes == b"bad"));
        let reflected = run(["unknown-secret-command"], &scratch.path);
        assert_output(
            &reflected,
            usage,
            b"",
            fixed_stderr(&contract, "invalidCommandLine"),
        );
        assert!(
            !reflected
                .stderr
                .windows("unknown-secret-command".len())
                .any(|part| part == b"unknown-secret-command")
        );

        fs::write(scratch.path.join("-"), ACCEPTED_BYTES).unwrap();
        let literal_dash = run(["check", "-"], &scratch.path);
        assert_eq!(literal_dash.status.code(), Some(0));
        assert!(literal_dash.stderr.is_empty());
        assert!(literal_dash.stdout.starts_with(b"accepted "));

        fs::write(scratch.path.join("-secret.eqi"), ACCEPTED_BYTES).unwrap();
        assert_output(
            &run(["check", "-secret.eqi"], &scratch.path),
            usage,
            b"",
            fixed_stderr(&contract, "invalidCommandLine"),
        );
        let escaped = run(["check", "--", "-secret.eqi"], &scratch.path);
        assert_eq!(escaped.status.code(), Some(0));
        assert!(escaped.stderr.is_empty());
    }

    #[test]
    fn accepted_and_rejected_files_match_the_independent_direct_operation() {
        assert_eq!(ACCEPTED_BYTES, ACCEPTED_LITERAL);
        assert_eq!(REJECTED_BYTES, REJECTED_LITERAL);
        let scratch = Scratch::new("parity");
        let accepted_path = scratch.path.join("accepted-secret.eqi");
        let rejected_path = scratch.path.join("rejected-secret.eqi");
        fs::write(&accepted_path, ACCEPTED_BYTES).unwrap();
        fs::write(&rejected_path, REJECTED_BYTES).unwrap();
        let accepted_spelling = accepted_path.to_str().unwrap();
        let first = ModelDocument::compile(
            accepted_spelling,
            std::str::from_utf8(ACCEPTED_BYTES).unwrap(),
        )
        .unwrap();
        let second = ModelDocument::compile(
            accepted_spelling,
            std::str::from_utf8(ACCEPTED_BYTES).unwrap(),
        )
        .unwrap();
        let third = ModelDocument::compile(
            accepted_spelling,
            std::str::from_utf8(ACCEPTED_BYTES).unwrap(),
        )
        .unwrap();
        let fingerprints = [
            first.structural_fingerprint().unwrap(),
            second.structural_fingerprint().unwrap(),
            third.structural_fingerprint().unwrap(),
        ];
        assert_eq!(
            fingerprints[0].generation(),
            SemanticFingerprintGeneration::V22
        );
        assert_eq!(fingerprints[0], fingerprints[1]);
        assert_eq!(fingerprints[0], fingerprints[2]);
        // Closed source compilation shares canonical identities with selected entries.
        assert_eq!(first.program().model(), second.program().model());
        assert_eq!(first.program().model(), third.program().model());
        assert_eq!(first.digest().unwrap(), second.digest().unwrap());
        assert_eq!(first.digest().unwrap(), third.digest().unwrap());
        let changed_source = std::str::from_utf8(ACCEPTED_BYTES)
            .unwrap()
            .replace("parameter rate: 1 / s = 1;", "parameter rate: 1 / s = 2;");
        assert_ne!(changed_source.as_bytes(), ACCEPTED_BYTES);
        let changed = ModelDocument::compile(accepted_spelling, &changed_source).unwrap();
        assert_ne!(first.digest().unwrap(), changed.digest().unwrap());

        let accepted = run(["check", accepted_spelling], &scratch.path);
        let accepted_line = format!("accepted {}\n", fingerprints[0]);
        assert_output(&accepted, 0, accepted_line.as_bytes(), b"");
        let repeated = run(["check", accepted_spelling], &scratch.path);
        assert_output(&repeated, 0, accepted_line.as_bytes(), b"");

        let rejected_spelling = rejected_path.to_str().unwrap();
        let direct = ModelDocument::compile(
            rejected_spelling,
            std::str::from_utf8(REJECTED_BYTES).unwrap(),
        )
        .unwrap_err();
        let expected_stderr = reference_render(&direct).expect("direct diagnostics fit");
        let rejected = run(["check", rejected_spelling], &scratch.path);
        assert_output(&rejected, 1, b"", &expected_stderr);
        assert_eq!(direct.len(), 1);
        let frozen = &expected()["rejectedDiagnostic"];
        assert_eq!(direct[0].severity(), Severity::Error);
        assert_eq!(frozen["severity"], "error");
        assert_eq!(direct[0].code().to_string(), frozen["code"]);
        assert_eq!(direct[0].message(), frozen["message"]);
        let span = direct[0].source_span().unwrap();
        assert_eq!(span.file, rejected_spelling);
        assert_eq!(span.start as u64, frozen["start"].as_u64().unwrap());
        assert_eq!(span.end as u64, frozen["end"].as_u64().unwrap());

        for output in [&accepted, &repeated, &rejected] {
            let streams = [output.stdout.as_slice(), output.stderr.as_slice()].concat();
            assert!(
                !streams
                    .windows(ACCEPTED_SENTINEL.len())
                    .any(|part| part == ACCEPTED_SENTINEL.as_bytes())
            );
            assert!(
                !streams
                    .windows(REJECTED_SENTINEL.len())
                    .any(|part| part == REJECTED_SENTINEL.as_bytes())
            );
            assert!(
                !streams
                    .windows(ACCEPTED_BYTES.len())
                    .any(|part| part == ACCEPTED_BYTES)
            );
            assert!(
                !streams
                    .windows(REJECTED_BYTES.len())
                    .any(|part| part == REJECTED_BYTES)
            );
        }
        let direct_ids = [
            first.program().model().to_string(),
            second.program().model().to_string(),
            third.program().model().to_string(),
            first.digest().unwrap().to_string(),
            second.digest().unwrap().to_string(),
            third.digest().unwrap().to_string(),
        ];
        for secret in direct_ids {
            assert!(
                !accepted
                    .stdout
                    .windows(secret.len())
                    .any(|part| part == secret.as_bytes())
            );
        }
    }

    fn escaped(value: &str) -> String {
        let mut output = String::new();
        for character in value.chars() {
            match character {
                '\u{20}'..='\u{7e}' if character != '\\' => output.push(character),
                '\\' => output.push_str("\\\\"),
                other => output.push_str(&format!("\\u{{{:x}}}", other as u32)),
            }
        }
        output
    }

    fn bounded_member(value: &str, scalar_limit: usize, byte_limit: usize, nonempty: bool) -> bool {
        (!nonempty || !value.is_empty())
            && value.len() <= byte_limit
            && value.chars().count() <= scalar_limit
    }

    fn reference_render(diagnostics: &[Diagnostic]) -> Option<Vec<u8>> {
        if diagnostics.len() > 1024 {
            return None;
        }
        let mut output = Vec::new();
        for diagnostic in diagnostics {
            let code = diagnostic.code().to_string();
            if code.len() != 6
                || !code[..2].bytes().all(|byte| byte.is_ascii_uppercase())
                || !code[2..].bytes().all(|byte| byte.is_ascii_digit())
                || !bounded_member(diagnostic.message(), 1_048_576, 1_048_576, true)
            {
                return None;
            }
            let mut line = String::new();
            if let Some(span) = diagnostic.source_span() {
                if span.end < span.start || !bounded_member(&span.file, 4096, 4096, false) {
                    return None;
                }
                line.push_str(&escaped(&span.file));
                line.push_str(&format!(":{}:{}: ", span.start, span.end));
            }
            let severity = match diagnostic.severity() {
                Severity::Error => "error",
                Severity::Warning => "warning",
                Severity::Note => "note",
            };
            line.push_str(severity);
            line.push('[');
            line.push_str(&code);
            line.push_str("]: ");
            line.push_str(&escaped(diagnostic.message()));
            if let Some(path) = diagnostic.graph_path() {
                if path.segments().len() > 256
                    || path
                        .segments()
                        .iter()
                        .any(|segment| !bounded_member(segment, 4096, 4096, true))
                {
                    return None;
                }
                line.push_str(" (at ");
                line.push_str(
                    &path
                        .segments()
                        .iter()
                        .map(|segment| escaped(segment))
                        .collect::<Vec<_>>()
                        .join("."),
                );
                line.push(')');
            }
            if let Some(patch) = diagnostic.suggestion() {
                if !bounded_member(&patch.summary, 4096, 4096, true) {
                    return None;
                }
                line.push_str("; help: ");
                line.push_str(&escaped(&patch.summary));
            }
            line.push('\n');
            let next = output.len().checked_add(line.len())?;
            if next > 1_048_576 {
                return None;
            }
            output.extend_from_slice(line.as_bytes());
        }
        Some(output)
    }

    fn synthetic_document(rate: u8) -> Result<ModelDocument, Vec<Diagnostic>> {
        let source = std::str::from_utf8(ACCEPTED_BYTES).unwrap().replace(
            "parameter rate: 1 / s = 1;",
            &format!("parameter rate: 1 / s = {rate};"),
        );
        ModelDocument::compile("oracle-returned.eqi", &source)
    }

    fn synthetic_rich(filename: &str) -> Vec<Diagnostic> {
        vec![
            Diagnostic::warning(Code("WN0042"), "slash\\\n\té")
                .with_span(Span {
                    file: filename.to_owned(),
                    start: 3,
                    end: 7,
                })
                .with_graph_path(GraphPath::new(["root", "child\\segment"]))
                .with_suggestion(Patch::new("fix\nnow")),
        ]
    }

    #[test]
    fn synthetic_returned_documents_are_distinct() {
        let first = synthetic_document(2).unwrap();
        let second = synthetic_document(3).unwrap();
        assert_ne!(
            first.structural_fingerprint().unwrap(),
            second.structural_fingerprint().unwrap()
        );
    }

    #[cfg(unix)]
    #[test]
    fn selected_input_and_valid_decoy_have_distinct_results() {
        let scratch = Scratch::new("selected-decoy");
        let selected = scratch.path.join("selected.eqi");
        let decoy = scratch.path.join("decoy.eqi");
        let source = std::str::from_utf8(ACCEPTED_BYTES).unwrap();
        let selected_source =
            source.replace("parameter rate: 1 / s = 1;", "parameter rate: 1 / s = 2;");
        let decoy_source =
            source.replace("parameter rate: 1 / s = 1;", "parameter rate: 1 / s = 3;");
        fs::write(&selected, &selected_source).unwrap();
        fs::write(&decoy, &decoy_source).unwrap();
        let selected_name = selected.to_str().unwrap();
        let decoy_name = decoy.to_str().unwrap();
        let selected_document = ModelDocument::compile(selected_name, &selected_source).unwrap();
        let decoy_document = ModelDocument::compile(decoy_name, &decoy_source).unwrap();
        let selected_fingerprint = selected_document.structural_fingerprint().unwrap();
        let decoy_fingerprint = decoy_document.structural_fingerprint().unwrap();
        assert_ne!(selected_fingerprint, decoy_fingerprint);
        let selected_stdout = format!("accepted {selected_fingerprint}\n").into_bytes();
        let decoy_stdout = format!("accepted {decoy_fingerprint}\n").into_bytes();
        assert_output(
            &run(["check", selected_name], &scratch.path),
            0,
            &selected_stdout,
            b"",
        );
        assert_output(
            &run(["check", decoy_name], &scratch.path),
            0,
            &decoy_stdout,
            b"",
        );
        let substituted = run_oracle_child(
            &scratch,
            &["check".into(), selected.as_os_str().to_owned()],
            "valid-decoy-substitution",
            "none",
            Some(&decoy),
            Some(selected_name),
        );
        assert_eq!(substituted.count, 1);
        assert_eq!(substituted.exit, 0);
        assert!(substituted.stderr.is_empty());
        assert_eq!(substituted.stdout, decoy_stdout);
        assert_ne!(substituted.stdout, selected_stdout);
    }

    #[test]
    fn synthetic_diagnostics_cover_fields_escaping_and_every_bound() {
        assert_eq!(
            cli_main::severity_label_for_oracle(Severity::Error),
            "error"
        );
        assert_eq!(
            cli_main::severity_label_for_oracle(Severity::Warning),
            "warning"
        );
        assert_eq!(cli_main::severity_label_for_oracle(Severity::Note), "note");

        let complete = vec![
            Diagnostic::error(Code("ER0001"), "ascii\\\n\t\u{7f}é")
                .with_span(Span {
                    file: "dir\\file\né.eqi".to_owned(),
                    start: 4,
                    end: 9,
                })
                .with_graph_path(GraphPath::new(["root", "slash\\", "line\n", "é"]))
                .with_suggestion(Patch::new("replace\\\té")),
            Diagnostic::warning(Code("WN0002"), "warning"),
        ];
        assert_eq!(
            cli_main::render_for_oracle(&complete),
            reference_render(&complete)
        );
        let rendered = reference_render(&complete).unwrap();
        assert!(rendered.windows(b"\\\\".len()).any(|part| part == b"\\\\"));
        for escape in [b"\\u{a}".as_slice(), b"\\u{9}", b"\\u{7f}", b"\\u{e9}"] {
            assert!(rendered.windows(escape.len()).any(|part| part == escape));
        }
        assert!(!rendered.windows(2).any(|part| part == b"\r\n"));

        let exact_message = Diagnostic::error(Code("ER0003"), "x".repeat(1_048_576));
        assert_eq!(
            cli_main::render_for_oracle(std::slice::from_ref(&exact_message)),
            reference_render(std::slice::from_ref(&exact_message))
        );
        for rejected in [
            Diagnostic::error(Code("bad"), "message"),
            Diagnostic::error(Code("ER0004"), ""),
            Diagnostic::error(Code("ER0005"), "x".repeat(1_048_577)),
            Diagnostic::error(Code("ER0006"), "message").with_span(Span {
                file: "x".repeat(4097),
                start: 0,
                end: 0,
            }),
            Diagnostic::error(Code("ER0007"), "message").with_span(Span {
                file: "x".to_owned(),
                start: 2,
                end: 1,
            }),
            Diagnostic::error(Code("ER0008"), "message").with_graph_path(GraphPath::new([""])),
            Diagnostic::error(Code("ER0009"), "message")
                .with_graph_path(GraphPath::new(["x".repeat(4097)])),
            Diagnostic::error(Code("ER0010"), "message").with_suggestion(Patch::new("")),
            Diagnostic::error(Code("ER0011"), "message")
                .with_suggestion(Patch::new("x".repeat(4097))),
        ] {
            assert_eq!(reference_render(std::slice::from_ref(&rejected)), None);
            assert_eq!(cli_main::render_for_oracle(&[rejected]), None);
        }
        for (accepted_member, rejected_member) in [
            ("é".repeat(2048), "é".repeat(2049)),
            ("x".repeat(4096), "x".repeat(4097)),
        ] {
            let accepted_graph = Diagnostic::error(Code("ER0020"), "message")
                .with_graph_path(GraphPath::new([accepted_member.clone()]));
            assert_eq!(
                cli_main::render_for_oracle(std::slice::from_ref(&accepted_graph)),
                reference_render(std::slice::from_ref(&accepted_graph))
            );
            let rejected_graph = Diagnostic::error(Code("ER0021"), "message")
                .with_graph_path(GraphPath::new([rejected_member.clone()]));
            assert_eq!(cli_main::render_for_oracle(&[rejected_graph]), None);
            let accepted_patch = Diagnostic::error(Code("ER0022"), "message")
                .with_suggestion(Patch::new(accepted_member));
            assert_eq!(
                cli_main::render_for_oracle(std::slice::from_ref(&accepted_patch)),
                reference_render(std::slice::from_ref(&accepted_patch))
            );
            let rejected_patch = Diagnostic::error(Code("ER0023"), "message")
                .with_suggestion(Patch::new(rejected_member));
            assert_eq!(cli_main::render_for_oracle(&[rejected_patch]), None);
        }
        let graph_256 = Diagnostic::error(Code("ER0012"), "message")
            .with_graph_path(GraphPath::new(std::iter::repeat_n("x", 256)));
        let graph_257 = Diagnostic::error(Code("ER0013"), "message")
            .with_graph_path(GraphPath::new(std::iter::repeat_n("x", 257)));
        assert_eq!(
            cli_main::render_for_oracle(std::slice::from_ref(&graph_256)),
            reference_render(std::slice::from_ref(&graph_256))
        );
        assert_eq!(cli_main::render_for_oracle(&[graph_257]), None);

        let count_1024 = vec![Diagnostic::error(Code("ER0014"), "x"); 1024];
        let count_1025 = vec![Diagnostic::error(Code("ER0015"), "x"); 1025];
        assert_eq!(
            cli_main::render_for_oracle(&count_1024),
            reference_render(&count_1024)
        );
        assert_eq!(cli_main::render_for_oracle(&count_1025), None);

        let prefix = b"error[ER0016]: ".len() + 1;
        let exact_stream = Diagnostic::error(Code("ER0016"), "x".repeat(1_048_576 - prefix));
        let over_stream = Diagnostic::error(Code("ER0016"), "x".repeat(1_048_577 - prefix));
        assert_eq!(
            reference_render(std::slice::from_ref(&exact_stream))
                .unwrap()
                .len(),
            1_048_576
        );
        assert_eq!(
            cli_main::render_for_oracle(&[exact_stream]),
            reference_render(&[Diagnostic::error(
                Code("ER0016"),
                "x".repeat(1_048_576 - prefix)
            )])
        );
        assert_eq!(cli_main::render_for_oracle(&[over_stream]), None);
        assert_eq!(
            usize::MAX.checked_add(1),
            None,
            "checked-length overflow witness"
        );
    }

    #[cfg(unix)]
    #[test]
    fn filesystem_admission_symlinks_sizes_and_abstract_reader_are_closed() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        use std::os::unix::net::UnixListener;

        let scratch = Scratch::new("filesystem");
        let contract = expected();
        let accepted = scratch.path.join("accepted.eqi");
        fs::write(&accepted, ACCEPTED_BYTES).unwrap();
        let direct = run(
            ["check".into(), accepted.as_os_str().to_owned()],
            &scratch.path,
        );
        assert_eq!(direct.status.code(), Some(0));

        let link = scratch.path.join("accepted-link.eqi");
        symlink(&accepted, &link).unwrap();
        let linked = run(["check".into(), link.as_os_str().to_owned()], &scratch.path);
        assert_eq!(linked.status.code(), Some(0));
        assert_eq!(linked.stdout, direct.stdout);

        let real_parent = scratch.path.join("real-parent");
        fs::create_dir(&real_parent).unwrap();
        fs::write(real_parent.join("accepted.eqi"), ACCEPTED_BYTES).unwrap();
        let parent_link = scratch.path.join("parent-link");
        symlink(&real_parent, &parent_link).unwrap();
        assert_eq!(
            run(
                [
                    "check".into(),
                    parent_link.join("accepted.eqi").into_os_string()
                ],
                &scratch.path
            )
            .status
            .code(),
            Some(0)
        );

        let unavailable = fixed_stderr(&contract, "unavailable");
        for path in [
            scratch.path.join("missing.eqi"),
            scratch.path.clone(),
            scratch.path.join("broken.eqi"),
        ] {
            if path.ends_with("broken.eqi") {
                symlink(scratch.path.join("absent-target"), &path).unwrap();
            }
            assert_output(
                &run(["check".into(), path.into_os_string()], &scratch.path),
                exit(&contract, "unavailable"),
                b"",
                unavailable,
            );
        }
        let socket = scratch.path.join("socket");
        let _listener = UnixListener::bind(&socket).unwrap();
        assert_output(
            &run(["check".into(), socket.into_os_string()], &scratch.path),
            66,
            b"",
            unavailable,
        );
        let loop_a = scratch.path.join("loop-a");
        let loop_b = scratch.path.join("loop-b");
        symlink(&loop_b, &loop_a).unwrap();
        symlink(&loop_a, &loop_b).unwrap();
        assert_output(
            &run(["check".into(), loop_a.into_os_string()], &scratch.path),
            66,
            b"",
            unavailable,
        );

        let unreadable = scratch.path.join("unreadable.eqi");
        fs::write(&unreadable, ACCEPTED_BYTES).unwrap();
        fs::set_permissions(&unreadable, fs::Permissions::from_mode(0o0)).unwrap();
        let unreadable_output = run(
            ["check".into(), unreadable.as_os_str().to_owned()],
            &scratch.path,
        );
        fs::set_permissions(&unreadable, fs::Permissions::from_mode(0o600)).unwrap();
        assert_output(&unreadable_output, 66, b"", unavailable);

        let invalid_utf8 = scratch.path.join("invalid-utf8.eqi");
        fs::write(&invalid_utf8, [0xff, 0xfe]).unwrap();
        assert_output(
            &run(
                ["check".into(), invalid_utf8.into_os_string()],
                &scratch.path,
            ),
            65,
            b"",
            fixed_stderr(&contract, "invalidUtf8"),
        );

        let exact = scratch.path.join("exact-limit.eqi");
        let mut exact_file = File::create(&exact).unwrap();
        exact_file.write_all(ACCEPTED_BYTES).unwrap();
        exact_file
            .write_all(&vec![b' '; 8_388_608 - ACCEPTED_BYTES.len()])
            .unwrap();
        drop(exact_file);
        let exact_output = run(["check".into(), exact.into_os_string()], &scratch.path);
        assert_eq!(
            exact_output.status.code(),
            Some(0),
            "exact maximum must reach compilation"
        );
        let over = scratch.path.join("over-limit.eqi");
        File::create(&over).unwrap().set_len(8_388_609).unwrap();
        assert_output(
            &run(["check".into(), over.into_os_string()], &scratch.path),
            65,
            b"",
            fixed_stderr(&contract, "sourceTooLarge"),
        );

        let exact_read =
            cli_main::read_bounded_for_oracle(8_388_608, &mut Cursor::new(vec![b'x'; 8_388_608]));
        let exact_read = exact_read.unwrap();
        assert_eq!(exact_read.as_ref().len(), 8_388_608);
        struct Meter {
            remaining: usize,
            consumed: usize,
        }
        impl Read for Meter {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                assert!(self.consumed + out.len() <= 8_388_609);
                let n = out.len().min(self.remaining);
                out[..n].fill(b'x');
                self.remaining -= n;
                self.consumed += n;
                Ok(n)
            }
        }
        let mut meter = Meter {
            remaining: 8_388_610,
            consumed: 0,
        };
        let grew = cli_main::read_bounded_for_oracle(8_388_608, &mut meter);
        assert!(matches!(grew, Err(cli_main::OracleReadError::TooLarge)));
        assert_eq!(meter.consumed, 8_388_609);
        struct PanicRead;
        impl Read for PanicRead {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                panic!("over-limit reader touched")
            }
        }
        let huge_reported = cli_main::read_bounded_for_oracle(u64::MAX, &mut PanicRead);
        assert!(matches!(
            huge_reported,
            Err(cli_main::OracleReadError::TooLarge)
        ));
        struct FailedRead;
        impl Read for FailedRead {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("read-error-secret"))
            }
        }
        let failed = cli_main::read_bounded_for_oracle(0, &mut FailedRead);
        assert!(matches!(
            failed,
            Err(cli_main::OracleReadError::Unavailable)
        ));
    }

    #[derive(Debug)]
    struct OracleReceipt {
        exit: i32,
        count: u64,
        stdout: Vec<u8>,
        stderr: Vec<u8>,
    }

    fn write_receipt(path: &Path, receipt: &OracleReceipt) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&receipt.exit.to_le_bytes());
        bytes.extend_from_slice(&receipt.count.to_le_bytes());
        bytes.extend_from_slice(&(receipt.stdout.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(receipt.stderr.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&receipt.stdout);
        bytes.extend_from_slice(&receipt.stderr);
        fs::write(path, bytes).unwrap();
    }

    fn read_u64(bytes: &[u8], at: &mut usize) -> u64 {
        let value = u64::from_le_bytes(bytes[*at..*at + 8].try_into().unwrap());
        *at += 8;
        value
    }

    fn read_receipt(path: &Path) -> OracleReceipt {
        let bytes = fs::read(path).expect("contained oracle child receipt");
        let mut at = 0;
        let exit = i32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        at += 4;
        let count = read_u64(&bytes, &mut at);
        let stdout_len = read_u64(&bytes, &mut at) as usize;
        let stderr_len = read_u64(&bytes, &mut at) as usize;
        let stdout = bytes[at..at + stdout_len].to_vec();
        at += stdout_len;
        let stderr = bytes[at..at + stderr_len].to_vec();
        at += stderr_len;
        assert_eq!(at, bytes.len());
        OracleReceipt {
            exit,
            count,
            stdout,
            stderr,
        }
    }

    fn child_args() -> Vec<OsString> {
        if std::env::var("EQIORA_CLI_ORACLE_MODE").as_deref() == Ok("panic-arguments") {
            panic!("{PANIC_SENTINEL}");
        }
        let mut args = vec![OsString::from("eqiora")];
        for index in 0..16 {
            let key = format!("EQIORA_CLI_ORACLE_ARG_{index}");
            let Some(value) = std::env::var_os(key) else {
                break;
            };
            args.push(value);
        }
        args
    }

    #[test]
    fn cli_oracle_child() {
        let Some(receipt_path) = std::env::var_os("EQIORA_CLI_ORACLE_RECEIPT") else {
            return;
        };
        let mode = std::env::var("EQIORA_CLI_ORACLE_MODE").unwrap();
        let expected_file = std::env::var_os("EQIORA_CLI_ORACLE_EXPECTED_FILE").map(PathBuf::from);
        let expected_filename = std::env::var("EQIORA_CLI_ORACLE_EXPECTED_FILENAME").ok();
        let expected_source = expected_file.as_ref().map(|path| fs::read(path).unwrap());
        let count = Cell::new(0u64);
        let operation = |filename: &str, source: &str| {
            count.set(count.get() + 1);
            if let Some(expected) = expected_filename.as_deref() {
                assert_eq!(filename, expected, "path spelling changed before operation");
            }
            if let Some(expected) = expected_source.as_deref()
                && mode != "valid-decoy-substitution"
            {
                assert_eq!(
                    source.as_bytes(),
                    expected,
                    "source bytes changed before operation"
                );
            }
            match mode.as_str() {
                "synthetic-overflow" => Err(vec![Diagnostic::error(Code("ER0099"), "x"); 1025]),
                "synthetic-escape" => Err(synthetic_rich(filename)),
                "synthetic-accepted-2" => synthetic_document(2),
                "synthetic-accepted-3" => synthetic_document(3),
                "valid-decoy-substitution" => ModelDocument::compile(
                    expected_file.as_ref().unwrap().to_str().unwrap(),
                    std::str::from_utf8(expected_source.as_ref().unwrap()).unwrap(),
                ),
                _ => ModelDocument::compile(filename, source),
            }
        };
        let panic_point = match std::env::var("EQIORA_CLI_ORACLE_PANIC")
            .unwrap_or_else(|_| "none".to_owned())
            .as_str()
        {
            "none" => cli_main::OraclePanicPoint::None,
            "before-operation" => cli_main::OraclePanicPoint::BeforeOperation(PANIC_SENTINEL),
            "during-projection" => cli_main::OraclePanicPoint::DuringProjection(PANIC_SENTINEL),
            other => panic!("unknown oracle panic point {other}"),
        };
        let outcome = cli_main::run_for_oracle(child_args, operation, panic_point);
        let (exit, stdout, stderr) = outcome.into_parts();
        write_receipt(
            Path::new(&receipt_path),
            &OracleReceipt {
                exit,
                count: count.get(),
                stdout,
                stderr,
            },
        );
    }

    fn run_oracle_child(
        scratch: &Scratch,
        args: &[OsString],
        mode: &str,
        panic_point: &str,
        expected_file: Option<&Path>,
        expected_filename: Option<&str>,
    ) -> OracleReceipt {
        let receipt = scratch.path.join(format!(
            "receipt-{}-{}",
            std::process::id(),
            SCRATCH_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "full::cli_oracle_child", "--test-threads=1"])
            .env("EQIORA_CLI_ORACLE_RECEIPT", &receipt)
            .env("EQIORA_CLI_ORACLE_MODE", mode)
            .env("EQIORA_CLI_ORACLE_PANIC", panic_point)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (index, arg) in args.iter().enumerate() {
            command.env(format!("EQIORA_CLI_ORACLE_ARG_{index}"), arg);
        }
        if let Some(path) = expected_file {
            command.env("EQIORA_CLI_ORACLE_EXPECTED_FILE", path);
        }
        if let Some(filename) = expected_filename {
            command.env("EQIORA_CLI_ORACLE_EXPECTED_FILENAME", filename);
        }
        let child = command.output().unwrap();
        assert!(
            child.status.success(),
            "oracle child did not contain its route: stdout={} stderr={}",
            String::from_utf8_lossy(&child.stdout),
            String::from_utf8_lossy(&child.stderr)
        );
        read_receipt(&receipt)
    }

    #[test]
    fn injected_once_operation_counts_and_source_forwarding_are_dynamic() {
        let scratch = Scratch::new("counts");
        let contract = expected();
        let accepted = scratch.path.join("accepted.eqi");
        let rejected = scratch.path.join("rejected.eqi");
        let empty = scratch.path.join("empty.eqi");
        let invalid_utf8 = scratch.path.join("invalid-utf8.eqi");
        fs::write(&accepted, ACCEPTED_BYTES).unwrap();
        fs::write(&rejected, REJECTED_BYTES).unwrap();
        fs::write(&empty, []).unwrap();
        fs::write(&invalid_utf8, [0xff]).unwrap();
        let accepted_text = accepted.to_str().unwrap();
        let rejected_text = rejected.to_str().unwrap();

        let accepted_result = run_oracle_child(
            &scratch,
            &["check".into(), accepted.as_os_str().to_owned()],
            "delegate",
            "none",
            Some(&accepted),
            Some(accepted_text),
        );
        assert_eq!(accepted_result.count, 1);
        assert_eq!(accepted_result.exit, 0);
        assert!(accepted_result.stdout.starts_with(b"accepted "));
        assert!(accepted_result.stderr.is_empty());
        let mut projections = Vec::new();
        let accepted_args = ["check".into(), accepted.as_os_str().to_owned()];
        for (mode, rate) in [("synthetic-accepted-2", 2), ("synthetic-accepted-3", 3)] {
            let document = synthetic_document(rate).unwrap();
            let expected =
                format!("accepted {}\n", document.structural_fingerprint().unwrap()).into_bytes();
            let result = run_oracle_child(&scratch, &accepted_args, mode, "none", None, None);
            assert_eq!(result.count, 1);
            assert_eq!(result.exit, 0);
            assert_eq!(result.stdout, expected);
            assert!(result.stderr.is_empty());
            projections.push(expected);
        }
        assert_ne!(projections[0], projections[1]);

        let rejected_result = run_oracle_child(
            &scratch,
            &["check".into(), rejected.as_os_str().to_owned()],
            "delegate",
            "none",
            Some(&rejected),
            Some(rejected_text),
        );
        assert_eq!(rejected_result.count, 1);
        assert_eq!(rejected_result.exit, 1);
        assert!(rejected_result.stdout.is_empty());
        assert_eq!(
            rejected_result.stderr,
            reference_render(
                &ModelDocument::compile(
                    rejected_text,
                    std::str::from_utf8(REJECTED_BYTES).unwrap()
                )
                .unwrap_err()
            )
            .unwrap()
        );

        let empty_result = run_oracle_child(
            &scratch,
            &["check".into(), empty.as_os_str().to_owned()],
            "delegate",
            "none",
            Some(&empty),
            Some(empty.to_str().unwrap()),
        );
        assert_eq!(empty_result.count, 1, "empty source is operation data");
        assert_eq!(empty_result.exit, 1);

        let pre_operation: &[&[&str]] = &[
            &["--help"],
            &["check", "--help"],
            &["--version"],
            &["check"],
            &["unknown"],
            &["check", "missing.eqi"],
        ];
        for args in pre_operation {
            let args = args.iter().map(OsString::from).collect::<Vec<_>>();
            let result = run_oracle_child(&scratch, &args, "delegate", "none", None, None);
            assert_eq!(result.count, 0, "pre-operation route executed for {args:?}");
        }

        let invalid_utf8_result = run_oracle_child(
            &scratch,
            &["check".into(), invalid_utf8.into_os_string()],
            "delegate",
            "none",
            None,
            None,
        );
        assert_eq!(invalid_utf8_result.count, 0);
        assert_eq!(invalid_utf8_result.exit, 65);
        let overlong = run_oracle_child(
            &scratch,
            &["check".into(), "x".repeat(4097).into()],
            "delegate",
            "none",
            None,
            None,
        );
        assert_eq!(overlong.count, 0);
        assert_eq!(overlong.exit, 64);

        let over = scratch.path.join("over.eqi");
        File::create(&over).unwrap().set_len(8_388_609).unwrap();
        let over_result = run_oracle_child(
            &scratch,
            &["check".into(), over.into_os_string()],
            "delegate",
            "none",
            None,
            None,
        );
        assert_eq!(over_result.count, 0);
        assert_eq!(over_result.exit, exit(&contract, "data"));

        let exact = scratch.path.join("exact-operation-limit.eqi");
        let mut exact_file = File::create(&exact).unwrap();
        exact_file.write_all(ACCEPTED_BYTES).unwrap();
        exact_file
            .write_all(&vec![b' '; 8_388_608 - ACCEPTED_BYTES.len()])
            .unwrap();
        drop(exact_file);
        let exact_result = run_oracle_child(
            &scratch,
            &["check".into(), exact.as_os_str().to_owned()],
            "synthetic-escape",
            "none",
            Some(&exact),
            Some(exact.to_str().unwrap()),
        );
        assert_eq!(exact_result.count, 1);
        assert_eq!(exact_result.exit, 1);

        let normalized = scratch.path.join("normalization.eqi");
        let normalized_bytes = [
            "\u{feff}// é\r\n  trailing  \r\n".as_bytes(),
            REJECTED_BYTES,
        ]
        .concat();
        fs::write(&normalized, &normalized_bytes).unwrap();
        let dotted = format!("{}/./normalization.eqi", scratch.path.display());
        let forwarded = run_oracle_child(
            &scratch,
            &["check".into(), dotted.clone().into()],
            "synthetic-escape",
            "none",
            Some(&normalized),
            Some(&dotted),
        );
        assert_eq!(forwarded.count, 1);
        assert_eq!(forwarded.exit, 1);
        assert_eq!(
            forwarded.stderr,
            reference_render(&synthetic_rich(&dotted)).unwrap()
        );

        let overflow = run_oracle_child(
            &scratch,
            &["check".into(), accepted.into_os_string()],
            "synthetic-overflow",
            "none",
            None,
            None,
        );
        assert_eq!(overflow.count, 1);
        assert_eq!(overflow.exit, 1);
        assert!(overflow.stdout.is_empty());
        assert_eq!(
            overflow.stderr,
            fixed_stderr(&contract, "diagnosticOverflow")
        );
    }

    #[test]
    fn both_precommit_panic_witnesses_are_silent_and_atomic() {
        let scratch = Scratch::new("panics");
        let contract = expected();
        let accepted = scratch.path.join("panic-payload-secret.eqi");
        fs::write(&accepted, ACCEPTED_BYTES).unwrap();
        for (mode, point, expected_count) in [
            ("panic-arguments", "none", 0),
            ("delegate", "before-operation", 0),
            ("delegate", "during-projection", 1),
        ] {
            let result = run_oracle_child(
                &scratch,
                &["check".into(), accepted.as_os_str().to_owned()],
                mode,
                point,
                None,
                None,
            );
            assert_eq!(result.count, expected_count, "panic point {point}");
            assert_eq!(result.exit, exit(&contract, "internal"));
            assert!(result.stdout.is_empty());
            assert_eq!(result.stderr, fixed_stderr(&contract, "internal"));
            let text = String::from_utf8(result.stderr).unwrap();
            for forbidden in [
                "panic-payload-secret",
                PANIC_SENTINEL,
                "panicked at",
                "backtrace",
                "accepted ",
                ACCEPTED_SENTINEL,
            ] {
                assert!(!text.contains(forbidden));
            }
        }
    }

    #[test]
    fn output_write_failures_use_the_fixed_io_exit_without_payload_echo() {
        let scratch = Scratch::new("output-io");
        let accepted = scratch.path.join("accepted-output-secret.eqi");
        fs::write(&accepted, ACCEPTED_BYTES).unwrap();

        let read_only_stdout = File::open(binary()).unwrap();
        let accepted_failure = Command::new(binary())
            .args([OsStr::new("check"), accepted.as_os_str()])
            .current_dir(&scratch.path)
            .stdout(Stdio::from(read_only_stdout))
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert_eq!(accepted_failure.status.code(), Some(74));
        let accepted_stderr = String::from_utf8_lossy(&accepted_failure.stderr);
        assert!(!accepted_stderr.contains("accepted-output-secret"));
        assert!(!accepted_stderr.contains(ACCEPTED_SENTINEL));

        let read_only_stderr = File::open(binary()).unwrap();
        let rejected_failure = Command::new(binary())
            .arg("invalid-output-secret")
            .current_dir(&scratch.path)
            .stdout(Stdio::piped())
            .stderr(Stdio::from(read_only_stderr))
            .output()
            .unwrap();
        assert_eq!(rejected_failure.status.code(), Some(74));
        assert!(
            !rejected_failure
                .stdout
                .windows("invalid-output-secret".len())
                .any(|part| part == b"invalid-output-secret")
        );
    }
}
