//! Command-line frontend for repository verification evidence.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use eqiora_verify::{
    CommandKind, EvidenceEnvironment, ExecutionPolicy, Request, RunnerKind, SystemEvidenceRunner,
    capability_evidence_index, execute,
};

#[derive(Debug, Parser)]
#[command(version, about = "Validate and run Eqiora verification evidence")]
struct Cli {
    /// Repository to inspect; defaults to the current working directory.
    #[arg(long, global = true, default_value = ".")]
    root: PathBuf,

    /// Stable report representation.
    #[arg(long, global = true, value_enum, default_value_t = OutputFormat::Human)]
    format: OutputFormat,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputFormat {
    Human,
    Json,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Environment {
    HostCpu,
    PhysicalMpiCuda,
}

impl From<Environment> for EvidenceEnvironment {
    fn from(value: Environment) -> Self {
        match value {
            Environment::HostCpu => Self::HostCpu,
            Environment::PhysicalMpiCuda => Self::PhysicalMpiCuda,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum RunnerKindArgument {
    Cargo,
    PythonInstalledWheel,
}

impl From<RunnerKindArgument> for RunnerKind {
    fn from(value: RunnerKindArgument) -> Self {
        match value {
            RunnerKindArgument::Cargo => Self::Cargo,
            RunnerKindArgument::PythonInstalledWheel => Self::PythonInstalledWheel,
        }
    }
}

#[derive(Debug, Subcommand)]
enum Command {
    /// List validated cases in deterministic ID order.
    List {
        /// Limit the report to these exact case IDs.
        #[arg(long)]
        case: Vec<String>,
    },
    /// Project validated case manifests into a capability-to-evidence index.
    Index {
        /// Limit the index to one exact capability identifier.
        #[arg(long)]
        capability: Option<String>,
    },
    /// Validate manifests, evidence artifacts, and typed targets without running them.
    Check {
        /// Limit validation output to these exact case IDs.
        #[arg(long)]
        case: Vec<String>,
    },
    /// Run structured evidence targets without invoking a shell.
    Run {
        /// Limit execution to these exact case IDs.
        #[arg(long)]
        case: Vec<String>,
        /// Maximum number of evidence targets executing at once.
        #[arg(
            long,
            value_parser = parse_jobs,
            default_value_t = default_jobs()
        )]
        jobs: usize,
        /// Continue after an evidence failure.
        #[arg(long, conflicts_with = "fail_fast")]
        keep_going: bool,
        /// Stop after the first evidence failure (the default).
        #[arg(long, conflicts_with = "keep_going")]
        fail_fast: bool,
        /// Run only evidence declared for this exact environment.
        #[arg(long, value_enum)]
        environment: Option<Environment>,
        /// Run only evidence using this exact runner kind.
        #[arg(long, value_enum)]
        runner_kind: Option<RunnerKindArgument>,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let root = match cli.root.canonicalize() {
        Ok(root) => root,
        Err(error) => {
            eprintln!(
                "cannot resolve verification repository {}: {error}",
                cli.root.display()
            );
            return ExitCode::FAILURE;
        }
    };
    let request = match cli.command {
        Command::Index { capability } => {
            let index = capability_evidence_index(&root, capability.as_deref());
            match cli.format {
                OutputFormat::Human => print!("{}", index.render_human()),
                OutputFormat::Json => match serde_json::to_string(&index) {
                    Ok(json) => println!("{json}"),
                    Err(error) => {
                        eprintln!("cannot serialize capability evidence index: {error}");
                        return ExitCode::FAILURE;
                    }
                },
            }
            return if index.success {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            };
        }
        Command::List { case } => Request::new(CommandKind::List, case, ExecutionPolicy::FailFast),
        Command::Check { case } => {
            Request::new(CommandKind::Check, case, ExecutionPolicy::FailFast)
        }
        Command::Run {
            case,
            jobs,
            keep_going,
            fail_fast: _,
            environment,
            runner_kind,
        } => {
            let mut request = Request::new(
                CommandKind::Run,
                case,
                if keep_going {
                    ExecutionPolicy::KeepGoing
                } else {
                    ExecutionPolicy::FailFast
                },
            )
            .with_jobs(jobs);
            if let Some(environment) = environment {
                request = request.for_environment(environment.into());
            }
            if let Some(runner_kind) = runner_kind {
                request = request.for_runner_kind(runner_kind.into());
            }
            request
        }
    };
    let report = execute(&root, &request, &SystemEvidenceRunner::from_environment());

    match cli.format {
        OutputFormat::Human => print!("{}", report.render_human()),
        OutputFormat::Json => match serde_json::to_string(&report) {
            Ok(json) => println!("{json}"),
            Err(error) => {
                eprintln!("cannot serialize verification report: {error}");
                return ExitCode::FAILURE;
            }
        },
    }

    if report.success {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn default_jobs() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
}

fn parse_jobs(value: &str) -> Result<usize, String> {
    value
        .parse::<std::num::NonZeroUsize>()
        .map(std::num::NonZeroUsize::get)
        .map_err(|_| "jobs must be a positive integer".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evidence_execution_has_one_command() {
        let current = Cli::try_parse_from(["eqiora-verify", "run"]).unwrap();
        assert!(matches!(current.command, Command::Run { .. }));
        let retired = Cli::try_parse_from(["eqiora-verify", "verify"]).unwrap_err();
        assert_eq!(retired.kind(), clap::error::ErrorKind::InvalidSubcommand);
    }

    #[test]
    fn case_selector_is_repeatable() {
        let cli = Cli::try_parse_from([
            "eqiora-verify",
            "run",
            "--case",
            "z.case",
            "--case",
            "a.case",
            "--case",
            "z.case",
        ])
        .unwrap();
        let Command::Run { case, .. } = cli.command else {
            panic!("fixture selects run");
        };
        assert_eq!(case, ["z.case", "a.case", "z.case"]);
        let request = Request::new(CommandKind::Run, case, ExecutionPolicy::FailFast);
        assert_eq!(
            request,
            Request::new(
                CommandKind::Run,
                vec!["a.case".to_owned(), "z.case".to_owned()],
                ExecutionPolicy::FailFast,
            )
        );
    }
}
