//! Repository development tasks.

mod architecture;
mod facade;

use std::collections::{HashMap, HashSet};
use std::env;
use std::path::Path;
use std::process::{Command, ExitCode};

fn main() -> ExitCode {
    match env::args().nth(1).as_deref() {
        Some("check-layers") => match check_layers() {
            Ok(()) => {
                println!("dependency layers are valid");
                ExitCode::SUCCESS
            }
            Err(message) => {
                eprintln!("{message}");
                ExitCode::FAILURE
            }
        },
        Some("check-facade") => match facade::check() {
            Ok(()) => {
                println!("public facade matches api/eqiora-facade-v1.json");
                ExitCode::SUCCESS
            }
            Err(message) => {
                eprintln!("{message}");
                ExitCode::FAILURE
            }
        },
        Some("check-architecture") => match architecture::check() {
            Ok(()) => ExitCode::SUCCESS,
            Err(message) => {
                eprintln!("{message}");
                ExitCode::FAILURE
            }
        },
        Some(command) => {
            eprintln!("unknown xtask command: {command}");
            usage();
            ExitCode::FAILURE
        }
        None => {
            usage();
            ExitCode::FAILURE
        }
    }
}

fn usage() {
    eprintln!("usage: cargo xtask <check-architecture|check-facade|check-layers>");
}

fn check_layers() -> Result<(), String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| "cannot locate workspace root".to_owned())?;
    let output = Command::new(env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(root)
        .output()
        .map_err(|error| format!("failed to run cargo metadata: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }

    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("invalid cargo metadata: {error}"))?;
    let members: HashSet<&str> = metadata["workspace_members"]
        .as_array()
        .ok_or_else(|| "metadata has no workspace_members".to_owned())?
        .iter()
        .filter_map(serde_json::Value::as_str)
        .collect();
    let packages = metadata["packages"]
        .as_array()
        .ok_or_else(|| "metadata has no packages".to_owned())?;
    let workspace_names: HashSet<&str> = packages
        .iter()
        .filter(|package| {
            package["id"]
                .as_str()
                .is_some_and(|id| members.contains(id))
        })
        .filter_map(|package| package["name"].as_str())
        .collect();
    let layers = checked_layer_map(&workspace_names, LAYER_DECLARATIONS)?;
    let mut violations = Vec::new();

    for package in packages {
        let Some(id) = package["id"].as_str() else {
            continue;
        };
        if !members.contains(id) {
            continue;
        }
        let Some(name) = package["name"].as_str() else {
            continue;
        };
        if name == "eqiora" || name == "xtask" {
            continue;
        }
        let Some(&layer) = layers.get(name) else {
            continue;
        };
        let Some(dependencies) = package["dependencies"].as_array() else {
            continue;
        };
        for dependency in dependencies {
            let Some(dependency_name) = dependency["name"].as_str() else {
                continue;
            };
            if !workspace_names.contains(dependency_name) {
                continue;
            }
            if name == "eqiora-python" && forbidden_python_dependency(dependency_name) {
                violations.push(format!(
                    "eqiora-python must cross the public eqiora facade, not depend on {dependency_name}"
                ));
                continue;
            }
            let Some(&dependency_layer) = layers.get(dependency_name) else {
                continue;
            };
            if dependency_layer > layer {
                violations.push(format!(
                    "{name} (L{layer}) must not depend on {dependency_name} (L{dependency_layer})"
                ));
            } else if dependency_layer == layer
                && !same_layer_dependency_is_allowed(name, dependency_name)
            {
                violations.push(format!(
                    "{name} and {dependency_name} are both L{layer}; this dependency requires an ADR"
                ));
            }
        }
    }

    if violations.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "dependency layer violations:\n{}",
            violations.join("\n")
        ))
    }
}

fn forbidden_python_dependency(dependency: &str) -> bool {
    // The private PyO3 adapter owns no numerical meaning; it projects the one
    // common Plan representation into Python. Every other Eqiora dependency
    // must cross the public facade.
    dependency.starts_with("eqiora-") && dependency != "eqiora-numerics"
}

fn same_layer_dependency_is_allowed(package: &str, dependency: &str) -> bool {
    matches!(
        (package, dependency),
        // RFC 0010: realization/distribution compose pure solver vocabulary.
        ("eqiora-schema", "eqiora-core")
            | ("eqiora-realization", "eqiora-solver")
            | ("eqiora-distributed", "eqiora-solver")
            | ("eqiora-assembly", "eqiora-solver")
            // RFC 0060: spatial distribution joins accepted mesh ownership,
            // canonical assembly deltas, and pure distributed algebra without
            // moving those meanings into any input crate.
            | ("eqiora-spatial-distribution", "eqiora-assembly")
            | ("eqiora-spatial-distribution", "eqiora-distributed")
            | ("eqiora-spatial-distribution", "eqiora-meshing")
            | ("eqiora-spatial-distribution", "eqiora-solver")
            // RFC 0058: execution consumes the accepted portable Realization
            // plus sole distributed/device/solver algebra evidence without any
            // of those contracts owning a reverse edge.
            | ("eqiora-execution", "eqiora-device")
            | ("eqiora-execution", "eqiora-distributed")
            | ("eqiora-execution", "eqiora-realization")
            | ("eqiora-execution", "eqiora-solver")
            // RFC 0053: the public canonical transient bridge consumes the
            // immutable mesh envelope as one authenticated revision instead
            // of accepting independently forgeable digest and mesh values.
            | ("eqiora-numerics", "eqiora-artifact")
            // #535: the common numerical resolver composes bounded external
            // mesh observations and canonical CPU lowering. Both adapters
            // remain one-way leaves; neither owns numerical policy.
            | ("eqiora-numerics", "eqiora-io-gmsh")
            | ("eqiora-numerics", "eqiora-runtime")
            // RFC 0063: this single L3 composition owns the joint lifecycle
            // of otherwise isolated MPI transport and CUDA action adapters.
            | ("eqiora-backend-mpi-cuda", "eqiora-backend-mpi")
            | ("eqiora-backend-mpi-cuda", "eqiora-backend-cuda")
            // RFC 0049: Geometry Identity composes the existing revision-local
            // mesh entity vocabulary without moving mesh meaning into geometry.
            | ("eqiora-geometry", "eqiora-meshing")
            // RFC 0080: semantic admission derives non-forgeable spatial
            // support from canonical geometry rather than caller-owned facts.
            | ("eqiora-sem", "eqiora-geometry")
            // #1002; docs/architecture.md, Common initialization derivative
            // owner: one-way reuse of pure typed derivatives, never a backend
            // or a reverse IR-to-SEM dependency.
            | ("eqiora-sem", "eqiora-ir")
            // #1002: common execution shares the pure DAE regularity proof;
            // no time backend or integration policy enters SEM.
            | ("eqiora-sem", "eqiora-time")
            // RFC 0080/#520: external Component lowering consumes the same
            // opaque canonical-geometry projection so no caller can forge a
            // digest, dimension, or entity-set fact at the compiler boundary.
            | ("eqiora-compiler", "eqiora-geometry")
    )
}

const LAYER_DECLARATIONS: &[(&str, u8)] = &[
    ("eqiora-core", 0),
    ("eqiora-schema", 0),
    ("eqiora-graph", 1),
    ("eqiora-lang", 1),
    ("eqiora-sem", 2),
    ("eqiora-ir", 2),
    ("eqiora-meshing", 2),
    ("eqiora-compiler", 2),
    ("eqiora-geometry", 2),
    ("eqiora-realization", 2),
    ("eqiora-device", 2),
    ("eqiora-distributed", 2),
    ("eqiora-execution", 2),
    ("eqiora-solver", 2),
    ("eqiora-time", 2),
    ("eqiora-assembly", 2),
    ("eqiora-spatial-distribution", 2),
    ("eqiora-numerics", 3),
    ("eqiora-package", 3),
    ("eqiora-differentiation", 3),
    ("eqiora-backend-faer", 3),
    ("eqiora-backend-mpi", 3),
    ("eqiora-backend-mpi-cuda", 3),
    ("eqiora-backend-diffsol", 3),
    ("eqiora-backend-cuda", 3),
    ("eqiora-io-gmsh", 3),
    ("eqiora-io-hdf5", 3),
    ("eqiora-io-vtu", 3),
    ("eqiora-io-xdmf", 3),
    ("eqiora-cad-truck", 3),
    ("eqiora-runtime", 3),
    ("eqiora-backend-rayon", 3),
    ("eqiora-artifact", 3),
    ("eqiora-verify", 4),
    ("eqiora-api", 4),
    ("eqiora-language-server", 4),
    ("eqiora-python", 4),
];

fn checked_layer_map(
    workspace_names: &HashSet<&str>,
    declarations: &[(&'static str, u8)],
) -> Result<HashMap<&'static str, u8>, String> {
    let mut layers = HashMap::with_capacity(declarations.len());
    let mut duplicates = Vec::new();
    for &(name, layer) in declarations {
        if layers.insert(name, layer).is_some() {
            duplicates.push(name);
        }
    }
    duplicates.sort_unstable();
    duplicates.dedup();

    let mut missing = workspace_names
        .iter()
        .filter(|name| !layer_exempt(name) && !layers.contains_key(**name))
        .copied()
        .collect::<Vec<_>>();
    missing.sort_unstable();

    let mut stale = layers
        .keys()
        .filter(|name| !workspace_names.contains(**name))
        .copied()
        .collect::<Vec<_>>();
    stale.sort_unstable();

    let mut forbidden = layers
        .keys()
        .filter(|name| layer_exempt(name))
        .copied()
        .collect::<Vec<_>>();
    forbidden.sort_unstable();

    let mut failures = Vec::new();
    if !duplicates.is_empty() {
        failures.push(format!(
            "duplicate dependency-layer declarations: {}",
            duplicates.join(", ")
        ));
    }
    if !missing.is_empty() {
        failures.push(format!(
            "workspace crates without dependency-layer declarations: {}",
            missing.join(", ")
        ));
    }
    if !stale.is_empty() {
        failures.push(format!(
            "dependency-layer declarations without workspace crates: {}",
            stale.join(", ")
        ));
    }
    if !forbidden.is_empty() {
        failures.push(format!(
            "dependency-layer declarations for exempt workspace crates: {}",
            forbidden.join(", ")
        ));
    }
    if failures.is_empty() {
        Ok(layers)
    } else {
        Err(failures.join("\n"))
    }
}

fn layer_exempt(name: &str) -> bool {
    matches!(name, "eqiora" | "xtask")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_has_one_private_common_plan_dependency() {
        assert!(!forbidden_python_dependency("eqiora"));
        assert!(!forbidden_python_dependency("eqiora-numerics"));
        assert!(forbidden_python_dependency("eqiora-runtime"));
        assert!(forbidden_python_dependency("eqiora-meshing"));
    }

    #[test]
    fn layer_declarations_fail_closed_for_unknown_workspace_crates() {
        let workspace = HashSet::from(["eqiora-core", "eqiora-new-adapter", "eqiora", "xtask"]);
        let error = checked_layer_map(&workspace, &[("eqiora-core", 0)]).unwrap_err();
        assert_eq!(
            error,
            "workspace crates without dependency-layer declarations: eqiora-new-adapter"
        );
    }

    #[test]
    fn duplicate_layer_declarations_are_not_silently_overwritten() {
        let workspace = HashSet::from(["eqiora-core"]);
        let error =
            checked_layer_map(&workspace, &[("eqiora-core", 0), ("eqiora-core", 1)]).unwrap_err();
        assert_eq!(
            error,
            "duplicate dependency-layer declarations: eqiora-core"
        );
    }

    #[test]
    fn declarations_for_nonexistent_future_crates_fail_closed() {
        let workspace = HashSet::from(["eqiora-core", "eqiora", "xtask"]);
        let error = checked_layer_map(
            &workspace,
            &[("eqiora-core", 0), ("eqiora-future-provider", 3)],
        )
        .unwrap_err();
        assert_eq!(
            error,
            "dependency-layer declarations without workspace crates: eqiora-future-provider"
        );
    }

    #[test]
    fn declarations_for_unlayered_facade_and_tool_fail_closed() {
        let workspace = HashSet::from(["eqiora-core", "eqiora", "xtask"]);
        let error = checked_layer_map(
            &workspace,
            &[("eqiora-core", 0), ("xtask", 4), ("eqiora", 4)],
        )
        .unwrap_err();
        assert_eq!(
            error,
            "dependency-layer declarations for exempt workspace crates: eqiora, xtask"
        );
    }
}
