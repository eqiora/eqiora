use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

const EXACT_CYLINDER_SOURCE: &str = "../../examples/steady-flow-past-cylinder.eqi";
const TRANSIENT_CYLINDER_SOURCE: &str = "../../examples/transient-flow-past-cylinder.eqi";
const MIXED_BOUNDARY_ELASTICITY_MODEL: &str = "../../examples/mixed-boundary-elasticity.eqi";
const FIXED_REFERENCE_FSI_MODEL: &str = "../../examples/fixed-reference-fsi.eqi";
const RESEARCH_CONDUCTIVITY_SOURCE: &str =
    "../../examples/research-conductivity/research_workflow.py";
const RESEARCH_CONDUCTIVITY_MODEL: &str = "../../examples/research-conductivity/conductivity.eqi";

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed={EXACT_CYLINDER_SOURCE}");
    println!("cargo:rerun-if-changed={TRANSIENT_CYLINDER_SOURCE}");
    println!("cargo:rerun-if-changed={MIXED_BOUNDARY_ELASTICITY_MODEL}");
    println!("cargo:rerun-if-changed={FIXED_REFERENCE_FSI_MODEL}");
    println!("cargo:rerun-if-changed={RESEARCH_CONDUCTIVITY_SOURCE}");
    println!("cargo:rerun-if-changed={RESEARCH_CONDUCTIVITY_MODEL}");

    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR")
            .ok_or("Cargo did not provide CARGO_MANIFEST_DIR to the Python package build")?,
    );
    let output_dir = PathBuf::from(
        env::var_os("OUT_DIR")
            .ok_or("Cargo did not provide OUT_DIR to the Python package build")?,
    );
    fs::copy(
        manifest_dir.join(EXACT_CYLINDER_SOURCE),
        output_dir.join("steady-flow-past-cylinder.eqi"),
    )?;
    fs::copy(
        manifest_dir.join(TRANSIENT_CYLINDER_SOURCE),
        output_dir.join("transient-flow-past-cylinder.eqi"),
    )?;
    fs::copy(
        manifest_dir.join(MIXED_BOUNDARY_ELASTICITY_MODEL),
        output_dir.join("mixed-boundary-elasticity.eqi"),
    )?;
    fs::copy(
        manifest_dir.join(FIXED_REFERENCE_FSI_MODEL),
        output_dir.join("fixed-reference-fsi.eqi"),
    )?;
    fs::copy(
        manifest_dir.join(RESEARCH_CONDUCTIVITY_SOURCE),
        output_dir.join("research_conductivity.py"),
    )?;
    fs::copy(
        manifest_dir.join(RESEARCH_CONDUCTIVITY_MODEL),
        output_dir.join("conductivity.eqi"),
    )?;
    Ok(())
}
