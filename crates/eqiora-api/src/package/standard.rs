//! Explicit transport of the exact semantic packages shipped with the distribution.

use eqiora_package::{
    BundleRoleV1, NormalizedRelativePath, PackageManifestV1, PackageReleaseV1, PackageSourcesV1,
    SourceFileV1,
};

use super::{PackagePreparationError, prepare_package_release_v1};

pub(super) fn closure(name: &str) -> Result<Vec<PackageReleaseV1>, PackagePreparationError> {
    let mut releases = Vec::new();
    if matches!(
        name,
        "Eqiora.Mechanics.Interfaces"
            | "Eqiora.Fluid.Incompressible"
            | "Eqiora.Solid.LinearElasticity"
    ) {
        let mechanics = sources(
            include_bytes!("../../packages/Eqiora.Mechanics.Interfaces/package.json"),
            include_bytes!("../../packages/Eqiora.Mechanics.Interfaces/README.md"),
            "src/interfaces.eqi",
            include_bytes!("../../packages/Eqiora.Mechanics.Interfaces/src/interfaces.eqi"),
        )?;
        releases.push(prepare_package_release_v1(mechanics, &[])?);
    }
    let source = match name {
        "Eqiora.Mechanics.Interfaces" => None,
        "Eqiora.Electrical.Basic" => Some(sources(
            include_bytes!("../../packages/Eqiora.Electrical.Basic/package.json"),
            include_bytes!("../../packages/Eqiora.Electrical.Basic/README.md"),
            "src/basic.eqi",
            include_bytes!("../../packages/Eqiora.Electrical.Basic/src/basic.eqi"),
        )?),
        "Eqiora.Controls.Sampled" => Some(sources(
            include_bytes!("../../packages/Eqiora.Controls.Sampled/package.json"),
            include_bytes!("../../packages/Eqiora.Controls.Sampled/README.md"),
            "src/sampled.eqi",
            include_bytes!("../../packages/Eqiora.Controls.Sampled/src/sampled.eqi"),
        )?),
        "Eqiora.Stochastic.Processes" => Some(sources(
            include_bytes!("../../packages/Eqiora.Stochastic.Processes/package.json"),
            include_bytes!("../../packages/Eqiora.Stochastic.Processes/README.md"),
            "src/processes.eqi",
            include_bytes!("../../packages/Eqiora.Stochastic.Processes/src/processes.eqi"),
        )?),
        "Eqiora.Fluid.Incompressible" => Some(sources(
            include_bytes!("../../packages/Eqiora.Fluid.Incompressible/package.json"),
            include_bytes!("../../packages/Eqiora.Fluid.Incompressible/README.md"),
            "src/incompressible.eqi",
            include_bytes!("../../packages/Eqiora.Fluid.Incompressible/src/incompressible.eqi"),
        )?),
        "Eqiora.Solid.LinearElasticity" => Some(sources(
            include_bytes!("../../packages/Eqiora.Solid.LinearElasticity/package.json"),
            include_bytes!("../../packages/Eqiora.Solid.LinearElasticity/README.md"),
            "src/linear_elasticity.eqi",
            include_bytes!(
                "../../packages/Eqiora.Solid.LinearElasticity/src/linear_elasticity.eqi"
            ),
        )?),
        _ => {
            return Err(PackagePreparationError::LocalDirectoryGraph(format!(
                "distribution has no bundled package `{name}`"
            )));
        }
    };
    if let Some(source) = source {
        releases.push(prepare_package_release_v1(source, &releases)?);
    }
    let identity = releases
        .last()
        .expect("nonempty bundled closure")
        .package_identity()?;
    if identity.name.as_str() != name {
        return Err(PackagePreparationError::LocalDirectoryGraph(format!(
            "bundled package name `{name}` differs from exact shipped release `{}@{}`",
            identity.name, identity.version
        )));
    }
    Ok(releases)
}

fn sources(
    manifest: &[u8],
    readme: &[u8],
    source_path: &str,
    source: &[u8],
) -> Result<PackageSourcesV1, PackagePreparationError> {
    Ok(PackageSourcesV1::new(
        PackageManifestV1::from_json(manifest)?,
        vec![
            SourceFileV1::new(
                NormalizedRelativePath::parse("README.md")?,
                BundleRoleV1::Documentation,
                readme.to_vec(),
            ),
            SourceFileV1::new(
                NormalizedRelativePath::parse(source_path)?,
                BundleRoleV1::ModelSource,
                source.to_vec(),
            ),
        ],
    )?)
}
