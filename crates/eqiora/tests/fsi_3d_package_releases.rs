use eqiora::kernel::BoundarySide;
use eqiora::package::{
    BundleEntryV1, BundleRoleV1, ExactVersion, InMemoryPackageStore, NormalizedRelativePath,
    PackageDependencyV1, PackageManifestV1, PackageReleaseV1, PackageSourcesV1,
    PackagedModelDocument, QualifiedName, ResolutionRecordV1, SourceFileV1,
    prepare_package_release_v1,
};
use eqiora_numerics::{ale::AleFsiCartesianModel, ale::lower_ale_fsi_cartesian_3d};

#[path = "support/embedded_package.rs"]
mod embedded_package;

const PACKAGED_ROOT: &str =
    include_str!("../../../verify/fsi/fixed-topology-ale-monolithic-3d/models/packaged.eqi");
const DIRECT_SOURCE: &str =
    include_str!("../../../verify/fsi/fixed-topology-ale-monolithic-3d/models/direct.eqi");

fn prepare_release(
    package: &str,
    version: &str,
    dependencies: &[PackageReleaseV1],
) -> PackageReleaseV1 {
    let sources = embedded_package::public_sources(package);
    prepare_package_release_v1(sources, dependencies)
        .unwrap_or_else(|error| panic!("prepare package release {package} {version}: {error:?}"))
}
#[test]
fn prepares_method_neutral_three_dimensional_fluid_and_solid_releases() {
    let mechanics = prepare_release("Eqiora.Mechanics.Interfaces", "0.3.0", &[]);
    for (package, version) in [
        ("Eqiora.Fluid.Incompressible", "0.6.0"),
        ("Eqiora.Solid.LinearElasticity", "0.6.0"),
    ] {
        let release = prepare_release(package, version, std::slice::from_ref(&mechanics));
        let identity = release.package_identity().expect("package identity");
        assert_eq!(identity.name.as_str(), package);
        assert_eq!(identity.version.as_str(), version);
    }
}

#[test]
fn exact_package_graph_lowers_to_three_dimensional_ale_fsi_roles() {
    let mechanics = prepare_release("Eqiora.Mechanics.Interfaces", "0.3.0", &[]);
    let fluid = prepare_release(
        "Eqiora.Fluid.Incompressible",
        "0.6.0",
        std::slice::from_ref(&mechanics),
    );
    let solid = prepare_release(
        "Eqiora.Solid.LinearElasticity",
        "0.6.0",
        std::slice::from_ref(&mechanics),
    );
    let root = prepare_root(&mechanics, &fluid, &solid);
    let resolution = ResolutionRecordV1::from_exact_releases(
        &root,
        &[mechanics.clone(), fluid.clone(), solid.clone()],
    )
    .expect("derive exact offline resolution");
    let mut store = InMemoryPackageStore::default();
    for release in [&mechanics, &fluid, &solid, &root] {
        store.insert(release).expect("install exact release");
    }
    let document = PackagedModelDocument::compile_locked(&store, &resolution, "Main")
        .expect("compile exact 3D FSI package graph offline");
    let model = lower_ale_fsi_cartesian_3d(document.model().program())
        .expect("package-expanded canonical 3D ALE FSI roles lower");
    let direct = eqiora::api::ModelDocument::compile("direct-ale-fsi-3d.eqi", DIRECT_SOURCE)
        .expect("direct 3D ALE FSI Model compiles");
    let direct_model = lower_ale_fsi_cartesian_3d(direct.program())
        .expect("direct canonical 3D ALE FSI roles lower");
    assert_same_canonical_roles(&direct_model, &model);

    assert_eq!(model.fluid().bounds(), &[[0.0, 1.0]; 3]);
    assert_eq!(
        model.solid().continuum().bounds(),
        &[[1.0, 2.0], [0.0, 1.0], [0.0, 1.0]]
    );
    assert_eq!(model.fluid().mass_density(), 1.0);
    assert_eq!(model.fluid().dynamic_viscosity(), 0.2);
    assert_eq!(model.solid().mass_density(), 1.0);
    assert_eq!(model.solid().continuum().shear_modulus(), 2.0);
    assert_eq!(model.solid().continuum().first_lame_parameter(), 1.0);
    assert_eq!(model.interface().axis(), 0);
    assert_eq!(
        model
            .interface()
            .endpoint(model.fluid().domain())
            .unwrap()
            .side(),
        BoundarySide::Upper
    );
    assert_eq!(
        model
            .interface()
            .endpoint(model.solid().continuum().domain())
            .unwrap()
            .side(),
        BoundarySide::Lower
    );
    assert_eq!(
        model
            .fluid()
            .conservative_body_force(&[0.25, 0.5, 0.75])
            .expect("fluid force"),
        [0.0; 3]
    );
    assert_eq!(
        model
            .solid()
            .continuum()
            .conservative_body_force(&[1.25, 0.5, 0.75])
            .expect("solid force"),
        [0.0; 3]
    );
}

fn assert_same_canonical_roles(
    direct: &AleFsiCartesianModel<3>,
    packaged: &AleFsiCartesianModel<3>,
) {
    assert_eq!(direct.fluid().bounds(), packaged.fluid().bounds());
    assert_eq!(
        direct.solid().continuum().bounds(),
        packaged.solid().continuum().bounds()
    );
    assert_eq!(
        direct.fluid().mass_density().to_bits(),
        packaged.fluid().mass_density().to_bits()
    );
    assert_eq!(
        direct.fluid().dynamic_viscosity().to_bits(),
        packaged.fluid().dynamic_viscosity().to_bits()
    );
    assert_eq!(
        direct.solid().mass_density().to_bits(),
        packaged.solid().mass_density().to_bits()
    );
    assert_eq!(
        direct.solid().continuum().shear_modulus().to_bits(),
        packaged.solid().continuum().shear_modulus().to_bits()
    );
    assert_eq!(
        direct.solid().continuum().first_lame_parameter().to_bits(),
        packaged
            .solid()
            .continuum()
            .first_lame_parameter()
            .to_bits()
    );
    assert_eq!(direct.interface().axis(), packaged.interface().axis());
    assert_eq!(
        direct
            .interface()
            .endpoint(direct.fluid().domain())
            .unwrap()
            .side(),
        packaged
            .interface()
            .endpoint(packaged.fluid().domain())
            .unwrap()
            .side()
    );
    assert_eq!(
        direct
            .interface()
            .endpoint(direct.solid().continuum().domain())
            .unwrap()
            .side(),
        packaged
            .interface()
            .endpoint(packaged.solid().continuum().domain())
            .unwrap()
            .side()
    );
    assert_eq!(
        direct
            .fluid()
            .conservative_body_force(&[0.25, 0.5, 0.75])
            .expect("direct fluid force"),
        packaged
            .fluid()
            .conservative_body_force(&[0.25, 0.5, 0.75])
            .expect("packaged fluid force")
    );
    assert_eq!(
        direct
            .solid()
            .continuum()
            .conservative_body_force(&[1.25, 0.5, 0.75])
            .expect("direct solid force"),
        packaged
            .solid()
            .continuum()
            .conservative_body_force(&[1.25, 0.5, 0.75])
            .expect("packaged solid force")
    );
}

fn prepare_root(
    mechanics: &PackageReleaseV1,
    fluid: &PackageReleaseV1,
    solid: &PackageReleaseV1,
) -> PackageReleaseV1 {
    let dependencies = [
        ("mechanics", mechanics),
        ("fluid_laws", fluid),
        ("solid_laws", solid),
    ]
    .into_iter()
    .map(|(_, release)| {
        PackageDependencyV1::new(release.package_identity().expect("dependency identity"))
    })
    .collect();
    let readme = NormalizedRelativePath::parse("README.md").expect("README path");
    let source = NormalizedRelativePath::parse("src/main.eqi").expect("source path");
    let model_source = format!(
        "import Eqiora.Mechanics.Interfaces.interfaces as mechanics;\n\
         import Eqiora.Fluid.Incompressible.incompressible as fluid_laws;\n\
         import Eqiora.Solid.LinearElasticity.linear_elasticity as solid_laws;\n{PACKAGED_ROOT}"
    );
    let manifest = PackageManifestV1::new(
        "main",
        QualifiedName::parse("org.eqiora.verify.ale_fsi_3d_packages").expect("root name"),
        ExactVersion::parse("0.1.0").expect("root version"),
        dependencies,
        vec![
            BundleEntryV1::new(readme.clone(), BundleRoleV1::Documentation),
            BundleEntryV1::new(source.clone(), BundleRoleV1::ModelSource),
        ],
    )
    .expect("root manifest");
    let sources = PackageSourcesV1::new(
        manifest,
        vec![
            SourceFileV1::new(
                readme,
                BundleRoleV1::Documentation,
                b"Exact-package authoring fixture for canonical tetrahedral ALE FSI.\n".to_vec(),
            ),
            SourceFileV1::new(source, BundleRoleV1::ModelSource, model_source.into_bytes()),
        ],
    )
    .expect("closed root source inventory");
    prepare_package_release_v1(sources, &[mechanics.clone(), fluid.clone(), solid.clone()])
        .expect("prepare exact 3D FSI root")
}
