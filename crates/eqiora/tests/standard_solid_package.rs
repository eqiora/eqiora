use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use eqiora::kernel::BoundarySide;
use eqiora::package::{
    BundleEntryV1, BundleRoleV1, DirectoryPackageStore, ExactVersion, InMemoryPackageStore,
    NormalizedRelativePath, PackageDependencyV1, PackageManifestV1, PackageReleaseV1,
    PackageSourcesV1, PackagedModelDocument, QualifiedName, ResolutionRecordV1, SourceFileV1,
    prepare_package_release_v1,
};
use eqiora_numerics::{
    common::PhysicalBoundaryDisposition,
    solid::{IsotropicElasticityContinuum, lower_isotropic_elasticity_cartesian_2d},
};

#[path = "support/embedded_package.rs"]
mod embedded_package;

const VERSION: &str = "0.4.0";
static NEXT_SCRATCH: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy)]
enum Assumption {
    PlaneStrain,
    PlaneStress,
}

#[test]
fn curated_and_explicit_standard_elasticity_share_the_package_neutral_path() {
    let solid = standard_release();
    for assumption in [Assumption::PlaneStrain, Assumption::PlaneStress] {
        let curated = compile_root(&solid, &root_source(true, assumption, 0.25));
        let explicit = compile_root(&solid, &root_source(false, assumption, 0.25));
        assert_eq!(curated.property_bindings().len(), 2);
        assert!(
            curated
                .property_bindings()
                .all(|binding| binding.0.is_some())
        );
        let curated = lower(curated.model().program());
        let explicit = lower(explicit.model().program());

        assert_eq!(curated.bounds(), explicit.bounds());
        assert_eq!(curated.shear_modulus(), 48.0);
        assert_eq!(curated.shear_modulus(), explicit.shear_modulus());
        assert_eq!(
            curated.first_lame_parameter(),
            explicit.first_lame_parameter()
        );
        assert_eq!(
            curated.load_potential_expression().evaluate(&[0.25, 0.75]),
            explicit.load_potential_expression().evaluate(&[0.25, 0.75])
        );
        assert_eq!(
            curated
                .boundary_inventory()
                .boundary(0, BoundarySide::Lower)
                .expect("fixed boundary")
                .disposition(),
            PhysicalBoundaryDisposition::TraceZero
        );
        for (axis, side) in [
            (0, BoundarySide::Upper),
            (1, BoundarySide::Lower),
            (1, BoundarySide::Upper),
        ] {
            assert_eq!(
                curated
                    .boundary_inventory()
                    .boundary(axis, side)
                    .expect("traction-free boundary")
                    .disposition(),
                PhysicalBoundaryDisposition::FluxZero
            );
        }
    }
}

#[test]
fn curated_assumptions_produce_distinct_expected_lame_parameters() {
    let solid = standard_release();
    let plane_strain = lower(
        compile_root(&solid, &root_source(true, Assumption::PlaneStrain, 0.25))
            .model()
            .program(),
    );
    let plane_stress = lower(
        compile_root(&solid, &root_source(true, Assumption::PlaneStress, 0.25))
            .model()
            .program(),
    );

    assert_eq!(plane_strain.first_lame_parameter(), 48.0);
    assert_eq!(plane_stress.first_lame_parameter(), 32.0);
}

#[test]
fn standard_vector_boundary_data_compiles_through_the_ordinary_package_path() {
    let solid = standard_release();
    compile_root(
        &solid,
        &root_source_with_boundaries(true, Assumption::PlaneStrain, 0.25, true),
    );
}

#[test]
fn singular_poisson_ratios_fail_during_compilation() {
    let solid = standard_release();
    for (assumption, poisson_ratio) in [
        (Assumption::PlaneStrain, 0.5),
        (Assumption::PlaneStress, 1.0),
    ] {
        compile_root(&solid, &root_source(true, assumption, 0.25));
        let error = prepare_package_release_v1(
            root_sources(&solid, &root_source(true, assumption, poisson_ratio)),
            std::slice::from_ref(&solid),
        )
        .expect_err("singular actual Component context must fail before publication");
        let eqiora::package::PackagePreparationError::Diagnostics(diagnostics) = error else {
            panic!("expected compiler diagnostics, got {error}");
        };
        assert!(
            diagnostics.iter().any(|diagnostic| {
                diagnostic.code() == eqiora::diagnostic::codes::LANGUAGE_TYPE_ERROR
                    && diagnostic.message() == "division by zero in static value"
                    && diagnostic.source_span().is_some()
            }),
            "{diagnostics:?}"
        );
    }
}

#[test]
fn standard_package_reopens_from_the_project_lock_and_offline_store() {
    let scratch = Scratch::create();
    let solid_sources = embedded_package::public_sources("Eqiora.Solid");
    let solid = prepare_package_release_v1(solid_sources.clone(), &[]).expect("solid release");
    let root_sources = root_sources(&solid, &root_source(true, Assumption::PlaneStrain, 0.25));

    write_package(&scratch.child("root"), &root_sources);
    write_package(&scratch.child("solid"), &solid_sources);
    let store_path = scratch.child("store");
    fs::write(
        scratch.0.join("eqiora.toml"),
        r#"[package]
name = "org.example.StandardSolid"
version = "1.0.0"
source = "root/src"
entry = "main"

[dependencies."Eqiora.Solid"]
version = "0.4.0"
sources = [{ path = "solid" }]
"#,
    )
    .expect("write project manifest");
    fs::write(
        scratch.0.join("solid/eqiora.toml"),
        r#"[package]
name = "Eqiora.Solid"
version = "0.4.0"
entry = "solid"
"#,
    )
    .expect("write solid manifest");

    let resolution =
        PackagedModelDocument::resolve_local_package_project_v1(&scratch.0, &store_path)
            .expect("resolve standard package project");
    let reopened = PackagedModelDocument::open_local_package_project_v1(&scratch.0, &store_path)
        .expect("reopen exact project lock");
    assert_eq!(reopened, resolution);
    let store = DirectoryPackageStore::open_ambient(store_path).expect("open offline store");
    let document = PackagedModelDocument::compile_locked(&store, &reopened, "Main")
        .expect("compile from exact offline store");
    assert_eq!(document.property_bindings().len(), 2);
    assert_eq!(lower(document.model().program()).shear_modulus(), 48.0);
}

fn standard_release() -> PackageReleaseV1 {
    prepare_package_release_v1(embedded_package::public_sources("Eqiora.Solid"), &[])
        .expect("standard solid release")
}

fn compile_root(solid: &PackageReleaseV1, source: &str) -> PackagedModelDocument {
    compile_root_result(solid, source).expect("compile standard solid project")
}

fn compile_root_result(
    solid: &PackageReleaseV1,
    source: &str,
) -> Result<PackagedModelDocument, eqiora::package::PackageCompilationError> {
    let root = prepare_package_release_v1(root_sources(solid, source), std::slice::from_ref(solid))
        .expect("standard solid project root");
    let resolution = ResolutionRecordV1::from_exact_releases(&root, std::slice::from_ref(solid))
        .expect("standard solid lock");
    let mut store = InMemoryPackageStore::default();
    store.insert(solid).expect("store solid");
    store.insert(&root).expect("store root");
    PackagedModelDocument::compile_locked(&store, &resolution, "Main")
}

fn root_sources(solid: &PackageReleaseV1, source: &str) -> PackageSourcesV1 {
    let path = NormalizedRelativePath::parse("src/main.eqi").expect("root path");
    let dependency = PackageDependencyV1::new(solid.package_identity().expect("solid identity"));
    let manifest = PackageManifestV1::new(
        "main",
        QualifiedName::parse("org.example.StandardSolid").expect("root name"),
        ExactVersion::parse(VERSION).expect("root version"),
        vec![dependency],
        vec![BundleEntryV1::new(path.clone(), BundleRoleV1::ModelSource)],
    )
    .expect("root manifest");
    let source = format!("import Eqiora.Solid.solid as solid;\n{source}");
    PackageSourcesV1::new(
        manifest,
        vec![SourceFileV1::new(
            path,
            BundleRoleV1::ModelSource,
            source.into_bytes(),
        )],
    )
    .expect("root sources")
}

fn lower(program: &eqiora::sem::KernelProgram) -> IsotropicElasticityContinuum<2> {
    lower_isotropic_elasticity_cartesian_2d(program)
        .expect("standard package reaches common elasticity lowering")
}

fn write_package(path: &Path, sources: &PackageSourcesV1) {
    fs::create_dir_all(path).expect("create package directory");
    for file in sources.files() {
        let destination = path.join(file.path().as_str());
        fs::create_dir_all(destination.parent().expect("package file parent"))
            .expect("create package source directory");
        fs::write(destination, file.bytes()).expect("write package source");
    }
}

struct Scratch(PathBuf);

impl Scratch {
    fn create() -> Self {
        let sequence = NEXT_SCRATCH.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "eqiora-standard-solid-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create scratch directory");
        Self(path)
    }

    fn child(&self, name: &str) -> PathBuf {
        let path = self.0.join(name);
        fs::create_dir(&path).expect("create scratch child");
        path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove scratch directory");
    }
}

fn root_source(curated: bool, assumption: Assumption, poisson_ratio: f64) -> String {
    root_source_with_boundaries(curated, assumption, poisson_ratio, false)
}

fn root_source_with_boundaries(
    curated: bool,
    assumption: Assumption,
    poisson_ratio: f64,
    prescribed_boundaries: bool,
) -> String {
    let (preamble, material_parameters, governing) = if curated {
        let component = match assumption {
            Assumption::PlaneStrain => "PlaneStrainMaterial2d",
            Assumption::PlaneStress => "PlaneStressMaterial2d",
        };
        (
            format!(
                r#"public property release ReferenceYoungModulus: solid.YoungModulus {{
  analytic {{ value = 120; source_unit: kg / (m * s ^ 2) = 1; }}
  validity unconditional; outside reject; branch single; citation org.example.reference; license spdx.CC0_1_0;
}}
public property release ReferencePoissonRatio: solid.PoissonRatio {{
  analytic {{ value = {poisson_ratio}; source_unit: 1 = 1; }}
  validity unconditional; outside reject; branch single; citation org.example.reference; license spdx.CC0_1_0;
}}
public material composition ReferenceMaterial {{
  property poisson_ratio = ReferencePoissonRatio;
  property young_modulus = ReferenceYoungModulus;
}}
"#
            ),
            String::new(),
            format!(
                r#"  instance governing: solid.{component}(
    body = body,
    exterior = boundaries(x_lower, x_upper, y_lower, y_upper),
    displacement = displacement,
    load_potential = load_potential,
    young_modulus = ReferenceMaterial.young_modulus,
    poisson_ratio = ReferenceMaterial.poisson_ratio
  );"#
            ),
        )
    } else {
        let lambda = match assumption {
            Assumption::PlaneStrain => 48,
            Assumption::PlaneStress => 32,
        };
        (
            String::new(),
            format!(
                "  parameter shear_modulus: kg / (m * s ^ 2) = 48;\n  parameter first_lame_parameter: kg / (m * s ^ 2) = {lambda};"
            ),
            r#"  instance balance: solid.IsotropicBalance2d(
    body = body,
    displacement = displacement,
    load_potential = load_potential,
    shear_modulus = shear_modulus,
    first_lame_parameter = first_lame_parameter
  );
  instance interface: solid.DisplacementTractionInterface2d(
    body = body,
    exterior = boundaries(x_lower, x_upper, y_lower, y_upper),
    displacement = displacement,
    shear_modulus = shear_modulus,
    first_lame_parameter = first_lame_parameter
  );"#
            .to_owned(),
        )
    };
    let interface = if curated { "governing" } else { "interface" };
    let (boundary_fields, x_lower_condition, x_upper_condition) = if prescribed_boundaries {
        (
            r#"  variable displacement_potential: m ^ 2 on body;
  variable boundary_displacement: vector<m, 2> on body in h1;
  variable traction_potential: kg / s ^ 2 on body;
  variable boundary_traction: vector<kg / (m * s ^ 2), 2> on body in h1;
  parameter displacement_scale: m = 1;
  parameter traction_scale: kg / (m * s ^ 2) = 2;
  relation displacement_potential_definition on body {
    displacement_potential - displacement_scale * coordinate(0) = 0;
  }
  relation boundary_displacement_definition on body {
    boundary_displacement - grad(displacement_potential) = 0;
  }
  relation traction_potential_definition on body {
    traction_potential - traction_scale * coordinate(0) = 0;
  }
  relation boundary_traction_definition on body {
    boundary_traction - grad(traction_potential) = 0;
  }
"#,
            r#"  instance x_lower_condition: solid.PrescribedDisplacement2d(
    body = body,
    face = x_lower,
    displacement = boundary_displacement
  );"#,
            r#"  instance x_upper_condition: solid.PrescribedTraction2d(
    body = body,
    face = x_upper,
    traction = boundary_traction
  );"#,
        )
    } else {
        (
            "",
            r#"  instance x_lower_condition: solid.FixedDisplacement2d(
    body = body, face = x_lower
  );"#,
            r#"  instance x_upper_condition: solid.TractionFree2d(
    body = body, face = x_upper
  );"#,
        )
    };
    format!(
        r#"{preamble}model Main() {{
  domain body = box(0, 4, 0, 2);
  domain x_lower = boundary(body, axis = 0, side = lower);
  domain x_upper = boundary(body, axis = 0, side = upper);
  domain y_lower = boundary(body, axis = 1, side = lower);
  domain y_upper = boundary(body, axis = 1, side = upper);

  variable displacement: vector<m, 2> on body in smooth;
  variable load_potential: kg / (m * s ^ 2) on body;
  parameter zero_load: kg / (m * s ^ 2) = 0;
{material_parameters}
  relation load_definition on body {{
    load_potential - zero_load = 0;
  }}
{boundary_fields}
{governing}
{x_lower_condition}
{x_upper_condition}
  instance y_lower_condition: solid.TractionFree2d(
    body = body, face = y_lower
  );
  instance y_upper_condition: solid.TractionFree2d(
    body = body, face = y_upper
  );

  connect {interface}.mechanical[boundary = x_lower],
    x_lower_condition.mechanical;
  connect {interface}.mechanical[boundary = x_upper],
    x_upper_condition.mechanical;
  connect {interface}.mechanical[boundary = y_lower],
    y_lower_condition.mechanical;
  connect {interface}.mechanical[boundary = y_upper],
    y_upper_condition.mechanical;
}}
"#
    )
}
