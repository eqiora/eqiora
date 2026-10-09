use std::collections::BTreeSet;

use eqiora::diagnostic::codes;
use eqiora::graph::EdgeKind;
use eqiora::kernel::BoundarySide;
use eqiora::package::{
    BundleEntryV1, BundleRoleV1, ExactVersion, InMemoryPackageStore, NormalizedRelativePath,
    PackageDependencyV1, PackageManifestV1, PackageReleaseV1, PackageSourcesV1,
    PackagedModelDocument, QualifiedName, ResolutionRecordV1, SourceFileV1,
    prepare_package_release_v1,
};
use eqiora::sem::KernelProgram;
use eqiora_numerics::{
    common::PhysicalBoundaryDisposition, solid::lower_isotropic_elasticity_cartesian_2d,
    solid::lower_isotropic_elastodynamics_cartesian_2d,
};

#[path = "support/embedded_package.rs"]
mod embedded_package;

const DIRECT: &str =
    include_str!("../../../verify/solid/dynamic-linear-solid-semantics-2d/models/direct.eqi");
const PACKAGED: &str =
    include_str!("../../../verify/solid/dynamic-linear-solid-semantics-2d/models/packaged.eqi");
const SOLID_SOURCE: &str = include_str!(
    "../../../crates/eqiora-api/packages/Eqiora.Solid.LinearElasticity/src/linear_elasticity.eqi"
);
const STATIC_PACKAGED: &str =
    include_str!("../../../verify/solid/mixed-boundary-elasticity-2d/models/packaged.eqi");

const ROOT_PACKAGE: &str = "org.eqiora.verify.dynamic_linear_solid_semantics_2d";
const ROOT_VERSION: &str = "0.1.0";

#[derive(Debug, PartialEq)]
struct Observation {
    bounds: [[u64; 2]; 2],
    density: u64,
    shear_modulus: u64,
    first_lame_parameter: u64,
    load_samples: [u64; 3],
    load_coordinate_jvp: [u64; 2],
    boundaries: [[PhysicalBoundaryDisposition; 2]; 2],
}

#[test]
fn direct_and_exact_package_models_have_equal_dynamic_solid_meaning() {
    let mechanics = public_release("Eqiora.Mechanics.Interfaces", &[]);
    let solid = public_solid_release(&mechanics);

    let direct = eqiora::api::ModelDocument::compile("direct.eqi", DIRECT)
        .expect("direct first-order solid Model compiles");
    let reordered_source = alias_and_permute_boundaries_and_connections(PACKAGED);
    let canonical_root = prepare_root_release(&solid, &mechanics, "solid", "mechanics", PACKAGED)
        .expect("prepare canonical root");
    let reordered_root = prepare_root_release(
        &solid,
        &mechanics,
        "structure",
        "power_boundary",
        &reordered_source,
    )
    .expect("prepare reordered root");
    assert_eq!(
        canonical_root
            .package_identity()
            .expect("canonical root identity"),
        reordered_root
            .package_identity()
            .expect("reordered root identity"),
        "import aliases, Cartesian boundary declaration/family order, and Connection order cannot enter root semantic identity"
    );
    let packaged = compile_prepared_root(&solid, &mechanics, &canonical_root);
    let reordered = compile_prepared_root(&solid, &mechanics, &reordered_root);
    assert_eq!(
        packaged
            .model()
            .canonical_json()
            .expect("canonical packaged Model"),
        reordered
            .model()
            .canonical_json()
            .expect("reordered packaged Model"),
        "equivalent Cartesian boundary and Connection ordering cannot enter canonical Model bytes"
    );

    let direct_observation = observe(direct.program());
    let packaged_observation = observe(packaged.model().program());
    let reordered_observation = observe(reordered.model().program());
    assert_eq!(direct_observation, packaged_observation);
    assert_eq!(packaged_observation, reordered_observation);
    assert_eq!(f64::from_bits(direct_observation.density), 5.0);
    assert_eq!(f64::from_bits(direct_observation.shear_modulus), 3.0);
    assert_eq!(f64::from_bits(direct_observation.first_lame_parameter), 4.0);
    assert_eq!(
        direct_observation.load_coordinate_jvp.map(f64::from_bits),
        [2.0, 0.0]
    );

    let direct_model = lower_isotropic_elastodynamics_cartesian_2d(direct.program())
        .expect("direct coefficient tapes lower");
    let packaged_model = lower_isotropic_elastodynamics_cartesian_2d(packaged.model().program())
        .expect("elaborated package coefficient tapes lower");
    assert_eq!(
        direct_model
            .mass_density_expression()
            .parameter_fields()
            .len(),
        1
    );
    assert_eq!(
        direct_model
            .continuum()
            .shear_modulus_expression()
            .parameter_fields()
            .len(),
        1
    );
    assert_eq!(
        direct_model
            .continuum()
            .first_lame_parameter_expression()
            .parameter_fields()
            .len(),
        1
    );
    assert_eq!(
        packaged_model
            .mass_density_expression()
            .parameter_fields()
            .len(),
        0
    );
    assert_eq!(
        packaged_model
            .continuum()
            .shear_modulus_expression()
            .parameter_fields()
            .len(),
        0
    );
    assert_eq!(
        packaged_model
            .continuum()
            .first_lame_parameter_expression()
            .parameter_fields()
            .len(),
        0
    );
    for alias in [
        "law.density",
        "law.mu",
        "law.lambda",
        "boundary.mu",
        "boundary.lambda",
    ] {
        assert!(
            packaged.model().aliases().get(alias).is_none(),
            "literal Component Parameter term `{alias}` has no Kernel identity"
        );
    }
}

#[test]
fn current_release_exposes_the_static_solid_surface() {
    let mechanics = public_release("Eqiora.Mechanics.Interfaces", &[]);
    let solid = public_solid_release(&mechanics);
    let packaged = compile_root(&solid, &mechanics, "solid", "mechanics", STATIC_PACKAGED);
    let model = lower_isotropic_elasticity_cartesian_2d(packaged.model().program())
        .expect("current static solid meaning is usable");
    assert_eq!(model.bounds(), &[[0.0, 1.0], [0.0, 1.0]]);
    assert_eq!(model.shear_modulus(), 3.0);
    assert_eq!(model.first_lame_parameter(), 0.0);
    assert_eq!(
        model
            .boundary_inventory()
            .boundary(0, BoundarySide::Lower)
            .expect("fixed x-lower side")
            .disposition(),
        PhysicalBoundaryDisposition::TraceZero
    );
    for (axis, side) in [
        (0, BoundarySide::Upper),
        (1, BoundarySide::Lower),
        (1, BoundarySide::Upper),
    ] {
        assert_eq!(
            model
                .boundary_inventory()
                .boundary(axis, side)
                .expect("free static side")
                .disposition(),
            PhysicalBoundaryDisposition::FluxZero
        );
    }
}

#[test]
fn zero_traction_and_live_velocity_ports_remain_explicit_canonical_meaning() {
    let mechanics = public_release("Eqiora.Mechanics.Interfaces", &[]);
    let solid = public_solid_release(&mechanics);

    let flux_source = PACKAGED.replacen(
        "instance x_lower_zero: mechanics.ZeroVelocity2d",
        "instance x_lower_zero: mechanics.ZeroTraction2d",
        1,
    );
    let flux = compile_root(&solid, &mechanics, "solid", "mechanics", &flux_source);
    let flux_model = lower_isotropic_elastodynamics_cartesian_2d(flux.model().program())
        .expect("zero traction is valid dynamic-solid boundary meaning");
    assert_eq!(
        flux_model
            .continuum()
            .boundary_inventory()
            .boundary(0, BoundarySide::Lower)
            .expect("x-lower boundary")
            .disposition(),
        PhysicalBoundaryDisposition::FluxZero
    );

    let live_source = transparent_open_terminal_source(PACKAGED);
    let live = compile_root(&solid, &mechanics, "solid", "mechanics", &live_source);
    let live_model = lower_isotropic_elastodynamics_cartesian_2d(live.model().program())
        .expect("compatible unresolved velocity Port remains canonical meaning");
    let PhysicalBoundaryDisposition::PortBinding { connection, port } = live_model
        .continuum()
        .boundary_inventory()
        .boundary(0, BoundarySide::Lower)
        .expect("x-lower boundary")
        .disposition()
    else {
        panic!("compatible unresolved velocity Port must remain a PortBinding");
    };
    let interface_port = live.model().aliases()["boundary.mechanical[axis=0,side=lower]"];
    let terminal_port = live.model().aliases()["x_lower_zero.mechanical"];
    assert_eq!(port, interface_port, "binding retains the exact solid Port");
    let members = live
        .model()
        .program()
        .edges()
        .iter()
        .filter(|edge| edge.kind() == EdgeKind::Connects && edge.from() == connection)
        .map(|edge| edge.to())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        members,
        BTreeSet::from([interface_port, terminal_port]),
        "the retained Connection is exactly the two intended velocity/traction Ports"
    );
    for (axis, side) in [
        (0, BoundarySide::Upper),
        (1, BoundarySide::Lower),
        (1, BoundarySide::Upper),
    ] {
        assert_eq!(
            live_model
                .continuum()
                .boundary_inventory()
                .boundary(axis, side)
                .expect("remaining boundary")
                .disposition(),
            PhysicalBoundaryDisposition::TraceZero
        );
    }
}

#[test]
fn dynamic_projection_normalizes_each_exact_global_residual_sign() {
    let reversed_kinematics = DIRECT.replace(
        "derivative(displacement) - velocity",
        "velocity - derivative(displacement)",
    );
    let negated_kinematics = DIRECT.replace(
        "derivative(displacement) - velocity",
        "-(derivative(displacement) - velocity)",
    );
    let reversed_momentum = DIRECT.replace(
        "density * derivative(velocity)\n      - div(\n        2 * mu * symmetric_part(grad(displacement))\n        + lambda * isotropic_lift(div(displacement))\n      )\n      - grad(load_potential)",
        "grad(load_potential)\n      - (density * derivative(velocity)\n        - div(\n          2 * mu * symmetric_part(grad(displacement))\n          + lambda * isotropic_lift(div(displacement))\n        ))",
    );
    assert!(reversed_kinematics.contains("velocity - derivative(displacement)"));
    assert!(negated_kinematics.contains("-(derivative(displacement) - velocity)"));
    assert!(
        reversed_momentum.contains("grad(load_potential)\n      - (density * derivative(velocity)")
    );
    let canonical = eqiora::api::ModelDocument::compile("canonical.eqi", DIRECT)
        .expect("canonical residuals compile");
    let expected = observe(canonical.program());
    for (filename, source) in [
        ("reversed-kinematics.eqi", reversed_kinematics),
        ("negated-kinematics.eqi", negated_kinematics),
        ("reversed-momentum.eqi", reversed_momentum),
    ] {
        let equivalent = eqiora::api::ModelDocument::compile(filename, &source)
            .expect("one globally sign-reversed dynamic residual compiles");
        assert_eq!(expected, observe(equivalent.program()));
    }
}

#[test]
fn kinematic_inertia_stress_density_and_closure_near_misses_fail_closed() {
    let missing_kinematics = DIRECT.replace(
        "  relation kinematics on body {\n    derivative(displacement) - velocity = 0;\n  }\n",
        "",
    );
    assert_lowering_rejects(&missing_kinematics, "kinematic Relation");

    let wrong_inertia = DIRECT.replace(
        "density * derivative(velocity)",
        "density * derivative(displacement)",
    );
    assert_dimension_or_lowering_rejects(&wrong_inertia, "inertia");

    let wrong_stress = DIRECT.replace("grad(displacement)", "grad(velocity)");
    assert_dimension_or_lowering_rejects(&wrong_stress, "stress");

    let zero_density = DIRECT.replace(
        "parameter density: kg / m ^ 3 = 5",
        "parameter density: kg / m ^ 3 = 0",
    );
    assert_lowering_rejects(&zero_density, "density");

    let negative_density = DIRECT.replace(
        "parameter density: kg / m ^ 3 = 5",
        "parameter density: kg / m ^ 3 = -1",
    );
    assert_lowering_rejects(&negative_density, "density");

    let zero_shear = DIRECT.replace(
        "parameter mu: kg / (m * s ^ 2) = 3",
        "parameter mu: kg / (m * s ^ 2) = 0",
    );
    assert_lowering_rejects(&zero_shear, "mu > 0");

    let negative_shear = DIRECT.replace(
        "parameter mu: kg / (m * s ^ 2) = 3",
        "parameter mu: kg / (m * s ^ 2) = -1",
    );
    assert_lowering_rejects(&negative_shear, "mu > 0");

    let zero_bulk_2d = DIRECT.replace(
        "parameter lambda: kg / (m * s ^ 2) = 4",
        "parameter lambda: kg / (m * s ^ 2) = -3",
    );
    assert_lowering_rejects(&zero_bulk_2d, "lambda + 2 mu / D > 0");

    let negative_bulk_2d = DIRECT.replace(
        "parameter lambda: kg / (m * s ^ 2) = 4",
        "parameter lambda: kg / (m * s ^ 2) = -4",
    );
    assert_lowering_rejects(&negative_bulk_2d, "lambda + 2 mu / D > 0");

    let retired_representation = DIRECT.replace(
        "model Main() {",
        "model Main() { representation separate = continuum;",
    );
    assert!(
        eqiora::language::parse("retired.eqi", &retired_representation)
            .into_document()
            .is_err()
    );

    let scalar_velocity = DIRECT.replace(
        "state velocity: vector<m / s, 2> on body in h1;",
        "state velocity: m / s on body in h1;",
    );
    assert_typed_source_rejects(&scalar_velocity, "shape");

    let wrong_boundary_parent = DIRECT.replace(
        "domain x_lower = boundary(body, axis = 0, side = lower);",
        "domain peer = box(0, 2, 0, 1);\n  domain x_lower = boundary(peer, axis = 0, side = lower);",
    );
    assert_typed_source_rejects(&wrong_boundary_parent, "support");

    let extra_relation = DIRECT.replace(
        "  relation x_lower_velocity",
        "  relation unexpected on body { velocity - velocity = 0; }\n\n  relation x_lower_velocity",
    );
    assert_lowering_rejects(&extra_relation, "volume Relations");
}

#[test]
fn nominal_connector_and_boundary_coefficients_cannot_be_substituted() {
    let mechanics = public_release("Eqiora.Mechanics.Interfaces", &[]);
    let solid = public_solid_release(&mechanics);

    let wrong_coefficients = PACKAGED.replace(
        "velocity = velocity,\n    mu = 3[kg / (m * s ^ 2)],\n    lambda = 4[kg / (m * s ^ 2)]\n  );\n\n  instance x_lower_zero",
        "velocity = velocity,\n    mu = 9[kg / (m * s ^ 2)],\n    lambda = 4[kg / (m * s ^ 2)]\n  );\n\n  instance x_lower_zero",
    );
    let mismatched = compile_root(
        &solid,
        &mechanics,
        "solid",
        "mechanics",
        &wrong_coefficients,
    );
    let diagnostic = lower_isotropic_elastodynamics_cartesian_2d(mismatched.model().program())
        .expect_err("boundary and volume Lamé coefficients must agree");
    assert!(
        diagnostic.message().contains("coefficients differ"),
        "unexpected coefficient diagnostic: {}",
        diagnostic.message()
    );

    let distinct_connector = r#"

public connector OtherVelocityTractionBoundary {
  trace velocity: m / s;
  flux traction: kg / (m * s ^ 2);
  shape spatial_vector;
  frame spatial;
  pairing euclidean_boundary_duality;
  orientation parent_outward;
}
"#;
    let wrong_connector_source = format!(
        "{}{}",
        SOLID_SOURCE.replace(
            "mechanics.VelocityTractionBoundary",
            "OtherVelocityTractionBoundary",
        ),
        distinct_connector
    );
    let wrong_solid = inline_solid_release(&mechanics, &wrong_connector_source);
    let error = prepare_root_release(&wrong_solid, &mechanics, "solid", "mechanics", PACKAGED)
        .expect_err("equal dimensions cannot substitute a distinct nominal Connector");
    let rendered = format!("{error:?}");
    assert!(
        rendered.contains("Connector")
            || rendered.contains("connector")
            || rendered.contains("nominal"),
        "unexpected nominal identity diagnostic: {rendered}"
    );
}

fn observe(program: &KernelProgram) -> Observation {
    let model = lower_isotropic_elastodynamics_cartesian_2d(program)
        .expect("canonical first-order elastodynamic meaning lowers");
    let bounds = model
        .continuum()
        .bounds()
        .map(|axis| axis.map(f64::to_bits));
    let load_samples = [0.0, 1.0, 2.0].map(|x| {
        model
            .continuum()
            .load_potential_expression()
            .evaluate(&[x, 0.25])
            .expect("load tape evaluates")
            .to_bits()
    });
    let zero_parameter_tangent = vec![
        0.0;
        model
            .continuum()
            .load_potential_expression()
            .parameter_fields()
            .len()
    ];
    let load_coordinate_jvp = [[1.0, 0.0], [0.0, 1.0]].map(|direction| {
        model
            .continuum()
            .load_potential_expression()
            .evaluate_jvp(&[0.5, 0.25], &direction, &zero_parameter_tangent)
            .expect("load tape coordinate JVP evaluates")
            .1
            .to_bits()
    });
    let boundaries = [0, 1].map(|axis| {
        [BoundarySide::Lower, BoundarySide::Upper].map(|side| {
            model
                .continuum()
                .boundary_inventory()
                .boundary(axis, side)
                .expect("complete Cartesian boundary")
                .disposition()
        })
    });
    assert!(
        boundaries
            .iter()
            .flatten()
            .all(|disposition| { *disposition == PhysicalBoundaryDisposition::TraceZero })
    );
    assert_ne!(model.continuum().displacement(), model.velocity());
    assert_eq!(
        model
            .mass_density_expression()
            .evaluate(&[0.5, 0.25])
            .unwrap(),
        5.0
    );
    assert_eq!(
        model
            .continuum()
            .shear_modulus_expression()
            .evaluate(&[0.5, 0.25])
            .unwrap(),
        3.0
    );
    assert_eq!(
        model
            .continuum()
            .first_lame_parameter_expression()
            .evaluate(&[0.5, 0.25])
            .unwrap(),
        4.0
    );
    Observation {
        bounds,
        density: model.mass_density().to_bits(),
        shear_modulus: model.continuum().shear_modulus().to_bits(),
        first_lame_parameter: model.continuum().first_lame_parameter().to_bits(),
        load_samples,
        load_coordinate_jvp,
        boundaries,
    }
}

fn assert_lowering_rejects(source: &str, message_fragment: &str) {
    let document = eqiora::api::ModelDocument::compile("near-miss.eqi", source)
        .expect("near miss remains valid Model meaning");
    let diagnostic = lower_isotropic_elastodynamics_cartesian_2d(document.program())
        .expect_err("near miss must not enter dynamic-solid lowering");
    assert!(
        diagnostic.message().contains(message_fragment),
        "unexpected lowering diagnostic: {}",
        diagnostic.message()
    );
}

fn assert_dimension_or_lowering_rejects(source: &str, context: &str) {
    match eqiora::api::ModelDocument::compile("near-miss.eqi", source) {
        Err(diagnostics) => assert!(
            diagnostics.iter().any(|diagnostic| {
                matches!(
                    diagnostic.code(),
                    codes::INVALID_RELATION_DIMENSION | codes::LANGUAGE_TYPE_ERROR
                ) && diagnostic.message().contains("dimension")
            }),
            "{context} must fail for a dimensional reason, received {diagnostics:?}"
        ),
        Ok(document) => {
            let diagnostic = lower_isotropic_elastodynamics_cartesian_2d(document.program())
                .expect_err("typed structural near miss must fail closed");
            assert_eq!(diagnostic.code(), codes::INVALID_SPATIAL_LOWERING);
        }
    }
}

fn assert_typed_source_rejects(source: &str, message_fragment: &str) {
    let diagnostics = eqiora::api::ModelDocument::compile("near-miss.eqi", source)
        .expect_err("ill-typed source near miss must fail before lowering");
    assert!(
        diagnostics.iter().any(|diagnostic| {
            matches!(
                diagnostic.code(),
                codes::LANGUAGE_TYPE_ERROR
                    | codes::INVALID_KERNEL_DEFINITION
                    | codes::INVALID_RELATION_DIMENSION
            ) && diagnostic.message().contains(message_fragment)
        }),
        "expected typed {message_fragment} diagnostic, received {diagnostics:?}"
    );
}

fn public_solid_release(mechanics: &PackageReleaseV1) -> PackageReleaseV1 {
    let current = eqiora::language::parse("linear-elasticity.eqi", SOLID_SOURCE)
        .into_document()
        .expect("current solid package source parses");
    assert_eq!(current.connectors().len(), 1);
    for name in [
        "IsotropicElastodynamicsWithPotential2d",
        "ElastodynamicMechanicalInterface2d",
    ] {
        assert!(
            current
                .components()
                .iter()
                .any(|component| component.name() == name)
        );
    }
    public_release(
        "Eqiora.Solid.LinearElasticity",
        std::slice::from_ref(mechanics),
    )
}

fn public_release(package: &str, dependencies: &[PackageReleaseV1]) -> PackageReleaseV1 {
    let sources = embedded_package::public_sources(package);
    prepare_package_release_v1(sources, dependencies)
        .unwrap_or_else(|error| panic!("prepare public package {package}: {error:?}"))
}

fn inline_solid_release(mechanics: &PackageReleaseV1, source: &str) -> PackageReleaseV1 {
    let dependency =
        PackageDependencyV1::new(mechanics.package_identity().expect("mechanics identity"));
    prepare_package_release_v1(
        inline_sources(
            "Eqiora.Solid.LinearElasticity",
            "0.4.0",
            vec![dependency],
            "src/linear_elasticity.eqi",
            source,
        ),
        std::slice::from_ref(mechanics),
    )
    .expect("prepare synthetic solid package")
}

fn compile_root(
    solid: &PackageReleaseV1,
    mechanics: &PackageReleaseV1,
    solid_alias: &str,
    mechanics_alias: &str,
    source: &str,
) -> PackagedModelDocument {
    let root = prepare_root_release(solid, mechanics, solid_alias, mechanics_alias, source)
        .expect("prepare exact root package");
    compile_prepared_root(solid, mechanics, &root)
}

fn compile_prepared_root(
    solid: &PackageReleaseV1,
    mechanics: &PackageReleaseV1,
    root: &PackageReleaseV1,
) -> PackagedModelDocument {
    let resolution =
        ResolutionRecordV1::from_exact_releases(root, &[solid.clone(), mechanics.clone()])
            .expect("exact dependency resolution");
    let mut store = InMemoryPackageStore::default();
    store.insert(mechanics).expect("install mechanics package");
    store.insert(solid).expect("install solid package");
    store.insert(root).expect("install verification root");
    PackagedModelDocument::compile_locked(&store, &resolution, "Main")
        .expect("exact package graph compiles offline")
}

fn prepare_root_release(
    solid: &PackageReleaseV1,
    mechanics: &PackageReleaseV1,
    solid_alias: &str,
    mechanics_alias: &str,
    source: &str,
) -> Result<PackageReleaseV1, eqiora::package::PackagePreparationError> {
    let solid_name = solid.package_identity().expect("solid identity").name;
    let mechanics_name = mechanics
        .package_identity()
        .expect("mechanics identity")
        .name;
    let source = format!(
        "import {solid_name}.linear_elasticity as {solid_alias};\nimport {mechanics_name}.interfaces as {mechanics_alias};\n{source}"
    );
    let dependencies = vec![
        PackageDependencyV1::new(solid.package_identity().expect("solid identity")),
        PackageDependencyV1::new(mechanics.package_identity().expect("mechanics identity")),
    ];
    prepare_package_release_v1(
        inline_sources(
            ROOT_PACKAGE,
            ROOT_VERSION,
            dependencies,
            "src/main.eqi",
            &source,
        ),
        &[solid.clone(), mechanics.clone()],
    )
}

fn inline_sources(
    name: &str,
    version: &str,
    dependencies: Vec<PackageDependencyV1>,
    model_path: &str,
    model_source: &str,
) -> PackageSourcesV1 {
    let readme = NormalizedRelativePath::parse("README.md").expect("README path");
    let model = NormalizedRelativePath::parse(model_path).expect("model path");
    let manifest = PackageManifestV1::new(
        &model_path
            .strip_prefix("src/")
            .unwrap()
            .strip_suffix(".eqi")
            .unwrap()
            .replace('/', "."),
        QualifiedName::parse(name).expect("package name"),
        ExactVersion::parse(version).expect("exact version"),
        dependencies,
        vec![
            BundleEntryV1::new(readme.clone(), BundleRoleV1::Documentation),
            BundleEntryV1::new(model.clone(), BundleRoleV1::ModelSource),
        ],
    )
    .expect("package manifest");
    PackageSourcesV1::new(
        manifest,
        vec![
            SourceFileV1::new(
                readme,
                BundleRoleV1::Documentation,
                format!("Exact test package {name}.\n").into_bytes(),
            ),
            SourceFileV1::new(
                model,
                BundleRoleV1::ModelSource,
                model_source.as_bytes().to_vec(),
            ),
        ],
    )
    .expect("closed author inventory")
}

fn alias_and_permute_boundaries_and_connections(source: &str) -> String {
    let mut source = source
        .replace("solid.", "structure.")
        .replace("mechanics.", "power_boundary.")
        .replace(
            "  domain x_lower = boundary(body, axis = 0, side = lower);\n  domain x_upper = boundary(body, axis = 0, side = upper);\n  domain y_lower = boundary(body, axis = 1, side = lower);\n  domain y_upper = boundary(body, axis = 1, side = upper);",
            "  domain y_upper = boundary(body, axis = 1, side = upper);\n  domain x_upper = boundary(body, axis = 0, side = upper);\n  domain y_lower = boundary(body, axis = 1, side = lower);\n  domain x_lower = boundary(body, axis = 0, side = lower);",
        )
        .replace(
            "boundaries(x_lower, x_upper, y_lower, y_upper)",
            "boundaries(y_upper, x_upper, y_lower, x_lower)",
        );
    assert!(
        source.find("domain y_upper").expect("permuted y-upper")
            < source.find("domain x_lower").expect("permuted x-lower"),
        "boundary declaration permutation must be effective"
    );
    assert!(source.contains("boundaries(y_upper, x_upper, y_lower, x_lower)"));
    for (side, terminal) in [
        ("x_lower", "x_lower_zero"),
        ("x_upper", "x_upper_zero"),
        ("y_lower", "y_lower_zero"),
        ("y_upper", "y_upper_zero"),
    ] {
        source = source.replace(
            &format!("boundary.mechanical[boundary = {side}], {terminal}.mechanical"),
            &format!("{terminal}.mechanical, boundary.mechanical[boundary = {side}]"),
        );
    }
    source
}

fn transparent_open_terminal_source(source: &str) -> String {
    let terminal = r#"
public component CompatibleOpenVelocityTerminal2d(
  support body: volume(ambient_dimension = 2),
  support face: boundary(parent = body),
  port mechanical: mechanics.VelocityTractionBoundary over face
) {

  relation transparent_carrier on face {
    mechanical.velocity - mechanical.velocity = 0;
    mechanical.traction - mechanical.traction = 0;
  }
}

"#;
    format!(
        "{terminal}{}",
        source.replacen(
            "instance x_lower_zero: mechanics.ZeroVelocity2d",
            "instance x_lower_zero: CompatibleOpenVelocityTerminal2d",
            1,
        )
    )
}
