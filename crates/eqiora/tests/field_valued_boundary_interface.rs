#[path = "support/exact_package.rs"]
#[allow(dead_code)] // Integration-test crates consume disjoint shared fixtures.
mod exact_package;

use eqiora::artifact::{
    ArtifactDigest, ModelEnvelope, ModelTransactionEnvelope, PhysicalExposureCatalogEnvelopeV1,
    PhysicalExposureContractV1, PhysicalExposureProjectionV1,
};
use eqiora::entity::kinds;
use eqiora::graph::EdgeKind;
use eqiora::ir::ComponentScalarization;
use eqiora::kernel::SymbolRef;
use eqiora::package::{
    BundleEntryV1, BundleRoleV1, ExactVersion, NormalizedRelativePath, PackageCompilationError,
    PackageDependencyV1, PackageManifestV1, PackagePreparationError, PackageReleaseV1,
    PackageSourcesV1, PackagedModelDocument, QualifiedName, ResolutionRecordV1, SourceFileV1,
    prepare_package_release_v1,
};
use eqiora::{Entity, Id};

const COMPONENTS: &str =
    include_str!("../../../verify/packages/field-valued-boundary-interface/models/components.eqi");
const COUPLED: &str =
    include_str!("../../../verify/packages/field-valued-boundary-interface/models/coupled.eqi");
const COUPLED_PERMUTED: &str = include_str!(
    "../../../verify/packages/field-valued-boundary-interface/models/coupled-permuted.eqi"
);
const NONCOINCIDENT: &str = include_str!(
    "../../../verify/packages/field-valued-boundary-interface/models/noncoincident.eqi"
);
const WRONG_PARENT: &str = include_str!(
    "../../../verify/packages/field-valued-boundary-interface/models/wrong-parent.eqi"
);
const PROJECTED: &str =
    include_str!("../../../verify/packages/field-valued-boundary-interface/models/projected.eqi");
const VERSION: &str = "0.1.0";
const ROOT_PACKAGE: &str = "org.eqiora.verify.field_valued_boundary_interface";

fn manifest(name: &str, dependencies: Vec<PackageDependencyV1>) -> PackageManifestV1 {
    PackageManifestV1::new(
        "model",
        QualifiedName::parse(name).expect("package name"),
        ExactVersion::parse(VERSION).expect("package version"),
        dependencies,
        vec![BundleEntryV1::new(
            NormalizedRelativePath::parse("src/model.eqi").expect("model path"),
            BundleRoleV1::ModelSource,
        )],
    )
    .expect("package manifest")
}

fn sources(manifest: PackageManifestV1, source: &str) -> PackageSourcesV1 {
    PackageSourcesV1::new(
        manifest,
        vec![SourceFileV1::new(
            NormalizedRelativePath::parse("src/model.eqi").expect("model path"),
            BundleRoleV1::ModelSource,
            source.as_bytes().to_vec(),
        )],
    )
    .expect("closed package sources")
}

fn components_release(source: &str) -> PackageReleaseV1 {
    prepare_package_release_v1(
        sources(manifest("Eqiora.Verify.FieldBoundary", vec![]), source),
        &[],
    )
    .expect("field-boundary component package release")
}

fn root_sources(
    components: &PackageReleaseV1,
    dependency_alias: &str,
    source: &str,
) -> PackageSourcesV1 {
    let dependency = PackageDependencyV1::new(
        components
            .package_identity()
            .expect("component package identity"),
    );
    let source =
        format!("import Eqiora.Verify.FieldBoundary.model as {dependency_alias};\n{source}");
    sources(manifest(ROOT_PACKAGE, vec![dependency]), &source)
}

fn root_release(
    components: &PackageReleaseV1,
    dependency_alias: &str,
    source: &str,
) -> Result<PackageReleaseV1, PackagePreparationError> {
    prepare_package_release_v1(
        root_sources(components, dependency_alias, source),
        std::slice::from_ref(components),
    )
}

fn compile_locked(
    components: &PackageReleaseV1,
    root: &PackageReleaseV1,
) -> Result<PackagedModelDocument, PackageCompilationError> {
    let (store, resolution) = exact_package::resolved_store(root, std::slice::from_ref(components));
    let packaged = PackagedModelDocument::compile_locked(&store, &resolution, "Main")?;
    packaged
        .compilation()
        .validate_against(&resolution)
        .expect("compilation matches exact resolution");
    Ok(packaged)
}

fn typed<I>(packaged: &PackagedModelDocument, name: &str) -> Id<I>
where
    I: Entity,
{
    packaged.model().aliases()[name]
        .downcast()
        .unwrap_or_else(|| panic!("`{name}` has the wrong entity kind"))
}

fn connection_containing(
    packaged: &PackagedModelDocument,
    port: Id<kinds::Port>,
) -> Id<kinds::Connection> {
    packaged
        .model()
        .program()
        .edges()
        .iter()
        .find(|edge| edge.kind() == EdgeKind::Connects && edge.to() == port.erase())
        .and_then(|edge| edge.from().downcast())
        .expect("boundary Port belongs to one conserving Connection")
}

#[derive(Debug, Clone, Copy)]
struct InterfaceSample {
    fluid_trace: [f64; 2],
    solid_trace: [f64; 2],
    fluid_flux: [f64; 2],
    solid_flux: [f64; 2],
}

#[derive(Debug, Clone, Copy)]
struct InterfaceDefects {
    trace: f64,
    outward_flux: f64,
    power: f64,
}

fn norm(values: [f64; 2]) -> f64 {
    values
        .into_iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
}

fn dot(left: [f64; 2], right: [f64; 2]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn defects(sample: InterfaceSample) -> InterfaceDefects {
    InterfaceDefects {
        trace: norm([
            sample.fluid_trace[0] - sample.solid_trace[0],
            sample.fluid_trace[1] - sample.solid_trace[1],
        ]),
        outward_flux: norm([
            sample.fluid_flux[0] + sample.solid_flux[0],
            sample.fluid_flux[1] + sample.solid_flux[1],
        ]),
        power: (dot(sample.fluid_trace, sample.fluid_flux)
            + dot(sample.solid_trace, sample.solid_flux))
        .abs(),
    }
}

fn diagnostic_messages(components: &PackageReleaseV1, source: &str) -> Vec<String> {
    match root_release(components, "mechanics", source) {
        Err(PackagePreparationError::Diagnostics(diagnostics)) => diagnostics
            .into_iter()
            .map(|diagnostic| diagnostic.message().to_owned())
            .collect(),
        Err(error) => panic!("unexpected package preparation failure: {error}"),
        Ok(root) => match compile_locked(components, &root) {
            Err(PackageCompilationError::Diagnostics(diagnostics)) => diagnostics
                .into_iter()
                .map(|diagnostic| diagnostic.message().to_owned())
                .collect(),
            Err(error) => panic!("unexpected package compilation failure: {error}"),
            Ok(_) => panic!("invalid boundary source exposed a packaged model"),
        },
    }
}

#[test]
fn packaged_field_valued_boundary_interface() {
    let components = components_release(COMPONENTS);
    let coupled_root =
        root_release(&components, "mechanics", COUPLED).expect("coupled root release");
    let permuted_root = root_release(&components, "mechanics", COUPLED_PERMUTED)
        .expect("permuted coupled root release");
    let short_alias_source = COUPLED.replace("mechanics.", "m.");
    let short_alias_root =
        root_release(&components, "m", &short_alias_source).expect("import-alias variant release");

    let root_identity = coupled_root.package_identity().expect("root identity");
    assert_eq!(
        root_identity,
        permuted_root.package_identity().expect("permuted identity"),
        "declaration and binding order must not enter package meaning"
    );
    assert_eq!(
        root_identity,
        short_alias_root.package_identity().expect("alias identity"),
        "import alias spelling must not enter package meaning"
    );

    let coupled = compile_locked(&components, &coupled_root).expect("compile coupled package");
    let permuted = compile_locked(&components, &permuted_root).expect("compile permuted package");
    let short_alias =
        compile_locked(&components, &short_alias_root).expect("compile alias variant package");
    let canonical_model = coupled
        .model()
        .canonical_json()
        .expect("canonical current Model");
    let canonical_digest = coupled.model().digest().expect("canonical current digest");
    for equivalent in [&permuted, &short_alias] {
        assert_eq!(
            canonical_model,
            equivalent
                .model()
                .canonical_json()
                .expect("equivalent Model"),
            "source order and import aliases must not change canonical bytes"
        );
        assert_eq!(
            canonical_digest,
            equivalent.model().digest().expect("equivalent digest")
        );
    }

    let program = coupled.model().program();
    let fluid_port = typed::<kinds::Port>(&coupled, "fluid_side.mechanical");
    let solid_port = typed::<kinds::Port>(&coupled, "solid_side.mechanical");
    let connection = connection_containing(&coupled, fluid_port);
    assert!(program.edges().iter().any(|edge| {
        edge.kind() == EdgeKind::Connects
            && edge.from() == connection.erase()
            && edge.to() == solid_port.erase()
    }));

    let junction = program
        .compose_boundary_physical_junction(connection)
        .expect("accepted boundary Connection composes one pointwise law");
    let scalarized = ComponentScalarization::lower(junction.typed())
        .expect("exact shaped junction scalarizes without a second semantics");
    assert_eq!(scalarized.rows().len(), 4);
    assert_eq!(scalarized.rows()[0].component_index(), [0]);
    assert_eq!(scalarized.rows()[1].component_index(), [1]);
    assert_eq!(scalarized.rows()[2].component_index(), [0]);
    assert_eq!(scalarized.rows()[3].component_index(), [1]);

    let balanced = InterfaceSample {
        fluid_trace: [2.0, 5.0],
        solid_trace: [2.0, 5.0],
        fluid_flux: [3.0, -7.0],
        solid_flux: [-3.0, 7.0],
    };
    let residuals = scalarized
        .evaluate(|coordinate| {
            let component = usize::try_from(*coordinate.component_index().first()?).ok()?;
            match coordinate.symbol() {
                SymbolRef::PortTrace(port) if port == fluid_port => {
                    Some(balanced.fluid_trace[component])
                }
                SymbolRef::PortTrace(port) if port == solid_port => {
                    Some(balanced.solid_trace[component])
                }
                SymbolRef::PortFlux(port) if port == fluid_port => {
                    Some(balanced.fluid_flux[component])
                }
                SymbolRef::PortFlux(port) if port == solid_port => {
                    Some(balanced.solid_flux[component])
                }
                _ => None,
            }
        })
        .expect("analytic interface sample evaluates");
    assert_eq!(residuals, [0.0, 0.0, 0.0, 0.0]);
    let balanced_defects = defects(balanced);
    assert_eq!(balanced_defects.trace, 0.0);
    assert_eq!(balanced_defects.outward_flux, 0.0);
    assert_eq!(balanced_defects.power, 0.0);

    let trace_broken = defects(InterfaceSample {
        solid_trace: [9.0, 8.0],
        ..balanced
    });
    assert!(trace_broken.trace > 0.0);
    assert_eq!(trace_broken.outward_flux, 0.0);
    let flux_broken = defects(InterfaceSample {
        solid_flux: [-2.0, 7.0],
        ..balanced
    });
    assert_eq!(flux_broken.trace, 0.0);
    assert!(flux_broken.outward_flux > 0.0);
    let power_broken = defects(InterfaceSample {
        solid_trace: [3.0, 5.0],
        ..balanced
    });
    assert!(power_broken.power > 0.0);

    let carrier = typed::<kinds::Relation>(&coupled, "fluid_side.carrier");
    let carrier_typed = program
        .typed_relation_residual(carrier)
        .expect("accepted Relation reconstructs its exact typing proof");
    let carrier_rows = ComponentScalarization::lower(&carrier_typed)
        .expect("typed Relation follows the same componentwise lowering");
    assert_eq!(carrier_rows.rows().len(), 4);
    assert_eq!(
        carrier_rows
            .evaluate(|coordinate| {
                let component = usize::try_from(*coordinate.component_index().first()?).ok()?;
                match coordinate.symbol() {
                    SymbolRef::PortTrace(_) => Some(balanced.fluid_trace[component]),
                    SymbolRef::PortFlux(_) => Some(balanced.fluid_flux[component]),
                    _ => None,
                }
            })
            .expect("carrier Relation evaluates"),
        [0.0, 0.0, 0.0, 0.0]
    );

    let model = ModelEnvelope::from_program(program).expect("complete current Model envelope");
    let model_bytes = model
        .canonical_json()
        .expect("canonical current Model bytes");
    let decoded = ModelEnvelope::from_json(&model_bytes, Default::default())
        .expect("decode complete current Model envelope");
    let replayed_program = decoded.to_program().expect("replay complete current Model");
    let replayed =
        ModelEnvelope::from_program(&replayed_program).expect("re-encode complete current Model");
    assert_eq!(replayed.canonical_json().unwrap(), model_bytes);
    assert_eq!(replayed.digest().unwrap(), model.digest().unwrap());
    let document = eqiora::api::ModelDocument::replay(&model_bytes)
        .expect("public document reconstructs the current Model");
    assert_eq!(document.canonical_json().unwrap(), model_bytes);
    assert_eq!(document.digest().unwrap(), canonical_digest);

    let (transaction, _) = model
        .to_transaction()
        .expect("reconstruct complete transaction");
    let transaction_envelope =
        ModelTransactionEnvelope::from_transaction(&transaction).expect("current transaction");
    let transaction_bytes = transaction_envelope.canonical_json().unwrap();
    assert_eq!(
        ModelTransactionEnvelope::from_json(&transaction_bytes, Default::default())
            .unwrap()
            .to_transaction()
            .unwrap()
            .ops(),
        transaction.ops()
    );

    let noncoincident = diagnostic_messages(&components, NONCOINCIDENT);
    assert!(
        noncoincident
            .iter()
            .any(|message| message.contains("NoncoincidentBoundaries")),
        "unexpected noncoincident diagnostics: {noncoincident:?}"
    );
    let wrong_parent = diagnostic_messages(&components, WRONG_PARENT);
    assert!(
        wrong_parent.iter().any(|message| {
            message.contains("is not BoundaryOf its exact bound parent slot `body`")
        }),
        "unexpected parent diagnostics: {wrong_parent:?}"
    );

    let renamed_components = components_release(&COMPONENTS.replace("velocity", "speed"));
    assert_ne!(
        components.package_identity().expect("original identity"),
        renamed_components
            .package_identity()
            .expect("renamed-member identity"),
        "public connector quantity names are semantic"
    );
}

#[test]
fn eliminated_field_exposures_replay_exact_support_contracts() {
    let components = components_release(COMPONENTS);
    let root = root_release(&components, "mechanics", PROJECTED).expect("projected root release");
    let resolution =
        ResolutionRecordV1::from_exact_releases(&root, std::slice::from_ref(&components))
            .expect("exact projected resolution");
    let packaged = compile_locked(&components, &root).expect("compile projected package");
    let catalog = packaged
        .physical_exposure_catalog()
        .expect("field exposure catalog");
    assert_eq!(catalog.projections().len(), 2);
    let mut selectors = catalog
        .projections()
        .iter()
        .map(|projection| projection.selector())
        .collect::<Vec<_>>();
    selectors.sort_unstable();
    assert_eq!(
        selectors,
        ["fluid_side.mechanical", "solid_side.mechanical"]
    );
    for projection in catalog.projections() {
        assert_eq!(projection.interior_port_sha256().len(), 1);
        assert!(matches!(
            projection.contract(),
            PhysicalExposureContractV1::FieldBoundary { .. }
        ));
    }
    let bytes = catalog.canonical_json().expect("field catalog JSON");
    let decoded = PhysicalExposureCatalogEnvelopeV1::from_json(&bytes, Default::default())
        .expect("decode field catalog");
    packaged
        .validate_physical_exposure_catalog(&decoded, &resolution)
        .expect("replay exact field supports and source provenance");

    let first = &catalog.projections()[0];
    let second = &catalog.projections()[1];
    let (
        PhysicalExposureContractV1::FieldBoundary {
            connector_sha256, ..
        },
        PhysicalExposureContractV1::FieldBoundary {
            boundary_sha256: wrong_boundary,
            ..
        },
    ) = (first.contract(), second.contract())
    else {
        panic!("both projections must be field-valued");
    };
    let wrong_support = PhysicalExposureProjectionV1::field_boundary(
        first.selector(),
        first.exposure_sha256(),
        first.connection_sha256(),
        first.interior_port_sha256(),
        connector_sha256,
        wrong_boundary,
        first.origins().to_vec(),
    )
    .expect("locally well-formed wrong-support projection");
    let model = ArtifactDigest::from_hex(packaged.model().digest().expect("Model digest"))
        .expect("Model artifact digest");
    let compilation = ArtifactDigest::from_hex(
        packaged
            .compilation()
            .digest()
            .expect("compilation digest")
            .to_hex(),
    )
    .expect("compilation artifact digest");
    assert!(
        PhysicalExposureCatalogEnvelopeV1::new(
            model,
            packaged.model().program(),
            compilation,
            vec![wrong_support],
        )
        .is_err(),
        "a stale/wrong boundary support fails before catalog exposure"
    );
}
