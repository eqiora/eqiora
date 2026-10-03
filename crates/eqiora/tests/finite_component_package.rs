//! Finite map signatures retain exact nominal spaces through locked package sources.
use eqiora::api::ModelDocument;
use eqiora::artifact::{ModelDecoderLimits, ModelEnvelope};
use eqiora::package::{
    BundleEntryV1, BundleRoleV1, ExactVersion, InMemoryPackageStore, NormalizedRelativePath,
    PackageManifestV1, PackageSourcesV1, PackagedModelDocument, QualifiedName, ResolutionRecordV1,
    SourceFileV1, prepare_package_release_v1,
};

fn packaged(source: &str) -> PackagedModelDocument {
    let path = NormalizedRelativePath::parse("src/finite_components.eqi").unwrap();
    let manifest = PackageManifestV1::new(
        "finite_components",
        QualifiedName::parse("org.eqiora.test.FiniteComponents").unwrap(),
        ExactVersion::parse("1.0.0").unwrap(),
        vec![],
        vec![BundleEntryV1::new(path.clone(), BundleRoleV1::ModelSource)],
    )
    .unwrap();
    let sources = PackageSourcesV1::new(
        manifest,
        vec![SourceFileV1::new(
            path,
            BundleRoleV1::ModelSource,
            source.as_bytes().to_vec(),
        )],
    )
    .unwrap();
    let release = prepare_package_release_v1(sources, &[]).unwrap();
    let mut store = InMemoryPackageStore::default();
    store.insert(&release).unwrap();
    let resolution = ResolutionRecordV1::from_exact_releases(&release, &[]).unwrap();
    PackagedModelDocument::compile_locked(&store, &resolution, "M").unwrap()
}

#[test]
fn finite_map_package_retains_signatures_and_operation_identity_without_source_order() {
    let source = r#"space Input=orthonormal(q,v); space Output=orthonormal(u,w);
component Transform(parameter a:map<1,Input,Output>, parameter rhs:coordinates<1,Output>) {
    variable state:coordinates<1,Input>;
    relation r { apply(a,state)=rhs; }
    observable norm:1=pair(transpose(state),state);
}
model M() {
    instance control:Transform(a=linear_map(Input,Output,[[2,1],[1,3]]),rhs=coordinates(Output,[4.0,7.0]));
}"#;
    let direct = ModelDocument::compile("finite.eqi", source).unwrap();
    for authored in [
        source.to_owned(),
        source.replace(
            "space Input=orthonormal(q,v); space Output=orthonormal(u,w);",
            "space Output=orthonormal(u,w); space Input=orthonormal(q,v);",
        ),
    ] {
        let packaged = packaged(&authored);
        assert!(packaged.model().structurally_equivalent(&direct).unwrap());
        let envelope = ModelEnvelope::from_program(packaged.model().program()).unwrap();
        let bytes = envelope.canonical_json().unwrap();
        let replay = ModelEnvelope::from_json(&bytes, ModelDecoderLimits::default())
            .unwrap()
            .to_program()
            .unwrap();
        assert_eq!(&replay, packaged.model().program());
    }
    // The same storage shape cannot satisfy a signature with reversed nominal endpoints.
    assert!(
        ModelDocument::compile(
            "foreign.eqi",
            &source.replace("a=linear_map(Input,Output", "a=linear_map(Output,Input",)
        )
        .is_err()
    );
}
