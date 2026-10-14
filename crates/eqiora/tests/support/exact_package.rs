use eqiora::compiler::CompilationNamespaceId;
use eqiora::package::{
    ExactVersion, InMemoryPackageStore, ModelPackageIdentityV1, PackageManifestV1,
    PackageReleaseV1, PackageSemanticDigest, QualifiedName, ResolutionRecordV1,
};

pub(crate) fn canonical_manifest(encoded: &[u8]) -> PackageManifestV1 {
    let manifest = PackageManifestV1::from_json(encoded).expect("package manifest");
    assert_eq!(
        manifest.canonical_json().expect("canonical manifest"),
        encoded.strip_suffix(b"\n").unwrap_or(encoded)
    );
    manifest
}

pub(crate) fn namespace(identity: &ModelPackageIdentityV1) -> CompilationNamespaceId {
    CompilationNamespaceId::new([
        identity.name.as_str(),
        identity.version.as_str(),
        &identity.semantic_digest.to_hex(),
    ])
    .expect("package namespace")
}

pub(crate) fn provisional_namespace(name: &str, version: &str) -> CompilationNamespaceId {
    namespace(&ModelPackageIdentityV1::new(
        QualifiedName::parse(name).expect("name"),
        ExactVersion::parse(version).expect("version"),
        PackageSemanticDigest::parse(&"00".repeat(32)).expect("provisional digest"),
    ))
}

pub(crate) fn resolved_store(
    root: &PackageReleaseV1,
    dependencies: &[PackageReleaseV1],
) -> (InMemoryPackageStore, ResolutionRecordV1) {
    let resolution = ResolutionRecordV1::from_exact_releases(root, dependencies)
        .expect("exact offline package resolution");
    let mut store = InMemoryPackageStore::default();
    for dependency in dependencies {
        store.insert(dependency).expect("insert dependency release");
    }
    store.insert(root).expect("insert root release");
    (store, resolution)
}
