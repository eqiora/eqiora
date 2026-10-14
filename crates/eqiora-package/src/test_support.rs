//! Ordinary compilation inputs shared by package binding tests.

use crate::PackageCompilationRecordV2;

pub(crate) fn compilation_with(
    model_byte: &str,
    resolution_byte: &str,
    source_byte: &str,
    compiler_version: &str,
) -> PackageCompilationRecordV2 {
    let model = model_byte.repeat(32);
    let resolution = resolution_byte.repeat(32);
    let semantic = "12".repeat(32);
    let source = source_byte.repeat(32);
    let json = format!(
        r#"{{"schema":"eqiora.package-compilation.v2","encoding":"eqiora.canonical-json.v1","model_sha256":"{model}","root":{{"name":"org.example.Main","version":"1.0.0","semantic_digest":"{semantic}"}},"resolution_digest":"{resolution}","packages":[{{"package":{{"name":"org.example.Main","version":"1.0.0","semantic_digest":"{semantic}"}},"source_digest":"{source}"}}],"toolchain":{{"compiler":"Eqiora.Compiler","compiler_version":"{compiler_version}","semantic_canonicalization_version":2,"source_bundle_version":1,"resolution_version":1}}}}"#,
    );
    PackageCompilationRecordV2::from_json(json.as_bytes()).expect("compilation")
}
