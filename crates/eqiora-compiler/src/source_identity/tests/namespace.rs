//! Source namespace version and independence from presentation locations.

use super::{LocalSourceIdentity, document, format, parse};

#[test]
fn canonical_source_namespace_tracks_the_current_encoding() {
    let document = document("model minimal() { parameter gain: 1 = 2; }");
    let digest = LocalSourceIdentity::from_document(&document).unwrap();
    let namespace = digest.namespace().unwrap();
    assert_eq!(namespace.segments()[0], "local-source-v19");
    assert_eq!(namespace.segments()[1], digest.to_string());
}

#[test]
fn formatting_file_and_span_changes_do_not_change_identity() {
    let compact = "model m() {parameter p:1=2;relation r{p-1=0;}}";
    let parsed = document(compact);
    let formatted = format(&parsed);
    let relocated = parse(
        "elsewhere/relocated.eqi",
        &format!("\n\n// shifts every source span\n{formatted}"),
    )
    .into_document()
    .unwrap();

    assert_eq!(
        LocalSourceIdentity::from_document(&parsed).unwrap(),
        LocalSourceIdentity::from_document(&relocated).unwrap()
    );
}
