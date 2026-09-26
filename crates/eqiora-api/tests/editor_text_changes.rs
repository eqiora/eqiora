use eqiora_api::editor::{
    EditorPosition as P, EditorService, EditorSnapshot, EditorTextChange as C,
};
use eqiora_core::diagnostic::codes;

#[test]
fn ordered_batch_uses_intermediate_utf16_text_and_analyzes_final_source() {
    let mut service = EditorService::new("edit.eqi", 1, "model Old() {}\n");
    let expected = "// 😀e\u{301}\r\nmodel Final() {relation r{1=1;}}\r// end\n";
    let snapshot = service
        .apply_changes(
            2,
            [
                C::replace_all("// 🧪e\u{301}\r\nmodel () {relation r{1=1;}}\r// end\n"),
                C::replace_range(P::new(0, 3), P::new(0, 5), "😀"),
                C::replace_range(P::new(1, 6), P::new(1, 6), "Bad"),
                C::replace_range(P::new(1, 6), P::new(1, 9), "Final"),
            ],
        )
        .unwrap();
    assert_eq!(snapshot.source(), expected);
    assert_eq!(snapshot.version(), 2);
    assert!(
        snapshot.diagnostics().is_empty(),
        "{:?}",
        snapshot.diagnostics()
    );
    assert_eq!(snapshot.symbols()[0].name(), "Final");
    assert_eq!(snapshot.position(10), Some(P::new(0, 7)));
    assert_eq!(snapshot.byte_offset(P::new(0, 4)), None);
    assert_eq!(
        snapshot.byte_offset(P::new(0, 99)),
        None,
        "exact query positions do not clamp"
    );
    assert_eq!(
        snapshot.byte_offset(P::new(3, 0)),
        Some(expected.len() as u32)
    );
}

#[test]
fn end_columns_clamp_but_invalid_ranges_reject_the_entire_batch() {
    let mut service = EditorService::new("edit.eqi", 1, "// 🧪\r\nmodel M() {}\r\n");
    service
        .apply_changes(2, [C::replace_range(P::new(0, 99), P::new(1, 0), "\n")])
        .unwrap();
    assert_eq!(service.current().source(), "// 🧪\nmodel M() {}\r\n");
    for (start, end) in [
        (P::new(0, 4), P::new(0, 5)),    // surrogate interior
        (P::new(0, 5), P::new(0, 4)),    // reversed
        (P::new(3, 0), P::new(3, 0)),    // absent line
        (P::new(0, 100), P::new(0, 99)), // reversed even before clamping
    ] {
        let before = service.current().clone();
        let error = service
            .apply_changes(
                3,
                [
                    C::replace_range(P::new(1, 6), P::new(1, 7), "Changed"),
                    C::replace_range(start, end, "bad"),
                ],
            )
            .unwrap_err();
        assert_eq!(error.code(), codes::PRECONDITION_FAILED);
        assert_eq!(service.current(), &before);
    }
    assert!(
        service
            .apply_changes(2, [C::replace_all("model Stale() {}")])
            .is_err()
    );
    assert_eq!(service.current().version(), 2);
    service.apply_changes(3, []).unwrap();
    assert_eq!(service.current().source(), "// 🧪\nmodel M() {}\r\n");
    assert_eq!(service.current().version(), 3);
}

#[test]
fn range_edits_recover_oversized_unindexed_source() {
    let mut service = EditorService::new(
        "edit.eqi",
        1,
        " ".repeat(EditorSnapshot::MAX_SOURCE_BYTES + 1),
    );
    assert!(!service.current().diagnostics().is_empty());
    let snapshot = service
        .apply_changes(
            2,
            [C::replace_range(
                P::new(0, 0),
                P::new(0, u32::MAX),
                "model Small() {relation r{1=1;}}",
            )],
        )
        .unwrap();
    assert_eq!(
        snapshot.source().len(),
        "model Small() {relation r{1=1;}}".len()
    );
    assert_eq!(snapshot.source(), "model Small() {relation r{1=1;}}");
    assert!(
        snapshot.diagnostics().is_empty(),
        "{:?}",
        snapshot.diagnostics()
    );
    assert_eq!(snapshot.symbols()[0].name(), "Small");
}
