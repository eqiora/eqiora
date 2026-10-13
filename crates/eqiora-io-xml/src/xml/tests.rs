use super::*;

fn limits() -> Limits {
    Limits {
        max_source_bytes: 4096,
        max_depth: 8,
        max_elements: 32,
        max_attributes_per_element: 4,
        max_text_bytes: 4096,
        attribute_references: false,
    }
}

fn parse(source: &[u8], limits: Limits, maximum_work: usize) -> Result<Node, Diagnostic> {
    parse_tree(
        source,
        limits,
        &mut WorkBudget::new(maximum_work, "test"),
        |_| Ok(()),
    )
}

#[test]
fn admitted_tree_keeps_element_paths_and_normalizes_attributes() {
    let root = parse(
        b"<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><r a=\"x\ty\"><!--skip--><c>z</c><c><d></d></c></r>",
        limits(),
        4096,
    ).unwrap();
    assert_eq!(root.name, "r");
    assert!(root.path.is_empty());
    assert_eq!(root.attributes["a"], "x y");
    assert_eq!(root.children[0].path, [0]);
    assert_eq!(root.children[0].text, "z");
    assert_eq!(root.children[1].path, [1]);
    assert_eq!(root.children[1].children[0].path, [1, 0]);
}

#[test]
fn independent_exact_resource_boundaries_include_bytes_and_events() {
    let source = b"<r a=\"b\"><c>z</c><c></c></r>";
    // Three start tags, three end tags, text, EOF: eight events. One
    // attribute. Three one-byte names, one-byte key/value, one-byte text.
    let maximum_work = source.len() + 8 + 1 + 6;
    let exact = Limits {
        max_source_bytes: source.len(),
        max_depth: 2,
        max_elements: 3,
        max_attributes_per_element: 1,
        max_text_bytes: 6,
        attribute_references: false,
    };
    assert!(parse(source, exact, maximum_work).is_ok());
    assert!(parse(source, exact, maximum_work - 1).is_err());
    for reduced in [
        Limits {
            max_source_bytes: source.len() - 1,
            ..exact
        },
        Limits {
            max_depth: 1,
            ..exact
        },
        Limits {
            max_elements: 2,
            ..exact
        },
        Limits {
            max_attributes_per_element: 0,
            ..exact
        },
        Limits {
            max_text_bytes: 5,
            ..exact
        },
    ] {
        assert!(parse(source, reduced, maximum_work).is_err());
    }
    // The declaration contributes one event and one attribute. The comment
    // contributes one event and two decoded bytes. Root start/end and EOF
    // contribute three events, plus the one-byte root name.
    let source = b"<?xml version=\"1.0\"?><!--xy--><r></r>";
    let work = source.len() + 5 + 1 + 3;
    assert!(parse(source, limits(), work).is_ok());
    assert!(parse(source, limits(), work - 1).is_err());
}

#[test]
fn both_attribute_policies_reject_malformed_or_unadmitted_xml() {
    for attribute_references in [false, true] {
        let policy = Limits {
            attribute_references,
            ..limits()
        };
        assert!(parse(b"<r></r>", policy, 4096).is_ok());
        for source in [
            "",
            "<r>",
            "<r></s>",
            "<r></r><s></s>",
            "x<r></r>",
            "<r></r>x",
            "<r/>",
            "<r><![CDATA[x]]></r>",
            "<r>&amp;</r>",
            "<!DOCTYPE r><r></r>",
            "<?instruction x?><r></r>",
            "<r a=\"x\" a=\"y\"></r>",
            "<r a=\"<\"></r>",
            "<r>]]></r>",
            "<r xmlns:a=\"x\"></r>",
            "<a:r></a:r>",
            "<r>\u{0}</r>",
            "<r>\u{fffe}</r>",
            "<r a=\"&#0;\"></r>",
            "<r a=\"&#xFFFE;\"></r>",
            "<?xml?><r></r>",
            "<?xml version=\"1.1\"?><r></r>",
            "<?xml encoding=\"UTF-8\" version=\"1.0\"?><r></r>",
            "<?xml version=\"1.0\" encoding=\"UTF-16\"?><r></r>",
            "<?xml version=\"1.0\" standalone=\"maybe\"?><r></r>",
            "<?xml version=\"1.0\" version=\"1.0\"?><r></r>",
            "<?xml version=\"1.0\" standalone=\"yes\" encoding=\"UTF-8\"?><r></r>",
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\" extra=\"x\"?><r></r>",
            " <!--x--><?xml version=\"1.0\"?><r></r>",
            "<?xml version=\"1.0\"?><?xml version=\"1.0\"?><r></r>",
            "<r><?xml version=\"1.0\"?></r>",
        ] {
            assert!(
                parse(source.as_bytes(), policy, 4096).is_err(),
                "{source:?}"
            );
        }
        assert!(parse(b"<r>\xff</r>", policy, 4096).is_err());
    }
}

#[test]
fn attribute_reference_policy_does_not_weaken_vtu() {
    let source = b"<r a=\"A&amp;B&#32;C\"></r>";
    assert!(parse(source, limits(), 4096).is_err());
    let root = parse(
        source,
        Limits {
            attribute_references: true,
            ..limits()
        },
        4096,
    )
    .unwrap();
    assert_eq!(root.attributes["a"], "A&B C");
}

#[test]
fn format_quota_is_checked_before_descendant_content() {
    let mut admitted = Vec::new();
    let result = parse_tree(
        b"<r><DataItem></DataItem><DataItem><bad></bad></DataItem></r>",
        limits(),
        &mut WorkBudget::new(4096, "test"),
        |name| {
            if name == "DataItem" && admitted.contains(&"DataItem".to_owned()) {
                return Err(invalid_import("DataItem quota"));
            }
            admitted.push(name.to_owned());
            Ok(())
        },
    );
    assert!(result.is_err());
    assert_eq!(admitted, ["r", "DataItem"]);
}

#[test]
fn allocation_and_counter_overflow_fail_closed() {
    assert!(crate::allocation::allocate_vec::<u64>(usize::MAX, "test").is_err());
    let mut work = WorkBudget::new(usize::MAX, "test");
    work.charge(usize::MAX).unwrap();
    assert!(work.charge(1).is_err());
    let root = parse(b"<r></r>", limits(), 4096).unwrap();
    let mut stack = [NodeBuilder {
        node: root,
        next_child: u32::MAX,
    }];
    assert!(next_path(&mut stack).is_err());
}
