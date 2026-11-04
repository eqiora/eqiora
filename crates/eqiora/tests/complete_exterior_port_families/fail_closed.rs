//! Fail-closed integration evidence for RFC 0041 complete exteriors.

use eqiora::Diagnostic;
use eqiora::compiler::compile;
use eqiora::compiler::source_identity::LocalSourceIdentityLimits;
use eqiora::diagnostic::codes;

const MISSING_SIDE: &str = include_str!(
    "../../../../verify/packages/complete-exterior-port-families/invalid/missing-side.eqi"
);
const DUPLICATE_EXACT: &str = include_str!(
    "../../../../verify/packages/complete-exterior-port-families/invalid/duplicate-exact.eqi"
);
const DUPLICATE_GEOMETRY: &str = include_str!(
    "../../../../verify/packages/complete-exterior-port-families/invalid/duplicate-geometry.eqi"
);
const WRONG_PARENT: &str = include_str!(
    "../../../../verify/packages/complete-exterior-port-families/invalid/wrong-parent.eqi"
);
const VOLUME_MEMBER: &str = include_str!(
    "../../../../verify/packages/complete-exterior-port-families/invalid/volume-member.eqi"
);
const BOUNDARY_OF_BOUNDARY: &str = include_str!(
    "../../../../verify/packages/complete-exterior-port-families/invalid/boundary-of-boundary.eqi"
);
const WRONG_DIMENSION: &str = include_str!(
    "../../../../verify/packages/complete-exterior-port-families/invalid/wrong-dimension.eqi"
);
const EMPTY_SET: &str = include_str!(
    "../../../../verify/packages/complete-exterior-port-families/invalid/empty-set.eqi"
);
const NON_BOUNDARY_SELECTOR: &str = include_str!(
    "../../../../verify/packages/complete-exterior-port-families/invalid/non-boundary-selector.eqi"
);
const SELECTOR_OUTSIDE_BOUND_SET: &str = include_str!(
    "../../../../verify/packages/complete-exterior-port-families/invalid/selector-outside-bound-set.eqi"
);
const DISTINCT_CONNECTOR: &str = include_str!(
    "../../../../verify/packages/complete-exterior-port-families/invalid/distinct-connector.eqi"
);
const UNCONNECTED_FAMILY: &str = include_str!(
    "../../../../verify/packages/complete-exterior-port-families/invalid/unconnected-family.eqi"
);

struct RejectionCase {
    name: &'static str,
    source: &'static str,
    required_fragments: &'static [&'static str],
}

const REJECTION_CASES: &[RejectionCase] = &[
    RejectionCase {
        name: "missing-side",
        source: MISSING_SIDE,
        required_fragments: &[
            "complete-exterior slot `exterior`",
            "missing Cartesian side (1, upper)",
        ],
    },
    RejectionCase {
        name: "duplicate-exact",
        source: DUPLICATE_EXACT,
        required_fragments: &[
            "complete-exterior slot `exterior`",
            "one exact Boundary more than once",
        ],
    },
    RejectionCase {
        name: "duplicate-geometry",
        source: DUPLICATE_GEOMETRY,
        required_fragments: &[
            "complete-exterior slot `exterior`",
            "Cartesian side (0, lower) more than once",
        ],
    },
    RejectionCase {
        name: "wrong-parent",
        source: WRONG_PARENT,
        required_fragments: &[
            "complete-exterior slot `exterior`",
            "Boundary of a different exact parent",
        ],
    },
    RejectionCase {
        name: "volume-member",
        source: VOLUME_MEMBER,
        required_fragments: &[
            "BoundarySet member `body`",
            "not an enclosing Cartesian Domain",
        ],
    },
    RejectionCase {
        name: "boundary-of-boundary",
        source: BOUNDARY_OF_BOUNDARY,
        required_fragments: &["Cartesian boundary parent must be a Cartesian box Domain"],
    },
    RejectionCase {
        name: "wrong-dimension",
        source: WRONG_DIMENSION,
        required_fragments: &["volume support slot `body` requires ambient dimension 3"],
    },
    RejectionCase {
        name: "empty-set",
        source: EMPTY_SET,
        required_fragments: &[
            "complete-exterior slot `exterior`",
            "requires a nonempty BoundarySet",
        ],
    },
    RejectionCase {
        name: "non-boundary-selector",
        source: NON_BOUNDARY_SELECTOR,
        required_fragments: &[
            "boundary selector target `body`",
            "is not a boundary Domain",
        ],
    },
    RejectionCase {
        name: "selector-outside-bound-set",
        source: SELECTOR_OUTSIDE_BOUND_SET,
        required_fragments: &[
            "Port family `solid.mechanical`",
            "has no member on exact Boundary `outside_x_lower`",
        ],
    },
    RejectionCase {
        name: "distinct-connector",
        source: DISTINCT_CONNECTOR,
        required_fragments: &[
            "exact boundary Connection",
            "requires the same specialized Connector",
        ],
    },
    RejectionCase {
        name: "unconnected-family",
        source: UNCONNECTED_FAMILY,
        required_fragments: &[
            "occurrence-level physical connection closure",
            "belongs to no normalized connection set",
        ],
    },
];

fn matching_diagnostic<'a>(
    diagnostics: &'a [Diagnostic],
    fragments: &[&str],
) -> Option<&'a Diagnostic> {
    diagnostics.iter().find(|diagnostic| {
        fragments
            .iter()
            .all(|fragment| diagnostic.message().contains(fragment))
    })
}

#[test]
fn invalid_complete_exteriors_and_selectors_fail_before_model_exposure() {
    for case in REJECTION_CASES {
        let file = format!("invalid/{}.eqi", case.name);
        let diagnostics = compile(&file, case.source).expect_err(case.name);
        let diagnostic =
            matching_diagnostic(&diagnostics, case.required_fragments).unwrap_or_else(|| {
                panic!(
                    "{}: expected one diagnostic containing {:?}, got {diagnostics:#?}",
                    case.name, case.required_fragments
                )
            });

        assert_eq!(
            diagnostic.code(),
            codes::LANGUAGE_TYPE_ERROR,
            "{}: stable diagnostic class",
            case.name
        );
        let span = diagnostic
            .source_span()
            .unwrap_or_else(|| panic!("{}: diagnostic must identify source", case.name));
        assert_eq!(span.file, file, "{}: diagnostic file", case.name);
        assert!(span.start < span.end, "{}: nonempty source span", case.name);
    }
}

#[test]
fn explicit_membership_limit_precedes_geometric_elaboration() {
    let limit = LocalSourceIdentityLimits::default().max_boundary_set_members;
    let member_count = limit.checked_add(1).expect("default limit is finite");
    let members = std::iter::repeat_n("x_lower", member_count)
        .collect::<Vec<_>>()
        .join(", ");
    let source = format!(
        r#"
public component ExteriorContract(
  support body: volume(ambient_dimension = 2),
  support exterior: complete_exterior(parent = body),
) {{

}}

model Main() {{
  domain body = box(0, 1, 0, 1);
  domain x_lower = boundary(body, axis = 0, side = lower);
  instance oversized: ExteriorContract(
    body = body,
    exterior = boundaries({members})
  );
}}
"#
    );

    let diagnostics = compile("invalid/oversized-boundary-set.eqi", &source)
        .expect_err("an oversized explicit set must expose no model");
    let expected = format!(
        "complete-exterior binding has {member_count} members, exceeding the {limit} member limit"
    );
    assert!(
        diagnostics.iter().any(|diagnostic| {
            diagnostic.code() == codes::LANGUAGE_LOWERING_ERROR
                && diagnostic.message().contains(&expected)
        }),
        "expected bounded rejection `{expected}`, got {diagnostics:#?}"
    );
    assert!(
        diagnostics.iter().all(|diagnostic| !diagnostic
            .message()
            .contains("one exact Boundary more than once")),
        "resource rejection must precede geometric proof"
    );
}

#[test]
fn unused_abstract_exterior_defers_cardinality_until_an_exact_occurrence() {
    // Ambient dimension does not determine a Geometry partition's part count.
    // No occurrence exists here, so no 2*dimension member list may be invented.
    let dimension = usize::MAX / 2 + 1;
    let declarations = format!(
        r#"
public connector BoundaryScalar {{
  trace value: 1;
  flux flux: 1;
  shape [];
  frame invariant;
  pairing euclidean_boundary_duality;
  orientation parent_outward;
}}
public component AbstractFamily(
  support body: volume(ambient_dimension = {dimension}),
  support exterior: complete_exterior(parent = body),
  port boundary[member in exterior]: BoundaryScalar over member
) {{}}
"#
    );
    let unused = format!(
        "{declarations} model Main() {{ variable retained:1; relation keep {{ retained=0; }} }}"
    );
    compile("abstract-family.eqi", &unused).expect("unused abstract topology allocates no members");

    // A real two-dimensional occurrence must still discharge the exact
    // dimension contract before any Port family output can be allocated.
    let bound = format!(
        "{declarations} model Main() {{ domain body=box(0,1,0,1);domain left=boundary(body,axis=0,side=lower);domain right=boundary(body,axis=0,side=upper);domain bottom=boundary(body,axis=1,side=lower);domain top=boundary(body,axis=1,side=upper);instance family:AbstractFamily(body=body,exterior=boundaries(left,right,bottom,top)); }}"
    );
    let diagnostics = compile("wrong-dimension-family.eqi", &bound)
        .expect_err("2D support cannot satisfy the huge exact dimension contract");
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code() == codes::LANGUAGE_TYPE_ERROR
                && diagnostic.message().contains("requires ambient dimension")),
        "{diagnostics:#?}"
    );
}
