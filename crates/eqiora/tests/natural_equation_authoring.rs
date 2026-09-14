//! Fixed, independently enumerated observations for uniform authored equality.

use eqiora::api::ModelDocument;
use eqiora::kernel::{ExprDag, ExprId, ExprNode, KernelNode, SymbolRef};
use eqiora::language::{
    BinaryOp, Document, DraftField, DraftRelation, Expr, ExprKind, Item, Module, RelationDecl,
    UnaryOp, format, parse,
};
use eqiora::package::{
    BundleEntryV1, BundleRoleV1, ExactVersion, InMemoryPackageStore, NormalizedRelativePath,
    PackageManifestV1, PackageSourcesV1, PackagedModelDocument, QualifiedName, ResolutionRecordV1,
    SourceFileV1, prepare_package_release_v1,
};
use eqiora::{DimExponents, ValueType};

const NATURAL: &str =
    include_str!("../../../verify/language/natural-equation-authoring/models/natural.eqi");
const EXPLICIT: &str = include_str!(
    "../../../verify/language/natural-equation-authoring/models/explicit-residual.eqi"
);

#[derive(Clone, Debug, PartialEq, Eq)]
enum Tree {
    Number(u64),
    Name(String),
    Neg(Box<Self>),
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
    Pow(Box<Self>, Box<Self>),
}

fn n(name: &str) -> Tree {
    Tree::Name(name.to_owned())
}
fn z() -> Tree {
    Tree::Number(0.0_f64.to_bits())
}
fn number(value: f64) -> Tree {
    Tree::Number(value.to_bits())
}
fn neg(value: Tree) -> Tree {
    Tree::Neg(Box::new(value))
}
fn sub(left: Tree, right: Tree) -> Tree {
    Tree::Sub(Box::new(left), Box::new(right))
}
fn add(left: Tree, right: Tree) -> Tree {
    Tree::Add(Box::new(left), Box::new(right))
}
fn mul(left: Tree, right: Tree) -> Tree {
    Tree::Mul(Box::new(left), Box::new(right))
}

fn source_tree(expression: &Expr) -> Tree {
    match expression.kind() {
        ExprKind::Number(value) => {
            number(value.canonical_text().parse().expect("finite real fixture"))
        }
        ExprKind::Name(name) => n(name),
        ExprKind::Unary {
            op: UnaryOp::Neg,
            value,
        } => neg(source_tree(value)),
        ExprKind::Binary { op, left, right } => {
            let (left, right) = (source_tree(left), source_tree(right));
            match op {
                BinaryOp::Add => add(left, right),
                BinaryOp::Sub => sub(left, right),
                BinaryOp::Mul => mul(left, right),
                BinaryOp::Div => Tree::Div(Box::new(left), Box::new(right)),
                BinaryOp::Pow => Tree::Pow(Box::new(left), Box::new(right)),
                other => panic!("outside this fixed arithmetic source-tree corpus: {other:?}"),
            }
        }
        other => panic!("outside this fixed source-tree corpus: {other:?}"),
    }
}

fn relation(document: &Document) -> &RelationDecl {
    let [model] = document.models() else {
        panic!("one model")
    };
    let mut relations = model.items().iter().filter_map(|item| {
        if let Item::Relation(value) = item {
            Some(value)
        } else {
            None
        }
    });
    let relation = relations.next().unwrap();
    assert!(relations.next().is_none());
    relation
}

fn compiled_roots(model: &ModelDocument) -> Vec<Tree> {
    fn project(model: &ModelDocument, dag: &ExprDag, id: ExprId) -> Tree {
        match &dag.nodes()[id.index() as usize] {
            ExprNode::Constant(value) => number(
                value
                    .real_scalar_value()
                    .expect("real scalar fixture")
                    .value(),
            ),
            ExprNode::Symbol(symbol) => {
                let id = match symbol {
                    SymbolRef::Field(id) => id.erase(),
                    SymbolRef::Parameter(id) => id.erase(),
                    other => panic!("unexpected fixed-corpus symbol: {other:?}"),
                };
                n(model
                    .aliases()
                    .iter()
                    .find_map(|(name, target)| (*target == id).then_some(name.as_str()))
                    .expect("named fixed-corpus symbol"))
            }
            ExprNode::Neg(value) => neg(project(model, dag, *value)),
            ExprNode::Add(left, right) => {
                add(project(model, dag, *left), project(model, dag, *right))
            }
            ExprNode::Sub(left, right) => {
                sub(project(model, dag, *left), project(model, dag, *right))
            }
            ExprNode::Mul(left, right) => {
                mul(project(model, dag, *left), project(model, dag, *right))
            }
            ExprNode::Div(left, right) => Tree::Div(
                Box::new(project(model, dag, *left)),
                Box::new(project(model, dag, *right)),
            ),
            other => panic!("outside this fixed typed-tree corpus: {other:?}"),
        }
    }
    let mut relations = model.program().nodes().filter_map(|node| {
        if let KernelNode::Relation(value) = node {
            Some(value)
        } else {
            None
        }
    });
    let relation = relations.next().unwrap();
    assert!(relations.next().is_none());
    let dag = &model
        .program()
        .numerical_residuals(relation.id().erase())
        .unwrap();
    assert!(dag.nodes().len() <= 32);
    dag.roots()
        .iter()
        .map(|root| project(model, dag, *root))
        .collect()
}

fn compile(source: &str) -> ModelDocument {
    assert!(source.len() <= 4096);
    ModelDocument::compile("oracle.eqi", source)
        .unwrap_or_else(|error| panic!("ordinary positive must compile: {error:?}\n{source}"))
}

fn statements(body: &str) -> String {
    format!(
        "// α\r\nmodel probe() {{ variable x: 1; variable y: 1; variable z: 1; parameter zero: 1 = 0; relation r {{ {body} }} }}"
    )
}

#[test]
fn authored_equations_and_typed_residuals_have_distinct_ordered_owners() {
    let natural = compile(NATURAL); // Public positive precedes all denials.
    let explicit = compile(EXPLICIT);
    let expected = vec![
        sub(n("a"), n("b")),
        sub(sub(n("a"), sub(n("b"), n("c"))), n("d")),
        sub(neg(n("a")), neg(n("b"))),
    ];
    assert_eq!(compiled_roots(&natural), expected);
    assert_eq!(compiled_roots(&explicit), expected);
    // Authored equation sides are authority; only the numeric projection agrees.
    assert!(!natural.structurally_equivalent(&explicit).unwrap());
    assert_ne!(
        natural.structural_fingerprint().unwrap(),
        explicit.structural_fingerprint().unwrap()
    );
    assert_ne!(
        natural.artifact_reference().unwrap(),
        explicit.artifact_reference().unwrap()
    );
    assert_ne!(natural.digest().unwrap(), explicit.digest().unwrap());
    let natural_bytes = natural.canonical_json().unwrap();
    let explicit_bytes = explicit.canonical_json().unwrap();
    assert!(natural_bytes.len() <= 256 * 1024 && explicit_bytes.len() <= 256 * 1024);
    assert_ne!(natural_bytes, explicit_bytes);

    for (source, expected_sides) in [
        (
            NATURAL,
            vec![
                (n("a"), n("b")),
                (sub(n("a"), sub(n("b"), n("c"))), n("d")),
                (neg(n("a")), neg(n("b"))),
            ],
        ),
        (
            EXPLICIT,
            expected.iter().cloned().map(|left| (left, z())).collect(),
        ),
    ] {
        let document = parse("oracle.eqi", source).into_document().unwrap();
        let actual = relation(&document)
            .conditions()
            .unwrap()
            .iter()
            .map(|equation| (source_tree(equation.left()), source_tree(equation.right())))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected_sides);
        assert_eq!(format(&document), source); // Each form owns its own canonical bytes.
        let reparsed = parse("oracle.eqi", &format(&document))
            .into_document()
            .unwrap();
        assert_eq!(format(&reparsed), source);
    }

    // Actual source mutations must change the ordered typed structure. None of
    // these is a comparator clone that can pass without executing the product.
    for mutant in [
        NATURAL.replace("a = b;", "a * b = 0;"),
        NATURAL.replace("a = b;", "a = 0;"),
        NATURAL.replace("a = b;", "b = a;"),
        NATURAL.replace("a = b;", "a + b = 0;"),
        NATURAL.replace("-a = -b;", "-a = b;"),
        NATURAL.replace("a - (b - c) = d;", "a - (c - b) = d;"),
        NATURAL.replace("a - (b - c) = d;", "a - b - c = d;"),
        NATURAL.replace(
            "    a = b;\n    a - (b - c) = d;",
            "    a - (b - c) = d;\n    a = b;",
        ),
    ] {
        let changed = compile(&mutant);
        assert_ne!(compiled_roots(&changed), expected);
        assert_ne!(
            changed.structural_fingerprint().unwrap(),
            natural.structural_fingerprint().unwrap()
        );
        assert!(!changed.structurally_equivalent(&natural).unwrap());
    }
}

#[test]
fn fixed_zero_precedence_and_range_table_has_no_parser_sentinel() {
    // (input, authored lhs, authored rhs, checked root, canonical statement).
    let cases = vec![
        ("x = 0;", n("x"), z(), n("x"), "x = 0;"),
        ("x = (0);", n("x"), z(), n("x"), "x = 0;"),
        ("x = ((0));", n("x"), z(), n("x"), "x = 0;"),
        ("x = -0;", n("x"), neg(z()), n("x"), "x = -0;"),
        ("x = (-0);", n("x"), neg(z()), n("x"), "x = -0;"),
        ("x = -(-0);", n("x"), neg(neg(z())), n("x"), "x = --0;"),
        ("x = 0e-999;", n("x"), z(), n("x"), "x = 0;"),
        ("0 = x;", z(), n("x"), sub(z(), n("x")), "0 = x;"),
        (
            "x - 0 = 0;",
            sub(n("x"), z()),
            z(),
            sub(n("x"), z()),
            "x - 0 = 0;",
        ),
        (
            "x = 0 * y;",
            n("x"),
            mul(z(), n("y")),
            sub(n("x"), mul(z(), n("y"))),
            "x = 0 * y;",
        ),
        (
            "x = y - y;",
            n("x"),
            sub(n("y"), n("y")),
            sub(n("x"), sub(n("y"), n("y"))),
            "x = y - y;",
        ),
        (
            "x = zero;",
            n("x"),
            n("zero"),
            sub(n("x"), n("zero")),
            "x = zero;",
        ),
        (
            "x - y = z;",
            sub(n("x"), n("y")),
            n("z"),
            sub(sub(n("x"), n("y")), n("z")),
            "x - y = z;",
        ),
        (
            "x = y - z;",
            n("x"),
            sub(n("y"), n("z")),
            sub(n("x"), sub(n("y"), n("z"))),
            "x = y - z;",
        ),
        (
            "-x = -y;",
            neg(n("x")),
            neg(n("y")),
            sub(neg(n("x")), neg(n("y"))),
            "-x = -y;",
        ),
        (
            "x - (y - z) = x;",
            sub(n("x"), sub(n("y"), n("z"))),
            n("x"),
            sub(sub(n("x"), sub(n("y"), n("z"))), n("x")),
            "x - (y - z) = x;",
        ),
    ];
    assert_eq!(cases.len(), 16);
    for (input, left, right, root, golden) in cases {
        let source = statements(input);
        let document = parse("range.eqi", &source).into_document().unwrap();
        let equation = &relation(&document).conditions().unwrap()[0];
        assert_eq!(source_tree(equation.left()), left);
        assert_eq!(source_tree(equation.right()), right);
        let statement_start = source.find(input).unwrap();
        let split = input.find('=').unwrap();
        let left_text = input[..split].trim_end();
        let right_text = input[split + 1..input.len() - 1].trim_start();
        let right_start = statement_start + input.len() - 1 - right_text.len();
        assert_eq!(
            (
                equation.range().start() as usize,
                equation.range().end() as usize
            ),
            (statement_start, statement_start + input.len() - 1)
        );
        assert_eq!(
            (
                equation.left().range().start() as usize,
                equation.left().range().end() as usize
            ),
            (statement_start, statement_start + left_text.len())
        );
        assert_eq!(
            (
                equation.right().range().start() as usize,
                equation.right().range().end() as usize
            ),
            (right_start, right_start + right_text.len())
        );
        let formatted = format(&document);
        assert!(formatted.len() <= 1024);
        assert!(formatted.lines().any(|line| line.trim() == golden));
        assert_eq!(
            format(&parse("range.eqi", &formatted).into_document().unwrap()),
            formatted
        );
        assert_eq!(compiled_roots(&compile(&source)), vec![root.clone()]);
        assert_eq!(compiled_roots(&compile(&formatted)), vec![root]);
    }
}

#[test]
fn zero_cannot_erase_an_unchecked_operand_or_underflow() {
    for rhs in [
        "0",
        "(0)",
        "-0",
        "(-0)",
        "-(-0)",
        "0[m]",
        "-0[m]",
        "0e-999[m]",
    ] {
        let source =
            format!("model typed() {{ variable force: m; relation r {{ force = {rhs}; }} }}");
        assert_eq!(compiled_roots(&compile(&source)), vec![n("force")]);
    }
    // All denials reach the source-owned boundary after a positive of the same
    // family; offsets are asserted against the authored input, never outputs.
    for (body, expected_code, message) in [
        (
            "variable x: m; relation r { x = 0[s]; }",
            "EQ0603",
            "incompatible types",
        ),
        (
            "variable x: m; relation r { x = 0[unknown]; }",
            "EQ0603",
            "unknown input-unit",
        ),
        (
            "variable x: m; relation r { x = 0 * missing; }",
            "EQ0603",
            "unresolved",
        ),
        (
            "variable x: m; relation r { x = 0 * 1[s]; }",
            "EQ0603",
            "incompatible types",
        ),
        (
            "variable x: 1; relation r { x = 1e-324; }",
            "EQ0603",
            "underflows",
        ),
        (
            "variable x: 1; relation r { x = (-1e-324); }",
            "EQ0603",
            "underflows",
        ),
        (
            "variable x: m; relation r { x = 5e-324[mm]; }",
            "EQ0603",
            "underflows",
        ),
        (
            "variable x: 1; relation r { x = 1e999; }",
            "EQ0603",
            "finite",
        ),
        (
            "variable x: 1; relation r { x = 0 = 0; }",
            "EQ0602",
            "after equation",
        ),
    ] {
        let source = format!("// α\nmodel bad() {{ {body} }}");
        let diagnostics = ModelDocument::compile("negative.eqi", &source).unwrap_err();
        let diagnostic = diagnostics
            .iter()
            .find(|diagnostic| {
                diagnostic.code().to_string() == expected_code
                    && diagnostic.message().contains(message)
            })
            .unwrap_or_else(|| panic!("{body}: {diagnostics:?}"));
        let span = diagnostic.source_span().expect("source-owned rejection");
        assert_eq!(span.file, "negative.eqi");
        assert!(span.start < span.end && span.end as usize <= source.len());
        assert!(
            source.is_char_boundary(span.start as usize)
                && source.is_char_boundary(span.end as usize)
        );
    }
    // Nonzero representable subnormal is not treated as exact zero.
    let subnormal = compile("model M() { variable x: 1; relation r { x = 5e-324; } }");
    assert_eq!(
        compiled_roots(&subnormal),
        vec![sub(n("x"), number(f64::from_bits(1)))]
    );
}

#[test]
fn exact_package_and_native_residuals_share_only_checked_structural_meaning() {
    let natural = compile(NATURAL);
    let path = NormalizedRelativePath::parse("src/models/natural.eqi").unwrap();
    let manifest = PackageManifestV1::new(
        "models.natural",
        QualifiedName::parse("org.eqiora.oracle.NaturalEquation").unwrap(),
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
            NATURAL.as_bytes().to_vec(),
        )],
    )
    .unwrap();
    let release = prepare_package_release_v1(sources, &[]).unwrap();
    let mut store = InMemoryPackageStore::default();
    store.insert(&release).unwrap();
    let resolution = ResolutionRecordV1::from_exact_releases(&release, &[]).unwrap();
    let packaged =
        PackagedModelDocument::compile_locked(&store, &resolution, "natural_equation_oracle")
            .unwrap();
    assert!(packaged.model().structurally_equivalent(&natural).unwrap());
    assert_eq!(
        packaged.model().structural_fingerprint().unwrap(),
        natural.structural_fingerprint().unwrap()
    );

    let field = |name| {
        DraftField::new(
            name,
            ValueType::scalar(eqiora::ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                .expect("valid scalar type"),
            eqiora::language::FieldRoleSyntax::Variable,
        )
    };
    let (a, b, c, d) = (field("a"), field("b"), field("c"), field("d"));
    let relation = DraftRelation::continuous(
        "balance",
        [
            (a.expression(), b.expression()),
            (
                a.expression() - (b.expression() - c.expression()),
                d.expression(),
            ),
            (-a.expression(), -b.expression()),
        ],
    );
    let draft = Module::new(
        "natural_equation_oracle",
        vec![a.into(), b.into(), c.into(), d.into(), relation.into()],
    )
    .unwrap();
    let native = ModelDocument::compile_module(&draft, None, &[]).unwrap();
    assert!(native.structurally_equivalent(&natural).unwrap());
    assert_eq!(
        native.structural_fingerprint().unwrap(),
        natural.structural_fingerprint().unwrap()
    );
}
