use eqiora_compiler::compile;
use eqiora_graph::{GraphStore, InMemoryGraphStore};

use super::*;

fn signature(roles: &EquationRoles) -> Vec<(u8, String, usize, usize)> {
    let mut result = roles
        .relations
        .values()
        .map(|entry| {
            let (kind, field) = match entry.kind {
                Role::Coefficient { field } => (0, field),
                Role::Kinematic { state, .. } => (1, state),
                Role::Residual { tested } => (2, tested),
            };
            (
                kind,
                format!("{:?}", roles.fields[&field].1),
                entry
                    .dependencies
                    .iter()
                    .filter(|id| roles.fields.contains_key(id))
                    .count(),
                entry
                    .dependencies
                    .iter()
                    .filter(|id| !roles.fields.contains_key(id))
                    .count(),
            )
        })
        .collect::<Vec<_>>();
    result.sort();
    result
}

fn derive(source: &str) -> Result<EquationRoles, Diagnostic> {
    let mut compiled = compile("roles.eqi", source).unwrap();
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    EquationRoles::derive(
        &program,
        program.nodes().filter_map(|node| match node {
            KernelNode::Domain(domain)
                if !continuum_fields_on(&program, domain.id().erase()).is_empty() =>
            {
                Some(domain.id().erase())
            }
            _ => None,
        }),
    )
}

const THREE: &str = "model Three() {
 domain body = box(0, 1, 0, 1);

 variable a: 1 on body;
 variable b: 1 on body;
 variable c: 1 on body;
 parameter k: 1 / m ^ 2 = 1;
 relation first on body { -div(grad(a)) + k * (a - b) = 0; }
 relation second on body { -div(grad(b)) + k * (b - a) + k * (b - c) = 0; }
 relation third on body { -div(grad(c)) + k * (c - b) = 0; }
}";

#[test]
fn three_principal_scalar_equations_keep_two_way_dependencies() {
    let roles = derive(THREE).unwrap();
    assert_eq!(roles.fields.len(), 3);
    let adjacency = roles
        .relations
        .values()
        .map(|entry| {
            let Role::Residual { tested } = entry.kind else {
                panic!("principal equation")
            };
            let neighbours = entry
                .dependencies
                .iter()
                .copied()
                .filter(|id| *id != tested && roles.fields.contains_key(id))
                .collect::<BTreeSet<_>>();
            (tested, neighbours)
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(adjacency.values().map(BTreeSet::len).sum::<usize>(), 4);
    for (field, neighbours) in &adjacency {
        for neighbour in neighbours {
            assert!(adjacency[neighbour].contains(field));
        }
    }
    let mut dependency_counts = roles
        .relations
        .values()
        .map(|entry| {
            assert!(matches!(entry.kind, Role::Residual { .. }));
            entry
                .dependencies
                .iter()
                .filter(|id| roles.fields.contains_key(id))
                .count()
        })
        .collect::<Vec<_>>();
    dependency_counts.sort_unstable();
    assert_eq!(dependency_counts, [2, 2, 3]);
    let renamed = THREE
        .replace("first", "zeta")
        .replace("second", "alpha")
        .replace("third", "omega")
        .replace(
            "variable a: 1 on body;\n variable b: 1 on body;",
            "variable b: 1 on body;\n variable a: 1 on body;",
        );
    let reordered = derive(&renamed).unwrap();
    assert_eq!(reordered.relations.len(), roles.relations.len());
    assert_eq!(reordered.fields.len(), roles.fields.len());
}

#[test]
fn ambiguous_principal_selection_rejects() {
    let error =
        derive(&THREE.replace("-div(grad(a))", "-div(grad(a)) - div(grad(b))")).unwrap_err();
    assert!(error.message().contains("unique supported principal"));
}

#[test]
fn coefficient_chains_resolve_but_cycles_and_duplicates_reject() {
    let source = "model Coefficients() {
 domain body = box(0, 1, 0, 1);

 variable u: 1 on body;
 variable a: 1 / m ^ 2 on body;
 variable b: 1 / m ^ 2 on body;
 parameter p: 1 / m ^ 2 = 2;
 relation a_def on body { a - b = 0; }
 relation b_def on body { b - p = 0; }
 relation balance on body { -div(grad(u)) - a = 0; }
}";
    let roles = derive(source).unwrap();
    let reordered = source.replace(
        "relation a_def on body { a - b = 0; }\n relation b_def on body { b - p = 0; }",
        "relation renamed_b on body { p - b = 0; }\n relation renamed_a on body { b - a = 0; }",
    );
    assert_ne!(source, reordered);
    assert_eq!(signature(&roles), signature(&derive(&reordered).unwrap()));
    assert_eq!(
        roles
            .relations
            .values()
            .filter(|entry| matches!(entry.kind, Role::Coefficient { .. }))
            .count(),
        2
    );
    assert!(
        derive(&source.replace("b - p", "b - a"))
            .unwrap_err()
            .message()
            .contains("cyclic or missing")
    );
    assert!(
        derive(&source.replace(
            "relation b_def",
            "relation duplicate on body { a - p = 0; }\n relation b_def"
        ))
        .unwrap_err()
        .message()
        .contains("unambiguous definition")
    );
    assert!(derive(&source.replace("relation b_def on body { b - p = 0; }", "")).is_err());
}

#[test]
fn existing_fsi_equations_derive_coefficients_kinematics_and_mixed_trials() {
    let source = include_str!(
        "../../../../../verify/fsi/fixed-reference-monolithic-step-2d/models/direct.eqi"
    );
    let roles = derive(source).unwrap();
    assert_eq!(roles.fields.len(), 6);
    assert_eq!(
        roles
            .relations
            .values()
            .filter(|entry| matches!(entry.kind, Role::Coefficient { .. }))
            .count(),
        2
    );
    assert_eq!(
        roles
            .relations
            .values()
            .filter(|entry| matches!(entry.kind, Role::Kinematic { .. }))
            .count(),
        1
    );
    assert_eq!(
        roles
            .relations
            .values()
            .filter(|entry| matches!(entry.kind, Role::Residual { .. }))
            .count(),
        3
    );
    let renamed = derive(
        &source
            .replace("fluid", "first_body")
            .replace("solid", "second_body"),
    )
    .unwrap();
    assert_eq!(renamed.fields.len(), roles.fields.len());
    assert_eq!(renamed.relations.len(), roles.relations.len());
}

#[test]
fn mixed_constraint_requires_one_equation_paired_multiplier() {
    let source = "model Mixed() {
 domain body = box(0, 1, 0, 1);

 variable u: vector<m / s, 2> on body;
 variable p: 1 / s on body;
 relation balance on body { -div(symmetric_part(grad(u))) + grad(p) = 0; }
 relation constraint on body { div(u) = 0; }
}";
    let roles = derive(source).unwrap();
    assert_eq!(roles.relations.len(), 2);
    let reordered = source.replace(
        "relation balance on body { -div(symmetric_part(grad(u))) + grad(p) = 0; }\n relation constraint on body { div(u) = 0; }",
        "relation renamed_constraint on body { -div(u) = 0; }\n relation renamed_balance on body { -div(symmetric_part(grad(u))) + grad(p) = 0; }",
    ).replace(" p:", " multiplier:").replace("(p)", "(multiplier)")
        .replace(" u:", " motion:").replace("(u)", "(motion)");
    assert_ne!(source, reordered);
    assert_eq!(signature(&roles), signature(&derive(&reordered).unwrap()));
    assert!(
        derive(&source.replace(" + grad(p)", ""))
            .unwrap_err()
            .message()
            .contains("unique gradient multiplier")
    );
    let ambiguous = source
        .replace("variable p:", "variable q: 1 / s on body;\n variable p:")
        .replace("+ grad(p)", "+ grad(p) + grad(q)");
    assert!(
        derive(&ambiguous)
            .unwrap_err()
            .message()
            .contains("unique gradient multiplier")
    );
}

#[test]
fn conservative_dyadic_role_uses_definition_identity_not_operator_name() {
    let source = include_str!(
        "../../../../../verify/fsi/fixed-reference-monolithic-step-2d/models/direct.eqi"
    );
    let source = format!("public operator outer_product(input left: spatial[1], input right: spatial[1]): spatial[2] = component(left, 0) * component(right, 1);\n{}", source.replace(
        "fluid_density * derivative(fluid_velocity)",
        "fluid_density * derivative(fluid_velocity) + div(fluid_density * outer_product(left = fluid_velocity, right = fluid_velocity))"));
    let roles = derive(&source).unwrap();
    assert_eq!(
        signature(&roles),
        signature(&derive(&source.replace("outer_product", "renamed_product")).unwrap())
    );
    let wrong_definition = source.replace(
        "component(left, 0) * component(right, 1)",
        "component(left, 0) * component(right, 1) + component(left, 0) * component(right, 1)",
    );
    assert!(derive(&wrong_definition).is_err());
}

#[test]
fn conservative_dyadic_role_rejects_other_trials_and_nondivergence_use() {
    use eqiora_core::{Id, entity::kinds};
    use eqiora_schema::kernel::ExprDagBuilder;
    let definition = eqiora_ir::PureOperatorDefinition::dyadic_product().unwrap();
    for (same_trial, in_divergence) in [(true, true), (false, true), (true, false)] {
        let mut builder = ExprDagBuilder::new();
        let velocity = Id::<kinds::Field>::new();
        let first = builder.symbol(SymbolRef::Field(velocity)).unwrap();
        let second = builder
            .symbol(SymbolRef::Field(if same_trial {
                velocity
            } else {
                Id::new()
            }))
            .unwrap();
        let product = builder.pure_operator(&definition, [first, second]).unwrap();
        let root = if in_divergence {
            builder.divergence(product).unwrap()
        } else {
            product
        };
        let dag = builder.finish([root]).unwrap();
        let derived = expression::principal(&dag, root);
        if same_trial && in_divergence {
            assert_eq!(
                derived.unwrap(),
                (BTreeSet::from([velocity.erase()]), BTreeSet::new())
            );
        } else {
            assert!(derived.is_err());
        }
    }
}
