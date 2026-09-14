use eqiora_compiler::compile;
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_meshing::{AffineGeometryMap, QuadratureRule, ReferenceCell};

use super::*;

fn program(source: &str) -> KernelProgram {
    let (transaction, model, _) = compile("linear.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
}

fn derive(source: &str) -> Result<CompiledLinearBlockForm, Diagnostic> {
    let program = program(source);
    let domain = program
        .nodes()
        .find_map(|node| match node {
            KernelNode::Domain(domain)
                if matches!(domain.kind(), DomainKind::CartesianBox { .. }) =>
            {
                Some(domain.id().erase())
            }
            _ => None,
        })
        .unwrap();
    let form =
        CompiledLinearBlockForm::derive(&program, domain, 1, &std::collections::BTreeSet::new())?;
    assert_eq!(form.domain(), domain);
    Ok(form)
}

fn source(reaction: &[Vec<f64>], reverse: bool) -> String {
    let count = reaction.len();
    let mut source = String::from(
        "model Linear() { domain body = box(0, 2); domain left = boundary(body, axis = 0, side = lower); domain right = boundary(body, axis = 0, side = upper); parameter unit: 1 / m ^ 2 = 1;\n",
    );
    let order = if reverse {
        (0..count).rev().collect::<Vec<_>>()
    } else {
        (0..count).collect()
    };
    for &i in &order {
        source += &format!("variable f{i}: 1 on body;\n");
    }
    for &i in &order {
        source += &format!("relation row{i} on body {{ -div({} * grad(f{i}))", i + 2);
        for (j, value) in reaction[i].iter().enumerate() {
            if *value != 0.0 {
                source += &format!(" + ({value}) * unit * f{j}");
            }
        }
        source += &format!(" - {} * unit = 0; }}\n", i + 1);
        for boundary in ["left", "right"] {
            source += &format!("relation {boundary}{i} on {boundary} {{ trace(f{i}) = 0; }}\n");
        }
    }
    source += "}";
    if reverse {
        source = source
            .replace("f0", "renamed_first")
            .replace("row", "equation");
    }
    source
}

fn geometry() -> AffineGeometryMap {
    // Reference interval [-1,1] maps to [0,2], so h = 2.
    AffineGeometryMap::new(
        ReferenceCell::hypercube(1).unwrap(),
        1,
        vec![1.0],
        vec![1.0],
    )
    .unwrap()
}

fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
}

#[test]
fn whole_row_reversal_preserves_diffusion_reaction_and_source() {
    let source = source(&[vec![3.0]], false);
    let reversed = source.replace(
        "-div(2 * grad(f0)) + (3) * unit * f0 - 1 * unit",
        "div(2 * grad(f0)) - (3) * unit * f0 + 1 * unit",
    );
    assert_ne!(source, reversed);
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(1, 2).unwrap();
    let original = derive(&source)
        .unwrap()
        .volume()
        .evaluate(&geometry(), &quadrature, &BTreeMap::new())
        .unwrap();
    let reversed = derive(&reversed)
        .unwrap()
        .volume()
        .evaluate(&geometry(), &quadrature, &BTreeMap::new())
        .unwrap();
    assert_eq!(original, reversed);
    let negative = source.replace("2 * grad(f0)", "(-2) * grad(f0)");
    assert!(
        derive(&negative)
            .unwrap()
            .volume()
            .evaluate(&geometry(), &quadrature, &BTreeMap::new())
            .is_err()
    );
}

#[test]
fn parameter_point_rebinding_preserves_the_original_compiled_form() {
    let source = source(&[vec![3.0]], false);
    let program = program(&source);
    let domain = program
        .nodes()
        .find_map(|node| match node {
            KernelNode::Domain(domain)
                if matches!(domain.kind(), DomainKind::CartesianBox { .. }) =>
            {
                Some(domain.id().erase())
            }
            _ => None,
        })
        .unwrap();
    let form =
        CompiledLinearBlockForm::derive(&program, domain, 1, &std::collections::BTreeSet::new())
            .unwrap();
    let fields = program
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Parameter(parameter) => Some(parameter.id()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(fields.len(), 1);
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(1, 2).unwrap();
    let original = form
        .volume()
        .evaluate(&geometry(), &quadrature, &BTreeMap::new())
        .unwrap();
    let rebound = form.bind_parameter_point(&fields, &[2.0]).unwrap();
    let changed_source = source.replace(
        "parameter unit: 1 / m ^ 2 = 1",
        "parameter unit: 1 / m ^ 2 = 2",
    );
    assert_ne!(source, changed_source);
    let expected = derive(&changed_source)
        .unwrap()
        .volume()
        .evaluate(&geometry(), &quadrature, &BTreeMap::new())
        .unwrap();
    assert_eq!(
        rebound
            .volume()
            .evaluate(&geometry(), &quadrature, &BTreeMap::new())
            .unwrap(),
        expected
    );
    assert_eq!(
        form.volume()
            .evaluate(&geometry(), &quadrature, &BTreeMap::new())
            .unwrap(),
        original
    );
    assert!(form.bind_parameter_point(&[], &[]).is_err());
    assert!(form.bind_parameter_point(&fields, &[f64::NAN]).is_err());
    assert!(
        form.bind_parameter_point(&[fields[0], fields[0]], &[1.0, 2.0])
            .is_err()
    );
}

#[test]
fn bound_volume_preserves_field_order_and_rebound_diffusion_positivity() {
    let source = source(&[vec![1.0, -2.0], vec![3.0, 4.0]], true)
        .replace("parameter unit:", "parameter k: 1 = 2; parameter unit:")
        .replace("2 * grad(renamed_first)", "(k - 1) * grad(renamed_first)");
    let (transaction, model, symbols) = compile("bound-volume.eqi", &source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let form = CompiledLinearBlockForm::derive(
        &program,
        symbols.get("body").unwrap(),
        1,
        &std::collections::BTreeSet::new(),
    )
    .unwrap();
    assert_eq!(form.fields().len(), form.volume().fields().len());
    for (index, ((field, value_type), layout)) in
        form.fields().iter().zip(form.volume().fields()).enumerate()
    {
        assert_eq!(*field, layout.field);
        assert_eq!(*value_type, layout.value_type);
        assert_eq!(layout.range, 2 * index..2 * index + 2);
    }
    assert!(form.volume().previous_fields().is_empty());
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(1, 2).unwrap();
    let original = form
        .volume()
        .evaluate(&geometry(), &quadrature, &BTreeMap::new())
        .unwrap();
    let parameters = [
        symbols.get("k").unwrap().downcast().unwrap(),
        symbols.get("unit").unwrap().downcast().unwrap(),
    ];
    for k in [0.0, 1.0] {
        let rebound = form.bind_parameter_point(&parameters, &[k, 1.0]).unwrap();
        let volume = rebound.volume().clone();
        let error = volume
            .evaluate(&geometry(), &quadrature, &BTreeMap::new())
            .unwrap_err();
        assert!(
            error.message().contains("positive finite diffusion"),
            "{error:?}"
        );
    }
    assert_eq!(
        form.volume()
            .evaluate(&geometry(), &quadrature, &BTreeMap::new())
            .unwrap(),
        original
    );
}

fn check(reaction: &[Vec<f64>], reverse: bool) {
    let form = derive(&source(reaction, reverse)).unwrap();
    assert!(form.fields().windows(2).all(|pair| pair[0].0 < pair[1].0));
    assert_eq!(form.dimension(), 1);
    assert_eq!(form.relations.len(), reaction.len());
    assert_eq!(form.residual_types.len(), reaction.len());
    assert_eq!(form.boundary_laws().len(), reaction.len());
    assert!(
        form.dependencies
            .values()
            .all(|dependencies| !dependencies.is_empty())
    );
    let local = form
        .volume()
        .evaluate(
            &geometry(),
            &QuadratureRule::tensor_product_gauss_legendre(1, 2).unwrap(),
            &BTreeMap::new(),
        )
        .unwrap();
    let count = reaction.len();
    // Unique independently prescribed forcing identifies rows across renamed IDs.
    let physical_rows = (0..count)
        .map(|row| local.rhs()[2 * row].round() as usize - 1)
        .collect::<Vec<_>>();
    for row in 0..count {
        let physical_row = physical_rows[row];
        for test in 0..2 {
            close(local.rhs()[2 * row + test], (physical_row + 1) as f64);
            for column in 0..count {
                for trial in 0..2 {
                    // Integrating Q1 gives diffusion k/h [1,-1;-1,1] and mass h/6 [2,1;1,2].
                    let diffusion = if row == column {
                        (physical_row + 2) as f64 / 2.0 * if test == trial { 1.0 } else { -1.0 }
                    } else {
                        0.0
                    };
                    let mass = 2.0 / 6.0 * if test == trial { 2.0 } else { 1.0 };
                    close(
                        local.matrix()[(row * 2 + test) * count * 2 + column * 2 + trial],
                        diffusion + mass * reaction[physical_row][physical_rows[column]],
                    );
                }
            }
        }
    }
}

#[test]
fn one_field_uses_diffusion_and_reaction_integrals() {
    check(&[vec![3.0]], false);
}

#[test]
fn nonsymmetric_two_and_three_field_blocks_keep_every_off_diagonal() {
    for reaction in [
        vec![vec![2.0, -3.0], vec![5.0, -1.0]],
        vec![
            vec![1.0, 2.0, 0.0],
            vec![0.0, -4.0, 3.0],
            vec![-5.0, 0.0, 6.0],
        ],
    ] {
        check(&reaction, false);
        check(&reaction, true);
    }
}

#[test]
fn nonlinear_coefficients_and_incomplete_boundaries_reject() {
    let valid = source(&[vec![1.0]], false);
    for mutated in [
        valid.replace("2 * grad(f0)", "f0 * grad(f0)"),
        valid.replace("unit * f0", "unit * f0 * f0"),
    ] {
        let error = derive(&mutated).unwrap_err();
        assert!(
            error.message().contains("unknown-dependent") || error.message().contains("nonlinear"),
            "{error:?}"
        );
    }
    assert!(
        derive(&valid.replace("relation right0 on right { trace(f0) = 0; }", ""))
            .unwrap_err()
            .message()
            .contains("coverage")
    );
    assert!(
        compile(
            "invalid.eqi",
            &valid.replace("parameter unit: 1 / m ^ 2", "parameter unit: m")
        )
        .is_err()
    );
}

#[test]
fn coefficient_chains_bind_the_exact_parameter_point_and_spatial_flux() {
    let authored = source(&[vec![1.0]], false)
        .replace("variable f0", "variable k: 1 on body; variable q: 1 on body; parameter slope: 1 / m = 1; relation first on body { k - (1 + slope * coordinate(0)) = 0; } relation second on body { q - k = 0; } variable f0")
        .replace("2 * grad(f0)", "q * grad(f0)");
    let form = derive(&authored).unwrap();
    assert_eq!(form.fields().len(), 1);
    assert_eq!(form.dependencies.len(), 5);
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(1, 2).unwrap();
    let local = form
        .volume()
        .evaluate(&geometry(), &quadrature, &BTreeMap::new())
        .unwrap();
    close(local.matrix()[0], 1.0 + 2.0 / 3.0);
    let changed = derive(&authored.replace("slope: 1 / m = 1", "slope: 1 / m = 2")).unwrap();
    let changed_local = changed
        .volume()
        .evaluate(&geometry(), &quadrature, &BTreeMap::new())
        .unwrap();
    close(changed_local.matrix()[0], 1.5 + 2.0 / 3.0);
    close(
        form.volume()
            .evaluate(&geometry(), &quadrature, &BTreeMap::new())
            .unwrap()
            .matrix()[0],
        local.matrix()[0],
    );
    let outside = authored.replace("-div(q * grad(f0))", "-q * div(grad(f0))");
    assert!(
        derive(&outside)
            .unwrap_err()
            .message()
            .contains("outside divergence")
    );
}

#[test]
fn rows_preserve_distinct_checked_physical_dimensions() {
    let authored = source(&[vec![0.0,3.0],vec![5.0,0.0]],false)
        .replace("variable f1: 1 on body", "variable f1: m on body")
        .replace("parameter unit:", "parameter uv: 1 / m ^ 3 = 3; parameter vu: 1 / m = 5; parameter fv: 1 / m = 2; parameter unit:")
        .replace("(3) * unit * f1", "uv * f1")
        .replace("(5) * unit * f0", "vu * f0")
        .replace("2 * unit =", "fv =");
    let form = derive(&authored).unwrap();
    assert_ne!(
        form.fields()[0].1.dimension(),
        form.fields()[1].1.dimension()
    );
    assert_ne!(
        form.residual_types[0].dimension(),
        form.residual_types[1].dimension()
    );
    let local = form
        .volume()
        .evaluate(
            &geometry(),
            &QuadratureRule::tensor_product_gauss_legendre(1, 2).unwrap(),
            &BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(local.matrix().len(), 16);
}

#[test]
fn conservation_preserves_physical_flux_orientation() {
    let source = source(&[vec![0.0]], false);
    let law = source.replace(
        "relation row0 on body { -div(2 * grad(f0)) - 1 * unit = 0; }",
        "law row0 on body { flux -2 * grad(f0); source unit; }",
    );
    assert_ne!(source, law);
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(1, 2).unwrap();
    for input in [&source, &law] {
        let local = derive(input)
            .unwrap()
            .volume()
            .evaluate(&geometry(), &quadrature, &BTreeMap::new())
            .unwrap();
        // h=2, diffusion=2: integral k N_i' N_j' = +/-1;
        // constant source=1: integral N_i = 1.
        for (actual, expected) in local.matrix().iter().zip([1.0, -1.0, -1.0, 1.0]) {
            close(*actual, expected);
        }
        for actual in local.rhs() {
            close(*actual, 1.0);
        }
    }
    let backwards = law.replace("flux -2", "flux 2");
    let error = derive(&backwards)
        .unwrap()
        .volume()
        .evaluate(&geometry(), &quadrature, &BTreeMap::new())
        .unwrap_err();
    assert!(error.message().contains("positive finite diffusion"));
}
