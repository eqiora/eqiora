use super::*;
use eqiora_meshing::QuadratureRule;

fn potential_source(expression: &str) -> String {
    format!(
        "model Potential() {{
        domain body = box(0, 1, 0, 1);
        parameter length: m = 1;
        parameter a: 1 = 3;
        variable u: vector<m, 2> on body;
        variable load: 1 on body;
        relation coefficient on body {{ load = {expression}; }}
        relation balance on body {{ -div(grad(u)) - grad(load) = 0; }}
    }}"
    )
}

#[test]
fn potential_gradient_preserves_chain_rules_and_primal_domains() {
    for (potential, expected) in [
        (
            "a * (coordinate(0)/length)^2 / (1 + 5*coordinate(0)/length)",
            13.0 / 27.0,
        ),
        ("math.sin(a*coordinate(0)/length)", 3.0 * 0.75_f64.cos()),
        (
            "math.sqrt(a*coordinate(0)/length)",
            3.0 / (2.0 * 0.75_f64.sqrt()),
        ),
    ] {
        let form = derive(&potential_source(potential)).unwrap();
        close(
            form.rows[0].forcing[0].evaluate(&[0.25, 0.5]).unwrap(),
            expected,
        );
        close(form.rows[0].forcing[1].evaluate(&[0.25, 0.5]).unwrap(), 0.0);
    }
    let quotient = derive(&potential_source(
        "a * (coordinate(0)/length)^2 / (1 + 5*coordinate(0)/length)",
    ))
    .unwrap();
    assert!(quotient.rows[0].forcing[0].evaluate(&[-0.2, 0.5]).is_err());
    let square_root = derive(&potential_source("math.sqrt(a*coordinate(0)/length)")).unwrap();
    for x in [0.0, -0.25] {
        assert!(square_root.rows[0].forcing[0].evaluate(&[x, 0.5]).is_err());
    }
    assert!(derive(&potential_source("math.sqrt(-a)")).is_err());
    let constant_root = derive(&potential_source("math.sqrt(a-a)")).unwrap();
    assert!(
        constant_root.rows[0].forcing[0]
            .evaluate(&[0.25, 0.5])
            .is_err()
    );
    // The derivative 2*x is finite here, but the demanded potential x^2 is not.
    let overflowing = derive(&potential_source("(coordinate(0)/length)^2")).unwrap();
    assert!(
        overflowing.rows[0].forcing[0]
            .evaluate(&[1.0e200, 0.5])
            .is_err()
    );
}

#[test]
fn potential_gradient_rebinds_exact_parameters_without_mutating_the_original() {
    let source = potential_source("math.sin(a*coordinate(0)/length)");
    let (transaction, model, _) = compile("potential.eqi", &source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let domain = program
        .nodes()
        .find_map(|node| match node {
            KernelNode::Domain(domain) => Some(domain.id().erase()),
            _ => None,
        })
        .unwrap();
    let parameters = program
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Parameter(parameter) => {
                Some((parameter.id(), parameter.value_type().dimension()))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(parameters.len(), 2);
    // Typed units distinguish these fixture Parameters independently of their ID order.
    let values = parameters
        .iter()
        .map(|(_, units)| {
            if *units == DimExponents::DIMENSIONLESS {
                5.0
            } else {
                2.0
            }
        })
        .collect::<Vec<_>>();
    let fields = parameters
        .iter()
        .map(|(field, _)| *field)
        .collect::<Vec<_>>();
    let form = CompiledRegionForm::derive(&program, domain, 2).unwrap();
    let original = &form.rows[0].forcing[0];
    let rebound = original.bind_parameter_point(&fields, &values).unwrap();
    close(
        rebound.evaluate(&[0.25, 0.5]).unwrap(),
        2.5 * 0.625_f64.cos(),
    );
    close(
        original.evaluate(&[0.25, 0.5]).unwrap(),
        3.0 * 0.75_f64.cos(),
    );
    let reversed = original
        .bind_parameter_point(
            &fields.iter().copied().rev().collect::<Vec<_>>(),
            &values.iter().copied().rev().collect::<Vec<_>>(),
        )
        .unwrap();
    close(
        reversed.evaluate(&[0.25, 0.5]).unwrap(),
        2.5 * 0.625_f64.cos(),
    );
    let invalid = parameters
        .iter()
        .map(|(_, units)| {
            if *units == DimExponents::DIMENSIONLESS {
                5.0
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    assert!(
        original
            .bind_parameter_point(&fields, &invalid)
            .unwrap()
            .evaluate(&[0.25, 0.5])
            .is_err()
    );
}

#[test]
fn scalar_q1_uses_the_same_value_and_gradient_contractions() {
    let source = "model Scalar() {
        domain body = box(0, 2, 0, 3);
        domain left = boundary(body, axis = 0, side = lower);
        domain right = boundary(body, axis = 0, side = upper);
        domain bottom = boundary(body, axis = 1, side = lower);
        domain top = boundary(body, axis = 1, side = upper);

        parameter reaction: 1 / m ^ 2 = 3;
        parameter load: 1 / m ^ 2 = 5;
        variable u: 1 on body;
        relation balance on body { -div(2 * grad(u)) + reaction * u - load = 0; }
        relation bc0 on left { trace(u) = 0; }
        relation bc1 on right { trace(u) = 0; }
        relation bc2 on bottom { trace(u) = 0; }
        relation bc3 on top { trace(u) = 0; }
    }";
    let (transaction, model, _) = compile("shared-q1.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let domain = program
        .nodes()
        .find_map(|node| match node {
            KernelNode::Domain(domain)
                if matches!(
                    domain.kind(),
                    eqiora_schema::kernel::DomainKind::CartesianBox { .. }
                ) =>
            {
                Some(domain.id().erase())
            }
            _ => None,
        })
        .unwrap();
    let form = CompiledRegionForm::derive(&program, domain, 2).unwrap();
    let reference = ReferenceCell::hypercube(2).unwrap();
    let fields = form
        .fields()
        .map(|(field, value_type)| RegionFieldBinding {
            field,
            space: p1(),
            scale: DynQuantity::new(1.0, value_type.dimension()),
        })
        .collect::<Vec<_>>();
    let rows = form
        .rows()
        .map(|(relation, _, _)| (relation, DynQuantity::new(1.0, DimExponents::DIMENSIONLESS)))
        .collect();
    let bound = form.bind(reference, &fields, &rows, None).unwrap();
    let geometry =
        AffineGeometryMap::new(reference, 2, vec![1.0, 1.5], vec![1.0, 0.0, 0.0, 1.5]).unwrap();
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(2, 2).unwrap();
    let local = bound
        .evaluate(&geometry, &quadrature, &BTreeMap::new())
        .unwrap();
    let scalar = crate::form_compiler::linear::CompiledLinearBlockForm::derive(
        &program,
        domain,
        2,
        &std::collections::BTreeSet::new(),
    )
    .unwrap()
    .volume()
    .unwrap()
    .evaluate(&geometry, &quadrature, &BTreeMap::new())
    .unwrap();
    for (a, b) in local.matrix().iter().zip(scalar.matrix()) {
        close(*a, *b);
    }
    for (a, b) in local.rhs().iter().zip(scalar.rhs()) {
        close(*a, *b);
    }
    // Constant forcing integrates to load * area / 4 at each Q1 vertex.
    for value in local.rhs() {
        close(*value, 5.0 * 6.0 / 4.0);
    }
}

#[test]
fn vector_potential_gradient_integrates_each_physical_component() {
    let form = derive(
        "model Loaded() {
        domain body = box(0, 1, 0, 1);
        parameter a: 1 / m ^ 2 = 3;
        parameter b: 1 / m ^ 2 = 5;
        variable u: vector<m, 2> on body;
        variable load: 1 on body;
        relation coefficient on body {
            load = a * coordinate(0)^2 + b * coordinate(1)^2;
        }
        relation balance on body { -div(grad(u)) - grad(load) = 0; }
    }",
    )
    .unwrap();
    let fields = form
        .fields()
        .map(|(field, value_type)| RegionFieldBinding {
            field,
            space: p1(),
            scale: DynQuantity::new(1.0, value_type.dimension()),
        })
        .collect::<Vec<_>>();
    let rows = form
        .rows()
        .map(|(relation, _, value_type)| {
            let dimension = value_type
                .dimension()
                .mul(dim([0, 2, 0, 0, 0, 0, 0]))
                .unwrap()
                .pow(-1, 1)
                .unwrap();
            (relation, DynQuantity::new(1.0, dimension))
        })
        .collect();
    let bound = form
        .bind(ReferenceCell::simplex(2).unwrap(), &fields, &rows, None)
        .unwrap();
    let local = bound
        .evaluate(
            &geometry(),
            &simplex_duffy_gauss_legendre(2, 3).unwrap(),
            &BTreeMap::new(),
        )
        .unwrap();
    // On the unit right triangle, x=N1 and y=N2. The load is (6x, 10y).
    // Integral Ni*Nj is 1/12 on the diagonal and 1/24 off the diagonal.
    let expected = [
        1.0 / 4.0,
        5.0 / 12.0,
        1.0 / 2.0,
        5.0 / 12.0,
        1.0 / 4.0,
        5.0 / 6.0,
    ];
    assert_eq!(local.rhs().len(), expected.len());
    for (actual, expected) in local.rhs().iter().zip(expected) {
        close(*actual, expected);
    }
}
