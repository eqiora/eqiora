use super::*;
use eqiora_core::{DimExponents, ScalarDomain, ValueLiteral, ValueType};

#[test]
fn structured_complex_initializers_keep_every_component_and_declared_unit() {
    let source = r#"model M() {
        parameter z:array<complex<m>,2> = math.complex([1,2],[-3,4]);
        parameter c:array<complex<m>,2> = math.conj(z);
        parameter r:array<m,2> = math.real(c);
        parameter i:array<m,2> = math.imag(c);
        variable u:array<complex<m>,2>;
        relation balance { u = math.complex(r,i); }
    }"#;
    let compiled = compile("structured-complex.eqi", source).unwrap();
    let values = compiled[0]
        .transaction()
        .ops()
        .iter()
        .filter_map(|op| match op {
            Op::DefineKernelNode {
                node: KernelNode::Parameter(parameter),
            } => Some(parameter.value().clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let metre = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    assert_eq!(values.len(), 4);
    for (domain, components) in [
        (ScalarDomain::Complex, [(1., -3.), (2., 4.)]),
        (ScalarDomain::Complex, [(1., 3.), (2., -4.)]),
        (ScalarDomain::Real, [(1., 0.), (2., 0.)]),
        (ScalarDomain::Real, [(3., 0.), (-4., 0.)]),
    ] {
        let expected = ValueLiteral::new(
            ValueType::scalar(domain, metre).unwrap().array(2).unwrap(),
            components,
        )
        .unwrap();
        assert!(
            values.contains(&expected),
            "missing {expected:?} in {values:?}"
        );
    }
    let formatted = eqiora_lang::format(
        &parse("structured-complex.eqi", source)
            .into_document()
            .unwrap(),
    );
    compile("formatted-structured-complex.eqi", &formatted).unwrap();
}
