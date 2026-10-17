use super::*;
use crate::{DifferentiationRole, LinearizedRelation, RelationCotangent, RelationTangent};
use eqiora_core::{DimExponents, FiniteBasis, Id, ValueFrame, ValueType};
use eqiora_schema::kernel::{ExprDagBuilder, typing::RootContract};

#[test]
fn structured_complex_projections_share_primal_and_real_linear_products() {
    let unit = DimExponents::from_integers([0, 1, -2, 0, 0, 0, 0]).unwrap();
    let domain = ScalarDomain::Complex;
    let left = FiniteBasis::new(Id::new(), 2).unwrap();
    let right = FiniteBasis::new(Id::new(), 3).unwrap();
    let vector = ValueType::shaped(
        domain,
        unit,
        ValueShape::new([2]).unwrap(),
        ValueFrame::SpatialCartesian,
    )
    .unwrap();
    for ty in [
        ValueType::scalar(domain, unit).unwrap().array(3).unwrap(),
        vector.clone(),
        vector.array(3).unwrap(),
        ValueType::shaped(
            domain,
            unit,
            ValueShape::new([2, 3]).unwrap(),
            ValueFrame::SpatialCartesian,
        )
        .unwrap(),
        ValueType::coordinates(left.dual(), domain, unit).unwrap(),
        ValueType::linear_map(left, right, domain, unit).unwrap(),
    ] {
        let field = SymbolRef::Field(Id::new());
        let mut dag = ExprDagBuilder::new();
        let z = dag.symbol(field).unwrap();
        let conjugate = dag.unary_math(UnaryMathFunction::Conj, z).unwrap();
        let real = dag.unary_math(UnaryMathFunction::Real, z).unwrap();
        let imag = dag.unary_math(UnaryMathFunction::Imag, z).unwrap();
        let rebuilt = dag.complex(real, imag).unwrap();
        let typed = TypedResidual::infer(
            dag.finish([conjugate, real, imag, rebuilt]).unwrap(),
            None,
            |_| None,
            RootContract::ComponentwiseResidual,
            |_| Ok::<_, ()>(ExpressionType::<()>::new(ty.clone(), None)),
        )
        .unwrap();
        assert_eq!(typed.node_type(conjugate).unwrap().value_type, ty);
        assert_eq!(typed.node_type(rebuilt).unwrap().value_type, ty);
        let flat = |coordinate: &ScalarSymbolCoordinate| {
            assert_eq!(coordinate.symbol(), field);
            coordinate
                .component_index()
                .iter()
                .zip(ty.shape().extents())
                .fold(0usize, |n, (i, size)| n * size.get() as usize + *i as usize)
        };
        let point = |i: usize| [2. + 3. * i as f64, -1. - 2. * i as f64];
        let direction = |i: usize| [1. - i as f64, 3. + i as f64];
        let count = ty.shape().component_count().unwrap();
        let expected = |input: &dyn Fn(usize) -> [f64; 2]| {
            let mut values = Vec::new();
            for i in 0..count {
                let [r, s] = input(i);
                values.extend([r, -s]);
            }
            values.extend((0..count).map(|i| input(i)[0]));
            values.extend((0..count).map(|i| input(i)[1]));
            for i in 0..count {
                values.extend(input(i));
            }
            values
        };
        let lowered = ComponentScalarization::lower(&typed).unwrap();
        assert_eq!(
            lowered
                .evaluate(|c| Some(point(flat(c))[usize::from(c.is_imaginary())]))
                .unwrap(),
            expected(&point)
        );
        let action = lowered
            .linearize(|c| {
                Some((
                    point(flat(c))[usize::from(c.is_imaginary())],
                    DifferentiationRole::Unknown,
                ))
            })
            .unwrap();
        let tangent = action
            .unknown_coordinates()
            .iter()
            .map(|c| direction(flat(c))[usize::from(c.is_imaginary())])
            .collect::<Vec<_>>();
        let mut product = vec![0.; 6 * count];
        action
            .jvp(RelationTangent::Unknown(&tangent), &mut product)
            .unwrap();
        assert_eq!(product, expected(&direction));
        let weight = |i: usize| (i + 1) as f64;
        let weights = (0..6 * count).map(weight).collect::<Vec<_>>();
        let mut adjoint = vec![0.; tangent.len()];
        action
            .vjp(&weights, RelationCotangent::Unknown(&mut adjoint))
            .unwrap();
        for (c, actual) in action.unknown_coordinates().iter().zip(adjoint) {
            let i = flat(c);
            let expected = if c.is_imaginary() {
                -weight(2 * i + 1) + weight(3 * count + i) + weight(4 * count + 2 * i + 1)
            } else {
                weight(2 * i) + weight(2 * count + i) + weight(4 * count + 2 * i)
            };
            assert_eq!(actual, expected);
        }
    }
}
