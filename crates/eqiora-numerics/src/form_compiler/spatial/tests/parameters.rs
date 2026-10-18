//! Complete Parameter points feed the same volume and trace component actions.
use super::*;
use crate::spatial_expression::Coefficient;
use eqiora_core::ScalarDomain;
use eqiora_realization::Space;
use num_complex::Complex64;

fn check<S: Coefficient>(complex: bool) {
    let scalar = if complex { "complex<1>" } else { "1" };
    let initial = if complex {
        "[math.complex(2,3),math.complex(-3,4)]"
    } else {
        "[2,-3]"
    };
    let source = format!(
        "model Channels() {{
        domain body=box(0,2);
        domain left=boundary(body,axis=0,side=lower);
        domain right=boundary(body,axis=0,side=upper);
        parameter unit:1/m^2=1;
        parameter gains:array<{scalar},2>={initial};
        variable u:array<{scalar},2> on body in h1;
        relation balance on body {{
            -div(grad(u))+unit*[gains[0]*u[0]+gains[1]*u[1],
                2*gains[1]*u[0]+gains[0]*u[1]]=unit*[11,13];
        }}
        relation left_value on left {{ trace(u)=gains; }}
        relation right_value on right {{ trace(u)=gains; }}
    }}"
    );
    let program = program(&source);
    let mut domain = None;
    let mut gains = None;
    let mut unit = None;
    for node in program.nodes() {
        match node {
            KernelNode::Domain(value)
                if matches!(value.kind(), DomainKind::CartesianBox { .. }) =>
            {
                domain = Some(value.id().erase())
            }
            KernelNode::Parameter(value) if value.value_type().array_rank() == 1 => {
                gains = Some(value.id())
            }
            KernelNode::Parameter(value) => unit = Some(value.id()),
            _ => {}
        }
    }
    let (gains, unit) = (gains.unwrap(), unit.unwrap());
    let form = CompiledSpatialForm::<S>::derive(&program, domain.unwrap(), 1, &Default::default())
        .unwrap();
    // A new point resets the unselected unit to the immutable Model default.
    let changed = form
        .bind_parameter_point(&ParameterPoint::bind(&program, &[unit], &[3.]).unwrap())
        .unwrap();
    let values = if complex {
        vec![5., -2., -7., 3.]
    } else {
        vec![5., -7.]
    };
    let point = ParameterPoint::bind(&program, &[gains], &values).unwrap();
    let bound = changed.bind_parameter_point(&point).unwrap();
    assert_eq!(
        point.get(gains).unwrap().value_type(),
        program.typed_value(gains.erase()).unwrap().value_type()
    );
    assert_eq!(
        point.get(unit).unwrap(),
        program.typed_value(unit.erase()).unwrap()
    );
    let lift = |real, imaginary| {
        <S as From<f64>>::from(real)
            + S::imaginary_unit().unwrap_or_else(|| <S as From<f64>>::from(0.))
                * <S as From<f64>>::from(imaginary)
    };
    let p = [lift(5., -2.), lift(-7., 3.)];
    let close = |actual: S, expected: S| {
        assert!(
            (actual - expected).abs() < 1e-12,
            "{actual:?} != {expected:?}"
        )
    };
    let space = Space::continuous_lagrange(std::num::NonZeroU16::MIN);
    let volume = bound
        .bind_space(ReferenceCell::hypercube(1).unwrap(), space)
        .unwrap();
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(1, 2).unwrap();
    let local = volume
        .prepare_cell(&geometry(), &quadrature)
        .unwrap()
        .evaluate(&BTreeMap::new())
        .unwrap();
    assert_eq!(local.rhs().len(), 4);
    // On [0,2], the P1 mass matrix has diagonal 2/3 and off-diagonal 1/3;
    // the stiffness entries are +/-1/2. Components are independent of nodes.
    for i in 0..4 {
        close(local.rhs()[i], lift(if i % 2 == 0 { 11. } else { 13. }, 0.));
        for j in 0..4 {
            let reaction = match (i % 2, j % 2) {
                (0, 0) | (1, 1) => p[0],
                (0, 1) => p[1],
                (1, 0) => p[1] * 2.,
                _ => unreachable!(),
            };
            let mass = if i / 2 == j / 2 { 2. / 3. } else { 1. / 3. };
            let stiffness = if i % 2 != j % 2 {
                0.
            } else if i / 2 == j / 2 {
                0.5
            } else {
                -0.5
            };
            close(
                local.matrix()[4 * i + j],
                reaction * mass + lift(stiffness, 0.),
            );
        }
    }
    for laws in bound.boundary_laws().values() {
        for law in laws.values() {
            let value = law.evaluate(&[0.], &[-1.]).unwrap();
            assert_eq!(value.len(), 2);
            for (actual, expected) in value.into_iter().zip(p) {
                close(actual, expected);
            }
        }
    }
    assert_eq!(
        program
            .typed_value(gains.erase())
            .unwrap()
            .value_type()
            .scalar_domain(),
        if complex {
            ScalarDomain::Complex
        } else {
            ScalarDomain::Real
        }
    );
    assert_ne!(
        point.get(gains).unwrap(),
        program.typed_value(gains.erase()).unwrap()
    );
}

#[test]
fn typed_parameter_points_rebind_each_real_channel() {
    check::<f64>(false);
}

#[test]
fn typed_parameter_points_rebind_each_complex_channel_and_phase() {
    check::<Complex64>(true);
}

#[test]
fn guarded_unitful_map_coefficients_reach_the_shared_volume_action() {
    for (inverse, scale, value) in [
        (false, "determinant(a)/1[m^2]", 5.),
        (true, "matrix_trace(inverse(a))*1[m]", 6. / 5.),
    ] {
        for varying in [false, true] {
            let coefficient = if varying {
                scale.replace("(a)", "(b)")
            } else {
                scale.into()
            };
            let profile = if varying {
                let expression = if inverse {
                    "a/(1+coordinate(0)/1[m])"
                } else {
                    "compose(identity(S)+(coordinate(0)/1[m])*linear_map(S,S,[[1,0],[0,0]]),a)"
                };
                format!(
                    "variable b:map<m,S,S> on body in smooth; relation profile on body {{b={expression};}}"
                )
            } else {
                String::new()
            };
            let source = format!(
                "space S=orthonormal(x,y);
                model GuardedMap() {{
                    domain body=box(0,2);
                    domain left=boundary(body,axis=0,side=lower);
                    domain right=boundary(body,axis=0,side=upper);
                    parameter a:map<m,S,S>=linear_map(S,S,[[2[m],1[m]],[3[m],4[m]]]);
                    {profile}
                    variable h:1 on body in smooth;
                    relation prescribed on body {{h=(1+coordinate(0)/1[m])^2;}}
                    variable u:1 on body in smooth;
                    relation balance on body {{-div(grad(u)+grad(math.sqrt(h)*({coefficient})))=0;}}
                    relation left_value on left {{trace(u)={scale};}}
                    relation right_value on right {{trace(u)=3*({scale});}}
                }}"
            );
            let form = derive(&source).unwrap();
            let local = form
                .volume()
                .unwrap()
                .prepare_cell(
                    &geometry(),
                    &QuadratureRule::tensor_product_gauss_legendre(1, 2).unwrap(),
                )
                .and_then(|cell| cell.evaluate(&BTreeMap::new()))
                .unwrap();
            // Q1 gradients on [0,2] are -1/2,+1/2. With t=scale(a),
            // f=t*(1+x) gives load [t,-t]; f=t*(1+x)^2 gives [4t,-4t].
            for (actual, expected) in local.matrix().iter().zip([0.5, -0.5, -0.5, 0.5]) {
                close(*actual, expected);
            }
            let load = if varying { 4. * value } else { value };
            for (actual, expected) in local.rhs().iter().zip([load, -load]) {
                close(*actual, expected);
            }
        }
    }
}
