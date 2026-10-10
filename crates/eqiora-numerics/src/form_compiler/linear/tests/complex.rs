use super::*;
use num_complex::Complex64 as C;

const HELMHOLTZ: &str = "model Wave() {
 domain body=box(0,6);
 domain left=boundary(body,axis=0,side=lower);
 domain right=boundary(body,axis=0,side=upper);
 parameter a:complex<m^2>=math.complex(-6[m^2],6[m^2]);
 parameter q:complex<1>=math.complex(3,-1);
 parameter f:complex<1>=math.complex(1,3);
 variable u:complex<1> on body in smooth;
 relation balance on body { -div(a*grad(u))+q*u=f; }
 relation fixed on left { trace(u)=math.complex(1,2); }
 relation flux on right { normal(a*grad(u))=math.complex(2[m],-4[m]); }
}";
fn complex_form(source: &str) -> Result<CompiledLinearBlockForm<C>, Diagnostic> {
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
    CompiledLinearBlockForm::<C>::derive(&program, domain, 1, &BTreeSet::new())
}
fn close(actual: C, expected: C) {
    assert!(
        (actual - expected).norm() < 1e-12,
        "{actual:?} != {expected:?}"
    );
}

#[test]
fn common_linear_admission_retains_nonhermitian_phase_and_complex_boundary_data() {
    let form = complex_form(HELMHOLTZ).unwrap();
    let geometry =
        AffineGeometryMap::new(ReferenceCell::hypercube(1).unwrap(), 1, vec![3.], vec![3.])
            .unwrap();
    let rule = QuadratureRule::gauss_legendre(2).unwrap();
    let local = form
        .volume()
        .unwrap()
        .prepare_cell(&geometry, &rule)
        .unwrap()
        .evaluate(&BTreeMap::new())
        .unwrap();
    // On [0,6], a/6 times the endpoint difference matrix plus q[[2,1],[1,2]].
    for (actual, expected) in local.matrix().iter().zip([
        C::new(5., -1.),
        C::new(4., -2.),
        C::new(4., -2.),
        C::new(5., -1.),
    ]) {
        close(*actual, expected);
    }
    for value in local.rhs() {
        close(*value, C::new(3., 9.));
    }
    assert_ne!(local.matrix()[1], local.matrix()[2].conj());
    let field = form.fields()[0].0;
    let laws = &form.boundary_laws()[&field];
    assert_eq!(laws.len(), 2);
    for law in laws.values() {
        match law.quantity {
            crate::canonical_boundary::PhysicalBoundaryQuantity::Trace => {
                assert_eq!(law.evaluate(&[0.], &[]).unwrap(), [C::new(1., 2.)])
            }
            crate::canonical_boundary::PhysicalBoundaryQuantity::Flux => {
                assert_eq!(law.evaluate(&[6.], &[1.]).unwrap(), [C::new(2., -4.)])
            }
        }
    }
    let reversed = HELMHOLTZ.replace("-div(a*grad(u))+q*u=f", "div(a*grad(u))-q*u=-f");
    let other = complex_form(&reversed)
        .unwrap()
        .volume()
        .unwrap()
        .prepare_cell(&geometry, &rule)
        .unwrap()
        .evaluate(&BTreeMap::new())
        .unwrap();
    assert_eq!(local, other);
    assert!(derive(HELMHOLTZ).is_err());
    let wrong = HELMHOLTZ.replace("normal(a*grad(u))", "normal(math.conj(a)*grad(u))");
    assert!(complex_form(&wrong).is_err());
}

#[test]
fn prepared_complex_storage_retains_initial_and_fresh_history_channels() {
    let source = HELMHOLTZ
        .replace("variable u", "state u")
        .replace(
            "-div(a*grad(u))+q*u=f",
            "2[s]*derivative(u)-div(a*grad(u))+q*u=f",
        )
        .replace(
            "relation balance",
            "initial { u=math.complex(1,2); } relation balance",
        );
    let form = complex_form(&source).unwrap();
    let field = form.fields()[0].0;
    assert_eq!(
        form.initial_values_at(&[0.0]).unwrap()[&field],
        C::new(1., 2.)
    );
    assert!(form.volume().is_err());
    let second = eqiora_core::DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap();
    let form = form
        .bind_backward_euler(DynQuantity::new(2., second))
        .unwrap();
    let geometry =
        AffineGeometryMap::new(ReferenceCell::hypercube(1).unwrap(), 1, vec![3.], vec![3.])
            .unwrap();
    let cell = form
        .volume()
        .unwrap()
        .prepare_cell(&geometry, &QuadratureRule::gauss_legendre(2).unwrap())
        .unwrap();
    // Capacity/step=1 gives M=[[2,1],[1,2]]. History is physical and supplied afresh.
    let uniform = cell
        .evaluate(&BTreeMap::from([(field, vec![C::new(1., 2.); 2])]))
        .unwrap();
    close(uniform.matrix()[0], C::new(7., -1.));
    close(uniform.matrix()[1], C::new(5., -2.));
    for value in uniform.rhs() {
        close(*value, C::new(6., 15.));
    }
    let next = cell
        .evaluate(&BTreeMap::from([(
            field,
            vec![C::new(1., 2.), C::new(-1., 3.)],
        )]))
        .unwrap();
    close(next.rhs()[0], C::new(4., 16.));
    close(next.rhs()[1], C::new(2., 17.));
    assert!(cell.evaluate(&BTreeMap::new()).is_err());
    assert!(
        cell.evaluate(&BTreeMap::from([(field, vec![C::new(1., f64::NAN); 2])]))
            .is_err()
    );
}
