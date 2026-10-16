use super::*;
use eqiora_meshing::{MeshQualityGate, SimplicialMesh};

fn equation(diffusion: &str, forcing: &str, extra: &str) -> BoundRegionForm<f64> {
    let source = format!(
        "model Diffusion() {{
        domain body = box(0,1,0,1);
        variable u:K on body in h1;
        relation balance on body {{ -div(({diffusion})*grad(u)) {extra} = {forcing}; }}
    }}"
    );
    let form = derive(&source).unwrap();
    let (fields, rows) = form.si_bindings(|_| Ok(p1())).unwrap();
    form.bind(ReferenceCell::simplex(2).unwrap(), &fields, &rows, None)
        .unwrap()
}

fn square() -> SimplicialMesh {
    SimplicialMesh::new(
        2,
        vec![vec![0., 0.], vec![1., 0.], vec![1., 1.], vec![0., 1.]],
        vec![vec![0, 1, 2], vec![0, 2, 3]],
        MeshQualityGate::new(0.1).unwrap(),
    )
    .unwrap()
}

#[test]
fn residual_indicators_separate_source_and_normal_gradient_jump() {
    let mesh = square();
    let rule = simplex_duffy_gauss_legendre(2, 3).unwrap();
    // h_K^2=2, |K|=1/2, f/a=2: the volume contribution is 4
    // per triangle. Gradients (0,1) and (1,0) have normal jump
    // sqrt(2) on a sqrt(2)-long edge: h_E*integral jump^2=4.
    // Each incident triangle receives half, giving 6 each.
    for diffusion in ["1", "7"] {
        let source = if diffusion == "1" {
            "2[K/m^2]"
        } else {
            "14[K/m^2]"
        };
        let indicators = equation(diffusion, source, "")
            .p1_residual_indicators(&mesh, &[0., 0., 1., 0.], &rule)
            .unwrap();
        for value in indicators {
            close(value, 6.0);
        }
    }
    // A globally affine field has no interior jump, and no boundary-edge
    // term is added by the homogeneous-essential estimator kernel.
    assert_eq!(
        equation("1", "0[K/m^2]", "")
            .p1_residual_indicators(&mesh, &[0., 1., 2., 1.], &rule)
            .unwrap(),
        vec![0.0; 2]
    );
}

#[test]
fn residual_indicators_reject_unsupported_operators_and_inputs() {
    let mesh = square();
    let rule = simplex_duffy_gauss_legendre(2, 2).unwrap();
    for form in [
        equation("1+coordinate(0)/1[m]", "2[K/m^2]", ""),
        equation("1", "2[K/m^2]", "+u/1[m^2]"),
    ] {
        assert!(form.p1_residual_indicators(&mesh, &[0.; 4], &rule).is_err());
    }
    let form = equation("1", "2[K/m^2]", "");
    for values in [
        vec![0.; 3],
        vec![f64::NAN; 4],
        vec![f64::MAX, -f64::MAX, f64::MAX, -f64::MAX],
    ] {
        assert!(form.p1_residual_indicators(&mesh, &values, &rule).is_err());
    }
    assert!(
        form.p1_residual_indicators(
            &mesh,
            &[0.; 4],
            &simplex_duffy_gauss_legendre(3, 2).unwrap()
        )
        .is_err()
    );
}
