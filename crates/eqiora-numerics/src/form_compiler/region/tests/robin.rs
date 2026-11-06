//! Pure compiler and facet-action checks; no global Robin solve is performed.
use super::*;
use crate::form_compiler::region::coordinates::Real;
use eqiora_meshing::{EntityIncidence, MeshEntity, OrientationCode, QuadratureRule};

fn source(alpha: &str, flux: &str) -> String {
    format!(
        "model Robin() {{
      domain body=box(0,2,0,2);
      domain wall=boundary(body,axis=0,side=upper);
      parameter k:W/(m*K)=3;
      parameter h:W/(m^2*K)=2;
      parameter ambient:K=5;
      variable temperature:K on body in smooth;
      law heat on body {{flux -k*grad(temperature);source 0[W/m^3];}}
      relation transfer on wall {{{flux}+({alpha})*trace(temperature)=h*ambient;}}
    }}"
    )
}

fn fixture(
    source: &str,
) -> (
    KernelProgram,
    CompiledRegionForm<f64>,
    BTreeMap<String, RawId>,
) {
    let (transaction, model, symbols) =
        compile("robin.eqi", source).unwrap().remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let ids = ["body", "wall", "heat", "transfer", "temperature", "h"]
        .into_iter()
        .map(|name| (name.to_owned(), symbols.get(name).unwrap()))
        .collect::<BTreeMap<_, _>>();
    let form = CompiledRegionForm::derive(&program, ids["body"], 2).unwrap();
    (program, form, ids)
}

fn law(
    program: &KernelProgram,
    form: &CompiledRegionForm<f64>,
    ids: &BTreeMap<String, RawId>,
) -> Result<RegionBoundaryLaw<f64>, Diagnostic> {
    Ok(form
        .boundary_laws(program, ids["wall"], ids["transfer"], None)?
        .remove(0))
}

#[test]
fn robin_compiler_keeps_known_matrix_and_load_and_rejects_invalid_coefficients() {
    for alpha in ["h", "0[W/(m^2*K)]"] {
        let (program, form, ids) = fixture(&source(alpha, "normal(k*grad(temperature))"));
        let law = law(&program, &form, &ids).unwrap();
        assert_eq!(
            law.quantity,
            crate::canonical_boundary::PhysicalBoundaryQuantity::Flux
        );
        assert_eq!(law.trace_field, None);
        assert_eq!(law.evaluate(&[2., 1.], &[1., 0.]).unwrap(), vec![10.]);
        assert_eq!(
            law.robin_coefficient()
                .unwrap()
                .evaluate(&[2., 1.])
                .unwrap(),
            if alpha == "h" { 2. } else { 0. }
        );
    }
    for (alpha, flux) in [
        ("-h", "normal(k*grad(temperature))"),
        ("h*coordinate(1)/1[m]", "normal(k*grad(temperature))"),
        ("h*trace(temperature)/1[K]", "normal(k*grad(temperature))"),
        ("h", "normal(2*k*grad(temperature))"),
        ("h", "normal(-k*grad(temperature))"),
    ] {
        let (program, form, ids) = fixture(&source(alpha, flux));
        assert!(law(&program, &form, &ids).is_err(), "{alpha}, {flux}");
    }
    assert!(
        compile(
            "wrong-unit.eqi",
            &source("1[W/(m*K)]", "normal(k*grad(temperature))")
        )
        .is_err()
    );
}

#[test]
fn robin_input_partial_and_rebinding_keep_alpha_and_beta() {
    let (program, form, ids) = fixture(&source("h", "normal(k*grad(temperature))"));
    let primal = law(&program, &form, &ids).unwrap();
    let coordinate = crate::parameter_point::coordinates(&program, &[ids["h"].downcast().unwrap()])
        .unwrap()
        .remove(0);
    let mut partial = primal.clone();
    partial.input_partial(&coordinate, 2).unwrap();
    assert_eq!(
        partial
            .robin_coefficient()
            .unwrap()
            .evaluate(&[2., 1.])
            .unwrap(),
        1.
    );
    assert_eq!(partial.evaluate(&[2., 1.], &[1., 0.]).unwrap(), vec![5.]);
    for h in [0., 4.] {
        let point = crate::parameter_point::ParameterPoint::bind(
            &program,
            &[ids["h"].downcast().unwrap()],
            &[h],
        )
        .unwrap();
        let mut rebound = primal.clone();
        rebound
            .bind_inputs(crate::form_compiler::spatial::InputPoint::Parameters(
                &point,
            ))
            .unwrap();
        assert_eq!(
            rebound
                .robin_coefficient()
                .unwrap()
                .evaluate(&[2., 1.])
                .unwrap(),
            h
        );
        assert_eq!(
            rebound.evaluate(&[2., 1.], &[1., 0.]).unwrap(),
            vec![5. * h]
        );
        let mut rebound_partial = partial.clone();
        rebound_partial
            .bind_inputs(crate::form_compiler::spatial::InputPoint::Parameters(
                &point,
            ))
            .unwrap();
        assert_eq!(
            rebound_partial
                .robin_coefficient()
                .unwrap()
                .evaluate(&[2., 1.])
                .unwrap(),
            1.
        );
        assert_eq!(
            rebound_partial.evaluate(&[2., 1.], &[1., 0.]).unwrap(),
            vec![5.]
        );
    }
    let point = crate::parameter_point::ParameterPoint::bind(
        &program,
        &[ids["h"].downcast().unwrap()],
        &[-1.],
    )
    .unwrap();
    let mut invalid = primal;
    assert!(
        invalid
            .bind_inputs(crate::form_compiler::spatial::InputPoint::Parameters(
                &point
            ))
            .is_err()
    );
}

#[test]
fn robin_facet_matrix_survives_projection_residual_jacobian_and_pullback() {
    let (program, form, ids) = fixture(&source("h", "normal(k*grad(temperature))"));
    let law = law(&program, &form, &ids).unwrap();
    for order in [1, 2] {
        let space = Space::continuous_lagrange(NonZeroU16::new(order).unwrap());
        let (fields, rows) = form.si_bindings(|_| Ok(space)).unwrap();
        let reference = ReferenceCell::hypercube(2).unwrap();
        let bound = form.bind(reference, &fields, &rows, None).unwrap();
        let topology = eqiora_meshing::ReferenceTopology::new(reference).unwrap();
        let ordinal = (0..4)
            .find(|&i| {
                topology
                    .entity(1, i)
                    .unwrap()
                    .vertex_ordinals()
                    .iter()
                    .all(|&v| (v & 1) == 1)
            })
            .unwrap();
        let vertices = topology.entity(1, ordinal).unwrap().vertex_ordinals();
        let parent =
            AffineGeometryMap::new(reference, 2, vec![1., 1.], vec![1., 0., 0., 1.]).unwrap();
        let facet = AffineGeometryMap::new(
            ReferenceCell::hypercube(1).unwrap(),
            2,
            vec![2., 1.],
            vec![0., 1.],
        )
        .unwrap();
        let incidence = EntityIncidence {
            entity: MeshEntity::new(2, 0),
            local_ordinal: ordinal,
            orientation: OrientationCode::new(0),
        };
        let degree = law.facet_datum_degree(&bound, ids["temperature"]);
        let rule = bound
            .natural_facet_quadrature(ids["temperature"], degree)
            .unwrap();
        let prepared = bound
            .prepare_affine_facet(
                ids["temperature"],
                &parent,
                (&facet, incidence, vertices),
                &rule,
                |p, n| law.evaluate(p, n),
                (degree, law.robin_coefficient()),
            )
            .unwrap()
            .project::<Real>()
            .unwrap();
        let count = prepared.affine().rows();
        let point = vec![3.; count];
        let action = prepared.linearize(&point).unwrap();
        let pulled = prepared
            .action(
                &point,
                Some(FieldPullback::Uniform {
                    current: Some(2.),
                    rate: None,
                }),
            )
            .unwrap();
        let mut total_matrix = 0.;
        let mut total_rhs = 0.;
        let mut total_residual = 0.;
        for i in 0..count {
            total_rhs += prepared.affine().rhs()[i];
            total_residual += action.residual[i];
            for j in 0..count {
                let matrix = prepared.affine().matrix()[i * count + j];
                assert!((action.jacobian[i * count + j] - matrix).abs() < 1e-13);
                assert!((pulled.jacobian[i * count + j] - 2. * matrix).abs() < 1e-13);
                total_matrix += matrix;
            }
        }
        // Partition of unity gives integral alpha=2*2 and integral beta=10*2.
        assert!((total_matrix - 4.).abs() < 1e-12);
        assert!((total_rhs - 20.).abs() < 1e-12);
        assert!((total_residual + 8.).abs() < 1e-12);
        if order == 2 {
            assert!(
                bound
                    .prepare_affine_facet(
                        ids["temperature"],
                        &parent,
                        (&facet, incidence, vertices),
                        &QuadratureRule::tensor_product_gauss_legendre(1, 2).unwrap(),
                        |p, n| law.evaluate(p, n),
                        degree,
                        law.robin_coefficient()
                    )
                    .is_err()
            );
        }
    }
}

#[test]
fn robin_authored_proofs_keep_trial_trace_and_only_essential_zero_test_trace() {
    let source = "model RobinProof() {
      domain body=box(0,2,0,2);
      domain left=boundary(body,axis=0,side=lower);
      domain wall=boundary(body,axis=0,side=upper);
      domain bottom=boundary(body,axis=1,side=lower);
      domain top=boundary(body,axis=1,side=upper);
      parameter k:W/(m*K)=3; parameter h:W/(m^2*K)=2; parameter ambient:K=5;
      variable temperature:K on body in smooth;
      relation heat on body {-div(k*grad(temperature))=0[W/m^3];}
      relation fixed on left {trace(temperature)=1[K];}
      relation transfer on wall {normal(k*grad(temperature))+h*trace(temperature)=h*ambient;}
      relation insulated_bottom on bottom {normal(k*grad(temperature))=0[W/m^2];}
      relation insulated_top on top {normal(k*grad(temperature))=0[W/m^2];}
      form weak for heat {
        test eta:1 for temperature zero_on left;
        integrate(body,dot(grad(eta),k*grad(temperature)))
          + integrate(wall,trace(eta)*h*trace(temperature))
          = integrate(wall,trace(eta)*h*ambient);
      }
    }";
    for (before, after, accepted) in [
        (
            "= integrate(wall,trace(eta)*h*ambient)",
            "= integrate(wall,trace(eta)*h*ambient)",
            true,
        ),
        (
            "+ integrate(wall,trace(eta)*h*trace(temperature))",
            "",
            false,
        ),
        (
            "+ integrate(wall,trace(eta)*h*trace(temperature))",
            "- integrate(wall,trace(eta)*h*trace(temperature))",
            false,
        ),
        ("zero_on left", "zero_on left,wall", false),
    ] {
        let candidate = source.replace(before, after);
        let compiled = compile("robin-proof.eqi", &candidate).unwrap().remove(0);
        let projection = compiled
            .authored_formulations()
            .next()
            .unwrap()
            .projection()
            .clone();
        let (transaction, model, symbols) = compiled.into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        let derived = crate::form_compiler::scalar::derive_candidate_with_dimension(
            &program,
            symbols.get("body").unwrap(),
            2,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            crate::form_compiler::scalar::authored::admit(&projection, &program, &derived).is_ok(),
            accepted
        );
        assert_eq!(
            crate::form_compiler::mixed::check(&program, &projection).is_ok(),
            accepted
        );
    }
}
