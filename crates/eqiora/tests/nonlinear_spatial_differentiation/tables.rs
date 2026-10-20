//! Packaged tables use the same native material and derivative consumers.
use super::*;
use eqiora::package::*;

fn document(right: f64) -> ModelDocument {
    let source = format!(
        r#"
        property contract Law(input value:1):1 {{derivatives first_open_intervals;}}
        property release Release:Law {{
            table {{data samples;axis value:1;value response:1;
                interpolation piecewise_affine;preprocessing identity;missing reject;
                knot_derivative reject;endpoint_derivative reject;}}
            validity value in [-0.5,5];outside reject;branch single;
            citation manufactured;license usage_license;
        }}
        component Equation(support body:volume(ambient_dimension=1),
            support left:boundary(parent=body),support right:boundary(parent=body),
            parameter p:1,parameter r:1,property response:Law) {{
            variable u:1 on body in smooth;
            relation balance on body {{-div(response(value=p*u)*grad(u))=0;}}
            relation lower on left {{trace(u)=1;}}
            relation upper on right {{trace(u)=r;}}
        }}
        public model Main() {{
            domain body=box(0,1);
            domain left=boundary(body,axis=0,side=lower);
            domain right=boundary(body,axis=0,side=upper);
            parameter p:1=1;parameter r:1={right};
            instance physics:Equation(body=body,left=left,right=right,p=p,r=r,response=Release);
        }}"#
    );
    let samples = eqiora_schema::resolved_array::ResolvedF64Array::new(
        vec![3, 2],
        vec![-0.5, 0.5, 3.5, 4.5, 5., 7.5],
    )
    .unwrap();
    let assets = [
        (
            "src/main.eqi",
            BundleRoleV1::ModelSource,
            source.into_bytes(),
        ),
        (
            "data/samples.json",
            BundleRoleV1::ResolvedArray,
            samples.canonical_json().unwrap(),
        ),
        (
            "docs/manufactured.md",
            BundleRoleV1::Documentation,
            b"Synthetic k(s)=1+s below 3.5 and 2s-2.5 above 3.5, with values .5,4.5,7.5.".to_vec(),
        ),
        (
            "docs/usage_license.md",
            BundleRoleV1::Documentation,
            b"CC0-1.0 synthetic definition.".to_vec(),
        ),
    ];
    let manifest = PackageManifestV1::new(
        "main",
        QualifiedName::parse("org.example.TableMaterial").unwrap(),
        ExactVersion::parse("1.0.0").unwrap(),
        vec![],
        assets
            .iter()
            .map(|(path, role, _)| {
                BundleEntryV1::new(NormalizedRelativePath::parse(*path).unwrap(), *role)
            })
            .collect(),
    )
    .unwrap();
    let files = assets
        .into_iter()
        .map(|(path, role, bytes)| {
            SourceFileV1::new(NormalizedRelativePath::parse(path).unwrap(), role, bytes)
        })
        .collect();
    let release =
        prepare_package_release_v1(PackageSourcesV1::new(manifest, files).unwrap(), &[]).unwrap();
    let resolution = ResolutionRecordV1::from_exact_releases(&release, &[]).unwrap();
    let mut store = InMemoryPackageStore::default();
    store.insert(&release).unwrap();
    PackagedModelDocument::compile_locked(&store, &resolution, "Main")
        .unwrap()
        .model()
        .clone()
}

fn check_point(program: &DifferentiableProgram, p: f64, r: f64, (u, dp, dr): (f64, f64, f64)) {
    let point = program
        .evaluate(&support::real_point(program, &[p, r]).unwrap())
        .unwrap();
    close(&point.primal().output().fixture_coordinates(), &[1., u, r]);
    for (direction, expected) in [([1., 0.], [0., dp, 0.]), ([0., 1.], [0., dr, 1.])] {
        let product = point
            .jvp(&tangent(program.identity(), &direction).unwrap())
            .unwrap();
        close(&product.derivative().fixture_coordinates(), &expected);
    }
    let product = point
        .vjp(&cotangent(program.identity(), &[2., 3., 5.]).unwrap())
        .unwrap();
    close(
        &product.derivative().fixture_coordinates(),
        &[3. * dp, 3. * dr + 5.],
    );
}

#[test]
fn table_material_primal_products_and_records_use_the_ordinary_owners() {
    let (document, plan, seed) = fixture_from_document(document(2.), 2.);
    let before = seed.to_bytes().unwrap();
    let CommonInitialState::Spatial(state) = &seed else {
        panic!("spatial fixture")
    };
    let result = plan
        .as_spatial()
        .unwrap()
        .run_result(Some(state), &FaerLinearSolver)
        .unwrap();
    assert_eq!(
        eqiora_numerics::CommonResult::from_bytes(&result.to_bytes().unwrap(), &plan).unwrap(),
        result
    );
    let program = program(&document, &plan, &seed);
    for (p, r) in [(0., 2.), (1., 2.), (1., 3.)] {
        check_point(&program, p, r, expected(p, r));
        let point = program
            .evaluate(&support::real_point(&program, &[p, r]).unwrap())
            .unwrap();
        let record = point.record().unwrap();
        assert_eq!(
            RecordedDifferentiableEvaluation::from_bytes(
                &record.to_bytes().unwrap(),
                &plan,
                Some(&seed)
            )
            .unwrap(),
            record
        );
    }
    assert_eq!(seed.to_bytes().unwrap(), before);
}

#[test]
fn table_material_rejects_knot_endpoint_and_outside_derivatives() {
    let (document, plan, seed) = fixture_from_document(document(1.), 1.);
    let before = seed.to_bytes().unwrap();
    let program = program(&document, &plan, &seed);
    for p in [-1., -0.5, 3.5, 5., 6.] {
        assert!(
            program
                .evaluate(&support::real_point(&program, &[p, 1.]).unwrap())
                .is_err()
        );
    }
    let valid = program
        .evaluate(&support::real_point(&program, &[1., 1.]).unwrap())
        .unwrap();
    close(&valid.primal().output().fixture_coordinates(), &[1.; 3]);
    assert_eq!(seed.to_bytes().unwrap(), before);
}

#[test]
fn table_material_uses_each_gauss_points_segment_slope() {
    let (document, plan, seed) = fixture_from_document(document(2.), 2.);
    let program = program(&document, &plan, &seed);
    // Discrete two-point Gauss Q1 oracle, not an exact continuum integral
    // across a knot. At p=r=2 only the last of four samples uses slope 2.
    let a = (1. - 1. / 3_f64.sqrt()) / 2.;
    let b = 1. - a;
    let qa = 2. + a;
    let qb = 0.25 + 2. / 3_f64.sqrt();
    let qc = -(4.5 + 4. * b);
    let u = (-qb + (qb * qb - 4. * qa * qc).sqrt()) / (2. * qa);
    let (p, r) = (2., 2.);
    let samples = [
        p * (b + a * u),
        p * (a + b * u),
        p * (b * u + a * r),
        p * (a * u + b * r),
    ];
    assert!(samples[..3].iter().all(|s| *s < 3.5));
    assert!(samples[3] > 3.5 && samples[3] < 5.);
    // F=KL*(u-1)+KR*(u-r); differentiate the two segment means explicitly.
    let c = b / 2. + a;
    let d = a / 2. + b;
    let kl = 1. + p * (1. + u) / 2.;
    let kr = -0.75 + p * (c * u + d * r);
    let fu = p * (u - 1.) / 2. + kl + p * c * (u - r) + kr;
    let fp = (1. + u) * (u - 1.) / 2. + (c * u + d * r) * (u - r);
    let fr = p * d * (u - r) - kr;
    check_point(&program, p, r, (u, -fp / fu, -fr / fu));
}
