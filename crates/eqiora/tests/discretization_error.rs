//! Manufactured solution and hand-integrated P1 hat, independent of the estimator.
use eqiora::api::ModelDocument;
use eqiora::artifact::{GmshMeshPolicyV1, ModelEnvelope};
use eqiora::compiler::StaticBindingValue;
use eqiora::geometry::{CanonicalGeometryV1, NamedEntitySet, PlanarFace, PlanarRegion};
use eqiora::meshing::simplex_duffy_gauss_legendre;
use eqiora::solver::{
    LinearSolver, REFERENCE_LINEAR_SOLVER, REFERENCE_SOLVER_PROVIDER, ReductionPolicy, SolverPlan,
};
use eqiora::{DimExponents, Id, kinds};
use eqiora_numerics::{
    AuthenticatedCommonMesh, CommonLinearRequest, CommonResult, CommonSolvePolicy,
    CommonSpatialPolicy, ResolvedCommonPlan,
};
use std::num::NonZeroUsize;

fn solve(boundary: &str, permute: bool) -> (CommonResult, Id<kinds::Field>) {
    let geometry = CanonicalGeometryV1::from_region(
        &PlanarRegion::new(
            vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
            vec![PlanarFace::new(vec![0, 1, 2, 3], vec![])],
            vec![
                NamedEntitySet::new("body", 2, vec![0]),
                NamedEntitySet::new("outer", 1, vec![0, 1, 2, 3]),
            ],
            1e-12,
        )
        .unwrap(),
    )
    .unwrap();
    let body = geometry.entity_set("body").unwrap();
    let bindings = [
        (
            "body",
            StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: body,
                parent: None,
            },
        ),
        (
            "outer",
            StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("outer").unwrap(),
                parent: Some(body),
            },
        ),
    ];
    let source = format!("public component Diffusion(support body:volume(ambient_dimension=2), support outer:boundary(parent=body)) {{
        variable u:K on body in h1;
        variable x:1 on body;
        variable y:1 on body;
        relation x_value on body {{ x=coordinate(0)/1[m]; }}
        relation y_value on body {{ y=coordinate(1)/1[m]; }}
        relation balance on body {{ -div(grad(u))=2[K/m^2]*(x*(1-x)+y*(1-y)); }}
        relation fixed on outer {{ trace(u)={boundary}; }}
    }}");
    let document =
        ModelDocument::compile_selected("error.eqi", &source, "Diffusion", &bindings).unwrap();
    let field = document.aliases()["definition.u"].downcast().unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let mut msh = String::from(
        "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n1 5 1 5\n2 1 0 5\n1\n2\n3\n4\n5\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n0.5 0.5 0\n$EndNodes\n$Elements\n1 4 1 4\n2 1 2 4\n",
    );
    for (index, mut cell) in [[1, 2, 5], [2, 3, 5], [3, 4, 5], [4, 1, 5]]
        .into_iter()
        .enumerate()
    {
        if permute {
            cell.rotate_left(1);
        }
        msh += &format!("{} {} {} {}\n", index + 1, cell[0], cell[1], cell[2]);
    }
    msh += "$EndElements\n";
    let mesh = AuthenticatedCommonMesh::gmsh_4152(
        geometry,
        GmshMeshPolicyV1::explicit(1e-12, 0.01, 8, 5.).unwrap(),
        msh.into_bytes(),
    )
    .unwrap();
    let plan = ResolvedCommonPlan::resolve(
        &model,
        mesh,
        CommonSpatialPolicy::P1,
        CommonSolvePolicy::Linear(
            CommonLinearRequest::exact(
                SolverPlan::new(
                    LinearSolver::BiConjugateGradientStabilized,
                    1e-12,
                    1e-14,
                    NonZeroUsize::new(100).unwrap(),
                )
                .unwrap()
                .with_reduction(ReductionPolicy::Reproducible),
                REFERENCE_SOLVER_PROVIDER,
            )
            .unwrap(),
        ),
        None,
        None,
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap();
    (
        plan.as_spatial()
            .unwrap()
            .run_result(None, &REFERENCE_LINEAR_SOLVER)
            .unwrap(),
        field,
    )
}

#[test]
fn manufactured_p1_error_remains_when_the_discrete_equations_are_solved() {
    let temperature = DimExponents::from_integers([0, 0, 0, 0, 1, 0, 0]).unwrap();
    let rule = simplex_duffy_gauss_legendre(2, 3).unwrap();
    for permute in [false, true] {
        let (result, field) = solve("0[K]", permute);
        let bytes = result.to_bytes().unwrap();
        let replay = CommonResult::from_bytes(&bytes, result.plan()).unwrap();
        for candidate in [&result, &replay] {
            assert!(candidate.solve_true_residual_norm(None, None).unwrap() < 1e-12);
            // u=x(1-x)y(1-y), ||grad u||^2=1/45. On the four-triangle
            // star, the center hat has stiffness 4 and load 4/15, so
            // u_h(center)=1/15 and ||grad(u-u_h)||^2=1/225.
            let (_, values, _) = candidate.field_block(0, 0).unwrap();
            assert!((values[4] - 1.0 / 15.0).abs() < 1e-13);
            assert!((1.0 / 45.0 - 4.0 * values[4].powi(2) - 1.0 / 225.0).abs() < 1e-13);
            // Each triangle has h^2=1 and integral f^2=11/90.
            // Its two interior edges contribute 4*(1/15)^2 in total.
            // Therefore eta_K^2=11/90+4/225=7/50.
            let estimates = candidate
                .discretization_error_indicators(field, &rule)
                .unwrap();
            assert_eq!(estimates.len(), 4);
            for estimate in estimates {
                assert_eq!(estimate.dim(), temperature);
                assert!((estimate.value().powi(2) - 7.0 / 50.0).abs() < 1e-13);
            }
            assert_eq!(candidate.to_bytes().unwrap(), bytes);
        }
    }
}

#[test]
fn nonhomogeneous_boundary_is_not_silently_estimated_as_homogeneous() {
    let (result, field) = solve("1[K]", false);
    let error = result
        .discretization_error_indicators(field, &simplex_duffy_gauss_legendre(2, 3).unwrap())
        .unwrap_err();
    assert!(error.message().contains("homogeneous essential"));
}

#[cfg(feature = "hdf5")]
#[test]
fn common_static_export_preserves_temperature_and_has_no_fabricated_time() {
    use eqiora::io::hdf5::{Hdf5DatasetRequest, Hdf5FileImage, Hdf5ResolvedValues, Hdf5ScalarType};
    for permute in [false, true] {
        let (result, field) = solve("0[K]", permute);
        let (xdmf, hdf5) = eqiora::api::export_common_result_xdmf_hdf5(
            &result,
            "result.h5",
            eqiora::api::XdmfHdf5ExportLimits::default(),
        )
        .unwrap();
        let mut reader = quick_xml::Reader::from_reader(xdmf.metadata_bytes());
        let mut in_field = false;
        let mut field_path = None;
        let mut field_information = std::collections::BTreeMap::new();
        loop {
            use quick_xml::events::Event;
            match reader.read_event().unwrap() {
                Event::Start(element) if element.name().as_ref() == "Attribute" => {
                    let attributes = element
                        .attributes()
                        .map(|attribute| {
                            let attribute = attribute.unwrap();
                            (
                                attribute.key.as_ref().to_owned(),
                                attribute
                                    .normalized_value(quick_xml::XmlVersion::Explicit1_0)
                                    .unwrap()
                                    .into_owned(),
                            )
                        })
                        .collect::<std::collections::BTreeMap<_, _>>();
                    assert_eq!(attributes["Name"], field.ulid().to_string());
                    assert_eq!(attributes["Center"], "Node");
                    assert_eq!(attributes["AttributeType"], "Scalar");
                    in_field = true;
                }
                Event::End(element) if element.name().as_ref() == "Attribute" => in_field = false,
                Event::Empty(element) if element.name().as_ref() == "Information" && in_field => {
                    let attributes = element
                        .attributes()
                        .map(|attribute| {
                            let attribute = attribute.unwrap();
                            (
                                attribute.key.as_ref().to_owned(),
                                attribute
                                    .normalized_value(quick_xml::XmlVersion::Explicit1_0)
                                    .unwrap()
                                    .into_owned(),
                            )
                        })
                        .collect::<std::collections::BTreeMap<_, _>>();
                    field_information
                        .insert(attributes["Name"].clone(), attributes["Value"].clone());
                }
                Event::Start(element) | Event::Empty(element)
                    if element.name().as_ref() == "Time" =>
                {
                    panic!("static Result has no model time")
                }
                Event::Text(text) if in_field => {
                    let text = text.as_ref().trim();
                    if !text.is_empty() {
                        field_path = Some(
                            text.strip_prefix("result.h5:")
                                .expect("declared sidecar")
                                .to_owned(),
                        );
                    }
                }
                Event::Eof => break,
                _ => {}
            }
        }
        assert_eq!(field_information["eqiora.units"], "coherent-SI");
        assert_eq!(field_information["eqiora.dimension"], "Θ");
        assert_eq!(field_information["eqiora.value-frame"], "invariant");
        let request =
            Hdf5DatasetRequest::new(field_path.unwrap(), Hdf5ScalarType::F64, vec![5]).unwrap();
        let arrays = eqiora::io::hdf5::resolve_hdf5_file_image(
            Hdf5FileImage::new(hdf5.bytes()),
            &[request],
            Default::default(),
        )
        .unwrap();
        let [Hdf5ResolvedValues::F64(values)] = arrays.values() else {
            panic!("one real temperature array")
        };
        assert_eq!(&values[..4], &[0.; 4]);
        // Independently integrated center equation: 4 u_c = 4/15.
        assert!((values[4] - 1. / 15.).abs() < 1e-13);
    }
}
