//! Common export is checked against material conservation and primitive HDF5 reads.
#![cfg(feature = "hdf5")]
use eqiora::api::{
    ModelDocument, XdmfHdf5ExportLimits, export_common_result_xdmf_hdf5,
    export_common_trajectory_xdmf_hdf5,
};
use eqiora::artifact::{GmshMeshPolicyV1, ModelEnvelope};
use eqiora::compiler::StaticBindingValue;
use eqiora::geometry::{CanonicalGeometryV1, NamedEntitySet, PlanarFace, PlanarRegion};
use eqiora::io::hdf5::{Hdf5DatasetRequest, Hdf5FileImage, Hdf5ResolvedValues, Hdf5ScalarType};
use eqiora::solver::{
    LinearSolver, REFERENCE_LINEAR_SOLVER, REFERENCE_SOLVER_PROVIDER, SolverPlan,
};
use eqiora_numerics::{
    AuthenticatedCommonMesh, CommonLinearRequest, CommonResult, CommonSolvePolicy,
    CommonSpatialPolicy, CommonTemporalPolicy, CommonTrajectory, CommonTransientRunRequest,
    ResolvedCommonPlan,
};
use std::num::NonZeroUsize;

const SOURCE: &str = r#"
public component Inventory(support body:volume(ambient_dimension=2),
    support physical:volume(ambient_dimension=2), support outer:boundary(parent=body)) {
    coordinate xi:m on body from body[0];
    coordinate eta:m on body from body[1];
    coordinate x:m on physical from physical[0];
    coordinate y:m on physical from physical[1];
    parameter rate:1/s=0.5[1/s];
    state u:1 on body in h1;
    initial {u=2;}
    law balance on body {
        storage 3[s/m^2]*u*volume_jacobian(from=(xi,eta),at=(
            x=(1+rate*time())*xi,y=(1+rate*time())*eta));
        flux -grad(u); source 0[1/m^2];
    }
    relation boundary on outer {trace(u)=2/(1+rate*time())^2;}
}
"#;

fn geometry(low: f64, high: f64) -> CanonicalGeometryV1 {
    CanonicalGeometryV1::from_region(
        &PlanarRegion::new(
            vec![[low, low], [high, low], [high, high], [low, high]],
            vec![PlanarFace::new(vec![0, 1, 2, 3], vec![])],
            vec![
                NamedEntitySet::new("body", 2, vec![0]),
                NamedEntitySet::new("outer", 1, vec![0, 1, 2, 3]),
            ],
            1e-12,
        )
        .unwrap(),
    )
    .unwrap()
}

fn plan(rate: f64, width: f64) -> ResolvedCommonPlan {
    resolve(
        &SOURCE.replace("0.5[1/s]", &format!("{rate}[1/s]")),
        width,
        Some(CommonTemporalPolicy::backward_euler(1.).unwrap()),
        Some(geometry(-10., 10.)),
    )
}

fn resolve(
    source: &str,
    width: f64,
    temporal: Option<CommonTemporalPolicy>,
    physical: Option<CanonicalGeometryV1>,
) -> ResolvedCommonPlan {
    let geometry = geometry(0., width);
    let body = geometry.entity_set("body").unwrap();
    let mut bindings = vec![
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
    if let Some(chart) = &physical {
        bindings.push((
            "physical",
            StaticBindingValue::GeometrySupport {
                geometry: chart,
                selection: chart.entity_set("body").unwrap(),
                parent: None,
            },
        ));
    }
    let document =
        ModelDocument::compile_selected("inventory.eqi", source, "Inventory", &bindings).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let half = width / 2.;
    let mesh = format!(
        "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n1 5 1 5\n2 1 0 5\n1\n2\n3\n4\n5\n0 0 0\n{width} 0 0\n{width} {width} 0\n0 {width} 0\n{half} {half} 0\n$EndNodes\n$Elements\n1 4 1 4\n2 1 2 4\n1 1 2 5\n2 2 3 5\n3 3 4 5\n4 4 1 5\n$EndElements\n"
    );
    let mesh = AuthenticatedCommonMesh::gmsh_4152(
        geometry,
        GmshMeshPolicyV1::explicit(1e-12, 0.01, 8, 5.).unwrap(),
        mesh.into_bytes(),
    )
    .unwrap()
    .with_model_geometries(physical.into_iter().collect())
    .unwrap();
    ResolvedCommonPlan::resolve(
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
                .unwrap(),
                REFERENCE_SOLVER_PROVIDER,
            )
            .unwrap(),
        ),
        None,
        temporal,
        &REFERENCE_LINEAR_SOLVER,
        None,
        None,
    )
    .unwrap()
}

fn trajectory(plan: ResolvedCommonPlan) -> CommonTrajectory {
    let initial = plan
        .as_spatial()
        .unwrap()
        .initial_state(Some(0.), Vec::new())
        .unwrap();
    let request = CommonTransientRunRequest::from_steps(plan, initial, 2, vec![1, 2]).unwrap();
    let std::ops::ControlFlow::Continue(states) = request
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("complete trajectory")
    };
    CommonTrajectory::accept_spatial_transient(request, states).unwrap()
}

#[test]
fn common_trajectories_export_current_geometry_and_conserved_material_values() {
    for rate in [0., 0.5] {
        let trajectory = trajectory(plan(rate, 1.));
        let result = CommonResult::accept_trajectory(0., trajectory.clone()).unwrap();
        let original = result.to_bytes().unwrap();
        for (metadata, bulk) in [
            export_common_trajectory_xdmf_hdf5(
                &trajectory,
                "inventory.h5",
                XdmfHdf5ExportLimits::default(),
            )
            .unwrap(),
            export_common_result_xdmf_hdf5(
                &result,
                "inventory.h5",
                XdmfHdf5ExportLimits::default(),
            )
            .unwrap(),
        ] {
            assert_eq!(metadata.frames().len(), 2);
            let mut geometry_ids = Vec::new();
            for (frame, time) in metadata.frames().iter().zip([1., 2.]) {
                assert_eq!(frame.time_s(), Some(time));
                assert_eq!(frame.dimension(), 2);
                assert_eq!(frame.information()["eqiora.time-unit"], "s");
                assert_eq!(frame.information()["eqiora.coordinates-unit"], "m");
                assert_eq!(
                    frame.information()["eqiora.trajectory"],
                    trajectory.identity()
                );
                geometry_ids.push(frame.information()["eqiora.geometry-state"].clone());
                let [field] = frame.fields() else {
                    panic!("one density")
                };
                assert_eq!(
                    field.association(),
                    eqiora::meshing::DiscreteFieldAssociation::Vertex
                );
                assert_eq!(field.information()["eqiora.units"], "coherent-SI");
                let requests = [
                    Hdf5DatasetRequest::new(frame.geometry_path(), Hdf5ScalarType::F64, vec![5, 2])
                        .unwrap(),
                    Hdf5DatasetRequest::new(frame.topology_path(), Hdf5ScalarType::U64, vec![4, 3])
                        .unwrap(),
                    Hdf5DatasetRequest::new(field.dataset_path(), Hdf5ScalarType::F64, vec![5])
                        .unwrap(),
                ];
                let arrays = eqiora::io::hdf5::resolve_hdf5_file_image(
                    Hdf5FileImage::new(bulk.bytes()),
                    &requests,
                    Default::default(),
                )
                .unwrap();
                let [
                    Hdf5ResolvedValues::F64(coordinates),
                    Hdf5ResolvedValues::U64(topology),
                    Hdf5ResolvedValues::F64(values),
                ] = arrays.values()
                else {
                    panic!("primitive geometry/topology/field arrays")
                };
                let scale = 1. + rate * time;
                for (actual, reference) in coordinates
                    .iter()
                    .zip([0., 0., 1., 0., 1., 1., 0., 1., 0.5, 0.5])
                {
                    assert!((actual - scale * reference).abs() < 1e-13);
                }
                assert_eq!(topology, &[0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4]);
                // The material box has area scale^2 and conserved inventory 2.
                for value in values {
                    assert!((value - 2. / (scale * scale)).abs() < 1e-11);
                }
            }
            assert_eq!(geometry_ids[0] == geometry_ids[1], rate == 0.);
        }
        assert_eq!(result.to_bytes().unwrap(), original);
    }
}

#[test]
fn common_export_rejects_incomplete_or_foreign_trajectory_before_projection() {
    let trajectory = trajectory(plan(0.5, 1.));
    for foreign in [plan(0., 1.), plan(0.5, 2.)] {
        assert!(CommonTrajectory::from_bytes(&trajectory.to_bytes().unwrap(), &foreign).is_err());
        let initial = foreign
            .as_spatial()
            .unwrap()
            .initial_state(Some(0.), Vec::new())
            .unwrap();
        let request =
            CommonTransientRunRequest::from_steps(foreign, initial, 2, vec![1, 2]).unwrap();
        assert!(
            CommonTrajectory::accept_spatial_transient(
                request,
                trajectory.spatial_steps().unwrap().to_vec()
            )
            .is_err()
        );
    }
    let mut states = trajectory.spatial_steps().unwrap().to_vec();
    states.pop();
    assert!(
        CommonTrajectory::accept_spatial_transient(
            trajectory.spatial_request().unwrap().clone(),
            states
        )
        .is_err()
    );
}

#[test]
fn common_vector_result_exports_every_physical_component() {
    let source = r#"public component Inventory(
        support body:volume(ambient_dimension=2),
        support outer:boundary(parent=body)) {
        variable u:vector<m/s,2> on body in smooth;
        variable potential:m^2/s on body in smooth;
        relation prescribed on body {potential=1[m/s]*coordinate(0)+2[m/s]*coordinate(1);}
        relation momentum on body {-div(2[kg/(m*s)]*symmetric_part(grad(u)))=0;}
        relation boundary on outer {trace(u)=trace(grad(potential));}
    }"#;
    let plan = resolve(source, 1., None, None);
    let result = plan
        .as_spatial()
        .unwrap()
        .run_result(None, &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    let (metadata, bulk) =
        export_common_result_xdmf_hdf5(&result, "velocity.h5", XdmfHdf5ExportLimits::default())
            .unwrap();
    let [frame] = metadata.frames() else {
        panic!("one static frame")
    };
    assert_eq!(frame.time_s(), None);
    let [field] = frame.fields() else {
        panic!("one vector field")
    };
    assert_eq!(
        field.information()["eqiora.value-frame"],
        "spatial-cartesian"
    );
    assert_eq!(field.information()["eqiora.units"], "coherent-SI");
    let request =
        Hdf5DatasetRequest::new(field.dataset_path(), Hdf5ScalarType::F64, vec![5, 2]).unwrap();
    let arrays = eqiora::io::hdf5::resolve_hdf5_file_image(
        Hdf5FileImage::new(bulk.bytes()),
        &[request],
        Default::default(),
    )
    .unwrap();
    let [Hdf5ResolvedValues::F64(values)] = arrays.values() else {
        panic!("physical components")
    };
    // Constant velocity [1,2] has zero strain and satisfies every boundary value.
    let (pairs, remainder) = values.as_chunks::<2>();
    assert!(remainder.is_empty());
    for pair in pairs {
        assert!((pair[0] - 1.).abs() < 1e-12);
        assert!((pair[1] - 2.).abs() < 1e-12);
    }
    let mut limited = XdmfHdf5ExportLimits::default();
    limited.hdf5.max_output_bytes = 1;
    assert!(export_common_result_xdmf_hdf5(&result, "velocity.h5", limited).is_err());
}
