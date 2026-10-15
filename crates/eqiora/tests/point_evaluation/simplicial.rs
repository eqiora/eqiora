//! Point observations retain the solved Field's physical type and exact occurrence.
use super::*;
use eqiora::artifact::{GmshMeshPolicyV1, ModelEnvelope};
use eqiora::geometry::{CanonicalGeometryV1, NamedEntitySet, PlanarFace, PlanarRegion};
use eqiora::solver::{
    LinearSolver, REFERENCE_LINEAR_SOLVER, REFERENCE_SOLVER_PROVIDER, ReductionPolicy, SolverPlan,
};
use eqiora_numerics::{
    AuthenticatedCommonMesh, CommonLinearRequest, CommonResult, CommonSolvePolicy,
    CommonSpatialPolicy, ResolvedCommonPlan,
};
use std::{collections::HashMap, num::NonZeroUsize};

fn geometry() -> CanonicalGeometryV1 {
    CanonicalGeometryV1::from_region(
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
    .unwrap()
}

// Bounded provider bytes authenticate the Mesh; this does not run external Gmsh.
fn resources(geometry: &CanonicalGeometryV1, permuted: bool) -> AuthenticatedCommonMesh {
    let mut msh = String::from(
        "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n1 5 1 5\n2 1 0 5\n1\n2\n3\n4\n5\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n0.5 0.5 0\n$EndNodes\n$Elements\n1 4 1 4\n2 1 2 4\n",
    );
    for (index, mut cell) in [[1, 2, 5], [2, 3, 5], [3, 4, 5], [4, 1, 5]]
        .into_iter()
        .enumerate()
    {
        if permuted {
            cell.rotate_left(1);
        }
        msh += &format!("{} {} {} {}\n", index + 1, cell[0], cell[1], cell[2]);
    }
    msh += "$EndElements\n";
    AuthenticatedCommonMesh::gmsh_4152(
        geometry.clone(),
        GmshMeshPolicyV1::explicit(1e-12, 0.01, 8, 5.).unwrap(),
        msh.into_bytes(),
    )
    .unwrap()
}

#[test]
fn affine_p1_point_observations_use_accepted_fields_and_replay() {
    let geometry = geometry();
    let body = geometry.entity_set("body").unwrap();
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let mut bindings = [
        ("body", body, None),
        ("outer", geometry.entity_set("outer").unwrap(), Some(body)),
    ]
    .map(|(name, selection, parent)| {
        (
            name,
            StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection,
                parent,
            },
        )
    })
    .to_vec();
    bindings.push((
        "path",
        StaticBindingValue::CoordinateInterval(
            AxisBounds::new(DynQuantity::new(0.0, length), DynQuantity::new(1.0, length)).unwrap(),
        ),
    ));
    let document = ModelDocument::compile_selected(
        "planar-point.eqi",
        r#"public component Affine(
            support body:volume(ambient_dimension=2),
            support outer:boundary(parent=body),
            support path:interval(m)
        ) {
            variable temperature:K on body in smooth;
            relation balance on body {-div(grad(temperature))=0[K/m^2];}
            relation prescribed on outer {
                trace(temperature)=300[K]+2[K/m]*coordinate(0)-3[K/m]*coordinate(1);
            }
            coordinate x:m on body from body[0];
            coordinate y:m on body from body[1];
            coordinate s:m on path from path;
            observable line:K on path=pullback(temperature,from=(s),at=(x=s,y=0.5[m]));
            observable profile:K on body=temperature;
            observable fixed:K=evaluate(profile,at=(x=0.2[m],y=0.3[m]));
            observable slope:K/m=evaluate(partial(temperature,wrt=x),at=(x=0.2[m],y=0.3[m]));
        }"#,
        "Affine",
        &bindings,
    )
    .unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let length = eqiora_core::DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let temperature = eqiora_core::DimExponents::from_integers([0, 0, 0, 0, 1, 0, 0]).unwrap();
    for permuted in [false, true] {
        let resolved = ResolvedCommonPlan::resolve(
            &model,
            resources(&geometry, permuted),
            CommonSpatialPolicy::P1,
            CommonSolvePolicy::Linear(
                CommonLinearRequest::exact(
                    SolverPlan::new(
                        LinearSolver::BiConjugateGradientStabilized,
                        1e-11,
                        1e-13,
                        NonZeroUsize::new(1000).unwrap(),
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
        let observable = document.aliases()["definition.profile"].downcast().unwrap();
        let fixed = document.aliases()["definition.fixed"].downcast().unwrap();
        let slope = document.aliases()["definition.slope"].downcast().unwrap();
        let line = document.aliases()["definition.line"].downcast().unwrap();
        let result = resolved
            .as_spatial()
            .unwrap()
            .run_result(&REFERENCE_LINEAR_SOLVER)
            .unwrap();
        let bytes = result.to_bytes().unwrap();
        let replay = CommonResult::from_bytes(&bytes, &resolved).unwrap();
        for result in [&result, &replay] {
            // T=300+2*x-3*y is harmonic. P1 contains this affine solution exactly.
            for (point, expected) in [
                ([0.2, 0.3], 299.5),
                ([0.5, 0.5], 299.5),
                ([0.0, 0.0], 300.0),
                ([0.75, 0.125], 301.125),
            ] {
                let coordinates = point.map(|value| eqiora_core::DynQuantity::new(value, length));
                let sample = result
                    .observe_at(&model, observable, &coordinates, &HashMap::new())
                    .unwrap();
                let value = sample.value().real_scalar_value().unwrap();
                assert!((value.value() - expected).abs() < 1e-10);
                assert_eq!(value.dim(), temperature);
                assert_eq!(sample.result_identity(), result.identity());
                assert_eq!(sample.point().unwrap().1, coordinates);
            }
            // Restriction to y=1/2 is T(s)=298.5+2*s, with no change of measure.
            for (position, expected) in [(0.0, 298.5), (0.25, 299.0), (0.75, 300.0), (1.0, 300.5)] {
                let sample = result
                    .observe_at(
                        &model,
                        line,
                        &[DynQuantity::new(position, length)],
                        &HashMap::new(),
                    )
                    .unwrap();
                let value = sample.value().real_scalar_value().unwrap();
                assert!((value.value() - expected).abs() < 1e-10);
                assert_eq!(value.dim(), temperature);
                assert_eq!(sample.result_identity(), result.identity());
            }
            assert!(
                result
                    .observe_at(
                        &model,
                        line,
                        &[DynQuantity::new(1.1, length)],
                        &HashMap::new()
                    )
                    .is_err()
            );
            for coordinates in [
                vec![
                    eqiora_core::DynQuantity::new(1.1, length),
                    eqiora_core::DynQuantity::new(0.5, length),
                ],
                vec![
                    eqiora_core::DynQuantity::new(0.5, temperature),
                    eqiora_core::DynQuantity::new(0.5, length),
                ],
                vec![eqiora_core::DynQuantity::new(0.5, length)],
            ] {
                assert!(
                    result
                        .observe_at(&model, observable, &coordinates, &HashMap::new())
                        .is_err()
                );
            }
            let fixed_value = result.observe(&model, fixed, &HashMap::new()).unwrap();
            assert!(
                (fixed_value.value().real_scalar_value().unwrap().value() - 299.5).abs() < 1e-10
            );
            assert!(
                result
                    .observe(&model, slope, &HashMap::new())
                    .unwrap_err()
                    .message()
                    .contains("admitted Q1 space")
            );
            assert_eq!(result.to_bytes().unwrap(), bytes);
        }
    }
}
