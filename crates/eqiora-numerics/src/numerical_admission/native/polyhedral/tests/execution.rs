use super::*;
use crate::spatial_expression::Coefficient;
use num_complex::Complex64 as C;

fn execute<S: Coefficient + crate::finalized_spatial::ResidualScalar + Send>(
    complex: bool,
    face: bool,
    backend: &dyn LinearSolverBackend<S>,
) {
    let space = if face {
        Space::tetrahedral_face()
    } else {
        Space::tetrahedral_edge()
    };
    let policy = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-13,
        1e-14,
        NonZeroUsize::new(2000).unwrap(),
    )
    .unwrap();
    for permuted in [false, true] {
        let (_, program, owner) = fixture_source(
            &[vec![0, 1, 2, 3]],
            false,
            permuted,
            2.,
            complex,
            |source| {
                let potential = "2*coordinate(0)+3*coordinate(1)+4*coordinate(2)";
                let (ty, value) = if complex {
                    (
                        "complex<m>",
                        format!("math.complex({potential},2*({potential}))"),
                    )
                } else {
                    ("m", potential.to_owned())
                };
                let operator = if face {
                    "-grad(div(u))"
                } else {
                    "curl(curl(u))"
                };
                let source = source.replace("relation balance", &format!("parameter a:m^2=2[m^2]; variable potential:{ty} on body; relation prescribed on body {{ potential={value}; }} relation balance"));
                let source = source.replace(
                    "curl(curl(u)) = 0",
                    &format!("a*({operator})+u=grad(potential)"),
                );
                source.replace(
                    "tangential_trace(-curl(u))",
                    if face {
                        "normal(a*isotropic_lift(div(u)))"
                    } else {
                        "tangential_trace(-a*curl(u))"
                    },
                )
            },
        );
        let equations =
            ExecutableLinearEquations::<S>::polyhedral(&program, &owner.resources).unwrap();
        let output = equations
            .execute(
                NonZeroUsize::MIN,
                LinearSolveRequest::new(backend, policy),
                &owner.resources,
                space,
                |reactions, values| reactions.recover(values),
            )
            .unwrap();
        assert_eq!(output.fields.len(), 1);
        let (_, ty, coefficients, actual_space) = &output.fields[0];
        assert_eq!(*actual_space, space);
        assert_eq!(ty.shape().component_count(), Some(3));
        assert_eq!(coefficients.len(), if face { 4 } else { 6 });
        let NativeMeshResources::GmshSimplicial { mesh, .. } = &owner.resources else {
            unreachable!()
        };
        // The exact constant solution (2,3,4), times (1+2i) for complex,
        // has zero curl/divergence. Canonical entity integrals are independent
        // of the local basis and of either positive cell vertex ordering.
        for (index, actual) in coefficients.iter().enumerate() {
            let vertices = mesh
                .mesh()
                .entity_vertices(MeshEntity::new(if face { 2 } else { 1 }, index))
                .unwrap();
            let a = &mesh.mesh().vertices()[vertices[0].index()];
            let b = &mesh.mesh().vertices()[vertices[1].index()];
            let v: [f64; 3] = std::array::from_fn(|i| b[i] - a[i]);
            let measure = if face {
                let c = &mesh.mesh().vertices()[vertices[2].index()];
                let w: [f64; 3] = std::array::from_fn(|i| c[i] - a[i]);
                [
                    v[1] * w[2] - v[2] * w[1],
                    v[2] * w[0] - v[0] * w[2],
                    v[0] * w[1] - v[1] * w[0],
                ]
                .map(|x| x * 0.5)
            } else {
                v
            };
            let moment = (0..3).map(|i| [2., 3., 4.][i] * measure[i]).sum::<f64>();
            let expected = C::new(moment, if complex { 2. * moment } else { 0. });
            assert!(
                (C::new(actual.re(), actual.im()) - expected).norm()
                    <= 1e-9 * expected.norm().max(1.)
            );
        }
        let (_, _, foreign) = fixture(&[vec![0, 1, 2, 3]], false, permuted, 5.);
        let error = equations
            .execute(
                NonZeroUsize::MIN,
                LinearSolveRequest::new(backend, policy),
                &foreign.resources,
                space,
                |reactions, values| reactions.recover(values),
            )
            .unwrap_err();
        assert!(
            error
                .message()
                .contains("Mesh differs from authenticated Model support")
        );
        let wrong = if face {
            Space::tetrahedral_edge()
        } else {
            Space::tetrahedral_face()
        };
        assert!(
            equations
                .execute(
                    NonZeroUsize::MIN,
                    LinearSolveRequest::new(backend, policy),
                    &owner.resources,
                    wrong,
                    |reactions, values| reactions.recover(values)
                )
                .is_err()
        );
        let error = equations
            .execute(
                NonZeroUsize::MIN,
                LinearSolveRequest::new(backend, policy),
                &owner.resources,
                space,
                |_, _| Err(invalid("injected recovery failure")),
            )
            .unwrap_err();
        assert!(error.message().contains("injected recovery failure"));
    }
}

#[test]
fn authenticated_polyhedral_equations_execute_real_and_complex_moments() {
    for face in [false, true] {
        execute::<f64>(false, face, &REFERENCE_LINEAR_SOLVER);
        execute::<C>(true, face, &REFERENCE_LINEAR_SOLVER);
    }
}
