use super::*;
use eqiora_compiler::compile;
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use num_complex::Complex64 as C;

fn execute(source: &str) -> CommonLinearRunOutput<C> {
    let (transaction, model, symbols) = compile("complex-spatial-execution.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let equations = ExecutableLinearEquations::<C>::new(
        &program,
        symbols.get("body").unwrap(),
        vec![[0., 6.]],
        BTreeMap::from([
            ((0, BoundarySide::Lower), symbols.get("left").unwrap()),
            ((0, BoundarySide::Upper), symbols.get("right").unwrap()),
        ]),
    )
    .unwrap();
    let mesh = eqiora_meshing::CartesianMesh::from_axes(vec![vec![0., 3., 6.]]).unwrap();
    let source_equations = ExecutableLinearEquations::<C>::source_regions(&program, &mesh).unwrap();
    assert_eq!(source_equations, equations);
    let equations = source_equations;
    // Source bounds must cover whole cells in the exact supplied Mesh.
    let crossed = eqiora_meshing::CartesianMesh::from_axes(vec![vec![-1., 3., 6.]]).unwrap();
    assert!(ExecutableLinearEquations::<C>::source_regions(&program, &crossed).is_err());
    let structure = equations.algebraic_structure(None).unwrap();
    let mut policy = NativeLinearPolicy::exact::<C>(
        SolverPlan::new(
            LinearSolver::BiConjugateGradientStabilized,
            1e-12,
            1e-14,
            NonZeroUsize::new(32).unwrap(),
        )
        .unwrap(),
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap();
    policy.planning_profile = Some(
        eqiora_solver::HostSerialSolverProfile::general_canonical_csr()
            .with_structure(structure.clone())
            .unwrap(),
    );
    let backend = policy
        .checked_complex_backend(&REFERENCE_LINEAR_SOLVER, Some(&structure))
        .unwrap();
    let output = equations
        .execute_cartesian(
            NonZeroUsize::MIN,
            LinearSolveRequest::new(&backend, policy.solver),
            &mesh,
            &equations
                .discretizations(Space::continuous_lagrange(std::num::NonZeroU16::MIN), None)
                .unwrap(),
            LinearOperatorProperties::General,
            |reactions, values| reactions.recover(values),
        )
        .unwrap();
    assert_eq!(output.fields.len(), 1);
    assert_eq!(output.fields[0].0.erase(), symbols.get("u").unwrap());
    assert_eq!(
        output.fields[0].1.scalar_domain(),
        eqiora_core::ScalarDomain::Complex
    );
    assert_eq!(output.fields[0].2.len(), 3);
    output
}

const CONSTANT: &str = "model Wave() {
 domain body=box(0,6);
 domain left=boundary(body,axis=0,side=lower);
 domain right=boundary(body,axis=0,side=upper);
 parameter a:complex<m^2>=math.complex(6[m^2],6[m^2]);
 parameter q:complex<1>=math.complex(1,1);
 variable u:complex<1> on body in h1;
 relation balance on body { -div(a*grad(u))+q*u=math.complex(-2,4); }
 relation fixed_left on left { trace(u)=math.complex(1,3); }
 relation fixed_right on right { trace(u)=math.complex(1,3); }
}";

#[test]
fn shared_spatial_executor_retains_complex_boundary_load_and_field_values() {
    // grad(c)=0 and (1+i)(1+3i)=-2+4i; Q1 represents c exactly.
    let output = execute(CONSTANT);
    for value in &output.fields[0].2 {
        assert!((*value - C::new(1., 3.)).norm() < 1e-11);
    }
    // Keep the prescribed values and change only the imaginary load.
    let changed = execute(&CONSTANT.replace("math.complex(-2,4)", "math.complex(-2,-4)"));
    assert!((changed.fields[0].2[1] - C::new(1., 3.)).norm() > 1.);
    assert_eq!(changed.fields[0].2[0], C::new(1., 3.));
    assert_eq!(changed.fields[0].2[2], C::new(1., 3.));
}

#[test]
fn shared_spatial_executor_retains_oriented_complex_natural_flux() {
    let source = CONSTANT
        .replace("on body in h1;", "on body in smooth;")
        .replace("math.complex(6[m^2],6[m^2])", "math.complex(3[m^2],1[m^2])")
        .replace("math.complex(1,1)", "math.complex(0,0)")
        .replace("math.complex(-2,4)", "math.complex(0,0)")
        .replace("trace(u)=math.complex(1,3)", "trace(u)=math.complex(0,0)")
        .replace(
            "relation fixed_right on right { trace(u)=math.complex(0,0); }",
            "relation natural on right { normal(a*grad(u))=math.complex(1[m],7[m]); }",
        );
    // (3+i)(1+2i)=1+7i, hence u=(1+2i)x/m for the right outward flux.
    let output = execute(&source);
    for (value, x) in output.fields[0].2.iter().zip([0., 3., 6.]) {
        assert!((*value - C::new(x, 2. * x)).norm() < 1e-10);
    }
    let reversed = execute(&source.replace("math.complex(1[m],7[m])", "math.complex(-1[m],-7[m])"));
    for (value, x) in reversed.fields[0].2.iter().zip([0., 3., 6.]) {
        assert!((*value + C::new(x, 2. * x)).norm() < 1e-10);
    }
}

#[test]
fn complex_corner_compatibility_checks_each_finite_coordinate() {
    use crate::cartesian_elliptic::support::require_compatible_boundary_value as check;
    assert!(check(Some(C::new(1e300, 1.)), C::new(1e300, 2.)).is_err());
    let large = C::new(1.7e308, 1.7e308);
    assert_eq!(check(Some(large), large).unwrap(), Some(large));
    assert!(check(Some(large), C::new(-1.7e308, 1.7e308)).is_err());
    for value in [C::new(f64::NAN, 0.), C::new(0., f64::INFINITY)] {
        assert!(check(None, value).is_err());
    }
    for (delta, accepted) in [(128. * f64::EPSILON, true), (512. * f64::EPSILON, false)] {
        assert_eq!(check(Some(1.), 1. + delta).is_ok(), accepted);
        assert_eq!(
            check(Some(C::new(1., 1.)), C::new(1. + delta, 1.)).is_ok(),
            accepted
        );
        assert_eq!(
            check(Some(C::new(1., 1.)), C::new(1., 1. + delta)).is_ok(),
            accepted
        );
    }
}
