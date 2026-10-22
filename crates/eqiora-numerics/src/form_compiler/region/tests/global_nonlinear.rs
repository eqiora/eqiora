//! Nonholomorphic tensor diffusion reaches the shared constrained Newton owner.
use super::*;
use crate::parameter_point::ParameterPoint;
use crate::region_assembly::mapping::{
    FieldDof, RegionDofMap, RegionRealCoordinates, RegionSolveInput, prepare_mesh_facet,
};
use eqiora_meshing::{CartesianMesh, MeshEntity, MeshTopology, QuadratureRule};
use eqiora_realization::NonlinearSolvePlan;
use eqiora_solver::LinearSolveRequest;
use eqiora_solver::{LinearOperatorProperties, LinearSolver, REFERENCE_LINEAR_SOLVER, SolverPlan};
use num_complex::Complex64 as C;
use std::num::NonZeroUsize;

#[test]
fn complex_tensor_materials_share_newton_physical_recovery_and_replay() {
    let mesh = CartesianMesh::from_axes(vec![vec![0., 0.5, 1.], vec![0., 0.5, 1.]]).unwrap();
    let reference = ReferenceCell::hypercube(2).unwrap();
    let cell_count = mesh.entity_count(2).unwrap();
    for property in [false, true] {
        let law = "math.complex(1+T,1)*delta(0,1)+p";
        let declarations = if property {
            format!("property contract Conductivity(input T:1,input p:1):tensor<complex<1>,2,2> {{derivatives first_partials;}}
property release Law:Conductivity {{analytic {{value={law};source_unit:1=1;}}
 validity T>=0 and T<=20 and p>=0 and p<=3;outside reject;branch single;
 citation org.example.manufactured;license spdx.CC0_1_0;}}")
        } else {
            format!("operator k(input T:1,input p:1):tensor<complex<1>,2,2>={law};")
        };
        let argument = if property {
            ",property k:Conductivity"
        } else {
            ""
        };
        let binding = if property { ",k=Law" } else { "" };
        let source = format!("{declarations}
component Equation(support body:volume(ambient_dimension=2),parameter p:1{argument}) {{
 variable u:complex<1> on body in smooth;
 relation balance on body {{
  -div(contract(k(T=math.real(u),p=p),grad(u),axes=((1,0),)))=-math.complex(5,1)*p*p/1[m^2];
 }}
}}
model Main() {{domain body=box(0,1,0,1);parameter p:1=1;instance physics:Equation(body=body,p=p{binding});}}");
        let (transaction, model, symbols) = compile("complex-tensor-newton.eqi", &source)
            .unwrap()
            .remove(0)
            .into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        let domain = symbols.get("body").unwrap();
        let original = CompiledRegionForm::<C>::derive(&program, domain, 2).unwrap();
        let field = original.fields().next().unwrap().0;
        let parameter = symbols.get("p").unwrap().downcast().unwrap();
        let key = |vertex| FieldDof {
            field,
            entity: vertex,
            slot: 0,
            component: 0,
        };
        for p in [0.5, 1., 2.] {
            let form = original
                .bind_inputs(&ParameterPoint::bind(&program, &[parameter], &[p]).unwrap())
                .unwrap();
            let (mut fields, rows) = form
                .si_bindings(|_| Ok(Space::continuous_lagrange(NonZeroU16::MIN)))
                .unwrap();
            fields[0].scale = DynQuantity::new(2., DimExponents::DIMENSIONLESS);
            let bound = form.bind(reference, &fields, &rows, None).unwrap();
            let exact = |x: &[f64]| C::new(p * (1. + x[0] + 2. * x[1]), p * (2. - x[0] + x[1]));
            let mut prescribed = BTreeMap::new();
            for index in 0..mesh.entity_count(0).unwrap() {
                let vertex = MeshEntity::new(0, index);
                let x = mesh.vertex_coordinates(vertex).unwrap();
                if x[0] == 0. || x[1] == 0. {
                    prescribed.insert(key(vertex), exact(&x));
                }
            }
            let mapping = RegionDofMap::new(
                &mesh,
                &BTreeMap::from([(domain, bound.fields().to_vec())]),
                reference,
                &vec![domain; cell_count],
                &[],
                &prescribed,
            )
            .unwrap();
            let mut natural = Vec::new();
            for index in 0..mesh.entity_count(1).unwrap() {
                let facet = MeshEntity::new(1, index);
                let parents = mesh.incidence(facet, 2).unwrap();
                if parents.len() != 1 {
                    continue;
                }
                let vertices = mesh.incidence(facet, 0).unwrap();
                let upper = (0..2).any(|axis| {
                    vertices
                        .iter()
                        .all(|v| mesh.vertex_coordinates(v.entity).unwrap()[axis] == 1.)
                });
                if !upper {
                    continue;
                }
                let parent = parents[0];
                let load = prepare_mesh_facet(&mesh, &bound, field, facet, parent, |x, n| {
                    // phi=1+x+2y, psi=2-x+y. K=(1+p*phi+i)I+pJ.
                    // grad(u)=p[(1,2)+i(-1,1)], so div(K grad(u))=p²(5+i).
                    // J grad(u)=3p on both axes, including its zero imaginary sum.
                    let diagonal = C::new(1. + exact(x).re, 1.);
                    let x_flux = diagonal * C::new(p, -p) + C::new(3. * p * p, 0.);
                    let y_flux = diagonal * C::new(2. * p, p) + C::new(3. * p * p, 0.);
                    Ok(vec![x_flux * n[0] + y_flux * n[1]])
                })
                .unwrap();
                natural.push((parent.entity.index(), load));
            }
            let input = RegionSolveInput {
                operator_properties: LinearOperatorProperties::General,
                forms: vec![(
                    bound,
                    QuadratureRule::tensor_product_gauss_legendre(2, 3).unwrap(),
                )],
                natural,
                previous: None,
                prescribed_states: BTreeMap::new(),
                geometry_action: None,
            };
            let coordinates = RegionRealCoordinates::new(&mapping).unwrap();
            let prepared = coordinates.prepare_assembly(&mesh, &input).unwrap();
            let target = prepared.plan.target_id(0).unwrap();
            let size = prepared.plan.target(target).unwrap().size();
            assert_eq!(size, 8);
            let mut seed = coordinates.recover(&vec![0.; size]).unwrap();
            for (key, value) in &mut seed.get_mut(&field).unwrap().coefficients {
                let x = mesh.vertex_coordinates(key.entity).unwrap();
                *value = exact(&x) - C::new(0.25, 0.125) * x[0] * x[1];
            }
            // A prior State may have old essential traces. Newton applies the
            // current traces, while acceptance must reject those stale values.
            for (key, value) in &mut seed.get_mut(&field).unwrap().coefficients {
                if mapping.free_dof(*key).is_none() {
                    *value += C::new(0.25, -0.5);
                }
            }
            assert!(coordinates.validated_algebraic_point(&seed).is_err());
            let before = seed.clone();
            let nonlinear =
                NonlinearSolvePlan::new(0., 1e-11, NonZeroUsize::new(16).unwrap(), 16).unwrap();
            let output = mapping
                .solve_nonlinear(
                    &mesh,
                    &input,
                    &seed,
                    nonlinear,
                    NonZeroUsize::MIN,
                    LinearSolveRequest::new(
                        &REFERENCE_LINEAR_SOLVER,
                        SolverPlan::new(
                            LinearSolver::BiConjugateGradientStabilized,
                            1e-13,
                            1e-14,
                            NonZeroUsize::new(100).unwrap(),
                        )
                        .unwrap(),
                    ),
                )
                .unwrap();
            let solved = output.solve;
            let recovered = output.fields;
            assert!(solved.evidence.iterations > 0);
            assert_eq!(seed, before);
            let validate = |fields, initial, iterations| {
                mapping.validate_nonlinear_coefficients(
                    &mesh, &input, fields, initial, nonlinear, iterations,
                )
            };
            let (norm, target_norm, packets, targets, initial_norm) =
                validate(&recovered, &seed, solved.evidence.iterations).unwrap();
            assert!(norm <= target_norm);
            assert_eq!(initial_norm, Some(solved.evidence.initial_residual_norm));
            assert_eq!(packets, solved.assembly.report.packet_count());
            assert_eq!(targets, solved.assembly.report.target_count());
            assert!(validate(&seed, &seed, solved.evidence.iterations).is_err());
            assert!(validate(&recovered, &seed, 0).is_err());
            assert!(validate(&recovered, &seed, 17).is_err());
            validate(&recovered, &recovered, 0).unwrap();
            assert!(validate(&recovered, &recovered, 1).is_err());
            let mut corrupt = recovered.clone();
            let fixed = mapping
                .keys()
                .find(|key| mapping.free_dof(*key).is_none())
                .unwrap();
            *corrupt
                .get_mut(&field)
                .unwrap()
                .coefficients
                .get_mut(&fixed)
                .unwrap() += C::new(0., 0.01);
            assert!(validate(&corrupt, &seed, solved.evidence.iterations).is_err());
            for (key, value) in &recovered[&field].coefficients {
                let expected = exact(&mesh.vertex_coordinates(key.entity).unwrap());
                assert!((*value - expected).norm() < 1e-10, "{value} != {expected}");
            }
            let residual = prepared
                .work
                .residual(
                    &prepared.plan,
                    target,
                    crate::region_assembly::RegionPoint::Bound(&solved.values),
                )
                .unwrap();
            assert!(residual.iter().all(|value| value.abs() < 1e-11));
            let mut wrong = solved.values.clone();
            wrong[0] += 0.01;
            assert!(
                validate(
                    &coordinates.recover(&wrong).unwrap(),
                    &seed,
                    solved.evidence.iterations
                )
                .is_err()
            );
            assert!(
                prepared
                    .work
                    .residual(
                        &prepared.plan,
                        target,
                        crate::region_assembly::RegionPoint::Bound(&wrong)
                    )
                    .unwrap()
                    .iter()
                    .any(|value| value.abs() > 1e-6)
            );
        }
    }
}
