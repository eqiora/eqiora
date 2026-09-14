use std::num::{NonZeroU16, NonZeroUsize};

use eqiora_assembly::{
    AssemblyBackend, AssemblyPacketSetIdentityV1, AssemblyPlan, AssemblyTarget,
    REFERENCE_ASSEMBLY_BACKEND, TargetAssemblyMap,
};
use eqiora_compiler::compile;
use eqiora_core::DimExponents;
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_meshing::{CartesianMesh, MeshGeometry, QuadratureRule};
use eqiora_realization::{FieldSpaceBinding, Space, TraceFieldEndpoint};
use eqiora_solver::{
    LinearOperatorProperties, LinearProblem, LinearSolver, LinearSolverBackend,
    REFERENCE_LINEAR_SOLVER, SolverPlan,
};

use super::*;
use crate::form_compiler::region::{CompiledRegionForm, RegionFieldBinding};
use crate::region_assembly::{PreparedRegionAssembly, RegionAssemblyCell};
use crate::scalar_conservation::recognize_scalar_conservation;

fn source(count: usize) -> String {
    let mut source = String::from(
        r#"
public connector ScalarBoundary {
  trace value: 1;
  flux outward_flux: 1 / m;
  shape [];
  frame invariant;
  pairing euclidean_boundary_duality;
  orientation parent_outward;
}
public component Interface(
  support body: volume(ambient_dimension = 1),
  support face: boundary(parent = body),
  variable value: 1 on body,
  parameter conductivity: 1,
  port edge: ScalarBoundary over face
) {
  relation carrier on face {
    trace(value) - edge.value = 0;
    normal(conductivity * grad(value)) - edge.outward_flux = 0;
  }
}
model Chain() {
  parameter conductivity: 1 = 2;
"#,
    );
    for index in 0..count {
        source += &format!(
            r#"
  domain body{index} = box({index}, {end});
  domain lower{index} = boundary(body{index}, axis = 0, side = lower);
  domain upper{index} = boundary(body{index}, axis = 0, side = upper);
  variable value{index}: 1 on body{index};
  relation balance{index} on body{index} {{ -div(conductivity * grad(value{index})) = 0; }}
"#,
            end = index + 1
        );
        for side in ["lower", "upper"] {
            if (index == 0 && side == "lower") || (index + 1 == count && side == "upper") {
                let value = usize::from(side == "upper");
                source += &format!(
                    "relation fixed{index} on {side}{index} {{ trace(value{index}) = {value}; }}\n"
                );
            } else {
                source += &format!(
                    "instance {side}_carrier{index}: Interface(body = body{index}, face = {side}{index}, value = value{index}, conductivity = conductivity);\n"
                );
            }
        }
    }
    for index in 0..count - 1 {
        source += &format!(
            "connect upper_carrier{index}.edge, lower_carrier{}.edge;\n",
            index + 1
        );
    }
    source + "}"
}

#[test]
fn model_derived_chain_assembles_solves_and_recovers_every_exact_field() {
    for count in [2, 3, 4] {
        let (transaction, model, _) = compile("chain.eqi", &source(count))
            .unwrap()
            .remove(0)
            .into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        let descriptor = recognize_scalar_conservation(&program).unwrap();
        assert_eq!(descriptor.interfaces().len(), count - 1);
        let mesh =
            CartesianMesh::from_axes(vec![(0..=2 * count).map(|i| i as f64 / 2.0).collect()])
                .unwrap();
        let reference = ReferenceCell::hypercube(1).unwrap();
        let mut regions = descriptor.regions().collect::<Vec<_>>();
        regions.sort_by(|a, b| a.bounds()[0][0].total_cmp(&b.bounds()[0][0]));
        let spatial = regions
            .iter()
            .map(|region| {
                DomainFieldDiscretization::new(
                    region.domain().downcast().unwrap(),
                    [FieldSpaceBinding::new(
                        region.field().downcast().unwrap(),
                        Space::continuous_lagrange(NonZeroU16::MIN),
                    )],
                    [],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        // A nonunit scale exposes accidental physical/algebraic coefficient mixing.
        let scales = regions
            .iter()
            .map(|region| {
                (
                    region.field(),
                    DynQuantity::new(2.0, DimExponents::DIMENSIONLESS),
                )
            })
            .collect();
        let layouts = field_layouts(&program, &spatial, reference, &scales).unwrap();
        let domains = regions
            .iter()
            .flat_map(|region| [region.domain(); 2])
            .collect::<Vec<_>>();
        let traces = descriptor
            .interfaces()
            .map(|interface| {
                let endpoints: [_; 2] = std::array::from_fn(|index| {
                    let side = &interface.sides()[index];
                    let region = regions
                        .iter()
                        .find(|region| region.domain() == side.domain())
                        .unwrap();
                    TraceFieldEndpoint::new(
                        region.domain().downcast().unwrap(),
                        region.field().downcast().unwrap(),
                    )
                });
                let quotient = ConformingTraceQuotient::new(
                    interface.connection().downcast().unwrap(),
                    endpoints[0],
                    endpoints[1],
                )
                .unwrap();
                let endpoints = quotient.endpoints();
                let facets = (1..2 * count)
                    .filter_map(|index| {
                        let facet = MeshEntity::new(0, index);
                        let actual = mesh.incidence(facet, 1).unwrap();
                        let sides = endpoints.map(|endpoint| {
                            actual.iter().copied().find(|side| {
                                domains[side.entity.index()] == endpoint.domain().erase()
                            })
                        });
                        match sides {
                            [Some(a), Some(b)] if a.entity != b.entity => Some(TraceFacet {
                                facet,
                                sides: [a, b],
                            }),
                            _ => None,
                        }
                    })
                    .collect();
                TraceBinding { quotient, facets }
            })
            .collect::<Vec<_>>();
        let membership = domains
            .iter()
            .copied()
            .enumerate()
            .map(|(cell, domain)| (eqiora_meshing::CellId::new(cell), domain))
            .collect::<Vec<_>>();
        let quotients = traces
            .iter()
            .map(|trace| trace.quotient)
            .collect::<Vec<_>>();
        let bound = bind_region_topology(&mesh, membership.clone(), &quotients).unwrap();
        assert_eq!(bound, (domains.clone(), traces.clone()));
        let mut reverse_membership = membership.clone();
        reverse_membership.reverse();
        assert_eq!(
            bound,
            bind_region_topology(&mesh, reverse_membership, &quotients).unwrap()
        );
        let mut duplicate = membership.clone();
        duplicate.push(membership[0]);
        assert!(bind_region_topology(&mesh, duplicate, &quotients).is_err());
        assert!(bind_region_topology(&mesh, membership[1..].iter().copied(), &quotients).is_err());
        let mut outside = membership.clone();
        outside[0].0 = eqiora_meshing::CellId::new(domains.len());
        assert!(bind_region_topology(&mesh, outside, &quotients).is_err());
        assert!(bind_region_topology(&mesh, membership.clone(), &quotients[1..]).is_err());
        let mut duplicate_quotient = quotients.clone();
        duplicate_quotient.push(quotients[0]);
        assert!(bind_region_topology(&mesh, membership.clone(), &duplicate_quotient).is_err());
        let endpoints = quotients[0].endpoints();
        let foreign = ConformingTraceQuotient::new(
            quotients[0].connection(),
            TraceFieldEndpoint::new(eqiora_core::Id::new(), endpoints[0].field()),
            endpoints[1],
        )
        .unwrap();
        let mut stale = quotients.clone();
        stale[0] = foreign;
        assert!(bind_region_topology(&mesh, membership, &stale).is_err());
        let (domains, traces) = bound;
        let key = |field, vertex| FieldDof {
            field,
            entity: MeshEntity::new(0, vertex),
            slot: 0,
            component: 0,
        };
        let prescribed = BTreeMap::from([
            (key(regions[0].field(), 0), 0.0),
            (key(regions[count - 1].field(), 2 * count), 1.0),
        ]);
        let mapping =
            RegionDofMap::new(&mesh, &layouts, reference, &domains, &traces, &prescribed).unwrap();
        assert_eq!(mapping.full_count(), 2 * count + 1);
        assert_eq!(mapping.free_count(), 2 * count - 1);
        let plan = AssemblyPlan::new(vec![
            AssemblyTarget::new(mapping.free_count()).unwrap(),
            AssemblyTarget::new(mapping.full_count()).unwrap(),
        ])
        .unwrap();
        let quadrature = QuadratureRule::tensor_product_gauss_legendre(1, 2).unwrap();
        let forms = regions
            .iter()
            .map(|region| {
                let form = CompiledRegionForm::derive(&program, region.domain(), 1).unwrap();
                let fields = layouts[&region.domain()]
                    .iter()
                    .map(|layout| RegionFieldBinding {
                        field: layout.field,
                        space: layout.space,
                        scale: scales[&layout.field],
                    })
                    .collect::<Vec<_>>();
                let rows = form
                    .rows()
                    .map(|(relation, _, _)| {
                        (
                            relation,
                            DynQuantity::new(
                                1.0,
                                DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap(),
                            ),
                        )
                    })
                    .collect();
                (
                    form.bind(reference, &fields, &rows, None).unwrap(),
                    quadrature.clone(),
                )
            })
            .collect();
        let cells = (0..2 * count)
            .map(|index| RegionAssemblyCell {
                index,
                geometry: mesh.geometry_map(MeshEntity::new(1, index)).unwrap(),
                mappings: vec![
                    TargetAssemblyMap::new(
                        plan.target_id(0).unwrap(),
                        mapping.cell_map(index, true).unwrap(),
                    ),
                    TargetAssemblyMap::new(
                        plan.target_id(1).unwrap(),
                        mapping.cell_map(index, false).unwrap(),
                    ),
                ],
                previous: BTreeMap::new(),
            })
            .collect();
        let work = PreparedRegionAssembly::new(
            AssemblyPacketSetIdentityV1::Unbound,
            &plan,
            forms,
            &domains,
            cells,
            vec![],
        )
        .unwrap();
        let assembled = REFERENCE_ASSEMBLY_BACKEND.assemble(&plan, &work).unwrap();
        let system = assembled.system(plan.target_id(0).unwrap()).unwrap();
        let problem = LinearProblem::new(
            system.matrix(),
            system.rhs(),
            LinearOperatorProperties::General,
        )
        .unwrap();
        let solver = SolverPlan::new(
            LinearSolver::BiConjugateGradientStabilized,
            1e-12,
            1e-14,
            NonZeroUsize::new(100).unwrap(),
        )
        .unwrap();
        let solution = REFERENCE_LINEAR_SOLVER.solve(&problem, solver).unwrap();
        let reactions = crate::region_assembly::InterfaceReactions::prepare(
            &work,
            plan.target_id(1).unwrap(),
            &mapping,
            &domains,
        )
        .unwrap();
        let full = mapping.lift(solution.values(), false).unwrap();
        let actions = reactions.recover(&full).unwrap();
        // Independent weak action: k=2 and u=x/L give endpoint actions ±2/L.
        // For three Regions this is ±2/3, including BOTH ends of the middle Region.
        for (quotient, keys) in mapping.traces() {
            for key in keys {
                let region = regions
                    .iter()
                    .find(|region| region.field() == key.field)
                    .unwrap();
                let x = key.entity.index() as f64 / 2.0;
                let sign = if x == region.bounds()[0][0] {
                    -1.0
                } else {
                    1.0
                };
                let actual = actions.action(quotient.connection().erase(), *key).unwrap();
                assert!((actual - sign * 2.0 / count as f64).abs() < 1e-11);
            }
        }
        assert!(actions.imbalance_norm < 1e-11);
        let mut nonfinite = full.clone();
        nonfinite[0] = f64::NAN;
        assert!(reactions.recover(&nonfinite).is_err());
        assert!(
            crate::region_assembly::InterfaceReactions::prepare(
                &work,
                plan.target_id(1).unwrap(),
                &mapping,
                &domains[..domains.len() - 1],
            )
            .is_err()
        );
        let mut wrong_domain = domains.clone();
        wrong_domain[0] = domains[domains.len() - 1];
        assert!(
            crate::region_assembly::InterfaceReactions::prepare(
                &work,
                plan.target_id(1).unwrap(),
                &mapping,
                &wrong_domain,
            )
            .is_err()
        );
        let mut permuted_traces = traces.clone();
        permuted_traces.reverse();
        let permuted_map = RegionDofMap::new(
            &mesh,
            &layouts,
            reference,
            &domains,
            &permuted_traces,
            &prescribed,
        )
        .unwrap();
        let permuted_reactions = crate::region_assembly::InterfaceReactions::prepare(
            &work,
            plan.target_id(1).unwrap(),
            &permuted_map,
            &domains,
        )
        .unwrap();
        assert_eq!(actions, permuted_reactions.recover(&full).unwrap());
        let mut junction_traces = traces.clone();
        let mut duplicate_trace = traces[0].clone();
        let [first, second] = duplicate_trace.quotient.endpoints();
        duplicate_trace.quotient =
            ConformingTraceQuotient::new(eqiora_core::Id::new(), first, second).unwrap();
        junction_traces.push(duplicate_trace);
        let junction_mapping = RegionDofMap::new(
            &mesh,
            &layouts,
            reference,
            &domains,
            &junction_traces,
            &prescribed,
        )
        .unwrap();
        let error = crate::region_assembly::InterfaceReactions::prepare(
            &work,
            plan.target_id(1).unwrap(),
            &junction_mapping,
            &domains,
        )
        .unwrap_err();
        assert!(error.message().contains("exact facet dual"));
        let (quotient, keys) = &mapping.traces()[0];
        let key = *keys.first().unwrap();
        assert!(
            actions
                .action(
                    eqiora_core::Id::<eqiora_core::entity::kinds::Connection>::new().erase(),
                    key
                )
                .is_err()
        );
        assert!(
            actions
                .action(
                    quotient.connection().erase(),
                    FieldDof {
                        field: eqiora_core::Id::<eqiora_core::entity::kinds::Field>::new().erase(),
                        ..key
                    }
                )
                .is_err()
        );
        let inventory = regions
            .iter()
            .map(|region| region.field())
            .collect::<Vec<_>>();
        let recovered = mapping.recover(solution.values(), &inventory).unwrap();
        assert_eq!(recovered.len(), count);
        let mut permuted = inventory.clone();
        permuted.reverse();
        assert_eq!(
            recovered,
            mapping.recover(solution.values(), &permuted).unwrap()
        );
        assert!(mapping.recover(solution.values(), &inventory[1..]).is_err());
        let mut duplicate = inventory.clone();
        duplicate.push(inventory[0]);
        assert!(mapping.recover(solution.values(), &duplicate).is_err());
        let mut foreign = inventory.clone();
        foreign[0] = eqiora_core::Id::<eqiora_core::entity::kinds::Field>::new().erase();
        assert!(mapping.recover(solution.values(), &foreign).is_err());
        // Independent exact steady solution: k=2, no source, u(0)=0,u(L)=1 => u=x/L.
        for region in &regions {
            let field = &recovered[&region.field()];
            assert_eq!(field.domain, region.domain());
            assert_eq!(field.value_type, layouts[&region.domain()][0].value_type);
            assert_eq!(field.coefficients.len(), 3);
            for (dof, value) in &field.coefficients {
                assert_eq!(dof.field, region.field());
                assert!((value - dof.entity.index() as f64 / (2 * count) as f64).abs() < 1e-11);
            }
        }
        let mut reversed = traces.clone();
        reversed.reverse();
        assert_eq!(
            mapping,
            RegionDofMap::new(&mesh, &layouts, reference, &domains, &reversed, &prescribed)
                .unwrap()
        );
        assert!(
            RegionDofMap::new(
                &mesh,
                &layouts,
                reference,
                &domains,
                &traces[..traces.len() - 1],
                &prescribed
            )
            .is_err()
        );
        let mut stale = traces.clone();
        stale[0].facets[0].sides.swap(0, 1);
        assert!(
            RegionDofMap::new(&mesh, &layouts, reference, &domains, &stale, &prescribed).is_err()
        );
        assert!(
            mapping
                .recover(&vec![0.0; mapping.free_count() + 1], &inventory)
                .is_err()
        );
    }
}

#[test]
fn vector_q1_consumer_uses_same_entity_mapping_and_physical_recovery() {
    let source = r#"
model VectorRegion() {
  domain body = box(0,1,0,1);
  domain left = boundary(body,axis=0,side=lower);
  domain right = boundary(body,axis=0,side=upper);
  domain bottom = boundary(body,axis=1,side=lower);
  domain top = boundary(body,axis=1,side=upper);
  variable value: vector<1,2> on body;
  parameter conductivity: 1 = 3;
  parameter endpoint: vector<1,2> = tensor_value(frame=body,components=[1,2]);
  relation balance on body { -div(conductivity * grad(value)) = 0; }
  relation l on left { trace(value) = 0; }
  relation r on right { trace(value) - endpoint = 0; }
  relation b on bottom { normal(conductivity * grad(value)) = 0; }
  relation t on top { normal(conductivity * grad(value)) = 0; }
}
"#;
    let (transaction, model, symbols) = compile("vector.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let field = symbols.get("value").unwrap();
    let domain = symbols.get("body").unwrap();
    let reference = ReferenceCell::hypercube(2).unwrap();
    let mesh = CartesianMesh::from_axes(vec![vec![0.0, 0.5, 1.0], vec![0.0, 0.5, 1.0]]).unwrap();
    let space = Space::continuous_lagrange(NonZeroU16::MIN);
    let spatial = [DomainFieldDiscretization::new(
        domain.downcast().unwrap(),
        [FieldSpaceBinding::new(field.downcast().unwrap(), space)],
        [],
    )
    .unwrap()];
    let scale = DynQuantity::new(4.0, DimExponents::DIMENSIONLESS);
    let scales = BTreeMap::from([(field, scale)]);
    let layouts = field_layouts(&program, &spatial, reference, &scales).unwrap();
    let mut fixed = BTreeMap::new();
    for vertex in 0..9 {
        let entity = MeshEntity::new(0, vertex);
        let x = mesh.vertex_coordinates(entity).unwrap()[0];
        if x == 0.0 || x == 1.0 {
            for component in 0..2 {
                fixed.insert(
                    FieldDof {
                        field,
                        entity,
                        slot: 0,
                        component,
                    },
                    x * (component + 1) as f64,
                );
            }
        }
    }
    let mapping = RegionDofMap::new(&mesh, &layouts, reference, &[domain; 4], &[], &fixed).unwrap();
    assert_eq!(mapping.full_count(), 18);
    assert_eq!(mapping.field_free_dofs(field).unwrap().len(), 6);
    let plan = AssemblyPlan::new(vec![
        AssemblyTarget::new(mapping.free_count()).unwrap(),
        AssemblyTarget::new(mapping.full_count()).unwrap(),
    ])
    .unwrap();
    let form = CompiledRegionForm::derive(&program, domain, 2).unwrap();
    let rows = form
        .rows()
        .map(|(relation, _, _)| (relation, DynQuantity::new(1.0, DimExponents::DIMENSIONLESS)))
        .collect();
    let bound = form
        .bind(
            reference,
            &[RegionFieldBinding {
                field,
                space,
                scale,
            }],
            &rows,
            None,
        )
        .unwrap();
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(2, 2).unwrap();
    let cells = (0..4)
        .map(|index| RegionAssemblyCell {
            index,
            geometry: mesh.geometry_map(MeshEntity::new(2, index)).unwrap(),
            previous: BTreeMap::new(),
            mappings: vec![
                TargetAssemblyMap::new(
                    plan.target_id(0).unwrap(),
                    mapping.cell_map(index, true).unwrap(),
                ),
                TargetAssemblyMap::new(
                    plan.target_id(1).unwrap(),
                    mapping.cell_map(index, false).unwrap(),
                ),
            ],
        })
        .collect();
    let work = PreparedRegionAssembly::new(
        AssemblyPacketSetIdentityV1::Unbound,
        &plan,
        vec![(bound, quadrature)],
        &[domain; 4],
        cells,
        vec![],
    )
    .unwrap();
    let assembled = REFERENCE_ASSEMBLY_BACKEND.assemble(&plan, &work).unwrap();
    let system = assembled.system(plan.target_id(0).unwrap()).unwrap();
    let problem = LinearProblem::new(
        system.matrix(),
        system.rhs(),
        LinearOperatorProperties::General,
    )
    .unwrap();
    let solver = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-12,
        1e-14,
        NonZeroUsize::new(100).unwrap(),
    )
    .unwrap();
    let solution = REFERENCE_LINEAR_SOLVER.solve(&problem, solver).unwrap();
    let recovered = mapping.recover(solution.values(), &[field]).unwrap();
    assert_eq!(recovered[&field].domain, domain);
    assert_eq!(recovered[&field].value_type, layouts[&domain][0].value_type);
    for (dof, value) in &recovered[&field].coefficients {
        let x = mesh.vertex_coordinates(dof.entity).unwrap()[0];
        assert!((value - x * (dof.component + 1) as f64).abs() < 1e-11);
        assert!(mapping.global_dof(*dof).is_some());
        assert_eq!(mapping.free_dof(*dof).is_some(), x != 0.0 && x != 1.0);
    }
    assert!(
        mapping
            .recover(&vec![f64::NAN; mapping.free_count()], &[field])
            .is_err()
    );
    let mut wrong_space = layouts.clone();
    wrong_space.get_mut(&domain).unwrap()[0].space = Space::simplex_p1_bubble();
    assert!(RegionDofMap::new(&mesh, &wrong_space, reference, &[domain; 4], &[], &fixed).is_err());
}
