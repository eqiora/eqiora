use super::*;
use crate::region_assembly::{DomainReactions, InterfaceReactions};
use num_complex::Complex64 as C;

#[test]
fn complex_domain_and_interface_actions_retain_phase_and_exact_owners() {
    // On two unit intervals, u=(1+2i)x and a=3+i give a*u'=1+7i.
    // The outward weak actions at the shared endpoint are +(1+7i), -(1+7i).
    // Changing only the right coefficient to 3-i makes its action -(5+5i),
    // hence the interface defect is -4+2i with Euclidean norm sqrt(20).
    for (right_imaginary, right_flux, expected_imbalance) in
        [(1, C::new(1., 7.), 0.), (-1, C::new(5., 5.), 20_f64.sqrt())]
    {
        let source = format!(
            r#"model Wave() {{
                domain left = box(0, 1);
                domain right = box(1, 2);
                parameter a:complex<m^2> = math.complex(3[m^2], 1[m^2]);
                parameter b:complex<m^2> = math.complex(3[m^2], {right_imaginary}[m^2]);
                variable u:complex<1> on left;
                variable v:complex<1> on right;
                relation first on left {{ -div(a*grad(u))=0; }}
                relation second on right {{ -div(b*grad(v))=0; }}
            }}"#
        );
        let (transaction, model, symbols) = compile("complex-reactions.eqi", &source)
            .unwrap()
            .remove(0)
            .into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        let domains = [symbols.get("left").unwrap(), symbols.get("right").unwrap()];
        let fields = [symbols.get("u").unwrap(), symbols.get("v").unwrap()];
        let reference = ReferenceCell::hypercube(1).unwrap();
        let mesh = CartesianMesh::from_axes(vec![vec![0., 1., 2.]]).unwrap();
        let space = Space::continuous_lagrange(NonZeroU16::MIN);
        let spatial = domains
            .into_iter()
            .zip(fields)
            .map(|(domain, field)| {
                DomainFieldDiscretization::new(
                    domain.downcast().unwrap(),
                    [FieldSpaceBinding::new(field.downcast().unwrap(), space)],
                    [],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let scales = fields
            .into_iter()
            .map(|field| (field, DynQuantity::new(2., DimExponents::DIMENSIONLESS)))
            .collect();
        let layouts = field_layouts(&program, &spatial, reference, &scales).unwrap();
        // This private assembly fixture supplies an exact quotient explicitly;
        // it does not assert source-level complex Connection admission.
        let quotient = ConformingTraceQuotient::new(
            eqiora_core::Id::new(),
            TraceFieldEndpoint::new(
                domains[0].downcast().unwrap(),
                fields[0].downcast().unwrap(),
            ),
            TraceFieldEndpoint::new(
                domains[1].downcast().unwrap(),
                fields[1].downcast().unwrap(),
            ),
        )
        .unwrap();
        let (_, traces) = bind_region_topology(
            &mesh,
            domains
                .into_iter()
                .enumerate()
                .map(|(index, domain)| (eqiora_meshing::CellId::new(index), domain)),
            &[quotient],
        )
        .unwrap();
        let mapping = RegionDofMap::<C>::new(
            &mesh,
            &layouts,
            reference,
            &domains,
            &traces,
            &BTreeMap::new(),
        )
        .unwrap();
        let plan =
            AssemblyPlan::new(vec![AssemblyTarget::new(mapping.full_count()).unwrap()]).unwrap();
        let target = plan.target_id(0).unwrap();
        let forms = domains
            .into_iter()
            .zip(fields)
            .map(|(domain, field)| {
                let form = CompiledRegionForm::<C>::derive(&program, domain, 1).unwrap();
                let row_dimension = DimExponents::from_integers([0, -1, 0, 0, 0, 0, 0]).unwrap();
                let rows = form
                    .rows()
                    .map(|(id, _, _)| (id, DynQuantity::new(1., row_dimension)))
                    .collect();
                (
                    form.bind(
                        reference,
                        &[RegionFieldBinding {
                            field,
                            space,
                            scale: scales[&field],
                        }],
                        &rows,
                        None,
                    )
                    .unwrap(),
                    QuadratureRule::gauss_legendre(2).unwrap(),
                )
            })
            .collect();
        let cells = (0..2)
            .map(|index| RegionAssemblyCell {
                index,
                geometry: mesh.geometry_map(MeshEntity::new(1, index)).unwrap(),
                mappings: vec![TargetAssemblyMap::new(
                    target,
                    mapping.cell_map(index, false).unwrap(),
                )],
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
        let key = |field, vertex| FieldDof {
            field,
            entity: MeshEntity::new(0, vertex),
            slot: 0,
            component: 0,
        };
        let mut full = vec![C::new(0., 0.); mapping.full_count()];
        for (field, vertices) in [(fields[0], [0, 1]), (fields[1], [1, 2])] {
            for vertex in vertices {
                // Algebraic coefficients are physical values divided by scale 2.
                full[mapping.global_dof(key(field, vertex)).unwrap()] =
                    C::new(0.5, 1.) * vertex as f64;
            }
        }
        let recovered = DomainReactions::prepare(
            &work,
            target,
            mapping.full_count(),
            &domains,
            &(0..mapping.full_count()).collect(),
        )
        .unwrap()
        .recover(&full)
        .unwrap();
        for (index, flux) in [C::new(1., 7.), right_flux].into_iter().enumerate() {
            let actual = &recovered.values[&domains[index]];
            assert!(
                (actual[mapping.global_dof(key(fields[index], index)).unwrap()] + flux).norm()
                    < 1e-12
            );
            assert!(
                (actual[mapping.global_dof(key(fields[index], index + 1)).unwrap()] - flux).norm()
                    < 1e-12
            );
        }
        let reactions = InterfaceReactions::prepare(&work, target, &mapping, &domains).unwrap();
        let actions = reactions.recover(&full).unwrap();
        assert!(
            (actions
                .action(quotient.connection().erase(), key(fields[0], 1))
                .unwrap()
                - C::new(1., 7.))
            .norm()
                < 1e-12
        );
        assert!(
            (actions
                .action(quotient.connection().erase(), key(fields[1], 1))
                .unwrap()
                + right_flux)
                .norm()
                < 1e-12
        );
        assert!((actions.imbalance_norm - expected_imbalance).abs() < 1e-12);
        assert!(
            actions
                .action(
                    eqiora_core::Id::<eqiora_core::entity::kinds::Connection>::new().erase(),
                    key(fields[0], 1)
                )
                .is_err()
        );
        full[0].im = f64::NAN;
        assert!(reactions.recover(&full).is_err());
    }
}
