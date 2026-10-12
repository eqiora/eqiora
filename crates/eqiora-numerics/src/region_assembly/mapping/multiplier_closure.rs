//! Closure of the complete constant-multiplier subspace, after constraints.
use super::*;
use eqiora_assembly::CsrMatrix;
use eqiora_realization::{Space, SpaceFamily};

impl<S: Coefficient + Send + Sync> RegionDofMap<S> {
    /// Replay the same assembled operator check without invoking a solver.
    pub(crate) fn verify_static_multiplier_closure<'mesh>(
        &self,
        mesh: &'mesh impl eqiora_meshing::MeshGeometry<Map<'mesh> = eqiora_meshing::AffineGeometryMap>,
        forms: Vec<(
            crate::form_compiler::region::BoundRegionForm<S>,
            eqiora_meshing::QuadratureRule,
        )>,
        natural: Vec<(usize, eqiora_assembly::LocalContribution<S>)>,
    ) -> Result<(), Diagnostic> {
        use eqiora_assembly::AssemblyBackend;
        let multipliers = forms
            .iter()
            .flat_map(|(form, _)| form.constraint_multipliers())
            .collect();
        let prepared = self.prepare_assembly(
            eqiora_assembly::AssemblyPacketSetIdentityV1::Unbound,
            mesh,
            forms,
            natural,
            None,
            None,
        )?;
        let assembled =
            eqiora_assembly::REFERENCE_ASSEMBLY_BACKEND.assemble(&prepared.plan, &prepared.work)?;
        self.require_multiplier_closure(assembled.systems()[0].matrix(), &multipliers)
    }

    pub(crate) fn require_multiplier_closure(
        &self,
        matrix: &CsrMatrix<S>,
        multipliers: &BTreeSet<RawId>,
    ) -> Result<(), Diagnostic> {
        if multipliers.is_empty() {
            return Ok(());
        }
        for field in multipliers {
            let (_, layout) = self
                .fields
                .get(field)
                .ok_or_else(|| invalid("constant multiplier is absent from the mapped Fields"))?;
            if layout.components != 1
                || !(layout.space == Space::continuous_lagrange(std::num::NonZeroU16::MIN)
                    || layout.space.family() == SpaceFamily::SimplexP1Bubble)
            {
                return Err(invalid(
                    "constant multiplier closure requires scalar P1 coefficients",
                ));
            }
        }
        // Join vertices within each Field/cell, using already-quotiented global
        // indices. Thus disconnected supports retain separate constant modes,
        // while a conforming multiplier trace joins the same physical constant.
        let mut adjacency = BTreeMap::<usize, BTreeSet<usize>>::new();
        for keys in &self.cell_keys {
            let mut local = BTreeMap::<RawId, Vec<usize>>::new();
            for key in keys {
                if multipliers.contains(&key.field) && key.entity.dimension() == 0 {
                    local.entry(key.field).or_default().push(self.globals[key]);
                }
            }
            for vertices in local.values() {
                let root = vertices[0];
                for &vertex in vertices {
                    adjacency.entry(root).or_default().insert(vertex);
                    adjacency.entry(vertex).or_default().insert(root);
                }
            }
        }
        let mut remaining = adjacency.keys().copied().collect::<BTreeSet<_>>();
        let mut modes = Vec::new();
        while let Some(root) = remaining.pop_first() {
            let mut component = vec![root];
            let mut cursor = 0;
            while cursor < component.len() {
                for neighbor in &adjacency[&component[cursor]] {
                    if remaining.remove(neighbor) {
                        component.push(*neighbor);
                    }
                }
                cursor += 1;
            }
            // A prescribed multiplier removes its component's constant shift.
            // Hierarchical bubbles contribute zero to a physical constant.
            if component
                .iter()
                .any(|&global| self.constraints.free_index(global).is_none())
            {
                continue;
            }
            modes.push(
                component
                    .into_iter()
                    .map(|global| {
                        self.constraints
                            .free_index(global)
                            .expect("free component")
                            .index()
                    })
                    .collect::<Vec<_>>(),
            );
        }
        require_independent_actions(matrix, &modes)
    }
}

fn norm<S: Coefficient>(values: &[S]) -> f64 {
    values
        .iter()
        .fold(0.0, |length, value| length.hypot(value.abs()))
}

fn require_independent_actions<S: Coefficient + Sync>(
    matrix: &CsrMatrix<S>,
    modes: &[Vec<usize>],
) -> Result<(), Diagnostic> {
    if modes.is_empty() {
        return Ok(());
    }
    // Use unit coefficient modes and the complete matrix's Frobenius scale.
    // Retain the existing closure policy's 8192-epsilon roundoff margin; the
    // sqrt(mode count) factor bounds accumulation across the tested subspace.
    // This is a numerical constant-mode closure criterion, not a proof of full
    // operator rank, an inf-sup bound, or solver convergence.
    let tolerance = 8192.0 * f64::EPSILON * norm(matrix.values()) * (modes.len() as f64).sqrt();
    if !tolerance.is_finite() {
        return Err(invalid(
            "constant multiplier closure has a nonfinite operator scale",
        ));
    }
    let mut basis: Vec<Vec<S>> = Vec::new();
    for mode in modes {
        let mut coefficients = vec![S::zero(); matrix.columns()];
        let amplitude = 1.0 / (mode.len() as f64).sqrt();
        for &index in mode {
            coefficients[index] = <S as From<f64>>::from(amplitude);
        }
        let mut action = matrix.multiply(&coefficients)?;
        // Reorthogonalization detects dependent nonzero images, unlike one
        // independent norm check per multiplier. Complex images use Hermitian
        // inner products; real images use the same scalar-generic operation.
        for _ in 0..2 {
            for previous in &basis {
                let projection = previous
                    .iter()
                    .zip(&action)
                    .fold(S::zero(), |sum, (&q, &value)| sum + q.conj() * value);
                for (value, &q) in action.iter_mut().zip(previous) {
                    *value -= q * projection;
                }
            }
        }
        let length = norm(&action);
        if !length.is_finite() || length <= tolerance {
            return Err(invalid(
                "complete operator has an unclosed constant multiplier subspace",
            ));
        }
        for value in &mut action {
            *value = *value / length;
        }
        basis.push(action);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix<S: Coefficient + Sync>(values: [S; 4]) -> CsrMatrix<S> {
        CsrMatrix::from_sorted_csr(2, 2, vec![0, 2, 4], vec![0, 1, 0, 1], values.to_vec()).unwrap()
    }

    #[test]
    fn nonzero_but_dependent_multiplier_images_do_not_prove_closure() {
        let modes = [vec![0], vec![1]];
        // Both columns are nonzero, but A*(-2,1)=0 exactly.
        for scale in [1e-100, 1.0, 1e100] {
            let dependent = matrix([scale, 2.0 * scale, 2.0 * scale, 4.0 * scale]);
            assert!(require_independent_actions(&dependent, &modes).is_err());
            let independent = matrix([2.0 * scale, 0.0, 0.0, 3.0 * scale]);
            require_independent_actions(&independent, &modes).unwrap();
            require_independent_actions(&independent, &[vec![1], vec![0]]).unwrap();
        }
    }

    #[test]
    fn complex_multiplier_closure_uses_the_whole_complex_subspace() {
        use num_complex::Complex64 as C;
        let modes = [vec![0], vec![1]];
        let dependent = matrix([
            C::new(1., 0.),
            C::new(2., 0.),
            C::new(0., 1.),
            C::new(0., 2.),
        ]);
        assert!(require_independent_actions(&dependent, &modes).is_err());
        let independent = matrix([
            C::new(1., 0.),
            C::new(2., 0.),
            C::new(0., 1.),
            C::new(0., -2.),
        ]);
        require_independent_actions(&independent, &modes).unwrap();
    }

    #[test]
    fn disconnected_multiplier_support_retains_each_constant_mode_and_its_reference() {
        use eqiora_core::{DimExponents, Id, ScalarDomain, ValueType, entity::kinds};
        use eqiora_meshing::{MeshQualityGate, SimplicialMesh};
        let domain = Id::<kinds::Domain>::new().erase();
        let field = Id::<kinds::Field>::new().erase();
        let layout = RegionFieldLayout {
            field,
            value_type: ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap(),
            space: Space::continuous_lagrange(std::num::NonZeroU16::MIN),
            range: 0..3,
            components: 1,
            scale: 1.,
        };
        let key = |vertex| FieldDof {
            field,
            entity: MeshEntity::new(0, vertex),
            slot: 0,
            component: 0,
        };
        for reversed in [false, true] {
            let mut cells = vec![vec![0, 1, 2], vec![3, 4, 5]];
            if reversed {
                cells.reverse();
            }
            let mesh = SimplicialMesh::new(
                2,
                vec![
                    vec![0., 0.],
                    vec![1., 0.],
                    vec![0., 1.],
                    vec![3., 0.],
                    vec![4., 0.],
                    vec![3., 1.],
                ],
                cells,
                MeshQualityGate::new(0.01).unwrap(),
            )
            .unwrap();
            for pinned in [false, true] {
                let prescribed = if pinned {
                    BTreeMap::from([(key(3), 0.0)])
                } else {
                    BTreeMap::new()
                };
                let mapping = RegionDofMap::new(
                    &mesh,
                    &BTreeMap::from([(domain, vec![layout.clone()])]),
                    ReferenceCell::simplex(2).unwrap(),
                    &[domain, domain],
                    &[],
                    &prescribed,
                )
                .unwrap();
                let operator = |closed_second: bool| {
                    let n = mapping.free_count();
                    let mut values = vec![0.0; n * n];
                    for i in 0..6 {
                        let Some(row) = mapping
                            .constraints
                            .free_index(mapping.global_dof(key(i)).unwrap())
                        else {
                            continue;
                        };
                        for j in 0..6 {
                            let Some(column) = mapping
                                .constraints
                                .free_index(mapping.global_dof(key(j)).unwrap())
                            else {
                                continue;
                            };
                            values[row.index() * n + column.index()] =
                                if closed_second && i >= 3 && j >= 3 {
                                    if i == j { 2.0 } else { -1.0 }
                                } else if i == j {
                                    1.0
                                } else {
                                    0.0
                                };
                        }
                    }
                    CsrMatrix::from_sorted_csr(
                        n,
                        n,
                        (0..=n).map(|i| i * n).collect(),
                        (0..n).cycle().take(n * n).collect(),
                        values,
                    )
                    .unwrap()
                };
                let multipliers = BTreeSet::from([field]);
                mapping
                    .require_multiplier_closure(&operator(false), &multipliers)
                    .unwrap();
                // I on the first island, triangle graph Laplacian on the
                // second: the global all-ones vector has nonzero action, but
                // (0,0,0,1,1,1) is a null vector until vertex 3 is prescribed.
                let result = mapping.require_multiplier_closure(&operator(true), &multipliers);
                if pinned {
                    result.unwrap();
                } else {
                    assert!(result.is_err());
                }
            }
        }
    }
}
