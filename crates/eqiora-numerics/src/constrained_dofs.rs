//! Realization-neutral strong-constraint algebra for assembled systems.

use eqiora_assembly::{
    AssemblyBackend, AssemblyMap, AssemblyPacket, AssemblyPlan, AssemblyReport, AssemblyTarget,
    DofId, IndexedAssemblyWork, LinearSystem, LocalContribution, LocalUnknown, TargetAssemblyMap,
};
use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

use crate::interleaved_dofs::InterleavedDofValues;
use crate::spatial_expression::Coefficient;

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_DISCRETIZATION, message)
}

fn fallible_zeroed<S: Coefficient>(
    length: usize,
    message: &'static str,
) -> Result<Vec<S>, Diagnostic> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message))?;
    values.resize(length, <S as From<f64>>::from(0.0));
    Ok(values)
}

/// One exact partition of global degrees of freedom into fixed values and
/// reduced-system equations.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ConstrainedDofLayout<S: Coefficient> {
    fixed_values: Vec<Option<S>>,
    free_indices: Vec<Option<DofId>>,
    free_count: usize,
}

impl<S: Coefficient + Send + Sync> ConstrainedDofLayout<S> {
    pub(crate) fn new(fixed_values: Vec<Option<S>>) -> Result<Self, Diagnostic> {
        if fixed_values
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(invalid("fixed degree-of-freedom value is non-finite"));
        }
        let mut free_count = 0_usize;
        let free_indices = fixed_values
            .iter()
            .map(|fixed| {
                fixed.is_none().then(|| {
                    let equation = DofId::new(free_count);
                    free_count += 1;
                    equation
                })
            })
            .collect();
        Ok(Self {
            fixed_values,
            free_indices,
            free_count,
        })
    }

    pub(crate) const fn free_count(&self) -> usize {
        self.free_count
    }

    pub(crate) fn free_index(&self, global: usize) -> Option<DofId> {
        self.free_indices.get(global).copied().flatten()
    }

    pub(crate) fn assemble(
        &self,
        backend: &dyn AssemblyBackend<S>,
        packet_count: usize,
        contribution: impl Fn(usize) -> Result<(LocalContribution<S>, Vec<usize>), Diagnostic> + Sync,
    ) -> Result<(LinearSystem<S>, LinearSystem<S>, AssemblyReport), Diagnostic> {
        let plan = AssemblyPlan::new(vec![
            AssemblyTarget::new(self.free_count)?,
            AssemblyTarget::new(self.fixed_values.len())?,
        ])?;
        let reduced_target = plan.target_id(0).expect("reduced target");
        let full_target = plan.target_id(1).expect("full target");
        let work = IndexedAssemblyWork::new(packet_count, |index| {
            let (local, globals) = contribution(index)?;
            AssemblyPacket::new(
                local,
                vec![
                    TargetAssemblyMap::new(reduced_target, self.reduced_map(&globals)?),
                    TargetAssemblyMap::new(full_target, self.full_map(&globals)?),
                ],
            )
        });
        let (systems, report) = backend.assemble(&plan, &work)?.into_parts();
        let mut systems = systems.into_iter();
        let reduced = systems.next().expect("validated reduced assembly target");
        let full = systems.next().expect("validated full assembly target");
        debug_assert!(systems.next().is_none());
        Ok((reduced, full, report))
    }

    pub(crate) fn is_free(&self, global: usize) -> Result<bool, Diagnostic> {
        self.free_indices
            .get(global)
            .map(Option::is_some)
            .ok_or_else(|| invalid("degree of freedom is outside the constrained layout"))
    }

    pub(crate) fn free_globals(&self) -> Vec<usize> {
        let mut globals = vec![0; self.free_count];
        for (global, free) in self.free_indices.iter().enumerate() {
            if let Some(free) = free {
                globals[free.index()] = global;
            }
        }
        globals
    }

    pub(crate) fn reduced_map(&self, global_dofs: &[usize]) -> Result<AssemblyMap<S>, Diagnostic> {
        let mut equations = Vec::with_capacity(global_dofs.len());
        let mut unknowns = Vec::with_capacity(global_dofs.len());
        for &global in global_dofs {
            let fixed = self
                .fixed_values
                .get(global)
                .ok_or_else(|| invalid("local degree of freedom is outside the global layout"))?;
            let free = self.free_indices[global];
            equations.push(free);
            unknowns.push(match fixed {
                Some(value) => LocalUnknown::Fixed(*value),
                None => LocalUnknown::Free(
                    free.expect("every unfixed degree of freedom owns a reduced equation"),
                ),
            });
        }
        AssemblyMap::new(equations, unknowns)
    }

    pub(crate) fn full_map(&self, global_dofs: &[usize]) -> Result<AssemblyMap<S>, Diagnostic> {
        for &global in global_dofs {
            if global >= self.fixed_values.len() {
                return Err(invalid(
                    "local degree of freedom is outside the full global layout",
                ));
            }
        }
        AssemblyMap::new(
            global_dofs
                .iter()
                .map(|&global| Some(DofId::new(global)))
                .collect(),
            global_dofs
                .iter()
                .map(|&global| LocalUnknown::Free(DofId::new(global)))
                .collect(),
        )
    }

    pub(crate) fn lift(&self, free_values: &[S]) -> Result<Vec<S>, Diagnostic> {
        self.lift_with_constraints(free_values, false)
    }

    pub(crate) fn lift_direction(&self, free_values: &[S]) -> Result<Vec<S>, Diagnostic> {
        self.lift_with_constraints(free_values, true)
    }

    fn lift_with_constraints(
        &self,
        free_values: &[S],
        direction: bool,
    ) -> Result<Vec<S>, Diagnostic> {
        if free_values.len() != self.free_count {
            return Err(invalid(
                "reduced solution shape differs from its constrained layout",
            ));
        }
        let mut values = fallible_zeroed(
            self.fixed_values.len(),
            "constrained solution allocation exceeds platform capacity",
        )?;
        for ((fixed, free), value) in self
            .fixed_values
            .iter()
            .zip(&self.free_indices)
            .zip(&mut values)
        {
            *value = fixed
                .map(|fixed| {
                    if direction {
                        <S as From<f64>>::from(0.0)
                    } else {
                        fixed
                    }
                })
                .unwrap_or_else(|| {
                    free_values[free
                        .expect("every unfixed degree of freedom owns a reduced equation")
                        .index()]
                });
        }
        Ok(values)
    }

    pub(crate) fn restrict(&self, values: &[S]) -> Result<Vec<S>, Diagnostic> {
        if values.len() != self.fixed_values.len() || values.iter().any(|value| !value.is_finite())
        {
            return Err(invalid(
                "full solution must be finite and match its constrained layout",
            ));
        }
        let mut free_values = fallible_zeroed(
            self.free_count,
            "reduced solution allocation exceeds platform capacity",
        )?;
        for ((&value, fixed), free) in values
            .iter()
            .zip(&self.fixed_values)
            .zip(&self.free_indices)
        {
            if fixed.is_some_and(|fixed| {
                fixed.re().to_bits() != value.re().to_bits()
                    || fixed.im().to_bits() != value.im().to_bits()
            }) {
                return Err(invalid(
                    "reduction requires each fixed value to match its exact prescribed word",
                ));
            }
            if let Some(free) = free {
                free_values[free.index()] = value;
            }
        }
        Ok(free_values)
    }

    pub(crate) fn full_residual(
        &self,
        system: &LinearSystem<S>,
        values: &[S],
    ) -> Result<Vec<S>, Diagnostic> {
        if values.len() != self.fixed_values.len() {
            return Err(invalid(
                "full solution shape differs from its constrained layout",
            ));
        }
        let mut residual = fallible_zeroed(
            system.matrix().rows(),
            "constrained residual allocation exceeds platform capacity",
        )?;
        system.matrix().multiply_into(values, &mut residual)?;
        for (value, right_hand_side) in residual.iter_mut().zip(system.rhs()) {
            *value -= *right_hand_side;
        }
        Ok(residual)
    }
}

impl ConstrainedDofLayout<f64> {
    pub(crate) fn reaction_sum<const D: usize>(
        &self,
        residual: &[f64],
    ) -> Result<[f64; D], Diagnostic> {
        if residual.len() != self.fixed_values.len() {
            return Err(invalid(
                "reaction residual shape differs from its constrained layout",
            ));
        }
        Ok(InterleavedDofValues::<D>::new(residual)?
            .sum_where(|global| self.fixed_values[global].is_some()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primal_direction_and_restriction_share_one_exact_constraint_inventory() {
        let layout = ConstrainedDofLayout::<f64>::new(vec![Some(-0.0), None, Some(1.25)]).unwrap();
        let full = layout.lift(&[7.5]).unwrap();
        assert_eq!(full[0].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(&full[1..], &[7.5, 1.25]);
        assert_eq!(layout.lift_direction(&[7.5]).unwrap(), vec![0.0, 7.5, 0.0]);
        assert_eq!(layout.restrict(&full).unwrap(), vec![7.5]);
        assert_eq!(layout.free_index(0), None);
        assert_eq!(layout.free_index(1), Some(DofId::new(0)));
        assert_eq!(layout.free_index(3), None);
        assert!(layout.restrict(&[0.0, 7.5, 1.25]).is_err());
        assert!(layout.restrict(&[-0.0, f64::NAN, 1.25]).is_err());
        assert!(layout.restrict(&full[..2]).is_err());
        assert!(layout.lift_direction(&[]).is_err());
    }

    #[test]
    fn assembly_preserves_nonsymmetric_cross_terms_and_fixed_values() {
        let layout = ConstrainedDofLayout::new(vec![Some(11.0), None]).unwrap();
        let (reduced, full, _) = layout
            .assemble(&eqiora_assembly::REFERENCE_ASSEMBLY_BACKEND, 1, |_| {
                Ok((
                    LocalContribution::new(2, 2, vec![2.0, 3.0, 5.0, 7.0], vec![13.0, 17.0])?,
                    vec![1, 0],
                ))
            })
            .unwrap();
        assert_eq!(reduced.matrix().values(), &[2.0]);
        assert_eq!(reduced.rhs(), &[-20.0]);
        assert_eq!(full.matrix().values(), &[7.0, 5.0, 3.0, 2.0]);
        assert_eq!(full.rhs(), &[17.0, 13.0]);
    }

    #[test]
    fn complex_constraints_preserve_both_words_and_eliminate_without_conjugation() {
        use num_complex::Complex64 as C;

        let fixed = C::new(-0.0, -0.0);
        let exact = ConstrainedDofLayout::new(vec![Some(fixed), None]).unwrap();
        let free = C::new(3.0, -4.0);
        let lifted = exact.lift(&[free]).unwrap();
        assert_eq!(lifted[0].re.to_bits(), (-0.0_f64).to_bits());
        assert_eq!(lifted[0].im.to_bits(), (-0.0_f64).to_bits());
        assert_eq!(exact.restrict(&lifted).unwrap(), vec![free]);
        assert!(exact.restrict(&[C::new(0.0, -0.0), free]).is_err());
        assert!(exact.restrict(&[C::new(-0.0, 0.0), free]).is_err());
        assert!(
            exact
                .restrict(&[fixed, C::new(3.0, f64::INFINITY)])
                .is_err()
        );
        assert!(ConstrainedDofLayout::new(vec![Some(C::new(1.0, f64::NAN))]).is_err());
        let direction = exact.lift_direction(&[free]).unwrap();
        assert_eq!(direction[0].re.to_bits(), 0.0_f64.to_bits());
        assert_eq!(direction[0].im.to_bits(), 0.0_f64.to_bits());
        assert_eq!(direction[1], free);

        let layout = ConstrainedDofLayout::new(vec![Some(C::new(1.0, 2.0)), None]).unwrap();
        let (reduced, full, _) = layout
            .assemble(&eqiora_assembly::REFERENCE_ASSEMBLY_BACKEND, 1, |_| {
                Ok((
                    LocalContribution::new(
                        2,
                        2,
                        vec![
                            C::new(7.0, -1.0),
                            C::new(2.0, -2.0),
                            C::new(4.0, 3.0),
                            C::new(5.0, -1.0),
                        ],
                        vec![C::new(3.0, 9.0), C::new(-2.0, 1.0)],
                    )?,
                    vec![1, 0],
                ))
            })
            .unwrap();
        // (2-2i)(1+2i) = 6+2i, so the free RHS is -3+7i.
        assert_eq!(reduced.matrix().values(), &[C::new(7.0, -1.0)]);
        assert_eq!(reduced.rhs(), &[C::new(-3.0, 7.0)]);
        // At global u=[1+2i, 3-4i], A*u-b = [33+1i, 20-38i].
        assert_eq!(
            layout
                .full_residual(&full, &layout.lift(&[free]).unwrap())
                .unwrap(),
            vec![C::new(33.0, 1.0), C::new(20.0, -38.0)]
        );
    }

    #[test]
    fn constrained_allocation_failure_is_stable() {
        let error = fallible_zeroed::<f64>(
            usize::MAX,
            "constrained allocation exceeds platform capacity",
        )
        .unwrap_err();
        assert_eq!(error.code(), codes::NUMERICAL_SOLVE_FAILED);
        assert_eq!(
            error.message(),
            "constrained allocation exceeds platform capacity"
        );
    }
}
