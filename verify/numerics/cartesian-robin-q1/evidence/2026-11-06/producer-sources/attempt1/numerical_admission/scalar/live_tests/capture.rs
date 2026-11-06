//! Opt-in immutable observation of the finalized occurrence retained by public Run.
use super::*;
use crate::region_assembly::mapping::RegionDofMap;
use eqiora_assembly::LinearSystem;
use eqiora_core::{DynQuantity, RawId};
use eqiora_meshing::QuadratureRule;
use std::cell::RefCell;

pub(in crate::numerical_admission) struct CapturedStatic {
    pub(in crate::numerical_admission) plan_identity: String,
    pub(in crate::numerical_admission) full: LinearSystem<f64>,
    pub(in crate::numerical_admission) mapping: RegionDofMap<f64>,
    pub(in crate::numerical_admission) dual_scales: Vec<(RawId, DynQuantity)>,
    pub(in crate::numerical_admission) quadrature: Vec<QuadratureRule>,
}

thread_local! {
    static ACTIVE:RefCell<Option<Vec<CapturedStatic>>>=const{RefCell::new(None)};
}

pub(in crate::numerical_admission) struct CaptureGuard(std::marker::PhantomData<std::rc::Rc<()>>);
impl CaptureGuard {
    pub(in crate::numerical_admission) fn start() -> Self {
        ACTIVE.with(|active| {
            let mut active = active.borrow_mut();
            assert!(active.is_none(), "static capture cannot be nested");
            *active = Some(Vec::new());
        });
        Self(std::marker::PhantomData)
    }
    pub(in crate::numerical_admission) fn finish(self) -> CapturedStatic {
        let mut captured = ACTIVE.with(|active| active.borrow_mut().take()).unwrap();
        assert_eq!(
            captured.len(),
            1,
            "one public Run finalized occurrence required"
        );
        captured.pop().unwrap()
    }
}
impl Drop for CaptureGuard {
    fn drop(&mut self) {
        ACTIVE.with(|active| {
            active.borrow_mut().take();
        });
    }
}

pub(in crate::numerical_admission) fn record(
    plan: &CommonSpatialPlan,
    system: Option<&regions::StaticRegionSystem<f64>>,
) -> Result<(), Diagnostic> {
    ACTIVE.with(|active| {
        let mut active = active.borrow_mut();
        let Some(captures) = active.as_mut() else {
            return Ok(());
        };
        let system = system.ok_or_else(|| {
            invalid("static capture requires the actual retained real FEM system")
        })?;
        let dual_scales = system
            .assembly
            .input
            .forms
            .iter()
            .flat_map(|(form, _)| {
                form.fields().iter().map(|field| {
                    form.physical_dual_scale(field.field)
                        .map(|scale| (field.field, scale))
                })
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        captures.push(CapturedStatic {
            plan_identity: plan.identity().to_owned(),
            full: system.assembly.full.clone(),
            mapping: system.mapping.clone(),
            dual_scales,
            quadrature: system
                .assembly
                .input
                .forms
                .iter()
                .map(|(_, rule)| rule.clone())
                .collect(),
        });
        Ok(())
    })
}

#[test]
fn capture_scope_disarms_on_drop_and_unwind_without_a_solve() {
    drop(CaptureGuard::start());
    let failed = std::panic::catch_unwind(|| {
        let _guard = CaptureGuard::start();
        panic!("capture cleanup witness");
    });
    assert!(failed.is_err());
    let no_occurrence = std::panic::catch_unwind(|| CaptureGuard::start().finish());
    assert!(no_occurrence.is_err());
    drop(CaptureGuard::start());
}
