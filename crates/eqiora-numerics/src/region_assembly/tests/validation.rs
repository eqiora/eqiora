use eqiora_assembly::{AssemblyMap, AssemblyTarget, DofId};
use eqiora_core::{Id, entity::kinds};
use eqiora_meshing::ReferenceCell;

use super::*;

fn rejects(change: impl FnOnce(&mut Fixture), message: &str) {
    let mut fixture = Fixture::new(3, false);
    change(&mut fixture);
    let error = fixture.prepare().unwrap_err();
    assert!(
        error.message().contains(message),
        "{} does not name {message}",
        error.message()
    );
}

#[test]
fn preparation_rejects_inexact_cell_and_domain_form_coverage() {
    rejects(
        |f| {
            f.cells.pop();
        },
        "mesh cell coverage",
    );
    rejects(
        |f| {
            f.cells[1].index = 0;
        },
        "exactly once",
    );
    rejects(
        |f| {
            f.cells[0].index = 1000;
        },
        "exactly once",
    );
    rejects(
        |f| {
            f.forms.pop();
        },
        "exactly cover",
    );
    rejects(
        |f| {
            f.forms.push(f.forms[0].clone());
        },
        "duplicate assembly form",
    );
    rejects(
        |f| {
            f.domains[0] = Id::<kinds::Domain>::new().erase();
        },
        "exactly cover",
    );
    rejects(
        |f| {
            f.forms.push(Fixture::new(2, false).forms.remove(0));
        },
        "exactly cover",
    );
    rejects(
        |f| {
            f.cells[0].geometry =
                AffineGeometryMap::new(ReferenceCell::simplex(1).unwrap(), 1, vec![0.0], vec![0.5])
                    .unwrap();
        },
        "cell reference differs",
    );
    rejects(
        |f| {
            f.forms[0].1 = QuadratureRule::tensor_product_gauss_legendre(1, 1).unwrap();
        },
        "basis-product degree",
    );
}

#[test]
fn preparation_rejects_inexact_history_and_algebraic_maps() {
    rejects(
        |f| {
            f.cells[0].previous.clear();
        },
        "exact consumed Field coverage",
    );
    rejects(
        |f| {
            f.cells[0].previous.values_mut().next().unwrap().pop();
        },
        "exact consumed Field coverage",
    );
    rejects(
        |f| {
            f.cells[0].previous.values_mut().next().unwrap()[0] = f64::NAN;
        },
        "exact consumed Field coverage",
    );
    rejects(
        |f| {
            f.cells[0]
                .previous
                .insert(Id::<kinds::Field>::new().erase(), vec![0.0; 2]);
        },
        "exact consumed Field coverage",
    );
    rejects(
        |f| {
            f.cells[0].mappings = [].into();
        },
        "target map",
    );
    rejects(
        |f| {
            let duplicate = f.cells[0].mappings[0].clone();
            let mut mappings = f.cells[0].mappings.to_vec();
            mappings.push(duplicate);
            f.cells[0].mappings = mappings.into();
        },
        "duplicate assembly targets",
    );
    rejects(
        |f| {
            std::sync::Arc::make_mut(&mut f.cells[0].mappings)[0] = TargetAssemblyMap::new(
                f.plan.target_id(0).unwrap(),
                AssemblyMap::new(vec![], vec![]).unwrap(),
            );
        },
        "exact local Field ranges",
    );
    rejects(
        |f| {
            let map = f.cells[0].mappings[0].map();
            let mut equations = map.equations().to_vec();
            equations[0] = Some(DofId::new(1000));
            std::sync::Arc::make_mut(&mut f.cells[0].mappings)[0] = TargetAssemblyMap::new(
                f.plan.target_id(0).unwrap(),
                AssemblyMap::new(equations, map.unknowns().to_vec()).unwrap(),
            );
        },
        "global DOF",
    );
    rejects(
        |f| {
            let map = f.cells[0].mappings[0].map();
            let mut unknowns = map.unknowns().to_vec();
            unknowns[0] = LocalUnknown::Free(DofId::new(1000));
            std::sync::Arc::make_mut(&mut f.cells[0].mappings)[0] = TargetAssemblyMap::new(
                f.plan.target_id(0).unwrap(),
                AssemblyMap::new(map.equations().to_vec(), unknowns).unwrap(),
            );
        },
        "global DOF",
    );
    rejects(
        |f| {
            let other_plan = AssemblyPlan::new(vec![AssemblyTarget::new(1).unwrap(); 3]).unwrap();
            std::sync::Arc::make_mut(&mut f.cells[0].mappings)[0] = TargetAssemblyMap::new(
                other_plan.target_id(2).unwrap(),
                f.cells[0].mappings[0].map().clone(),
            );
        },
        "outside the assembly Plan",
    );
}
