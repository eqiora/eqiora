//! Independent affine change-of-variables values through the ordinary Plan/Result path.
use eqiora_artifact::ModelEnvelope;
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_compiler::{CompiledModel, StaticBindingValue};
use eqiora_core::{DimExponents, DynQuantity};
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_meshing::QuadratureRule;
use eqiora_numerics::{CommonAlgebraicPlan, CommonLinearRequest, CommonSolvePolicy};
use eqiora_schema::kernel::AxisBounds;
use eqiora_solver::{LinearSolver, LinearSolverBackend, ReductionPolicy, SolverPlan};
use std::{collections::HashMap, num::NonZeroUsize};

#[test]
fn accepted_result_integrates_affine_pullback_with_absolute_volume_factor() {
    let source = r#"model Affine(support a:interval(m),support b:interval(m),support c:interval(m),support d:interval(m)) {
        support reference:product(a,b);
        support body:product(c,d);
        coordinate xi:m on reference from a;
        coordinate eta:m on reference from b;
        coordinate x:m on body from c;
        coordinate y:m on body from d;
        variable anchor:1;
        relation retained {anchor=2;}
        observable transformed:m^4=integral(
            pullback(x*x+x*y,from=(xi,eta),at=(x=2*xi+eta,y=3*eta))
            *volume_jacobian(from=(xi,eta),at=(x=2*xi+eta,y=3*eta)),measure(reference));
        observable omitted:m^4=integral(
            pullback(x*x+x*y,from=(xi,eta),at=(x=2*xi+eta,y=3*eta)),measure(reference));
        observable reflected:m^4=integral(
            pullback(x*x+x*y,from=(xi,eta),at=(x=2[m]-2*xi+eta,y=3*eta))
            *volume_jacobian(from=(xi,eta),at=(x=2[m]-2*xi+eta,y=3*eta)),measure(reference));
        observable rotated:m^4=integral(
            pullback(x*x+x*y,from=(xi,eta),at=(x=1[m]-eta,y=xi))
            *volume_jacobian(from=(xi,eta),at=(x=1[m]-eta,y=xi)),measure(reference));
        observable physical_x:m=evaluate(partial(x*x+x*y,wrt=x),at=(x=1[m],y=1.5[m]));
        observable physical_y:m=evaluate(partial(x*x+x*y,wrt=y),at=(x=1[m],y=1.5[m]));
        observable mapped_x:m=evaluate(
            partial(pullback(x*x+x*y,from=(xi,eta),at=(x=2*xi+eta,y=3*eta)),wrt=xi)/2,
            at=(xi=0.25[m],eta=0.5[m]));
        observable mapped_y:m=evaluate((
            partial(pullback(x*x+x*y,from=(xi,eta),at=(x=2*xi+eta,y=3*eta)),wrt=eta)
            -partial(pullback(x*x+x*y,from=(xi,eta),at=(x=2*xi+eta,y=3*eta)),wrt=xi)/2)/3,
            at=(xi=0.25[m],eta=0.5[m]));
        observable bound:m^4=integral(
            pullback(x*x+x*y,from=(xi,eta),at=(x=anchor*xi+eta,y=3*eta))
            *volume_jacobian(from=(xi,eta),at=(x=anchor*xi+eta,y=3*eta)),measure(reference));
        observable singular:m^2=integral(
            volume_jacobian(from=(xi,eta),at=(x=xi+eta,y=xi+eta)),measure(reference));
        observable nonlinear:m^2=integral(
            volume_jacobian(from=(xi,eta),at=(x=xi+xi*xi/2[m],y=eta)),measure(reference));
        observable pole:1=integral(
            pullback(1/(x-0.5[m])^2,from=(xi,eta),at=(x=xi,y=eta)),measure(reference));
    }"#;
    // I(k)=3k*(k²/3+5k/4+4/3)=k³+(15/4)k²+4k.
    // Two independently solved Models must not reuse the first map coefficient.
    for (scale, bound_integral) in [(2, 31.0), (1, 35.0 / 4.0)] {
        let source = source.replace("anchor=2", &format!("anchor={scale}"));
        let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
        let interval = |upper| {
            StaticBindingValue::CoordinateInterval(
                AxisBounds::new(
                    DynQuantity::new(0.0, length),
                    DynQuantity::new(upper, length),
                )
                .unwrap(),
            )
        };
        let compiled = CompiledModel::compile_selected(
            "affine.eqi",
            &source,
            "Affine",
            &[
                ("a", interval(1.0)),
                ("b", interval(1.0)),
                ("c", interval(3.0)),
                ("d", interval(3.0)),
            ],
        )
        .unwrap();
        let symbols = compiled.symbols().clone();
        let (transaction, model_id, _) = compiled.into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program =
            eqiora_sem::KernelProgram::from_snapshot(&store.snapshot(), model_id).unwrap();
        let density = symbols.get("transformed").unwrap().downcast().unwrap();
        let coordinates = vec![
            (
                (symbols.get("a").unwrap(), 0),
                DynQuantity::new(0.25, length),
            ),
            (
                (symbols.get("b").unwrap(), 0),
                DynQuantity::new(0.5, length),
            ),
        ];
        let point = eqiora_sem::EvaluationPoint::new(
            &program,
            symbols.get("reference").unwrap().downcast().unwrap(),
            coordinates.clone(),
            None,
        )
        .unwrap();
        // u(1,3/2)=5/2 m² and |det J|=6, hence the density is 15 m².
        let sampled = program
            .evaluate_observable_density_with_points(density, &point, &mut |_, _| {
                panic!("coordinate-only density")
            })
            .unwrap()
            .real_scalar_value()
            .unwrap();
        assert!((sampled.value() - 15.0).abs() <= 64.0 * f64::EPSILON * 15.0);
        assert_eq!(
            sampled.dim(),
            DimExponents::from_integers([0, 2, 0, 0, 0, 0, 0]).unwrap()
        );
        let foreign = eqiora_sem::EvaluationPoint::new(
            &program,
            symbols.get("body").unwrap().downcast().unwrap(),
            vec![
                (
                    (symbols.get("c").unwrap(), 0),
                    DynQuantity::new(1.0, length),
                ),
                (
                    (symbols.get("d").unwrap(), 0),
                    DynQuantity::new(1.5, length),
                ),
            ],
            None,
        )
        .unwrap();
        assert!(
            program
                .evaluate_observable_density_with_points(density, &foreign, &mut |_, _| panic!(
                    "foreign point must reject before resolving"
                ))
                .unwrap_err()
                .message()
                .contains("exact integral input support")
        );
        for invalid in [vec![coordinates[0]], vec![coordinates[0], coordinates[0]]] {
            assert!(
                eqiora_sem::EvaluationPoint::new(&program, point.domain(), invalid, None).is_err()
            );
        }
        let original = ModelEnvelope::from_program(&program).unwrap();
        let model =
            ModelEnvelope::from_json(&original.canonical_json().unwrap(), Default::default())
                .unwrap();
        let plan = CommonAlgebraicPlan::resolve(
            &model,
            CommonSolvePolicy::Linear(
                CommonLinearRequest::exact(
                    SolverPlan::new(
                        LinearSolver::SparseLu,
                        1e-12,
                        1e-12,
                        NonZeroUsize::new(10).unwrap(),
                    )
                    .unwrap()
                    .with_reduction(ReductionPolicy::Fast),
                    FaerLinearSolver.provider(),
                )
                .unwrap(),
            ),
            None,
            None,
            &FaerLinearSolver,
        )
        .unwrap();
        let result = plan
            .run_result(&plan.initial_state(&[]).unwrap(), &FaerLinearSolver)
            .unwrap();
        let rules = HashMap::from([(
            symbols.get("reference").unwrap().downcast().unwrap(),
            QuadratureRule::tensor_product_gauss_legendre(2, 2).unwrap(),
        )]);
        // On [0,1]² the pullback is 4xi²+10xi*eta+4eta², whose integral
        // is 4/3+5/2+4/3=31/6. The sheared image has |det J|=6, giving 31.
        // Reflection xi -> 1-xi covers the same image with opposite orientation.
        // Rotation (x,y)=(1-eta,xi) preserves the unit square: integral(x²+xy)=1/3+1/4.
        // Two-point Gauss is exact for each polynomial degree; 256 eps covers
        // floating-point evaluation, LU/log-exp and the four-point accumulation.
        let unit = DimExponents::from_integers([0, 4, 0, 0, 0, 0, 0]).unwrap();
        for (name, expected) in [
            ("transformed", 31.0),
            ("omitted", 31.0 / 6.0),
            ("reflected", 31.0),
            ("bound", bound_integral),
            ("rotated", 7.0 / 12.0),
        ] {
            let observation = result
                .observe(
                    &model,
                    symbols.get(name).unwrap().downcast().unwrap(),
                    &rules,
                )
                .unwrap();
            let actual = observation.value().real_scalar_value().unwrap();
            assert_eq!(actual.dim(), unit);
            assert!(
                (actual.value() - expected).abs() <= 256.0 * f64::EPSILON * expected.abs(),
                "{name}: {actual:?}"
            );
        }
        // J=[2,1;0,3], so J^-T*[7,13/2]=[7/2,1] at (xi,eta)=(1/4,1/2).
        // This agrees with [2x+y,x] at (x,y)=(1,3/2).
        for (name, expected) in [
            ("physical_x", 3.5),
            ("mapped_x", 3.5),
            ("physical_y", 1.0),
            ("mapped_y", 1.0),
        ] {
            let observation = result
                .observe(
                    &model,
                    symbols.get(name).unwrap().downcast().unwrap(),
                    &HashMap::new(),
                )
                .unwrap();
            assert_eq!(
                observation.value().real_scalar_value().unwrap(),
                DynQuantity::new(expected, length)
            );
        }
        for (name, gate) in [
            ("singular", "singular"),
            ("nonlinear", "affine"),
            ("pole", "regular factor density"),
        ] {
            let error = result
                .observe(
                    &model,
                    symbols.get(name).unwrap().downcast().unwrap(),
                    &rules,
                )
                .unwrap_err();
            assert!(error.message().contains(gate), "{name}: {error:?}");
        }
        let typed = program
            .typed_observable(symbols.get("bound").unwrap().downcast().unwrap())
            .unwrap();
        let error = eqiora_ir::ScalarOperatorIr::lower_affine_map_density(&typed, &mut |_| {
            Some(eqiora_core::ValueLiteral::try_from(DynQuantity::new(2.0, length)).unwrap())
        })
        .unwrap_err();
        assert!(error.message().contains("retained type"), "{error:?}");
    }
}
