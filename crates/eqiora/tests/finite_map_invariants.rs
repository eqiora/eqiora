//! Local finite-map invariants preserve nominal spaces, units and regularity.
use eqiora::api::ModelDocument;
use eqiora_schema::kernel::SymbolRef;

fn source(outputs: &str) -> String {
    format!(
        r#"
space Channels=orthonormal(first,second);
component Local(parameter a:map<1,Channels,Channels>, parameter x:coordinates<V,Channels>) {{
    variable anchor:1; relation fixed {{anchor=1;}}
    {outputs}
}}
model M() {{
    instance material:Local(a=linear_map(Channels,Channels,[[2,3],[5,7]]),x=coordinates(Channels,[11[V],13[V]]));
}}
"#
    )
}

fn observation(document: &ModelDocument, name: &str) -> eqiora_core::ValueLiteral {
    let program = document.program();
    let id = document.aliases()[&format!("material.{name}")]
        .downcast()
        .unwrap();
    program
        .evaluate_finite_observable(id, &mut |symbol| match symbol {
            SymbolRef::Parameter(id) => program.typed_value(id.erase()).cloned(),
            _ => None,
        })
        .unwrap()
}

#[test]
fn existing_nonsymmetric_map_action_reaches_typed_evaluation() {
    let document = ModelDocument::compile(
        "map-action.eqi",
        &source("observable mapped:coordinates<V,Channels>=apply(a,x);"),
    )
    .unwrap();
    let value = observation(&document, "mapped");
    // [[2,3],[5,7]]*[11,13]=[61,146], independently by row multiplication.
    assert_eq!(value.component(0).unwrap().0, 61.0);
    assert_eq!(value.component(1).unwrap().0, 146.0);
}

#[test]
fn nonsymmetric_local_inverse_and_invariants_have_an_ordinary_source_path() {
    let document = ModelDocument::compile(
        "local-map.eqi",
        &source(
            r#"
    observable determinant_value:1=determinant(a);
    observable trace_value:1=matrix_trace(a);
    observable recovered:coordinates<V,Channels>=apply(inverse(a),apply(a,x));
"#,
        ),
    )
    .unwrap();
    for (name, expected) in [("determinant_value", -1.0), ("trace_value", 9.0)] {
        let value = observation(&document, name);
        assert!((value.real_scalar_value().unwrap().value() - expected).abs() < 1e-12);
    }
    // Hand inversion: [[2,3],[5,7]]^-1=[[-7,3],[5,-2]], no symmetry assumption.
    let value = observation(&document, "recovered");
    assert_eq!(value.component_count(), 2);
    for (i, expected) in [11.0, 13.0].into_iter().enumerate() {
        assert!((value.component(i).unwrap().0 - expected).abs() < 1e-12);
    }
}

#[test]
fn inverse_direction_and_adjoint_follow_an_independent_quotient_derivative() {
    use eqiora_ir::{
        ComponentScalarization, DifferentiationRole, LinearizedRelation, RelationCotangent,
        RelationTangent,
    };
    let document = ModelDocument::compile(
        "inverse-actions.eqi",
        r#"
space Channels=orthonormal(first,second);
model M() {
    variable anchor:1; relation fixed {anchor=1;}
    parameter a:map<1,Channels,Channels>=linear_map(Channels,Channels,[[2,3],[5,7]]);
    observable inverse_map:map<1,Channels,Channels>=inverse(a);
}
"#,
    )
    .unwrap();
    let id = document.aliases()["inverse_map"].downcast().unwrap();
    let typed = document.program().typed_observable(id).unwrap();
    let rows = ComponentScalarization::lower(&typed).unwrap();
    // For A(t)=[[2+t,3],[5,7]], inverse entries are [7,-3,-5,2+t]/(7t-1).
    // Direct quotient differentiation at zero gives [-49,21,35,-15].
    let expected = [-49.0, 21.0, 35.0, -15.0];
    let cotangent = [1.0, 2.0, 3.0, 4.0];
    for scale in [-3.0, 1e-6, 1.0, 1e6] {
        let mut total_adjoint = [0.0; 4];
        let mut pairing = 0.0;
        for (row, (expected, seed)) in rows.rows().iter().zip(expected.into_iter().zip(cotangent)) {
            let components = row
                .symbols()
                .iter()
                .map(|coordinate| {
                    let [r, c] = coordinate.component_index() else {
                        panic!("matrix coordinate");
                    };
                    (*r as usize) * 2 + *c as usize
                })
                .collect::<Vec<_>>();
            assert_eq!(
                components.len(),
                4,
                "the inverse must retain four independent Parameter coordinates"
            );
            let values = components
                .iter()
                .map(|i| scale * [2.0, 3.0, 5.0, 7.0][*i])
                .collect::<Vec<_>>();
            let roles = vec![DifferentiationRole::Parameter; values.len()];
            let bound = row.linearize(&values, &roles).unwrap();
            let direction = components
                .iter()
                .map(|i| if *i == 0 { 1.0 } else { 0.0 })
                .collect::<Vec<_>>();
            let mut tangent = [0.0];
            bound
                .jvp(RelationTangent::Parameter(&direction), &mut tangent)
                .unwrap();
            assert!(
                (tangent[0] * scale * scale - expected).abs() < 1e-10,
                "scale={scale}, tangent={}, expected={expected}",
                tangent[0]
            );
            pairing += tangent[0] * seed * scale * scale;
            let mut adjoint = vec![0.0; values.len()];
            bound
                .vjp(&[seed], RelationCotangent::Parameter(&mut adjoint))
                .unwrap();
            for (i, value) in components.iter().zip(adjoint) {
                total_adjoint[*i] += value * scale * scale;
            }
        }
        // G=[[1,2],[3,4]] pairs with the direct derivative above to give 38.
        assert!((pairing - 38.0).abs() < 1e-10);
        for (actual, expected) in total_adjoint.into_iter().zip([38.0, -28.0, -15.0, 11.0]) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }
}

#[test]
fn inverse_rejects_singularity_and_reports_its_conditioning_policy() {
    for (matrix, gate) in [
        ("[[1,2],[2,4]]", "singular or numerically unresolved"),
        ("[[1,0],[0,1e-18]]", "reciprocal infinity-norm estimate"),
    ] {
        let source = source("observable inverse_map:map<1,Channels,Channels>=inverse(a);")
            .replace("[[2,3],[5,7]]", matrix);
        let document = ModelDocument::compile("inverse-rejection.eqi", &source).unwrap();
        let program = document.program();
        let id = document.aliases()["material.inverse_map"]
            .downcast()
            .unwrap();
        let error = program
            .evaluate_finite_observable(id, &mut |symbol| match symbol {
                SymbolRef::Parameter(id) => program.typed_value(id.erase()).cloned(),
                _ => None,
            })
            .unwrap_err();
        assert!(error.message().contains(gate), "{error:?}");
    }
}

#[test]
fn dimensioned_constitutive_map_and_inverse_action_use_the_ordinary_solver() {
    use eqiora::artifact::ModelEnvelope;
    use eqiora::solver::{LinearSolver, LinearSolverBackend, ReductionPolicy, SolverPlan};
    use eqiora_backend_faer::FaerLinearSolver;
    use eqiora_numerics::{
        CommonAlgebraicPlan, CommonLinearRequest, CommonResult, CommonSolvePolicy,
        ResolvedCommonPlan,
    };
    use std::{collections::HashMap, num::NonZeroUsize};
    let base_source = r#"
space Inputs=orthonormal(q0,q1); space Outputs=orthonormal(s0,s1);
model M() {
    parameter a:map<Pa,Inputs,Outputs>=linear_map(Inputs,Outputs,[[2[Pa],3[Pa]],[5[Pa],7[Pa]]]);
    parameter stress:coordinates<Pa,Outputs>=coordinates(Outputs,[61[Pa],146[Pa]]);
    variable strain:coordinates<1,Inputs>;
    relation constitutive {apply(a,strain)=stress;}
    observable inverse_action:coordinates<1,Inputs>=apply(inverse(a),stress);
    observable solved:coordinates<1,Inputs>=strain;
}
"#;
    for equation in ["apply(a,strain)=stress", "strain=apply(inverse(a),stress)"] {
        let source = base_source.replace("apply(a,strain)=stress", equation);
        let document = ModelDocument::compile("constitutive-map.eqi", &source).unwrap();
        let model = ModelEnvelope::from_program(document.program()).unwrap();
        let model =
            ModelEnvelope::from_json(&model.canonical_json().unwrap(), Default::default()).unwrap();
        let policy = CommonSolvePolicy::Linear(
            CommonLinearRequest::exact(
                SolverPlan::new(
                    LinearSolver::SparseLu,
                    1e-12,
                    1e-14,
                    NonZeroUsize::new(8).unwrap(),
                )
                .unwrap()
                .with_reduction(ReductionPolicy::Fast),
                FaerLinearSolver.provider(),
            )
            .unwrap(),
        );
        let plan =
            CommonAlgebraicPlan::resolve(&model, policy, None, None, &FaerLinearSolver).unwrap();
        let result = plan
            .run_result(&plan.initial_state(&[]).unwrap(), &FaerLinearSolver)
            .unwrap();
        let replay = ResolvedCommonPlan::from_bytes(
            &result.plan().to_bytes().unwrap(),
            &FaerLinearSolver,
            eqiora::time::TimeBackendIdentity::new("eqiora.test.time", "1"),
        )
        .unwrap();
        let result = CommonResult::from_bytes(&result.to_bytes().unwrap(), &replay).unwrap();
        for name in ["inverse_action", "solved"] {
            let id = document.aliases()[name].downcast().unwrap();
            let value = result.observe(&model, id, &HashMap::new()).unwrap();
            assert_eq!(
                value.value().value_type().dimension(),
                eqiora::DimExponents::DIMENSIONLESS
            );
            for (i, expected) in [11.0, 13.0].into_iter().enumerate() {
                assert!((value.value().component(i).unwrap().0 - expected).abs() < 1e-11);
            }
        }
        // Same cardinality cannot replace the inverse's exact target (the original Inputs).
        let wrong = source.replace(
            "inverse_action:coordinates<1,Inputs>",
            "inverse_action:coordinates<1,Outputs>",
        );
        assert!(ModelDocument::compile("foreign-inverse.eqi", &wrong).is_err());
        // Trace/determinant have no implicit identification between the distinct spaces.
        for (operation, unit) in [("matrix_trace", "Pa"), ("determinant", "Pa^2")] {
            let wrong = source.replace(
                "observable solved:coordinates<1,Inputs>=strain;",
                &format!("observable forbidden:{unit}={operation}(a);"),
            );
            assert!(ModelDocument::compile("foreign-invariant.eqi", &wrong).is_err());
        }
    }
}

#[test]
fn identity_and_dimensioned_invariants_use_closed_nominal_constructors() {
    let source = r#"
space S=orthonormal(x,y); space T=orthonormal(x,y);
model M() {
    variable anchor:1; relation fixed {anchor=1;}
    let a:map<Pa,S,S>=linear_map(S,S,[[2[Pa],3[Pa]],[5[Pa],7[Pa]]]);
    let unit:map<1,S,S>=identity(S);
    let d:Pa^2=determinant(a);
    let t:Pa=matrix_trace(a);
    let inv:map<1/Pa,S,S>=inverse(a);
    let product:map<Pa,S,S>=compose(a,unit);
    observable determinant_value:Pa^2=d;
    observable trace_value:Pa=t;
    observable inverse_map:map<1/Pa,S,S>=inv;
    observable composed:map<Pa,S,S>=product;
}

"#;
    let document = ModelDocument::compile("map-units.eqi", source).unwrap();
    let pressure = eqiora::DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).unwrap();
    for (name, expected, dimension) in [
        ("determinant_value", vec![-1.0], pressure.pow(2, 1).unwrap()),
        ("trace_value", vec![9.0], pressure),
        (
            "inverse_map",
            vec![-7.0, 3.0, 5.0, -2.0],
            pressure.pow(-1, 1).unwrap(),
        ),
        ("composed", vec![2.0, 3.0, 5.0, 7.0], pressure),
    ] {
        let id = document.aliases()[name].downcast().unwrap();
        let value = document
            .program()
            .evaluate_finite_observable(id, &mut |_| None)
            .unwrap();
        assert_eq!(value.value_type().dimension(), dimension);
        for (i, expected) in expected.into_iter().enumerate() {
            assert!((value.component(i).unwrap().0 - expected).abs() < 1e-12);
        }
    }
    for wrong in [
        source.replace("identity(S)", "identity(T)"),
        source.replace("let unit:map<1,S,S>", "let unit:map<Pa,S,S>"),
    ] {
        assert!(ModelDocument::compile("wrong-identity.eqi", &wrong).is_err());
    }
}

#[test]
fn finite_map_size_is_not_the_physical_space_dimension() {
    use eqiora_ir::{
        ComponentScalarization, DifferentiationRole, LinearizedRelation, RelationCotangent,
        RelationTangent,
    };
    // B = I + u 1^T, u_i=i+1. On the subspace 1^T x=0 it is identity;
    // on u its eigenvalue is d=1+sum(u). Thus det(B)=d and direct multiplication
    // gives B^-1=I-u 1^T/d. Cyclically permuting its rows tests pivoting and sign.
    // These formulas are independent of a factorization or a computed inverse.
    for n in [1usize, 2, 3, 4, 5, 6, 9, 16] {
        let d = (1 + n * (n + 1) / 2) as f64;
        let determinant = if n % 2 == 0 { -d } else { d };
        let coefficient = |r: usize, c: usize| {
            let p = (r + 1) % n;
            (p + 1 + usize::from(p == c)) as f64
        };
        let inverse = |r: usize, c: usize| f64::from(r == (c + 1) % n) - (r + 1) as f64 / d;
        let basis = (0..n)
            .map(|i| format!("q{i}"))
            .collect::<Vec<_>>()
            .join(",");
        let matrix = (0..n)
            .map(|r| {
                format!(
                    "[{}]",
                    (0..n)
                        .map(|c| coefficient(r, c).to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let document = ModelDocument::compile(
            "general-finite-map.eqi",
            &format!(
                r#"
space S=orthonormal({basis});
model M() {{
    variable anchor:1; relation fixed {{anchor=1;}}
    parameter a:map<1,S,S>=linear_map(S,S,[{matrix}]);
    observable determinant_value:1=determinant(a);
    observable inverse_map:map<1,S,S>=inverse(a);
}}
"#
            ),
        )
        .unwrap();
        for name in ["determinant_value", "inverse_map"] {
            let id = document.aliases()[name].downcast().unwrap();
            let typed = document.program().typed_observable(id).unwrap();
            let rows = ComponentScalarization::lower(&typed)
                .unwrap_or_else(|error| panic!("{n}x{n} {name}: {error:?}"));
            for (output, row) in rows.rows().iter().enumerate() {
                let coordinates = row
                    .symbols()
                    .iter()
                    .map(|s| {
                        let [r, c] = s.component_index() else {
                            panic!("matrix coordinate")
                        };
                        (*r as usize, *c as usize)
                    })
                    .collect::<Vec<_>>();
                let values = coordinates
                    .iter()
                    .map(|&(r, c)| coefficient(r, c))
                    .collect::<Vec<_>>();
                let roles = vec![DifferentiationRole::Parameter; values.len()];
                let bound = row.linearize(&values, &roles).unwrap();
                let mut primal = [0.0];
                bound.primal(&mut primal).unwrap();
                let expected = if name == "determinant_value" {
                    determinant
                } else {
                    inverse(output / n, output % n)
                };
                assert!(
                    (primal[0] - expected).abs() < 1e-9,
                    "{n}x{n} {name}[{output}]: {} vs {expected}",
                    primal[0]
                );
                // H=E00: the inverse derivative is the rank-one outer product of
                // the closed-form inverse's first column and first row, negated.
                let direction = coordinates
                    .iter()
                    .map(|&p| f64::from(p == (0, 0)))
                    .collect::<Vec<_>>();
                let expected_derivative = if name == "determinant_value" {
                    determinant * inverse(0, 0)
                } else {
                    -inverse(output / n, 0) * inverse(0, output % n)
                };
                let mut tangent = [0.0];
                bound
                    .jvp(RelationTangent::Parameter(&direction), &mut tangent)
                    .unwrap();
                assert!((tangent[0] - expected_derivative).abs() < 1e-9);
                let mut adjoint = vec![0.0; values.len()];
                bound
                    .vjp(&[1.0], RelationCotangent::Parameter(&mut adjoint))
                    .unwrap();
                let pairing = adjoint
                    .iter()
                    .zip(&direction)
                    .map(|(g, h)| g * h)
                    .sum::<f64>();
                assert!((pairing - expected_derivative).abs() < 1e-9);
            }
        }
    }
}

#[test]
fn determinant_derivative_does_not_require_an_admitted_inverse() {
    use eqiora_ir::{
        ComponentScalarization, DifferentiationRole, LinearizedRelation, RelationCotangent,
    };
    for (matrix, expected) in [
        ("[[1,2],[2,4]]", [4.0_f64, -2.0, -2.0, 1.0]),
        ("[[1,0],[0,1e-320]]", [1e-320, 0.0, 0.0, 1.0]),
        ("[[0,0],[0,0]]", [0.0, 0.0, 0.0, 0.0]),
    ] {
        let source = format!(
            r#"
space S=orthonormal(x,y);
model M() {{
    variable anchor:1; relation fixed {{anchor=1;}}
    parameter a:map<1,S,S>=linear_map(S,S,{matrix});
    observable determinant_value:1=determinant(a);
}}
"#
        );
        let document = ModelDocument::compile("singular-determinant.eqi", &source).unwrap();
        let id = document.aliases()["determinant_value"].downcast().unwrap();
        let typed = document.program().typed_observable(id).unwrap();
        let scalar = ComponentScalarization::lower(&typed).unwrap();
        let row = &scalar.rows()[0];
        let values = match matrix {
            "[[1,2],[2,4]]" => [1.0, 2.0, 2.0, 4.0],
            "[[1,0],[0,1e-320]]" => [1.0, 0.0, 0.0, 1e-320],
            _ => [0.0; 4],
        };
        let coordinates = row
            .symbols()
            .iter()
            .map(|s| {
                let [r, c] = s.component_index() else {
                    panic!("matrix coordinate")
                };
                (*r as usize) * 2 + *c as usize
            })
            .collect::<Vec<_>>();
        let inputs = coordinates.iter().map(|&i| values[i]).collect::<Vec<_>>();
        let bound = row
            .linearize(&inputs, &[DifferentiationRole::Parameter; 4])
            .unwrap();
        let mut gradient = [0.0; 4];
        bound
            .vjp(&[1.0], RelationCotangent::Parameter(&mut gradient))
            .unwrap();
        // d(ad-bc)=[d,-c,-b,a], also at a singular matrix.
        for (actual, &i) in gradient.iter().zip(&coordinates) {
            let tolerance = if expected[i] == 0.0 {
                0.0
            } else {
                expected[i].abs() * 1e-12
            };
            assert!(
                (actual - expected[i]).abs() <= tolerance,
                "{matrix} coordinate {i}: {actual}"
            );
        }
    }
}

#[test]
fn inverse_admission_reason_survives_affine_binding_and_plan_resolution() {
    use eqiora::artifact::ModelEnvelope;
    use eqiora::solver::{LinearSolver, LinearSolverBackend, ReductionPolicy, SolverPlan};
    use eqiora_backend_faer::FaerLinearSolver;
    use eqiora_ir::ComponentScalarization;
    use eqiora_numerics::{CommonAlgebraicPlan, CommonLinearRequest, CommonSolvePolicy};
    use std::num::NonZeroUsize;
    for (matrix, gate) in [
        ("[[1,2],[2,4]]", "singular or numerically unresolved"),
        ("[[1,0],[0,1e-18]]", "reciprocal infinity-norm estimate"),
    ] {
        let document = ModelDocument::compile(
            "inverse-plan-admission.eqi",
            &format!(
                r#"
space S=orthonormal(x,y);
model M() {{
    parameter a:map<1,S,S>=linear_map(S,S,{matrix});
    variable x:coordinates<1,S>;
    parameter rhs:coordinates<1,S>=coordinates(S,[1,1]);
    relation local {{x=apply(inverse(a),rhs);}}
    observable inverse_map:map<1,S,S>=inverse(a);
}}
"#
            ),
        )
        .unwrap();
        let id = document.aliases()["inverse_map"].downcast().unwrap();
        let typed = document.program().typed_observable(id).unwrap();
        let scalar = ComponentScalarization::lower(&typed).unwrap();
        let values = if matrix == "[[1,2],[2,4]]" {
            [1.0, 2.0, 2.0, 4.0]
        } else {
            [1.0, 0.0, 0.0, 1e-18]
        };
        let row = &scalar.rows()[0];
        let bindings = row
            .symbols()
            .iter()
            .map(|symbol| {
                let [r, c] = symbol.component_index() else {
                    panic!("matrix coordinate")
                };
                (symbol.clone(), values[(*r as usize) * 2 + *c as usize])
            })
            .collect::<Vec<_>>();
        let error = match row.bind_affine(&[], &bindings) {
            Err(error) => error,
            Ok(_) => panic!("inverse admission must reject"),
        };
        assert!(error.message().contains(gate), "{error:?}");
        let model = ModelEnvelope::from_program(document.program()).unwrap();
        let policy = CommonSolvePolicy::Linear(
            CommonLinearRequest::exact(
                SolverPlan::new(
                    LinearSolver::SparseLu,
                    1e-12,
                    1e-14,
                    NonZeroUsize::new(8).unwrap(),
                )
                .unwrap()
                .with_reduction(ReductionPolicy::Fast),
                FaerLinearSolver.provider(),
            )
            .unwrap(),
        );
        let error =
            match CommonAlgebraicPlan::resolve(&model, policy, None, None, &FaerLinearSolver) {
                Err(error) => error,
                Ok(_) => panic!("Plan must preserve inverse admission"),
            };
        assert!(error.message().contains(gate), "{error:?}");
    }
}

#[test]
fn determinant_underflow_is_not_reported_as_singularity() {
    let document = ModelDocument::compile(
        "determinant-underflow.eqi",
        &source("observable determinant_value:1=determinant(a);")
            .replace("[[2,3],[5,7]]", "[[1e-200,0],[0,1e-200]]"),
    )
    .unwrap();
    let id = document.aliases()["material.determinant_value"]
        .downcast()
        .unwrap();
    let error = document
        .program()
        .evaluate_finite_observable(id, &mut |symbol| match symbol {
            SymbolRef::Parameter(id) => document.program().typed_value(id.erase()).cloned(),
            _ => None,
        })
        .unwrap_err();
    assert!(
        error.message().contains("determinant underflows binary64"),
        "{error:?}"
    );
}

#[test]
fn complex_trace_and_contextual_identity_reuse_continuous_coefficients() {
    let document = ModelDocument::compile(
        "complex-identity.eqi",
        r#"
space S=orthonormal(x,y);
model M() {
    variable anchor:1; relation fixed {anchor=1;}
    parameter i:map<complex<1>,S,S>=identity(S);
    parameter a:map<complex<1>,S,S>=linear_map(S,S,[[math.complex(2,3),5],[7,math.complex(11,-1)]]);
    observable trace_value:complex<1>=matrix_trace(a);
    observable composed:map<complex<1>,S,S>=compose(a,i);
}
"#,
    )
    .unwrap();
    for (name, expected) in [
        ("trace_value", vec![(13.0, 2.0)]),
        (
            "composed",
            vec![(2.0, 3.0), (5.0, 0.0), (7.0, 0.0), (11.0, -1.0)],
        ),
    ] {
        let id = document.aliases()[name].downcast().unwrap();
        let value = document
            .program()
            .evaluate_finite_observable(id, &mut |symbol| match symbol {
                SymbolRef::Parameter(id) => document.program().typed_value(id.erase()).cloned(),
                _ => None,
            })
            .unwrap();
        for (i, expected) in expected.into_iter().enumerate() {
            assert_eq!(value.component(i), Some(expected));
        }
    }
}
