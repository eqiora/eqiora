//! Exact Result replay and typed observations. The viewer performs no scientific arithmetic.
use eqiora::{
    ScalarDomain, ValueLiteral, ValueType,
    api::{MathRendering, ModelDocument},
    backends::{diffsol::DIFFSOL_TIME_BACKEND, faer::FaerLinearSolver},
    kernel::KernelNode,
    language::NotationProfile,
};
use eqiora_numerics::{CommonResult, ResolvedCommonPlan};
use serde_json::{Value, json};
use std::collections::HashMap;

pub(super) fn project(
    bytes: &[u8],
    plan: &[u8],
    selected: &ModelDocument,
) -> Result<Value, String> {
    let plan = ResolvedCommonPlan::from_bytes(plan, &FaerLinearSolver, DIFFSOL_TIME_BACKEND)
        .map_err(|error| format!("Cannot validate Result Plan: {}", error.message()))?;
    if plan.model_digest()
        != selected
            .digest()
            .map_err(|error| error.message().to_owned())?
    {
        return Err(
            "Result belongs to a different exact Model artifact; select its unchanged source"
                .to_owned(),
        );
    }
    let result = CommonResult::from_bytes(bytes, &plan)
        .map_err(|error| format!("Cannot validate Result: {}", error.message()))?;
    let mut observations = Vec::new();
    let mut remaining = 4096;
    for node in selected.program().nodes() {
        let KernelNode::Observable(observable) = node else {
            continue;
        };
        let id = observable.id();
        let names = selected
            .aliases()
            .iter()
            .filter(|(_, target)| **target == id.erase())
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>();
        let mut entry = json!({"id": id.to_string(), "names": names});
        match result.observe(plan.model_artifact(), id, &HashMap::new()) {
            Ok(observation) => {
                let value = observation.value();
                entry["valueType"] = json!(
                    MathRendering::value_type(value.value_type(), NotationProfile::Plain)
                        .map_err(|error| error.message().to_owned())?
                        .plain()
                );
                entry["shape"] = json!(
                    value
                        .value_type()
                        .shape()
                        .extents()
                        .iter()
                        .map(|n| n.get())
                        .collect::<Vec<_>>()
                );
                if value.component_count() > remaining {
                    entry["unsupported"] = json!(
                        "Observation exceeds the remaining 4096-component editor response budget"
                    );
                } else if value.components().is_none() {
                    entry["unsupported"] =
                        json!("Complex component projections require a real or complex value");
                } else {
                    remaining -= value.component_count();
                    entry["components"] = json!(
                        (0..value.component_count())
                            .map(|index| component(value, index))
                            .collect::<Vec<_>>()
                    );
                }
            }
            Err(error) => entry["unsupported"] = json!(error.message()),
        }
        observations.push(entry);
    }
    Ok(json!({
        "version": 1, "identity": result.identity(), "planIdentity": plan.identity(),
        "modelDigest": plan.model_digest(), "observations": observations,
        "interpretation": "Mathematical components in the declared basis. No peak/RMS phasor, power, probability or spectral convention is inferred. Author physical quantities as typed Observables.",
        "phaseConvention": "Principal argument in radians, undefined at exact zero. No threshold or unwrapping.",
        "modal": "Modal Result projections are unavailable; no normalization, phase reference or mode is inferred."
    }))
}

fn component(value: &ValueLiteral, index: usize) -> Value {
    let mut component = json!({"index": index});
    for (name, projection) in [
        ("real", value.component_real(index).map(Some)),
        ("imaginary", value.component_imaginary(index).map(Some)),
        ("magnitude", value.component_magnitude(index).map(Some)),
        (
            "squaredMagnitude",
            value.component_squared_magnitude(index).map(Some),
        ),
        ("phase", value.component_phase(index)),
    ] {
        component[name] = match projection {
            Ok(Some(value)) => {
                let unit = ValueType::scalar(ScalarDomain::Real, value.dim());
                match unit
                    .ok()
                    .and_then(|ty| MathRendering::value_type(&ty, NotationProfile::Plain).ok())
                {
                    Some(unit) => {
                        json!({"value": value.value(), "unit": if name == "phase" { "rad" } else { unit.plain() }})
                    }
                    None => json!({"unavailable": "Cannot render the derived dimension"}),
                }
            }
            Ok(None) => json!({"undefined": "Phase is undefined at zero magnitude"}),
            Err(error) => json!({"unavailable": error.to_string()}),
        };
    }
    component
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora::{
        artifact::ModelEnvelope,
        solver::{LinearSolver, REFERENCE_LINEAR_SOLVER, ReductionPolicy, SolverPlan},
    };
    use eqiora_numerics::{CommonAlgebraicPlan, CommonLinearRequest, CommonSolvePolicy};
    use std::num::NonZeroUsize;

    fn fixture() -> (ModelDocument, Vec<u8>, Vec<u8>) {
        fixture_source(
            "model M(){variable z:complex<V>;relation r{math.complex(1,-2)*z=math.complex(11[V],-2[V]);}observable response:complex<V>=z; observable energy:V^2=math.abs2(z); observable zero:complex<V>=math.complex(0[V],0[V]);}",
        )
    }

    fn fixture_source(source: &str) -> (ModelDocument, Vec<u8>, Vec<u8>) {
        let document = ModelDocument::compile("complex.eqi", source).unwrap();
        let model = ModelEnvelope::from_program(document.program()).unwrap();
        let solver = SolverPlan::new(
            LinearSolver::BiConjugateGradientStabilized,
            1e-13,
            1e-15,
            NonZeroUsize::new(8).unwrap(),
        )
        .unwrap()
        .with_reduction(ReductionPolicy::Reproducible);
        let request = CommonSolvePolicy::Linear(
            CommonLinearRequest::exact(solver, REFERENCE_LINEAR_SOLVER.provider()).unwrap(),
        );
        let plan = CommonAlgebraicPlan::resolve(
            &model,
            request,
            None,
            &[],
            None,
            &REFERENCE_LINEAR_SOLVER,
        )
        .unwrap();
        let result = plan
            .run_result(&plan.initial_state(&[]).unwrap(), &REFERENCE_LINEAR_SOLVER)
            .unwrap();
        (
            document,
            result.plan().to_bytes().unwrap(),
            result.to_bytes().unwrap(),
        )
    }

    #[test]
    fn result_components_and_authored_observables_have_exact_lineage() {
        let (model, plan, result) = fixture();
        let view = project(&result, &plan, &model).unwrap();
        assert_eq!(view["version"], 1);
        assert_eq!(view["modelDigest"], model.digest().unwrap());
        let observations = view["observations"].as_array().unwrap();
        let named = |name: &str| {
            observations
                .iter()
                .find(|entry| entry["names"].as_array().unwrap().contains(&json!(name)))
                .unwrap()
        };
        let row = &named("response")["components"][0];
        // (1-2i)(3+4i)=11-2i; 3²+4²=25 and the 3-4-5 triangle fixes magnitude.
        for (name, expected) in [
            ("real", 3.0),
            ("imaginary", 4.0),
            ("magnitude", 5.0),
            ("squaredMagnitude", 25.0),
        ] {
            assert!((row[name]["value"].as_f64().unwrap() - expected).abs() < 1e-10);
        }
        assert_eq!(row["real"]["unit"], row["magnitude"]["unit"]);
        assert_eq!(
            row["squaredMagnitude"]["unit"],
            named("energy")["components"][0]["real"]["unit"]
        );
        assert_eq!(row["phase"]["unit"], "rad");
        assert!(named("zero")["components"][0]["phase"]["undefined"].is_string());
        assert!(
            named("zero")["components"][0]["phase"]
                .get("value")
                .is_none()
        );
        assert!(
            (named("energy")["components"][0]["real"]["value"]
                .as_f64()
                .unwrap()
                - 25.0)
                .abs()
                < 1e-10
        );
    }

    #[test]
    fn explicit_finite_wavefunction_probabilities_are_observables_not_viewer_inference() {
        let (model, plan, result) = fixture_source(
            "model State(){variable psi:array<complex<1>,2>;relation fixed{psi=[math.complex(0.6,0),math.complex(0,0.8)];}observable amplitude:array<complex<1>,2>=psi;observable probability:1=math.abs2(psi[0])+math.abs2(psi[1]);}",
        );
        let view = project(&result, &plan, &model).unwrap();
        let observed = view["observations"].as_array().unwrap();
        let named = |name: &str| {
            observed
                .iter()
                .find(|entry| entry["names"].as_array().unwrap().contains(&json!(name)))
                .unwrap()
        };
        let amplitude = &named("amplitude")["components"];
        for (index, expected) in [0.36, 0.64].into_iter().enumerate() {
            assert!(
                (amplitude[index]["squaredMagnitude"]["value"]
                    .as_f64()
                    .unwrap()
                    - expected)
                    .abs()
                    < 1e-12
            );
        }
        assert!(
            (named("probability")["components"][0]["real"]["value"]
                .as_f64()
                .unwrap()
                - 1.0)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn stale_model_invalid_result_and_foreign_plan_are_rejected() {
        let (model, plan, result) = fixture();
        assert!(
            project(b"{}", &plan, &model)
                .unwrap_err()
                .contains("Cannot validate Result")
        );
        let other =
            ModelDocument::compile("other.eqi", "model Other(){variable x:1;relation r{x=1;}}")
                .unwrap();
        assert!(
            project(&result, &plan, &other)
                .unwrap_err()
                .contains("different exact Model")
        );
        let mut wire: Value = serde_json::from_slice(&result).unwrap();
        wire["identity"] = json!("invented");
        assert!(project(&serde_json::to_vec(&wire).unwrap(), &plan, &model).is_err());
        assert!(
            project(&result, b"{}", &model)
                .unwrap_err()
                .contains("Cannot validate Result Plan")
        );
    }
}
