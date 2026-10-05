//! Finite Hamiltonians consume the shared eigen and time owners.
use eqiora::api::ModelDocument;
use eqiora::artifact::ModelEnvelope;
use eqiora::runtime::FirstOrderProgram;
use eqiora::time::{ImplicitMidpointTimeBackend, TimeMethod};
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_numerics::{
    CommonEigenPlan, CommonEigenRequest, CommonOdePlan, CommonOdePolicy, CommonOdeRunRequest,
    CommonTimeTolerance, CommonTrajectory,
};
use std::num::NonZeroUsize;

const SPACES: &str =
    "space A=orthonormal(up,down); space B=orthonormal(first,second,third); space AB=product(A,B);";
const PARAMETERS: &str = r#"
 parameter hbar:J*s=1[J*s];
 parameter ha:map<complex<J>,A,A>=linear_map(A,A,[[0,2[J]],[2[J],0]]);
 parameter hb:map<complex<J>,B,B>=linear_map(B,B,[[3[J],0,0],[0,0,0],[0,0,-3[J]]]);
 parameter ia:map<complex<1>,A,A>=linear_map(A,A,[[1,0],[0,1]]);
 parameter ib:map<complex<1>,B,B>=linear_map(B,B,[[1,0,0],[0,1,0],[0,0,1]]);
"#;

#[test]
fn dimensioned_hamiltonians_share_eigenpairs_and_time_states() {
    for hbar in [1., 2.] {
        let parameters = PARAMETERS.replace("hbar:J*s=1[J*s]", &format!("hbar:J*s={hbar}[J*s]"));
        for (basis, operator, initial, spectrum, center, active) in [
            ("A", "ha", "[math.complex(1,0),0]", vec![-2., 2.], 0., 1),
            (
                "AB",
                "tensor_product(ha,ib)+tensor_product(ia,hb)",
                "[math.complex(1,0),0,0,0,0,0]",
                vec![-5., -2., -1., 1., 2., 5.],
                3.,
                3,
            ),
            (
                "AB",
                "linear_map(AB,AB,[[3[J],0,0,2[J],0,0],[0,0,0,0,2[J],0],[0,0,-3[J],0,0,2[J]],[2[J],0,0,3[J],0,0],[0,2[J],0,0,0,0],[0,0,2[J],0,0,-3[J]]])",
                "[math.complex(1,0),0,0,0,0,0]",
                vec![-5., -2., -1., 1., 2., 5.],
                3.,
                3,
            ),
        ] {
            let spectral_source = format!(
                "{SPACES} model Spectrum(){{{parameters} variable u:coordinates<complex<1>,{basis}>; variable lambda:J; relation spectrum{{apply({operator},u)=lambda*u;}}}}"
            );
            let spectral = ModelDocument::compile("spectrum.eqi", &spectral_source).unwrap();
            let spectral_model = ModelEnvelope::from_program(spectral.program()).unwrap();
            let eigen = CommonEigenPlan::resolve(
                &spectral_model,
                CommonEigenRequest::dense(NonZeroUsize::new(spectrum.len()).unwrap(), 1e-12, 1e-12)
                    .unwrap(),
                &FaerLinearSolver,
            )
            .unwrap()
            .run_result(&FaerLinearSolver)
            .unwrap();
            assert_eq!(eigen.eigenpair_count(), spectrum.len());
            for (index, expected) in spectrum.iter().enumerate() {
                let (value, _, residual, normalization) = eigen.eigenpair(index).unwrap();
                let actual = value.component(0).unwrap();
                assert!((actual.0 - expected).abs() < 1e-11);
                assert_eq!(actual.1, 0.);
                assert!(residual < 1e-11 && normalization < 1e-11);
            }

            let source = format!(
                "{SPACES} model Evolution(){{{parameters} parameter H:map<complex<J>,{basis},{basis}>={operator}; state psi:coordinates<complex<1>,{basis}>; initial{{psi=coordinates({basis},{initial});}} relation flow{{derivative(psi)=math.complex(0,-1)/hbar*apply(H,psi);}} observable norm:1=math.real(pair(adjoint(psi),psi)); observable energy:J=math.real(pair(adjoint(psi),apply(H,psi)));}}"
            );
            let document = ModelDocument::compile("evolution.eqi", &source).unwrap();
            let model = ModelEnvelope::from_program(document.program()).unwrap();
            let flow = FirstOrderProgram::lower(
                document.program(),
                document.aliases()["flow"].downcast().unwrap(),
            )
            .unwrap();
            let generator = flow.constant_generator().unwrap();
            let dimension = flow.state_coordinates().len();
            for row in 0..dimension {
                for column in 0..dimension {
                    assert_eq!(
                        generator[row * dimension + column],
                        -generator[column * dimension + row]
                    );
                }
            }
            let step = 0.01;
            let end = 0.2;
            let plan = CommonOdePlan::resolve(
                &model,
                document.program(),
                CommonOdePolicy::new(
                    TimeMethod::ImplicitMidpoint,
                    step,
                    1e-12,
                    flow.state_coordinates()
                        .iter()
                        .map(|&c| CommonTimeTolerance::new(c, 1e-14).unwrap())
                        .collect(),
                )
                .unwrap()
                .with_hermitian_parameter(document.aliases()["H"].downcast().unwrap())
                .unwrap()
                .with_conserved_norm(
                    vec![document.aliases()["psi"].downcast().unwrap()],
                    eqiora::DynQuantity::new(1., eqiora::DimExponents::DIMENSIONLESS),
                    eqiora::DynQuantity::new(1e-10, eqiora::DimExponents::DIMENSIONLESS),
                )
                .unwrap(),
                ImplicitMidpointTimeBackend::CAPABILITIES,
            )
            .unwrap();
            let resolved = eqiora_numerics::ResolvedCommonPlan::Ode(Box::new(plan.clone()));
            let bytes = resolved.to_bytes().unwrap();
            assert_eq!(
                eqiora_numerics::ResolvedCommonPlan::from_bytes(
                    &bytes,
                    &FaerLinearSolver,
                    ImplicitMidpointTimeBackend::CAPABILITIES,
                )
                .unwrap(),
                resolved
            );
            let state = plan.initial_state(0.).unwrap();
            let request = CommonOdeRunRequest::new(plan, state, end, vec![end]).unwrap();
            let solution = ImplicitMidpointTimeBackend::new()
                .solve(&request.problem().unwrap(), request.time_plan())
                .unwrap();
            let trajectory = CommonTrajectory::accept_ode(request, solution).unwrap();
            let value = trajectory
                .ode_states()
                .unwrap()
                .last()
                .unwrap()
                .field_value(&model, document.aliases()["psi"].downcast().unwrap(), 0)
                .unwrap();
            // The initial first basis vector is the equal superposition of the
            // eigenvectors at center ± 2 J. Explicit hbar (J s) scales each frequency.
            // Cayley rotates each
            // by -2 N atan(lambda h/(2 hbar)), independently of the implementation.
            let phase = |lambda: f64| 2. * 20. * (lambda * step / (2. * hbar)).atan();
            let plus = phase(center + 2.);
            let minus = phase(center - 2.);
            for index in 0..value.component_count() {
                let expected = if index == 0 {
                    (
                        (plus.cos() + minus.cos()) / 2.,
                        -(plus.sin() + minus.sin()) / 2.,
                    )
                } else if index == active {
                    (
                        (plus.cos() - minus.cos()) / 2.,
                        -(plus.sin() - minus.sin()) / 2.,
                    )
                } else {
                    (0., 0.)
                };
                let actual = value.component(index).unwrap();
                assert!((actual.0 - expected.0).abs() < 1e-11);
                assert!((actual.1 - expected.1).abs() < 1e-11);
            }
            for (name, expected) in [("norm", 1.), ("energy", center)] {
                let value = trajectory
                    .observe_terminal(&model, document.aliases()[name].downcast().unwrap())
                    .unwrap();
                assert!((value.value().component(0).unwrap().0 - expected).abs() < 1e-11);
            }
        }
    }
}

#[test]
fn constant_generator_uses_structural_proof_and_common_mass_solve() {
    let model = |equations: &str| {
        let source = format!(
            "model Evolution(){{state x:1; state y:1; initial{{x=1;y=0;}} relation flow{{{equations}}}}}"
        );
        let document = ModelDocument::compile("generator.eqi", &source).unwrap();
        FirstOrderProgram::lower(
            document.program(),
            document.aliases()["flow"].downcast().unwrap(),
        )
        .unwrap()
    };
    // M=[[2,1],[1,2]], F=[[1,-2],[2,-1]]; M^-1 F=[[0,-1],[1,0]].
    let mass = model(
        "2*derivative(x)+derivative(y)=(x-2*y)/1[s]; derivative(x)+2*derivative(y)=(2*x-y)/1[s];",
    );
    assert_eq!(mass.constant_generator().unwrap(), [0., -1., 1., 0.]);
    for equations in [
        // x^3-x vanishes at all coordinate probes (0 and 1), but is nonlinear.
        "derivative(x)=(-y+x*x*x-x)/1[s]; derivative(y)=x/1[s];",
        "derivative(x)=(-y+1)/1[s]; derivative(y)=x/1[s];",
        "derivative(x)=-time()*y/1[s^2]; derivative(y)=x/1[s];",
        "derivative(x)=-y/1[s]+time()/1[s^2]; derivative(y)=x/1[s];",
    ] {
        assert!(model(equations).constant_generator().is_err());
    }
}

#[test]
fn conserved_norm_rejects_nonhermitian_flow_and_wrong_initial_normalization() {
    let build = |parameters: &str, amplitude: f64| {
        let source = format!(
            "{SPACES} model Evolution(){{{parameters} state psi:coordinates<complex<1>,A>; initial{{psi=coordinates(A,[math.complex({amplitude},0),0]);}} relation flow{{derivative(psi)=math.complex(0,-1)/hbar*apply(ha,psi);}}}}"
        );
        let document = ModelDocument::compile("norm.eqi", &source).unwrap();
        let model = ModelEnvelope::from_program(document.program()).unwrap();
        let flow = FirstOrderProgram::lower(
            document.program(),
            document.aliases()["flow"].downcast().unwrap(),
        )
        .unwrap();
        let policy = CommonOdePolicy::new(
            TimeMethod::ImplicitMidpoint,
            0.01,
            1e-12,
            flow.state_coordinates()
                .iter()
                .map(|&c| CommonTimeTolerance::new(c, 1e-14).unwrap())
                .collect(),
        )
        .unwrap()
        .with_hermitian_parameter(document.aliases()["ha"].downcast().unwrap())
        .unwrap()
        .with_conserved_norm(
            vec![document.aliases()["psi"].downcast().unwrap()],
            eqiora::DynQuantity::new(1., eqiora::DimExponents::DIMENSIONLESS),
            eqiora::DynQuantity::new(1e-10, eqiora::DimExponents::DIMENSIONLESS),
        )
        .unwrap();
        CommonOdePlan::resolve(
            &model,
            document.program(),
            policy,
            ImplicitMidpointTimeBackend::CAPABILITIES,
        )
    };
    let valid = build(PARAMETERS, 1.).unwrap();
    let state = valid.initial_state(0.).unwrap();
    assert_eq!(
        eqiora_numerics::CommonOdeState::from_bytes(&state.to_bytes().unwrap(), &valid).unwrap(),
        state
    );
    // The lower triangle is significant; treating only the upper triangle as H
    // would incorrectly accept this norm-changing generator.
    assert!(
        build(
            &PARAMETERS.replace("[[0,2[J]],[2[J],0]]", "[[0,2[J]],[1[J],0]]"),
            1.
        )
        .is_err()
    );
    assert!(build(PARAMETERS, 2.).unwrap().initial_state(0.).is_err());
}

#[test]
fn raw_hermitian_contract_rejects_coefficients_erased_by_evolution_scaling() {
    let parameters = PARAMETERS
        .replace("[[0,2[J]],[2[J],0]]", "[[0,1e-300[J]],[0,0]]")
        .replace("hbar:J*s=1[J*s]", "hbar:J*s=1e100[J*s]");
    let source = format!(
        "{SPACES} model Evolution(){{{parameters} state psi:coordinates<complex<1>,A>; initial{{psi=coordinates(A,[math.complex(1,0),0]);}} relation flow{{derivative(psi)=math.complex(0,-1)/hbar*apply(ha,psi);}}}}"
    );
    let document = ModelDocument::compile("tiny-hermitian.eqi", &source).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let flow = FirstOrderProgram::lower(
        document.program(),
        document.aliases()["flow"].downcast().unwrap(),
    )
    .unwrap();
    // The represented generator is zero after multiplying 1e-300 by 1e-100.
    assert!(
        flow.constant_generator()
            .unwrap()
            .iter()
            .all(|&value| value == 0.)
    );
    let policy = CommonOdePolicy::new(
        TimeMethod::ImplicitMidpoint,
        0.01,
        1e-12,
        flow.state_coordinates()
            .iter()
            .map(|&c| CommonTimeTolerance::new(c, 1e-14).unwrap())
            .collect(),
    )
    .unwrap()
    .with_hermitian_parameter(document.aliases()["ha"].downcast().unwrap())
    .unwrap();
    let error = CommonOdePlan::resolve(
        &model,
        document.program(),
        policy,
        ImplicitMidpointTimeBackend::CAPABILITIES,
    )
    .unwrap_err();
    assert!(
        error.message().contains("complete matrix is not Hermitian"),
        "{error:?}"
    );
}

#[test]
fn finite_hamiltonian_types_reject_missing_hbar_units_and_foreign_coordinate_roles() {
    let source = format!(
        "{SPACES} space BA=product(B,A); model Evolution(){{{PARAMETERS} parameter H:map<complex<J>,AB,AB>=tensor_product(ha,ib)+tensor_product(ia,hb); state psi:coordinates<complex<1>,AB>; initial{{psi=coordinates(AB,[math.complex(1,0),0,0,0,0,0]);}} relation flow{{derivative(psi)=math.complex(0,-1)/hbar*apply(H,psi);}}}}"
    );
    ModelDocument::compile("valid.eqi", &source).unwrap();
    let spatial = "space S=orthonormal(a,b,c); model M(){domain body=box(0,1,0,1,0,1); parameter hbar:J*s=1[J*s]; parameter H:map<complex<J>,S,S>=linear_map(S,S,[[1[J],0,0],[0,1[J],0],[0,0,1[J]]]); state psi:vector<complex<1>,3> on body; relation flow on body{derivative(psi)=math.complex(0,-1)/hbar*apply(H,psi);}}";
    // The spatial Field and its time equation are independently valid. Only
    // substituting it into the finite-map action must cause this rejection.
    ModelDocument::compile(
        "spatial-control.eqi",
        &spatial.replace("apply(H,psi)", "1[J]*psi"),
    )
    .unwrap();
    let errors = ModelDocument::compile("spatial.eqi", spatial).unwrap_err();
    assert!(
        errors.iter().any(|error| error
            .message()
            .contains("matching declared basis identities and dualities")),
        "{errors:?}"
    );
    for bad in [
        source.replace("hbar:J*s=1[J*s]", "hbar:1=1"),
        source
            .replace(
                "state psi:coordinates<complex<1>,AB>",
                "state psi:coordinates<complex<1>,BA>",
            )
            .replace("psi=coordinates(AB,", "psi=coordinates(BA,"),
        source.replace(
            "state psi:coordinates<complex<1>,AB>",
            "state psi:array<complex<1>,6>",
        ),
    ] {
        assert!(ModelDocument::compile("bad.eqi", &bad).is_err());
    }
}
