use super::super::tests::{compile_process, process_run};
use super::*;
use crate::SamplingIdentity;

pub(super) const WIENER: &str = r#"model Process() {
    noise W: wiener(covariance=1);
    state x: 1;
    initial { x=1; }
    law dynamics for x {
        calculus ito;
        drift -x * 2[1/s];
        diffusion W=x * 0.5[1/s^(1/2)];
    }
}"#;
pub(super) const POISSON: &str = r#"model Process() {
    noise N: poisson(intensity=8[1/s]);
    state x: 1;
    initial { x=1; }
    law dynamics for x { jump N=x; }
}"#;

pub(super) fn request(source: &str, method: ProcessMethod) -> Arc<ProcessRun> {
    let kernel = compile_process(source);
    let program = ProcessProgram::new(&kernel).unwrap();
    let state = program.initialize(0.).unwrap();
    process_run(
        &kernel,
        state,
        &[31; 32],
        &SamplingIdentity::new(b"sample", b"checkpoint", &[5, 2], None).unwrap(),
        1.,
        method,
    )
    .unwrap()
}
