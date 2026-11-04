mod command;
mod terminal;

use eqiora::package::{SourceImportContext, compile_source};

#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use command::contained_run as run_for_oracle;
#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use command::read_bounded as read_bounded_for_oracle;
#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use command::{OraclePanicPoint, OracleReadError};
#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use terminal::render_diagnostics as render_for_oracle;
#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use terminal::severity_label as severity_label_for_oracle;

#[cfg_attr(test, allow(dead_code))]
fn main() {
    let outcome = command::contained_run(
        command::process_args,
        |filename, source| {
            compile_source(filename, source, None, &[], SourceImportContext::Bundled)
                .map_err(|error| error.diagnostics().map(<[_]>::to_vec).unwrap_or_else(|| {
                    vec![eqiora::Diagnostic::error(eqiora::diagnostic::codes::LANGUAGE_LOWERING_ERROR, error.to_string())]
                }))
        },
        command::OraclePanicPoint::None,
    );
    std::process::exit(terminal::commit(outcome));
}
