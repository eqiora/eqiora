//! Thin Python construction over the compiler's owned module AST.

mod boundaries;
mod compile;
mod connections;
mod declaration;
mod definition;
mod dimensions;
mod expression;
mod formulation;
mod imported_properties;
mod imports;
mod law;
mod module;
mod records;
pub(crate) use declaration::PyAstType;
pub(crate) use expression::PyAstExpression;
pub(crate) use module::PyAstModule;

use pyo3::prelude::*;
use pyo3::types::PyModule;

pyo3::create_exception!(
    eqiora,
    ModuleError,
    pyo3::exceptions::PyValueError,
    "A Module value violates the bounded authoring contract."
);

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("ModuleError", module.py().get_type::<ModuleError>())?;
    module.add_class::<PyAstExpression>()?;
    declaration::register(module)?;
    module.add_class::<definition::PyAstDefinition>()?;
    module.add_class::<formulation::PyAstFormulation>()?;
    module.add_class::<PyAstModule>()?;
    module.add_function(wrap_pyfunction!(compile::_compile_module, module)?)?;
    Ok(())
}
