//! Closed compiler-owned Formulation variants used by both Module definition kinds.
use super::{
    declaration::PyAstType,
    expression::{PyAstExpression, syntax_error},
};
use eqiora::language::{
    ActivationSyntax, Expr, FieldRoleSyntax, FormulationBinding, SourceAstFactory as Ast, TextRange,
};
use pyo3::prelude::*;

type TestInput<'py> = (
    String,
    String,
    Vec<String>,
    PyRef<'py, crate::modeling::PyValueType>,
    Option<String>,
);
type AmplitudeInput<'py> = (String, String, PyRef<'py, PyAstType>, Option<String>);

#[pyclass(name = "_AstFormulation", module = "eqiora._eqiora", frozen)]
pub(super) struct PyAstFormulation {
    pub(super) name: String,
    pub(super) relations: Vec<String>,
    pub(super) binding: FormulationBinding,
    pub(super) equations: Vec<(Expr, Expr)>,
}

#[pymethods]
impl PyAstFormulation {
    #[staticmethod]
    fn weak(
        name: String,
        relations: Vec<String>,
        tests: Vec<TestInput<'_>>,
        equations: Vec<(PyRef<'_, PyAstExpression>, PyRef<'_, PyAstExpression>)>,
    ) -> PyResult<Self> {
        if relations.len() > 8 || tests.len() > 8 || equations.len() > 8 {
            return Err(syntax_error("weak form exceeds the 8-item inventory limit"));
        }
        let tests = tests
            .into_iter()
            .map(|(name, trial, zero_on, kind, regularity)| {
                if !kind.value.shape().is_scalar()
                    || kind.value.scalar_domain() != eqiora::ScalarDomain::Real
                {
                    return Err(syntax_error("test dimension requires a real scalar type"));
                }
                Ok((
                    name,
                    trial,
                    zero_on,
                    super::boundaries::dimension(&kind)?,
                    regularity,
                ))
            })
            .collect::<PyResult<_>>()?;
        Ok(Self {
            name,
            relations,
            binding: FormulationBinding::WeakTests { tests },
            equations: equations
                .into_iter()
                .map(|(a, b)| (a.value.clone(), b.value.clone()))
                .collect(),
        })
    }

    #[staticmethod]
    fn harmonic(
        name: String,
        relations: Vec<String>,
        angular_frequency: &PyAstExpression,
        excitations: Vec<(String, PyRef<'_, PyAstExpression>)>,
        amplitudes: Vec<AmplitudeInput<'_>>,
    ) -> PyResult<Self> {
        if relations.len() > 256 || excitations.len() > 256 || amplitudes.len() > 256 {
            return Err(syntax_error(
                "harmonic form exceeds the definition inventory limit",
            ));
        }
        let amplitudes = amplitudes
            .into_iter()
            .map(|(name, original, kind, support)| {
                Ast::field(
                    name,
                    support,
                    FieldRoleSyntax::Variable,
                    eqiora::kernel::SpatialRegularity::Unspecified,
                    ActivationSyntax::Continuous,
                    kind.value.clone(),
                    TextRange::default(),
                )
                .map(|field| (field, original))
                .map_err(syntax_error)
            })
            .collect::<PyResult<_>>()?;
        Ok(Self {
            name,
            relations,
            binding: FormulationBinding::Harmonic {
                angular_frequency: angular_frequency.value.clone(),
                excitations: excitations
                    .into_iter()
                    .map(|(name, expression)| (name, expression.value.clone()))
                    .collect(),
                amplitudes,
            },
            equations: Vec::new(),
        })
    }
}
