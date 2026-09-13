//! Explicit module values cross the existing resolved-source admission boundary.

use eqiora::language::{
    Module, PureValueClassSyntax, SignatureItem, SourceAstFactory as Ast, TextRange,
    VisibilitySyntax,
};
use pyo3::prelude::*;

use super::boundaries::FieldConnectorInput;
use super::connections::{ConnectorInput, connectors};
use super::declaration::PyAstType;
use super::definition::{Definition, PyAstDefinition};
use super::expression::{PyAstExpression, path, syntax_error};

type OperatorInput<'py> = (
    String,
    Vec<(String, PyRef<'py, PyAstType>, u32)>,
    PyRef<'py, PyAstType>,
    PyRef<'py, PyAstExpression>,
    u32,
);

type PropertyContractInput<'py> = (Vec<(String, PyRef<'py, PyAstType>)>, String, Option<String>);

#[pyclass(name = "_AstModule", module = "eqiora._eqiora", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyAstModule {
    pub(crate) value: Module,
}

fn range(ordinal: u32) -> TextRange {
    TextRange::new(ordinal, ordinal.saturating_add(1))
}

impl PyAstModule {
    pub(super) fn admit_metadata(&self) -> PyResult<()> {
        let docs = self
            .value
            .document()
            .doc_comments()
            .map(|(_, doc)| doc.text().len());
        let notation = self
            .value
            .document()
            .notations()
            .map(|(_, notation)| notation.canonical().len());
        docs.chain(notation).try_fold(0usize, |total, bytes| {
            total
                .checked_add(bytes)
                .filter(|total| *total <= 8 * 1024 * 1024)
                .ok_or_else(|| syntax_error("Module exceeds the 8388608-byte limit"))
        })?;
        Ok(())
    }

    fn document_for_edit(&self) -> PyResult<eqiora::language::Document> {
        if self.value.has_native_metadata() {
            return Err(syntax_error("a closed native Module is immutable"));
        }
        Ok(self.value.document().clone())
    }
}

#[pymethods]
impl PyAstModule {
    #[staticmethod]
    fn namespace(package: &str, name: &str) -> PyResult<String> {
        let package = eqiora::package::QualifiedName::parse(package)
            .map_err(|error| syntax_error(error.to_string()))?;
        let namespace = eqiora::compiler::CompilationNamespaceId::new([package.as_str()])
            .map_err(|error| syntax_error(error.message()))?;
        eqiora::compiler::ResolvedSourceUnit::new(
            namespace,
            format!("src/{}.eqi", name.replace('.', "/")),
            "",
        )
        .map_err(|error| syntax_error(error.message()))?;
        Ok(package.as_str().to_owned())
    }

    fn same_graph(&self, other: &Self) -> bool {
        self.value == other.value
    }

    #[new]
    #[pyo3(signature = (definitions, operators, connector_inputs, field_connector_inputs, dimension_inputs=Vec::new()))]
    fn new(
        definitions: Vec<PyRef<'_, PyAstDefinition>>,
        operators: Vec<OperatorInput<'_>>,
        connector_inputs: Vec<ConnectorInput<'_>>,
        field_connector_inputs: Vec<FieldConnectorInput<'_>>,
        dimension_inputs: Vec<(String, PyRef<'_, PyAstType>, u32)>,
    ) -> PyResult<Self> {
        if definitions.len()
            + operators.len()
            + connector_inputs.len()
            + field_connector_inputs.len()
            + dimension_inputs.len()
            > 256
        {
            return Err(syntax_error("module exceeds 256 definitions"));
        }
        let mut components = Vec::new();
        let mut models = Vec::new();
        for definition in definitions {
            match &definition.value {
                Definition::Component(value) => components.push(value.clone()),
                Definition::Model(value) => models.push(value.clone()),
            }
        }
        let operators = operators
            .into_iter()
            .map(|(name, inputs, result, body, ordinal)| {
                if inputs.len() > 256 {
                    return Err(syntax_error("operator formal count exceeds 256"));
                }
                let formals = inputs
                    .into_iter()
                    .map(|(name, kind, formal_ordinal)| {
                        Ast::pure_operator_formal(
                            name,
                            PureValueClassSyntax::Typed(kind.value.clone()),
                            range(formal_ordinal),
                        )
                        .map_err(syntax_error)
                    })
                    .collect::<PyResult<_>>()?;
                Ast::pure_operator(
                    VisibilitySyntax::Public,
                    name,
                    formals,
                    PureValueClassSyntax::Typed(result.value.clone()),
                    body.value.clone(),
                    range(ordinal),
                )
                .map_err(syntax_error)
            })
            .collect::<PyResult<_>>()?;
        let mut connectors = connectors(connector_inputs)?;
        connectors.extend(super::boundaries::connectors(field_connector_inputs)?);
        let dimensions = dimension_inputs
            .into_iter()
            .map(|(name, kind, ordinal)| {
                super::dimensions::declaration(name, &kind.value, range(ordinal))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let document = Ast::document_with_dimensions(
            Vec::new(),
            dimensions,
            connectors,
            components,
            operators,
            models,
        )
        .map_err(syntax_error)?;
        Ok(Self {
            value: Module::from_document(document),
        })
    }

    #[staticmethod]
    fn parse(filename: &str, source: &str) -> PyResult<Self> {
        eqiora::compiler::preflight_resolved_hierarchy([source.len()], 0).map_err(syntax_error)?;
        Ok(Self {
            value: Module::parse(filename, source)
                .map_err(|errors| syntax_error(format!("{errors:?}")))?,
        })
    }

    fn source(&self) -> PyResult<String> {
        // Admission bounds the AST before the recursive formatter runs.
        eqiora::compiler::source_identity::LocalSourceIdentity::from_document(
            self.value.document(),
        )
        .map_err(syntax_error)?;
        self.admit_metadata()?;
        let source = eqiora::language::format(self.value.document());
        if source.len() > 8 * 1024 * 1024 {
            return Err(syntax_error("Module exceeds the 8388608-byte limit"));
        }
        Ok(source)
    }

    fn check(&self) -> PyResult<()> {
        self.admit_metadata()
    }

    fn with_import(&self, name: &str, alias: String, ordinal: u32) -> PyResult<Self> {
        let document = self.document_for_edit()?;
        if let Some((existing, _, _)) = document
            .imports()
            .find(|(_, existing, _)| *existing == alias)
        {
            if existing != &path(name)? {
                return Err(syntax_error("import alias already names another module"));
            }
            return Ok(self.clone());
        }
        let document = Ast::with_import(document, path(name)?, Some(alias), range(ordinal))
            .map_err(syntax_error)?;
        Ok(Self {
            value: Module::from_document(document),
        })
    }

    fn imports(&self) -> Vec<(String, String)> {
        self.value
            .document()
            .imports()
            .map(|(module, alias, _)| (alias.to_owned(), module.to_string()))
            .collect()
    }

    fn dimension_descriptor(
        &self,
        root: (String, String),
        units: Vec<(String, String, PyRef<'_, PyAstModule>)>,
        dependencies: Vec<(String, String)>,
        name: &str,
    ) -> PyResult<crate::modeling::PyDimension> {
        super::dimensions::resolve(self, root, units, dependencies, name)
    }

    fn component(&self, name: &str) -> PyResult<Vec<(String, String, bool)>> {
        let definition = self
            .value
            .document()
            .components()
            .iter()
            .find(|value| value.name() == name && value.visibility() == VisibilitySyntax::Public)
            .ok_or_else(|| syntax_error("import requires one public Component"))?;
        definition
            .signature()
            .iter()
            .map(|item| {
                let role = match item {
                    SignatureItem::Input(_) => "input",
                    SignatureItem::Output(_) => "output",
                    SignatureItem::Clock(_) => "clock",
                    SignatureItem::Support(_) => "support",
                    SignatureItem::Field(_) => "field",
                    SignatureItem::Property(_) => "property",
                    SignatureItem::Parameter(_) => "parameter",
                    SignatureItem::Port(_) | SignatureItem::PortFamily(_) => "port",
                    _ => return Err(syntax_error("unsupported signature declaration")),
                };
                let required =
                    !matches!(item, SignatureItem::Parameter(value) if value.default().is_some());
                Ok((item.name().to_owned(), role.to_owned(), required))
            })
            .collect::<PyResult<_>>()
    }

    fn operator_descriptor(&self, name: &str) -> PyResult<super::imports::OperatorDescriptor> {
        super::imports::operator(&self.value, name)
    }

    fn connector_descriptor(&self, name: &str, public: bool) -> PyResult<(String, String, String)> {
        super::imports::connector(&self.value, name, public)
    }

    fn component_ports(&self, name: &str) -> PyResult<Vec<super::imports::PortDescriptor>> {
        super::imports::ports(&self.value, name)
    }

    fn component_supports(&self, name: &str) -> PyResult<Vec<super::imports::SupportDescriptor>> {
        super::imports::supports(&self.value, name)
    }

    fn record_descriptor(&self, name: &str) -> PyResult<Vec<(String, String)>> {
        super::imports::record(&self.value, name)
    }

    fn enum_descriptor(&self, name: &str) -> PyResult<Vec<String>> {
        super::imports::enumeration(&self.value, name)
    }

    fn finite_space_descriptor(&self, name: &str) -> PyResult<Vec<String>> {
        super::imports::finite_space(&self.value, name)
    }

    fn with_space(&self, name: String, labels: Vec<String>, ordinal: u32) -> PyResult<Self> {
        let declaration = Ast::finite_space(VisibilitySyntax::Public, name, labels, range(ordinal))
            .map_err(syntax_error)?;
        let document =
            Ast::with_finite_space(self.document_for_edit()?, declaration).map_err(syntax_error)?;
        Ok(Self {
            value: Module::from_document(document),
        })
    }

    #[staticmethod]
    fn record_type(name: &str) -> PyResult<PyAstType> {
        super::records::record_type(name)
    }

    fn with_record(
        &self,
        name: String,
        members: Vec<(String, PyRef<'_, PyAstType>, u32)>,
        ordinal: u32,
    ) -> PyResult<Self> {
        Ok(Self {
            value: Module::from_document(super::records::with_record(
                self.document_for_edit()?,
                name,
                members,
                ordinal,
            )?),
        })
    }

    fn with_enum(&self, name: String, members: Vec<String>, ordinal: u32) -> PyResult<Self> {
        let tags = members
            .iter()
            .map(|name| path(name))
            .collect::<PyResult<_>>()?;
        let declaration = Ast::enumeration(VisibilitySyntax::Public, name, tags, range(ordinal))
            .map_err(syntax_error)?;
        Ok(Self {
            value: Module::from_document(Ast::with_enumeration(
                self.document_for_edit()?,
                declaration,
            )),
        })
    }

    fn property_contract_descriptor(
        &self,
        root: (String, String),
        units: Vec<(String, String, PyRef<'_, PyAstModule>)>,
        dependencies: Vec<(String, String)>,
        name: &str,
    ) -> PyResult<super::imported_properties::ContractDescriptor> {
        super::imported_properties::contract(root, units, dependencies, name)
    }

    fn property_release_descriptor(&self, name: &str) -> PyResult<String> {
        super::imported_properties::release(&self.value, name)
    }

    #[staticmethod]
    fn with_contract(
        module: Option<&Self>,
        name: String,
        kind: &PyAstType,
        profile: PropertyContractInput<'_>,
        ordinal: u32,
    ) -> PyResult<Self> {
        let document = Ast::with_property_contract(
            module.map(Self::document_for_edit).transpose()?,
            name,
            kind.value.clone(),
            (
                profile
                    .0
                    .into_iter()
                    .map(|(name, kind)| (name, kind.value.clone()))
                    .collect(),
                match profile.1.as_str() {
                    "value_only" => eqiora::kernel::PropertyDerivatives::ValueOnly,
                    "first_partials" => eqiora::kernel::PropertyDerivatives::FirstPartials,
                    "first_open_intervals" => {
                        eqiora::kernel::PropertyDerivatives::FirstOpenIntervals
                    }
                    _ => return Err(syntax_error("unsupported property derivative profile")),
                },
                profile.2.as_deref().map(path).transpose()?,
            ),
            range(ordinal),
        )
        .map_err(syntax_error)?;
        Ok(Self {
            value: Module::from_document(document),
        })
    }

    #[staticmethod]
    fn with_table_release(
        module: Option<&Self>,
        name: String,
        contract: &str,
        table: (
            String,
            String,
            PyRef<'_, PyAstExpression>,
            PyRef<'_, PyAstExpression>,
            PyRef<'_, PyAstExpression>,
            PyRef<'_, PyAstExpression>,
        ),
        attribution: (String, String, String),
        ordinal: u32,
    ) -> PyResult<Self> {
        let range = range(ordinal);
        let definition = Ast::property_table(
            path(&table.0)?,
            (table.1, table.2.value.clone()),
            ("sample".into(), table.3.value.clone()),
            [table.4.value.clone(), table.5.value.clone()],
            range,
        )
        .map_err(syntax_error)?;
        let scale = Ast::expression(
            eqiora::language::ExprKind::Number(
                eqiora::language::DecimalLiteral::parse("1").map_err(syntax_error)?,
            ),
            range,
        )
        .map_err(syntax_error)?;
        let document = Ast::with_property_release(
            module.map(Self::document_for_edit).transpose()?,
            name,
            path(contract)?,
            (definition, table.3.value.clone(), scale),
            (path(&attribution.0)?, path(&attribution.1)?),
            (None, Some(path(&attribution.2)?)),
            range,
        )
        .map_err(syntax_error)?;
        Ok(Self {
            value: Module::from_document(document),
        })
    }

    #[staticmethod]
    fn with_release(
        module: Option<&Self>,
        name: String,
        contract: &str,
        source: (
            PyRef<'_, PyAstExpression>,
            PyRef<'_, PyAstExpression>,
            PyRef<'_, PyAstExpression>,
        ),
        attribution: (String, String),
        profile: (Option<PyRef<'_, PyAstExpression>>, Option<String>, String),
        ordinal: u32,
    ) -> PyResult<Self> {
        if profile.2 != "reject" {
            return Err(syntax_error(
                "property outside-domain policy must be reject",
            ));
        }
        let document = Ast::with_property_release(
            module.map(Self::document_for_edit).transpose()?,
            name,
            path(contract)?,
            (
                source.0.value.clone().into(),
                source.1.value.clone(),
                source.2.value.clone(),
            ),
            (path(&attribution.0)?, path(&attribution.1)?),
            (
                profile.0.map(|value| value.value.clone()),
                profile.1.as_deref().map(path).transpose()?,
            ),
            range(ordinal),
        )
        .map_err(syntax_error)?;
        Ok(Self {
            value: Module::from_document(document),
        })
    }

    fn with_material(
        &self,
        name: String,
        properties: Vec<(String, String, u32)>,
        ordinal: u32,
    ) -> PyResult<Self> {
        let properties = properties
            .into_iter()
            .map(|(name, release, ordinal)| Ok((name, path(&release)?, range(ordinal))))
            .collect::<PyResult<_>>()?;
        let document = Ast::with_material_composition(
            self.document_for_edit()?,
            name,
            properties,
            range(ordinal),
        )
        .map_err(syntax_error)?;
        Ok(Self {
            value: Module::from_document(document),
        })
    }

    fn metadata(
        &self,
        ordinal: u32,
        doc: Option<String>,
        notation: Option<&crate::notation::PyNotation>,
    ) -> PyResult<Self> {
        let document = self
            .document_for_edit()?
            .with_declaration_metadata(range(ordinal), doc, notation.map(|value| value.0.clone()))
            .map_err(syntax_error)?;
        Ok(Self {
            value: Module::from_document(document),
        })
    }
}
