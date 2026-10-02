use crate::ast::{Document, Expr, NamePath, TextRange, VisibilitySyntax};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PropertyContractDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) visibility: VisibilitySyntax,
    pub(crate) name: String,
    pub(crate) value_type: crate::ValueTypeSyntax,
    pub(crate) inputs: Vec<(String, crate::ValueTypeSyntax)>,
    pub(crate) derivatives: eqiora_schema::kernel::PropertyDerivatives,
    pub(crate) branch: Option<NamePath>,
    pub(crate) range: TextRange,
}

/// One closed property definition, with data references kept outside value scope.
#[derive(Debug, Clone, PartialEq)]
pub enum PropertySourceSyntax {
    /// Existing typed constant or analytic expression.
    Expression(Expr),
    /// Exact package-owned resolved-array reference and column meaning.
    Table(Box<PropertyTableSyntax>),
}
impl From<Expr> for PropertySourceSyntax {
    fn from(value: Expr) -> Self {
        Self::Expression(value)
    }
}
/// The sole supported table profile: exact piecewise-affine values and open derivatives.
#[derive(Debug, Clone, PartialEq)]
pub struct PropertyTableSyntax {
    pub(crate) data: NamePath,
    pub(crate) axis: String,
    pub(crate) axis_dimension: Expr,
    pub(crate) value: String,
    pub(crate) value_dimension: Expr,
    pub(crate) validity: [Expr; 2],
    pub(crate) range: TextRange,
}
impl PropertyTableSyntax {
    /// Exact package-owned asset reference.
    #[must_use]
    pub const fn data(&self) -> &NamePath {
        &self.data
    }
    /// Contract input assigned to the first column.
    #[must_use]
    pub fn axis(&self) -> &str {
        &self.axis
    }
    /// Coherent-SI dimension of the first column.
    #[must_use]
    pub const fn axis_dimension(&self) -> &Expr {
        &self.axis_dimension
    }
    /// Authored name of the result column.
    #[must_use]
    pub fn value(&self) -> &str {
        &self.value
    }
    /// Coherent-SI dimension of the result column.
    #[must_use]
    pub const fn value_dimension(&self) -> &Expr {
        &self.value_dimension
    }
    /// Declared closed interval, which must be contained by the exact array coverage.
    #[must_use]
    pub const fn validity(&self) -> &[Expr; 2] {
        &self.validity
    }
    /// Original declaration range.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PropertyReleaseDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) visibility: VisibilitySyntax,
    pub(crate) name: String,
    pub(crate) contract: NamePath,
    pub(crate) source_value: PropertySourceSyntax,
    pub(crate) validity: Option<Expr>,
    pub(crate) branch: Option<NamePath>,
    pub(crate) source_dimension: Option<Expr>,
    pub(crate) coherent_si_scale: Expr,
    pub(crate) citation: NamePath,
    pub(crate) license: NamePath,
    pub(crate) range: TextRange,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MaterialCompositionDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) visibility: VisibilitySyntax,
    pub(crate) name: String,
    pub(crate) properties: Vec<PropertyBindingDecl>,
    pub(crate) range: TextRange,
}

/// One exact nominal property contract in a shared external signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentPropertyDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) name: String,
    pub(crate) contract: NamePath,
    pub(crate) range: TextRange,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PropertyBindingDecl {
    pub(crate) comments: crate::ast::comments::SourceComments,
    pub(crate) property: String,
    pub(crate) release: NamePath,
    pub(crate) range: TextRange,
}

impl Document {
    /// Ordered independent inputs, derivative requirement, and declared phase branch.
    pub fn property_contract_profiles(
        &self,
    ) -> impl ExactSizeIterator<
        Item = (
            &str,
            &[(String, crate::ValueTypeSyntax)],
            eqiora_schema::kernel::PropertyDerivatives,
            Option<&NamePath>,
        ),
    > {
        self.property_contracts.iter().map(|value| {
            (
                value.name.as_str(),
                value.inputs.as_slice(),
                value.derivatives,
                value.branch.as_ref(),
            )
        })
    }

    /// Exact operating-domain predicate and phase branch; absent predicate is unconditional.
    pub fn property_release_profiles(
        &self,
    ) -> impl ExactSizeIterator<Item = (&str, Option<&Expr>, Option<&NamePath>)> {
        self.property_releases.iter().map(|value| {
            (
                value.name.as_str(),
                value.validity.as_ref(),
                value.branch.as_ref(),
            )
        })
    }
    pub fn property_contract_syntax(
        &self,
    ) -> impl ExactSizeIterator<Item = (VisibilitySyntax, &str, &crate::ValueTypeSyntax, TextRange)>
    {
        self.property_contracts.iter().map(|value| {
            (
                value.visibility,
                value.name.as_str(),
                &value.value_type,
                value.range,
            )
        })
    }

    pub fn property_release_syntax(
        &self,
    ) -> impl ExactSizeIterator<
        Item = (
            VisibilitySyntax,
            &str,
            &NamePath,
            &PropertySourceSyntax,
            Option<&Expr>,
            &Expr,
            &NamePath,
            &NamePath,
            TextRange,
        ),
    > {
        self.property_releases.iter().map(|value| {
            (
                value.visibility,
                value.name.as_str(),
                &value.contract,
                &value.source_value,
                value.source_dimension.as_ref(),
                &value.coherent_si_scale,
                &value.citation,
                &value.license,
                value.range,
            )
        })
    }

    pub fn material_composition_syntax(
        &self,
    ) -> impl ExactSizeIterator<
        Item = (
            VisibilitySyntax,
            &str,
            Vec<(&str, &NamePath, TextRange)>,
            TextRange,
        ),
    > {
        self.material_compositions.iter().map(|value| {
            (
                value.visibility,
                value.name.as_str(),
                value
                    .properties
                    .iter()
                    .map(|binding| (binding.property.as_str(), &binding.release, binding.range))
                    .collect(),
                value.range,
            )
        })
    }

    #[must_use]
    pub fn isolated_property_declarations(&self) -> Vec<(String, VisibilitySyntax, Self)> {
        let mut values = self
            .property_contracts
            .iter()
            .map(|declaration| {
                (
                    declaration.name.clone(),
                    declaration.visibility,
                    Self {
                        comments: Default::default(),
                        imports: Vec::new(),
                        records: self.records.clone(),
                        enumerations: self.enumerations.clone(),
                        finite_spaces: self.finite_spaces.clone(),
                        dimensions: self.dimensions.clone(),
                        property_contracts: vec![declaration.clone()],
                        property_releases: Vec::new(),
                        material_compositions: Vec::new(),
                        connectors: Vec::new(),
                        components: Vec::new(),
                        pure_operators: Vec::new(),
                        models: Vec::new(),
                    },
                )
            })
            .collect::<Vec<_>>();
        values.extend(self.property_releases.iter().map(|declaration| {
            (
                declaration.name.clone(),
                declaration.visibility,
                Self {
                    comments: Default::default(),
                    imports: Vec::new(),
                    records: self.records.clone(),
                    enumerations: self.enumerations.clone(),
                    finite_spaces: self.finite_spaces.clone(),
                    dimensions: self.dimensions.clone(),
                    property_contracts: Vec::new(),
                    property_releases: vec![declaration.clone()],
                    material_compositions: Vec::new(),
                    connectors: Vec::new(),
                    components: Vec::new(),
                    pure_operators: Vec::new(),
                    models: Vec::new(),
                },
            )
        }));
        values.extend(self.material_compositions.iter().map(|declaration| {
            (
                declaration.name.clone(),
                declaration.visibility,
                Self {
                    comments: Default::default(),
                    imports: Vec::new(),
                    records: self.records.clone(),
                    enumerations: self.enumerations.clone(),
                    finite_spaces: self.finite_spaces.clone(),
                    dimensions: self.dimensions.clone(),
                    property_contracts: Vec::new(),
                    property_releases: Vec::new(),
                    material_compositions: vec![declaration.clone()],
                    connectors: Vec::new(),
                    components: Vec::new(),
                    pure_operators: Vec::new(),
                    models: Vec::new(),
                },
            )
        }));
        values
    }
}

impl fmt::Display for NamePath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.text)
    }
}

impl NamePath {
    #[must_use]
    pub fn is_qualified(&self) -> bool {
        self.segments.len() > 1
    }
}
use core::fmt;

impl ComponentPropertyDecl {
    /// Public signature name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Exact nominal property contract path.
    #[must_use]
    pub const fn contract(&self) -> &NamePath {
        &self.contract
    }
    /// Full signature-entry range.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }
}
