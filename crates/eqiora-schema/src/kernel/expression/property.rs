//! Scientific identity attached to an exact expression occurrence.
//!
//! The value and calculus remain owned by the existing typed literal and pure
//! operator vocabulary. These declarations record scientific release meaning;
//! they do not assert external validation of a citation or material.

use super::super::pure_operator::{CalculusBuilder, CalculusNode, PureOperatorDefinition};
use eqiora_core::{Diagnostic, ValueLiteral};

/// Exact derivative products promised by a nominal property contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PropertyDerivatives {
    /// No independent-input derivative products.
    ValueOnly,
    /// First formal partials under the analytic operating-domain guard.
    FirstPartials,
    /// First derivatives only on the open intervals of a one-axis release.
    FirstOpenIntervals,
}
impl PropertyDerivatives {
    /// Canonical declaration spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ValueOnly => "value_only",
            Self::FirstPartials => "first_partials",
            Self::FirstOpenIntervals => "first_open_intervals",
        }
    }
}

/// The mathematical value bound by one exact scientific release.
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyMeaning {
    /// A zero-input value, including every existing typed literal domain.
    Constant(ValueLiteral),
    /// A typed, closed analytic value and its checked operating-domain guard.
    Analytic(PureOperatorDefinition),
    /// Exact resolved-array closure and its specified piecewise-affine calculus.
    Table(Box<crate::property_table::AcceptedRealTable>),
}

impl PropertyMeaning {
    /// A normalized immutable value exists only for a zero-input contract.
    #[must_use]
    pub const fn constant_value(&self) -> Option<&ValueLiteral> {
        match self {
            Self::Constant(value) => Some(value),
            Self::Analytic(_) | Self::Table(_) => None,
        }
    }

    /// Shared executable definition for analytic and data-backed releases.
    #[must_use]
    pub const fn definition(&self) -> Option<&PureOperatorDefinition> {
        match self {
            Self::Constant(_) => None,
            Self::Analytic(value) => Some(value),
            Self::Table(value) => Some(value.value()),
        }
    }

    /// Complete mathematical result type, independent of a chosen operating point.
    #[must_use]
    pub fn value_type(&self) -> Option<eqiora_core::ValueType> {
        match self {
            Self::Constant(value) => Some(value.value_type().clone()),
            Self::Analytic(definition) => eqiora_core::ValueType::scalar(
                definition.result_rule().scalar_domain()?,
                definition.result_rule().dimension()?,
            )
            .ok(),
            Self::Table(table) => eqiora_core::ValueType::scalar(
                eqiora_core::ScalarDomain::Real,
                table.value_dimension(),
            )
            .ok(),
        }
    }
}

/// One exact nominal contract/release binding, independent of provider choice.
#[derive(Debug, Clone, PartialEq)]
pub struct PropertyRelease {
    contract: String,
    release: String,
    composition: Option<String>,
    consumer: Option<(String, String)>,
    inputs: Vec<String>,
    branch: Option<String>,
    derivatives: PropertyDerivatives,
    citation: String,
    license: String,
    meaning: PropertyMeaning,
}

impl PropertyRelease {
    /// Bind exact scientific identities to the existing mathematical owner.
    ///
    /// # Errors
    /// Rejects unbounded or malformed identity strings, duplicate formal names,
    /// formal count disagreement, and derivative claims for constant contracts.
    pub fn new(
        identities: (String, String),
        inputs: Vec<String>,
        branch: Option<String>,
        derivatives: PropertyDerivatives,
        attribution: (String, String),
        meaning: PropertyMeaning,
    ) -> Result<Self, Diagnostic> {
        let (contract, release) = identities;
        let (citation, license) = attribution;
        for value in [&contract, &release, &citation, &license]
            .into_iter()
            .chain(branch.iter())
        {
            if value.is_empty() || value.len() > 4096 || value.chars().any(char::is_control) {
                return Err(invalid("property identities must be nonempty bounded text"));
            }
        }
        let mut names = std::collections::BTreeSet::new();
        for name in &inputs {
            if name.is_empty()
                || name.len() > 256
                || !name.chars().enumerate().all(|(index, value)| {
                    value == '_'
                        || value.is_ascii_alphabetic()
                        || index > 0 && value.is_ascii_digit()
                })
                || !names.insert(name)
            {
                return Err(invalid(
                    "property inputs require distinct bounded identifiers",
                ));
            }
        }
        if let Some(definition) = meaning.definition()
            && (!definition.result_rule().is_invariant_scalar()
                || definition.result_rule().dimension().is_none()
                || definition.result_rule().scalar_domain()
                    != Some(eqiora_core::ScalarDomain::Real)
                || definition.formals().iter().any(|formal| {
                    !formal.is_invariant_scalar()
                        || formal.dimension().is_none()
                        || formal.scalar_domain() != Some(eqiora_core::ScalarDomain::Real)
                }))
        {
            return Err(invalid(
                "analytic property requires concrete real scalar input and output contracts",
            ));
        }
        match &meaning {
            PropertyMeaning::Constant(_)
                if !inputs.is_empty() || derivatives != PropertyDerivatives::ValueOnly =>
            {
                return Err(invalid(
                    "constant property contracts have no independent inputs or partial products",
                ));
            }
            value
                if value
                    .definition()
                    .is_some_and(|definition| inputs.len() != definition.formals().len()) =>
            {
                return Err(invalid(
                    "property input names must match the exact calculus formal slots",
                ));
            }
            _ => {}
        }
        match (&meaning, derivatives) {
            (PropertyMeaning::Analytic(definition), PropertyDerivatives::FirstOpenIntervals) => {
                partial_definition(&open_interval_definition(definition)?, 0)?;
            }
            (PropertyMeaning::Table(_), PropertyDerivatives::FirstPartials) => {
                return Err(invalid(
                    "table derivatives require the explicit first_open_intervals profile",
                ));
            }
            (PropertyMeaning::Analytic(definition), PropertyDerivatives::FirstPartials) => {
                for formal in 0..inputs.len() {
                    partial_definition(
                        definition,
                        u16::try_from(formal)
                            .map_err(|_| invalid("property formal limit exceeded"))?,
                    )?;
                }
            }
            _ => {}
        }
        Ok(Self {
            contract,
            release,
            composition: None,
            consumer: None,
            inputs,
            branch,
            derivatives,
            citation,
            license,
            meaning,
        })
    }

    /// Retain the authored nominal material grouping of this exact release.
    ///
    /// # Errors
    /// Rejects malformed or unbounded composition identities.
    pub fn with_composition(mut self, composition: Option<String>) -> Result<Self, Diagnostic> {
        if composition.as_ref().is_some_and(|name| {
            name.is_empty() || name.len() > 4096 || name.chars().any(char::is_control)
        }) {
            return Err(invalid("property composition requires a bounded identity"));
        }
        self.composition = composition;
        Ok(self)
    }

    /// Retain the exact named requirement whose expression occurrence uses this release.
    ///
    /// # Errors
    /// Rejects empty, malformed, or unbounded context labels.
    pub fn for_requirement(
        mut self,
        component: String,
        requirement: String,
    ) -> Result<Self, Diagnostic> {
        if [&component, &requirement]
            .into_iter()
            .any(|name| name.is_empty() || name.len() > 4096 || name.chars().any(char::is_control))
        {
            return Err(invalid(
                "property consumer requires bounded component and requirement identities",
            ));
        }
        self.consumer = Some((component, requirement));
        Ok(self)
    }

    /// Nominal material composition, when explicitly selected by the binding.
    #[must_use]
    pub fn composition(&self) -> Option<&str> {
        self.composition.as_deref()
    }
    /// Named component requirement; the containing DAG root supplies occurrence identity.
    #[must_use]
    pub fn consumer(&self) -> Option<(&str, &str)> {
        self.consumer
            .as_ref()
            .map(|(component, requirement)| (component.as_str(), requirement.as_str()))
    }

    /// Exact nominal contract identity.
    #[must_use]
    pub fn contract(&self) -> &str {
        &self.contract
    }
    /// Exact release identity, including its accepted source/package identity.
    #[must_use]
    pub fn release(&self) -> &str {
        &self.release
    }
    /// Independent input names in exact formal-slot order.
    #[must_use]
    pub fn inputs(&self) -> &[String] {
        &self.inputs
    }
    /// Declared phase branch; no branch is inferred from a material name.
    #[must_use]
    pub fn branch(&self) -> Option<&str> {
        self.branch.as_deref()
    }
    /// Whether the contract requires ordinary first formal partials.
    #[must_use]
    pub const fn first_partials(&self) -> bool {
        !matches!(self.derivatives, PropertyDerivatives::ValueOnly)
    }
    /// Exact derivative product profile.
    #[must_use]
    pub const fn derivatives(&self) -> PropertyDerivatives {
        self.derivatives
    }
    /// Authored scientific citation identity.
    #[must_use]
    pub fn citation(&self) -> &str {
        &self.citation
    }
    /// Authored scientific license identity.
    #[must_use]
    pub fn license(&self) -> &str {
        &self.license
    }
    /// Derive an admitted first formal partial through the shared calculus owner.
    /// All other declared inputs remain fixed and the exact validity guard remains.
    ///
    /// # Errors
    /// Rejects undeclared derivative products or an absent independent input.
    pub fn partial(&self, input: &str) -> Result<PureOperatorDefinition, Diagnostic> {
        if !self.first_partials() {
            return Err(invalid("property contract does not admit first partials"));
        }
        let formal = self
            .inputs
            .iter()
            .position(|name| name == input)
            .ok_or_else(|| invalid("unknown independent property input"))?;
        if let PropertyMeaning::Table(table) = &self.meaning {
            return Ok(table.derivative().clone());
        }
        let PropertyMeaning::Analytic(definition) = &self.meaning else {
            return Err(invalid("constant contract has no partial products"));
        };
        let open_definition;
        let definition = if self.derivatives == PropertyDerivatives::FirstOpenIntervals {
            open_definition = open_interval_definition(definition)?;
            &open_definition
        } else {
            definition
        };
        partial_definition(
            definition,
            u16::try_from(formal).map_err(|_| invalid("property formal limit exceeded"))?,
        )
    }

    /// Exact existing mathematical value owner.
    #[must_use]
    pub const fn meaning(&self) -> &PropertyMeaning {
        &self.meaning
    }
    /// Whether the exact analytic root retains an operating-domain guard.
    #[must_use]
    pub fn guarded(&self) -> bool {
        self.meaning.definition().is_some_and(|definition| {
            matches!(
                definition.nodes()[definition.root().index() as usize],
                CalculusNode::Require { .. }
            )
        })
    }
}

/// Tighten only the authored interval guard; the value law keeps closed validity.
fn open_interval_definition(
    definition: &PureOperatorDefinition,
) -> Result<PureOperatorDefinition, Diagnostic> {
    use super::super::ComparisonOp;
    let fail =
        || invalid("first_open_intervals requires one input and a closed interval validity guard");
    if definition.formals().len() != 1 {
        return Err(fail());
    }
    let nodes = definition.nodes();
    let CalculusNode::Require { condition, .. } = nodes[definition.root().index() as usize] else {
        return Err(fail());
    };
    let CalculusNode::And(lower, upper) = nodes[condition.index() as usize] else {
        return Err(fail());
    };
    let CalculusNode::Compare(ComparisonOp::GreaterEqual, low_input, _) =
        nodes[lower.index() as usize]
    else {
        return Err(fail());
    };
    let CalculusNode::Compare(ComparisonOp::LessEqual, high_input, _) =
        nodes[upper.index() as usize]
    else {
        return Err(fail());
    };
    if nodes[low_input.index() as usize] != nodes[high_input.index() as usize]
        || !matches!(&nodes[low_input.index() as usize], CalculusNode::FormalComponent { formal: 0, axes } if axes.is_empty())
    {
        return Err(fail());
    }
    let mut builder = CalculusBuilder::new(
        definition.formals().iter().copied(),
        definition.result_rule(),
    )
    .map_err(|_| fail())?;
    for (index, node) in nodes.iter().enumerate() {
        let node = match node {
            CalculusNode::Compare(_, left, right) if index == lower.index() as usize => {
                CalculusNode::Compare(ComparisonOp::Greater, *left, *right)
            }
            CalculusNode::Compare(_, left, right) if index == upper.index() as usize => {
                CalculusNode::Compare(ComparisonOp::Less, *left, *right)
            }
            node => node.clone(),
        };
        builder.push(node).map_err(|_| fail())?;
    }
    builder.finish(definition.root()).map_err(|_| fail())
}

fn partial_definition(
    definition: &PureOperatorDefinition,
    formal: u16,
) -> Result<PureOperatorDefinition, Diagnostic> {
    let failure = |error: super::super::pure_operator::PureOperatorError| {
        invalid(&format!("property first partial is unavailable: {error}"))
    };
    let input = definition
        .formals()
        .get(usize::from(formal))
        .ok_or_else(|| invalid("invalid property formal"))?;
    let dimension = definition
        .result_rule()
        .dimension()
        .and_then(|output| input.dimension().and_then(|input| output.div(input)))
        .ok_or_else(|| invalid("property partial requires exact input/output dimensions"))?;
    let mut builder = CalculusBuilder::new(
        definition.formals().iter().copied(),
        definition.result_rule().with_dimension(dimension),
    )
    .map_err(failure)?;
    let arguments = (0..definition.formals().len())
        .map(|slot| {
            builder.push(CalculusNode::FormalComponent {
                formal: u16::try_from(slot).expect("bounded formal count"),
                axes: Box::new([]),
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(failure)?;
    let root = builder
        .apply_scalar(definition, &arguments)
        .map_err(failure)?;
    let partial = builder.partial(root, formal).map_err(failure)?;
    builder.finish(partial).map_err(failure)
}

fn invalid(message: &str) -> Diagnostic {
    Diagnostic::error(
        eqiora_core::diagnostic::codes::INVALID_EXPRESSION_DAG,
        message,
    )
}

impl super::ExprDagBuilder {
    /// Construct a fresh, scientifically identified property occurrence.
    /// Equal values from different releases deliberately have distinct arena IDs.
    ///
    /// # Errors
    /// Rejects arguments to a constant or invalid typed application structure.
    pub fn property(
        &mut self,
        release: PropertyRelease,
        arguments: impl IntoIterator<Item = super::ExprId>,
    ) -> Result<super::ExprId, Diagnostic> {
        let arguments = arguments.into_iter().collect::<Vec<_>>();
        let id = match release.meaning() {
            PropertyMeaning::Constant(value) => {
                if !arguments.is_empty() {
                    return Err(invalid("constant property release takes no arguments"));
                }
                self.push(super::ExprNode::Constant(value.clone()))?
            }
            PropertyMeaning::Analytic(definition) => self.pure_operator(definition, arguments)?,
            PropertyMeaning::Table(table) => self.pure_operator(table.value(), arguments)?,
        };
        self.properties.insert(id, release);
        Ok(id)
    }

    /// Restore the scientific identity on an existing exact occurrence during replay.
    ///
    /// # Errors
    /// Rejects a foreign root, duplicate annotation, changed constant, or substituted
    /// analytic definition. Every annotation authenticates its actual value owner.
    pub fn bind_property(
        &mut self,
        id: super::ExprId,
        release: PropertyRelease,
    ) -> Result<(), Diagnostic> {
        let matches = match (self.nodes.get(id.index() as usize), release.meaning()) {
            (Some(super::ExprNode::Constant(actual)), PropertyMeaning::Constant(expected)) => {
                actual == expected
            }
            (
                Some(super::ExprNode::PureOperatorApplication(actual)),
                expected @ (PropertyMeaning::Analytic(_) | PropertyMeaning::Table(_)),
            ) => {
                actual.definition()
                    == expected
                        .definition()
                        .expect("nonconstant definition")
                        .digest()
            }
            _ => false,
        };
        if !matches || self.properties.contains_key(&id) {
            return Err(invalid(
                "property annotation must identify one exact unannotated value occurrence",
            ));
        }
        self.properties.insert(id, release);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::{ExprDagBuilder, ExprNode};
    use super::*;

    fn release(name: &str, value: ValueLiteral) -> PropertyRelease {
        PropertyRelease::new(
            (
                "package@exact::Coefficient".into(),
                format!("package@exact::{name}"),
            ),
            vec![],
            None,
            PropertyDerivatives::ValueOnly,
            ("org.example.measurement".into(), "spdx.CC0_1_0".into()),
            PropertyMeaning::Constant(value),
        )
        .unwrap()
    }

    #[test]
    fn equal_values_keep_distinct_release_occurrences() {
        let value = ValueLiteral::from_real(
            eqiora_core::ValueType::scalar(
                eqiora_core::ScalarDomain::Real,
                eqiora_core::DimExponents::DIMENSIONLESS,
            )
            .unwrap(),
            2.0,
        )
        .unwrap();
        let mut builder = ExprDagBuilder::new();
        let first = builder
            .property(release("First", value.clone()), [])
            .unwrap();
        let second = builder
            .property(release("Second", value.clone()), [])
            .unwrap();
        assert_ne!(first, second);
        let sum = builder.add(first, second).unwrap();
        let dag = builder.finish([sum]).unwrap();
        assert_eq!(dag.properties().len(), 2);
        assert_ne!(
            dag.properties()[&first].release(),
            dag.properties()[&second].release()
        );
        assert_eq!(dag.node(first), dag.node(second));
    }

    #[test]
    fn replay_rejects_changed_value_and_duplicate_claim() {
        let value_type = eqiora_core::ValueType::scalar(
            eqiora_core::ScalarDomain::Real,
            eqiora_core::DimExponents::DIMENSIONLESS,
        )
        .unwrap();
        let first = ValueLiteral::from_real(value_type.clone(), 2.0).unwrap();
        let foreign = ValueLiteral::from_real(value_type, 3.0).unwrap();
        let mut builder = ExprDagBuilder::new();
        let root = builder.push(ExprNode::Constant(first.clone())).unwrap();
        assert!(
            builder
                .bind_property(root, release("Foreign", foreign))
                .is_err()
        );
        builder
            .bind_property(root, release("Exact", first.clone()))
            .unwrap();
        assert!(
            builder
                .bind_property(root, release("Other", first))
                .is_err()
        );
    }
}
