//! Exact scientific release identity on an existing expression occurrence.
use super::*;
use eqiora_schema::kernel::{PropertyDerivatives, PropertyMeaning, PropertyRelease};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireProperty {
    pub(super) root: u32,
    contract: String,
    release: String,
    composition: Option<String>,
    consumer: Option<(String, String)>,
    inputs: Vec<String>,
    branch: Option<String>,
    derivatives: PropertyDerivatives,
    citation: String,
    license: String,
    meaning: WirePropertyMeaning,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum WirePropertyMeaning {
    Constant {
        value: WireValueLiteral,
    },
    Analytic {
        definition: String,
    },
    Table {
        definition: String,
        array: serde_json::Value,
        digest: String,
        axis_dimension: WireDimension,
        value_dimension: WireDimension,
        profile: eqiora_schema::property_table::RealTableProfile,
        validity: [eqiora_schema::kernel::pure_operator::ExactRational; 2],
    },
}

impl WireProperty {
    pub(super) fn encode(root: ExprId, release: &PropertyRelease) -> Result<Self, Diagnostic> {
        Ok(Self {
            root: root.index(),
            contract: release.contract().to_owned(),
            release: release.release().to_owned(),
            composition: release.composition().map(str::to_owned),
            consumer: release
                .consumer()
                .map(|(component, requirement)| (component.to_owned(), requirement.to_owned())),
            inputs: release.inputs().to_vec(),
            branch: release.branch().map(str::to_owned),
            derivatives: release.derivatives(),
            citation: release.citation().to_owned(),
            license: release.license().to_owned(),
            meaning: match release.meaning() {
                PropertyMeaning::Constant(value) => WirePropertyMeaning::Constant {
                    value: WireValueLiteral::encode(value)?,
                },
                PropertyMeaning::Analytic(definition) => WirePropertyMeaning::Analytic {
                    definition: definition.digest().to_string(),
                },
                PropertyMeaning::Table(table) => WirePropertyMeaning::Table {
                    definition: table.value().digest().to_string(),
                    array: serde_json::from_slice(&table.array().canonical_json()?)
                        .map_err(|error| invalid_artifact(error.to_string()))?,
                    digest: crate::ArtifactDigest::from_sha256(table.digest()?).to_string(),
                    axis_dimension: WireDimension::encode(table.axis_dimension()),
                    value_dimension: WireDimension::encode(table.value_dimension()),
                    profile: table.profile(),
                    validity: table.validity(),
                },
            },
        })
    }

    pub(super) fn decode(
        &self,
        builder: &mut ExprDagBuilder,
        ids: &[ExprId],
        definitions: &BTreeMap<String, PureOperatorDefinition>,
    ) -> Result<(), Diagnostic> {
        let meaning = match &self.meaning {
            WirePropertyMeaning::Constant { value } => PropertyMeaning::Constant(value.decode()?),
            WirePropertyMeaning::Analytic { definition } => PropertyMeaning::Analytic(
                definitions
                    .get(definition)
                    .ok_or_else(|| {
                        invalid_artifact("property release references an absent exact definition")
                    })?
                    .clone(),
            ),
            WirePropertyMeaning::Table {
                definition,
                array,
                digest,
                axis_dimension,
                value_dimension,
                profile,
                validity,
            } => {
                let bytes = serde_json::to_vec(array)
                    .map_err(|error| invalid_artifact(error.to_string()))?;
                let validity = validity.clone();
                let table = crate::decode_real_table(
                    &crate::ArtifactDigest::from_hex(digest)?,
                    Some(&bytes),
                    Default::default(),
                    axis_dimension.decode(),
                    value_dimension.decode(),
                    *profile,
                    validity,
                )?;
                if table.value().digest().to_string() != *definition
                    || definitions.get(definition) != Some(table.value())
                {
                    return Err(invalid_artifact(
                        "table closure does not rederive the exact application definition",
                    ));
                }
                PropertyMeaning::Table(Box::new(table))
            }
        };
        let release = PropertyRelease::new(
            (self.contract.clone(), self.release.clone()),
            self.inputs.clone(),
            self.branch.clone(),
            self.derivatives,
            (self.citation.clone(), self.license.clone()),
            meaning,
        )
        .and_then(|release| release.with_composition(self.composition.clone()))
        .and_then(|release| {
            if let Some((component, requirement)) = &self.consumer {
                release.for_requirement(component.clone(), requirement.clone())
            } else {
                Ok(release)
            }
        })
        .map_err(|error| invalid_artifact(error.message()))?;
        builder
            .bind_property(operand(ids, self.root)?, release)
            .map_err(|error| invalid_artifact(error.message()))
    }

    pub(super) fn ensure_limits(&self, limits: ModelDecoderLimits) -> Result<(), Diagnostic> {
        if let WirePropertyMeaning::Constant { value } = &self.meaning {
            value.ensure_limits(limits)?;
        }
        Ok(())
    }

    pub(super) fn literal_component_count(&self) -> usize {
        match &self.meaning {
            WirePropertyMeaning::Constant { value } => value.component_payload_count(),
            WirePropertyMeaning::Analytic { .. } => 0,
            WirePropertyMeaning::Table { array, .. } => array
                .get("values")
                .and_then(serde_json::Value::as_array)
                .map_or(0, Vec::len),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source() -> ExprDag {
        let value_type = eqiora_core::ValueType::scalar(
            eqiora_core::ScalarDomain::Real,
            eqiora_core::DimExponents::DIMENSIONLESS,
        )
        .unwrap();
        let value = eqiora_core::ValueLiteral::from_real(value_type, 2.0).unwrap();
        let mut builder = ExprDagBuilder::new();
        let mut roots = Vec::new();
        for name in ["First", "Second"] {
            let release = PropertyRelease::new(
                ("exact::Contract".into(), format!("exact::{name}")),
                vec![],
                None,
                PropertyDerivatives::ValueOnly,
                ("org.example.source".into(), "spdx.CC0_1_0".into()),
                PropertyMeaning::Constant(value.clone()),
            )
            .unwrap();
            roots.push(builder.property(release, []).unwrap());
        }
        builder.finish(roots).unwrap()
    }

    #[test]
    fn replay_retains_equal_value_distinct_release_occurrences() {
        let source = source();
        let wire = WireExpression::encode(&source).unwrap();
        let bytes = serde_json::to_vec(&wire).unwrap();
        let reopened: WireExpression = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(reopened.decode().unwrap(), source);
        assert_eq!(reopened.properties.len(), 2);
        assert_ne!(
            reopened.properties[0].release,
            reopened.properties[1].release
        );
    }

    #[test]
    fn replay_rejects_substituted_value_duplicate_and_foreign_occurrences() {
        let original = WireExpression::encode(&source()).unwrap();
        let mut changed = original.clone();
        changed.properties[0].root = 999;
        assert!(changed.decode().is_err());
        let mut duplicate = original.clone();
        duplicate.properties.push(duplicate.properties[0].clone());
        assert!(duplicate.decode().is_err());
        let mut substituted = original;
        let value_type = eqiora_core::ValueType::scalar(
            eqiora_core::ScalarDomain::Real,
            eqiora_core::DimExponents::DIMENSIONLESS,
        )
        .unwrap();
        substituted.properties[0].meaning = WirePropertyMeaning::Constant {
            value: WireValueLiteral::encode(
                &eqiora_core::ValueLiteral::from_real(value_type, 3.0).unwrap(),
            )
            .unwrap(),
        };
        assert!(substituted.decode().is_err());
    }
    #[test]
    fn table_replay_authenticates_array_interval_and_generated_definition() {
        use eqiora_schema::kernel::pure_operator::ExactRational;
        use eqiora_schema::property_table::{AcceptedRealTable, RealTableProfile};
        use eqiora_schema::resolved_array::ResolvedF64Array;
        let dimension = eqiora_core::DimExponents::DIMENSIONLESS;
        let array = ResolvedF64Array::new(vec![2, 2], vec![0., 0., 1., 1.]).unwrap();
        let table = AcceptedRealTable::from_array(
            array,
            dimension,
            dimension,
            RealTableProfile::PiecewiseAffineOpenIntervalsV1,
            [
                ExactRational::new(0, 1).unwrap(),
                ExactRational::new(1, 1).unwrap(),
            ],
        )
        .unwrap();
        let release = PropertyRelease::new(
            ("Contract".into(), "Release".into()),
            vec!["x".into()],
            Some("single".into()),
            PropertyDerivatives::FirstOpenIntervals,
            ("citation".into(), "license".into()),
            PropertyMeaning::Table(Box::new(table)),
        )
        .unwrap();
        let mut builder = ExprDagBuilder::new();
        let argument = builder
            .push(ExprNode::Constant(
                eqiora_core::ValueLiteral::from_real(
                    eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, dimension)
                        .unwrap(),
                    0.5,
                )
                .unwrap(),
            ))
            .unwrap();
        let root = builder.property(release, [argument]).unwrap();
        let source = builder.finish([root]).unwrap();
        let original = WireExpression::encode(&source).unwrap();
        assert_eq!(original.decode().unwrap(), source);
        let changed_array = ResolvedF64Array::new(vec![2, 2], vec![0., 0., 1., 2.]).unwrap();
        let mut substituted = original.clone();
        let WirePropertyMeaning::Table { array, .. } = &mut substituted.properties[0].meaning
        else {
            panic!("table")
        };
        *array = serde_json::from_slice(&changed_array.canonical_json().unwrap()).unwrap();
        assert!(substituted.decode().is_err());
        // Even a matching replacement array digest cannot authorize the original law.
        let WirePropertyMeaning::Table { digest, .. } = &mut substituted.properties[0].meaning
        else {
            panic!("table")
        };
        *digest = crate::ArtifactDigest::from_sha256(changed_array.digest().unwrap()).to_string();
        let errors = substituted.decode().unwrap_err();
        assert!(
            errors
                .message()
                .contains("rederive the exact application definition"),
            "{errors:?}"
        );
        let mut altered_interval = original.clone();
        let WirePropertyMeaning::Table { validity, .. } =
            &mut altered_interval.properties[0].meaning
        else {
            panic!("table")
        };
        validity[0] = eqiora_schema::kernel::pure_operator::ExactRational::new(1, 4).unwrap();
        assert!(altered_interval.decode().is_err());
        let mut missing = original;
        let WirePropertyMeaning::Table { array, .. } = &mut missing.properties[0].meaning else {
            panic!("table")
        };
        *array = serde_json::Value::Null;
        assert!(missing.decode().is_err());
    }
}
