//! One checked nominal property catalog reused by authored and selected bindings.
use super::*;
#[derive(Clone, Debug, Default)]
pub(crate) struct Catalog {
    pub(crate) contracts: BTreeMap<Key, Contract>,
    pub(crate) releases: BTreeMap<Key, Release>,
    pub(super) compositions: BTreeMap<Key, Composition>,
}
pub(crate) fn build(
    units: &[AnalyzedSourceUnit],
    aliases: &[ResolvedAlias],
) -> Result<Catalog, Vec<Diagnostic>> {
    let mut diagnostics = Vec::new();
    let mut contracts = BTreeMap::new();
    for unit in units.iter() {
        for ((visibility, name, value_type, range), (_, inputs, derivatives, branch)) in unit
            .document
            .property_contract_syntax()
            .zip(unit.document.property_contract_profiles())
        {
            if name == crate::math::ROOT {
                diagnostics.push(error(
                    &unit.file,
                    range,
                    "identifier `math` is reserved for compiler-owned scalar mathematics",
                ));
                continue;
            }
            let contract = Contract {
                file: unit.file.clone(),
                visibility,
                value_type: value_type.clone(),
                inputs: inputs.to_vec(),
                derivatives,
                branch: branch.map(ToString::to_string),
            };
            if let Err(diagnostic) = contract.result_type::<()>(None) {
                diagnostics.push(diagnostic);
                continue;
            }
            let key = (unit.module.clone(), name.to_owned());
            if contracts.insert(key, contract).is_some() {
                diagnostics.push(error(
                    &unit.file,
                    range,
                    format!("duplicate property contract `{name}`"),
                ));
            }
        }
    }

    let mut releases = BTreeMap::new();
    for unit in units.iter() {
        for (
            visibility,
            name,
            contract_path,
            source_value_expr,
            source_dimension_expr,
            scale_expr,
            citation,
            license,
            range,
        ) in unit.document.property_release_syntax()
        {
            if name == crate::math::ROOT {
                diagnostics.push(error(
                    &unit.file,
                    range,
                    "identifier `math` is reserved for compiler-owned scalar mathematics",
                ));
                continue;
            }
            let Some(contract_key) = resolve_path(
                &unit.module,
                contract_path,
                aliases,
                &contracts,
                |value| value.visibility,
                &unit.file,
                &mut diagnostics,
            ) else {
                continue;
            };
            let contract = &contracts[&contract_key];
            let contract_type = match contract.result_type::<()>(None) {
                Ok(value) => value,
                Err(value) => {
                    diagnostics.push(value);
                    continue;
                }
            };
            let source_dimension = match source_dimension_expr {
                Some(expression) => match lower_dimension(&unit.file, expression) {
                    Ok(value) => value,
                    Err(diagnostic) => {
                        diagnostics.push(diagnostic);
                        continue;
                    }
                },
                None => contract_type.dimension(),
            };
            if source_dimension != contract_type.dimension() {
                diagnostics.push(error(
                    &unit.file,
                    source_dimension_expr.map_or(range, Expr::range),
                    "property release source unit does not match its contract dimension",
                ));
                continue;
            }
            let (_, validity, branch) = unit
                .document
                .property_release_profiles()
                .find(|profile| profile.0 == name)
                .expect("one profile per declaration");
            let expected_branch = contract.branch.as_deref().or(Some("single"));
            if expected_branch != branch.map(NamePath::as_str) {
                diagnostics.push(error(
                    &unit.file,
                    range,
                    "property release branch does not match its nominal contract",
                ));
                continue;
            }
            let attribution = if matches!(
                source_value_expr,
                eqiora_lang::PropertySourceSyntax::Table(_)
            ) {
                super::table::attribution(unit, citation).and_then(|citation| {
                    super::table::attribution(unit, license).map(|license| (citation, license))
                })
            } else {
                Ok((citation.to_string(), license.to_string()))
            };
            let (citation, license) = match attribution {
                Ok(value) => value,
                Err(diagnostic) => {
                    diagnostics.push(diagnostic);
                    continue;
                }
            };
            if !contract.inputs.is_empty() {
                let scale = constant(&unit.file, scale_expr);
                if !matches!(scale, Ok(value) if value.is_finite() && value > 0.0) {
                    diagnostics.push(error(
                        &unit.file,
                        scale_expr.range(),
                        "coherent-SI scale must be finite and strictly positive",
                    ));
                    continue;
                }
                let meaning = match source_value_expr {
                    eqiora_lang::PropertySourceSyntax::Expression(value) => {
                        crate::pure_operator::property::compile_property(
                            &unit.file,
                            &unit.document,
                            &contract.inputs,
                            &contract.value_type,
                            value,
                            scale_expr,
                            validity,
                        )
                        .map(PropertyMeaning::Analytic)
                    }
                    eqiora_lang::PropertySourceSyntax::Table(table) => {
                        super::table::compile(unit, table, contract, scale_expr)
                    }
                };
                match meaning {
                    Ok(meaning) => {
                        let key = (unit.module.clone(), name.to_owned());
                        match eqiora_schema::kernel::PropertyRelease::new(
                            (qualified(&contract_key), qualified(&key)),
                            contract
                                .inputs
                                .iter()
                                .map(|(name, _)| name.clone())
                                .collect(),
                            expected_branch.map(str::to_owned),
                            contract.derivatives,
                            (citation.to_string(), license.to_string()),
                            meaning.clone(),
                        ) {
                            Ok(_) => {
                                if releases
                                    .insert(
                                        key,
                                        Release {
                                            visibility,
                                            contract: contract_key,
                                            meaning,
                                            branch: expected_branch.map(str::to_owned),
                                            citation: citation.to_string(),
                                            license: license.to_string(),
                                        },
                                    )
                                    .is_some()
                                {
                                    diagnostics.push(error(
                                        &unit.file,
                                        range,
                                        format!("duplicate property release `{name}`"),
                                    ));
                                }
                            }
                            Err(diagnostic) => {
                                diagnostics.push(error(&unit.file, range, diagnostic.message()))
                            }
                        }
                    }
                    Err(diagnostic) => diagnostics.push(diagnostic),
                }
                continue;
            }
            if contract.derivatives != eqiora_schema::kernel::PropertyDerivatives::ValueOnly
                || validity.is_some()
            {
                diagnostics.push(error(&unit.file, range, "constant property requires unconditional validity and no input partial products"));
                continue;
            }
            let eqiora_lang::PropertySourceSyntax::Expression(source_value_expr) =
                source_value_expr
            else {
                diagnostics.push(error(
                    &unit.file,
                    range,
                    "table requires one independent input",
                ));
                continue;
            };
            let source_value = match crate::hierarchy::closed_value(
                &unit.file,
                source_value_expr,
                contract_type.clone(),
            ) {
                Ok(value) => value,
                Err(value) => {
                    diagnostics.push(value);
                    continue;
                }
            };
            let scale = match constant(&unit.file, scale_expr) {
                Ok(value) if value.is_finite() && value > 0.0 => value,
                Ok(_) => {
                    diagnostics.push(error(
                        &unit.file,
                        scale_expr.range(),
                        "coherent-SI scale must be finite and strictly positive",
                    ));
                    continue;
                }
                Err(value) => {
                    diagnostics.push(value);
                    continue;
                }
            };
            let normalized = if source_value.value_type().scalar_domain()
                == eqiora_core::ScalarDomain::Integer
            {
                if scale != 1.0 {
                    Err("integer property values require unit scale one")
                } else {
                    crate::typed_values::retype(&source_value, contract_type.clone())
                        .map_err(|_| "invalid integer property type")
                }
            } else {
                source_value
                    .components()
                    .expect("real or complex property")
                    .map(|(real, imag)| {
                        Ok((
                            crate::units::normalize_value(real, scale)?,
                            crate::units::normalize_value(imag, scale)?,
                        ))
                    })
                    .collect::<Result<Vec<_>, &'static str>>()
                    .and_then(|values| {
                        eqiora_core::ValueLiteral::new(contract_type.clone(), values)
                            .map_err(|_| "invalid normalized property value")
                    })
            };
            let value = match normalized {
                Ok(value) => value,
                Err(message) => {
                    diagnostics.push(error(&unit.file, range, message));
                    continue;
                }
            };
            let key = (unit.module.clone(), name.to_owned());
            if releases
                .insert(
                    key,
                    Release {
                        visibility,
                        contract: contract_key,
                        meaning: PropertyMeaning::Constant(value),
                        branch: expected_branch.map(str::to_owned),
                        citation: citation.to_string(),
                        license: license.to_string(),
                    },
                )
                .is_some()
            {
                diagnostics.push(error(
                    &unit.file,
                    range,
                    format!("duplicate property release `{name}`"),
                ));
            }
        }
    }

    if !diagnostics.is_empty() {
        return Err(diagnostics);
    }

    let mut compositions = BTreeMap::new();
    for unit in units.iter() {
        for (visibility, name, properties, range) in unit.document.material_composition_syntax() {
            if properties.is_empty() {
                diagnostics.push(error(
                    &unit.file,
                    range,
                    "material composition requires at least one property",
                ));
                continue;
            }
            let mut seen = BTreeSet::new();
            let mut resolved = Vec::new();
            for (property, release_path, binding_range) in properties {
                if !seen.insert(property) {
                    diagnostics.push(error(
                        &unit.file,
                        binding_range,
                        format!("duplicate material property `{property}`"),
                    ));
                    continue;
                }
                if let Some(release) = resolve_path(
                    &unit.module,
                    release_path,
                    aliases,
                    &releases,
                    |value| value.visibility,
                    &unit.file,
                    &mut diagnostics,
                ) {
                    resolved.push((property.to_owned(), release, binding_range));
                }
            }
            let key = (unit.module.clone(), name.to_owned());
            if compositions
                .insert(
                    key,
                    Composition {
                        visibility,
                        properties: resolved,
                    },
                )
                .is_some()
            {
                diagnostics.push(error(
                    &unit.file,
                    range,
                    format!("duplicate material composition `{name}`"),
                ));
            }
        }
    }

    if !diagnostics.is_empty() {
        return Err(diagnostics);
    }

    Ok(Catalog {
        contracts,
        releases,
        compositions,
    })
}

impl Catalog {
    pub(crate) fn contract(
        &self,
        namespace: &CompilationModuleId,
        path: &NamePath,
        aliases: &[ResolvedAlias],
        file: &str,
    ) -> Result<&Contract, Diagnostic> {
        let mut diagnostics = Vec::new();
        let key = resolve_path(
            namespace,
            path,
            aliases,
            &self.contracts,
            |value| value.visibility,
            file,
            &mut diagnostics,
        )
        .ok_or_else(|| {
            diagnostics
                .into_iter()
                .next()
                .expect("failed resolution diagnostic")
        })?;
        Ok(&self.contracts[&key])
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn bind(
        &self,
        requirement_namespace: &CompilationModuleId,
        requirement: &eqiora_lang::ComponentPropertyDecl,
        binding_namespace: &CompilationModuleId,
        value: &Expr,
        aliases: &[ResolvedAlias],
        file: &str,
    ) -> Result<eqiora_schema::kernel::PropertyRelease, Vec<Diagnostic>> {
        let path = match value.kind() {
            eqiora_lang::ExprKind::Name(name) => {
                NamePath::from_segments([name], value.range()).expect("parsed name")
            }
            eqiora_lang::ExprKind::Path(path) => path.clone(),
            _ => {
                return Err(vec![error(
                    file,
                    value.range(),
                    "property binding requires an exact nominal release or composition member",
                )]);
            }
        };
        let mut diagnostics = Vec::new();
        let required = resolve_path(
            requirement_namespace,
            requirement.contract(),
            aliases,
            &self.contracts,
            |value| value.visibility,
            file,
            &mut diagnostics,
        );
        let supplied = resolve_property_value(
            binding_namespace,
            &path,
            aliases,
            &self.releases,
            &self.compositions,
            file,
            &mut diagnostics,
        );
        if !diagnostics.is_empty() {
            return Err(diagnostics);
        }
        let required = required.expect("resolved contract");
        let (supplied, composition) = supplied.expect("resolved release");
        let contract = &self.contracts[&required];
        let release = &self.releases[&supplied];
        if required != release.contract {
            return Err(vec![error(
                file,
                value.range(),
                "property release implements a different nominal contract",
            )]);
        }
        eqiora_schema::kernel::PropertyRelease::new(
            (qualified(&required), qualified(&supplied)),
            contract
                .inputs
                .iter()
                .map(|(name, _)| name.clone())
                .collect(),
            release.branch.clone(),
            contract.derivatives,
            (release.citation.clone(), release.license.clone()),
            release.meaning.clone(),
        )
        .and_then(|release| release.with_composition(composition.as_ref().map(qualified)))
        .map_err(|diagnostic| vec![error(file, value.range(), diagnostic.message())])
    }

    pub(crate) fn contract_identity(
        &self,
        namespace: &CompilationModuleId,
        path: &NamePath,
        aliases: &[ResolvedAlias],
        file: &str,
    ) -> Result<String, Diagnostic> {
        let mut diagnostics = Vec::new();
        let key = resolve_path(
            namespace,
            path,
            aliases,
            &self.contracts,
            |value| value.visibility,
            file,
            &mut diagnostics,
        )
        .ok_or_else(|| {
            diagnostics
                .into_iter()
                .next()
                .expect("failed resolution diagnostic")
        })?;
        Ok(qualified(&key))
    }
}
