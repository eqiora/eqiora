//! Exact package data references resolve before entering the value namespace.
use super::*;
use eqiora_core::ScalarDomain;
use eqiora_lang::PropertyTableSyntax;
use eqiora_schema::property_table::{AcceptedRealTable, RealTableProfile};

pub(super) fn compile(
    unit: &AnalyzedSourceUnit,
    table: &PropertyTableSyntax,
    contract: &Contract,
    scale: &Expr,
) -> Result<PropertyMeaning, Diagnostic> {
    let fail = |message: &str| error(&unit.file, table.range(), message);
    let [(input, syntax)] = contract.inputs.as_slice() else {
        return Err(fail(
            "table requires one exact real scalar independent input",
        ));
    };
    if input != table.axis() {
        return Err(fail("table axis must name its exact contract input"));
    }
    let axis_type = crate::value_types::lower_value_type::<()>(&contract.file, syntax, None)?;
    let result_type = contract.result_type::<()>(None)?;
    if [&axis_type, &result_type]
        .iter()
        .any(|value| value.scalar_domain() != ScalarDomain::Real || !value.shape().is_scalar())
    {
        return Err(fail(
            "table input and result must be concrete real scalar types",
        ));
    }
    if lower_dimension(&unit.file, table.axis_dimension())? != axis_type.dimension()
        || lower_dimension(&unit.file, table.value_dimension())? != result_type.dimension()
    {
        return Err(fail(
            "table column dimensions differ from the exact contract",
        ));
    }
    if constant(&unit.file, scale)? != 1.0 {
        return Err(fail(
            "identity-preprocessed table data requires coherent-SI scale one",
        ));
    }
    let array = unit
        .resolved_arrays
        .get(table.data().as_str())
        .ok_or_else(|| {
            fail("exact table asset is absent from this package's resolved-array closure")
        })?;
    // Table admission bounds rows and scalar count before retaining a release copy.
    if array.shape().len() != 2
        || array.values().len() > 2 * eqiora_schema::kernel::property_table::MAX_TABLE_POINTS
    {
        return Err(fail("table array exceeds the exact row-major table bounds"));
    }
    let validity = table
        .validity()
        .iter()
        .map(|endpoint| {
            let bound = crate::hierarchy::closed_value(&unit.file, endpoint, axis_type.clone())?;
            let value = bound
                .real_scalar_value()
                .ok_or_else(|| fail("table validity endpoint must be a real scalar quantity"))?;
            eqiora_schema::kernel::pure_operator::ExactRational::from_binary64(value.value())
                .ok_or_else(|| fail("table validity endpoint exceeds exact arithmetic bounds"))
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    let accepted = AcceptedRealTable::from_array(
        array.clone(),
        axis_type.dimension(),
        result_type.dimension(),
        RealTableProfile::PiecewiseAffineOpenIntervalsV1,
        [validity[0].clone(), validity[1].clone()],
    )
    .map_err(|diagnostic| fail(diagnostic.message()))?;
    Ok(PropertyMeaning::Table(Box::new(accepted)))
}

/// Resolve attribution only from the declaring package's admitted document closure.
pub(super) fn attribution(
    unit: &AnalyzedSourceUnit,
    name: &NamePath,
) -> Result<String, Diagnostic> {
    use sha2::{Digest, Sha256};
    let bytes = unit.resolved_documents.get(name.as_str()).ok_or_else(|| {
        error(
            &unit.file,
            name.range(),
            "table attribution must name an admitted document owned by this package",
        )
    })?;
    let digest = Sha256::digest(bytes.as_ref());
    let hex = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    // Canonical package name avoids embedding the package's own eventual digest.
    Ok(format!(
        "{}::documentation:docs/{}.md#sha256:{hex}",
        unit.module.owner().package_name(),
        name.as_str().replace('.', "/")
    ))
}

/// Extend the existing declaration identity with only its referenced admitted assets.
pub(crate) fn canonical_asset_identity(
    unit: &AnalyzedSourceUnit,
    table: &PropertyTableSyntax,
    citation: &NamePath,
    license: &NamePath,
    declaration: &str,
) -> Result<String, Diagnostic> {
    use sha2::{Digest, Sha256};
    let array = unit
        .resolved_arrays
        .get(table.data().as_str())
        .ok_or_else(|| {
            error(
                &unit.file,
                table.range(),
                "exact table asset is absent from this package's resolved-array closure",
            )
        })?;
    let citation = attribution(unit, citation)?;
    let license = attribution(unit, license)?;
    let array = array.canonical_json()?;
    let mut hash = Sha256::new();
    for bytes in [
        declaration.as_bytes(),
        array.as_slice(),
        citation.as_bytes(),
        license.as_bytes(),
    ] {
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    Ok(format!(
        "eqiora.source-declaration.v1:sha256:{}",
        hash.finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    ))
}
