//! Bounded tabular projection of accepted observations, not a Result artifact codec.
use super::CommonObservation;
use eqiora_compiler::UnitCatalog;
use eqiora_core::{
    Diagnostic, DimExponents, DynQuantity, ScalarDomain, ValueFrame, diagnostic::codes,
};
use eqiora_lang::Expr;
use serde_json::json;
use std::collections::BTreeMap;

impl CommonObservation {
    /// Maximum selected occurrences in one CSV projection.
    pub const CSV_MAX_ROWS: usize = 65_536;
    /// Maximum complete UTF-8 CSV size, including its header and escaped metadata.
    pub const CSV_MAX_BYTES: usize = 16 * 1024 * 1024;

    /// Export one explicitly selected component per observation, preserving order and duplicates.
    ///
    /// The profile accepts primal real/complex observations, including point samples
    /// and retained temporal outputs. It retains selected original real/imaginary
    /// values, intrinsic type/basis, support, exact source identities, effective
    /// quadrature and optional time. Projection and unit conversion delegate to the
    /// existing value and unit owners. Phase is principal radians and requires unit
    /// `1`; its undefined zero-amplitude value is an explicit empty value/status.
    ///
    /// This is a lossy tabular view: other components, the original Model/Mesh,
    /// full Result/history, and executable equations are not embedded. It is not a
    /// restart/import artifact. Direction-dependent observations reject because
    /// their complete directional provenance is not retained in this owner.
    /// Numbers use round-trip decimal spelling; display text additionally carries
    /// the explicit C locale and fixed decimal-place rounding policy.
    /// # Errors
    /// Rejects empty/oversized collections, invalid selections/units/presentation,
    /// non-primal observations, and output exceeding the fixed byte budget.
    pub fn export_csv<'a>(
        observations: impl IntoIterator<Item = &'a Self>,
        projection: &str,
        component: usize,
        unit: &Expr,
        decimal_places: u8,
        locale: &str,
    ) -> Result<Vec<u8>, Diagnostic> {
        export(
            observations,
            projection,
            component,
            unit,
            decimal_places,
            locale,
            Self::CSV_MAX_ROWS,
            Self::CSV_MAX_BYTES,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn export<'a>(
    observations: impl IntoIterator<Item = &'a CommonObservation>,
    projection: &str,
    component: usize,
    unit: &Expr,
    decimal_places: u8,
    locale: &str,
    max_rows: usize,
    max_bytes: usize,
) -> Result<Vec<u8>, Diagnostic> {
    let mut writer = Csv {
        bytes: Vec::new(),
        limit: max_bytes,
    };
    writer.record(&[
        "occurrence",
        "result_identity",
        "observable_id",
        "evaluation_kind",
        "value_type",
        "component",
        "source_real_si",
        "source_imaginary_si",
        "projection",
        "projection_dimension",
        "projected_value_si",
        "status",
        "angular_convention",
        "display_value",
        "display_unit",
        "display_text",
        "decimal_places",
        "rounding",
        "locale",
        "output_index",
        "time_s",
        "state_identity",
        "point_domain",
        "point_si",
        "quadratures",
    ])?;
    let unit_label = unit.to_source();
    if projection == "phase" && unit_label != "1" {
        return Err(error(
            "phase CSV presentation requires unit 1 and the explicit radians convention",
        ));
    }
    let mut count = 0;
    for (ordinal, observation) in observations.into_iter().enumerate() {
        if ordinal >= max_rows {
            return Err(error("CSV exceeds its selected observation budget"));
        }
        if observation.evaluation_kind() != "value" {
            return Err(error(
                "CSV requires primal observations with retained source provenance",
            ));
        }
        let projected = observation.project_component(projection, component)?;
        let dimension = projected.map_or(DimExponents::DIMENSIONLESS, |value| value.dim());
        // Validate the presentation even for undefined phase; do not silently
        // accept an incompatible unit/locale merely because there is no value.
        let (display_value, display_text) = UnitCatalog::present_quantity(
            projected.unwrap_or(DynQuantity::new(0., dimension)),
            unit,
            decimal_places,
            locale,
        )
        .map_err(error)?;
        let value = observation.value();
        let (real, imaginary) = value
            .component(component)
            .expect("projected numeric component");
        let ty = value.value_type();
        let bases = ty
            .finite_bases()
            .map(|basis| {
                basis
                    .atoms()
                    .map(|atom| {
                        json!({
                            "space": atom.space().expect("atomic basis").ulid().to_string(),
                            "extent": atom.extent(), "dual": atom.is_dual(),
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let value_type = json!({
            "scalar_domain": match ty.scalar_domain() { ScalarDomain::Real => "real", ScalarDomain::Complex => "complex", _ => unreachable!("numeric projection admitted") },
            "dimension": ty.dimension().exponents(),
            "shape": ty.shape().extents().iter().map(|extent| extent.get()).collect::<Vec<_>>(),
            "array_rank": ty.array_rank(),
            "frame": match ty.frame() { ValueFrame::Invariant => "invariant", ValueFrame::SpatialCartesian => "spatial-cartesian" },
            "basis_kind": if ty.map_bases().is_some() { "linear-map" } else if ty.coordinate_basis().is_some() { "coordinates" } else { "ordinary" },
            "bases": bases,
        });
        let output = observation.output();
        let point = observation.point();
        let coordinates = point.map(|(_, coordinates)| {
            coordinates
                .iter()
                .map(|value| {
                    json!({
                        "value_si": value.value(), "dimension": value.dim().exponents(),
                    })
                })
                .collect::<Vec<_>>()
        });
        let rules = observation.quadratures().iter().map(|(domain, rule)| {
            use eqiora_meshing::ReferenceCellFamily;
            (domain.ulid().to_string(), json!({
                "family": match rule.reference_cell().family() { ReferenceCellFamily::Point => "point", ReferenceCellFamily::Simplex => "simplex", ReferenceCellFamily::Hypercube => "hypercube" },
                "dimension": rule.reference_cell().dimension(),
                "polynomial_exactness": rule.polynomial_exactness(),
                "points": rule.points().iter().map(|point| json!({"coordinates": point.coordinates, "weight": point.weight})).collect::<Vec<_>>(),
            }))
        }).collect::<BTreeMap<_, _>>();
        let fields = [
            ordinal.to_string(),
            observation.result_identity().into(),
            observation.observable().ulid().to_string(),
            observation.evaluation_kind().into(),
            value_type.to_string(),
            component.to_string(),
            real.to_string(),
            imaginary.to_string(),
            projection.into(),
            json!(dimension.exponents()).to_string(),
            projected
                .map(|value| value.value().to_string())
                .unwrap_or_default(),
            if projected.is_some() {
                "defined"
            } else {
                "undefined"
            }
            .into(),
            if projection == "phase" { "radians" } else { "" }.into(),
            if projected.is_some() {
                display_value.to_string()
            } else {
                String::new()
            },
            unit_label.clone(),
            if projected.is_some() {
                display_text
            } else {
                String::new()
            },
            decimal_places.to_string(),
            UnitCatalog::PRESENTATION_ROUNDING.into(),
            locale.into(),
            output
                .map(|(index, _, _)| index.to_string())
                .unwrap_or_default(),
            output
                .map(|(_, time, _)| time.to_string())
                .unwrap_or_default(),
            output
                .map(|(_, _, state)| state.to_owned())
                .unwrap_or_default(),
            point
                .map(|(domain, _)| domain.ulid().to_string())
                .unwrap_or_default(),
            json!(coordinates).to_string(),
            json!(rules).to_string(),
        ];
        writer.record(&fields.iter().map(String::as_str).collect::<Vec<_>>())?;
        count += 1;
    }
    if count == 0 {
        return Err(error("CSV requires at least one selected observation"));
    }
    Ok(writer.bytes)
}

struct Csv {
    bytes: Vec<u8>,
    limit: usize,
}
impl Csv {
    fn record(&mut self, fields: &[&str]) -> Result<(), Diagnostic> {
        for (index, field) in fields.iter().enumerate() {
            // Quote every cell; all literal quotes are doubled, including JSON
            // metadata. Bound each escaped field before appending its content.
            let size = field
                .len()
                .checked_add(field.bytes().filter(|byte| *byte == b'"').count())
                .and_then(|size| size.checked_add(2 + usize::from(index != 0)))
                .ok_or_else(|| error("CSV field size overflows"))?;
            if self
                .bytes
                .len()
                .checked_add(size)
                .is_none_or(|size| size > self.limit)
            {
                return Err(error("CSV exceeds its output byte budget"));
            }
            if index != 0 {
                self.bytes.push(b',');
            }
            self.bytes.push(b'"');
            for byte in field.bytes() {
                self.bytes.push(byte);
                if byte == b'"' {
                    self.bytes.push(byte);
                }
            }
            self.bytes.push(b'"');
        }
        if self
            .bytes
            .len()
            .checked_add(2)
            .is_none_or(|size| size > self.limit)
        {
            return Err(error("CSV exceeds its output byte budget"));
        }
        self.bytes.extend_from_slice(b"\r\n");
        Ok(())
    }
}

fn error(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_EXTERNAL_DATA_EXPORT, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{Id, ValueLiteral, ValueType};
    use eqiora_lang::{ExprKind, SourceAstFactory, TextRange};
    use std::collections::HashMap;

    fn unit(name: &str) -> Expr {
        SourceAstFactory::expression(
            if name == "1" {
                ExprKind::Number(eqiora_lang::DecimalLiteral::parse("1").unwrap())
            } else {
                ExprKind::Name(name.into())
            },
            TextRange::new(0, 0),
        )
        .unwrap()
    }

    fn observation(real: f64, imaginary: f64) -> CommonObservation {
        CommonObservation {
            evaluation_kind: "value",
            observable: Id::from_ulid("00000000000000000000000001".parse().unwrap()),
            result_identity: "a".repeat(64),
            value: ValueLiteral::new(
                ValueType::scalar(
                    ScalarDomain::Complex,
                    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap(),
                )
                .unwrap(),
                [(real, imaginary)],
            )
            .unwrap(),
            output: Some((0, 0.25, "b".repeat(64))),
            point: None,
            quadratures: HashMap::new(),
        }
    }

    #[test]
    fn csv_quotes_and_byte_limits_follow_the_complete_encoded_record() {
        let mut csv = Csv {
            bytes: Vec::new(),
            limit: 100,
        };
        csv.record(&["quoted,\"text", "line\nbreak", ""]).unwrap();
        assert_eq!(csv.bytes, b"\"quoted,\"\"text\",\"line\nbreak\",\"\"\r\n");
        let mut exact = Csv {
            bytes: Vec::new(),
            limit: 5,
        };
        exact.record(&["x"]).unwrap();
        assert_eq!(exact.bytes, b"\"x\"\r\n");
        assert!(
            Csv {
                bytes: Vec::new(),
                limit: 4
            }
            .record(&["x"])
            .is_err()
        );
    }

    #[test]
    fn typed_csv_uses_shared_components_units_and_explicit_missing_phase() {
        let value = observation(3., 4.);
        let before = value.clone();
        let bytes =
            CommonObservation::export_csv([&value, &value], "magnitude", 0, &unit("mm"), 3, "C")
                .unwrap();
        let text = String::from_utf8(bytes).unwrap();
        assert_eq!(text.matches("5000.000 mm").count(), 2);
        assert!(text.contains("\"3\",\"4\",\"magnitude\""));
        assert_eq!(value, before);
        let zero = observation(0., 0.);
        let phase = CommonObservation::export_csv([&zero], "phase", 0, &unit("1"), 3, "C").unwrap();
        assert!(
            String::from_utf8(phase)
                .unwrap()
                .contains("\"undefined\",\"radians\",\"\",\"1\",\"\"")
        );
        assert!(CommonObservation::export_csv([&zero], "phase", 0, &unit("m"), 3, "C").is_err());
        assert!(
            CommonObservation::export_csv([&zero], "phase", 0, &unit("1"), 3, "de_DE").is_err()
        );
        assert!(
            CommonObservation::export_csv([&value], "magnitude", 1, &unit("mm"), 3, "C").is_err()
        );
        assert!(
            CommonObservation::export_csv([&value], "magnitude", 0, &unit("Pa"), 3, "C").is_err()
        );
        assert!(CommonObservation::export_csv([], "real", 0, &unit("m"), 3, "C").is_err());
        assert!(export([&value, &value], "real", 0, &unit("m"), 3, "C", 1, 1 << 20).is_err());
        assert!(export([&value], "real", 0, &unit("m"), 3, "C", 1, 10).is_err());
        let mut derivative = value;
        derivative.evaluation_kind = "state-jvp";
        assert!(
            CommonObservation::export_csv([&derivative], "real", 0, &unit("m"), 3, "C").is_err()
        );
    }
}
