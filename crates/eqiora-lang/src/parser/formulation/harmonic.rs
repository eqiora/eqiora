//! The fixed negative-exponential, peak harmonic request from the RC specimen.
use super::*;

impl Parser<'_> {
    pub(super) fn parse_harmonic_form(
        &mut self,
        start: u32,
        name: String,
        relations: Vec<String>,
    ) -> Option<FormulationDecl> {
        self.expect_keyword("harmonic")?;
        self.expect(TokenKind::LeftParen, "`(` before harmonic options")?;
        let mut frequency = None;
        let mut convention = false;
        let mut normalization = false;
        loop {
            let option = self.expect_identifier("harmonic option")?;
            self.expect(TokenKind::Equal, "`=` in harmonic option")?;
            match option.text() {
                "angular_frequency" if frequency.is_none() => {
                    frequency = Some(self.parse_expression(0)?);
                }
                "convention" if !convention => {
                    self.expect_keyword("negative_exponential")?;
                    convention = true;
                }
                "normalization" if !normalization => {
                    self.expect_keyword("peak")?;
                    normalization = true;
                }
                _ => {
                    self.error_token(&option, "unknown or repeated harmonic option");
                    return None;
                }
            }
            if !self.at(TokenKind::Comma) {
                break;
            }
            self.bump();
        }
        self.expect(TokenKind::RightParen, "`)` after harmonic options")?;
        self.expect(TokenKind::Semicolon, "`;` after harmonic request")?;
        if frequency.is_none() || !convention || !normalization {
            self.error_here("harmonic requires angular_frequency, convention and normalization");
            return None;
        }
        let mut excitations = Vec::new();
        let mut amplitudes = Vec::new();
        while !self.at(TokenKind::RightBrace) && !self.at(TokenKind::Eof) {
            if self.at_keyword("excitation") {
                self.bump();
                let input = self.expect_identifier("original input")?.text().to_owned();
                self.expect(TokenKind::Equal, "`=` before excitation amplitude")?;
                let value = self.parse_expression(0)?;
                self.expect(TokenKind::Semicolon, "`;` after excitation")?;
                excitations.push((input, value));
            } else if self.at_keyword("amplitude") {
                let amplitude_start = self.bump().range().start();
                let mut amplitude = self.parse_field_head(
                    amplitude_start,
                    crate::FieldRoleSyntax::Variable,
                    false,
                )?;
                if amplitude.activation != crate::ActivationSyntax::Continuous {
                    self.error_here("harmonic amplitudes have no temporal activation");
                    return None;
                }
                self.expect_keyword("for")?;
                let original = self
                    .expect_identifier("original unknown")?
                    .text()
                    .to_owned();
                let end = self
                    .expect(TokenKind::Semicolon, "`;` after amplitude")?
                    .range()
                    .end();
                amplitude.range = TextRange::new(amplitude_start, end);
                amplitudes.push((amplitude, original));
            } else {
                self.error_here("expected `excitation` or `amplitude` in harmonic request");
                return None;
            }
        }
        if amplitudes.is_empty() {
            self.error_here("harmonic request requires amplitude mappings");
            return None;
        }
        let end = self
            .expect(TokenKind::RightBrace, "`}` after harmonic request")?
            .range()
            .end();
        Some(FormulationDecl {
            comments: Default::default(),
            name,
            relations,
            binding: FormulationBinding::Harmonic {
                angular_frequency: frequency.expect("checked frequency option"),
                excitations,
                amplitudes,
            },
            equations: Vec::new(),
            gauge: None,
            range: TextRange::new(start, end),
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{FormulationBinding, SourceAstFactory, format, parse};

    fn specimen() -> &'static str {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../docs/language/harmonic-rc.md"
        ))
        .split_once("```eqiora\n")
        .unwrap()
        .1
        .split_once("\n```")
        .unwrap()
        .0
    }

    #[test]
    fn rc_request_retains_amplitudes_without_adding_original_unknowns() {
        let source = specimen();
        let document = parse("rc.eqi", source).into_document().unwrap();
        let model = &document.models()[0];
        let original = format!(
            "{} }}",
            source.split_once("  form harmonic_response").unwrap().0
        );
        let original = parse("rc.eqi", &original).into_document().unwrap();
        assert_eq!(model.items(), original.models()[0].items());
        let Some(FormulationBinding::Harmonic {
            excitations,
            amplitudes,
            ..
        }) = model.formulation_binding("harmonic_response")
        else {
            panic!("harmonic request");
        };
        assert_eq!(excitations.len(), 1);
        assert_eq!(excitations[0].0, "source");
        assert_eq!(
            amplitudes
                .iter()
                .map(|(a, original)| (a.name(), original.as_str()))
                .collect::<Vec<_>>(),
            [("voltage_hat", "voltage"), ("current_hat", "current")]
        );
        let (name, relations, equations, range) = model.formulations().next().unwrap();
        assert_eq!(relations, ["network"]);
        assert!(equations.is_empty());
        let native = SourceAstFactory::model_with_form(
            model.visibility(),
            model.name(),
            model.signature().to_vec(),
            model.items().to_vec(),
            (
                name.into(),
                relations.to_vec(),
                model.formulation_binding(name).unwrap().clone(),
            ),
            (Vec::new(), range),
            model.range(),
        )
        .unwrap();
        assert_eq!(
            native.formulation_binding(name),
            model.formulation_binding(name)
        );
        let formatted = format(&document);
        assert_eq!(
            format(&parse("rc.eqi", &formatted).into_document().unwrap()),
            formatted
        );
    }

    #[test]
    fn amplitude_notation_support_and_comments_survive_roundtrip() {
        let source = specimen().replace(
            "amplitude voltage_hat: complex<V> for voltage;",
            "// peak voltage\n    amplitude voltage_hat @{V}: complex<V> on body for voltage;",
        );
        let formatted = format(&parse("rc.eqi", &source).into_document().unwrap());
        assert!(formatted.contains("// peak voltage"));
        assert!(formatted.contains("amplitude voltage_hat @{V}: complex<V> on body for voltage;"));
        assert_eq!(
            format(&parse("rc.eqi", &formatted).into_document().unwrap()),
            formatted
        );
    }

    #[test]
    fn options_and_amplitude_ownership_syntax_are_closed() {
        for (from, to) in [
            ("angular_frequency = omega, ", ""),
            ("convention = negative_exponential, ", ""),
            (", normalization = peak", ""),
            ("angular_frequency", "cyclic_frequency"),
            ("negative_exponential", "positive_exponential"),
            ("normalization = peak", "normalization = rms"),
            (
                "normalization = peak",
                "normalization = peak, normalization = peak",
            ),
            (
                "complex<V> for voltage;",
                "complex<V> at clock for voltage;",
            ),
            ("for voltage;", "for voltage = 0;"),
        ] {
            assert!(
                parse("rc.eqi", &specimen().replace(from, to))
                    .into_document()
                    .is_err(),
                "{to}"
            );
        }
    }
}
