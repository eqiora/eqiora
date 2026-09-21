//! Closed authored-Formulation parsing.

use crate::ast::formulation::{FormulationBinding, FormulationDecl};
use crate::ast::{ComponentDecl, TextRange, VisibilitySyntax};
use crate::lexer::TokenKind;

use super::{ParsedComponentItem, Parser};

impl Parser<'_> {
    pub(super) fn parse_component(
        &mut self,
        start: u32,
        visibility: VisibilitySyntax,
    ) -> Option<ComponentDecl> {
        self.expect_keyword("component")?;
        let name = self.declaration_name("component name")?.text().to_owned();
        let signature = self.parse_signature()?;
        let mut items = Vec::new();
        self.expect(TokenKind::LeftBrace, "`{` after component name")?;
        let mut formulations = Vec::new();
        let mut formulations_started = false;
        while !self.at(TokenKind::RightBrace) && !self.at(TokenKind::Eof) {
            if self.at_keyword("form") {
                formulations_started = true;
                match self.parse_formulation() {
                    Some(formulation) => formulations.push(formulation),
                    None => self.recover_item(),
                }
                continue;
            }
            if formulations_started {
                self.error_here("component declarations must precede authored forms");
                self.recover_item();
                continue;
            }
            if self.at_component_property() {
                self.error_here("property requirements belong in the signature");
                self.recover_item();
                continue;
            }
            match self.parse_component_item() {
                Some(ParsedComponentItem::Retained(item)) => items.push(*item),
                Some(ParsedComponentItem::Discarded) => {}
                None => self.recover_item(),
            }
        }
        let end = self
            .expect(TokenKind::RightBrace, "`}` to close component")
            .map_or_else(|| self.current().range().end(), |token| token.range().end());
        Some(ComponentDecl {
            comments: Default::default(),
            visibility,
            name,
            signature,
            items,
            formulations,
            range: TextRange::new(start, end),
        })
    }

    pub(super) fn parse_formulation(&mut self) -> Option<FormulationDecl> {
        let start = self.expect_keyword("form")?.range().start();
        let name = self
            .expect_identifier("Formulation name")?
            .text()
            .to_owned();
        self.expect_keyword("for")?;
        let mut relations = Vec::new();
        loop {
            relations.push(
                self.expect_identifier("Formulation Relation")?
                    .text()
                    .to_owned(),
            );
            if !self.at(TokenKind::Comma) {
                break;
            }
            self.bump();
        }
        self.expect(TokenKind::LeftBrace, "`{` before authored Formulation")?;
        let binding = if self.at_keyword("interval") {
            self.bump();
            let name = self
                .expect_identifier("mathematical interval name")?
                .text()
                .to_owned();
            self.expect(TokenKind::LeftParen, "`(` before interval endpoints")?;
            let lower = self
                .expect_identifier("lower endpoint binder")?
                .text()
                .to_owned();
            self.expect(TokenKind::Comma, "`,` between endpoint binders")?;
            let upper = self
                .expect_identifier("upper endpoint binder")?
                .text()
                .to_owned();
            self.expect(TokenKind::RightParen, "`)` after interval endpoints")?;
            self.expect_keyword("on")?;
            let domain = self.expect_identifier("parent Domain")?.text().to_owned();
            self.expect(TokenKind::Semicolon, "`;` after interval binder")?;
            FormulationBinding::Interval {
                name,
                lower,
                upper,
                domain,
            }
        } else {
            let mut tests = Vec::new();
            loop {
                self.expect_keyword("test")?;
                let name = self
                    .expect_identifier("test-function name")?
                    .text()
                    .to_owned();
                self.expect(TokenKind::Colon, "`:` before test dimension")?;
                let dimension = self.parse_expression(0)?;
                if !matches!(dimension.kind(), crate::ast::ExprKind::Number(value) if value.to_i64().ok() == Some(1))
                {
                    self.error_here(
                        "weak forms require an explicit dimensionless test (`test w: 1`)",
                    );
                    return None;
                }
                self.expect_keyword("for")?;
                let trial = self
                    .expect_identifier("trial Field name")?
                    .text()
                    .to_owned();
                let mut zero_on = Vec::new();
                if self.at_keyword("zero_on") {
                    self.bump();
                    loop {
                        zero_on.push(
                            self.expect_identifier("essential test boundary")?
                                .text()
                                .to_owned(),
                        );
                        if !self.at(TokenKind::Comma) {
                            break;
                        }
                        self.bump();
                    }
                }
                self.expect(TokenKind::Semicolon, "`;` after test declaration")?;
                tests.push((name, trial, zero_on));
                if !self.at_keyword("test") {
                    break;
                }
            }
            FormulationBinding::WeakTests { tests }
        };
        let mut equations = Vec::new();
        loop {
            let left = self.parse_expression(0)?;
            self.expect(TokenKind::Equal, "`=` in authored Formulation")?;
            let right = self.parse_expression(0)?;
            self.expect(
                TokenKind::Semicolon,
                "`;` after authored Formulation equality",
            )?;
            equations.push((left, right));
            if self.at(TokenKind::RightBrace) {
                break;
            }
        }
        let end = self
            .expect(
                TokenKind::RightBrace,
                "`}` after authored Formulation equalities",
            )?
            .range()
            .end();
        Some(FormulationDecl {
            comments: Default::default(),
            name,
            binding,
            relations,
            equations,
            range: TextRange::new(start, end),
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::{ComponentItem, ExprKind};
    use crate::parse;

    #[test]
    fn retains_one_component_primal_form_outside_model_items() {
        let source = r#"
component Diffusion(
  support region: volume(ambient_dimension = 2)
) {
  variable potential: 1 on region;
  parameter diffusion: 1 = 1;
  parameter source: 1 / m ^ 2 = 1;
  relation balance on region { -div(diffusion * grad(potential)) = source; }
  form weak for balance { test w: 1 for potential zero_on surface;
    integrate(region, dot(grad(w), diffusion * grad(potential)))
      = integrate(region, w * source);
  }
}
"#;
        let document = parse("form.eqi", source).into_document().unwrap();
        let component = &document.components()[0];
        let forms = component.formulations().collect::<Vec<_>>();
        let [(_, relations, equations, _)] = forms.as_slice() else {
            panic!("one form expected")
        };
        assert_eq!(*relations, ["balance"]);
        let [(left, right)] = *equations else {
            panic!("one equality expected")
        };
        assert!(
            matches!(left.kind(), ExprKind::Call { callee, arguments } if callee.as_str() == "integrate" && arguments.expressions().len() == 2)
        );
        assert!(
            matches!(right.kind(), ExprKind::Call { callee, arguments } if callee.as_str() == "integrate" && arguments.expressions().len() == 2)
        );
        assert!(component.items().iter().all(
            |item| !matches!(item, ComponentItem::Relation(relation) if relation.name() == "primal")
        ));

        let misplaced = parse(
            "misplaced.eqi",
            "component C() { relation r { 1 = 0; } form weak for r { test w: 1 for x zero_on surface; integrate(d, test(x)) = integrate(d, test(x)); } parameter p: 1 = 1; }",
        );
        assert!(misplaced.diagnostics().iter().any(|diagnostic| {
            diagnostic
                .message()
                .contains("declarations must precede authored forms")
        }));
    }
    #[test]
    fn dimensionless_test_is_explicit_and_old_implicit_syntax_rejects() {
        for body in [
            "integrate(body, test(u)) = integrate(body, test(u));",
            "test w: K for u zero_on surface; integrate(body, w)=integrate(body,w);",
            "test w: 1.00000000000000000001 for u zero_on surface; integrate(body, w)=integrate(body,w);",
        ] {
            let source = format!("component C() {{ form weak for balance {{ {body} }} }}");
            assert!(parse("invalid.eqi", &source).into_document().is_err());
        }
    }
}
