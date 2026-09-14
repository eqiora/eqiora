//! Homogeneous positional or named argument lists for shared calls.
use super::*;
impl Parser<'_> {
    pub(super) fn parse_call_arguments(
        &mut self,
        path: &NamePath,
    ) -> Option<(CallArguments, usize)> {
        let mut positional = Vec::new();
        let mut named = Vec::new();
        let mut child_depth = 0;
        if self.at(TokenKind::RightParen) && !matches!(path.as_str(), "boundaries" | "time") {
            self.error_here("operator call requires at least one argument");
            return None;
        }
        while !self.at(TokenKind::RightParen) {
            self.skip_trivia();
            let is_named = self.current().kind() == TokenKind::Identifier
                && self
                    .tokens
                    .iter()
                    .skip(self.cursor + 1)
                    .find(|token| !token.kind().is_trivia())
                    .is_some_and(|token| token.kind() == TokenKind::Equal);
            if is_named {
                if !positional.is_empty() {
                    self.error_here("cannot mix positional and named arguments");
                    return None;
                }
                let name = self.bump();
                if named
                    .iter()
                    .any(|binding: &NamedBindingDecl| binding.name() == name.text())
                {
                    self.error_token(&name, "duplicate named argument");
                    return None;
                }
                self.expect(TokenKind::Equal, "`=` after argument name")?;
                let (value, depth) = self.parse_expression_with_depth(0)?;
                child_depth = child_depth.max(depth);
                let range = TextRange::new(name.range().start(), value.range().end());
                named.push(NamedBindingDecl {
                    comments: Default::default(),
                    name: name.text().to_owned(),
                    value,
                    range,
                });
            } else {
                if !named.is_empty() {
                    self.error_here("cannot mix named and positional arguments");
                    return None;
                }
                let (value, depth) = self.parse_expression_with_depth(0)?;
                child_depth = child_depth.max(depth);
                positional.push(value);
            }
            if !self.at(TokenKind::Comma) {
                break;
            }
            self.bump();
            if self.at(TokenKind::RightParen) {
                self.error_here("expected argument after comma");
                return None;
            }
        }
        if path.as_str() == "tensor_value"
            && (named.len() != 2
                || !named.iter().any(|binding| binding.name() == "components")
                || !named.iter().any(|binding| {
                    binding.name() == "frame"
                        && matches!(
                            binding.value().kind(),
                            ExprKind::Name(_) | ExprKind::Path(_)
                        )
                }))
        {
            self.error_here("tensor_value requires frame = support_name, components = expression");
            return None;
        }
        Some((
            if named.is_empty() {
                CallArguments::Positional(positional)
            } else {
                CallArguments::Named(named)
            },
            child_depth,
        ))
    }
}
