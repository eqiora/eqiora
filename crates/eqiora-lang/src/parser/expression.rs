//! Ordered expression parsing with bounded recursion and constructed-tree depth.

use super::*;

// Matches the compiler's existing canonical expression-depth admission. This
// guard runs before allocating a deeper tree or entering another recursive call.
const MAX_EXPRESSION_DEPTH: usize = 256;

impl Parser<'_> {
    pub(super) fn parse_exact_expression(&mut self) -> Option<Expr> {
        let previous = self.exact_numeric;
        self.exact_numeric = true;
        let result = self.parse_expression(0);
        self.exact_numeric = previous;
        result
    }

    pub(super) fn parse_expression(&mut self, minimum_binding_power: u8) -> Option<Expr> {
        self.parse_expression_with_depth(minimum_binding_power)
            .map(|(expression, _)| expression)
    }

    pub(super) fn parse_expression_with_depth(
        &mut self,
        minimum_binding_power: u8,
    ) -> Option<(Expr, usize)> {
        if self.expression_recursion >= MAX_EXPRESSION_DEPTH {
            self.error_here("expression recursion exceeds the 256-level limit");
            return None;
        }
        self.expression_recursion += 1;
        let parsed = self.parse_expression_inner(minimum_binding_power);
        self.expression_recursion -= 1;
        parsed
    }

    pub(super) fn parent_depth(&mut self, child_depth: usize) -> Option<usize> {
        if child_depth >= MAX_EXPRESSION_DEPTH {
            self.error_here("expression tree exceeds the 256-level limit");
            None
        } else {
            Some(child_depth + 1)
        }
    }

    fn parse_expression_inner(&mut self, minimum_binding_power: u8) -> Option<(Expr, usize)> {
        if minimum_binding_power > 0 && (self.at_keyword("if") || self.at_keyword("case")) {
            self.error_here("conditional operand requires parentheses");
            return None;
        }
        if minimum_binding_power > 5 && self.at_keyword("not") {
            self.error_here(
                "Boolean negation in an arithmetic or comparison operand requires parentheses",
            );
            return None;
        }
        let (mut left, mut depth) = self.parse_primary()?;
        let mut compared = false;
        loop {
            if self.at(TokenKind::Dot) {
                if !matches!(
                    left.kind(),
                    ExprKind::Index { .. }
                        | ExprKind::Member { .. }
                        | ExprKind::BoundaryPortSelection { .. }
                        | ExprKind::Name(_)
                        | ExprKind::Path(_)
                ) {
                    self.error_here(
                        "member access requires a Port or indexed component occurrence",
                    );
                    return None;
                }
                self.bump();
                let member = self.expect_identifier("indexed occurrence member")?;
                depth = self.parent_depth(depth)?;
                let range = TextRange::new(left.range.start(), member.range().end());
                left = Expr {
                    resolved_enum: None,
                    resolved_nominal: None,
                    kind: ExprKind::Member {
                        value: Box::new(left),
                        member: member.text().to_owned(),
                    },
                    range,
                };
                continue;
            }
            if self.at(TokenKind::LeftBracket) {
                self.bump();
                let (index, index_depth) = self.parse_expression_with_depth(0)?;
                depth = self.parent_depth(depth.max(index_depth))?;
                let upper = if self.at(TokenKind::Colon) {
                    self.bump();
                    let (upper, upper_depth) = self.parse_expression_with_depth(0)?;
                    depth = depth.max(self.parent_depth(upper_depth)?);
                    Some(upper)
                } else {
                    None
                };
                let end = self
                    .expect(TokenKind::RightBracket, "`]` after index")?
                    .range()
                    .end();
                let range = TextRange::new(left.range.start(), end);
                left = Expr {
                    resolved_enum: None,
                    resolved_nominal: None,
                    kind: match upper {
                        Some(upper) => ExprKind::Slice {
                            value: Box::new(left),
                            lower: Box::new(index),
                            upper: Box::new(upper),
                        },
                        None => ExprKind::Index {
                            value: Box::new(left),
                            index: Box::new(index),
                        },
                    },
                    range,
                };
                continue;
            }
            let token = self.current().clone();
            let angle_equal = matches!(token.kind(), TokenKind::LeftAngle | TokenKind::RightAngle)
                && self.tokens.get(self.cursor + 1).is_some_and(|next| {
                    next.kind() == TokenKind::Equal && token.range().end() == next.range().start()
                });
            let (operator, left_power, right_power) = match token.kind() {
                TokenKind::Identifier if token.text() == "or" => (BinaryOp::Or, 1, 2),
                TokenKind::Identifier if token.text() == "and" => (BinaryOp::And, 3, 4),
                TokenKind::EqualEqual => (BinaryOp::Equal, 7, 8),
                TokenKind::NotEqual => (BinaryOp::NotEqual, 7, 8),
                TokenKind::LeftAngle => (
                    if angle_equal {
                        BinaryOp::LessEqual
                    } else {
                        BinaryOp::Less
                    },
                    7,
                    8,
                ),
                TokenKind::RightAngle => (
                    if angle_equal {
                        BinaryOp::GreaterEqual
                    } else {
                        BinaryOp::Greater
                    },
                    7,
                    8,
                ),
                TokenKind::Plus => (BinaryOp::Add, 9, 10),
                TokenKind::Minus => (BinaryOp::Sub, 9, 10),
                TokenKind::Star => (BinaryOp::Mul, 11, 12),
                TokenKind::Slash => (BinaryOp::Div, 11, 12),
                TokenKind::Caret => (BinaryOp::Pow, 15, 14),
                _ => break,
            };
            if left_power < minimum_binding_power {
                break;
            }
            if left_power == 7 {
                if compared {
                    self.error_here("comparison chaining requires explicit Boolean composition");
                    return None;
                }
                compared = true;
            }
            self.bump();
            if angle_equal {
                self.bump();
            }
            let (right, right_depth) = self.parse_expression_with_depth(right_power)?;
            depth = self.parent_depth(depth.max(right_depth))?;
            let range = TextRange::new(left.range.start(), right.range.end());
            left = Expr {
                resolved_enum: None,
                resolved_nominal: None,
                kind: ExprKind::Binary {
                    op: operator,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                range,
            };
        }
        Some((left, depth))
    }
    fn parse_primary(&mut self) -> Option<(Expr, usize)> {
        if self.at_keyword("if") {
            return self.parse_select();
        }
        if self.at_keyword("case") {
            return self.parse_case();
        }
        if self.at_keyword("then") || self.at_keyword("else") {
            self.error_here("expected value expression before conditional branch keyword");
            return None;
        }
        let result = if self.at(TokenKind::Minus) {
            let start = self.bump().range().start();
            // Power binds inside unary minus, including a signed right power:
            // -x^2 is -(x^2), while x^-2 is x^(-2).
            let (value, child_depth) = self.parse_expression_with_depth(14)?;
            let depth = self.parent_depth(child_depth)?;
            (
                Expr {
                    resolved_enum: None,
                    resolved_nominal: None,
                    range: TextRange::new(start, value.range.end()),
                    kind: ExprKind::Unary {
                        op: UnaryOp::Neg,
                        value: Box::new(value),
                    },
                },
                depth,
            )
        } else if self.at_keyword("not") {
            let start = self.bump().range().start();
            let (value, child_depth) = self.parse_expression_with_depth(5)?;
            let depth = self.parent_depth(child_depth)?;
            (
                Expr {
                    resolved_enum: None,
                    resolved_nominal: None,
                    range: TextRange::new(start, value.range().end()),
                    kind: ExprKind::Unary {
                        op: UnaryOp::Not,
                        value: Box::new(value),
                    },
                },
                depth,
            )
        } else if self.at_keyword("true") || self.at_keyword("false") {
            let token = self.bump();
            (
                Expr {
                    resolved_enum: None,
                    resolved_nominal: None,
                    range: token.range(),
                    kind: ExprKind::Boolean(token.text() == "true"),
                },
                1,
            )
        } else if self.at(TokenKind::LeftBracket) {
            return self.parse_array();
        } else if self.at(TokenKind::Number) {
            let expression = self.parse_quantity_or_number()?;
            // The unit grammar owns the separate dimension tree; the quantity
            // is one scalar operand in this value-expression depth bound.
            (expression, 1)
        } else if self.at(TokenKind::Identifier) {
            return self.parse_named();
        } else if self.at(TokenKind::LeftParen) {
            return self.parse_group();
        } else {
            self.error_here("expected expression");
            return None;
        };
        Some(result)
    }

    fn parse_array(&mut self) -> Option<(Expr, usize)> {
        let result = {
            let start = self.bump().range().start();
            if self.at(TokenKind::RightBracket) {
                self.error_here("array literal requires at least one element");
                return None;
            }
            let (first, mut child_depth) = self.parse_expression_with_depth(0)?;
            let mut elements = vec![first];
            while self.at(TokenKind::Comma) {
                self.bump();
                let (element, depth) = self.parse_expression_with_depth(0)?;
                child_depth = child_depth.max(depth);
                elements.push(element);
            }
            let depth = self.parent_depth(child_depth)?;
            let end = self
                .expect(TokenKind::RightBracket, "`]` after array elements")?
                .range()
                .end();
            (
                Expr {
                    resolved_enum: None,
                    resolved_nominal: None,
                    kind: ExprKind::Array(elements),
                    range: TextRange::new(start, end),
                },
                depth,
            )
        };
        Some(result)
    }

    fn parse_reduction(&mut self, path: NamePath) -> Option<(Expr, usize)> {
        let operation = match path.as_str() {
            "sum" => crate::ReductionOp::Sum,
            "product" => crate::ReductionOp::Product,
            "min" => crate::ReductionOp::Min,
            "max" => crate::ReductionOp::Max,
            _ => unreachable!("recognized finite reduction"),
        };
        self.expect(TokenKind::LeftParen, "`(` after reduction operation")?;
        let (value, child_depth) = self.parse_expression_with_depth(0)?;
        self.expect(TokenKind::Comma, "`,` before reduction binder")?;
        if !self.at_keyword("over") {
            self.error_here("reduction requires `over = (member in IndexSet)`");
            return None;
        }
        self.bump();
        self.expect(TokenKind::Equal, "`=` after `over`")?;
        let start = self
            .expect(TokenKind::LeftParen, "`(` before reduction binder")?
            .range()
            .start();
        let member = self.expect_identifier("reduction member")?;
        if !self.at_keyword("in") {
            self.error_here("reduction binder requires `in`");
            return None;
        }
        self.bump();
        let set = self.parse_name_path("reduction index set")?;
        let end = self
            .expect(TokenKind::RightParen, "`)` after reduction binder")?
            .range()
            .end();
        let binder = FamilyBinderSyntax {
            member: member.text().to_owned(),
            set,
            range: TextRange::new(start, end),
        };
        let end = self
            .expect(TokenKind::RightParen, "`)` after reduction")?
            .range()
            .end();
        let depth = self.parent_depth(child_depth)?;
        Some((
            Expr {
                resolved_enum: None,
                resolved_nominal: None,
                kind: ExprKind::Reduction {
                    operation,
                    binder,
                    value: Box::new(value),
                },
                range: TextRange::new(path.range().start(), end),
            },
            depth,
        ))
    }

    fn parse_named(&mut self) -> Option<(Expr, usize)> {
        let result = {
            let token = self.bump();
            let name = token.text().to_owned();
            let path = if self.at(TokenKind::Dot) {
                self.parse_name_path_from_first(token, "qualified name segment")?
            } else {
                NamePath::single(name, token.range())
            };
            if self.at(TokenKind::LeftParen) && path.as_str() == "evaluate" {
                return self.parse_evaluate(path);
            }
            if self.at(TokenKind::LeftParen) && path.as_str() == "partial" {
                return self.parse_partial(path);
            }
            if self.at(TokenKind::LeftParen)
                && matches!(path.as_str(), "sum" | "product" | "min" | "max")
            {
                return self.parse_reduction(path);
            }
            if self.at(TokenKind::LeftParen) {
                self.bump();
                let (arguments, child_depth) = self.parse_call_arguments(&path)?;
                let depth = self.parent_depth(child_depth)?;
                let end = self
                    .expect(TokenKind::RightParen, "`)` after operator arguments")?
                    .range()
                    .end();
                (
                    Expr {
                        resolved_enum: None,
                        resolved_nominal: None,
                        kind: ExprKind::Call {
                            callee: path.clone(),
                            arguments,
                        },
                        range: TextRange::new(path.range().start(), end),
                    },
                    depth,
                )
            } else if self.at_boundary_selection() {
                let selector = self.parse_boundary_port_selector()?;
                let range = TextRange::new(path.range().start(), selector.range().end());
                (
                    Expr {
                        resolved_enum: None,
                        resolved_nominal: None,
                        kind: ExprKind::BoundaryPortSelection {
                            port: Box::new(path),
                            selector: Box::new(selector),
                        },
                        range,
                    },
                    1,
                )
            } else {
                let range = path.range();
                let kind = if path.is_qualified() {
                    ExprKind::Path(path)
                } else {
                    ExprKind::Name(path.as_str().to_owned())
                };
                (
                    Expr {
                        resolved_enum: None,
                        resolved_nominal: None,
                        kind,
                        range,
                    },
                    1,
                )
            }
        };
        Some(result)
    }

    fn parse_group(&mut self) -> Option<(Expr, usize)> {
        let start = self.bump().range().start();
        let mut values = Vec::new();
        let mut depth = 0;
        if !self.at(TokenKind::RightParen) {
            let (value, child_depth) = self.parse_expression_with_depth(0)?;
            values.push(value);
            depth = child_depth;
        }
        let tuple = values.is_empty() || self.at(TokenKind::Comma);
        while self.at(TokenKind::Comma) {
            self.bump();
            if self.at(TokenKind::RightParen) {
                break;
            }
            let (value, child_depth) = self.parse_expression_with_depth(0)?;
            values.push(value);
            depth = depth.max(child_depth);
        }
        let end = self
            .expect(TokenKind::RightParen, "`)` after expression")?
            .range()
            .end();
        let range = TextRange::new(start, end);
        if tuple {
            let depth = self.parent_depth(depth)?;
            Some((
                Expr {
                    resolved_enum: None,
                    resolved_nominal: None,
                    kind: ExprKind::Tuple(values),
                    range,
                },
                depth,
            ))
        } else {
            let mut value = values.pop()?;
            value.range = range;
            Some((value, depth))
        }
    }

    fn at_boundary_selection(&mut self) -> bool {
        if !self.at(TokenKind::LeftBracket) {
            return false;
        }
        let mut tokens = self.tokens[self.cursor..]
            .iter()
            .filter(|token| !token.kind().is_trivia());
        tokens.next();
        matches!(tokens.next().map(Token::kind), Some(TokenKind::Identifier))
            && matches!(tokens.next().map(Token::kind), Some(TokenKind::Equal))
    }
}
