//! Print source-owned trivia over the canonical AST output's nested owner spans.

use std::fmt;
use std::ops::{Deref, DerefMut, Range};

use crate::ast::comments::{CommentPosition, SourceComments, SyntaxAnchor};
use crate::lexer::{Token, TokenKind, lex};

#[derive(Default)]
pub(super) struct Output {
    text: String,
    owners: Vec<Owner>,
    stack: Vec<usize>,
}

type PrintedAnchor = (usize, usize, SyntaxAnchor);

struct Owner {
    span: Range<usize>,
    parent: Option<usize>,
    comments: SourceComments,
}

impl Deref for Output {
    type Target = String;
    fn deref(&self) -> &String {
        &self.text
    }
}

impl DerefMut for Output {
    fn deref_mut(&mut self) -> &mut String {
        &mut self.text
    }
}

impl fmt::Write for Output {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        self.text.push_str(text);
        Ok(())
    }
}

impl Output {
    pub(super) fn begin(&mut self, comments: &SourceComments) {
        let mut comments = comments.clone();
        // Parsed docs own exact trivia; directly constructed docs use this
        // same formatter's canonical leading-comment projection.
        if let Some(doc) = &comments.doc
            && !comments
                .comments
                .iter()
                .any(|comment| comment.text.starts_with("///") && !comment.detached_doc)
        {
            comments.comments.extend(doc.text().split('\n').map(|line| {
                crate::ast::comments::CommentTrivia {
                    text: if line.is_empty() {
                        "///".to_owned()
                    } else {
                        format!("/// {line}")
                    },
                    range: doc.range(),
                    position: CommentPosition::Leading,
                    detached_doc: false,
                }
            }));
        }
        let index = self.owners.len();
        self.owners.push(Owner {
            span: self.text.len()..self.text.len(),
            parent: self.stack.last().copied(),
            comments,
        });
        self.stack.push(index);
    }

    pub(super) fn end(&mut self) {
        let index = self.stack.pop().expect("balanced formatter owner");
        self.owners[index].span.end = self.text.len();
    }

    fn canonical_anchors(&self, tokens: &[&Token]) -> Vec<Vec<PrintedAnchor>> {
        let mut anchors = vec![Vec::new(); self.owners.len()];
        for (index, token) in tokens.iter().enumerate() {
            let mut owner = self
                .owners
                .partition_point(|owner| owner.span.start <= token.range().start() as usize)
                - 1;
            while token.range().end() as usize > self.owners[owner].span.end {
                owner = self.owners[owner]
                    .parent
                    .expect("token remains inside its parent owner");
            }
            anchors[owner].push((index, index + 1, SyntaxAnchor::Token(token.kind())));
        }
        for child in &self.owners {
            if let Some(parent) = child.parent {
                let start = tokens
                    .partition_point(|token| (token.range().start() as usize) < child.span.start);
                let end =
                    tokens.partition_point(|token| token.range().end() as usize <= child.span.end);
                anchors[parent].push((start, end, SyntaxAnchor::Child(child.comments.range)));
            }
        }
        for local in &mut anchors {
            local.sort_by_key(|(index, _, _)| *index);
        }
        anchors
    }

    pub(super) fn finish(self) -> String {
        assert!(self.stack.is_empty(), "balanced formatter owners");
        if self
            .owners
            .iter()
            .all(|owner| owner.comments.comments.is_empty())
        {
            return self.text;
        }
        let (lexed, _) = lex("<formatted>", &self.text);
        let tokens: Vec<_> = lexed
            .iter()
            .filter(|token| !token.kind().is_trivia() && token.kind() != TokenKind::Eof)
            .collect();
        let anchors = self.canonical_anchors(&tokens);
        let mut gaps: Vec<Vec<(u8, &str, bool, bool)>> = vec![Vec::new(); tokens.len() + 1];
        for (index, owner) in self.owners.iter().enumerate() {
            if owner.comments.comments.is_empty() {
                continue;
            }
            let local = &anchors[index];
            let old: Vec<_> = owner
                .comments
                .tokens
                .iter()
                .map(|(kind, _)| *kind)
                .collect();
            let new: Vec<_> = local.iter().map(|(_, _, anchor)| *anchor).collect();
            let mapping = align(&old, &new);
            for comment in &owner.comments.comments {
                let (gap, inline) = match comment.position {
                    CommentPosition::Leading => (
                        tokens.partition_point(|token| {
                            (token.range().start() as usize) < owner.span.start
                        }),
                        false,
                    ),
                    CommentPosition::Trailing => (
                        tokens.partition_point(|token| {
                            token.range().end() as usize <= owner.span.end
                        }),
                        true,
                    ),
                    CommentPosition::Gap { token, inline } => {
                        let following = mapping
                            .get(token..)
                            .unwrap_or_default()
                            .iter()
                            .flatten()
                            .next()
                            .copied();
                        let preceding = mapping[..token.min(mapping.len())]
                            .iter()
                            .rev()
                            .flatten()
                            .next()
                            .copied();
                        let gap = following
                            .map(|position| local[position].0)
                            .or_else(|| preceding.map(|position| local[position].1))
                            .unwrap_or_else(|| {
                                tokens.partition_point(|token| {
                                    token.range().end() as usize <= owner.span.end
                                })
                            });
                        (gap, inline)
                    }
                };
                let priority = match comment.position {
                    CommentPosition::Trailing => 0,
                    CommentPosition::Gap { .. } => 1,
                    CommentPosition::Leading => 2,
                };
                gaps[gap].push((priority, &comment.text, inline, comment.detached_doc));
            }
        }
        let mut result = String::new();
        for gap in &mut gaps {
            gap.sort_by_key(|(priority, _, _, _)| *priority);
        }
        let mut previous = 0;
        for (index, gap) in gaps.iter().enumerate() {
            let next = tokens
                .get(index)
                .map_or(self.text.len(), |token| token.range().start() as usize);
            let whitespace = &self.text[previous..next];
            if gap.is_empty() {
                result.push_str(whitespace);
            } else {
                let line_start = self.text[..next]
                    .rfind('\n')
                    .map_or(0, |position| position + 1);
                let indent: String = self.text[line_start..]
                    .chars()
                    .take_while(|character| *character == ' ')
                    .collect();
                let inline = gap[0].2 && previous != 0;
                if inline {
                    result.push(' ');
                } else if previous == 0 {
                    result.push_str(&indent);
                } else if whitespace.contains('\n') {
                    result.push_str(whitespace);
                } else {
                    result.push('\n');
                    result.push_str(&indent);
                }
                for (offset, (_, comment, _, detached)) in gap.iter().enumerate() {
                    if offset != 0 {
                        result.push_str(&indent);
                    }
                    result.push_str(comment);
                    result.push('\n');
                    if *detached {
                        result.push('\n');
                    }
                }
                if index < tokens.len() {
                    result.push_str(&indent);
                }
            }
            if let Some(token) = tokens.get(index) {
                let end = token.range().end() as usize;
                result.push_str(&self.text[next..end]);
                previous = end;
            }
        }
        result
    }
}

/// Align only syntax leaves of the same owner. Child declarations are excluded.
/// Equal prefixes/suffixes handle normal formatting and local edits in linear
/// time. A bounded middle avoids quadratic work on hostile or extensively edited
/// input; unmatched deleted syntax keeps its trivia at the surviving owner edge.
fn align(old: &[SyntaxAnchor], new: &[SyntaxAnchor]) -> Vec<Option<usize>> {
    let mut result = vec![None; old.len()];
    let prefix = old
        .iter()
        .zip(new)
        .take_while(|(left, right)| left == right)
        .count();
    for (index, slot) in result[..prefix].iter_mut().enumerate() {
        *slot = Some(index);
    }
    let suffix = old[prefix..]
        .iter()
        .rev()
        .zip(new[prefix..].iter().rev())
        .take_while(|(left, right)| left == right)
        .count();
    for index in 0..suffix {
        result[old.len() - index - 1] = Some(new.len() - index - 1);
    }
    let left = &old[prefix..old.len() - suffix];
    let right = &new[prefix..new.len() - suffix];
    if left.len().saturating_mul(right.len()) > 65_536 {
        return result;
    }
    let width = right.len() + 1;
    let mut lengths = vec![0_u32; (left.len() + 1) * width];
    for row in (0..left.len()).rev() {
        for column in (0..right.len()).rev() {
            lengths[row * width + column] = if left[row] == right[column] {
                1 + lengths[(row + 1) * width + column + 1]
            } else {
                lengths[(row + 1) * width + column].max(lengths[row * width + column + 1])
            };
        }
    }
    let (mut row, mut column) = (0, 0);
    while row < left.len() && column < right.len() {
        if left[row] == right[column] {
            result[prefix + row] = Some(prefix + column);
            row += 1;
            column += 1;
        } else if lengths[(row + 1) * width + column] >= lengths[row * width + column + 1] {
            row += 1;
        } else {
            column += 1;
        }
    }
    result
}
