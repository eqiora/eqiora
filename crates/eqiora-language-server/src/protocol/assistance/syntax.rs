//! Cursor recovery uses the language lexer, including unfinished calls.
use eqiora::language::{Token, TokenKind as K, lex};

pub(super) fn tokens(source: &str) -> Vec<Token> {
    lex("editor", source)
        .0
        .into_iter()
        .filter(|t| !t.kind().is_trivia() && t.kind() != K::Eof)
        .collect()
}

pub(super) fn in_comment(source: &str, offset: u32) -> bool {
    lex("editor", source).0.iter().any(|t| {
        matches!(t.kind(), K::LineComment | K::DocComment | K::Notation)
            && t.range().start() <= offset
            && offset <= t.range().end()
    })
}

// The returned start includes the full qualification, so completion replaces
// `math.sq` once instead of inserting a second `math.` prefix.
pub(super) fn name_at(source: &str, offset: u32, prefix: bool) -> Option<(String, u32, u32)> {
    if in_comment(source, offset) {
        return None;
    }
    let tokens = tokens(source);
    let index = tokens.iter().position(|t| {
        matches!(t.kind(), K::Identifier | K::Dot)
            && t.range().start() <= offset
            && (offset < t.range().end() || (prefix && offset == t.range().end()))
    });
    let Some(mut first) = index else {
        return prefix.then(|| (String::new(), offset, offset));
    };
    let mut last = first;
    while first > 0
        && matches!(
            (tokens[first - 1].kind(), tokens[first].kind()),
            (K::Identifier, K::Dot) | (K::Dot, K::Identifier)
        )
    {
        first -= 1;
    }
    while last + 1 < tokens.len()
        && matches!(
            (tokens[last].kind(), tokens[last + 1].kind()),
            (K::Identifier, K::Dot) | (K::Dot, K::Identifier)
        )
    {
        last += 1;
    }
    let start = tokens[first].range().start();
    let end = tokens[last].range().end();
    let name = source.get(start as usize..if prefix { offset } else { end } as usize)?;
    Some((
        name.chars().filter(|c| !c.is_whitespace()).collect(),
        start,
        end,
    ))
}

pub(super) struct Call {
    pub name: String,
    pub start: u32,
    pub argument: usize,
    pub named: Option<String>,
}

pub(super) fn call_at(source: &str, offset: u32) -> Option<Call> {
    if in_comment(source, offset) {
        return None;
    }
    let tokens = tokens(source);
    let before: Vec<_> = tokens
        .iter()
        .filter(|t| t.range().start() < offset)
        .collect();
    let mut stack: Vec<(K, usize, usize, usize)> = Vec::new();
    for (i, token) in before.iter().enumerate() {
        match token.kind() {
            K::LeftParen | K::LeftBracket | K::LeftBrace => stack.push((token.kind(), i, 0, i + 1)),
            K::RightParen | K::RightBracket | K::RightBrace => {
                stack.pop();
            }
            K::Comma => {
                if let Some((_, _, argument, begin)) = stack.last_mut() {
                    *argument += 1;
                    *begin = i + 1;
                }
            }
            _ => {}
        }
    }
    for (kind, open, argument, begin) in stack.into_iter().rev() {
        if kind != K::LeftParen || open == 0 || before[open - 1].kind() != K::Identifier {
            continue;
        }
        let end = before[open - 1].range().end();
        let (name, start, _) = name_at(source, end - 1, false)?;
        // Declaration heads are not call sites.
        if before
            .iter()
            .position(|t| t.range().start() == start)
            .and_then(|i| i.checked_sub(1))
            .is_some_and(|i| {
                matches!(
                    before[i].text(),
                    "component" | "model" | "operator" | "connector"
                )
            })
        {
            return None;
        }
        let named = before
            .get(begin)
            .filter(|t| t.kind() == K::Identifier)
            .filter(|_| before.get(begin + 1).is_some_and(|t| t.kind() == K::Equal))
            .map(|t| t.text().to_owned());
        return Some(Call {
            name,
            start,
            argument,
            named,
        });
    }
    None
}

pub(super) fn clean(source: &str) -> String {
    lex("editor", source)
        .0
        .into_iter()
        .filter(|t| !matches!(t.kind(), K::LineComment | K::DocComment | K::Eof))
        .map(|t| t.text().to_owned())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub(super) fn head_end(source: &str, callable: bool) -> usize {
    let mut depth = 0usize;
    for token in tokens(source) {
        match token.kind() {
            K::LeftParen | K::LeftBracket => depth += 1,
            K::RightParen | K::RightBracket => depth = depth.saturating_sub(1),
            K::LeftBrace | K::Semicolon if depth == 0 => return token.range().start() as usize,
            K::Equal if depth == 0 && callable => return token.range().start() as usize,
            _ => {}
        }
    }
    source.len()
}
