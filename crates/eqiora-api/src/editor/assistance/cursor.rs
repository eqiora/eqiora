//! Cursor recovery uses the language lexer, including unfinished calls.
use eqiora_lang::{Token, TokenKind as K, lex};

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

/// Recovered innermost call, including unfinished argument lists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditorCall {
    /// Qualified callable spelling.
    pub name: String,
    /// UTF-8 start of the callable name.
    pub start: u32,
    /// Zero-based argument at the cursor.
    pub argument: usize,
    /// Named binding whose value contains the cursor.
    pub named: Option<String>,
    pub(super) open: usize,
    pub(super) begin: usize,
}

pub(super) fn call_at(source: &str, offset: u32) -> Option<EditorCall> {
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
        return Some(EditorCall {
            name,
            start,
            open,
            begin,
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

pub(super) fn context(source: &str, start: u32) -> super::EditorCompletionContext {
    use super::EditorCompletionContext as C;
    let tokens = tokens(source);
    let before: Vec<_> = tokens
        .iter()
        .take_while(|t| t.range().end() <= start)
        .collect();
    let mut owners = Vec::new();
    let mut begin = 0;
    for (i, t) in before.iter().enumerate() {
        match t.kind() {
            K::LeftBrace => {
                owners.push(
                    before[begin..i]
                        .iter()
                        .any(|t| matches!(t.text(), "relation" | "initial" | "form")),
                );
                begin = i + 1;
            }
            K::RightBrace => {
                owners.pop();
                begin = i + 1;
            }
            K::Semicolon => begin = i + 1,
            _ => {}
        }
    }
    let statement = &before[begin..];
    if statement.first().is_some_and(|t| t.text() == "import")
        && !statement.iter().any(|t| t.text() == "as")
    {
        return C::Import;
    }
    if statement.iter().any(|t| t.kind() == K::Equal) {
        return C::Expression;
    }
    if let Some(colon) = statement.iter().rposition(|t| t.kind() == K::Colon)
        && !statement[colon + 1..]
            .iter()
            .any(|t| t.kind() == K::LeftParen)
    {
        return C::Type;
    }
    if owners.last() == Some(&true) || statement.iter().any(|t| t.kind() == K::LeftParen) {
        C::Expression
    } else {
        C::Declaration
    }
}

pub(super) fn binding_position(source: &str, offset: u32, call: &EditorCall) -> bool {
    let tokens = tokens(source);
    let current: Vec<_> = tokens[call.begin..]
        .iter()
        .take_while(|t| t.range().start() < offset)
        .collect();
    current.is_empty() || (current.len() == 1 && current[0].kind() == K::Identifier)
}

pub(super) fn positional_before(source: &str, call: &EditorCall) -> bool {
    let tokens = tokens(source);
    let mut depth = 0;
    let mut begin = call.open + 1;
    for i in call.open + 1..call.begin {
        match tokens[i].kind() {
            K::Comma if depth == 0 => {
                if i > begin && !tokens.get(begin + 1).is_some_and(|t| t.kind() == K::Equal) {
                    return true;
                }
                begin = i + 1;
            }
            K::LeftParen | K::LeftBracket | K::LeftBrace => depth += 1,
            K::RightParen | K::RightBracket | K::RightBrace => depth -= 1,
            _ => {}
        }
    }
    false
}

pub(super) fn supplied(
    source: &str,
    offset: u32,
    call: &EditorCall,
) -> std::collections::BTreeSet<String> {
    let tokens = tokens(source);
    let mut result = std::collections::BTreeSet::new();
    let mut begin = call.open + 1;
    let mut depth = 0;
    let record = |result: &mut std::collections::BTreeSet<String>, begin: usize, end: usize| {
        let argument = &tokens[begin..end];
        if let [name, equals, ..] = argument
            && name.kind() == K::Identifier
            && equals.kind() == K::Equal
            && !(name.range().start() <= offset && offset <= name.range().end())
        {
            result.insert(name.text().to_owned());
        }
    };
    // Record disjoint argument slices; nested commas never split bindings.
    for (i, token) in tokens.iter().enumerate().skip(call.open + 1) {
        match token.kind() {
            K::RightParen if depth == 0 => {
                record(&mut result, begin, i);
                return result;
            }
            K::Comma if depth == 0 => {
                record(&mut result, begin, i);
                begin = i + 1;
            }
            K::LeftParen | K::LeftBracket | K::LeftBrace => depth += 1,
            K::RightParen | K::RightBracket | K::RightBrace => depth -= 1,
            _ => {}
        }
    }
    record(&mut result, begin, tokens.len());
    result
}
