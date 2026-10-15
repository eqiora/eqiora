use crate::allocation::{copy_slice, copy_string};
use std::collections::BTreeMap;

use eqiora_core::{Diagnostic, diagnostic::codes};
use quick_xml::{
    Reader, XmlVersion,
    events::{BytesDecl, BytesStart, Event},
};

#[derive(Clone, Copy)]
pub(crate) struct Limits {
    pub(crate) max_source_bytes: usize,
    pub(crate) max_depth: usize,
    pub(crate) max_elements: usize,
    pub(crate) max_attributes_per_element: usize,
    pub(crate) max_text_bytes: usize,
    pub(crate) attribute_references: bool,
}

pub(crate) fn invalid_import(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_EXTERNAL_DATA_IMPORT, message)
}

#[derive(Debug)]
pub(crate) struct Node {
    pub(crate) name: String,
    pub(crate) path: Vec<u32>,
    pub(crate) attributes: BTreeMap<String, String>,
    pub(crate) text: String,
    pub(crate) children: Vec<Node>,
}

#[derive(Debug)]
struct NodeBuilder {
    node: Node,
    next_child: u32,
}

#[derive(Debug)]
pub(crate) struct WorkBudget {
    used: usize,
    maximum: usize,
    format: &'static str,
}

impl WorkBudget {
    pub(crate) fn new(maximum: usize, format: &'static str) -> Self {
        Self {
            used: 0,
            maximum,
            format,
        }
    }

    pub(crate) fn charge(&mut self, amount: usize) -> Result<(), Diagnostic> {
        self.used = self.used.checked_add(amount).ok_or_else(|| {
            invalid_import(format!("{} parser work overflows usize", self.format))
        })?;
        if self.used > self.maximum {
            return Err(invalid_import(format!(
                "{} parser work exceeds the configured limit",
                self.format
            )));
        }
        Ok(())
    }
}

pub(crate) fn parse_tree(
    source: &[u8],
    limits: Limits,
    work: &mut WorkBudget,
    mut admit_element: impl FnMut(&str) -> Result<(), Diagnostic>,
) -> Result<Node, Diagnostic> {
    if source.len() > limits.max_source_bytes {
        return Err(invalid_import(
            "XML source exceeds the configured byte limit",
        ));
    }
    work.charge(source.len())?;
    let source_text = std::str::from_utf8(source)
        .map_err(|_| invalid_import("XML source must be valid UTF-8"))?;
    if !source_text.chars().all(is_xml_1_0_character) {
        return Err(invalid_import(
            "XML source contains a character outside XML 1.0",
        ));
    }
    let mut reader = Reader::from_reader(source);
    reader.config_mut().trim_text(false);
    reader.config_mut().enable_all_checks(true);
    let mut stack = Vec::<NodeBuilder>::new();
    let mut root = None;
    let mut elements = 0_usize;
    let mut text_bytes = 0_usize;
    let mut saw_declaration = false;
    let mut saw_event_before_declaration = false;
    loop {
        work.charge(1)?;
        match reader
            .read_event()
            .map_err(|_| invalid_import("XML source is not well-formed XML"))?
        {
            Event::Start(start) => {
                saw_event_before_declaration = true;
                if root.is_some() && stack.is_empty() {
                    return Err(invalid_import("XML source contains multiple root elements"));
                }
                elements = checked_add(elements, 1, "XML element count")?;
                if elements > limits.max_elements {
                    return Err(invalid_import(
                        "XML element count exceeds the configured limit",
                    ));
                }
                if stack.len() >= limits.max_depth {
                    return Err(invalid_import(
                        "XML nesting depth exceeds the configured limit",
                    ));
                }
                let path = next_path(&mut stack)?;
                let (name, attributes, decoded) = parse_start(&start, limits, work)?;
                admit_element(&name)?;
                text_bytes = checked_add(text_bytes, decoded, "XML decoded text bytes")?;
                if text_bytes > limits.max_text_bytes {
                    return Err(invalid_import(
                        "XML decoded text exceeds the configured limit",
                    ));
                }
                stack
                    .try_reserve(1)
                    .map_err(|_| invalid_import("XML stack allocation failed"))?;
                stack.push(NodeBuilder {
                    node: Node {
                        name,
                        path,
                        attributes,
                        text: String::new(),
                        children: Vec::new(),
                    },
                    next_child: 0,
                });
            }
            Event::End(end) => {
                saw_event_before_declaration = true;
                let builder = stack
                    .pop()
                    .ok_or_else(|| invalid_import("XML end tag has no start tag"))?;
                if end.name().as_ref() != builder.node.name {
                    return Err(invalid_import("XML end tag differs from its start tag"));
                }
                if let Some(parent) = stack.last_mut() {
                    parent
                        .node
                        .children
                        .try_reserve(1)
                        .map_err(|_| invalid_import("XML child allocation failed"))?;
                    parent.node.children.push(builder.node);
                } else if root.replace(builder.node).is_some() {
                    return Err(invalid_import("XML source contains multiple root elements"));
                }
            }
            Event::Text(text) => {
                saw_event_before_declaration = true;
                if text.contains('&') || text.contains("]]>") {
                    return Err(invalid_import(
                        "XML text references and forbidden XML sequences are unsupported",
                    ));
                }
                let decoded = text.as_ref();
                text_bytes = checked_add(text_bytes, decoded.len(), "XML decoded text bytes")?;
                if text_bytes > limits.max_text_bytes {
                    return Err(invalid_import(
                        "XML decoded text exceeds the configured limit",
                    ));
                }
                work.charge(decoded.len())?;
                if let Some(current) = stack.last_mut() {
                    current
                        .node
                        .text
                        .try_reserve(decoded.len())
                        .map_err(|_| invalid_import("XML text allocation failed"))?;
                    current.node.text.push_str(decoded);
                } else if !contains_only_xml_space(decoded) {
                    return Err(invalid_import("XML source has text outside its root"));
                }
            }
            Event::Comment(comment) => {
                saw_event_before_declaration = true;
                let decoded = comment.as_ref();
                text_bytes = checked_add(text_bytes, decoded.len(), "XML decoded text bytes")?;
                if text_bytes > limits.max_text_bytes {
                    return Err(invalid_import(
                        "XML decoded text exceeds the configured limit",
                    ));
                }
                work.charge(decoded.len())?;
            }
            Event::Decl(declaration) => {
                if saw_declaration
                    || saw_event_before_declaration
                    || root.is_some()
                    || !stack.is_empty()
                {
                    return Err(invalid_import(
                        "XML declaration must be the first event and occur once",
                    ));
                }
                validate_declaration(&declaration, work)?;
                saw_declaration = true;
            }
            Event::Empty(_) => {
                return Err(invalid_import(
                    "empty XML elements are outside the admitted XML subset",
                ));
            }
            Event::CData(_) | Event::PI(_) | Event::DocType(_) | Event::GeneralRef(_) => {
                return Err(invalid_import(
                    "DTD, entities, CDATA, and processing instructions are outside the admitted XML subset",
                ));
            }
            Event::Eof => break,
        }
    }
    if !stack.is_empty() {
        return Err(invalid_import("XML source ended inside an element"));
    }
    root.ok_or_else(|| invalid_import("XML source has no root element"))
}

fn next_path(stack: &mut [NodeBuilder]) -> Result<Vec<u32>, Diagnostic> {
    let Some(parent) = stack.last_mut() else {
        return Ok(Vec::new());
    };
    let child = parent.next_child;
    parent.next_child = parent
        .next_child
        .checked_add(1)
        .ok_or_else(|| invalid_import("XML structural selector overflows u32"))?;
    let mut path = copy_slice(&parent.node.path, "XML selector")?;
    path.try_reserve_exact(1)
        .map_err(|_| invalid_import("XML selector allocation failed"))?;
    path.push(child);
    Ok(path)
}

fn parse_start(
    start: &BytesStart<'_>,
    limits: Limits,
    work: &mut WorkBudget,
) -> Result<(String, BTreeMap<String, String>, usize), Diagnostic> {
    let name = copy_string(start.name().as_ref(), "XML element name")?;
    if name.contains(':') {
        return Err(invalid_import(
            "XML namespaces are outside the admitted XML subset",
        ));
    }
    let mut attributes = BTreeMap::new();
    let mut decoded = name.len();
    for (index, attribute) in start.attributes().with_checks(true).enumerate() {
        work.charge(1)?;
        if index >= limits.max_attributes_per_element {
            return Err(invalid_import(
                "XML element attribute count exceeds the configured limit",
            ));
        }
        let attribute =
            attribute.map_err(|_| invalid_import("XML element has malformed attributes"))?;
        if attribute.value.contains('<')
            || (!limits.attribute_references && attribute.value.contains('&'))
        {
            return Err(invalid_import(
                "XML attributes may not contain references or literal less-than signs",
            ));
        }
        let key = copy_string(attribute.key.as_ref(), "XML attribute name")?;
        if key.contains(':') {
            return Err(invalid_import(
                "XML namespaces are outside the admitted XML subset",
            ));
        }
        let normalized = attribute
            .normalized_value(XmlVersion::Explicit1_0)
            .map_err(|_| invalid_import("XML attribute contains an invalid reference"))?;
        if !normalized.chars().all(is_xml_1_0_character) {
            return Err(invalid_import(
                "XML attribute contains a character outside XML 1.0",
            ));
        }
        let value = copy_string(normalized.as_ref(), "XML attribute value")?;
        decoded = checked_add(decoded, key.len(), "XML decoded attribute bytes")?;
        decoded = checked_add(decoded, value.len(), "XML decoded attribute bytes")?;
        if attributes.insert(key, value).is_some() {
            return Err(invalid_import("XML element repeats an attribute"));
        }
    }
    work.charge(decoded)?;
    Ok((name, attributes, decoded))
}

fn validate_declaration(
    declaration: &BytesDecl<'_>,
    work: &mut WorkBudget,
) -> Result<(), Diagnostic> {
    let raw = declaration.as_ref();
    let start = BytesStart::from_content(raw, 3);
    let mut count = 0_usize;
    let mut second_was_encoding = false;
    for attribute in start.attributes().with_checks(true) {
        work.charge(1)?;
        if count >= 3 {
            return Err(invalid_import(
                "XML declaration requires version and at most encoding and standalone",
            ));
        }
        let attribute =
            attribute.map_err(|_| invalid_import("XML declaration has malformed attributes"))?;
        let key = attribute.key.as_ref();
        let value = attribute.value.as_ref();
        match (count, key) {
            (0, "version") if value == "1.0" => {}
            (1, "encoding") if value.eq_ignore_ascii_case("UTF-8") => {
                second_was_encoding = true;
            }
            (1, "standalone") if matches!(value, "yes" | "no") => {}
            (2, "standalone") if second_was_encoding && matches!(value, "yes" | "no") => {}
            _ => {
                return Err(invalid_import(
                    "XML declaration attributes are unknown, repeated, or out of order",
                ));
            }
        }
        count += 1;
    }
    if count == 0 {
        return Err(invalid_import(
            "XML declaration requires a version attribute",
        ));
    }
    Ok(())
}

fn checked_add(left: usize, right: usize, label: &str) -> Result<usize, Diagnostic> {
    left.checked_add(right)
        .ok_or_else(|| invalid_import(format!("{label} overflows usize")))
}

fn is_xml_1_0_character(character: char) -> bool {
    matches!(character, '\u{0009}' | '\u{000A}' | '\u{000D}')
        || ('\u{0020}'..='\u{D7FF}').contains(&character)
        || ('\u{E000}'..='\u{FFFD}').contains(&character)
        || ('\u{10000}'..='\u{10FFFF}').contains(&character)
}

pub(crate) fn contains_only_xml_space(value: &str) -> bool {
    value.chars().all(is_xml_space)
}

pub(crate) const fn is_xml_space(character: char) -> bool {
    matches!(character, '\u{0009}' | '\u{000A}' | '\u{000D}' | '\u{0020}')
}
#[cfg(test)]
mod tests;
