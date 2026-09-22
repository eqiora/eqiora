use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;

use eqiora_core::Diagnostic;
use eqiora_meshing::{DiscreteFieldAssociation, DiscreteFieldShape};
use quick_xml::Reader;
use quick_xml::XmlVersion;
use quick_xml::events::{BytesDecl, BytesStart, Event};

use crate::plan::{
    FieldPlan, GeometryKind, XdmfArrayRequest, XdmfArrayRole, XdmfImportLimits, XdmfImportPlan,
    XdmfScalarType, XdmfSelection, invalid_import,
};

#[derive(Debug)]
struct Node {
    name: String,
    path: Vec<u32>,
    attributes: BTreeMap<String, String>,
    text: String,
    children: Vec<Node>,
}

#[derive(Debug)]
struct NodeBuilder {
    node: Node,
    next_child: u32,
}

#[derive(Debug, Clone)]
struct DataItem {
    path: Vec<u32>,
    locator: String,
    dataset: String,
    scalar: XdmfScalarType,
    shape: Vec<u64>,
}

#[derive(Debug)]
struct AttributePlan {
    field: FieldPlan,
    item: DataItem,
}

impl XdmfImportPlan {
    /// Parse bounded XDMF 3 metadata into canonical typed array requests.
    ///
    /// No path or URL is opened. Requests are normalized to Geometry,
    /// Topology, then caller-selected Attributes regardless of XML order.
    ///
    /// # Errors
    /// Returns `EQ0810` for malformed XML, an unsupported construct, an
    /// invalid structural selection, or a resource-budget violation.
    pub fn parse(
        metadata: &[u8],
        selection: XdmfSelection,
        limits: XdmfImportLimits,
    ) -> Result<Self, Diagnostic> {
        let limits = limits.validate()?;
        parse_document(metadata, selection, limits)
    }
}

pub(crate) fn parse_document(
    metadata: &[u8],
    selection: XdmfSelection,
    limits: XdmfImportLimits,
) -> Result<XdmfImportPlan, Diagnostic> {
    if metadata.len() > limits.max_metadata_bytes {
        return Err(invalid_import(
            "XDMF metadata exceeds the configured byte limit",
        ));
    }
    let metadata_text = std::str::from_utf8(metadata)
        .map_err(|_| invalid_import("XDMF metadata must be valid UTF-8"))?;
    if !metadata_text.chars().all(is_xml_1_0_character) {
        return Err(invalid_import(
            "XDMF metadata contains a character outside XML 1.0",
        ));
    }
    let root = parse_xml_tree(metadata, limits)?;
    validate_element(&root, "Xdmf", &["Version"])?;
    require_whitespace(&root)?;
    require_attr(&root, "Version", "XDMF root")
        .and_then(|value| require_exact(value, "3.0", "XDMF Version"))?;
    let domain = only_child(&root, "Domain")?;
    validate_element(domain, "Domain", &[])?;
    require_whitespace(domain)?;
    let grid = only_child(domain, "Grid")?;
    validate_element(grid, "Grid", &["Name", "GridType"])?;
    require_whitespace(grid)?;
    require_attr(grid, "GridType", "XDMF Grid")
        .and_then(|value| require_exact(value, "Uniform", "XDMF GridType"))?;
    if grid.path != selection.grid() {
        return Err(invalid_import(
            "XDMF selection does not identify the admitted Uniform Grid",
        ));
    }

    let mut topology = None;
    let mut geometry = None;
    let mut attributes = Vec::new();
    for child in &grid.children {
        match child.name.as_str() {
            "Topology" if topology.is_none() => topology = Some(parse_topology(child, limits)?),
            "Geometry" if geometry.is_none() => geometry = Some(parse_geometry(child, limits)?),
            "Attribute" => attributes.push(parse_attribute(child, limits)?),
            "Topology" => return Err(invalid_import("XDMF Grid contains duplicate Topology")),
            "Geometry" => return Err(invalid_import("XDMF Grid contains duplicate Geometry")),
            _ => return Err(invalid_import("XDMF Grid contains an unsupported element")),
        }
    }
    let (dimension, cell_count, topology_item) =
        topology.ok_or_else(|| invalid_import("XDMF Uniform Grid requires one Topology"))?;
    let (geometry_kind, vertex_count, geometry_item) =
        geometry.ok_or_else(|| invalid_import("XDMF Uniform Grid requires one Geometry"))?;
    match (dimension, geometry_kind) {
        (2, GeometryKind::Xy | GeometryKind::Xyz) | (3, GeometryKind::Xyz) => {}
        _ => {
            return Err(invalid_import(
                "XDMF geometry kind is incompatible with topology",
            ));
        }
    }

    for attribute in &attributes {
        let expected = match attribute.field.association {
            DiscreteFieldAssociation::Vertex => vertex_count,
            DiscreteFieldAssociation::Cell => cell_count,
        };
        validate_field_shape(&attribute.item, attribute.field.shape, expected, dimension)?;
    }
    let available = attributes
        .into_iter()
        .map(|attribute| (attribute.field.origin_selector.clone(), attribute))
        .collect::<BTreeMap<_, _>>();
    let mut selected_fields = Vec::new();
    let mut selected_items = Vec::new();
    for path in selection.attributes() {
        let attribute = available.get(path).ok_or_else(|| {
            invalid_import("XDMF selection references a missing Attribute element")
        })?;
        selected_fields.push(attribute.field.clone());
        selected_items.push(attribute.item.clone());
    }

    let mut requests = Vec::new();
    requests
        .try_reserve_exact(2 + selected_items.len())
        .map_err(|_| invalid_import("XDMF request allocation failed"))?;
    push_request(&mut requests, XdmfArrayRole::Geometry, geometry_item);
    push_request(&mut requests, XdmfArrayRole::Topology, topology_item);
    for item in selected_items {
        push_request(&mut requests, XdmfArrayRole::Attribute, item);
    }
    Ok(XdmfImportPlan {
        metadata: metadata.to_vec(),
        selection,
        limits,
        grid_name: grid.attributes.get("Name").cloned(),
        dimension,
        geometry_kind,
        cell_count,
        fields: selected_fields,
        requests,
    })
}

fn parse_xml_tree(metadata: &[u8], limits: XdmfImportLimits) -> Result<Node, Diagnostic> {
    let mut reader = Reader::from_reader(metadata);
    reader.config_mut().trim_text(false);
    reader.config_mut().enable_all_checks(true);
    let mut stack = Vec::<NodeBuilder>::new();
    let mut root = None;
    let mut elements = 0_usize;
    let mut data_items = 0_usize;
    let mut text_bytes = 0_usize;
    let mut work = 0_usize;
    let mut saw_declaration = false;
    let mut saw_event_before_declaration = false;
    loop {
        work = checked_add(work, 1, "XDMF parser work")?;
        if work > limits.max_parser_work {
            return Err(invalid_import(
                "XDMF parser work exceeds the configured limit",
            ));
        }
        match reader
            .read_event()
            .map_err(|_| invalid_import("XDMF metadata is not well-formed XML"))?
        {
            Event::Start(start) => {
                saw_event_before_declaration = true;
                if root.is_some() && stack.is_empty() {
                    return Err(invalid_import(
                        "XDMF metadata contains multiple root elements",
                    ));
                }
                elements = checked_add(elements, 1, "XDMF element count")?;
                if elements > limits.max_elements {
                    return Err(invalid_import(
                        "XDMF element count exceeds the configured limit",
                    ));
                }
                if stack.len() >= limits.max_depth {
                    return Err(invalid_import(
                        "XDMF nesting depth exceeds the configured limit",
                    ));
                }
                let path = next_path(&mut stack)?;
                let (name, attributes, decoded) = parse_start(&start, limits)?;
                if name == "DataItem" {
                    data_items = checked_add(data_items, 1, "XDMF DataItem count")?;
                    if data_items > limits.max_data_items {
                        return Err(invalid_import(
                            "XDMF DataItem count exceeds the configured limit",
                        ));
                    }
                }
                text_bytes = checked_add(text_bytes, decoded, "XDMF decoded text bytes")?;
                if text_bytes > limits.max_text_bytes {
                    return Err(invalid_import(
                        "XDMF decoded text exceeds the configured limit",
                    ));
                }
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
                    .ok_or_else(|| invalid_import("XDMF XML end tag has no start tag"))?;
                if end.name().as_ref() != builder.node.name {
                    return Err(invalid_import(
                        "XDMF XML end tag differs from its start tag",
                    ));
                }
                if let Some(parent) = stack.last_mut() {
                    parent.node.children.push(builder.node);
                } else if root.replace(builder.node).is_some() {
                    return Err(invalid_import(
                        "XDMF metadata contains multiple root elements",
                    ));
                }
            }
            Event::Text(text) => {
                saw_event_before_declaration = true;
                if text.contains("]]>") {
                    return Err(invalid_import(
                        "XDMF character data contains the forbidden XML sequence ]]>",
                    ));
                }
                let decoded = text.as_ref();
                text_bytes = checked_add(text_bytes, decoded.len(), "XDMF decoded text bytes")?;
                if text_bytes > limits.max_text_bytes {
                    return Err(invalid_import(
                        "XDMF decoded text exceeds the configured limit",
                    ));
                }
                if let Some(current) = stack.last_mut() {
                    current
                        .node
                        .text
                        .try_reserve(decoded.len())
                        .map_err(|_| invalid_import("XDMF text allocation failed"))?;
                    current.node.text.push_str(decoded);
                } else if !contains_only_xml_space(decoded) {
                    return Err(invalid_import("XDMF metadata has text outside its root"));
                }
            }
            Event::Comment(comment) => {
                saw_event_before_declaration = true;
                let decoded = comment.as_ref();
                text_bytes = checked_add(text_bytes, decoded.len(), "XDMF decoded text bytes")?;
                if text_bytes > limits.max_text_bytes {
                    return Err(invalid_import(
                        "XDMF decoded text exceeds the configured limit",
                    ));
                }
            }
            Event::Decl(decl) => {
                if saw_declaration
                    || saw_event_before_declaration
                    || root.is_some()
                    || !stack.is_empty()
                {
                    return Err(invalid_import(
                        "XDMF XML declaration must be the first event and occur once",
                    ));
                }
                validate_declaration(&decl)?;
                saw_declaration = true;
            }
            Event::Empty(_) => {
                return Err(invalid_import(
                    "empty XML elements are outside the admitted XDMF subset",
                ));
            }
            Event::CData(_) | Event::PI(_) | Event::DocType(_) | Event::GeneralRef(_) => {
                return Err(invalid_import(
                    "DTD, references, CDATA, and processing instructions are outside the admitted XDMF subset",
                ));
            }
            Event::Eof => break,
        }
    }
    if !stack.is_empty() {
        return Err(invalid_import("XDMF metadata ended inside an element"));
    }
    root.ok_or_else(|| invalid_import("XDMF metadata has no root element"))
}

fn next_path(stack: &mut [NodeBuilder]) -> Result<Vec<u32>, Diagnostic> {
    let Some(parent) = stack.last_mut() else {
        return Ok(Vec::new());
    };
    let child = parent.next_child;
    parent.next_child = parent
        .next_child
        .checked_add(1)
        .ok_or_else(|| invalid_import("XDMF structural selector overflows u32"))?;
    let mut path = parent.node.path.clone();
    path.try_reserve(1)
        .map_err(|_| invalid_import("XDMF selector allocation failed"))?;
    path.push(child);
    Ok(path)
}

fn parse_start(
    start: &BytesStart<'_>,
    limits: XdmfImportLimits,
) -> Result<(String, BTreeMap<String, String>, usize), Diagnostic> {
    let name = start.name().as_ref().to_owned();
    let mut attributes = BTreeMap::new();
    let mut decoded = name.len();
    for (index, attribute) in start.attributes().with_checks(true).enumerate() {
        if index >= limits.max_attributes_per_element {
            return Err(invalid_import(
                "XDMF element attribute count exceeds the configured limit",
            ));
        }
        let attribute =
            attribute.map_err(|_| invalid_import("XDMF element has malformed attributes"))?;
        if attribute.value.contains('<') {
            return Err(invalid_import(
                "XDMF attribute contains a literal less-than sign",
            ));
        }
        let key = attribute.key.as_ref().to_owned();
        let value = attribute
            .normalized_value(XmlVersion::Explicit1_0)
            .map_err(|_| invalid_import("XDMF attribute contains an invalid reference"))?
            .into_owned();
        decoded = checked_add(decoded, key.len(), "XDMF decoded attribute bytes")?;
        decoded = checked_add(decoded, value.len(), "XDMF decoded attribute bytes")?;
        if attributes.insert(key, value).is_some() {
            return Err(invalid_import("XDMF element repeats an attribute"));
        }
    }
    Ok((name, attributes, decoded))
}

fn parse_topology(
    node: &Node,
    limits: XdmfImportLimits,
) -> Result<(usize, usize, DataItem), Diagnostic> {
    validate_element(node, "Topology", &["TopologyType", "NumberOfElements"])?;
    require_whitespace(node)?;
    let topology_type = require_attr(node, "TopologyType", "XDMF Topology")?;
    let (dimension, arity) = match topology_type {
        "Triangle" => (2, 3_u64),
        "Tetrahedron" => (3, 4_u64),
        _ => {
            return Err(invalid_import(
                "XDMF TopologyType must be Triangle or Tetrahedron",
            ));
        }
    };
    let cell_count = parse_positive_usize(
        require_attr(node, "NumberOfElements", "XDMF Topology")?,
        "XDMF NumberOfElements",
    )?;
    let mut item = parse_data_item(only_child(node, "DataItem")?, XdmfScalarType::U64, limits)?;
    item.path.clone_from(&node.path);
    let expected = [
        u64::try_from(cell_count).map_err(|_| invalid_import("XDMF cell count exceeds u64"))?,
        arity,
    ];
    if item.shape != expected {
        return Err(invalid_import(
            "XDMF topology shape differs from NumberOfElements and cell arity",
        ));
    }
    Ok((dimension, cell_count, item))
}

fn parse_geometry(
    node: &Node,
    limits: XdmfImportLimits,
) -> Result<(GeometryKind, usize, DataItem), Diagnostic> {
    validate_element(node, "Geometry", &["GeometryType"])?;
    require_whitespace(node)?;
    let (kind, width) = match require_attr(node, "GeometryType", "XDMF Geometry")? {
        "XY" => (GeometryKind::Xy, 2_u64),
        "XYZ" => (GeometryKind::Xyz, 3_u64),
        _ => return Err(invalid_import("XDMF GeometryType must be XY or XYZ")),
    };
    let mut item = parse_data_item(only_child(node, "DataItem")?, XdmfScalarType::F64, limits)?;
    item.path.clone_from(&node.path);
    if item.shape.len() != 2 || item.shape[1] != width {
        return Err(invalid_import(
            "XDMF geometry shape differs from GeometryType",
        ));
    }
    let vertices = usize::try_from(item.shape[0])
        .map_err(|_| invalid_import("XDMF vertex count exceeds local usize"))?;
    Ok((kind, vertices, item))
}

fn parse_attribute(node: &Node, limits: XdmfImportLimits) -> Result<AttributePlan, Diagnostic> {
    validate_element(node, "Attribute", &["Name", "AttributeType", "Center"])?;
    require_whitespace(node)?;
    let association = match require_attr(node, "Center", "XDMF Attribute")? {
        "Node" => DiscreteFieldAssociation::Vertex,
        "Cell" => DiscreteFieldAssociation::Cell,
        _ => return Err(invalid_import("XDMF Attribute Center must be Node or Cell")),
    };
    let mut item = parse_data_item(only_child(node, "DataItem")?, XdmfScalarType::F64, limits)?;
    item.path.clone_from(&node.path);
    let shape = match require_attr(node, "AttributeType", "XDMF Attribute")? {
        "Scalar" => DiscreteFieldShape::Scalar,
        "Vector" => {
            if item.shape.len() != 2 {
                return Err(invalid_import(
                    "XDMF Vector Attribute requires rank-two Dimensions",
                ));
            }
            let components = u32::try_from(item.shape[1])
                .ok()
                .and_then(NonZeroU32::new)
                .ok_or_else(|| {
                    invalid_import("XDMF vector component count must fit positive u32")
                })?;
            DiscreteFieldShape::Vector { components }
        }
        _ => {
            return Err(invalid_import(
                "XDMF AttributeType must be Scalar or Vector",
            ));
        }
    };
    Ok(AttributePlan {
        field: FieldPlan {
            name: node.attributes.get("Name").cloned(),
            origin_selector: node.path.clone(),
            association,
            shape,
        },
        item,
    })
}

fn parse_data_item(
    node: &Node,
    scalar: XdmfScalarType,
    limits: XdmfImportLimits,
) -> Result<DataItem, Diagnostic> {
    validate_element(
        node,
        "DataItem",
        &["Format", "DataType", "Precision", "Dimensions"],
    )?;
    if !node.children.is_empty() {
        return Err(invalid_import(
            "XDMF DataItem must not contain child elements",
        ));
    }
    require_attr(node, "Format", "XDMF DataItem")
        .and_then(|value| require_exact(value, "HDF", "XDMF DataItem Format"))?;
    let required_type = match scalar {
        XdmfScalarType::U64 => "UInt",
        XdmfScalarType::F64 => "Float",
    };
    require_attr(node, "DataType", "XDMF DataItem")
        .and_then(|value| require_exact(value, required_type, "XDMF DataItem DataType"))?;
    require_attr(node, "Precision", "XDMF DataItem")
        .and_then(|value| require_exact(value, "8", "XDMF DataItem Precision"))?;
    let shape = parse_shape(require_attr(node, "Dimensions", "XDMF DataItem")?, limits)?;
    let (locator, dataset) = parse_hdf_reference(trim_xml_space(&node.text))?;
    Ok(DataItem {
        path: node.path.clone(),
        locator,
        dataset,
        scalar,
        shape,
    })
}

fn parse_shape(value: &str, limits: XdmfImportLimits) -> Result<Vec<u64>, Diagnostic> {
    let mut shape = Vec::new();
    for component in value.split_ascii_whitespace() {
        if shape.len() >= limits.max_array_rank {
            return Err(invalid_import(
                "XDMF array rank exceeds the configured limit",
            ));
        }
        let dimension = component.parse::<u64>().map_err(|_| {
            invalid_import("XDMF Dimensions must contain positive decimal integers")
        })?;
        if dimension == 0 {
            return Err(invalid_import("XDMF Dimensions must be positive"));
        }
        shape.push(dimension);
    }
    if shape.is_empty() {
        return Err(invalid_import("XDMF Dimensions must not be empty"));
    }
    let product = shape.iter().try_fold(1_usize, |product, dimension| {
        let dimension = usize::try_from(*dimension)
            .map_err(|_| invalid_import("XDMF array extent exceeds local usize"))?;
        product
            .checked_mul(dimension)
            .ok_or_else(|| invalid_import("XDMF array shape product overflows usize"))
    })?;
    if product > limits.max_array_values {
        return Err(invalid_import(
            "XDMF array shape exceeds the configured scalar-value limit",
        ));
    }
    Ok(shape)
}

fn parse_hdf_reference(value: &str) -> Result<(String, String), Diagnostic> {
    if value.is_empty() || value.chars().any(char::is_control) {
        return Err(invalid_import(
            "XDMF HDF DataItem reference must be nonempty printable text",
        ));
    }
    let delimiter = value
        .rfind(':')
        .ok_or_else(|| invalid_import("XDMF HDF DataItem requires locator:/dataset"))?;
    let (locator, dataset) = value.split_at(delimiter);
    let dataset = &dataset[1..];
    if locator.is_empty() || !dataset.starts_with('/') || dataset == "/" {
        return Err(invalid_import(
            "XDMF HDF DataItem requires a nonempty locator and absolute dataset path",
        ));
    }
    if dataset
        .split('/')
        .skip(1)
        .any(|part| part.is_empty() || matches!(part, "." | ".."))
    {
        return Err(invalid_import(
            "XDMF HDF dataset path must be canonical and absolute",
        ));
    }
    Ok((locator.to_owned(), dataset.to_owned()))
}

fn validate_field_shape(
    item: &DataItem,
    shape: DiscreteFieldShape,
    entities: usize,
    topological_dimension: usize,
) -> Result<(), Diagnostic> {
    let entities = u64::try_from(entities)
        .map_err(|_| invalid_import("XDMF field entity count exceeds u64"))?;
    let valid = match shape {
        DiscreteFieldShape::Scalar => item.shape.as_slice() == [entities],
        DiscreteFieldShape::Vector { components } => {
            usize::try_from(components.get()) == Ok(topological_dimension)
                && item.shape.as_slice() == [entities, u64::from(components.get())]
        }
    };
    if !valid {
        return Err(invalid_import(
            "XDMF Attribute Dimensions differ from Center and AttributeType",
        ));
    }
    Ok(())
}

fn validate_declaration(declaration: &BytesDecl<'_>) -> Result<(), Diagnostic> {
    let raw = declaration.as_ref();
    let start = BytesStart::from_content(raw, 3);
    let attributes = start
        .attributes()
        .with_checks(true)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| invalid_import("XDMF XML declaration has malformed attributes"))?;
    if attributes.is_empty() || attributes.len() > 3 {
        return Err(invalid_import(
            "XDMF XML declaration requires version and at most encoding and standalone",
        ));
    }
    for (index, attribute) in attributes.iter().enumerate() {
        let key = attribute.key.as_ref();
        let value = attribute.value.as_ref();
        match (index, key) {
            (0, "version") if value == "1.0" => {}
            (1, "encoding") if value.eq_ignore_ascii_case("UTF-8") => {}
            (1, "standalone") if matches!(value, "yes" | "no") => {}
            (2, "standalone")
                if attributes[1].key.as_ref() == "encoding" && matches!(value, "yes" | "no") => {}
            _ => {
                return Err(invalid_import(
                    "XDMF XML declaration attributes are unknown, repeated, or out of order",
                ));
            }
        }
    }
    Ok(())
}

fn is_xml_1_0_character(character: char) -> bool {
    matches!(character, '\u{0009}' | '\u{000A}' | '\u{000D}')
        || ('\u{0020}'..='\u{D7FF}').contains(&character)
        || ('\u{E000}'..='\u{FFFD}').contains(&character)
        || ('\u{10000}'..='\u{10FFFF}').contains(&character)
}

fn push_request(requests: &mut Vec<XdmfArrayRequest>, role: XdmfArrayRole, item: DataItem) {
    requests.push(XdmfArrayRequest {
        ordinal: requests.len(),
        role,
        origin_selector: item.path,
        source_locator: item.locator,
        dataset_path: item.dataset,
        scalar: item.scalar,
        shape: item.shape,
    });
}

fn validate_element(node: &Node, name: &str, allowed: &[&str]) -> Result<(), Diagnostic> {
    if node.name != name {
        return Err(invalid_import(format!(
            "XDMF expected {name}, found {}",
            node.name
        )));
    }
    let allowed = allowed.iter().copied().collect::<BTreeSet<_>>();
    if node
        .attributes
        .keys()
        .any(|key| !allowed.contains(key.as_str()))
    {
        return Err(invalid_import(format!(
            "XDMF {name} contains an unsupported attribute"
        )));
    }
    Ok(())
}

fn require_whitespace(node: &Node) -> Result<(), Diagnostic> {
    if contains_only_xml_space(&node.text) {
        Ok(())
    } else {
        Err(invalid_import(format!(
            "XDMF {} contains unsupported text",
            node.name
        )))
    }
}

fn contains_only_xml_space(value: &str) -> bool {
    value.chars().all(is_xml_space)
}

fn trim_xml_space(value: &str) -> &str {
    value.trim_matches(is_xml_space)
}

const fn is_xml_space(character: char) -> bool {
    matches!(character, '\u{0009}' | '\u{000A}' | '\u{000D}' | '\u{0020}')
}

fn only_child<'a>(node: &'a Node, name: &str) -> Result<&'a Node, Diagnostic> {
    if node.children.len() != 1 || node.children[0].name != name {
        return Err(invalid_import(format!(
            "XDMF {} requires exactly one {name} child",
            node.name
        )));
    }
    Ok(&node.children[0])
}

fn require_attr<'a>(node: &'a Node, name: &str, owner: &str) -> Result<&'a str, Diagnostic> {
    node.attributes
        .get(name)
        .map(String::as_str)
        .ok_or_else(|| invalid_import(format!("{owner} requires attribute {name}")))
}

fn require_exact(value: &str, expected: &str, label: &str) -> Result<(), Diagnostic> {
    if value == expected {
        Ok(())
    } else {
        Err(invalid_import(format!("{label} must be {expected}")))
    }
}

fn parse_positive_usize(value: &str, label: &str) -> Result<usize, Diagnostic> {
    let value = value
        .parse::<usize>()
        .map_err(|_| invalid_import(format!("{label} must be a positive decimal integer")))?;
    if value == 0 {
        Err(invalid_import(format!("{label} must be positive")))
    } else {
        Ok(value)
    }
}

fn checked_add(left: usize, right: usize, label: &str) -> Result<usize, Diagnostic> {
    left.checked_add(right)
        .ok_or_else(|| invalid_import(format!("{label} overflows usize")))
}
