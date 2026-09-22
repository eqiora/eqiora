use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;

use eqiora_core::Diagnostic;
use eqiora_meshing::{DiscreteFieldAssociation, DiscreteFieldShape};
use quick_xml::Reader;
use quick_xml::XmlVersion;
use quick_xml::events::{BytesDecl, BytesStart, Event};

use crate::plan::{
    FieldPlan, VtuCellKind, VtuImportLimits, VtuImportPlan, VtuSelection, invalid_import,
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

#[derive(Debug)]
struct WorkBudget {
    used: usize,
    maximum: usize,
}

impl WorkBudget {
    fn new(maximum: usize) -> Self {
        Self { used: 0, maximum }
    }

    fn charge(&mut self, amount: usize) -> Result<(), Diagnostic> {
        self.used = self
            .used
            .checked_add(amount)
            .ok_or_else(|| invalid_import("VTU parser work overflows usize"))?;
        if self.used > self.maximum {
            return Err(invalid_import(
                "VTU parser work exceeds the configured limit",
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
struct ResolvedBudget {
    values: usize,
    bytes: usize,
    maximum_values: usize,
    maximum_bytes: usize,
}

impl ResolvedBudget {
    fn new(limits: VtuImportLimits) -> Self {
        Self {
            values: 0,
            bytes: 0,
            maximum_values: limits.max_resolved_values,
            maximum_bytes: limits.max_resolved_bytes,
        }
    }

    fn charge<T>(&mut self, count: usize, label: &str) -> Result<(), Diagnostic> {
        self.values = checked_add(self.values, count, "VTU resolved value count")?;
        let bytes = count
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| invalid_import(format!("{label} byte count overflows usize")))?;
        self.bytes = checked_add(self.bytes, bytes, "VTU resolved byte count")?;
        if self.values > self.maximum_values || self.bytes > self.maximum_bytes {
            return Err(invalid_import(
                "VTU normalized content exceeds the configured aggregate resolved limit",
            ));
        }
        Ok(())
    }
}

struct ParseBudgets<'a> {
    work: &'a mut WorkBudget,
    resolved: &'a mut ResolvedBudget,
}

#[derive(Debug)]
struct PointsArray {
    selector: Vec<u32>,
    values: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VtkPrimitive {
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
    Float32,
    Float64,
}

impl VtkPrimitive {
    fn parse(value: &str) -> Result<Self, Diagnostic> {
        match value {
            "Int8" => Ok(Self::Int8),
            "UInt8" => Ok(Self::UInt8),
            "Int16" => Ok(Self::Int16),
            "UInt16" => Ok(Self::UInt16),
            "Int32" => Ok(Self::Int32),
            "UInt32" => Ok(Self::UInt32),
            "Int64" => Ok(Self::Int64),
            "UInt64" => Ok(Self::UInt64),
            "Float32" => Ok(Self::Float32),
            "Float64" => Ok(Self::Float64),
            _ => Err(invalid_import(
                "VTU field DataArray type must be a VTK primitive numeric type",
            )),
        }
    }
}

pub(crate) fn parse_document(
    source: &[u8],
    selection: VtuSelection,
    limits: VtuImportLimits,
) -> Result<VtuImportPlan, Diagnostic> {
    selection.validate_against(limits)?;
    if source.len() > limits.max_source_bytes {
        return Err(invalid_import(
            "VTU source exceeds the configured byte limit",
        ));
    }
    let mut work = WorkBudget::new(limits.max_parser_work);
    work.charge(source.len())?;
    let source_text = std::str::from_utf8(source)
        .map_err(|_| invalid_import("VTU source must be valid UTF-8"))?;
    if !source_text.chars().all(is_xml_1_0_character) {
        return Err(invalid_import(
            "VTU source contains a character outside XML 1.0",
        ));
    }
    let root = parse_xml_tree(source, limits, &mut work)?;
    let mut resolved = ResolvedBudget::new(limits);
    validate_element(
        &root,
        "VTKFile",
        &["type", "version", "byte_order", "header_type", "compressor"],
    )?;
    require_whitespace(&root)?;
    require_exact(
        require_attr(&root, "type", "VTKFile")?,
        "UnstructuredGrid",
        "VTKFile type",
    )?;
    match require_attr(&root, "version", "VTKFile")? {
        "0.1" | "1.0" => {}
        _ => return Err(invalid_import("VTKFile version must be 0.1 or 1.0")),
    }
    match require_attr(&root, "byte_order", "VTKFile")? {
        "LittleEndian" | "BigEndian" => {}
        _ => {
            return Err(invalid_import(
                "VTKFile byte_order must be LittleEndian or BigEndian",
            ));
        }
    }
    match root.attributes.get("header_type").map(String::as_str) {
        None | Some("UInt32" | "UInt64") => {}
        Some(_) => {
            return Err(invalid_import(
                "VTKFile header_type must be UInt32 or UInt64",
            ));
        }
    }
    if root.attributes.contains_key("compressor") {
        return Err(invalid_import(
            "VTKFile compressor is outside the admitted ASCII VTU subset",
        ));
    }

    let grid = only_child(&root, "UnstructuredGrid")?;
    validate_element(grid, "UnstructuredGrid", &[])?;
    require_whitespace(grid)?;
    let piece = only_child(grid, "Piece")?;
    validate_element(piece, "Piece", &["NumberOfPoints", "NumberOfCells"])?;
    require_whitespace(piece)?;
    if piece.path != selection.piece() {
        return Err(invalid_import(
            "VTU selection does not identify the admitted Piece",
        ));
    }
    let point_count = parse_positive_usize(
        require_attr(piece, "NumberOfPoints", "VTU Piece")?,
        "VTU NumberOfPoints",
    )?;
    let cell_count = parse_positive_usize(
        require_attr(piece, "NumberOfCells", "VTU Piece")?,
        "VTU NumberOfCells",
    )?;
    if point_count > limits.max_points {
        return Err(invalid_import(
            "VTU point count exceeds the configured limit",
        ));
    }
    if cell_count > limits.max_cells {
        return Err(invalid_import(
            "VTU cell count exceeds the configured limit",
        ));
    }

    let mut points = None;
    let mut cells = None;
    let mut point_data = None;
    let mut cell_data = None;
    for child in &piece.children {
        match child.name.as_str() {
            "Points" if points.replace(child).is_none() => {}
            "Cells" if cells.replace(child).is_none() => {}
            "PointData" if point_data.replace(child).is_none() => {}
            "CellData" if cell_data.replace(child).is_none() => {}
            "Points" | "Cells" | "PointData" | "CellData" => {
                return Err(invalid_import("VTU Piece repeats a structural child"));
            }
            _ => return Err(invalid_import("VTU Piece contains an unsupported element")),
        }
    }
    let points = points.ok_or_else(|| invalid_import("VTU Piece requires one Points"))?;
    let cells = cells.ok_or_else(|| invalid_import("VTU Piece requires one Cells"))?;
    let point_data =
        point_data.ok_or_else(|| invalid_import("VTU Piece requires one PointData"))?;
    let cell_data = cell_data.ok_or_else(|| invalid_import("VTU Piece requires one CellData"))?;

    validate_element(points, "Points", &[])?;
    require_whitespace(points)?;
    let geometry_node = only_child(points, "DataArray")?;
    let expected_geometry_values = point_count
        .checked_mul(3)
        .ok_or_else(|| invalid_import("VTU point/component product overflows usize"))?;
    let geometry_array =
        parse_points_array(geometry_node, expected_geometry_values, limits, &mut work)?;

    validate_element(cells, "Cells", &[])?;
    require_whitespace(cells)?;
    let mut connectivity = None;
    let mut offsets = None;
    let mut types = None;
    for child in &cells.children {
        validate_element(
            child,
            "DataArray",
            &[
                "type",
                "Name",
                "NumberOfComponents",
                "format",
                "RangeMin",
                "RangeMax",
            ],
        )?;
        let name = require_attr(child, "Name", "VTU Cells DataArray")?;
        match name {
            "connectivity" if connectivity.is_none() => {
                connectivity = Some(parse_integer_array(
                    child,
                    IntegerPolicy::General,
                    limits,
                    &mut work,
                )?);
            }
            "offsets" if offsets.is_none() => {
                offsets = Some(parse_integer_array(
                    child,
                    IntegerPolicy::General,
                    limits,
                    &mut work,
                )?);
            }
            "types" if types.is_none() => {
                types = Some(parse_integer_array(
                    child,
                    IntegerPolicy::UInt8,
                    limits,
                    &mut work,
                )?);
            }
            "connectivity" | "offsets" | "types" => {
                return Err(invalid_import("VTU Cells repeats a required DataArray"));
            }
            _ => {
                return Err(invalid_import(
                    "VTU Cells contains an unsupported DataArray",
                ));
            }
        }
    }
    let (_, connectivity) =
        connectivity.ok_or_else(|| invalid_import("VTU Cells requires connectivity DataArray"))?;
    let (_, offsets) =
        offsets.ok_or_else(|| invalid_import("VTU Cells requires offsets DataArray"))?;
    let (_, types) = types.ok_or_else(|| invalid_import("VTU Cells requires types DataArray"))?;
    require_value_count(offsets.len(), cell_count, "VTU offsets")?;
    require_value_count(types.len(), cell_count, "VTU cell types")?;
    let first_type = types
        .first()
        .copied()
        .ok_or_else(|| invalid_import("VTU cell type array must not be empty"))?;
    if types.iter().any(|cell_type| *cell_type != first_type) {
        return Err(invalid_import("VTU cell types must be homogeneous"));
    }
    let cell_kind = match first_type {
        5 => VtuCellKind::Triangle,
        10 => VtuCellKind::Tetrahedron,
        _ => {
            return Err(invalid_import(
                "VTU cell type must be Tri3 (5) or Tet4 (10)",
            ));
        }
    };
    let arity = cell_kind.arity();
    let expected_connectivity = cell_count
        .checked_mul(arity)
        .ok_or_else(|| invalid_import("VTU cell/arity product overflows usize"))?;
    require_value_count(
        connectivity.len(),
        expected_connectivity,
        "VTU connectivity",
    )?;
    for (cell, offset) in offsets.iter().copied().enumerate() {
        let expected = cell
            .checked_add(1)
            .and_then(|count| count.checked_mul(arity))
            .and_then(|value| u64::try_from(value).ok())
            .ok_or_else(|| invalid_import("VTU expected offset overflows u64"))?;
        if offset != expected {
            return Err(invalid_import(
                "VTU offsets do not describe fixed-width homogeneous simplices",
            ));
        }
    }
    let point_count_u64 =
        u64::try_from(point_count).map_err(|_| invalid_import("VTU point count exceeds u64"))?;
    if connectivity.iter().any(|index| *index >= point_count_u64) {
        return Err(invalid_import(
            "VTU connectivity references a missing point",
        ));
    }

    resolved.charge::<u64>(connectivity.len(), "VTU normalized topology")?;

    let normalized_geometry_values = point_count
        .checked_mul(cell_kind.dimension())
        .ok_or_else(|| invalid_import("VTU normalized geometry size overflows usize"))?;
    resolved.charge::<f64>(normalized_geometry_values, "VTU normalized geometry")?;
    let mut geometry = allocate_vec(normalized_geometry_values, "VTU normalized geometry")?;
    for point in geometry_array.values.as_chunks::<3>().0 {
        if cell_kind == VtuCellKind::Triangle && point[2] != 0.0 {
            return Err(invalid_import(
                "VTU Tri3 Points require exactly zero z coordinates",
            ));
        }
        for coordinate in &point[..cell_kind.dimension()] {
            geometry.push(*coordinate);
        }
    }

    let selected_fields = selection
        .fields()
        .iter()
        .map(Vec::as_slice)
        .collect::<BTreeSet<_>>();
    let mut available_fields = BTreeMap::<Vec<u32>, FieldPlan>::new();
    let mut budgets = ParseBudgets {
        work: &mut work,
        resolved: &mut resolved,
    };
    parse_field_container(
        point_data,
        DiscreteFieldAssociation::Vertex,
        point_count,
        limits,
        &selected_fields,
        &mut budgets,
        &mut available_fields,
    )?;
    parse_field_container(
        cell_data,
        DiscreteFieldAssociation::Cell,
        cell_count,
        limits,
        &selected_fields,
        &mut budgets,
        &mut available_fields,
    )?;
    let mut fields = Vec::new();
    fields
        .try_reserve_exact(selection.fields().len())
        .map_err(|_| invalid_import("VTU selected field allocation failed"))?;
    for selector in selection.fields() {
        let field = available_fields.remove(selector).ok_or_else(|| {
            invalid_import("VTU field selection references a missing PointData/CellData DataArray")
        })?;
        fields.push(field);
    }

    let source = copy_slice(source, "VTU source copy")?;
    let geometry_selector = copy_slice(&geometry_array.selector, "VTU geometry selector")?;
    let topology_selector = copy_slice(&cells.path, "VTU topology selector")?;

    Ok(VtuImportPlan {
        source,
        selection,
        limits,
        cell_kind,
        geometry_selector,
        topology_selector,
        geometry_shape: vec![
            u64::try_from(point_count)
                .map_err(|_| invalid_import("VTU point count exceeds u64"))?,
            u64::try_from(cell_kind.dimension())
                .map_err(|_| invalid_import("VTU dimension exceeds u64"))?,
        ],
        topology_shape: vec![
            u64::try_from(cell_count).map_err(|_| invalid_import("VTU cell count exceeds u64"))?,
            u64::try_from(arity).map_err(|_| invalid_import("VTU arity exceeds u64"))?,
        ],
        geometry,
        topology: connectivity,
        fields,
    })
}

fn parse_field_container(
    container: &Node,
    association: DiscreteFieldAssociation,
    entity_count: usize,
    limits: VtuImportLimits,
    selected_fields: &BTreeSet<&[u32]>,
    budgets: &mut ParseBudgets<'_>,
    fields: &mut BTreeMap<Vec<u32>, FieldPlan>,
) -> Result<(), Diagnostic> {
    const ACTIVE_ARRAY_ATTRIBUTES: &[&str] =
        &["Scalars", "Vectors", "Normals", "Tensors", "TCoords"];

    let expected_name = match association {
        DiscreteFieldAssociation::Vertex => "PointData",
        DiscreteFieldAssociation::Cell => "CellData",
    };
    validate_element(container, expected_name, ACTIVE_ARRAY_ATTRIBUTES)?;
    require_whitespace(container)?;
    for active_name in ACTIVE_ARRAY_ATTRIBUTES {
        if container
            .attributes
            .get(*active_name)
            .is_some_and(String::is_empty)
        {
            return Err(invalid_import(format!(
                "VTU {expected_name} {active_name} display name must not be empty"
            )));
        }
    }
    for child in &container.children {
        let selected = selected_fields.contains(child.path.as_slice());
        if let Some((selector, field)) =
            parse_field_array(child, association, entity_count, selected, limits, budgets)?
            && fields.insert(selector, field).is_some()
        {
            return Err(invalid_import(
                "VTU selected field selectors must be unique",
            ));
        }
    }
    for active_name in ACTIVE_ARRAY_ATTRIBUTES {
        if let Some(name) = container.attributes.get(*active_name)
            && !container.children.iter().any(|child| {
                child
                    .attributes
                    .get("Name")
                    .is_some_and(|candidate| candidate == name)
            })
        {
            return Err(invalid_import(format!(
                "VTU {expected_name} {active_name} names no contained DataArray"
            )));
        }
    }
    Ok(())
}

fn parse_field_array(
    node: &Node,
    association: DiscreteFieldAssociation,
    entity_count: usize,
    selected: bool,
    limits: VtuImportLimits,
    budgets: &mut ParseBudgets<'_>,
) -> Result<Option<(Vec<u32>, FieldPlan)>, Diagnostic> {
    validate_element(
        node,
        "DataArray",
        &[
            "type",
            "Name",
            "NumberOfComponents",
            "format",
            "RangeMin",
            "RangeMax",
        ],
    )?;
    validate_range_metadata(node)?;
    validate_information_keys(node, limits, budgets.work)?;
    require_ascii_format(node)?;
    let name = require_attr(node, "Name", "VTU field DataArray")?;
    if name.is_empty() {
        return Err(invalid_import(
            "VTU field DataArray requires a nonempty Name",
        ));
    }
    let primitive = VtkPrimitive::parse(require_attr(node, "type", "VTU field DataArray")?)?;
    let components = node
        .attributes
        .get("NumberOfComponents")
        .map(|value| parse_positive_u32(value, "VTU NumberOfComponents"))
        .transpose()?;
    let component_count = match components {
        None => 1,
        Some(components) => usize::try_from(components.get())
            .map_err(|_| invalid_import("VTU field component count exceeds local usize"))?,
    };
    let expected_values = entity_count
        .checked_mul(component_count)
        .ok_or_else(|| invalid_import("VTU field shape product overflows usize"))?;
    require_token_count(
        &node.text,
        expected_values,
        limits,
        budgets.work,
        "VTU field",
    )?;

    if !selected {
        validate_primitive_tokens(&node.text, primitive, budgets.work)?;
        return Ok(None);
    }
    if primitive != VtkPrimitive::Float64 {
        return Err(invalid_import(
            "VTU selected fields must use Float64 in the first import profile",
        ));
    }
    budgets
        .resolved
        .charge::<f64>(expected_values, "VTU selected field")?;
    let values = decode_f64_tokens(&node.text, expected_values, budgets.work)?;
    let entity_count_u64 = u64::try_from(entity_count)
        .map_err(|_| invalid_import("VTU field entity count exceeds u64"))?;
    let (shape, raw_shape) = match components {
        None => (DiscreteFieldShape::Scalar, vec![entity_count_u64]),
        Some(components) if components.get() == 1 => {
            (DiscreteFieldShape::Scalar, vec![entity_count_u64])
        }
        Some(components) => (
            DiscreteFieldShape::Vector { components },
            vec![entity_count_u64, u64::from(components.get())],
        ),
    };
    let selector = copy_slice(&node.path, "VTU selected field selector")?;
    let field = FieldPlan {
        selector: copy_slice(&selector, "VTU selected field selector metadata")?,
        name: Some(copy_string(name, "VTU selected field name")?),
        association,
        shape,
        raw_shape,
        values,
    };
    Ok(Some((selector, field)))
}

fn parse_points_array(
    node: &Node,
    expected_values: usize,
    limits: VtuImportLimits,
    work: &mut WorkBudget,
) -> Result<PointsArray, Diagnostic> {
    validate_element(
        node,
        "DataArray",
        &[
            "type",
            "Name",
            "NumberOfComponents",
            "format",
            "RangeMin",
            "RangeMax",
        ],
    )?;
    validate_range_metadata(node)?;
    validate_information_keys(node, limits, work)?;
    require_exact(
        require_attr(node, "type", "VTU Points DataArray")?,
        "Float64",
        "VTU Points DataArray type",
    )?;
    require_ascii_format(node)?;
    require_exact(
        require_attr(node, "NumberOfComponents", "VTU Points DataArray")?,
        "3",
        "VTU Points NumberOfComponents",
    )?;
    require_token_count(&node.text, expected_values, limits, work, "VTU Points")?;
    Ok(PointsArray {
        selector: copy_slice(&node.path, "VTU Points selector")?,
        values: decode_f64_tokens(&node.text, expected_values, work)?,
    })
}

#[derive(Debug, Clone, Copy)]
enum IntegerPolicy {
    General,
    UInt8,
}

fn parse_integer_array(
    node: &Node,
    policy: IntegerPolicy,
    limits: VtuImportLimits,
    work: &mut WorkBudget,
) -> Result<(Vec<u32>, Vec<u64>), Diagnostic> {
    if !node.children.is_empty() {
        return Err(invalid_import(
            "VTU integer DataArray must not contain child elements",
        ));
    }
    validate_range_metadata(node)?;
    require_ascii_format(node)?;
    match node
        .attributes
        .get("NumberOfComponents")
        .map(String::as_str)
    {
        None | Some("1") => {}
        Some(_) => {
            return Err(invalid_import(
                "VTU Cells DataArray must have one component",
            ));
        }
    }
    let scalar = require_attr(node, "type", "VTU Cells DataArray")?;
    match policy {
        IntegerPolicy::General if matches!(scalar, "Int32" | "Int64" | "UInt32" | "UInt64") => {}
        IntegerPolicy::UInt8 if scalar == "UInt8" => {}
        IntegerPolicy::General => {
            return Err(invalid_import(
                "VTU connectivity and offsets require Int32/Int64/UInt32/UInt64",
            ));
        }
        IntegerPolicy::UInt8 => {
            return Err(invalid_import("VTU cell types require UInt8"));
        }
    }
    let value_count = count_tokens(&node.text, limits, work)?;
    let mut values = allocate_vec(value_count, "VTU integer DataArray")?;
    for token in node.text.split_ascii_whitespace() {
        work.charge(1)?;
        let value = match scalar {
            "Int32" => u64::try_from(token.parse::<i32>().map_err(|_| {
                invalid_import("VTU integer DataArray contains a value outside Int32")
            })?)
            .map_err(|_| invalid_import("VTU integer DataArray values must be non-negative"))?,
            "Int64" => u64::try_from(token.parse::<i64>().map_err(|_| {
                invalid_import("VTU integer DataArray contains a value outside Int64")
            })?)
            .map_err(|_| invalid_import("VTU integer DataArray values must be non-negative"))?,
            "UInt32" => u64::from(token.parse::<u32>().map_err(|_| {
                invalid_import("VTU integer DataArray contains a value outside UInt32")
            })?),
            "UInt64" => token.parse::<u64>().map_err(|_| {
                invalid_import("VTU integer DataArray contains a value outside UInt64")
            })?,
            "UInt8" => u64::from(token.parse::<u8>().map_err(|_| {
                invalid_import("VTU integer DataArray contains a value outside UInt8")
            })?),
            _ => {
                return Err(invalid_import(
                    "VTU integer DataArray uses an unsupported scalar type",
                ));
            }
        };
        values.push(value);
    }
    Ok((
        copy_slice(&node.path, "VTU Cells DataArray selector")?,
        values,
    ))
}

fn validate_range_metadata(node: &Node) -> Result<(), Diagnostic> {
    match (
        node.attributes.get("RangeMin"),
        node.attributes.get("RangeMax"),
    ) {
        (None, None) => Ok(()),
        (Some(minimum), Some(maximum)) => {
            let minimum = minimum
                .parse::<f64>()
                .map_err(|_| invalid_import("VTU RangeMin must be finite Float64 metadata"))?;
            let maximum = maximum
                .parse::<f64>()
                .map_err(|_| invalid_import("VTU RangeMax must be finite Float64 metadata"))?;
            if !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
                return Err(invalid_import(
                    "VTU RangeMin/RangeMax must be finite and ordered",
                ));
            }
            Ok(())
        }
        _ => Err(invalid_import(
            "VTU RangeMin and RangeMax metadata must occur together",
        )),
    }
}

fn validate_information_keys(
    node: &Node,
    limits: VtuImportLimits,
    work: &mut WorkBudget,
) -> Result<(), Diagnostic> {
    if node.children.len() > 1 {
        return Err(invalid_import(
            "VTU DataArray permits at most one known InformationKey",
        ));
    }
    let Some(key) = node.children.first() else {
        return Ok(());
    };
    validate_element(key, "InformationKey", &["name", "location", "length"])?;
    require_whitespace(key)?;
    require_exact(
        require_attr(key, "name", "VTU InformationKey")?,
        "L2_NORM_RANGE",
        "VTU InformationKey name",
    )?;
    require_exact(
        require_attr(key, "location", "VTU InformationKey")?,
        "vtkDataArray",
        "VTU InformationKey location",
    )?;
    require_exact(
        require_attr(key, "length", "VTU InformationKey")?,
        "2",
        "VTU InformationKey length",
    )?;
    if key.children.len() != 2 {
        return Err(invalid_import(
            "VTU L2_NORM_RANGE InformationKey requires two Value children",
        ));
    }
    for (index, value) in key.children.iter().enumerate() {
        validate_element(value, "Value", &["index"])?;
        if !value.children.is_empty() {
            return Err(invalid_import(
                "VTU InformationKey Value must not contain child elements",
            ));
        }
        let expected_index = index.to_string();
        require_exact(
            require_attr(value, "index", "VTU InformationKey Value")?,
            &expected_index,
            "VTU InformationKey Value index",
        )?;
        require_token_count(&value.text, 1, limits, work, "VTU InformationKey Value")?;
        validate_primitive_tokens(&value.text, VtkPrimitive::Float64, work)?;
    }
    Ok(())
}

fn count_tokens(
    text: &str,
    limits: VtuImportLimits,
    work: &mut WorkBudget,
) -> Result<usize, Diagnostic> {
    let mut count = 0_usize;
    for _ in text.split_ascii_whitespace() {
        work.charge(1)?;
        if count >= limits.max_array_values {
            return Err(invalid_import(
                "VTU DataArray value count exceeds the configured limit",
            ));
        }
        count += 1;
    }
    if count == 0 {
        return Err(invalid_import("VTU DataArray values must not be empty"));
    }
    Ok(count)
}

fn require_token_count(
    text: &str,
    expected: usize,
    limits: VtuImportLimits,
    work: &mut WorkBudget,
    label: &str,
) -> Result<(), Diagnostic> {
    if expected > limits.max_array_values {
        return Err(invalid_import(format!(
            "{label} value count exceeds the configured per-array limit"
        )));
    }
    let actual = count_tokens(text, limits, work)?;
    require_value_count(actual, expected, label)
}

fn decode_f64_tokens(
    text: &str,
    expected: usize,
    work: &mut WorkBudget,
) -> Result<Vec<f64>, Diagnostic> {
    let mut values = allocate_vec(expected, "VTU Float64 DataArray")?;
    for token in text.split_ascii_whitespace() {
        work.charge(1)?;
        let value = token
            .parse::<f64>()
            .map_err(|_| invalid_import("VTU Float64 DataArray contains an invalid value"))?;
        if !value.is_finite() {
            return Err(invalid_import(
                "VTU Float64 DataArray values must all be finite",
            ));
        }
        values.push(value);
    }
    Ok(values)
}

fn validate_primitive_tokens(
    text: &str,
    primitive: VtkPrimitive,
    work: &mut WorkBudget,
) -> Result<(), Diagnostic> {
    for token in text.split_ascii_whitespace() {
        work.charge(1)?;
        let valid = match primitive {
            VtkPrimitive::Int8 => token.parse::<i8>().is_ok(),
            VtkPrimitive::UInt8 => token.parse::<u8>().is_ok(),
            VtkPrimitive::Int16 => token.parse::<i16>().is_ok(),
            VtkPrimitive::UInt16 => token.parse::<u16>().is_ok(),
            VtkPrimitive::Int32 => token.parse::<i32>().is_ok(),
            VtkPrimitive::UInt32 => token.parse::<u32>().is_ok(),
            VtkPrimitive::Int64 => token.parse::<i64>().is_ok(),
            VtkPrimitive::UInt64 => token.parse::<u64>().is_ok(),
            VtkPrimitive::Float32 => token.parse::<f32>().is_ok_and(f32::is_finite),
            VtkPrimitive::Float64 => token.parse::<f64>().is_ok_and(f64::is_finite),
        };
        if !valid {
            return Err(invalid_import(
                "VTU field DataArray contains a value outside its declared primitive type",
            ));
        }
    }
    Ok(())
}

fn require_ascii_format(node: &Node) -> Result<(), Diagnostic> {
    require_exact(
        require_attr(node, "format", "VTU DataArray")?,
        "ascii",
        "VTU DataArray format",
    )
}

fn parse_xml_tree(
    source: &[u8],
    limits: VtuImportLimits,
    work: &mut WorkBudget,
) -> Result<Node, Diagnostic> {
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
            .map_err(|_| invalid_import("VTU source is not well-formed XML"))?
        {
            Event::Start(start) => {
                saw_event_before_declaration = true;
                if root.is_some() && stack.is_empty() {
                    return Err(invalid_import("VTU source contains multiple root elements"));
                }
                elements = checked_add(elements, 1, "VTU element count")?;
                if elements > limits.max_elements {
                    return Err(invalid_import(
                        "VTU element count exceeds the configured limit",
                    ));
                }
                if stack.len() >= limits.max_depth {
                    return Err(invalid_import(
                        "VTU nesting depth exceeds the configured limit",
                    ));
                }
                let path = next_path(&mut stack)?;
                let (name, attributes, decoded) = parse_start(&start, limits, work)?;
                text_bytes = checked_add(text_bytes, decoded, "VTU decoded text bytes")?;
                if text_bytes > limits.max_text_bytes {
                    return Err(invalid_import(
                        "VTU decoded text exceeds the configured limit",
                    ));
                }
                stack
                    .try_reserve(1)
                    .map_err(|_| invalid_import("VTU XML stack allocation failed"))?;
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
                    .ok_or_else(|| invalid_import("VTU XML end tag has no start tag"))?;
                if end.name().as_ref() != builder.node.name {
                    return Err(invalid_import("VTU XML end tag differs from its start tag"));
                }
                if let Some(parent) = stack.last_mut() {
                    parent
                        .node
                        .children
                        .try_reserve(1)
                        .map_err(|_| invalid_import("VTU XML child allocation failed"))?;
                    parent.node.children.push(builder.node);
                } else if root.replace(builder.node).is_some() {
                    return Err(invalid_import("VTU source contains multiple root elements"));
                }
            }
            Event::Text(text) => {
                saw_event_before_declaration = true;
                if text.contains('&') || text.contains("]]>") {
                    return Err(invalid_import(
                        "VTU text references and forbidden XML sequences are unsupported",
                    ));
                }
                let decoded = text.as_ref();
                text_bytes = checked_add(text_bytes, decoded.len(), "VTU decoded text bytes")?;
                if text_bytes > limits.max_text_bytes {
                    return Err(invalid_import(
                        "VTU decoded text exceeds the configured limit",
                    ));
                }
                work.charge(decoded.len())?;
                if let Some(current) = stack.last_mut() {
                    current
                        .node
                        .text
                        .try_reserve(decoded.len())
                        .map_err(|_| invalid_import("VTU text allocation failed"))?;
                    current.node.text.push_str(decoded);
                } else if !contains_only_xml_space(decoded) {
                    return Err(invalid_import("VTU source has text outside its root"));
                }
            }
            Event::Comment(comment) => {
                saw_event_before_declaration = true;
                let decoded = comment.as_ref();
                text_bytes = checked_add(text_bytes, decoded.len(), "VTU decoded text bytes")?;
                if text_bytes > limits.max_text_bytes {
                    return Err(invalid_import(
                        "VTU decoded text exceeds the configured limit",
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
                        "VTU XML declaration must be the first event and occur once",
                    ));
                }
                validate_declaration(&declaration, work)?;
                saw_declaration = true;
            }
            Event::Empty(_) => {
                return Err(invalid_import(
                    "empty XML elements are outside the admitted VTU subset",
                ));
            }
            Event::CData(_) | Event::PI(_) | Event::DocType(_) | Event::GeneralRef(_) => {
                return Err(invalid_import(
                    "DTD, entities, CDATA, and processing instructions are outside the admitted VTU subset",
                ));
            }
            Event::Eof => break,
        }
    }
    if !stack.is_empty() {
        return Err(invalid_import("VTU source ended inside an element"));
    }
    root.ok_or_else(|| invalid_import("VTU source has no root element"))
}

fn next_path(stack: &mut [NodeBuilder]) -> Result<Vec<u32>, Diagnostic> {
    let Some(parent) = stack.last_mut() else {
        return Ok(Vec::new());
    };
    let child = parent.next_child;
    parent.next_child = parent
        .next_child
        .checked_add(1)
        .ok_or_else(|| invalid_import("VTU structural selector overflows u32"))?;
    let mut path = copy_slice(&parent.node.path, "VTU selector")?;
    path.try_reserve_exact(1)
        .map_err(|_| invalid_import("VTU selector allocation failed"))?;
    path.push(child);
    Ok(path)
}

fn parse_start(
    start: &BytesStart<'_>,
    limits: VtuImportLimits,
    work: &mut WorkBudget,
) -> Result<(String, BTreeMap<String, String>, usize), Diagnostic> {
    let name = copy_string(start.name().as_ref(), "VTU element name")?;
    if name.contains(':') {
        return Err(invalid_import(
            "XML namespaces are outside the admitted VTU subset",
        ));
    }
    let mut attributes = BTreeMap::new();
    let mut decoded = name.len();
    for (index, attribute) in start.attributes().with_checks(true).enumerate() {
        work.charge(1)?;
        if index >= limits.max_attributes_per_element {
            return Err(invalid_import(
                "VTU element attribute count exceeds the configured limit",
            ));
        }
        let attribute =
            attribute.map_err(|_| invalid_import("VTU element has malformed attributes"))?;
        if attribute.value.contains('<') || attribute.value.contains('&') {
            return Err(invalid_import(
                "VTU attributes may not contain references or literal less-than signs",
            ));
        }
        let key = copy_string(attribute.key.as_ref(), "VTU attribute name")?;
        if key.contains(':') {
            return Err(invalid_import(
                "XML namespaces are outside the admitted VTU subset",
            ));
        }
        let normalized = attribute
            .normalized_value(XmlVersion::Explicit1_0)
            .map_err(|_| invalid_import("VTU attribute contains an invalid reference"))?;
        let value = copy_string(normalized.as_ref(), "VTU attribute value")?;
        decoded = checked_add(decoded, key.len(), "VTU decoded attribute bytes")?;
        decoded = checked_add(decoded, value.len(), "VTU decoded attribute bytes")?;
        if attributes.insert(key, value).is_some() {
            return Err(invalid_import("VTU element repeats an attribute"));
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
                "VTU XML declaration requires version and at most encoding and standalone",
            ));
        }
        let attribute = attribute
            .map_err(|_| invalid_import("VTU XML declaration has malformed attributes"))?;
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
                    "VTU XML declaration attributes are unknown, repeated, or out of order",
                ));
            }
        }
        count += 1;
    }
    if count == 0 {
        return Err(invalid_import(
            "VTU XML declaration requires a version attribute",
        ));
    }
    Ok(())
}

fn validate_element(node: &Node, name: &str, allowed: &[&str]) -> Result<(), Diagnostic> {
    if node.name != name {
        return Err(invalid_import(format!(
            "VTU expected {name}, found {}",
            node.name
        )));
    }
    if node
        .attributes
        .keys()
        .any(|key| !allowed.contains(&key.as_str()))
    {
        return Err(invalid_import(format!(
            "VTU {name} contains an unsupported attribute"
        )));
    }
    Ok(())
}

fn require_whitespace(node: &Node) -> Result<(), Diagnostic> {
    if contains_only_xml_space(&node.text) {
        Ok(())
    } else {
        Err(invalid_import(format!(
            "VTU {} contains unsupported text",
            node.name
        )))
    }
}

fn only_child<'a>(node: &'a Node, name: &str) -> Result<&'a Node, Diagnostic> {
    if node.children.len() != 1 || node.children[0].name != name {
        return Err(invalid_import(format!(
            "VTU {} requires exactly one {name} child",
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
    let parsed = value
        .parse::<usize>()
        .map_err(|_| invalid_import(format!("{label} must be a positive decimal integer")))?;
    if parsed == 0 {
        Err(invalid_import(format!("{label} must be positive")))
    } else {
        Ok(parsed)
    }
}

fn parse_positive_u32(value: &str, label: &str) -> Result<NonZeroU32, Diagnostic> {
    value
        .parse::<u32>()
        .ok()
        .and_then(NonZeroU32::new)
        .ok_or_else(|| invalid_import(format!("{label} must fit positive u32")))
}

fn require_value_count(actual: usize, expected: usize, label: &str) -> Result<(), Diagnostic> {
    if actual == expected {
        Ok(())
    } else {
        Err(invalid_import(format!(
            "{label} requires {expected} values, received {actual}"
        )))
    }
}

fn checked_add(left: usize, right: usize, label: &str) -> Result<usize, Diagnostic> {
    left.checked_add(right)
        .ok_or_else(|| invalid_import(format!("{label} overflows usize")))
}

fn allocate_vec<T>(capacity: usize, label: &str) -> Result<Vec<T>, Diagnostic> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| invalid_import(format!("{label} allocation failed")))?;
    Ok(values)
}

fn copy_slice<T: Copy>(source: &[T], label: &str) -> Result<Vec<T>, Diagnostic> {
    let mut copy = allocate_vec(source.len(), label)?;
    copy.extend_from_slice(source);
    Ok(copy)
}

fn copy_string(source: &str, label: &str) -> Result<String, Diagnostic> {
    let mut copy = String::new();
    copy.try_reserve_exact(source.len())
        .map_err(|_| invalid_import(format!("{label} allocation failed")))?;
    copy.push_str(source);
    Ok(copy)
}

fn is_xml_1_0_character(character: char) -> bool {
    matches!(character, '\u{0009}' | '\u{000A}' | '\u{000D}')
        || ('\u{0020}'..='\u{D7FF}').contains(&character)
        || ('\u{E000}'..='\u{FFFD}').contains(&character)
        || ('\u{10000}'..='\u{10FFFF}').contains(&character)
}

fn contains_only_xml_space(value: &str) -> bool {
    value.chars().all(is_xml_space)
}

const fn is_xml_space(character: char) -> bool {
    matches!(character, '\u{0009}' | '\u{000A}' | '\u{000D}' | '\u{0020}')
}

#[cfg(test)]
mod tests {
    use eqiora_core::diagnostic::codes;
    use eqiora_meshing::{
        DiscreteFieldAssociation, DiscreteFieldShape, MeshQualityGate, MeshTopology,
    };

    use crate::plan::{PORTABLE_MAX_SELECTED_FIELDS, PORTABLE_MAX_SELECTOR_DEPTH};

    use super::*;

    const PIECE: &[u32] = &[0, 0];
    const POINT_FIELD: &[u32] = &[0, 0, 2, 0];
    const CELL_FIELD: &[u32] = &[0, 0, 3, 0];
    const OFFICIAL_VTK: &str = include_str!(
        "../../../verify/artifacts/vtu-unstructured-grid-import/fixtures/unit-square-tri3-ascii.vtu"
    );

    fn source(offset: &str, cell_type: &str, point_values: &str) -> String {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">
  <UnstructuredGrid>
    <Piece NumberOfPoints="3" NumberOfCells="1">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">0 0 0  1 0 0  0 1 0</DataArray>
      </Points>
      <Cells>
        <DataArray type="Int64" Name="connectivity" format="ascii">0 1 2</DataArray>
        <DataArray type="Int64" Name="offsets" format="ascii">{offset}</DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">{cell_type}</DataArray>
      </Cells>
      <PointData>
        <DataArray type="Float64" Name="temperature" NumberOfComponents="1" format="ascii">{point_values}</DataArray>
      </PointData>
      <CellData>
        <DataArray type="Float64" Name="flux" NumberOfComponents="2" format="ascii">4 5</DataArray>
      </CellData>
    </Piece>
  </UnstructuredGrid>
</VTKFile>"#
        )
    }

    fn tetrahedron_source(connectivity: &str) -> String {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="4" NumberOfCells="1">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
          0 0 0  1 0 0  0 1 0  0 0 1
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">{connectivity}</DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">4</DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">10</DataArray>
      </Cells>
      <PointData>
        <DataArray type="Float64" Name="temperature" format="ascii">1 2 3 4</DataArray>
      </PointData>
      <CellData>
        <DataArray type="Float64" Name="flux" NumberOfComponents="2" format="ascii">5 6</DataArray>
      </CellData>
    </Piece>
  </UnstructuredGrid>
</VTKFile>"#
        )
    }

    fn selection() -> VtuSelection {
        VtuSelection::new(
            PIECE.to_vec(),
            vec![POINT_FIELD.to_vec(), CELL_FIELD.to_vec()],
        )
        .unwrap()
    }

    #[test]
    fn selection_constructor_enforces_portable_metadata_bounds() {
        assert_eq!(
            VtuSelection::new(vec![0; PORTABLE_MAX_SELECTOR_DEPTH + 1], Vec::new())
                .unwrap_err()
                .code(),
            codes::INVALID_EXTERNAL_DATA_IMPORT
        );
        assert_eq!(
            VtuSelection::new(vec![0], vec![vec![0]; PORTABLE_MAX_SELECTED_FIELDS + 1],)
                .unwrap_err()
                .code(),
            codes::INVALID_EXTERNAL_DATA_IMPORT
        );

        let fields = (0..PORTABLE_MAX_SELECTED_FIELDS)
            .map(|index| {
                let mut selector = vec![0; PORTABLE_MAX_SELECTOR_DEPTH];
                selector[0] = u32::try_from(index).unwrap();
                selector
            })
            .collect();
        assert_eq!(
            VtuSelection::new(vec![u32::MAX; PORTABLE_MAX_SELECTOR_DEPTH], fields)
                .unwrap_err()
                .code(),
            codes::INVALID_EXTERNAL_DATA_IMPORT
        );
    }

    #[test]
    fn one_ascii_piece_reconstructs_shared_mesh_and_selected_fields() {
        let source = source("3", "5", "1 2 3");
        let plan = VtuImportPlan::parse(source.as_bytes(), selection(), VtuImportLimits::default())
            .unwrap();
        assert_eq!(plan.source_bytes(), source.as_bytes());
        assert_eq!(plan.selection().piece(), PIECE);
        assert_eq!(plan.cell_kind(), VtuCellKind::Triangle);
        assert_eq!(plan.geometry_selector(), &[0, 0, 0, 0]);
        assert_eq!(plan.topology_selector(), &[0, 0, 1]);
        assert_eq!(plan.geometry_shape(), &[3, 2]);
        assert_eq!(plan.topology_shape(), &[1, 3]);
        assert_eq!(plan.normalized_geometry(), &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        assert_eq!(plan.normalized_topology(), &[0, 1, 2]);

        let imported = plan.accept(MeshQualityGate::new(0.5).unwrap()).unwrap();
        assert_eq!(imported.mesh().topological_dimension(), 2);
        assert_eq!(imported.mesh().entity_count(0), Some(3));
        assert_eq!(imported.mesh().entity_count(2), Some(1));
        assert_eq!(imported.fields().len(), 2);
        assert_eq!(imported.fields()[0].selector(), POINT_FIELD);
        assert_eq!(imported.fields()[0].name(), Some("temperature"));
        assert_eq!(imported.fields()[0].raw_shape(), &[3]);
        assert_eq!(
            imported.fields()[0].payload().association(),
            DiscreteFieldAssociation::Vertex
        );
        assert_eq!(
            imported.fields()[0].payload().component_shape(),
            DiscreteFieldShape::Scalar
        );
        assert_eq!(imported.fields()[0].payload().values(), &[1.0, 2.0, 3.0]);
        assert_eq!(imported.fields()[1].selector(), CELL_FIELD);
        assert_eq!(imported.fields()[1].raw_shape(), &[1, 2]);
        assert_eq!(
            imported.fields()[1].payload().association(),
            DiscreteFieldAssociation::Cell
        );
    }

    #[test]
    fn official_vtk_metadata_is_bounded_and_nonsemantic() {
        let selection =
            VtuSelection::new(PIECE.to_vec(), vec![vec![0, 0, 0, 0], vec![0, 0, 1, 0]]).unwrap();
        let plan = VtuImportPlan::parse(
            OFFICIAL_VTK.as_bytes(),
            selection,
            VtuImportLimits::default(),
        )
        .unwrap();
        assert_eq!(plan.geometry_selector(), &[0, 0, 2, 0]);
        assert_eq!(plan.topology_selector(), &[0, 0, 3]);
        assert_eq!(plan.geometry_shape(), &[4, 2]);
        assert_eq!(plan.topology_shape(), &[2, 3]);
        let imported = plan.accept(MeshQualityGate::new(0.5).unwrap()).unwrap();
        assert_eq!(imported.mesh().entity_count(0), Some(4));
        assert_eq!(imported.mesh().entity_count(2), Some(2));
        assert_eq!(imported.fields()[0].name(), Some("temperature"));
        assert_eq!(imported.fields()[1].name(), Some("flux"));
    }

    #[test]
    fn wrong_offsets_fail_closed() {
        let error = VtuImportPlan::parse(
            source("2", "5", "1 2 3").as_bytes(),
            selection(),
            VtuImportLimits::default(),
        )
        .unwrap_err();
        assert_eq!(error.code(), codes::INVALID_EXTERNAL_DATA_IMPORT);
    }

    #[test]
    fn positive_tetrahedron_is_admitted_and_negative_orientation_is_rejected() {
        let positive = VtuImportPlan::parse(
            tetrahedron_source("0 1 2 3").as_bytes(),
            selection(),
            VtuImportLimits::default(),
        )
        .unwrap();
        assert_eq!(positive.cell_kind(), VtuCellKind::Tetrahedron);
        assert_eq!(positive.geometry_shape(), &[4, 3]);
        assert_eq!(positive.topology_shape(), &[1, 4]);
        assert_eq!(positive.normalized_topology(), &[0, 1, 2, 3]);
        assert_eq!(
            positive
                .accept(MeshQualityGate::new(1.0e-12).unwrap())
                .unwrap()
                .mesh()
                .topological_dimension(),
            3
        );

        let negative = VtuImportPlan::parse(
            tetrahedron_source("0 2 1 3").as_bytes(),
            selection(),
            VtuImportLimits::default(),
        )
        .unwrap();
        assert_eq!(
            negative
                .accept(MeshQualityGate::new(1.0e-12).unwrap())
                .unwrap_err()
                .code(),
            codes::INVALID_EXTERNAL_DATA_IMPORT
        );
    }

    #[test]
    fn unselected_primitive_fields_are_preflighted_without_retention() {
        let source = source("3", "5", "1 2 3")
            .replace("<PointData>", "<PointData Normals=\"temperature\">")
            .replace("<CellData>", "<CellData Tensors=\"flux\">")
            .replace(
                "type=\"Float64\" Name=\"temperature\"",
                "type=\"Int16\" Name=\"temperature\"",
            )
            .replace(
                "type=\"Float64\" Name=\"flux\"",
                "type=\"Float32\" Name=\"flux\"",
            );
        let unselected = VtuSelection::new(PIECE.to_vec(), Vec::new()).unwrap();
        let plan = VtuImportPlan::parse(
            source.as_bytes(),
            unselected.clone(),
            VtuImportLimits::default(),
        )
        .unwrap();
        assert!(
            plan.accept(MeshQualityGate::new(0.5).unwrap())
                .unwrap()
                .fields()
                .is_empty()
        );

        let invalid = source.replace(">1 2 3</DataArray>", ">1 invalid 3</DataArray>");
        assert_eq!(
            VtuImportPlan::parse(invalid.as_bytes(), unselected, VtuImportLimits::default(),)
                .unwrap_err()
                .code(),
            codes::INVALID_EXTERNAL_DATA_IMPORT
        );
    }

    #[test]
    fn field_shape_count_and_required_containers_fail_closed() {
        let wrong_count = source("3", "5", "1 2");
        assert_eq!(
            VtuImportPlan::parse(
                wrong_count.as_bytes(),
                selection(),
                VtuImportLimits::default(),
            )
            .unwrap_err()
            .code(),
            codes::INVALID_EXTERNAL_DATA_IMPORT
        );

        let missing_cell_data = source("3", "5", "1 2 3").replace(
            "      <CellData>\n        <DataArray type=\"Float64\" Name=\"flux\" NumberOfComponents=\"2\" format=\"ascii\">4 5</DataArray>\n      </CellData>\n",
            "",
        );
        assert_eq!(
            VtuImportPlan::parse(
                missing_cell_data.as_bytes(),
                selection(),
                VtuImportLimits::default(),
            )
            .unwrap_err()
            .code(),
            codes::INVALID_EXTERNAL_DATA_IMPORT
        );
    }

    #[test]
    fn unsupported_cell_type_fails_closed() {
        let error = VtuImportPlan::parse(
            source("3", "9", "1 2 3").as_bytes(),
            selection(),
            VtuImportLimits::default(),
        )
        .unwrap_err();
        assert_eq!(error.code(), codes::INVALID_EXTERNAL_DATA_IMPORT);
    }

    #[test]
    fn nonfinite_field_fails_closed() {
        let error = VtuImportPlan::parse(
            source("3", "5", "1 NaN 3").as_bytes(),
            selection(),
            VtuImportLimits::default(),
        )
        .unwrap_err();
        assert_eq!(error.code(), codes::INVALID_EXTERNAL_DATA_IMPORT);
    }

    #[test]
    fn point_and_work_limits_fail_closed() {
        let source = source("3", "5", "1 2 3");
        let limits = VtuImportLimits {
            max_points: 2,
            ..VtuImportLimits::default()
        };
        assert_eq!(
            VtuImportPlan::parse(source.as_bytes(), selection(), limits)
                .unwrap_err()
                .code(),
            codes::INVALID_EXTERNAL_DATA_IMPORT
        );

        let limits = VtuImportLimits {
            max_parser_work: 8,
            ..VtuImportLimits::default()
        };
        let error = VtuImportPlan::parse(source.as_bytes(), selection(), limits).unwrap_err();
        assert_eq!(error.code(), codes::INVALID_EXTERNAL_DATA_IMPORT);
        assert!(
            error
                .message()
                .contains("parser work exceeds the configured limit"),
            "source scan reached the wrong rejection gate: {error}",
        );

        let limits = VtuImportLimits {
            max_selected_fields: 1,
            ..VtuImportLimits::default()
        };
        assert_eq!(
            VtuImportPlan::parse(source.as_bytes(), selection(), limits)
                .unwrap_err()
                .code(),
            codes::INVALID_EXTERNAL_DATA_IMPORT
        );

        let limits = VtuImportLimits {
            max_selector_depth: 3,
            ..VtuImportLimits::default()
        };
        assert_eq!(
            VtuImportPlan::parse(source.as_bytes(), selection(), limits)
                .unwrap_err()
                .code(),
            codes::INVALID_EXTERNAL_DATA_IMPORT
        );

        let limits = VtuImportLimits {
            max_selector_values: 9,
            ..VtuImportLimits::default()
        };
        assert_eq!(
            VtuImportPlan::parse(source.as_bytes(), selection(), limits)
                .unwrap_err()
                .code(),
            codes::INVALID_EXTERNAL_DATA_IMPORT
        );

        let limits = VtuImportLimits {
            max_resolved_bytes: 8,
            ..VtuImportLimits::default()
        };
        assert_eq!(
            VtuImportPlan::parse(source.as_bytes(), selection(), limits)
                .unwrap_err()
                .code(),
            codes::INVALID_EXTERNAL_DATA_IMPORT
        );
    }
}
