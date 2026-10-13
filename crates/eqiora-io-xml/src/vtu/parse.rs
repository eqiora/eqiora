use crate::allocation::{allocate_vec, copy_slice, copy_string};
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;

use crate::xml::{self, Node, WorkBudget, contains_only_xml_space};
use eqiora_core::Diagnostic;
use eqiora_meshing::{DiscreteFieldAssociation, DiscreteFieldShape};

use super::plan::{
    FieldPlan, VtuCellKind, VtuImportLimits, VtuImportPlan, VtuSelection, invalid_import,
};

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
    let mut work = WorkBudget::new(limits.max_parser_work, "VTU");
    let root = xml::parse_tree(
        source,
        xml::Limits {
            max_source_bytes: limits.max_source_bytes,
            max_depth: limits.max_depth,
            max_elements: limits.max_elements,
            max_attributes_per_element: limits.max_attributes_per_element,
            max_text_bytes: limits.max_text_bytes,
            attribute_references: false,
        },
        &mut work,
        |_| Ok(()),
    )?;
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

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
