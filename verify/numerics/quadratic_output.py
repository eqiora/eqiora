"""Authenticate and export complete quadratic scalar coefficients through public APIs.

Consumers own mesh authentication, expected entity inventories, and element-specific
closure checks. This module does not solve or evaluate a reference function.
"""
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math


class ProductionError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ProductionError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def decimal_binary64(value):
    """Exact decimal spelling of a finite binary64 physical coordinate."""
    require(math.isfinite(value), "nonfinite coordinate")
    return str(Decimal.from_float(value))


def authenticate_output(plan, result, field, mesh):
    result_raw = result.to_bytes()
    try:
        result_wire = json.loads(result_raw)
    except (ValueError, TypeError) as error:
        raise ProductionError("Result is not a canonical JSON artifact") from error
    require(isinstance(result_wire, dict) and result_wire.get("schema") == "eqiora.common-result/v28", "unsupported Result artifact schema")
    content = result_wire.get("content")
    require(isinstance(content, dict) and content.get("plan_identity") == plan.identity
            and result.model_digest == plan.model_digest, "Result does not belong to the actual Plan/Model")
    output = result.output(field)
    require(output.field == field, "Result selected a different Field")
    require(output.mesh is not None and output.mesh.digest == mesh.digest, "Result output is not on the actual Plan mesh")
    require(output.space == "continuous-lagrange-p2", "Result output is not the canonical quadratic space")
    require(output.value_type.scalar_domain == "real" and output.value_type.shape == [], "candidate must be a real scalar Field")
    require(tuple(output.coefficient_dimension) == (0,)*7, "candidate coefficients are not dimensionless")
    return output


def extract_coefficients(plan, result, field, mesh, coordinates, expected, validate_closure):
    """Export each declared (association, entity tuple) once in canonical order."""
    output = authenticate_output(plan, result, field, mesh)
    entities = tuple(plan.field_coefficient_entities(field))
    require(entities == tuple(sorted(set(entities))), "coefficient entities are missing canonical uniqueness/order")
    require(entities == tuple(entity for _, selected in expected for entity in selected),
            "bound coefficient entities differ from the expected complete inventory")
    require(tuple(output.associations) == tuple(name for name, _ in expected),
            "Result lost or added a coefficient association")
    records, retained = [], []
    for association, selected in expected:
        count = len(selected)
        require(output.coefficient_count(association) == count, "Result coefficient block count differs from bound entities")
        require(tuple(output.logical_shape(association)) == (count,), "Result scalar block has an incorrect logical shape")
        values = output.values(association)
        require(len(values) == count, "Result block drops or adds coefficients")
        for entity, value in zip(selected, values, strict=True):
            value = float(value)
            require(math.isfinite(value), "nonfinite candidate coefficient")
            vertices = tuple(mesh.entity_vertices(entity))
            require(vertices and len(set(vertices)) == len(vertices), "coefficient entity has invalid closure")
            require(all(0 <= index < len(coordinates) for index in vertices), "coefficient closure vertex out of bounds")
            corners = [coordinates[index] for index in vertices]
            validate_closure(entity, vertices, corners)
            point = tuple(sum((Fraction.from_float(p[a]) for p in corners), Fraction()) / len(corners)
                          for a in range(mesh.dimension))
            physical = tuple(float(x) for x in point)
            require(all(Fraction.from_float(x) == exact for x, exact in zip(physical, point)),
                    "centroid is not exactly binary64 representable")
            coordinate_strings = [decimal_binary64(x) for x in physical]
            records.append({"coordinates_m": coordinate_strings, "value_binary64_hex": value.hex()})
            retained.append({"entity": list(entity), "association": association, "coordinates_m": coordinate_strings})
    require(len({tuple(record["coordinates_m"]) for record in records}) == len(records),
            "complete physical node inventory is not bijective")
    return records, retained


def artifact_identity(plan, result, mesh, plan_raw, retained):
    return {"schema": 1, "plan_sha256": sha256(plan_raw), "result_sha256": sha256(result.to_bytes()),
            "plan_identity": plan.identity, "run_request_key": result.plan_key, "model_digest": plan.model_digest,
            "mesh_sha256": sha256(mesh.to_bytes()), "mesh_digest": mesh.digest, "coefficient_entities": retained}
