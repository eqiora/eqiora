"""Authenticate complete ordinary Q2 outputs before the independent comparator.

This module never solves or evaluates the manufactured function. Coordinates are
actual affine entity centroids, the nodal interpolation points of full Q2.
"""
from decimal import Decimal
from fractions import Fraction
import hashlib
import itertools
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


def authenticate_plan(plan, lengths, n):
    require(tuple(lengths) in ((2, 3), (2, 3, 4)) and n in (2, 4, 8), "undeclared box or level")
    raw = plan.to_bytes()
    wire = json.loads(raw)
    require(wire.get("schema") == "eqiora.resolved-common-plan/v28", "unexpected Plan schema")
    require(wire.get("assembly_quadrature") == {"kind": "gauss-legendre", "points_per_axis": 4}, "actual effective Plan quadrature is not four-point Gauss")
    require(wire.get("spatial") == {"kind": "uniform", "policy": "q2"}, "actual Plan is not uniform Q2")
    mesh = plan.mesh
    require(mesh is not None and mesh.dimension == len(lengths), "missing or incorrect actual Mesh")
    coordinates = tuple(tuple(float(x) for x in point) for point in mesh.coordinates)
    expected_vertices = set(itertools.product(*[[float(Fraction(length*k, n)) for k in range(n+1)] for length in lengths]))
    require(len(coordinates) == len(expected_vertices) and set(coordinates) == expected_vertices, "actual mesh vertex grid differs from declared physical box")
    actual_cells = []
    for cell in mesh.cells:
        vertices = tuple(int(index) for index in cell)
        require(len(vertices) == 2**len(lengths) and len(set(vertices)) == len(vertices), "incorrect cell connectivity")
        require(all(0 <= index < len(coordinates) for index in vertices), "cell vertex index out of bounds")
        actual_cells.append(frozenset(coordinates[index] for index in vertices))
    expected_cells = []
    for anchor in itertools.product(range(n), repeat=len(lengths)):
        expected_cells.append(frozenset(tuple(float(Fraction(lengths[a]*(anchor[a]+corner[a]), n)) for a in range(len(lengths))) for corner in itertools.product((0, 1), repeat=len(lengths))))
    require(len(actual_cells) == len(expected_cells) and set(actual_cells) == set(expected_cells), "actual mesh cell inventory differs from declared Cartesian grid")
    return raw, mesh, coordinates


def extract_candidate(plan, result, field, lengths, n):
    plan_raw, mesh, coordinates = authenticate_plan(plan, lengths, n)
    result_raw = result.to_bytes()
    try:
        result_wire = json.loads(result_raw)
    except (ValueError, TypeError) as error:
        raise ProductionError("Result is not a canonical JSON artifact") from error
    require(isinstance(result_wire, dict) and result_wire.get("schema") == "eqiora.common-result/v28", "unsupported Result artifact schema")
    content = result_wire.get("content")
    require(isinstance(content, dict) and content.get("plan_identity") == plan.identity
            and result.model_digest == plan.model_digest, "Result does not belong to the actual Plan/Model")
    dimension = len(lengths)
    output = result.output(field)
    require(output.field == field, "Result selected a different Field")
    require(output.mesh is not None and output.mesh.digest == mesh.digest, "Result output is not on the actual Plan mesh")
    require(output.space == "continuous-lagrange-p2", "Result output is not the canonical quadratic space")
    require(output.value_type.scalar_domain == "real" and output.value_type.shape == [], "candidate must be a real scalar Field")
    require(tuple(output.coefficient_dimension) == (0,)*7, "candidate coefficients are not dimensionless")
    entities = tuple(plan.field_coefficient_entities(field))
    require(entities == tuple(sorted(set(entities))), "coefficient entities are missing canonical uniqueness/order")
    associations = ["vertex", "edge"] + (["face"] if dimension == 3 else []) + ["cell"]
    require(tuple(output.associations) == tuple(associations), "Result lost or added a coefficient association")
    records = []
    retained = []
    for stratum, association in enumerate(associations):
        selected = [entity for entity in entities if entity[0] == stratum]
        expected_count = math.comb(dimension, stratum)*n**stratum*(n+1)**(dimension-stratum)
        require(len(selected) == expected_count, "bound Q2 entity stratum has an incorrect count")
        require(output.coefficient_count(association) == expected_count, "Result coefficient block count differs from bound entities")
        require(tuple(output.logical_shape(association)) == (expected_count,), "Result scalar block has an incorrect logical shape")
        values = output.values(association)
        require(len(values) == expected_count, "Result block drops or adds coefficients")
        for entity, value in zip(selected, values, strict=True):
            value = float(value)
            require(math.isfinite(value), "nonfinite candidate coefficient")
            vertices = tuple(mesh.entity_vertices(entity))
            require(len(vertices) == 2**stratum and len(set(vertices)) == len(vertices), "coefficient entity has invalid closure")
            require(all(0 <= index < len(coordinates) for index in vertices), "coefficient closure vertex out of bounds")
            corners = [coordinates[index] for index in vertices]
            varying = [a for a in range(dimension) if len({point[a] for point in corners}) == 2]
            require(len(varying) == stratum and all(len({point[a] for point in corners}) == (2 if a in varying else 1) for a in range(dimension)), "coefficient entity is not the proper affine Cartesian stratum")
            require(all(max(point[a] for point in corners)-min(point[a] for point in corners) == float(Fraction(lengths[a], n)) for a in varying), "coefficient entity spans an incorrect cell extent")
            point = tuple(sum((Fraction.from_float(p[a]) for p in corners), Fraction())/len(corners) for a in range(dimension))
            # These boxes and levels are dyadic: conversion preserves the exact centroid.
            physical = tuple(float(x) for x in point)
            require(all(Fraction.from_float(x) == exact for x, exact in zip(physical, point)), "centroid is not exactly binary64 representable")
            records.append({"coordinates_m": [decimal_binary64(x) for x in physical], "value_binary64_hex": value.hex()})
            retained.append({"entity": list(entity), "association": association, "coordinates_m": [decimal_binary64(x) for x in physical]})
    require(len(records) == (2*n+1)**dimension and len({tuple(record["coordinates_m"]) for record in records}) == len(records), "complete Q2 physical node inventory is not bijective")
    candidate = {"schema": 1, "lengths_m": list(lengths), "N": n, "gauss_points": 4, "records": records}
    identity = {"schema": 1, "plan_sha256": sha256(plan_raw), "result_sha256": sha256(result.to_bytes()), "plan_identity": plan.identity, "run_request_key": result.plan_key, "model_digest": plan.model_digest, "mesh_sha256": sha256(mesh.to_bytes()), "mesh_digest": mesh.digest, "coefficient_entities": retained}
    return candidate, identity
