"""Authenticate complete ordinary Q2 outputs before the independent comparator.

This module never solves or evaluates the manufactured function. Coordinates are
actual affine entity centroids, the nodal interpolation points of full Q2.
"""
from decimal import Decimal
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quadratic_output import (ProductionError, require, sha256, decimal_binary64,
                              extract_coefficients, artifact_identity)


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
    dimension = len(lengths)
    associations = ["vertex", "edge"] + (["face"] if dimension == 3 else []) + ["cell"]
    expected = tuple((association, tuple((stratum, index) for index in range(
        math.comb(dimension, stratum)*n**stratum*(n+1)**(dimension-stratum))))
        for stratum, association in enumerate(associations))

    def validate_closure(entity, vertices, corners):
        stratum = entity[0]
        require(len(vertices) == 2**stratum, "coefficient entity has invalid closure")
        varying = [a for a in range(dimension) if len({point[a] for point in corners}) == 2]
        require(len(varying) == stratum and all(len({point[a] for point in corners}) == (2 if a in varying else 1) for a in range(dimension)), "coefficient entity is not the proper affine Cartesian stratum")
        require(all(max(point[a] for point in corners)-min(point[a] for point in corners) == float(Fraction(lengths[a], n)) for a in varying), "coefficient entity spans an incorrect cell extent")

    records, retained = extract_coefficients(plan, result, field, mesh, coordinates, expected, validate_closure)
    require(len(records) == (2*n+1)**dimension, "complete Q2 physical node inventory is not bijective")
    candidate = {"schema": 1, "lengths_m": list(lengths), "N": n, "gauss_points": 4, "records": records}
    return candidate, artifact_identity(plan, result, mesh, plan_raw, retained)
