#!/usr/bin/env python3
"""Independent exact General-profile oracle for host-serial solver planning v10.

This script deliberately contains a literal copy of the public policy table. It
uses no Rust code and does not discover candidates from the implementation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import struct
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any


POLICY_ID = "eqiora.solver-planning/v10"
EXPECTED_PATH = Path(__file__).resolve().parents[1] / "expected" / "policy-v10.json"

REFERENCE_ID = "eqiora.reference.bicgstab-general-jacobi-reproducible-f64"
FAER_BICGSTAB_ID = "eqiora.faer.bicgstab-general-jacobi-fast-f64"
REFERENCE_IDENTITY_ID = "eqiora.reference.bicgstab-general-identity-reproducible-f64"
FAER_IDENTITY_ID = "eqiora.faer.bicgstab-general-identity-fast-f64"
FAER_SPARSE_LU_ID = "eqiora.faer.sparse-lu-general-identity-fast-f64"

# This is an independent literal policy table, not a transcription of Rust.
# Lower integer values rank first; candidate ID is always the final key.
POLICY_TABLE = {
    "Robust": (
        ("reduction", {"Reproducible": 0, "Fast": 1}),
        ("algorithm", {"ConjugateGradient": 0, "MinimumResidual": 0,
                       "BiConjugateGradientStabilized": 1, "SparseLu": 2}),
    ),
    "Fast": (
        ("reduction", {"Fast": 0, "Reproducible": 1}),
        (
            "algorithm",
            {"SparseLu": 0, "ConjugateGradient": 1, "MinimumResidual": 1, "BiConjugateGradientStabilized": 2},
        ),
    ),
    "LowMemory": (
        (
            "algorithm",
            {"BiConjugateGradientStabilized": 0, "ConjugateGradient": 0, "MinimumResidual": 0, "SparseLu": 1},
        ),
        ("algorithm", {"ConjugateGradient": 0, "MinimumResidual": 0,
                       "BiConjugateGradientStabilized": 1, "SparseLu": 2}),
    ),
}

SELECTED_REASON = {
    "Robust": "candidate.selected.robust-preference",
    "Fast": "candidate.selected.fast-preference",
    "LowMemory": "candidate.selected.low-memory-preference",
}

EXPECTED_SELECTION = {
    "Robust": REFERENCE_IDENTITY_ID,
    "Fast": FAER_SPARSE_LU_ID,
    "LowMemory": FAER_IDENTITY_ID,
}

REFERENCE_PROVIDER = {
    "id": "eqiora.reference",
    "implementation_version": "0.1.5",
    "libraries": [],
}
FAER_PROVIDER = {
    "id": "eqiora.faer",
    "implementation_version": "0.1.5",
    "libraries": [{"name": "faer", "version": "0.24.4"}],
}
EXECUTION_PROVIDER = {
    "id": "eqiora.host.serial",
    "implementation_version": "0.1.5",
    "libraries": [],
}

CANDIDATES = (
    {
        "id": REFERENCE_ID,
        "evidence_case": "fluid.cartesian-advection-diffusion-fvm-2d",
        "solver_provider": REFERENCE_PROVIDER,
        "tuple": {
            "algorithm": "BiConjugateGradientStabilized",
            "operator_properties": "General",
            "preconditioner": "Jacobi",
            "reduction": "Reproducible",
            "scalar_type": "F64",
        },
    },
    {
        "id": FAER_BICGSTAB_ID,
        "evidence_case": "numerics.linear-backends",
        "solver_provider": FAER_PROVIDER,
        "tuple": {
            "algorithm": "BiConjugateGradientStabilized",
            "operator_properties": "General",
            "preconditioner": "Jacobi",
            "reduction": "Fast",
            "scalar_type": "F64",
        },
    },
    {
        "id": REFERENCE_IDENTITY_ID,
        "evidence_case": "fluid.cell-centered-navier-stokes-fvm-2d",
        "solver_provider": REFERENCE_PROVIDER,
        "tuple": {
            "algorithm": "BiConjugateGradientStabilized",
            "operator_properties": "General",
            "preconditioner": "Identity",
            "reduction": "Reproducible",
            "scalar_type": "F64",
        },
    },
    {
        "id": FAER_IDENTITY_ID,
        "evidence_case": "electrical.parallel-dc-network",
        "solver_provider": FAER_PROVIDER,
        "tuple": {
            "algorithm": "BiConjugateGradientStabilized",
            "operator_properties": "General",
            "preconditioner": "Identity",
            "reduction": "Fast",
            "scalar_type": "F64",
        },
    },
    {
        "id": FAER_SPARSE_LU_ID,
        "evidence_case": "numerics.linear-backends",
        "solver_provider": FAER_PROVIDER,
        "tuple": {
            "algorithm": "SparseLu",
            "operator_properties": "General",
            "preconditioner": "Identity",
            "reduction": "Fast",
            "scalar_type": "F64",
        },
    },
)

REASON_VOCABULARY = (
    "catalog.evidence-mismatch",
    "catalog.provider-mismatch",
    "catalog.provider-unavailable",
    "catalog.scalar-domain-mismatch",
    "catalog.plan-mismatch",
    "profile.general-required",
    "profile.normal-required",
    "profile.canonical-csr-required",
    "profile.complete-diagonal-required",
    "capability.exact-tuple-required",
    "candidate.admitted",
    "cost.shape-unavailable-preference-fallback",
    "candidate.not-selected",
    "candidate.selected.robust-preference",
    "candidate.selected.fast-preference",
    "candidate.selected.low-memory-preference",
)

INVENTORY_FAILURES = (
    "catalog.missing-id",
    "catalog.duplicate-id",
    "catalog.unknown-id",
    "catalog.control-mismatch",
)

FALSIFIERS = (
    "input-permutation",
    "missing-candidate-id",
    "foreign-domain-complete-catalog",
    "foreign-operator-class-complete-catalog",
    "typed-backend-with-real-only-capabilities",
    "absent-optional-typed-provider",
    "absent-optional-provider-with-required-fast-reduction",
    "duplicate-candidate-id",
    "unknown-candidate-id",
    "stale-evidence-id",
    "simultaneous-stale-evidence-and-provider-precedence",
    "changed-provider-release",
    "changed-provider-dependency",
    "missing-provider-dependency",
    "extra-provider-dependency",
    "changed-algorithm",
    "changed-operator-properties",
    "changed-preconditioner",
    "changed-reduction",
    "changed-scalar",
    "changed-relative-tolerance-bits",
    "changed-absolute-tolerance-bits",
    "changed-iteration-limit",
    "non-general-properties",
    "transposed-orientation",
    "hand-built-problem",
    "matrix-free-problem",
    "missing-first-diagonal",
    "missing-second-diagonal",
    "missing-exact-capability",
    "rank-rejected-candidate",
    "reverse-robust-reduction-precedence",
    "reverse-fast-reduction-precedence",
    "reverse-fast-algorithm-precedence",
    "reverse-low-memory-algorithm-precedence",
    "input-enumeration-tie-break",
    "mutate-selected-plan",
    "substitute-problem-after-resolution",
    "backend-received-problem-pointer",
    "backend-received-exact-operator-identity",
    "failed-preflight-numerical-call",
    "extra-direct-actual-operator-apply",
    "extra-direct-actual-operator-diagonal",
    "execute-more-than-one-backend",
    "retry-after-selected-failure",
    "manual-decision-value-or-report-difference",
    "registered-private-aggregator-omission",
)


def f64_bits(value: float) -> str:
    return f"0x{struct.unpack('>Q', struct.pack('>d', value))[0]:016x}"


def rank_key(candidate: dict[str, Any], objective: str) -> tuple[Any, ...]:
    table = POLICY_TABLE[objective]
    return tuple(order[candidate["tuple"][axis]] for axis, order in table) + (
        candidate["id"],
    )


def ordered_trace(
    candidates: tuple[dict[str, Any], ...], objective: str
) -> list[list[str]]:
    selected = min(candidates, key=lambda candidate: rank_key(candidate, objective))[
        "id"
    ]
    trace: list[list[str]] = []
    for candidate in sorted(candidates, key=lambda candidate: candidate["id"]):
        trace.append([candidate["id"], "candidate.admitted"])
        trace.append([candidate["id"], "cost.shape-unavailable-preference-fallback"])
        trace.append(
            [
                candidate["id"],
                SELECTED_REASON[objective]
                if candidate["id"] == selected
                else "candidate.not-selected",
            ]
        )
    return trace


def ordered_rejection_trace(
    candidates: tuple[dict[str, Any], ...],
    objective: str,
    rejection_reasons: dict[str, str],
) -> tuple[str, list[list[str]]]:
    admitted = tuple(
        candidate for candidate in candidates if candidate["id"] not in rejection_reasons
    )
    if not admitted:
        raise AssertionError("reranking trace requires at least one admitted candidate")
    selected = min(admitted, key=lambda candidate: rank_key(candidate, objective))["id"]
    trace: list[list[str]] = []
    for candidate in sorted(candidates, key=lambda candidate: candidate["id"]):
        candidate_id = candidate["id"]
        if candidate_id in rejection_reasons:
            trace.append([candidate_id, rejection_reasons[candidate_id]])
        else:
            trace.append([candidate_id, "candidate.admitted"])
            trace.append([candidate_id, "cost.shape-unavailable-preference-fallback"])
            trace.append(
                [
                    candidate_id,
                    SELECTED_REASON[objective]
                    if candidate_id == selected
                    else "candidate.not-selected",
                ]
            )
    return selected, trace


def zero_numerical_ledger() -> dict[str, Any]:
    return {
        "faer_bicgstab_jacobi_backend_solve_calls": 0,
        "faer_bicgstab_identity_backend_solve_calls": 0,
        "faer_sparse_lu_backend_solve_calls": 0,
        "operator_identity": "exact-owned-canonical-view",
        "operator_apply_calls": 0,
        "operator_diagonal_calls": 0,
        "reference_jacobi_backend_solve_calls": 0,
        "reference_identity_backend_solve_calls": 0,
    }


def execution_ledgers(operator_apply_calls: int) -> dict[str, dict[str, Any]]:
    return {
        objective: {
            "backend_received_problem": "exact-resolved-problem-pointer",
            "backend_received_operator": "exact-owned-canonical-view",
            "operator_identity": "exact-owned-canonical-view",
            "operator_apply_calls": operator_apply_calls,
            "operator_diagonal_calls": 0,
            "retry_calls": 0,
            "selected_backend_solve_calls": 1,
            "selected_candidate_id": EXPECTED_SELECTION[objective],
            "unselected_backend_solve_calls": 0,
        }
        for objective in ("Robust", "Fast", "LowMemory")
    }


def admitted_subset_oracles(
    candidates: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    subsets = []
    candidate_ids = [candidate["id"] for candidate in candidates]
    for admitted_count in range(1, len(candidate_ids) + 1):
        for admitted_ids_tuple in itertools.combinations(candidate_ids, admitted_count):
            admitted_ids = set(admitted_ids_tuple)
            rejection_reasons = {
                candidate_id: "catalog.evidence-mismatch"
                for candidate_id in candidate_ids
                if candidate_id not in admitted_ids
            }
            objectives = []
            for objective in ("Robust", "Fast", "LowMemory"):
                selected, trace = ordered_rejection_trace(
                    candidates, objective, rejection_reasons
                )
                objectives.append(
                    {
                        "objective": objective,
                        "ordered_reasons": trace,
                        "selected_candidate_id": selected,
                    }
                )
            subsets.append(
                {
                    "admitted_candidate_ids": sorted(admitted_ids),
                    "objectives": objectives,
                    "rejected_candidate_ids": sorted(rejection_reasons),
                }
            )
    return subsets


def candidate_rejection_oracles(
    candidates: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    oracles = []
    for reason in (
        "catalog.evidence-mismatch",
        "catalog.provider-mismatch",
        "catalog.plan-mismatch",
        "capability.exact-tuple-required",
    ):
        objectives = []
        for objective in ("Robust", "Fast", "LowMemory"):
            selected, trace = ordered_rejection_trace(
                candidates, objective, {FAER_BICGSTAB_ID: reason}
            )
            objectives.append(
                {
                    "objective": objective,
                    "ordered_reasons": trace,
                    "selected_candidate_id": selected,
                }
            )
        oracles.append(
            {
                "objectives": objectives,
                "reason": reason,
                "rejected_candidate_id": FAER_BICGSTAB_ID,
            }
        )
    return oracles


def simultaneous_evidence_provider_precedence_oracle(
    candidates: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    reason = "catalog.evidence-mismatch"
    objectives = []
    for objective in ("Robust", "Fast", "LowMemory"):
        selected, trace = ordered_rejection_trace(
            candidates, objective, {FAER_BICGSTAB_ID: reason}
        )
        objectives.append(
            {
                "objective": objective,
                "ordered_reasons": trace,
                "selected_candidate_id": selected,
            }
        )
    return {
        "first_and_only_rejection_reason": reason,
        "mutated_candidate_id": FAER_BICGSTAB_ID,
        "simultaneous_mutations": [
            "registered_evidence_identity",
            "solver_provider_descriptor",
        ],
        "objectives": objectives,
    }


def solve_exact() -> tuple[list[Fraction], list[Fraction]]:
    matrix = [[Fraction(4), Fraction(1)], [Fraction(2), Fraction(3)]]
    rhs = [Fraction(6), Fraction(8)]
    determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    solution = [
        (rhs[0] * matrix[1][1] - matrix[0][1] * rhs[1]) / determinant,
        (matrix[0][0] * rhs[1] - rhs[0] * matrix[1][0]) / determinant,
    ]
    residual = [
        rhs[row] - sum(matrix[row][column] * solution[column] for column in range(2))
        for row in range(2)
    ]
    return solution, residual


def ratio(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def build_expected() -> dict[str, Any]:
    solution, residual = solve_exact()
    sorted_candidates = tuple(sorted(CANDIDATES, key=lambda candidate: candidate["id"]))
    no_admitted_trace = [
        [candidate["id"], "catalog.evidence-mismatch"]
        for candidate in sorted_candidates
    ]
    diagnostic_trace = ",".join(
        f"{candidate_id}={reason}" for candidate_id, reason in no_admitted_trace
    )
    objectives = []
    for objective in ("Robust", "Fast", "LowMemory"):
        selected = min(
            sorted_candidates, key=lambda candidate: rank_key(candidate, objective)
        )["id"]
        if selected != EXPECTED_SELECTION[objective]:
            raise AssertionError(f"literal {objective} table selected {selected}")
        objectives.append(
            {
                "objective": objective,
                "precedence": [
                    {
                        "axis": axis,
                        "ordered_values": [
                            value
                            for value, _ in sorted(
                                order.items(), key=lambda entry: entry[1]
                            )
                        ],
                    }
                    for axis, order in POLICY_TABLE[objective]
                ]
                + [{"axis": "candidate_id", "order": "ascending"}],
                "selected_candidate_id": selected,
                "ordered_reasons": ordered_trace(sorted_candidates, objective),
            }
        )

    return {
        "admitted_subsets": admitted_subset_oracles(sorted_candidates),
        "call_ledgers": {
            "direct_operator_self_control": {
                "operator_identity": "exact-owned-canonical-view",
                "problem_operator_apply_calls_before_reset": 1,
                "problem_operator_diagonal_calls_before_reset": 1,
                "operator_apply_calls_after_reset": 0,
                "operator_diagonal_calls_after_reset": 0,
            },
            "candidate_rejection_before_execution": {
                gate: zero_numerical_ledger()
                for gate in ("evidence", "provider", "plan", "capability")
            },
            "failed_preflight": {
                gate: zero_numerical_ledger()
                for gate in (
                    "catalog-inventory",
                    "common-controls",
                    "profile-general",
                    "profile-normal-or-canonical-csr",
                    "profile-canonical-csr",
                    "profile-jacobi-diagonal",
                    "capability-exact-tuple",
                )
            },
            "selected_failure": execution_ledgers(operator_apply_calls=0),
            "selected_success": execution_ledgers(operator_apply_calls=2),
        },
        "catalog_validation_precedence": [
            "evidence_identity",
            "solver_provider_descriptor",
            "exact_plan_tuple",
            "problem_profile",
            "backend_exact_capability",
        ],
        "catalog_validation_precedence_observation": simultaneous_evidence_provider_precedence_oracle(
            sorted_candidates
        ),
        "candidates": list(sorted_candidates),
        "candidate_rejection_reranking": candidate_rejection_oracles(
            sorted_candidates
        ),
        "common_controls": {
            "absolute_tolerance": "1e-14",
            "absolute_tolerance_f64_bits": f64_bits(float("1e-14")),
            "execution_provider": EXECUTION_PROVIDER,
            "initial_guess": "implicit-zero",
            "maximum_iterations": 100,
            "relative_tolerance": "1e-12",
            "relative_tolerance_f64_bits": f64_bits(float("1e-12")),
        },
        "fixture": {
            "canonical_csr": {
                "column_indices": [0, 1, 0, 1],
                "columns": 2,
                "properties": "General",
                "right_hand_side": [6, 8],
                "row_offsets": [0, 2, 4],
                "rows": 2,
                "values": [4, 1, 2, 3],
            },
            "componentwise_solution_bound": {
                "comparison": "absolute-error-less-than-or-equal",
                "power_of_two": -40,
                **ratio(Fraction(1, 2**40)),
            },
            "diagonal_inventory": [
                {"entry_index": 0, "row": 0, "value": 4},
                {"entry_index": 3, "row": 1, "value": 3},
            ],
            "matrix": [[4, 1], [2, 3]],
            "orientation": "Normal",
            "right_hand_side": [6, 8],
            "solution": [ratio(value) for value in solution],
            "true_residual": [ratio(value) for value in residual],
        },
        "falsifiers": list(FALSIFIERS),
        "inventory_failure_fragments": list(INVENTORY_FAILURES),
        "manual_decision_equality": {
            "linear_solution": "exact-PartialEq",
            "solve_report": "exact-PartialEq",
        },
        "no_admitted_candidate": {
            "diagnostic_code": "EQ0807",
            "diagnostic_message": f"{POLICY_ID} no admitted candidate; trace=[{diagnostic_trace}]",
            "ordered_reasons": no_admitted_trace,
        },
        "objectives": objectives,
        "policy_id": POLICY_ID,
        "reason_ordering": [
            "ascending_candidate_id",
            "one_first_rejection_or_candidate.admitted",
            "immediate_candidate.not-selected_for_admitted_nonselection",
            "immediate_single_objective_selected_reason_for_selection",
        ],
        "reason_vocabulary": list(REASON_VOCABULARY),
    }


# Each pair is an exact rational complex number (real, imaginary). These literal
# fixtures are independent of provider assembly and every Rust catalog record.
def cadd(a, b):
    return (a[0] + b[0], a[1] + b[1])


def cneg(a):
    return (-a[0], -a[1])


def cmul(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def cdiv(a, b):
    denominator = b[0] ** 2 + b[1] ** 2
    return (Fraction(a[0] * b[0] + a[1] * b[1], denominator),
            Fraction(a[1] * b[0] - a[0] * b[1], denominator))


def cadjoint(matrix):
    return [[(matrix[j][i][0], -matrix[j][i][1]) for j in range(2)] for i in range(2)]


def csolve(matrix, rhs):
    a, b = matrix[0]
    c, d = matrix[1]
    determinant = cadd(cmul(a, d), cneg(cmul(b, c)))
    return [cdiv(cadd(cmul(rhs[0], d), cneg(cmul(b, rhs[1]))), determinant),
            cdiv(cadd(cmul(a, rhs[1]), cneg(cmul(rhs[0], c))), determinant)]


def complex_oracles():
    fixtures = [
        ("General", "general", [[(1, 1), (2, 0)], [(0, 3), (4, -1)]], [(-5, 5), (-13, 9)], [(10, -6), (5, 11)]),
        ("ComplexSymmetric", "complex-symmetric", [[(2, 1), (1, 2)], [(1, 2), (4, -1)]], [(-4, 2), (-10, 10)], [(10, -3), (1, 8)]),
        ("Hermitian", "hermitian", [[(1, 0), (0, 2)], [(0, -2), (1, 0)]], [(-1, -2), (2, -1)], [(-4, 1), (-1, -1)]),
        ("HermitianPositiveDefinite", "hpd", [[(4, 0), (0, 1)], [(0, -1), (3, 0)]], [(3, 6), (-4, 2)], [(5, -3), (2, 7)]),
    ]
    oracles = []
    for properties, suffix, matrix, rhs, adjoint_rhs in fixtures:
        solution = csolve(matrix, rhs)
        adjoint_solution = csolve(cadjoint(matrix), adjoint_rhs)
        assert solution == [(1, 2), (-2, 1)]
        assert adjoint_solution == [(2, -1), (1, 3)]
        unconjugated = [[matrix[j][i] for j in range(2)] for i in range(2)]
        assert csolve(unconjugated, adjoint_rhs) != adjoint_solution
        if properties.startswith("Hermitian"):
            assert matrix == cadjoint(matrix)
            determinant = cadd(cmul(matrix[0][0], matrix[1][1]), cneg(cmul(matrix[0][1], matrix[1][0])))
            if properties == "HermitianPositiveDefinite":
                assert matrix[0][0][0] > 0 and determinant[0] > 0 and determinant[1] == 0
            else:
                assert determinant[0] < 0 and determinant[1] == 0
        else:
            assert matrix != cadjoint(matrix)
            assert (matrix[0][1] == matrix[1][0]) == (properties == "ComplexSymmetric")
        direct = {
            "id": f"eqiora.faer.sparse-lu-{suffix}-identity-fast-complex-f64",
            "evidence_case": "numerics.host-serial-solver-planning",
            "solver_provider": FAER_PROVIDER,
            "tuple": {"algorithm": "SparseLu", "operator_properties": properties,
                      "preconditioner": "Identity", "reduction": "Fast", "scalar_domain": "Complex", "scalar_type": "F64"},
        }
        # Literal existing mathematical tuples, independent of Rust inventory:
        # BiCGSTAB accepts every listed nonsingular complex class; CG additionally
        # accepts HPD, whose positive leading minors were proved above.
        reference = []
        for algorithm, spelling in [("BiConjugateGradientStabilized", "bicgstab")] + (
                [("ConjugateGradient", "cg")] if properties == "HermitianPositiveDefinite" else []):
            for preconditioner in ("Identity", "Jacobi"):
                reference.append({
                    "id": f"eqiora.reference.{spelling}-{suffix}-{preconditioner.lower()}-reproducible-complex-f64",
                    "evidence_case": "numerics.host-serial-solver-planning",
                    "solver_provider": REFERENCE_PROVIDER,
                    "tuple": {"algorithm": algorithm, "operator_properties": properties,
                              "preconditioner": preconditioner, "reduction": "Reproducible",
                              "scalar_domain": "Complex", "scalar_type": "F64"},
                })
        if properties in ("Hermitian", "HermitianPositiveDefinite"):
            # Hermitian realification is symmetric and retains the same spectrum;
            # identity MINRES needs neither a positive diagonal nor definiteness.
            for preconditioner in ("Identity", "Jacobi"):
                # This independent Hermitian witness has positive diagonal;
                # determinant<0 still forbids promoting it to HPD.
                reference.append({
                    "id": f"eqiora.reference.minres-{suffix}-{preconditioner.lower()}-reproducible-complex-f64",
                    "evidence_case": "numerics.host-serial-solver-planning",
                    "solver_provider": REFERENCE_PROVIDER,
                    "tuple": {"algorithm": "MinimumResidual", "operator_properties": properties,
                              "preconditioner": preconditioner, "reduction": "Reproducible",
                              "scalar_domain": "Complex", "scalar_type": "F64"},
                })
        iterative_faer = []
        for algorithm, spelling in [("BiConjugateGradientStabilized", "bicgstab")] + (
                [("ConjugateGradient", "cg")] if properties == "HermitianPositiveDefinite" else []):
            for preconditioner in ("Identity", "Jacobi"):
                iterative_faer.append({
                    "id": f"eqiora.faer.{spelling}-{suffix}-{preconditioner.lower()}-fast-complex-f64",
                    "evidence_case": "numerics.host-serial-solver-planning",
                    "solver_provider": FAER_PROVIDER,
                    "tuple": {"algorithm": algorithm, "operator_properties": properties,
                              "preconditioner": preconditioner, "reduction": "Fast",
                              "scalar_domain": "Complex", "scalar_type": "F64"},
                })
        candidates = tuple(reference + iterative_faer + [direct])
        objectives = []
        without_faer = []
        for complete_diagonal in (None, True):
            rejected = {candidate["id"]: "profile.complete-diagonal-required"
                        for candidate in candidates
                        if candidate["tuple"]["preconditioner"] == "Jacobi" and complete_diagonal is not True}
            for objective in POLICY_TABLE:
                selected, trace = ordered_rejection_trace(candidates, objective, rejected)
                # The unchanged unknown-size axes rank reproducible first for
                # Robust, iterative first for LowMemory, and the declared
                # specialized CG preference on HPD before the final literal ID.
                spelling = ("cg" if properties == "HermitianPositiveDefinite" else
                            "minres" if properties == "Hermitian" else "bicgstab")
                expected_id = (direct["id"] if objective == "Fast" else
                    f"eqiora.reference.{spelling}-{suffix}-identity-reproducible-complex-f64")
                if objective == "LowMemory" and properties in ("General", "ComplexSymmetric", "HermitianPositiveDefinite"):
                    preferred = "cg" if properties == "HermitianPositiveDefinite" else "bicgstab"
                    expected_id = f"eqiora.faer.{preferred}-{suffix}-identity-fast-complex-f64"
                assert selected == expected_id
                # Every rank key ends with a distinct candidate ID, hence sorting
                # defines a total order independent of input order. Exercise every
                # pair swap and cyclic placement without factorial enumeration.
                keys = [rank_key(candidate, objective) for candidate in candidates]
                assert len(set(keys)) == len(candidates)
                permutations = [candidates[i:] + candidates[:i] for i in range(len(candidates))]
                for i in range(len(candidates)):
                    for j in range(i + 1, len(candidates)):
                        swapped = list(candidates)
                        swapped[i], swapped[j] = swapped[j], swapped[i]
                        permutations.append(tuple(swapped))
                for permutation in permutations:
                    assert ordered_rejection_trace(permutation, objective, rejected) == (selected, trace)
                objectives.append({"complete_diagonal": complete_diagonal, "objective": objective,
                                   "selected_candidate_id": selected, "ordered_reasons": trace})
                no_provider = {**rejected, **{candidate["id"]: "catalog.provider-unavailable" for candidate in iterative_faer + [direct]}}
                reference_id, reference_trace = ordered_rejection_trace(candidates, objective, no_provider)
                assert reference_id == f"eqiora.reference.{spelling}-{suffix}-identity-reproducible-complex-f64"
                without_faer.append({"complete_diagonal": complete_diagonal, "objective": objective,
                                     "selected_candidate_id": reference_id, "ordered_reasons": reference_trace})
        oracles.append({"properties": properties, "matrix": matrix, "right_hand_side": rhs,
                        "solution": [[ratio(v) for v in z] for z in solution],
                        "adjoint_right_hand_side": adjoint_rhs,
                        "adjoint_solution": [[ratio(v) for v in z] for z in adjoint_solution],
                        "componentwise_error_bound": "2^-40", "candidates": candidates, "objectives": objectives,
                        "without_faer_objectives": without_faer})
    return oracles


def validate_mutants(expected: dict[str, Any]) -> None:
    candidates = tuple(expected["candidates"])
    by_id = {candidate["id"]: candidate for candidate in candidates}

    # Every input permutation must produce the same selection and ID-ordered trace.
    for permutation in itertools.permutations(candidates):
        for objective in POLICY_TABLE:
            selected = min(
                permutation, key=lambda candidate: rank_key(candidate, objective)
            )["id"]
            assert selected == EXPECTED_SELECTION[objective]
            assert ordered_trace(permutation, objective) == ordered_trace(
                candidates, objective
            )

    # Every nonempty admitted subset has one exact selection and complete
    # ID-ordered trace. Rejection is represented independently of ranking.
    for subset in expected["admitted_subsets"]:
        admitted_ids = set(subset["admitted_candidate_ids"])
        rejection_reasons = {
            candidate["id"]: "catalog.evidence-mismatch"
            for candidate in candidates
            if candidate["id"] not in admitted_ids
        }
        for objective_oracle in subset["objectives"]:
            selected, trace = ordered_rejection_trace(
                candidates, objective_oracle["objective"], rejection_reasons
            )
            assert selected == objective_oracle["selected_candidate_id"]
            assert trace == objective_oracle["ordered_reasons"]

    # A candidate failing evidence and provider identity simultaneously must
    # expose evidence as its first and only rejection. A provider-first mutant
    # changes every complete objective trace even though reranking is unchanged.
    simultaneous = expected["catalog_validation_precedence_observation"]
    assert (
        simultaneous["first_and_only_rejection_reason"]
        == "catalog.evidence-mismatch"
    )
    for objective_oracle in simultaneous["objectives"]:
        selected, trace = ordered_rejection_trace(
            candidates,
            objective_oracle["objective"],
            {FAER_BICGSTAB_ID: "catalog.evidence-mismatch"},
        )
        _, provider_first_trace = ordered_rejection_trace(
            candidates,
            objective_oracle["objective"],
            {FAER_BICGSTAB_ID: "catalog.provider-mismatch"},
        )
        assert selected == objective_oracle["selected_candidate_id"]
        assert trace == objective_oracle["ordered_reasons"]
        assert provider_first_trace != trace

    # Each meaningful precedence reversal on this catalog changes the decision.
    assert (
        min(
            candidates,
            key=lambda c: (
                {"Fast": 0, "Reproducible": 1}[c["tuple"]["reduction"]],
                c["id"],
            ),
        )["id"]
        != EXPECTED_SELECTION["Robust"]
    )
    assert (
        min(
            candidates,
            key=lambda c: (
                {"Reproducible": 0, "Fast": 1}[c["tuple"]["reduction"]],
                c["id"],
            ),
        )["id"]
        != EXPECTED_SELECTION["Fast"]
    )
    fast_candidates = tuple(c for c in candidates if c["tuple"]["reduction"] == "Fast")
    assert (
        min(
            fast_candidates,
            key=lambda c: (
                {"BiConjugateGradientStabilized": 0, "SparseLu": 1}[
                    c["tuple"]["algorithm"]
                ],
                c["id"],
            ),
        )["id"]
        != EXPECTED_SELECTION["Fast"]
    )
    assert (
        min(
            candidates,
            key=lambda c: (
                {"SparseLu": 0, "BiConjugateGradientStabilized": 1}[
                    c["tuple"]["algorithm"]
                ],
                c["id"],
            ),
        )["id"]
        != EXPECTED_SELECTION["LowMemory"]
    )

    # Reordering Fast's two axes is observationally identical for this exact
    # catalog and every admitted subset, so it is deliberately not a falsifier.
    fast_axis_swapped_table = (
        (
            "algorithm",
            {"SparseLu": 0, "BiConjugateGradientStabilized": 1},
        ),
        ("reduction", {"Fast": 0, "Reproducible": 1}),
    )
    for admitted_count in range(1, len(candidates) + 1):
        for admitted in itertools.combinations(candidates, admitted_count):
            ordinary = min(admitted, key=lambda candidate: rank_key(candidate, "Fast"))[
                "id"
            ]
            swapped = min(
                admitted,
                key=lambda candidate: tuple(
                    order[candidate["tuple"][axis]]
                    for axis, order in fast_axis_swapped_table
                )
                + (candidate["id"],),
            )["id"]
            assert swapped == ordinary

    # Input enumeration cannot replace the final ID tie-break.
    reversed_bicgstab = [by_id[REFERENCE_IDENTITY_ID], by_id[FAER_IDENTITY_ID]]
    assert reversed_bicgstab[0]["id"] != EXPECTED_SELECTION["LowMemory"]

    assert solve_exact() == ([Fraction(1), Fraction(2)], [Fraction(0), Fraction(0)])
    csr = expected["fixture"]["canonical_csr"]
    diagonal = expected["fixture"]["diagonal_inventory"]
    assert [csr["column_indices"][entry["entry_index"]] for entry in diagonal] == [0, 1]
    assert [csr["values"][entry["entry_index"]] for entry in diagonal] == [4, 3]
    assert (
        expected["common_controls"]["relative_tolerance_f64_bits"]
        == "0x3d719799812dea11"
    )
    assert (
        expected["common_controls"]["absolute_tolerance_f64_bits"]
        == "0x3d06849b86a12b9b"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", action="store_true", dest="print_expected")
    arguments = parser.parse_args()

    expected = build_expected()
    expected["complex_profiles"] = complex_oracles()
    validate_mutants(expected)
    rendered = json.dumps(expected, indent=2, sort_keys=True) + "\n"
    if arguments.print_expected:
        sys.stdout.write(rendered)
        return 0
    try:
        committed = EXPECTED_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"missing frozen oracle: {EXPECTED_PATH}", file=sys.stderr)
        return 1
    if committed != rendered:
        print("policy-v10.json differs from the independent derivation", file=sys.stderr)
        return 1
    print("host-serial solver planning v10 oracle: exact derivation and mutants passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
