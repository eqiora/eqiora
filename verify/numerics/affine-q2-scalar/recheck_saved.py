#!/usr/bin/env python3
"""Integrity and independent-mathematics replay of committed original Q2 fields.

No Eqiora import, PDE solve, native artifact reopen or process-receipt replay.
"""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "reference"))
import compare
import polynomial


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def checked_file(root, relative, expected):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "inventory path escapes its owner")
    require(digest(path) == expected, f"changed retained file: {relative}")



def checked_projection(root, relative, projection):
    require(projection["omitted_fields"] == ["native_module", "python"], "unexpected receipt projection scope")
    checked_file(root, relative, projection["public_sha256"])
    public = read(root / relative)
    require(set(public) == set(projection["original_field_names"]) - set(projection["omitted_fields"]), "projection changed fields outside its omission scope")
    require(set(public) == set(projection["retained_field_sha256"]), "projection retained-field inventory changed")
    for key, value in public.items():
        actual = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        require(actual == projection["retained_field_sha256"][key], f"projection changed retained field: {key}")


def main():
    patch = HERE / "evidence/polynomial-5f8130"
    for relative, expected in read(patch / "inventory.json")["files_sha256"].items():
        checked_file(patch, relative, expected)
    for dimension in (2, 3):
        result = polynomial.compare(read(patch / f"d{dimension}/polynomial-input.json"))
        require(result["accepted"], f"polynomial {dimension}D rejected")

    root = HERE / "evidence/sine-5f8130"
    inventory = read(root / "inventory.json")
    for relative, expected in inventory["original_files_sha256"].items():
        checked_file(root, relative, expected)
    for relative, expected in inventory["reference_files_sha256"].items():
        checked_file(HERE, relative, expected)
    for relative, projection in inventory["public_receipt_projections"].items():
        checked_projection(root, relative, projection)
    for dimension in (2, 3):
        candidates = []
        for n in (2, 4, 8):
            directory = root / f"d{dimension}-n{n}"
            candidate = read(directory / "candidate.json")
            identity = read(directory / "export-identity.json")
            for name in ("plan", "result", "mesh", "candidate"):
                require(identity[f"{name}_sha256"] == digest(directory / f"{name}.json"), f"{name} export binding changed")
            result = read(directory / "result.json")
            require(result["content"]["plan_identity"] == identity["plan_identity"], "Result/Plan binding changed")
            fields = result["content"]["payload"]["fields"]
            require(len(fields) == 1, "retained Result is not the single-field witness")
            values = [float(value).hex() for block in fields[0]["blocks"] for value in block["values"]]
            require(values == [record["value_binary64_hex"] for record in candidate["records"]], "candidate differs from original complete Result field")
            require([record["coordinates_m"] for record in identity["coefficient_entities"]] == [record["coordinates_m"] for record in candidate["records"]], "export coordinates changed")
            require(compare.compare(candidate)["accepted"], f"sine {dimension}D N={n} rejected")
            candidates.append(candidate)
        require(compare.compare_levels(candidates)["accepted"], f"{dimension}D finite-level set rejected")
    print("saved-q2: integrity, original full-field bindings, two polynomial and six sine gates, both finite-level sets PASS; zero PDE solves")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(f"saved-q2: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(2)
