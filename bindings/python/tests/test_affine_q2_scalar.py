"""Ordinary installed Q2 fields against the independent full-vector reference."""
from pathlib import Path
import subprocess
import sys


def test_affine_q2_scalar_full_fields_and_finite_level_errors():
    root = Path(__file__).resolve().parents[3]
    gate = root / "verify/numerics/affine-q2-scalar/registered.py"
    # Fresh interpreter prevents generic verification module names from shadowing
    # pytest or another case. The child owns serial controls and retains failures.
    completed = subprocess.run([sys.executable, "-I", str(gate)], cwd=root, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
