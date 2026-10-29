"""Run the README model listing so it cannot drift from the installed API."""

from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path


README = Path(__file__).resolve().parents[3] / "README.md"
SECTION = "## ✏️ What a model looks like"


def _listings() -> tuple[str, str]:
    text = README.read_text(encoding="utf-8")
    start = text.index(SECTION)
    end = text.index("\n## ", start + len(SECTION))
    section = text[start:end]
    model = re.search(r"```eqiora\n(.*?)```", section, re.DOTALL)
    program = re.search(r"```python\n(.*?)```", section, re.DOTALL)
    assert model is not None and program is not None
    return model.group(1), program.group(1)


def test_readme_model_runs_and_prints_the_documented_value(tmp_path: Path) -> None:
    model, program = _listings()
    assert 'model.field("x")' in program
    assert "absolute_tolerance=1e-11" in program
    (tmp_path / "decay.eqi").write_text(model, encoding="utf-8")
    (tmp_path / "run.py").write_text(program, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-I", "run.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    printed = float(result.stdout.strip().strip("[]"))
    assert abs(printed - math.exp(-1.0)) <= 2.0e-8
    documented = re.search(r"# \[([0-9.]+)\]", program)
    assert documented is not None
    assert result.stdout.strip() == f"[{documented.group(1)}]"
