"""The lesson runs entirely from its cells with the installed public package."""
import json
from pathlib import Path


def test_stochastic_processes_notebook_is_self_contained(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[3] / "examples/python/stochastic_processes.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace = {}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), f"{path.name}:cell-{index}", "exec"), namespace)
    assert len(namespace["members"]) == namespace["sample_count"]
    assert set(namespace["statistics"]) == {"OU", "GBM"}
    if "plt" in namespace:
        namespace["plt"].close("all")
