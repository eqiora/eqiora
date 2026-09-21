# Eqiora notebook editor

This prebuilt Jupyter extension highlights `%%eqiora` source cells using the
[canonical TextMate grammar](../eqiora/syntaxes/eqiora.tmLanguage.json). A small
CodeMirror adapter maps TextMate presentation scopes to editor styles. It does
not parse semantics, maintain another vocabulary, or execute source.

The header selects Eqiora for that cell; removing the header restores the host
language. Selection follows the current source, including before execution and
when reopening a saved notebook. Ordinary Python cells retain their language.
Types and units use the host's builtin color. No notebook metadata is changed.

The Python kernel independently needs `%load_ext eqiora.jupyter` from the same
development revision. In a split server/kernel installation, install the frontend
assets in the **Jupyter server environment** and Eqiora in the **kernel environment**.
This development feature is absent from the published Eqiora 0.1.2 package.
Colab and VS Code notebook editors do not load this Jupyter plugin; their
highlighting is not claimed. Completion and hover are not implemented here.

## Build and inspect

From this directory, with the repository's Node/npm toolchain:

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python jupyterlab==4.5.11 notebook==7.5.6
export PATH="$PWD/.venv/bin:$PATH"
npm ci --ignore-scripts
npm run typecheck
npm run build
```

The build copies the canonical grammar into an ignored temporary source file.
It writes a single prebuilt output under
`wheel-data/data/share/jupyter/labextensions/@eqiora/jupyter/`, including dependency
licenses. These generated files are checked in for Python packaging: ordinary
package installation must not invoke Node or fetch frontend dependencies.
Rebuild and commit this output after changes to the adapter, grammar or build inputs.
Root Python packaging owns installing the files into `sys.prefix/share/jupyter`.

For frontend development without rebuilding the Python wheel, copy the prebuilt
output into this isolated environment:

```bash
python - <<'PY'
import shutil
import sys
from pathlib import Path
source = Path('wheel-data/data/share/jupyter/labextensions/@eqiora/jupyter')
target = Path(sys.prefix) / 'share/jupyter/labextensions/@eqiora/jupyter'
shutil.copytree(source, target, dirs_exist_ok=True)
PY
jupyter labextension list
python example.py
jupyter lab --no-browser --ServerApp.ip=127.0.0.1 --ServerApp.port=18927 --ServerApp.port_retries=0 --IdentityProvider.token=eqiora-test --ServerApp.root_dir="$PWD"
```

Open `decay.ipynb` under `/lab/tree/decay.ipynb` or `/notebooks/decay.ipynb`.
`example.py` generates its source cell directly from `examples/decay.eqi`; it
contains no second equation definition. The final Python cell uses the compiled
Model. Execution requires a development Eqiora package in that kernel; editor
highlighting is available before any cell runs.

## Focused checks

```bash
python -m unittest discover -s tests -p 'test_*.py'
npm run typecheck
npx playwright install chromium
npm test
```

With the server above running, use another terminal:

```bash
EQIORA_JUPYTER_URL=http://127.0.0.1:18927 EQIORA_JUPYTER_TOKEN=eqiora-test npm run test:host
```

Tests use temporary notebooks and remove them afterward. Browser tests cover
canonical keywords/types/units/numbers/comments, initial unexecuted cells,
header edits, restoration of Python, saved reopening and unchanged execution
counts. Verified on Linux x86_64, Chromium 151.0.7922.34 (Playwright 1.62.1),
JupyterLab 4.5.11 and Notebook 7.5.6. Host checks used locally built prebuilt assets;
installed-wheel delivery must also be checked by the Python package gate.
The notebook execution contract is covered separately by
`bindings/python/tests/test_jupyter.py`; frontend tests do not establish Colab
Restart and Run All behavior.
