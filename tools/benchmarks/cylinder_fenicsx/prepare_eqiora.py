"""Export the ordinary example mesh and compile its packaged model; never solve."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eqiora
from public_mesh import compile_model, export_mesh, geometry_and_mesh
from topology import mesh_arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    geometry, mesh = geometry_and_mesh(eqiora)
    data = export_mesh(geometry, mesh)
    mesh_arrays(data)
    model = compile_model(eqiora, geometry)
    data['model_digest'] = model.digest
    data['package_compilation_digest'] = model.package_compilation_digest
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    print(f'PREPARATION_PASS vertices={mesh.vertex_count} cells={mesh.cell_count}; no solve')


if __name__ == '__main__':
    main()
