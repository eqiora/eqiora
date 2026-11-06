"""Prepare the independent MINI reference on the public exported mesh; never solve."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fenicsx_adapter import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, required=True)
    args = parser.parse_args()
    prepared = prepare(json.loads(args.mesh.read_text()))
    print(f'PREPARATION_PASS vertices={len(prepared.coordinates)} cells={len(prepared.cells)} '
          f'full_unknowns={3*len(prepared.coordinates)+2*len(prepared.cells)}; no solve')


if __name__ == '__main__':
    main()
