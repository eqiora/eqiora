#!/usr/bin/env python3
"""Prepare the ordinary Hex case; --execute requires a reviewed producer receipt."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from production import HERE, extract, prepare, require, sha256, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--build-receipt', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    require(not args.output.exists(), 'output must not exist; preserve previous attempts')
    args.output.mkdir(parents=True)
    state = {'ordinary_run_attempted': 0, 'ordinary_run_completed': 0}
    try:
        import eqiora
        import eqiora._eqiora as native
        receipt_raw = args.build_receipt.read_bytes()
        receipt = json.loads(receipt_raw)
        extension = Path(native.__file__).resolve()
        extension_hash = sha256(extension.read_bytes())
        require(receipt['extension_sha256'] == extension_hash, 'loaded extension differs from reviewed build receipt')
        require(receipt['source_commit'] and receipt['wheel_sha256'] and receipt['binary_sha256'], 'incomplete build receipt')
        source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE).decode().strip()
        source_tree = subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=HERE).decode().strip()
        write(args.output, 'launch.json', {'started_at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(), 'argv': sys.argv,
            'python': sys.executable, 'python_version': sys.version, 'eqiora_version': importlib.metadata.version('eqiora'),
            'source_commit': source_commit, 'source_tree': source_tree, 'runtime_build': receipt, 'build_receipt_sha256': sha256(receipt_raw),
            'extension': str(extension), 'extension_sha256': extension_hash, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
            'environment': {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'LD_LIBRARY_PATH')},
            'files': {str(path.relative_to(HERE)): sha256(path.read_bytes()) for path in (HERE/'model.eqi', HERE/'production.py', HERE/'check.py', HERE/'reference/reference.py', HERE/'reference/protocol.py')}})
        model, plan, field, observables = prepare(eqiora, args.output)
        write(args.output, 'preparation.json', {'status': 'prepared-no-numerical-candidate', 'plan_identity': plan.identity, 'model_digest': model.digest,
            'mesh_digest': plan.mesh.digest, 'field': field.id, 'observables': [v.id for v in observables]})
        if not args.execute:
            return 0
        require(receipt.get('producer_review') == 'accepted', 'producer receipt has not passed explicit review')
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA_VISIBLE_DEVICES must be empty')
        require(len(os.sched_getaffinity(0)) == 1, 'execution requires one pinned CPU')
        require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS')), 'execution requires one thread')
        state['ordinary_run_attempted'] = 1
        result = eqiora.run(plan)
        state['ordinary_run_completed'] = 1
        # Preserve the actual Result immediately, before observation/assertions.
        result_bytes = result.to_bytes()
        write(args.output, 'result.json', result_bytes)
        replay = eqiora.Result.from_bytes(plan, result_bytes)
        require(replay.to_bytes() == result_bytes, 'canonical Result replay changed')
        sys.path.insert(0, str(HERE/'reference'))
        import protocol
        limits = protocol.r.budgets()
        for name, lane in (('primary', result), ('replay', replay)):
            document = extract(plan, model, lane, field, observables)
            write(args.output, name+'.json', document)
            protocol.check(json.dumps(document), limits)
        write(args.output, 'qualification.json', {**state, 'status': 'passed', 'pde_solves': 1, 'profile_calls': 0, 'warmups': 0})
        return 0
    except Exception as error:
        write(args.output, 'failure.json', {**state, 'type': type(error).__name__, 'message': str(error)})
        print(f'cartesian-hex-elasticity: {type(error).__name__}: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
