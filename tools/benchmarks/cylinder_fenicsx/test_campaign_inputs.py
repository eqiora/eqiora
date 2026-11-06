"""Declared case descriptors and launch failures cannot silently change the workload."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from run import FrozenCase, load_cases
from topology import ComparisonError


class CampaignInputTests(unittest.TestCase):
    def test_case_arguments_are_problem_specific_and_complete(self):
        case = FrozenCase('coarse', Path('/reference/coarse.json'), 'a'*64, None)
        self.assertEqual(case.worker_arguments(), ('--case-label', 'coarse', '--reference-manifest',
                         '/reference/coarse.json', '--reference-sha256', 'a'*64))
        self.assertNotIn('--cells', case.worker_arguments())
        self.assertNotIn('--mode', case.worker_arguments())

    def test_undeclared_levels_fail_before_reference_or_worker_access(self):
        scratch = Path.home()/'.cache/eqiora/cylinder-campaign-input-tests'; scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path = Path(directory)/'references.json'
            for cases in ({}, {'standard': {}}, {'coarse': {}, 'standard': {}, 'fine': {}, 'best': {}}):
                path.write_text(json.dumps({'cases': cases}))
                with self.assertRaises(ComparisonError): load_cases(path)

    def test_bad_cpu_has_typed_failure_and_no_output(self):
        scratch = Path.home()/'.cache/eqiora/cylinder-campaign-input-tests'; scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory); output = root/'not-created'
            command = [sys.executable, '-I', str(Path(__file__).with_name('run.py')), '--references', __file__,
                       '--output', str(output), '--eqiora-python', sys.executable, '--fenicsx-python', sys.executable,
                       '--eqiora-wheel', __file__, '--eqiora-revision', 'a'*40, '--eqiora-build-profile', 'release',
                       '--mode', 'benchmark', '--cpu', '-1']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(json.loads(result.stdout)['complete_comparison'])
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
