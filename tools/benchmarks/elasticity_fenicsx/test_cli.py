"""Invalid launch controls must fail before creating a campaign directory."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CommandLineTests(unittest.TestCase):
    def test_invalid_inputs_fail_before_launch(self):
        scratch = Path.home()/'.cache/eqiora/elasticity-cli-tests'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            output = root/'uncreated-output'
            command = [sys.executable, '-I', str(Path(__file__).with_name('run.py')),
                       '--output', str(output), '--eqiora-python', sys.executable,
                       '--fenicsx-python', sys.executable, '--eqiora-wheel', __file__,
                       '--eqiora-revision', 'a'*40, '--eqiora-build-profile', 'release',
                       '--mode', 'correctness-only', '--cpu', str(min(os.sched_getaffinity(0)))]
            for flag, value in (('--cpu', '-1'), ('--eqiora-revision', 'not-a-source-id'),
                                ('--eqiora-python', str(root/'missing-python'))):
                corrupt = command.copy(); corrupt[corrupt.index(flag)+1] = value
                completed = subprocess.run(corrupt, capture_output=True, text=True)
                self.assertEqual(completed.returncode, 2, completed.stderr)
                self.assertFalse(json.loads(completed.stdout)['complete_comparison'])
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
