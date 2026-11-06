"""Exercise fresh-child orchestration without importing either numerical library."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from shared_campaign import ComparisonCase, collect_reports
from shared_measurement import ComparisonError


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.args = SimpleNamespace(output=self.root / 'campaign', cpu=3,
                                    eqiora_python=self.root / 'eqiora-python',
                                    fenicsx_python=self.root / 'fenicsx-python',
                                    eqiora_wheel=self.root / 'eqiora.whl',
                                    eqiora_revision='a' * 40, eqiora_build_profile='release',
                                    mode='benchmark')
        self.commands = []

    def worker(self, command, **kwargs):
        self.commands.append((command, kwargs))
        output = Path(command[command.index('--output') + 1])
        output.write_text(json.dumps({'case': output.parent.parent.name,
                                      'tool': output.parent.name}))
        return SimpleNamespace(returncode=0)

    def test_problem_arguments_order_environment_and_fresh_outputs(self):
        cases = [ComparisonCase('n4', ('--dimension', '3', '--cells', '4')),
                 ComparisonCase('curved-fine', ('--reference', '/data/frozen.json',
                                                '--reference-sha256', 'b' * 64))]
        with patch('shared_campaign.os.sched_setaffinity') as affinity, \
             patch('shared_campaign.subprocess.run', side_effect=self.worker), \
             patch.dict(os.environ, {'PYTHONPATH': '/foreign', 'LD_LIBRARY_PATH': '/foreign-mpi'}):
            reports = collect_reports(self.args, cases, problem_root=self.root)
        affinity.assert_called_once_with(0, {3})
        self.assertEqual([(r['case'], r['tool']) for r in reports],
                         [('n4', 'eqiora'), ('n4', 'fenicsx'),
                          ('curved-fine', 'fenicsx'), ('curved-fine', 'eqiora')])
        for index, (command, kwargs) in enumerate(self.commands):
            case = cases[index // 2]
            self.assertEqual(command[1], '-I')
            self.assertEqual(command[3:3 + len(case.worker_arguments)], list(case.worker_arguments))
            self.assertNotIn('PYTHONPATH', kwargs['env'])
            self.assertEqual(kwargs['env']['OMP_NUM_THREADS'], '1')
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            if 'fenicsx_case.py' in command[2]:
                self.assertNotIn('LD_LIBRARY_PATH', kwargs['env'])
                self.assertNotIn('--wheel', command)
            else:
                self.assertEqual(command[-6:], ['--wheel', str(self.args.eqiora_wheel),
                                                '--revision', 'a' * 40, '--build-profile', 'release'])
        self.assertEqual(len(list(self.args.output.glob('*/*/worker.log'))), 4)

    def test_failed_worker_stops_and_retains_diagnostics(self):
        def fail(command, **kwargs):
            kwargs['stdout'].write('external solver failure\n')
            return SimpleNamespace(returncode=17)
        with patch('shared_campaign.os.sched_setaffinity'), \
             patch('shared_campaign.subprocess.run', side_effect=fail) as run:
            result = collect_reports(self.args, [ComparisonCase('coarse', ('--mesh', 'frozen'))],
                                     problem_root=self.root)
        self.assertIsNone(result)
        self.assertEqual(run.call_count, 1)
        failure = json.loads((self.args.output / 'failure.json').read_text())
        self.assertEqual((failure['case'], failure['backend'], failure['exit_code']), ('coarse', 'eqiora', 17))
        self.assertFalse(failure['complete_comparison'])
        self.assertEqual(Path(failure['log']).read_text(), 'external solver failure\n')

    def test_invalid_cases_fail_before_output_or_affinity_change(self):
        invalid = [[], [ComparisonCase('../escape', ())], [ComparisonCase('/absolute', ())],
                   [ComparisonCase('same', ()), ComparisonCase('same', ())],
                   [ComparisonCase('valid', ('--output=elsewhere',))],
                   [ComparisonCase('valid', ('--mode', 'benchmark'))],
                   [ComparisonCase('valid', ('--',))]]
        for cases in invalid:
            with self.subTest(cases=cases), patch('shared_campaign.os.sched_setaffinity') as affinity:
                with self.assertRaises(ComparisonError):
                    collect_reports(self.args, cases, problem_root=self.root)
                affinity.assert_not_called()
                self.assertFalse(self.args.output.exists())


if __name__ == '__main__':
    unittest.main()
