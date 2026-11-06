"""Failed, dirty or differently built correctness cannot authorize timing."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from qualification import check_process, verified_descriptor, validate_qualification, check_report_reference, check_cold_pair
from topology import ComparisonError


class QualificationTests(unittest.TestCase):
    def test_real_terminal_clean_source_and_same_producer_are_required(self):
        artifact = {'sha256': 'a'*64, 'source_revision': 'b'*40, 'build_profile': 'release'}
        receipt = {'case': 'coarse', 'backend': 'eqiora', 'terminal_exit_code': 0,
                   'harness_dirty': False, 'actual_harness_revision': 'c'*40,
                   'finished_at': '2026-11-06T00:00:00+00:00', 'artifact': dict(artifact)}
        check_process(receipt, label='coarse', backend='eqiora', expected_artifact=artifact)
        for key, value in (('terminal_exit_code', None), ('terminal_exit_code', 1),
                           ('harness_dirty', True), ('case', 'fine'), ('artifact', None), ('finished_at', None)):
            bad = copy.deepcopy(receipt); bad[key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaises(ComparisonError):
                    check_process(bad, label='coarse', backend='eqiora', expected_artifact=artifact)
        bad = copy.deepcopy(receipt); bad['artifact']['sha256'] = 'd'*64
        with self.assertRaises(ComparisonError):
            check_process(bad, label='coarse', backend='eqiora', expected_artifact=artifact)

    def test_reference_binding_is_required_for_both_backends(self):
        for backend in ('eqiora', 'fenicsx'):
            report = {'tool': backend, 'environment': {'frozen_reference_manifest_sha256': 'a'*64}}
            check_report_reference(report, 'a'*64)
            with self.assertRaises(ComparisonError): check_report_reference(report, 'b'*64)
            del report['environment']['frozen_reference_manifest_sha256']
            with self.assertRaises(ComparisonError): check_report_reference(report, 'a'*64)

    def test_retained_candidate_must_be_the_recorded_cold_arrays(self):
        report = {'coordinates': [[0., 1.]], 'values': [[2., 3.]],
                  'associated_fields': {'bubble': {'association': 'cell-bubble', 'values': [[4., 5.]]}}}
        check_cold_pair(report, report['coordinates'], report['values'], report['associated_fields'])
        for name in ('coordinates', 'values', 'associated_fields'):
            bad = copy.deepcopy(report)
            if name == 'associated_fields': bad[name]['bubble']['values'][0][0] += 1
            else: bad[name][0][0] += 1
            with self.subTest(name=name):
                with self.assertRaises(ComparisonError):
                    check_cold_pair(bad, report['coordinates'], report['values'], report['associated_fields'])

    def test_bad_hash_or_incomplete_case_qualification_is_rejected(self):
        scratch = Path.home()/'.cache/eqiora/cylinder-qualification-tests'; scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory); path = root/'qualification.json'; path.write_text(json.dumps({'cases': {}}))
            with self.assertRaises(ComparisonError): validate_qualification(path, [], {})
            with self.assertRaises(ComparisonError): verified_descriptor(root, {'path': path.name, 'sha256': '0'*64})


if __name__ == '__main__':
    unittest.main()
