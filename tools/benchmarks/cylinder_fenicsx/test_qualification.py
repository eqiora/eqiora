"""Failed, dirty or differently built correctness cannot authorize timing."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from qualification import check_process, verified_descriptor, validate_qualification
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

    def test_bad_hash_or_incomplete_case_qualification_is_rejected(self):
        scratch = Path.home()/'.cache/eqiora/cylinder-qualification-tests'; scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory); path = root/'qualification.json'; path.write_text(json.dumps({'cases': {}}))
            with self.assertRaises(ComparisonError): validate_qualification(path, [], {})
            with self.assertRaises(ComparisonError): verified_descriptor(root, {'path': path.name, 'sha256': '0'*64})


if __name__ == '__main__':
    unittest.main()
