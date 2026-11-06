"""Public projections cannot silently remove or alter scientific observations."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent/'multimesh'))
from recheck import check_inventory_entry
from shared_measurement import ComparisonError


class PublicProjectionTests(unittest.TestCase):
    def test_only_enumerated_metadata_projection_is_accepted(self):
        name = 'benchmark/fine/fenicsx/result.json'
        entry = {'original_sha256': 'a'*64, 'public_sha256': 'b'*64,
                 'scope': 'explicit public metadata projection', 'omitted_fields': ['environment.mpi_library']}
        check_inventory_entry(name, entry, 'b'*64)
        for fields in ([], ['values'], ['environment.mpi_library', 'accuracy'], ['environment.PATH']):
            bad = copy.deepcopy(entry); bad['omitted_fields'] = fields
            with self.subTest(fields=fields):
                with self.assertRaises(ComparisonError): check_inventory_entry(name, bad, 'b'*64)
        with self.assertRaises(ComparisonError): check_inventory_entry(name, entry, 'c'*64)
        with self.assertRaises(ComparisonError): check_inventory_entry('references/fine/reference-launch.json', entry, 'b'*64)

    def test_original_identity_and_outer_control_projection_are_distinct(self):
        original = {'original_sha256': 'a'*64, 'public_sha256': 'a'*64,
                    'scope': 'original bytes', 'omitted_fields': []}
        check_inventory_entry('references/fine/reference-launch.json', original, 'a'*64)
        bad = copy.deepcopy(original); bad['original_sha256'] = 'b'*64
        with self.assertRaises(ComparisonError): check_inventory_entry('references/fine/reference-launch.json', bad, 'a'*64)
        projected = {'original_sha256': 'a'*64, 'public_sha256': 'b'*64,
                     'scope': 'explicit public metadata projection',
                     'omitted_fields': [f'entries.{i}.environment_controls.PATH' for i in range(3)]}
        check_inventory_entry('reference-controls/launch-plan.json', projected, 'b'*64)
        projected['omitted_fields'].append('entries.0.command')
        with self.assertRaises(ComparisonError): check_inventory_entry('reference-controls/launch-plan.json', projected, 'b'*64)


if __name__ == '__main__':
    unittest.main()
