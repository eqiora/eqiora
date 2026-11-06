"""Refreshing an artifact digest cannot hide broken saved-field associations."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import recheck_saved as saved


class SavedEvidence(unittest.TestCase):
    def test_changed_result_or_lineage_rejects_after_digest_refresh(self):
        parent = Path.home()/'.cache/eqiora/hex-saved-tests'
        parent.mkdir(parents=True,exist_ok=True)
        for control in ('changed-result-value','wrong-result-identity','wrong-component-map'):
            with self.subTest(control=control), tempfile.TemporaryDirectory(dir=parent) as temporary:
                root = Path(temporary)/'evidence'
                shutil.copytree(saved.HERE/'evidence',root)
                inventory = saved.read(root/'inventory.json')
                relative = 'python-7612/result.json' if control == 'changed-result-value' else 'python-7612/primary.json'
                path = root/relative
                value = saved.read(path)
                if control == 'changed-result-value':
                    value['content']['payload']['fields'][0]['blocks'][0]['values'][30] += .01
                elif control == 'wrong-result-identity':
                    value['lineage']['result'] = value['lineage']['model']
                else:
                    # This reordered projection retains all 81 records and values.
                    # It must remain bound to physical coordinates, not array position.
                    value['coefficients'][30]['coordinate_m'],value['coefficients'][31]['coordinate_m'] = value['coefficients'][31]['coordinate_m'],value['coefficients'][30]['coordinate_m']
                    value['coefficients'][30]['component'],value['coefficients'][31]['component'] = value['coefficients'][31]['component'],value['coefficients'][30]['component']
                path.write_text(json.dumps(value))
                inventory['files_sha256'][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
                (root/'inventory.json').write_text(json.dumps(inventory))
                with self.assertRaises(ValueError):
                    saved.main(root)


if __name__ == '__main__':
    unittest.main()
