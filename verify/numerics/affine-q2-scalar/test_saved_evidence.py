"""Projection tampering must not gain authority by updating its file digest."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import recheck_saved as saved


class SavedProjection(unittest.TestCase):
    def test_only_declared_machine_local_fields_can_be_omitted(self):
        root = saved.HERE / "evidence/sine-5f8130"
        inventory = saved.read(root / "inventory.json")
        relative, original = next(iter(inventory["public_receipt_projections"].items()))
        saved.checked_projection(root, relative, original)
        parent = Path.home() / ".cache/eqiora/saved-evidence-tests"
        parent.mkdir(parents=True, exist_ok=True)
        public = saved.read(root / relative)
        for edit in ("changed-value", "extra-field", "extra-omission"):
            with self.subTest(edit=edit), tempfile.TemporaryDirectory(dir=parent) as temporary:
                directory = Path(temporary)
                projection = copy.deepcopy(original)
                changed = copy.deepcopy(public)
                if edit == "changed-value":
                    changed["pde_solves"] = 100
                elif edit == "extra-field":
                    changed["invented"] = True
                else:
                    changed.pop("pde_solves")
                path = directory / "public.json"
                path.write_text(json.dumps(changed))
                projection["public_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaises(ValueError):
                    saved.checked_projection(directory, "public.json", projection)


if __name__ == "__main__":
    unittest.main()
