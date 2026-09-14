from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fixture import SOURCE_SHA, checker, make_fixture


class ArtifactBoundTests(unittest.TestCase):
    def test_oversize_file_is_rejected_before_any_content_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact, identities = make_fixture(root)
            oversized = artifact / "oversized.bin"
            with oversized.open("wb") as target:
                target.truncate(checker.MAX_FILE_BYTES + 1)
            with mock.patch.object(
                checker,
                "sha256",
                side_effect=AssertionError("content read after raw-cap rejection"),
            ):
                errors = checker.check_artifact(
                    artifact, SOURCE_SHA, "0.1.0", identities
                )
            self.assertTrue(any("exceeds read cap" in error for error in errors))

    def test_count_and_total_caps_stop_before_hashing_the_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact, identities = make_fixture(root)
            for attribute, value, phrase in (
                ("MAX_FILES", 1, "exceeds 1 files"),
                ("MAX_TOTAL_BYTES", 1, "exceeds 1 bytes"),
            ):
                with (
                    self.subTest(bound=attribute),
                    mock.patch.object(checker, attribute, value),
                    mock.patch.object(
                        checker,
                        "sha256",
                        side_effect=AssertionError(
                            "hashed prefix after raw-cap rejection"
                        ),
                    ),
                ):
                    errors = checker.check_artifact(
                        artifact, SOURCE_SHA, "0.1.0", identities
                    )
                self.assertTrue(any(phrase in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
