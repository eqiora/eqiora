from __future__ import annotations

import sys
import unittest
from collections.abc import Callable
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools/release"))

from python_candidate import (  # noqa: E402
    CandidateError,
    SourceIdentity,
    require_annotated_expected_tag,
)


class AnnotatedReleaseTagTests(unittest.TestCase):
    commit = "a" * 40
    tag = "v0.1.0"

    def source(self) -> SourceIdentity:
        return SourceIdentity(commit=self.commit, tags=(self.tag,))

    def git_query(self, object_type: str, peeled: str) -> Callable[..., str]:
        def query(argv: list[str], *, capture: bool = False) -> str:
            self.assertTrue(capture)
            if argv[:3] == ["git", "cat-file", "-t"]:
                return object_type
            if argv[:2] == ["git", "rev-parse"]:
                return peeled
            self.fail(f"unexpected git query: {argv}")

        return query

    def test_accepts_only_an_annotated_tag_peeled_to_the_candidate(self) -> None:
        require_annotated_expected_tag(
            self.source(),
            self.tag,
            git_query=self.git_query("tag", self.commit),
        )

    def test_rejects_lightweight_and_misdirected_tags(self) -> None:
        with self.assertRaisesRegex(CandidateError, "annotated tag"):
            require_annotated_expected_tag(
                self.source(),
                self.tag,
                git_query=self.git_query("commit", self.commit),
            )
        with self.assertRaisesRegex(CandidateError, "does not peel"):
            require_annotated_expected_tag(
                self.source(),
                self.tag,
                git_query=self.git_query("tag", "b" * 40),
            )

    def test_rejects_a_different_tag_before_querying_git(self) -> None:
        def unexpected_query(*_arguments: object, **_keywords: object) -> str:
            self.fail("git must not be queried for an absent tag")

        with self.assertRaisesRegex(CandidateError, "requires exact tag"):
            require_annotated_expected_tag(
                SourceIdentity(commit=self.commit, tags=("v0.1.1",)),
                self.tag,
                git_query=unexpected_query,
            )


if __name__ == "__main__":
    unittest.main()
