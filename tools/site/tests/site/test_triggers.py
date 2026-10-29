from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from fixture import (
    REPOSITORY,
    _workflow,
    checker,
    git_object_authority,
    git_read_environment,
)


BASIS_SHA = "68c9c2fe245ac52cc20dcf5a65a2455de507f0dc"
ARCHIVE_ERROR = "Pages archive must bind the tracked link after extraction"
BROWSER_ERROR = "Pages browser identity must precede execution and propagation"
ORDER_ERROR = "Pages archive/browser supply checks are out of causal order"
DEREFERENCE_ERROR = "Pages workflow uses forbidden supply substitution '--dereference'"
observe = checker.check_workflow_text


def omitted(token: str) -> str:
    return f"Pages workflow omits offline/supply boundary {token!r}"


def archive_case(root: Path, repo: Path, revision: str, mutation: str = ":") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    source, sentinel = root / "source", root / "source-used"
    script = r"""set -euo pipefail
source_links="$(git -C "$GIT_REPOSITORY" ls-tree -r "$REVISION" | awk '$1 == "120000" { print $4 }')"
case "$source_links" in ''|'CLAUDE.md') ;; *) exit 1 ;; esac
if test -n "$source_links"; then
  test "$(git -C "$GIT_REPOSITORY" cat-file blob "$REVISION:CLAUDE.md" | sha256sum | cut -d ' ' -f 1)" = "a54ff182c7e8acf56acfd6e4b9c3ff41e2c41a31c9b211b2deb9df75d9a478f9"
  git -C "$GIT_REPOSITORY" ls-tree "$REVISION" -- AGENTS.md | grep -F '100644 blob'
  git -C "$GIT_REPOSITORY" cat-file blob "$REVISION:AGENTS.md" > "$EXPECTED"
fi
mkdir -p "$SOURCE"
git -C "$GIT_REPOSITORY" archive --format=tar "$REVISION" | tar -xf - -C "$SOURCE"
eval "$MUTATION"
if test -n "$source_links"; then
  test -L "$SOURCE/CLAUDE.md"
  test "$(readlink -n "$SOURCE/CLAUDE.md" | sha256sum | cut -d ' ' -f 1)" = "a54ff182c7e8acf56acfd6e4b9c3ff41e2c41a31c9b211b2deb9df75d9a478f9"
  cmp "$SOURCE/AGENTS.md" "$EXPECTED"
elif test -e "$SOURCE/CLAUDE.md" || test -L "$SOURCE/CLAUDE.md"; then
  exit 1
fi
printf source-used > "$SENTINEL"
"""
    environment = {
        **git_read_environment(),
        "GIT_REPOSITORY": str(repo),
        "REVISION": revision,
        "SOURCE": str(source),
        "EXPECTED": str(root / "expected-AGENTS.md"),
        "SENTINEL": str(sentinel),
        "MUTATION": mutation,
    }
    subprocess.run(
        ["bash", "-c", script],
        cwd=root,
        env=environment,
        capture_output=True,
        check=False,
    )
    return sentinel


class TriggerContractTests(unittest.TestCase):
    def test_current_workflow_is_green(self) -> None:
        text = (REPOSITORY / ".github/workflows/pages.yml").read_text(encoding="utf-8")
        self.assertEqual(observe(text), [])

    def test_required_job_and_fail_closed_selection_mutants(self) -> None:
        ordinary = _workflow()
        filtered = ordinary.replace(
            "  pull_request:\n    types:",
            "  pull_request:\n    paths: [docs/**]\n    types:",
            1,
        )
        self.assertIn(
            "Pages must run one unfiltered required job for every pull request",
            observe(filtered),
        )
        stale_base_retarget = ordinary.replace(", edited]", "]", 1)
        self.assertIn(
            "Pages must run one unfiltered required job for every pull request",
            observe(stale_base_retarget),
        )
        self.assertIn(
            "Pages omits the repository-owned input-closure classifier",
            observe(ordinary.replace("tools/ci/classify_changes.py", "removed", 1)),
        )
        commented_classifier = ordinary.replace(
            "tools/ci/classify_changes.py", "removed", 1
        ) + "\n# tools/ci/classify_changes.py\n"
        self.assertIn(
            "Pages omits the repository-owned input-closure classifier",
            observe(commented_classifier),
        )
        unrelated_classifier = ordinary.replace(
            "tools/ci/classify_changes.py", "removed", 1
        ).replace(
            "      - name: Bind the classified source authority\n",
            "      - name: Unrelated token holder\n"
            "        run: tools/ci/classify_changes.py\n"
            "      - name: Bind the classified source authority\n",
            1,
        )
        self.assertIn(
            "Pages omits the repository-owned input-closure classifier",
            observe(unrelated_classifier),
        )
        classifier_run = (
            "        run: |\n"
            "          exec python3 tools/ci/classify_changes.py \\\n"
            "            --event \"$SITE_EVENT_NAME\" \\\n"
            "            --base \"$SITE_BASE_SHA\" \\\n"
            "            --head \"$SITE_HEAD_SHA\" \\\n"
            "            --github-output \"$GITHUB_OUTPUT\"\n"
        )
        forged = ordinary.replace(
            classifier_run,
            "        run: |\n"
            "          if false; then\n"
            "            exec python3 tools/ci/classify_changes.py \\\n"
            "              --event \"$SITE_EVENT_NAME\" \\\n"
            "              --base \"$SITE_BASE_SHA\" \\\n"
            "              --head \"$SITE_HEAD_SHA\"\n"
            "          fi\n"
            "          printf '%s\\n' \\\n"
            "            'site=false' \\\n"
            "            'site_source_sha=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' \\\n"
            "            'site_reason=unchanged input closure' >> \"$GITHUB_OUTPUT\"\n",
            1,
        )
        self.assertNotEqual(forged, ordinary)
        self.assertIn(
            "Pages omits the repository-owned input-closure classifier",
            observe(forged),
        )
        self.assertIn(
            "Pages full build step is not fail-closed: Configure GitHub Pages",
            observe(
                ordinary.replace(
                    "      - name: Configure GitHub Pages\n"
                    "        if: steps.site_closure.outputs.site == 'true'\n",
                    "      - name: Configure GitHub Pages\n",
                    1,
                )
            ),
        )
        self.assertIn(
            "Pages deployment is not bound to an authenticated main full build",
            observe(
                ordinary.replace(
                    "needs.build.outputs.full_build == 'true'",
                    "needs.build.result == 'success'",
                    1,
                )
            ),
        )
        commented_deploy = ordinary.replace(
            "      needs.build.outputs.full_build == 'true'",
            "      needs.build.result == 'success'",
            1,
        ) + "\n# needs.build.outputs.full_build == 'true'\n"
        self.assertIn(
            "Pages deployment is not bound to an authenticated main full build",
            observe(commented_deploy),
        )
        unrelated_deploy = ordinary.replace(
            "      needs.build.outputs.full_build == 'true'",
            "      needs.build.result == 'success'",
            1,
        ).replace(
            "      - name: Deploy GitHub Pages artifact\n",
            "      - name: Unrelated deploy token holder\n"
            "        run: echo \"needs.build.outputs.full_build == 'true'\"\n"
            "      - name: Deploy GitHub Pages artifact\n",
            1,
        )
        self.assertIn(
            "Pages deployment is not bound to an authenticated main full build",
            observe(unrelated_deploy),
        )
        duplicate_deploy_if = ordinary.replace(
            "    needs: build\n",
            "    if: always()\n    needs: build\n",
            1,
        )
        self.assertIn(
            "Pages deployment is not bound to an authenticated main full build",
            observe(duplicate_deploy_if),
        )

    def test_real_linked_and_genuine_no_link_archives_reach_source_use(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.home() / ".cache/eqiora") as value:
            root = Path(value)
            sentinel = archive_case(root, git_object_authority().root, BASIS_SHA)
            self.assertEqual(sentinel.read_bytes(), b"source-used")
            repository = root / "no-link"
            subprocess.run(["git", "init", "-q", str(repository)], check=True)
            git = ["git", "-C", str(repository)]
            (repository / "AGENTS.md").write_text("no link\n", encoding="utf-8")
            subprocess.run([*git, "add", "AGENTS.md"], check=True)
            git += ["-c", "user.name=oracle", "-c", "user.email=oracle@example.invalid"]
            subprocess.run([*git, "commit", "-qm", "no link"], check=True)
            revision = (
                subprocess.check_output([*git, "rev-parse", "HEAD"]).decode().strip()
            )
            sentinel = archive_case(root / "case", repository, revision)
            self.assertEqual(sentinel.read_bytes(), b"source-used")

    def test_archive_and_workflow_mutants_fail(self) -> None:
        mutations = (
            'unlink "$SOURCE/CLAUDE.md"',
            'rm -f "$SOURCE/CLAUDE.md"',
            'rm -f "$SOURCE/CLAUDE.md"; printf AGENTS.md > "$SOURCE/CLAUDE.md"',
        )
        for mutation in mutations:
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory(dir=Path.home() / ".cache/eqiora") as value,
            ):
                sentinel = archive_case(
                    Path(value), git_object_authority().root, BASIS_SHA, mutation
                )
                self.assertFalse(sentinel.exists())

        ordinary = _workflow()
        archive = checker.DIRECT_SOURCE_ARCHIVE_COMMAND
        mandatory = '            test -L "$scratch/source/CLAUDE.md"\n'
        self.assertIn(ARCHIVE_ERROR, observe(ordinary.replace(mandatory, "", 1)))
        dereference = ordinary.replace(
            archive, archive.replace("tar -xf", "tar --dereference -xf"), 1
        )
        self.assertCountEqual(
            observe(dereference),
            [omitted(archive), DEREFERENCE_ERROR, ARCHIVE_ERROR],
        )
        moved = ordinary.replace(archive, "ARCHIVE-MOVED", 1).replace(
            "unshare --net", f"{archive}\nunshare --net", 1
        )
        self.assertIn(ORDER_ERROR, observe(moved))

    def test_browser_identity_mutants_fail(self) -> None:
        ordinary = _workflow()
        replacements = (
            (
                "377a9fc07b495c3b8d0ab8ec1fdfd153641759f9756fea34d8dfce4de82f5025",
                "1" * 64,
            ),
            (
                'expected_browser_bytes="301186376"',
                'expected_browser_bytes="290614601"',
            ),
        )
        for old, new in replacements:
            with self.subTest(old=old):
                self.assertIn(BROWSER_ERROR, observe(ordinary.replace(old, new, 1)))
        for token in checker.OFFLINE_WORKFLOW_TOKENS:
            self.assertTrue(observe(ordinary.replace(token, "removed")))
        for token in checker.FORBIDDEN_WORKFLOW_TOKENS:
            self.assertTrue(observe(f"{ordinary}\n# {token}\n"))


if __name__ == "__main__":
    unittest.main()
