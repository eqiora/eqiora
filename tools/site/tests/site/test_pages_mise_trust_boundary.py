from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[4]
WORKFLOW = REPOSITORY / ".github/workflows/pages.yml"
SCRATCH_ROOT = Path.home() / ".cache/eqiora/site-trust-tests"

MISE_ACTION = (
    "jdx/mise-action@6be3c7ed4c21d4f76b208d351d95aac273ebb76d # v5.2.4"
)
CACHE_ACTION = (
    "actions/cache@55cc8345863c7cc4c66a329aec7e433d2d1c52a9 # v6.1.0"
)


def _named_step(workflow: str, name: str) -> str:
    marker = f"      - name: {name}\n"
    if workflow.count(marker) != 1:
        raise AssertionError(f"workflow step is not unique: {name}")
    start = workflow.index(marker)
    end = workflow.find("      - name:", start + len(marker))
    return workflow[start:] if end < 0 else workflow[start:end]


class PagesMiseTrustBoundaryTests(unittest.TestCase):
    def test_unused_chrome_feed_is_disabled_before_apt_updates(self) -> None:
        supply = _named_step(
            WORKFLOW.read_text(encoding="utf-8"),
            "Supply locked native, Rust, Python, Node, and browser inputs",
        )
        start = supply.index("          for source in /etc/apt/sources.list.d/")
        end = supply.index("          sudo apt-get update", start)
        script = textwrap.dedent(supply[start:end])
        SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as temporary:
            root = Path(temporary)
            feeds = {
                "chrome.list": "deb https://dl.google.com/linux/chrome/deb/ stable main\n",
                "chrome.sources": "Types: deb\nURIs: https://dl.google.com/linux/chrome-stable/deb\n",
                "ubuntu.sources": "Types: deb\nURIs: http://archive.ubuntu.com/ubuntu/\n",
                "unrelated.list": "deb https://example.com/linux/chrome/deb/ stable main\n",
            }
            for name, content in feeds.items():
                (root / name).write_text(content, encoding="utf-8")
            command = 'sudo() { "$@"; }\n' + script.replace(
                "/etc/apt/sources.list.d/", f"{root}/"
            )
            for _ in range(2):
                subprocess.run(["bash", "-eu", "-c", command], check=True)
            for name, content in feeds.items():
                disabled = name.startswith("chrome.")
                path = root / (name + ".disabled" if disabled else name)
                self.assertEqual(path.read_text(encoding="utf-8"), content)
                if disabled:
                    self.assertFalse((root / name).exists())

    def test_playwright_cache_uses_the_reviewed_action_release(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        step = _named_step(workflow, "Restore the exact Playwright browser cache")

        self.assertIn(f"        uses: {CACHE_ACTION}\n", step)

    def test_locked_toolchain_is_installed_without_a_split_mise_cache(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        step = _named_step(workflow, "Install the locked mise toolchain")

        self.assertIn(f"        uses: {MISE_ACTION}\n", step)
        self.assertIn("          version: 2026.5.10\n", step)
        self.assertIn("          cache: false\n", step)
        self.assertNotIn("cache: true", step)

        supply = _named_step(
            workflow, "Supply locked native, Rust, Python, Node, and browser inputs"
        )
        self.assertIn("          mise install\n", supply)
        self.assertIn(
            '          test "$(rustc -Vv)" = "$(rustc +stable -Vv)"\n', supply
        )

    def test_offline_build_trusts_only_the_archived_mise_config(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        step = _named_step(workflow, "Build and verify with only loopback networking")

        trust = (
            '                  MISE_TRUSTED_CONFIG_PATHS='
            '"$EQIORA_SITE_SOURCE_ROOT/mise.toml" \\\n'
        )
        chdir = '                  --chdir="$EQIORA_SITE_SOURCE_ROOT" \\\n'
        runner = (
            '                  "$EQIORA_SITE_SOURCE_ROOT/tools/site/'
            'run_offline_site_checks.sh"\n'
        )
        self.assertEqual(step.count(trust), 1)
        self.assertEqual(step.count(chdir), 1)
        self.assertIn(chdir + trust, step)
        self.assertEqual(step.count(runner), 1)
        self.assertIn("            unshare --net -- bash -ceu '\n", step)
        self.assertIn("              ip link set lo up\n", step)
        self.assertIn("                  CARGO_NET_OFFLINE=true \\\n", step)
        self.assertIn("                  npm_config_offline=true \\\n", step)
        self.assertNotIn("mise trust", workflow)
        self.assertNotIn("MISE_YES", workflow)
        self.assertNotIn("GITHUB_WORKSPACE/tools/site/run_offline_site_checks.sh", step)

    def test_exact_file_trust_is_effective_at_runtime(self) -> None:
        if not sys.platform.startswith("linux"):
            self.skipTest("mise trust behavior is exercised on Linux")
        mise = shutil.which("mise")
        if mise is None:
            self.skipTest("mise is provisioned by the repository setup gate")

        SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
        self.assertTrue(SCRATCH_ROOT.is_dir())
        self.assertFalse(SCRATCH_ROOT.is_symlink())
        with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as temporary:
            root = Path(temporary).resolve()
            source = root / "source"
            source.mkdir()
            for directory in ("config", "state", "cache", "data"):
                (root / directory).mkdir()
            (source / "mise.toml").write_text(
                '[tools]\nnode = "24.18.1"\n', encoding="utf-8"
            )
            probe = root / "probe.py"
            probe.write_text(
                textwrap.dedent(
                    """\
                    import os
                    import subprocess

                    result = subprocess.run(
                        [os.environ["PROBE_MISE"], "current", "node"],
                        check=False,
                        env=os.environ,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    print(result.stdout, end="")
                    print(result.stderr, end="", file=__import__("sys").stderr)
                    raise SystemExit(result.returncode)
                    """
                ),
                encoding="utf-8",
            )
            base = os.environ.copy()
            for name in tuple(base):
                if name.startswith("MISE_") or name in {"CI", "GITHUB_ACTIONS"}:
                    base.pop(name, None)
            base.update(
                {
                    "HOME": str(Path.home()),
                    "LC_ALL": "C",
                    "LANG": "C",
                    "TZ": "UTC",
                    "MISE_CONFIG_DIR": str(root / "config"),
                    "MISE_STATE_DIR": str(root / "state"),
                    "MISE_CACHE_DIR": str(root / "cache"),
                    "MISE_DATA_DIR": str(root / "data"),
                    "MISE_OFFLINE": "1",
                    "PROBE_MISE": str(Path(mise).resolve()),
                }
            )
            command = [
                shutil.which("env", path=os.defpath) or "env",
                f"--chdir={source}",
                str(Path(sys.executable).resolve()),
                str(probe),
            ]

            trusted = subprocess.run(
                command,
                check=False,
                env={
                    **base,
                    "MISE_TRUSTED_CONFIG_PATHS": str(source / "mise.toml"),
                },
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
            )
            self.assertEqual(
                trusted.returncode,
                0,
                f"stdout:\n{trusted.stdout}\nstderr:\n{trusted.stderr}",
            )
            self.assertEqual(trusted.stdout.strip(), "24.18.1")

            untrusted = subprocess.run(
                command,
                check=False,
                env=base,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
            )
            self.assertNotEqual(untrusted.returncode, 0)
            self.assertIn("not trusted", untrusted.stdout + untrusted.stderr)
            self.assertIn(str(source / "mise.toml"), untrusted.stdout + untrusted.stderr)


if __name__ == "__main__":
    unittest.main()
