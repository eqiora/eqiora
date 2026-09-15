"""Reference sources and commands share the current installed candidate."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "docs/site/src/content/docs/reference"
PAGES = sorted((REFERENCE / "language").glob("*.mdx")) + sorted(
    (REFERENCE / "standard-packages").glob("*.mdx")
)
RAW_IMPORT = re.compile(r"import (\w+) from '([^']+\.eqi)\?raw';")


class CurrentReferenceContentTests(unittest.TestCase):
    def test_rendered_examples_have_one_maintained_source(self):
        self.assertEqual(len(PAGES), 9)
        examples = set()
        for page in PAGES:
            content = page.read_text(encoding="utf-8")
            self.assertNotIn("0.1.0a7", content)
            self.assertNotIn("```eqiora", content)
            self.assertIn("editUrl: https://github.com/nkiyohara/eqiora/edit/main/", content)
            for name, relative in RAW_IMPORT.findall(content):
                path = (page.parent / relative).resolve()
                self.assertTrue(path.is_relative_to(ROOT))
                self.assertTrue(path.is_file())
                self.assertRegex(content, rf"<Code\s+code=\{{{name}(?:\}}|\.slice\()")
                examples.add(path)
            for code in re.findall(r"```python\n(.*?)```", content, re.DOTALL):
                compile(code, str(page), "exec")
        self.assertEqual(
            {path for path in examples if path.is_relative_to(REFERENCE)},
            set(REFERENCE.glob("*/_examples/*.eqi")),
        )

    def test_reference_links_resolve_to_current_sources_and_routes(self):
        content_root = ROOT / "docs/site/src/content/docs"
        for page in [REFERENCE / "index.mdx", *PAGES]:
            content = page.read_text(encoding="utf-8")
            for source in re.findall(r'<ExactSourceLink[^>]*path="([^"]+)"', content):
                self.assertTrue((ROOT / source).exists(), f"{page}: {source}")
            links = re.findall(r"\]\((/[^)#]*)(?:#[^)]+)?\)", content)
            for link in links:
                route = content_root / link.strip("/")
                candidates = [route / "index.mdx", route / "index.md",
                              route.with_suffix(".mdx"), route.with_suffix(".md")]
                self.assertTrue(any(path.is_file() for path in candidates), f"{page}: {link}")
            self.assertNotRegex(content, r"github\.com/nkiyohara/eqiora/(?:blob|tree)/")


if __name__ == "__main__":
    unittest.main()
