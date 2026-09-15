"""Focused checks for the current reader routes, without a second route registry."""

from pathlib import Path
import re
import unittest
from urllib.parse import urlsplit

from tools.site.check_site_sitemap import SITEMAP_ROUTES
from tools.site.check_site_starlight import STARLIGHT_ROUTES

ROOT = Path(__file__).resolve().parents[3]
CONTENT = ROOT / "docs/site/src/content/docs"


class ReaderDestinationTests(unittest.TestCase):
    def test_every_published_body_is_in_the_existing_route_owners(self):
        routes = {"/404.html"}
        for path in CONTENT.rglob("*.mdx"):
            relative = path.relative_to(CONTENT).with_suffix("").as_posix()
            route = relative.removesuffix("/index")
            routes.add("/" if route == "index" else f"/{route}/")
        self.assertEqual(set(STARLIGHT_ROUTES), routes)
        self.assertEqual(set(SITEMAP_ROUTES), routes - {"/404.html"})
        support = (ROOT / "docs/site/tests/support.ts").read_text()
        ordinary = support.split("export const SITE_ROUTES = [", 1)[1].split(
            "] as const;", 1
        )[0]
        browser_routes = re.findall(r"'(/[^']*)'", ordinary)
        self.assertEqual(set(browser_routes), routes)
        self.assertEqual(len(browser_routes), len(routes))

    def test_displaced_destinations_have_no_source_bodies(self):
        for name in ("textbooks", "python", "api", "examples", "concepts", "architecture"):
            self.assertEqual(list((CONTENT / name).rglob("*.mdx")), [], name)
        lessons = list((CONTENT / "learn/mathematical-modeling").glob("*.mdx"))
        self.assertEqual(len(lessons), 9)  # The path introduction plus eight existing lessons.

    def test_markdown_and_component_route_links_target_current_destinations(self):
        # Route props become anchors only after MDX rendering; cover them as well
        # as Markdown links without inventing another destination inventory.
        pattern = re.compile(r'\]\((/[^\s)]+)\)|(?:href|routeHref)="(/[^"\s]+)"')
        for source in CONTENT.rglob("*.mdx"):
            for match in pattern.finditer(source.read_text()):
                path = urlsplit(match.group(1) or match.group(2)).path
                if path.startswith("/reference/rust/api/"):
                    continue  # Existing owned assembly supplies Rustdoc and the schema.
                self.assertIn(path, STARLIGHT_ROUTES, f"{source}: {path}")

    def test_detailed_guide_pages_only_present_the_canonical_bodies(self):
        for name in ("modeling", "execution-and-arrays", "differentiation"):
            canonical = ROOT / f"docs/python/{name}.md"
            self.assertTrue(canonical.is_file())
            wrapper = (CONTENT / f"guides/{name}.mdx").read_text()
            self.assertIn(f'<CanonicalGuide name="{name}" />', wrapper)
            self.assertNotIn("```", wrapper)
        imports = (ROOT / "docs/site/src/canonical-guides.ts").read_text()
        self.assertIn("../../python/{modeling,execution-and-arrays,differentiation}.md", imports)


if __name__ == "__main__":
    unittest.main()
