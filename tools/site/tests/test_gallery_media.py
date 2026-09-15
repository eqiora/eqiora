"""Ordinary rendering checks; requires the plotting extra and ffmpeg."""

import importlib.util
import json
from pathlib import Path
import subprocess
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "docs/site/src/assets/gallery"
SPEC = importlib.util.spec_from_file_location(
    "wake_media", ROOT / "tools/site/produce_karman_vortex_street_media.py"
)
wake = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wake)


class GalleryMediaTests(unittest.TestCase):
    def test_wake_frame_preserves_domain_scale_and_lift_history(self):
        coordinates = np.array([[0, 0], [2.2, 0], [2.2, 0.41], [0, 0.41]])
        triangles = np.array([[0, 1, 2], [0, 2, 3]])
        times = np.array([7.0, 7.1, 7.2])
        lift = np.array([-0.5, 0.25, 0.5])
        figure = wake._render_frame(
            coordinates,
            triangles,
            np.array([2.0, -2.0]),
            times,
            lift,
            frame_index=1,
            magnitude=3.0,
            strouhal=0.3,
        )
        flow, _, history = figure.axes
        self.assertEqual(flow.get_xlim(), (0.0, 2.2))
        self.assertEqual(flow.get_ylim(), (0.0, 0.41))
        self.assertEqual(flow.get_aspect(), 1)
        self.assertEqual(flow.collections[0].get_clim(), (-3.0, 3.0))
        self.assertEqual((flow.get_xlabel(), flow.get_ylabel()), ("x [m]", "y [m]"))
        self.assertEqual(history.get_xlim(), (7.0, 7.2))
        self.assertIn("lift coefficient", history.get_ylabel())
        labels = " ".join(text.get_text() for text in figure.texts)
        self.assertIn("Re = 100", labels)
        self.assertIn("sampled St = 0.300", labels)
        self.assertIn("Eqiora simulation", labels)

    def test_published_pngs_decode_and_have_visible_content(self):
        self.assertFalse((ASSETS / "exact-cylinder-pressure.png").exists())
        for path in ASSETS.glob("*.png"):
            with self.subTest(path=path.name), Image.open(path) as image:
                image.load()
                self.assertGreaterEqual(min(image.size), 300)
                self.assertLessEqual(max(image.size), 4096)
                self.assertTrue(any(lo != hi for lo, hi in image.convert("RGB").getextrema()))
        with Image.open(ASSETS / "exact-cylinder-pressure-presentation.png") as image:
            self.assertGreater(image.width / image.height, 3)

    def test_wake_videos_decode_100_frames_and_match_the_poster(self):
        with Image.open(ASSETS / "karman-vortex-street-poster.png") as poster:
            size = poster.size
        for suffix in ("mp4", "webm"):
            with self.subTest(format=suffix):
                result = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-count_frames",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=width,height,nb_read_frames,r_frame_rate",
                        "-of",
                        "json",
                        str(ASSETS / f"karman-vortex-street.{suffix}"),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                (stream,) = json.loads(result.stdout)["streams"]
                self.assertEqual((stream["width"], stream["height"]), size)
                self.assertEqual(stream["nb_read_frames"], "100")
                self.assertEqual(stream["r_frame_rate"], "8/1")


if __name__ == "__main__":
    unittest.main()
