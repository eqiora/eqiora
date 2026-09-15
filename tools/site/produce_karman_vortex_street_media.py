#!/usr/bin/env python3
"""Render caller-owned media from the unverified Reynolds-100 wake example."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "python"
FRAME_RATE = 8
FIGURE_SIZE = (10.0, 5.625)
DPI = 144


def _solve():
    sys.path.insert(0, str(EXAMPLES))
    try:
        from karman_vortex_street import solve

        return solve()
    finally:
        sys.path.pop(0)


def _render_frame(
    coordinates: np.ndarray,
    triangles: np.ndarray,
    vorticity: np.ndarray,
    times_s: np.ndarray,
    lift: np.ndarray,
    *,
    frame_index: int,
    magnitude: float,
    strouhal: float | None,
) -> Figure:
    figure = Figure(figsize=FIGURE_SIZE, dpi=DPI, facecolor="#f8fafc")
    time_s = float(times_s[frame_index])
    figure.text(
        0.06,
        0.94,
        "Kármán vortex street",
        color="#0f172a",
        fontsize=20,
        fontweight="bold",
        va="top",
    )
    figure.text(
        0.94,
        0.94,
        f"Re = 100   ·   t = {time_s:.2f} s",
        color="#334155",
        fontsize=11,
        ha="right",
        va="top",
    )

    flow = figure.add_axes((0.06, 0.49, 0.82, 0.30))
    flow.set_facecolor("#eef2f7")
    scalar = flow.tripcolor(
        coordinates[:, 0],
        coordinates[:, 1],
        triangles=triangles,
        facecolors=np.clip(vorticity, -magnitude, magnitude),
        shading="flat",
        cmap="RdBu_r",
        vmin=-magnitude,
        vmax=magnitude,
    )
    flow.set_xlim(0.0, 2.2)
    flow.set_ylim(0.0, 0.41)
    flow.set_aspect("equal", adjustable="box")
    flow.set_xlabel("x [m]")
    flow.set_ylabel("y [m]")
    flow.set_title("Cell-average vorticity ω [s⁻¹]", loc="left", fontsize=11)
    colorbar = figure.colorbar(scalar, cax=figure.add_axes((0.90, 0.49, 0.014, 0.30)))
    colorbar.set_label("ω [s⁻¹]")
    colorbar.ax.text(
        0.5,
        1.04,
        "clipped",
        transform=colorbar.ax.transAxes,
        ha="center",
        fontsize=8,
        color="#64748b",
    )

    history = figure.add_axes((0.09, 0.13, 0.82, 0.23))
    history.axhline(0.0, color="#94a3b8", linewidth=0.8)
    history.plot(times_s, lift, color="#17417e", linewidth=2.0)
    history.plot(time_s, lift[frame_index], "o", color="#dc2626", markersize=6)
    history.axvline(time_s, color="#dc2626", linewidth=1.0, alpha=0.45)
    history.set_xlim(float(times_s[0]), float(times_s[-1]))
    lift_limit = max(float(np.abs(lift).max()) * 1.15, 0.05)
    history.set_ylim(-lift_limit, lift_limit)
    history.set_xlabel("physical time [s]")
    history.set_ylabel("lift coefficient $C_L$")
    history.set_title("Cylinder lift over the sampled wake window", loc="left", fontsize=11)
    history.grid(axis="y", color="#cbd5e1", linewidth=0.6, alpha=0.7)

    st_label = "St not resolved" if strouhal is None else f"sampled St = {strouhal:.3f}"
    figure.text(
        0.06,
        0.035,
        f"MINI/P1 · backward Euler Δt = 0.01 s · {st_label}",
        color="#475569",
        fontsize=9,
    )
    figure.text(
        0.94,
        0.035,
        "UNVERIFIED PRODUCT EXAMPLE",
        color="#9f1239",
        fontsize=9,
        fontweight="bold",
        ha="right",
    )
    return figure


def _save_png(figure: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format="png",
        dpi=DPI,
        metadata={"Software": "Eqiora Karman vortex street presentation v1"},
    )


def _save_reduced_motion(poster: Path, output: Path) -> None:
    """Derive a visibly identified static alternative from the accepted poster."""
    with Image.open(poster) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    label = "STATIC PHASE SNAPSHOT"
    font = ImageFont.load_default(size=15)
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    width = right - left
    height = bottom - top
    x = image.width - width - 30
    y = 101
    draw.rounded_rectangle(
        (x - 10, y - 7, x + width + 10, y + height + 7),
        radius=6,
        fill="#f8fafc",
        outline="#94a3b8",
        width=1,
    )
    draw.text((x, y), label, fill="#475569", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(
        output,
        format="PNG",
        optimize=True,
        pnginfo=None,
    )


def _encode(ffmpeg: str, frames: Path, output: Path, *, codec: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(FRAME_RATE),
        "-i",
        str(frames / "frame-%03d.png"),
        "-an",
        "-c:v",
        codec,
        "-pix_fmt",
        "yuv420p",
    ]
    if codec == "libx264":
        command.extend(("-crf", "24", "-movflags", "+faststart"))
    else:
        command.extend(("-crf", "32", "-b:v", "0"))
    command.append(str(output))
    subprocess.run(command, check=True, timeout=300)


def produce(poster: Path, reduced_motion: Path, webm: Path, mp4: Path) -> None:
    wake = _solve()
    trajectory = wake.result.trajectory
    coordinates = np.asarray(trajectory.coordinates)
    triangles = np.asarray(trajectory.cells)
    states = trajectory.states
    values = np.asarray(
        [state.curl(wake.plan.capability.velocity).values("cell") for state in states]
    )
    if len(states) != 100:
        raise RuntimeError("the wake media requires exactly 100 sampled output states")
    centers = coordinates[triangles].mean(axis=1)
    visible_wake = values[:, centers[:, 0] >= 0.3]
    magnitude = float(np.quantile(np.abs(visible_wake), 0.995))
    if not np.isfinite(magnitude) or magnitude <= 0.0:
        raise RuntimeError("the wake vorticity scale must be finite and nonzero")
    if not np.isfinite(wake.lift_coefficients).all():
        raise RuntimeError("the wake lift history must be finite")

    poster_index = int(np.argmax(np.abs(wake.lift_coefficients)))
    poster_figure = _render_frame(
        coordinates,
        triangles,
        values[poster_index],
        wake.times_s,
        wake.lift_coefficients,
        frame_index=poster_index,
        magnitude=magnitude,
        strouhal=wake.strouhal_number,
    )
    _save_png(poster_figure, poster)
    _save_reduced_motion(poster, reduced_motion)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode wake video")
    with tempfile.TemporaryDirectory(prefix="eqiora-karman-wake-", dir=Path.home()) as raw:
        frame_directory = Path(raw)
        for index, frame_values in enumerate(values):
            _save_png(
                _render_frame(
                    coordinates,
                    triangles,
                    frame_values,
                    wake.times_s,
                    wake.lift_coefficients,
                    frame_index=index,
                    magnitude=magnitude,
                    strouhal=wake.strouhal_number,
                ),
                frame_directory / f"frame-{index:03d}.png",
            )
        _encode(ffmpeg, frame_directory, webm, codec="libvpx-vp9")
        _encode(ffmpeg, frame_directory, mp4, codec="libx264")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poster", type=Path, required=True)
    parser.add_argument("--reduced-motion", type=Path, required=True)
    parser.add_argument("--webm", type=Path, required=True)
    parser.add_argument("--mp4", type=Path, required=True)
    arguments = parser.parse_args()
    produce(arguments.poster, arguments.reduced_motion, arguments.webm, arguments.mp4)


if __name__ == "__main__":
    main()
