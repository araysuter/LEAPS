#!/usr/bin/env python3
"""Build LEAPS application and in-app brand assets from the approved logo PNG."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "leaps" / "assets"


def _square_logo(source: Image.Image) -> Image.Image:
    logo = source.convert("RGBA")
    alpha_box = logo.getchannel("A").getbbox()
    if alpha_box is None:
        raise ValueError("The source logo is fully transparent")
    logo = logo.crop(alpha_box)
    edge = max(logo.size)
    square = Image.new("RGBA", (edge, edge), (0, 0, 0, 0))
    square.alpha_composite(logo, ((edge - logo.width) // 2, (edge - logo.height) // 2))
    return square


def _superellipse_mask(size: int, inset: int, exponent: float = 5.0) -> Image.Image:
    axis = np.arange(size, dtype=np.float32) - ((size - 1) / 2)
    x, y = np.meshgrid(axis, axis)
    radius = (size - (2 * inset)) / 2
    distance = (np.abs(x / radius) ** exponent) + (np.abs(y / radius) ** exponent)
    # A narrow antialiased transition keeps the shape clean at every Dock size.
    alpha = np.clip((1.006 - distance) * 160.0, 0.0, 1.0)
    return Image.fromarray(np.asarray(alpha * 255, dtype=np.uint8), mode="L")


def _squircle_background(size: int, mask: Image.Image) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    diagonal = (xx + yy) / (2 * (size - 1))
    highlight = np.exp(-(((xx - size * 0.25) ** 2) + ((yy - size * 0.18) ** 2)) / (size**2 * 0.19))
    base = np.empty((size, size, 4), dtype=np.uint8)
    base[:, :, 0] = np.clip(12 + (16 * highlight) - (4 * diagonal), 0, 255)
    base[:, :, 1] = np.clip(42 + (20 * highlight) - (7 * diagonal), 0, 255)
    base[:, :, 2] = np.clip(61 + (24 * highlight) - (8 * diagonal), 0, 255)
    base[:, :, 3] = np.asarray(mask)
    return Image.fromarray(base, mode="RGBA")


def build_assets(source_path: Path, output_dir: Path = ASSETS) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    approved_source = output_dir / "leaps-logo-source.png"
    if source_path.resolve() != approved_source.resolve():
        shutil.copyfile(source_path, approved_source)

    source = Image.open(source_path)
    logo = _square_logo(source)
    resampling = Image.Resampling.LANCZOS

    mark = logo.resize((512, 512), resampling)
    mark.save(output_dir / "leaps-mark.png", optimize=True)

    size = 1024
    shape_mask = _superellipse_mask(size, inset=72)
    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    shadow_alpha = shape_mask.filter(ImageFilter.GaussianBlur(18))
    shadow.putalpha(shadow_alpha.point(lambda value: int(value * 0.48)))
    shadow_color = Image.new("RGBA", (size, size), (0, 7, 15, 255))
    shadow_color.putalpha(shadow.getchannel("A"))

    app_icon = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    app_icon.alpha_composite(shadow_color, (0, 10))
    app_icon.alpha_composite(_squircle_background(size, shape_mask))

    badge_size = 748
    badge = logo.resize((badge_size, badge_size), resampling)
    badge_shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    badge_alpha = badge.getchannel("A").filter(ImageFilter.GaussianBlur(13))
    badge_shadow_layer = Image.new("RGBA", badge.size, (0, 5, 12, 118))
    badge_shadow_layer.putalpha(badge_alpha.point(lambda value: int(value * 0.46)))
    badge_position = ((size - badge_size) // 2, ((size - badge_size) // 2) - 4)
    badge_shadow.alpha_composite(badge_shadow_layer, (badge_position[0], badge_position[1] + 12))
    app_icon.alpha_composite(badge_shadow)
    app_icon.alpha_composite(badge, badge_position)

    app_icon_path = output_dir / "leaps-app-icon.png"
    app_icon.save(app_icon_path, optimize=True)
    app_icon.save(output_dir / "leaps-app-icon.icns", format="ICNS")
    app_icon.save(
        output_dir / "leaps-app-icon.ico",
        format="ICO",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Approved transparent LEAPS logo PNG")
    parser.add_argument("--output-dir", type=Path, default=ASSETS)
    arguments = parser.parse_args()
    build_assets(arguments.source, arguments.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
