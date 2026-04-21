"""
Copy gender avatar PNGs from Cursor chat uploads into ./assets/ (same rules as app.py).

  python copy_gender_avatars.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST_DIR = ROOT / "assets"

_GENDER_UPLOAD_PATTERN_PAIRS = (
    ("*20260420191450_72_131*.png", "*20260420191452_73_131*.png"),
    ("*771d108b*.png", "*57d53876*.png"),
)


def _latest_png_match(directory: Path, pattern: str) -> Path | None:
    hits = [p for p in directory.glob(pattern) if p.is_file()]
    if not hits:
        return None
    return max(hits, key=lambda p: p.stat().st_mtime)


def _pick_gender_sources_from_dir(src_dir: Path) -> tuple[Path | None, Path | None]:
    for pat_m, pat_f in _GENDER_UPLOAD_PATTERN_PAIRS:
        m = _latest_png_match(src_dir, pat_m)
        f = _latest_png_match(src_dir, pat_f)
        if m and f:
            return m, f
    return None, None


def _gender_avatar_cursor_assets_dir() -> Path | None:
    root = Path.home() / ".cursor" / "projects"
    if not root.is_dir():
        return None
    explicit = root / "c-Users-Administrator-Desktop-Diet-Exercise-Plan-App" / "assets"
    if explicit.is_dir():
        return explicit
    for proj in root.iterdir():
        if not proj.is_dir():
            continue
        ad = proj / "assets"
        if not ad.is_dir():
            continue
        if _pick_gender_sources_from_dir(ad)[0]:
            return ad
    return None


def main() -> int:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    src_dir = _gender_avatar_cursor_assets_dir()
    if not src_dir:
        print("Could not find Cursor assets folder for this project.")
        return 1
    src_m, src_f = _pick_gender_sources_from_dir(src_dir)
    if not src_m or not src_f:
        print("No matching male+female PNG pair in:", src_dir)
        return 1
    shutil.copy2(src_m, DEST_DIR / "gender_avatar_male.png")
    shutil.copy2(src_f, DEST_DIR / "gender_avatar_female.png")
    print("Saved:", DEST_DIR / "gender_avatar_male.png")
    print("Saved:", DEST_DIR / "gender_avatar_female.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
