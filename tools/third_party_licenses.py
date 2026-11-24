"""Generate THIRD_PARTY_LICENSES.txt for code bundled into Proteus distributions.

Usage: python tools/third_party_licenses.py [--check]

Wheels contain the compiled Rust extension (its runtime crates) and the built web interface
(its runtime npm packages). Python dependencies are installed separately and are not listed.
Run after changing Cargo.lock or frontend/package-lock.json (needs cargo and an installed
frontend/node_modules, and network access for the few crates published without license
files); --check fails when the committed file is out of date.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "THIRD_PARTY_LICENSES.txt"
LICENSE_FILE = re.compile(r"^(licen[cs]e|copying|notice|unlicense)", re.IGNORECASE)
_LOFTY = "https://raw.githubusercontent.com/Serial-ATA/lofty-rs/0.25.5/"
# Published crates that ship no license file: texts from their pinned upstream sources.
# Keyed by version so an upgrade fails loudly until the source is re-checked.
UPSTREAM = {
    **{
        (name, version): [_LOFTY + "LICENSE-MIT", _LOFTY + "LICENSE-APACHE"]
        for name, version in (("lofty", "0.25.5"), ("lofty_attr", "0.13.0"), ("ogg_pager", "0.8.0"))
    },
    # Declares MIT without a copyright notice; it is a port of Chromaprint, whose MIT notice
    # (the license of Chromaprint's own code) is reproduced for attribution.
    ("rusty-chromaprint", "0.3.0"): [
        "https://raw.githubusercontent.com/acoustid/chromaprint/v1.5.1/LICENSE.md"
    ],
}


def license_texts(name: str, version: str, directory: Path) -> list[tuple[str, str]]:
    files = sorted(p for p in directory.iterdir() if p.is_file() and LICENSE_FILE.match(p.name))
    texts = [(p.name, p.read_text(encoding="utf-8", errors="replace").strip()) for p in files]
    for url in [] if texts else UPSTREAM.get((name, version), []):
        with urllib.request.urlopen(url, timeout=30) as response:
            texts.append((url, response.read().decode("utf-8").strip()))
    return texts


def rust_packages() -> list[tuple[str, str, str, Path]]:
    tree = subprocess.run(
        ["cargo", "tree", "-e", "normal", "--target", "all", "--prefix", "none", "--locked"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    used = {tuple(line.split()[:2]) for line in tree.splitlines() if line.strip()}
    meta = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--locked"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout
    )  # fmt: skip
    return sorted(
        (p["name"], p["version"], p.get("license") or "see files", Path(p["manifest_path"]).parent)
        for p in meta["packages"]
        if (p["name"], f"v{p['version']}") in used and p["source"]
    )


def npm_packages() -> list[tuple[str, str, str, Path]]:
    lock = json.loads((ROOT / "frontend/package-lock.json").read_text())
    return sorted(
        (path.removeprefix("node_modules/"), info["version"], info.get("license", "see files"),
         ROOT / "frontend" / path)
        for path, info in lock["packages"].items()
        if path.startswith("node_modules/") and not info.get("dev")
    )  # fmt: skip


def render() -> str:
    sections, missing, seen = [], [], {}
    for kind, packages in (("Rust crate", rust_packages()), ("npm package", npm_packages())):
        for name, version, license_, directory in packages:
            texts = license_texts(name, version, directory)
            if not texts:
                missing.append(f"{name} {version}")
            source = (
                f"https://crates.io/crates/{name}/{version}"
                if kind == "Rust crate"
                else f"https://www.npmjs.com/package/{name}/v/{version}"
            )
            parts = []
            for file, text in texts:  # Identical texts (e.g. Apache-2.0) are printed once.
                key = " ".join(text.split())
                if key in seen:
                    parts.append(f"--- {file}: identical to the text given for {seen[key]} ---")
                else:
                    seen[key] = f"{name} {version}"
                    parts.append(f"--- {file} ---\n{text}")
            body = "\n\n".join(parts)
            sections.append(
                f"{kind}: {name} {version}\nLicense: {license_}\nSource: {source}\n\n{body}"
            )
    if missing:
        sys.exit(f"no license file found for: {', '.join(missing)}")
    header = (
        "Third-party software bundled in Proteus distributions\n\n"
        "The Proteus native extension and web interface include the components below, each under\n"
        "its own license. MPL-2.0 components are used unmodified; their source code is available\n"
        "at the listed source URL.\n"
    )
    return "\n\n" + ("=" * 78 + "\n").join([header, *(f"{s}\n" for s in sections)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render().lstrip()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != text:
            sys.exit(f"{OUTPUT.name} is out of date; run python tools/third_party_licenses.py")
        return
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT.name}")


if __name__ == "__main__":
    main()
