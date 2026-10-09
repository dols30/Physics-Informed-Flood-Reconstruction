"""Build a shareable repository ZIP without local data, environments or checkpoints."""

import argparse
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FILES = ["README.md", "LICENSE", "NOTICE", ".gitignore", "pyproject.toml", "requirements.txt", "requirements-lock.txt",
         "requirements-notebook-lock.txt"]
DIRECTORIES = [".github", "assets", "docs", "notebooks", "reports", "scripts", "src", "tests"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/releases/Physics-Informed-Flood-Reconstruction.zip")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    files = [ROOT / name for name in FILES]
    for directory in DIRECTORIES:
        files.extend(p for p in (ROOT / directory).rglob("*") if p.is_file()
                     and not any(part in {"__pycache__", ".ipynb_checkpoints"} or part.endswith(".egg-info")
                                 for part in p.parts)
                     and p.suffix not in {".pyc", ".pyo"})
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, Path("Physics-Informed-Flood-Reconstruction") / path.relative_to(ROOT))
    print(f"Created {args.output} ({args.output.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
