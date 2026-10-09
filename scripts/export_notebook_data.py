"""Export compact, attributed arrays from the executed runs for an offline notebook."""

import hashlib
import json
from pathlib import Path
import shutil


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / "reports" / "artifacts"
    destination.mkdir(parents=True, exist_ok=True)
    files = {
        "O4_wave1.npz": root / "data/processed/O4_wave1.npz",
        "O4_wave3.npz": root / "data/processed/O4_wave3.npz",
        "measured_predictions.npz": root / "outputs/measured/O4_wave1/predictions.npz",
        "synthetic_predictions.npz": root / "outputs/synthetic/predictions.npz",
        "synthetic_reference.npz": root / "outputs/synthetic/reference.npz",
    }
    manifest = {}
    for name, source in files.items():
        if not source.is_file():
            raise FileNotFoundError(f"Run the README benchmark workflow first: {source.relative_to(root)}")
        target = destination / name
        shutil.copyfile(source, target)
        manifest[name] = {"source": str(source.relative_to(root)), "bytes": target.stat().st_size,
                          "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(files)} compact artifacts to {destination.relative_to(root)}")


if __name__ == "__main__":
    main()
