"""Package the submission from an explicit allowlist, deterministically.

    python tools/build_submission_zip.py [--with-labeling] [--out dist/paiec_irt.zip]

Only the listed source files and ``params.json`` enter the archive, with fixed
timestamps and permissions so identical inputs give byte-identical ZIPs. Every
member is scanned for credential-like strings before writing; any hit aborts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "submission"
BASE_FILES = ["model.py", "pirt_online.py", "params.json", "requirements.txt", "README.md"]
LABELING_FILES = ["labeling.py", "labeling_core.py"]

CREDENTIAL_PATTERNS = [
    r"sk-[A-Za-z0-9_-]{16,}",
    r"sk-ant-[A-Za-z0-9_-]{8,}",
    r"hf_[A-Za-z0-9]{20,}",
    r"gh[pousr]_[A-Za-z0-9]{20,}",
    r"github_pat_[A-Za-z0-9_]{20,}",
    r"AKIA[0-9A-Z]{16}",
    r"AIza[0-9A-Za-z_-]{30,}",
    r"xox[abpr]-[A-Za-z0-9-]{10,}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"(?i)(api[_-]?key|secret|password|bearer)\s*[:=]\s*['\"][^'\"]{8,}['\"]",
]


def scan_credentials(name: str, data: bytes) -> list:
    text = data.decode("utf-8", errors="replace")
    return [f"{name}: matches {p!r}" for p in CREDENTIAL_PATTERNS if re.search(p, text)]


def build(out: Path, with_labeling: bool, allow_synthetic: bool = False) -> list:
    params = json.loads((SRC / "params.json").read_text())
    if params.get("provenance", {}).get("synthetic") and not allow_synthetic:
        raise SystemExit("params.json was fitted on synthetic data; refit on measurement-db before packaging")
    names = BASE_FILES + (LABELING_FILES if with_labeling else [])
    members, problems = [], []
    for name in names:
        path = SRC / name
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"missing or symlinked source: {path}")
        data = path.read_bytes()
        problems += scan_credentials(name, data)
        members.append((name, data))
    if problems:
        raise SystemExit("credential-like content found:\n  " + "\n  ".join(problems))
    out.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(out, "w", compression=ZIP_DEFLATED) as zf:
        for name, data in sorted(members):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            zf.writestr(info, data)
    return [(n, len(d), hashlib.sha256(d).hexdigest()[:16]) for n, d in sorted(members)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "dist" / "paiec_irt.zip"))
    ap.add_argument("--with-labeling", action="store_true")
    ap.add_argument("--allow-synthetic", action="store_true", help="pipeline tests only")
    args = ap.parse_args(argv)
    out = Path(args.out)
    manifest = build(out, args.with_labeling, args.allow_synthetic)
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    for name, size, digest in manifest:
        print(f"  {name:<20} {size:>10,} B  sha256:{digest}")
    print("credential scan: clean")


if __name__ == "__main__":
    sys.exit(main())
