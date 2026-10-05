"""Download the pretrained DC-ELLA learners from a GitHub release and verify them.

The learners (ella_learner.pkl / ella_learner_final.pkl) are too large for the
repository, so they are published as release assets with flat names. This
script puts every asset at the path the experiment scripts expect and checks
its SHA-256 against tools/checkpoints_manifest.tsv.

Examples (run from the repository root):
    python tools/fetch_checkpoints.py --repo ning17822099770-svg/DC-ELLA-UAV-Metaverse --tag v1.0
    python tools/fetch_checkpoints.py --from-dir path/to/downloaded/assets
    python tools/fetch_checkpoints.py --verify-only
"""
import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = Path(__file__).with_name("checkpoints_manifest.tsv")


def read_manifest():
    lines = MANIFEST.read_text(encoding="utf-8").splitlines()[1:]
    return [dict(zip(("asset", "target", "size", "sha256"), l.split("\t"))) for l in lines if l.strip()]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", help="GitHub repository, e.g. owner/DC-ELLA-UAV-Metaverse")
    src.add_argument("--from-dir", type=Path, help="folder that already contains the release assets")
    src.add_argument("--verify-only", action="store_true", help="only check files already in place")
    ap.add_argument("--tag", default="v1.0", help="release tag (default: v1.0)")
    args = ap.parse_args()

    ok = True
    for row in read_manifest():
        target = ROOT / row["target"]
        if not args.verify_only and not (target.exists() and sha256(target) == row["sha256"]):
            target.parent.mkdir(parents=True, exist_ok=True)
            if args.from_dir:
                shutil.copyfile(args.from_dir / row["asset"], target)
            else:
                url = f"https://github.com/{args.repo}/releases/download/{args.tag}/{row['asset']}"
                print(f"downloading {url}")
                urllib.request.urlretrieve(url, target)
        good = target.exists() and sha256(target) == row["sha256"]
        ok &= good
        print(f"[{'OK' if good else 'MISSING/BAD'}] {row['target']}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
