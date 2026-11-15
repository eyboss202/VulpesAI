"""Copy a finished Kaggle session of a Vulpes run to this PC and check it (Vulpes Flos 1.0, K4).

Run it on the PC, from the repository folder, after a session has ended (docs/KAGGLE.md)::

    python scripts/kaggle/sync.py <user>/<notebook>            # e.g. someone/vulpes-flos-train
    python scripts/kaggle/sync.py <user>/<notebook> --keep 1

It asks Kaggle whether the notebook is still running; once it has ended, it downloads the
session's output with the Kaggle command-line tool (``kaggle kernels output``), checks the
newest step checkpoint and ``best.pt`` against the ``SHA256SUMS`` the session wrote
(``scripts/kaggle/session.py``), and only then moves them into ``runs/<run>/`` here, keeping the
newest ``--keep`` step checkpoints. If Kaggle is lost, this copy is where the run continues:
another Kaggle account, a rented GPU or this PC (docs/TRAINING.md, "Moving a run to another
machine").

A copy here that has trained further than the session (the run continued on the PC) is never
overwritten. Needs ``pip install kaggle`` and an API token (kaggle.com -> Settings -> API ->
Create New Token, saved as ``~/.kaggle/kaggle.json``); no Kaggle GPU quota is used.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from session import DONE, RUN, STEP, SUMS, find_runs, newest_step, sha256  # noqa: E402

ACTIVE = ("running", "queued")  # `kaggle kernels status` while the session hasn't ended


def run(args: list[str]) -> str:
    out = subprocess.run(args, check=True, text=True, capture_output=True)
    return out.stdout


def session_active(kaggle: str, notebook: str) -> tuple[bool, str]:
    """``(still running?, Kaggle's words)`` from ``kaggle kernels status``."""
    text = run([kaggle, "kernels", "status", notebook]).strip()
    return any(word in text.lower() for word in ACTIVE), text


def read_sums(path: Path) -> dict[str, str]:
    """``sha256sum`` lines -> {name: digest}."""
    sums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[name.strip().lstrip("*")] = digest
    return sums


def check_download(src: Path) -> dict[str, str]:
    """Check a downloaded run against its ``SHA256SUMS``; returns the checked {name: digest}."""
    if not (src / DONE).is_file() or not (src / SUMS).is_file():
        raise SystemExit(
            f"{src} has no {DONE}/{SUMS}: the session did not end normally (see its log on Kaggle)"
        )
    sums = read_sums(src / SUMS)
    newest = newest_step(p.name for p in src.iterdir())
    if newest and newest[1] not in sums:
        raise SystemExit(f"{SUMS} does not list {newest[1]}, the newest checkpoint in {src}")
    for name, digest in sums.items():
        if not (src / name).is_file():
            raise SystemExit(f"{name} is listed in {SUMS} but was not downloaded")
        if sha256(src / name) != digest:
            raise SystemExit(f"{name}: SHA-256 differs from the session's (download it again)")
        print(f"  {name}: SHA-256 matches the session's")
    return sums


def merge(src: Path, dest: Path) -> list[str]:
    """Move every file of the downloaded run ``src`` into ``dest`` (same relative paths); returns them."""
    moved = []
    for p in sorted(src.rglob("*")):
        if p.is_file():
            rel = p.relative_to(src)
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            os.replace(p, dest / rel)  # same disk: a rename, not a copy
            moved.append(rel.as_posix())
    return moved


def prune_local(run_dir: Path, keep: int) -> list[Path]:
    """Delete all but the newest ``keep`` step checkpoints here (``best.pt`` is never touched)."""
    steps = sorted((int(m.group(1)), p) for p in run_dir.iterdir() if (m := STEP.match(p.name)))
    removed = [p for _, p in steps[:-keep]] if keep > 0 else []
    for p in removed:
        p.unlink()
    return removed


def sync(download_dir: Path, run_name: str, local_dir: Path, keep: int) -> str:
    """Check the run downloaded under ``download_dir`` and move it into ``local_dir``; returns its status."""
    copies = find_runs(download_dir, run_name)
    flat = newest_step(p.name for p in download_dir.iterdir())
    if not copies and flat and (download_dir / "config.yaml").is_file():
        copies = [(flat[0], download_dir)]  # the output came without its runs/<run>/ folders
    if not copies:
        raise SystemExit(f"no checkpoint of {run_name} in the session's output ({download_dir})")
    there, src = copies[-1]
    here = newest_step(p.name for p in local_dir.iterdir()) if local_dir.is_dir() else None
    if here and here[0] > there:
        raise SystemExit(
            f"{local_dir} is ahead of the session ({here[1]} > step {there}): this copy has trained "
            "further on its own, so nothing is copied over it"
        )
    check_download(src)
    status = (src / DONE).read_text(encoding="utf-8").splitlines()
    local_dir.mkdir(parents=True, exist_ok=True)
    merge(src, local_dir)
    for p in prune_local(local_dir, keep):
        print(f"  deleted the older {p.name} here")
    if len(status) > 1:
        record = json.loads(status[1])
        tok_s = f", {record['tokens_per_sec']:,.0f} tok/s" if record.get("tokens_per_sec") else ""
        print(
            f"  session {record['started']} -> {record['ended']} UTC: steps {record['from_step']:,} -> "
            f"{record['to_step'] or 0:,} in {record['train_hours']:.2f} h on {record['gpus']} GPU(s){tok_s}"
        )
    return status[0] if status else ""


def main() -> None:
    fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=fmt)
    parser.add_argument("notebook", help="the training notebook, as <user>/<notebook-slug>")
    parser.add_argument("--run", default=RUN, help="run name (folder under runs/)")
    parser.add_argument("--dest", type=Path, help="local run folder (default: runs/<run>)")
    parser.add_argument("--keep", type=int, default=2, help="step checkpoints to keep here")
    parser.add_argument("--kaggle", default="kaggle", help="the Kaggle command-line tool")
    args = parser.parse_args()
    local_dir = args.dest or Path("runs") / args.run
    if shutil.which(args.kaggle) is None:
        parser.error(f"{args.kaggle!r} not found: pip install kaggle (and an API token, see --help)")

    active, words = session_active(args.kaggle, args.notebook)
    print(words)
    if active:
        raise SystemExit("the session hasn't ended yet: run this again after it has (nothing copied)")
    download = local_dir.parent / f".kaggle-download-{args.run}"
    shutil.rmtree(download, ignore_errors=True)  # never mix two sessions' files
    download.mkdir(parents=True)
    t0 = time.perf_counter()
    print(f"downloading the output of {args.notebook} to {download} ...", flush=True)
    subprocess.run([args.kaggle, "kernels", "output", args.notebook, "-p", str(download)], check=True)
    size = sum(p.stat().st_size for p in download.rglob("*") if p.is_file())
    print(f"  {size / 1e9:.2f} GB in {time.perf_counter() - t0:.0f} s")
    status = sync(download, args.run, local_dir, args.keep)
    shutil.rmtree(download, ignore_errors=True)
    print(f"\nthe session says: {status}\ncopied and checked: {local_dir.resolve()}")


if __name__ == "__main__":
    main()
