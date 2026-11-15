"""One Kaggle session of a Vulpes run: find the data and the newest checkpoint, train, hand over (K4).

Vulpes Flos 1.0 trains in many 12-hour Kaggle sessions (docs/PLAN-FLOS.md §7, docs/KAGGLE.md).
Each session is a notebook version ("Save & Run All") that clones this repository and runs::

    python scripts/kaggle/session.py --started-at <when the notebook started>

What it does:

1. **Data.** Finds the prepared shards (a ``meta.json`` of format ``vulpes-shards``) among the
   notebook's inputs under ``/kaggle/input``: the private dataset built in K3.
2. **The run so far.** Looks for ``runs/<run>/`` with step checkpoints in the inputs (the previous
   session's output, attached as an input, or a dataset holding it) and copies the newest step
   checkpoint, ``best.pt`` and the small files (config, logs, metrics, TensorBoard) to
   ``/kaggle/working/runs/<run>/``, the only place whose files outlive the session. Inputs are
   read-only, so the run can't continue where it is. No run anywhere: this session starts it.
3. **Training.** ``torchrun --standalone --nproc_per_node <GPUs> scripts/train.py --resume ...``
   (or ``--config configs/train/flos-kaggle.yaml`` the first time) with ``data_dir`` set to the
   dataset and ``--stop-after`` set so that training stops, saved, ``--margin`` hours before
   Kaggle's 12-hour limit, counted from ``--started-at``.
4. **Hand-over.** Writes ``SESSION_DONE`` (what happened) and ``SHA256SUMS`` (the newest step
   checkpoint and ``best.pt``) in the run directory, and one line per session to
   ``kaggle-sessions.jsonl`` (steps, hours, tokens/s, GPUs, commit). The notebook's output is
   then the next session's input, and ``scripts/kaggle/sync.py`` on the PC checks its copy
   against ``SHA256SUMS``.

It exits 0 whenever the run directory is in a good state (finished, or stopped and saved), so
Kaggle always keeps the session's output; a failed training is reported in ``SESSION_DONE``
and with exit code 1 (the previous checkpoints are still there: checkpoints are atomic).

Usage (in the notebook; everything has a default)::

    python scripts/kaggle/session.py --started-at 1763150000
    python scripts/kaggle/session.py --hours 2 max_steps=80      # K6's pilot: a short session
    python scripts/kaggle/session.py --dry-run                   # show what it would do

Only the standard library and PyTorch (to count the GPUs) are needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = "vulpes-flos-1.0-base"
CONFIG = "configs/train/flos-kaggle.yaml"
INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
SESSION_LIMIT_H = 12.0  # Kaggle stops a GPU session after 12 hours
DONE, SUMS, SESSIONS = "SESSION_DONE", "SHA256SUMS", "kaggle-sessions.jsonl"
SHARD_FORMATS = {"vulpes-shards", "aupus-shards"}  # aupus-*: datasets made before the rename
STEP = re.compile(r"^step_(\d{6,})\.pt$")
#: Never carried from one session to the next: half-written files and the previous hand-over.
NOT_RESTORED = re.compile(r"(\.tmp|\.part)$|^(SESSION_DONE|SHA256SUMS)$")


def newest_step(names: Iterable[str]) -> tuple[int, str] | None:
    """``(step, name)`` of the newest ``step_NNNNNN.pt`` among ``names``."""
    steps = sorted((int(m.group(1)), n) for n in names if (m := STEP.match(n)))
    return steps[-1] if steps else None


def find_datasets(root: Path) -> list[Path]:
    """Directories under ``root`` holding prepared shards (``meta.json`` of a shard format)."""
    found = []
    for meta in sorted(root.rglob("meta.json")) if root.is_dir() else []:
        try:
            fmt = json.loads(meta.read_text(encoding="utf-8")).get("format")
        except (OSError, ValueError):
            continue
        if fmt in SHARD_FORMATS:
            found.append(meta.parent)
    return found


def find_runs(root: Path, run: str) -> list[tuple[int, Path]]:
    """``(newest step, dir)`` of every copy of run ``run`` under ``root``, oldest first.

    A copy is a directory named ``run`` with a ``config.yaml`` and at least one step checkpoint.
    """
    found = []
    for config in root.rglob("config.yaml") if root.is_dir() else []:
        d = config.parent
        if d.name == run and (newest := newest_step(p.name for p in d.iterdir())):
            found.append((newest[0], d))
    return sorted(found)


def files_to_restore(src: Path) -> list[Path]:
    """Paths (relative to ``src``) a session needs: all but older step checkpoints and temp files."""
    newest = newest_step(p.name for p in src.iterdir())
    wanted = []
    for p in sorted(src.rglob("*")):
        if not p.is_file() or NOT_RESTORED.search(p.name):
            continue
        if STEP.match(p.name) and p.parent == src and (newest is None or p.name != newest[1]):
            continue
        wanted.append(p.relative_to(src))
    return wanted


def restore(src: Path, dest: Path, log: Callable[[str], None] = print) -> int:
    """Copy the run in ``src`` (read-only input) to ``dest``; returns the bytes copied."""
    copied = 0
    for rel in files_to_restore(src):
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        source = src / rel
        size = source.stat().st_size
        if target.is_file() and target.stat().st_size == size:
            continue
        t0 = time.perf_counter()
        tmp = target.with_name(target.name + ".part")
        shutil.copy2(source, tmp)
        if tmp.stat().st_size != size:
            raise OSError(f"{rel}: copied {tmp.stat().st_size} bytes, expected {size}")
        os.replace(tmp, target)
        copied += size
        if size > 50e6:
            log(f"  copied {rel.as_posix()} ({size / 1e9:.2f} GB, {time.perf_counter() - t0:.0f} s)")
    return copied


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(16 << 20):
            h.update(block)
    return h.hexdigest()


def write_sums(run_dir: Path) -> dict[str, str]:
    """``SHA256SUMS`` (``sha256sum`` format) for the newest step checkpoint and ``best.pt``."""
    names = [n for n in ((newest_step(p.name for p in run_dir.iterdir()) or (0, None))[1], "best.pt") if n]
    sums = {n: sha256(run_dir / n) for n in names if (run_dir / n).is_file()}
    (run_dir / SUMS).write_text("".join(f"{d}  {n}\n" for n, d in sums.items()), encoding="utf-8")
    return sums


def stop_after_hours(started_at: float, margin: float, now: float | None = None) -> float:
    """Hours of training left in this session: Kaggle's limit, minus time already used, minus the margin."""
    used = ((time.time() if now is None else now) - started_at) / 3600
    return round(SESSION_LIMIT_H - margin - used, 3)


def train_command(
    nproc: int,
    run_dir: Path,
    resume: bool,
    config: str,
    data_dir: Path,
    stop_after: float,
    overrides: list[str],
) -> list[str]:
    """The training command: ``torchrun`` on several GPUs, plain ``python`` on one."""
    if nproc > 1:
        launcher = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={nproc}",
        ]
    else:
        launcher = [sys.executable]
    start = ["--resume", str(run_dir)] if resume else ["--config", config]
    new_run = [f"out_dir={run_dir.parent}", f"run_name={run_dir.name}"]  # the config names the real run
    paths = [f"data_dir={data_dir}", *([] if resume else new_run)]
    return [*launcher, "scripts/train.py", *start, "--stop-after", f"{stop_after:g}", *paths, *overrides]


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def last_metrics(run_dir: Path) -> dict:
    """The newest step's train record and the newest val loss in ``metrics.jsonl``."""
    out: dict = {}
    path = run_dir / "metrics.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
        record = json.loads(line)
        if "train/loss" in record:
            out["step"], out["train_loss"] = record["step"], record["train/loss"]
            out["tokens_per_sec"] = record.get("perf/tokens_per_sec", out.get("tokens_per_sec"))
        if "val/loss" in record:
            out["val_loss"] = record["val/loss"]
    return out


def count_gpus() -> int:
    try:
        import torch
    except ImportError:
        return 0
    return torch.cuda.device_count()


def main(argv: list[str] | None = None) -> int:
    fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=fmt)
    parser.add_argument("--run", default=RUN, help="run name (folder under runs/)")
    parser.add_argument("--config", default=CONFIG, help="config of a new run (when no checkpoint is found)")
    parser.add_argument("--input", type=Path, default=INPUT, help="where Kaggle mounts the notebook's inputs")
    parser.add_argument("--working", type=Path, default=WORKING, help="the directory Kaggle saves as output")
    parser.add_argument("--data", type=Path, help="the prepared shards (default: found under --input)")
    parser.add_argument("--started-at", type=float, help="when the session started (Unix time; default: now)")
    parser.add_argument(
        "--margin", type=float, default=0.75, help="hours kept free before the 12-hour limit (default 0.75)"
    )
    parser.add_argument("--hours", type=float, help="train for this many hours instead (a short session)")
    parser.add_argument("--gpus", type=int, help="processes to start (default: every GPU there is)")
    parser.add_argument(
        "--dry-run", action="store_true", help="find everything, print the command, train nothing"
    )
    parser.add_argument("overrides", nargs="*", metavar="key=value", help="passed on to scripts/train.py")
    args = parser.parse_args(argv)
    started_at = args.started_at or time.time()
    os.chdir(ROOT)  # the configs name the tokenizer relative to the repository

    data = args.data
    if data is None:
        found = find_datasets(args.input)
        if len(found) != 1:
            listed = "".join(f"\n  {d}" for d in found) or " none"
            parser.error(f"expected one prepared dataset under {args.input}, found{listed}: pass --data")
        data = found[0]
    run_dir = args.working / "runs" / args.run
    here = newest_step(p.name for p in run_dir.iterdir()) if run_dir.is_dir() else None
    inputs = find_runs(args.input, args.run)
    print(f"data: {data}")
    if here and (not inputs or inputs[-1][0] <= here[0]):
        print(f"run: continuing {run_dir} from {here[1]} (already in this session's output)")
    elif inputs:
        step, src = inputs[-1]
        print(f"run: {src} is at step {step:,}; copying it to {run_dir}")
        for other_step, other in inputs[:-1]:
            print(f"  (older copy ignored: {other}, step {other_step:,})")
        if not args.dry_run:
            t0 = time.perf_counter()
            gb = restore(src, run_dir) / 1e9
            print(f"  restored {gb:.2f} GB in {time.perf_counter() - t0:.0f} s")
    else:
        print(f"run: no checkpoint of {args.run} in {args.input}: this session starts it from {args.config}")
    resume = bool(here or inputs)
    for stale in (DONE, SUMS):
        (run_dir / stale).unlink(missing_ok=True)  # the previous session's hand-over, not this one's

    hours = args.hours if args.hours is not None else stop_after_hours(started_at, args.margin)
    if hours <= 0:
        parser.error(f"no time left in this session ({hours:g} h): check --started-at and --margin")
    nproc = args.gpus if args.gpus is not None else count_gpus()
    if nproc < 1:
        parser.error("no GPU: pick the 'GPU T4 x2' accelerator in the notebook's settings")
    cmd = train_command(nproc, run_dir, resume, args.config, data, hours, args.overrides)
    print(f"training {hours:g} h on {nproc} process(es):\n  {' '.join(cmd)}", flush=True)
    if args.dry_run:
        return 0

    before = last_metrics(run_dir)
    t0 = time.time()
    code = subprocess.run(cmd, cwd=ROOT).returncode
    after = last_metrics(run_dir)
    if code == 0:
        status = "finished: the run reached max_steps"
    elif code == 130:
        status = "stopped and saved before the end: the next session continues it"
    else:
        status = f"training failed with exit code {code}: see train.log; the last checkpoint is intact"
    sums = write_sums(run_dir) if run_dir.is_dir() else {}
    record = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(started_at)),
        "ended": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "train_hours": round((time.time() - t0) / 3600, 3),
        "from_step": before.get("step", 0),
        "to_step": after.get("step"),
        "train_loss": after.get("train_loss"),
        "val_loss": after.get("val_loss"),
        "tokens_per_sec": after.get("tokens_per_sec"),
        "gpus": nproc,
        "exit_code": code,
        "commit": git_commit(),
        "overrides": args.overrides,
    }
    if run_dir.is_dir():
        with open(run_dir / SESSIONS, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        (run_dir / DONE).write_text(f"{status}\n{json.dumps(record)}\n", encoding="utf-8")
    print(f"\n{status}")
    for name, digest in sums.items():
        print(f"  {name}  {digest}")
    return 0 if code in (0, 130) else 1


if __name__ == "__main__":
    sys.exit(main())
