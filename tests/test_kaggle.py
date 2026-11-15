"""Tests for training on Kaggle (K4): scripts/kaggle/session.py, scripts/kaggle/sync.py and the notebook."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "scripts" / "kaggle"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"kaggle_{name}", KAGGLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def touch(path: Path, data: bytes = b"") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def fake_inputs(root: Path) -> Path:
    """A /kaggle/input with the shards, the previous session's output and an older copy of the run."""
    touch(root / "vulpes-shards" / "meta.json", json.dumps({"format": "vulpes-shards"}).encode())
    touch(root / "something-else" / "meta.json", json.dumps({"format": "other"}).encode())
    new = root / "train-notebook" / "runs" / "vulpes-flos-1.0-base"
    for name in (
        "config.yaml",
        "metrics.jsonl",
        "step_000040.pt",
        "step_000080.pt",
        "best.pt",
        "tb/events.1",
    ):
        touch(new / name, name.encode())
    for name in ("step_000120.pt.tmp", "SESSION_DONE", "SHA256SUMS"):
        touch(new / name)  # a half-written save and the previous hand-over: never carried on
    old = root / "older-dataset" / "vulpes-flos-1.0-base"
    touch(old / "config.yaml")
    touch(old / "step_000040.pt")
    touch(root / "elsewhere" / "other-run" / "config.yaml")
    touch(root / "elsewhere" / "other-run" / "step_000400.pt")
    return new


def test_session_finds_the_data_and_the_newest_copy_of_the_run(tmp_path):
    session = load("session")
    new = fake_inputs(tmp_path)
    assert session.find_datasets(tmp_path) == [tmp_path / "vulpes-shards"]
    assert session.find_runs(tmp_path, "vulpes-flos-1.0-base") == [
        (40, tmp_path / "older-dataset" / "vulpes-flos-1.0-base"),
        (80, new),
    ]
    assert session.find_runs(tmp_path / "missing", "vulpes-flos-1.0-base") == []
    wanted = [p.as_posix() for p in session.files_to_restore(new)]
    assert wanted == ["best.pt", "config.yaml", "metrics.jsonl", "step_000080.pt", "tb/events.1"]

    dest = tmp_path / "working" / "runs" / "vulpes-flos-1.0-base"
    assert session.restore(new, dest, log=lambda _: None) == sum(len(n) for n in wanted)
    assert sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()) == wanted
    assert (dest / "step_000080.pt").read_bytes() == b"step_000080.pt"
    assert session.restore(new, dest) == 0  # already there: nothing copied again


def test_session_command_and_time_budget(tmp_path):
    session = load("session")
    run_dir, data = tmp_path / "runs" / "r", tmp_path / "shards"
    two = session.train_command(2, run_dir, True, "c.yaml", data, 10.25, ["max_steps=80"])
    assert two[1:5] == ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"]
    assert two[5:] == [
        "scripts/train.py", "--resume", str(run_dir), "--stop-after", "10.25",
        f"data_dir={data}", "max_steps=80",
    ]  # fmt: skip
    one = session.train_command(1, run_dir, False, "c.yaml", data, 2, [])
    assert one[1:] == [
        "scripts/train.py", "--config", "c.yaml", "--stop-after", "2",
        f"data_dir={data}", f"out_dir={run_dir.parent}", "run_name=r",
    ]  # fmt: skip
    # 12 h, minus the 45 min margin, minus the hour the notebook has already run.
    assert session.stop_after_hours(started_at=1000.0, margin=0.75, now=1000.0 + 3600) == 10.25


def test_notebook_runs_the_session_script():
    nb = json.loads((KAGGLE / "vulpes-flos-train.ipynb").read_text(encoding="utf-8"))
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "scripts/kaggle/session.py --started-at {SESSION_START}" in code
    assert 'CODE = "/tmp/VulpesAI"' in code  # outside /kaggle/working: the code is not saved as output
    assert "--no-deps" in code  # Kaggle's own PyTorch is kept


def sync_dir(
    tmp_path: Path, name: str, files: dict[str, bytes], sums: dict[str, bytes] | None = None
) -> Path:
    """A downloaded session output: runs/<run>/ with files, SESSION_DONE and SHA256SUMS."""
    import hashlib

    run = tmp_path / name / "runs" / "flos"
    for rel, data in files.items():
        touch(run / rel, data)
    record = {"started": "a", "ended": "b", "train_hours": 1.0, "from_step": 0, "to_step": 8, "gpus": 2}
    touch(run / "SESSION_DONE", f"stopped and saved\n{json.dumps(record)}\n".encode())
    sums = files if sums is None else sums
    lines = "".join(f"{hashlib.sha256(d).hexdigest()}  {n}\n" for n, d in sums.items() if n.endswith(".pt"))
    touch(run / "SHA256SUMS", lines.encode())
    return tmp_path / name


def test_sync_checks_the_session_and_never_overwrites_a_newer_copy(tmp_path):
    sync = load("sync")
    pc = tmp_path / "pc" / "flos"
    files = {"config.yaml": b"c", "step_000008.pt": b"eight", "best.pt": b"best", "tb/e": b"e"}
    assert sync.sync(sync_dir(tmp_path, "s1", files), "flos", pc, keep=2) == "stopped and saved"
    assert sorted(p.relative_to(pc).as_posix() for p in pc.rglob("*") if p.is_file()) == [
        "SESSION_DONE", "SHA256SUMS", "best.pt", "config.yaml", "step_000008.pt", "tb/e"
    ]  # fmt: skip

    bad = sync_dir(tmp_path, "s2", {**files, "step_000016.pt": b"x"}, {"step_000016.pt": b"y"})
    with pytest.raises(SystemExit, match="SHA-256 differs"):
        sync.sync(bad, "flos", pc, keep=2)
    assert not (pc / "step_000016.pt").exists()  # nothing moved before the check

    unlisted = sync_dir(tmp_path, "s3", {**files, "step_000016.pt": b"x"}, {"best.pt": b"best"})
    with pytest.raises(SystemExit, match="does not list step_000016.pt"):
        sync.sync(unlisted, "flos", pc, keep=2)

    for step in (24, 32):
        sync.sync(sync_dir(tmp_path, f"s{step}", {**files, f"step_{step:06d}.pt": b"n"}), "flos", pc, keep=2)
    assert sorted(p.name for p in pc.glob("step_*.pt")) == ["step_000024.pt", "step_000032.pt"]

    touch(pc / "step_000040.pt")  # the run went on here, on the PC
    with pytest.raises(SystemExit, match="ahead of the session"):
        sync.sync(sync_dir(tmp_path, "s5", {**files, "step_000036.pt": b"n"}), "flos", pc, keep=2)

    flat = tmp_path / "flat"  # an output downloaded without its folders
    for rel, data in {"config.yaml": b"c", "step_000048.pt": b"f"}.items():
        touch(flat / rel, data)
    shutil.copy(sync_dir(tmp_path, "s6", {"step_000048.pt": b"f"}) / "runs" / "flos" / "SHA256SUMS", flat)
    touch(flat / "SESSION_DONE", b"finished\n")
    assert sync.sync(flat, "flos", pc, keep=2) == "finished"


@pytest.mark.slow
def test_two_sessions_train_hand_over_and_sync(tiny_run, tmp_path, monkeypatch):
    """Session 1 starts the run, session 2 continues it from session 1's output, the PC copies it."""
    monkeypatch.chdir(tmp_path)  # session.main moves to the repository; this puts it back afterwards
    session, sync = load("session"), load("sync")
    cfg = {
        "run_name": "flos-test",
        "model": str(tiny_run.model_yaml),
        "data_dir": "set-by-the-session",
        "tokenizer": str(tiny_run.tokenizer),
        "seq_len": 64,
        "micro_batch": 2,
        "grad_accum": 2,
        "max_steps": 4,
        "optim": {"lr": 3e-3, "min_lr": 3e-4, "warmup_steps": 1},  # cosine: may be lengthened anywhere
        "device": "cpu",
        "compile": False,
        "eval_every": 2,
        "eval_batches": 2,
        "save_every": 2,
        "log_every": 1,
    }
    config = tmp_path / "flos-test.yaml"
    config.write_text(yaml.safe_dump(cfg))
    common = ["--run", "flos-test", "--config", str(config), "--gpus", "1", "--hours", "1"]

    first_in, first_out = tmp_path / "in1", tmp_path / "out1"
    first_in.mkdir()
    assert (
        session.main(
            [*common, "--input", str(first_in), "--working", str(first_out), "--data", str(tiny_run.data_dir)]
        )
        == 0
    )
    run1 = first_out / "runs" / "flos-test"
    assert (run1 / "SESSION_DONE").read_text().startswith("finished")
    assert set(sync.read_sums(run1 / "SHA256SUMS")) == {"step_000004.pt", "best.pt"}

    # Session 2: the shards and session 1's output are its inputs; the run is lengthened.
    second_in, second_out = tmp_path / "in2", tmp_path / "out2"
    shutil.copytree(tiny_run.data_dir, second_in / "shards")
    shutil.copytree(first_out, second_in / "train-notebook")
    assert (
        session.main([*common, "--input", str(second_in), "--working", str(second_out), "max_steps=6"]) == 0
    )
    run2 = second_out / "runs" / "flos-test"
    sessions = [json.loads(line) for line in (run2 / "kaggle-sessions.jsonl").read_text().splitlines()]
    assert [(s["from_step"], s["to_step"], s["exit_code"]) for s in sessions] == [(0, 4, 0), (4, 6, 0)]
    assert "resumed from" in (run2 / "train.log").read_text(encoding="utf-8")

    pc = tmp_path / "pc" / "flos-test"
    trained = (run2 / "step_000006.pt").read_bytes()
    assert sync.sync(second_out, "flos-test", pc, keep=2).startswith("finished")
    assert (pc / "step_000006.pt").read_bytes() == trained  # moved out of the download, checked
    assert sorted(p.name for p in pc.glob("step_*.pt")) == ["step_000004.pt", "step_000006.pt"]
