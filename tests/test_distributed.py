"""Tests for data parallelism (DDP, ``vulpes/train/distributed.py``) on CPU: 2 processes over Gloo.

Kaggle's two T4s run the same code with NCCL; here two CPU processes stand in for them
(Vulpes Flos 1.0, milestone K2). What has to hold:

* two processes train the same run as one: the same sequences per step in the same order,
  so the same weights up to float rounding (the gradient sums run in another order);
* both processes end with identical weights, and only process 0 writes anything;
* a checkpoint made by two processes resumes on one (Kaggle -> the desktop), and the other way.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from vulpes.config import ConfigError
from vulpes.data import PretrainLoader
from vulpes.train import SFTConfig, SFTTrainer, TrainConfig, Trainer, distributed, pretrain
from vulpes.train import checkpoint as ckpt


def make_cfg(tiny_run, out_dir: Path, **overrides) -> dict:
    base = {
        "run_name": "run",
        "out_dir": str(out_dir),
        "model": str(tiny_run.model_yaml),
        "data_dir": str(tiny_run.data_dir),
        "tokenizer": str(tiny_run.tokenizer),
        "seq_len": 32,
        "micro_batch": 2,
        "grad_accum": 4,
        "max_steps": 6,
        "optim": {"lr": 3e-3, "min_lr": 3e-4, "warmup_steps": 2, "schedule": "wsd", "decay_fraction": 0.5},
        "device": "cpu",
        "compile": False,
        "eval_every": 3,
        "eval_batches": 4,
        "save_every": 3,
        "log_every": 1,
    }
    return {**base, **overrides}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def train_until(trainer: Trainer, stop_at: int | None) -> dict:
    """Train; with ``stop_at``, stop (and save) after that step, as Ctrl-C or ``--stop-after`` would."""
    real_lr_at = pretrain.lr_at

    def lr_at(step: int, *args: object) -> float:
        if stop_at is not None and step == stop_at - 1:
            trainer.request_stop("test")
        return real_lr_at(step, *args)

    pretrain.lr_at = lr_at
    try:
        return trainer.train()
    finally:
        pretrain.lr_at = real_lr_at


def _worker(rank: int, world: int, port: int, cfg: dict, resume: bool, stop_at: int | None, out: str) -> None:
    """One process of a ``torchrun --nproc_per_node world`` job (spawned by ``run_ddp``)."""
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        USE_LIBUV="0",  # PyTorch's Windows builds have no libuv for the rendezvous store
    )
    torch.set_num_threads(1)  # two processes on one CPU
    try:
        trainer = Trainer(TrainConfig.from_dict(cfg), resume=resume, log=lambda _msg: None)
        summary = train_until(trainer, stop_at)
        params = {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}
        torch.save({"summary": summary, "params": params, "loader": trainer.loader.state_dict()}, out % rank)
    finally:
        distributed.cleanup()


def run_ddp(
    cfg: dict, tmp_path: Path, world: int = 2, resume: bool = False, stop_at: int | None = None
) -> list[dict]:
    out = str(tmp_path / f"{cfg['run_name']}-rank%d.pt")
    mp.spawn(_worker, args=(world, free_port(), cfg, resume, stop_at, out), nprocs=world)
    return [torch.load(out % r, weights_only=True) for r in range(world)]


def run_single(cfg: dict, resume: bool = False, stop_at: int | None = None) -> Trainer:
    trainer = Trainer(TrainConfig.from_dict(cfg), resume=resume, log=lambda _msg: None)
    train_until(trainer, stop_at)
    return trainer


def val_losses(run_dir: Path) -> dict[int, float]:
    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return {r["step"]: r["val/loss"] for r in map(json.loads, lines) if "val/loss" in r}


@pytest.mark.slow
def test_two_processes_train_the_same_run_as_one(tiny_run, tmp_path):
    single = run_single(make_cfg(tiny_run, tmp_path, run_name="single"))
    ranks = run_ddp(make_cfg(tiny_run, tmp_path, run_name="ddp"), tmp_path)

    # Both processes hold the same model, and it is the single-process model up to rounding.
    p0, p1 = ranks[0]["params"], ranks[1]["params"]
    assert all(torch.equal(p0[k], p1[k]) for k in p0)
    ps = single.model.state_dict()
    assert all(torch.allclose(ps[k], p0[k], atol=1e-5, rtol=1e-4) for k in ps)
    # The same windows were read (each process walked the whole order, skipping the other's).
    assert ranks[0]["loader"] == ranks[1]["loader"] == single.loader.state_dict()
    assert ranks[0]["summary"]["finished"] and ranks[0]["summary"]["step"] == 6
    # Validation is split between the processes and summed: the same numbers as on one.
    v_single, v_ddp = val_losses(single.run_dir), val_losses(tmp_path / "ddp")
    assert v_single.keys() == v_ddp.keys() == {0, 3, 6}
    assert all(v_single[s] == pytest.approx(v_ddp[s], rel=1e-5) for s in v_single)

    # Only process 0 writes: one header line in the log, one record per step in the metrics.
    log = (tmp_path / "ddp" / "train.log").read_text(encoding="utf-8")
    assert log.count("run ddp | model") == 1 and "2 x cpu (DDP)" in log
    train_steps = [json.loads(line)["step"] for line in (tmp_path / "ddp" / "metrics.jsonl").open()]
    assert sorted(s for s in train_steps if s) == sorted([1, 2, 3, 3, 4, 5, 6, 6])  # train + val records
    assert [s for s, _ in ckpt.list_checkpoints(tmp_path / "ddp")] == [3, 6]


@pytest.mark.slow
def test_a_ddp_run_moves_to_one_process_and_back(tiny_run, tmp_path):
    # Kaggle's 2 GPUs (one session) -> the desktop (1 process, 1 x 8) -> Kaggle again, one run.
    cfg = make_cfg(tiny_run, tmp_path, max_steps=10, eval_every=0, save_every=0)
    assert not run_ddp(cfg, tmp_path, stop_at=3)[0]["summary"]["finished"]
    assert ckpt.latest_checkpoint(tmp_path / "run").name == "step_000003.pt"

    moved = {**cfg, "micro_batch": 1, "grad_accum": 8, "eval_batches": 8}  # same 8 sequences per step
    assert run_single(moved, resume=True, stop_at=6).step == 6
    ranks = run_ddp(cfg, tmp_path, resume=True)
    assert ranks[0]["summary"]["finished"] and ranks[0]["summary"]["step"] == 10

    # The same as one uninterrupted single-process run of 10 steps, up to rounding.
    straight = run_single({**cfg, "run_name": "straight", "max_steps": 10})
    ps, pd = straight.model.state_dict(), ranks[0]["params"]
    assert all(torch.allclose(ps[k], pd[k], atol=1e-5, rtol=1e-4) for k in ps)
    assert ranks[0]["loader"] == straight.loader.state_dict()


def test_grad_accum_must_split_evenly_between_processes(tiny_run, tmp_path, monkeypatch):
    monkeypatch.setattr(distributed, "setup", lambda device: distributed.DistInfo(0, 0, 2))
    with pytest.raises(ConfigError, match="grad_accum 3 must split evenly over 2 processes"):
        Trainer(TrainConfig.from_dict(make_cfg(tiny_run, tmp_path, grad_accum=3)), log=lambda _msg: None)


def test_from_env_reads_torchrun_variables(monkeypatch):
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    assert distributed.from_env() == distributed.DistInfo() and not distributed.from_env().enabled
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    info = distributed.from_env()
    assert info == distributed.DistInfo(1, 1, 2) and info.enabled and not info.is_main
    # Without a process group every collective is the identity, so single-process code pays nothing.
    single = distributed.DistInfo()
    cpu = torch.device("cpu")
    assert distributed.sum_([1.5, 2.0], cpu, single) == [1.5, 2.0]
    assert distributed.any_(True, cpu, single) and not distributed.all_(False, cpu, single)
    distributed.barrier(cpu, single)


def test_fine_tuning_refuses_torchrun(tiny_run, tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    cfg = SFTConfig(run_name="sft", init_from=str(tiny_run.checkpoint), out_dir=str(tmp_path))
    with pytest.raises(ValueError, match="one GPU"):
        SFTTrainer(cfg, log=lambda _msg: None)


@pytest.mark.parametrize("batch", [1, 3, 7])
def test_loader_skip_moves_exactly_like_next(tiny_run, batch):
    """With several GPUs each process skips the windows of the others' micro-batches."""
    read = PretrainLoader(tiny_run.data_dir, "train", batch, 32, seed=5)
    skipped = PretrainLoader(tiny_run.data_dir, "train", batch, 32, seed=5)
    n = read.num_windows * 2 + 5  # across two epoch boundaries
    for _ in range(n):
        next(read)
    skipped.skip(n * batch)
    assert skipped.state_dict() == read.state_dict()
    assert all(torch.equal(a, b) for a, b in zip(next(skipped), next(read), strict=True))
