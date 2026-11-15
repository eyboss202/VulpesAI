# Training Vulpes Flos 1.0 on Kaggle

The Vulpes Flos 1.0 Base run (`base`, 348M parameters, 10–15B tokens; [`PLAN-FLOS.md`](PLAN-FLOS.md))
on Kaggle's free **2× T4**, one 12-hour session at a time, at **no cost**. No session is long enough
for the run, and none needs to be: every session resumes the previous one bit for bit (0.1 already
moved between machines this way). Three files do the work:

| File | Where | What it does |
|---|---|---|
| `scripts/kaggle/vulpes-flos-train.ipynb` | Kaggle | the notebook: clones this repository at `COMMIT`, installs it (Kaggle's PyTorch is kept), runs `session.py` |
| `scripts/kaggle/session.py` | Kaggle | finds the shards and the newest checkpoint among the inputs, copies the run to `/kaggle/working`, trains with `torchrun` on both GPUs until ~45 min before the 12-hour limit, then writes `SESSION_DONE`, `SHA256SUMS` and a line in `kaggle-sessions.jsonl` |
| `scripts/kaggle/sync.py` | PC | after a session, downloads its output with the Kaggle API, checks the checkpoints against `SHA256SUMS` and moves them into `runs/vulpes-flos-1.0-base/` |

Status (2026-11-14): **written and tested on CPU** (`tests/test_kaggle.py`: two sessions hand a run
over and the PC copies it). Not yet run on Kaggle: that is the rest of K4, below.

## Once: the account, the data and the notebook

1. **Account.** Verify the phone number (Settings → Phone verification): without it there are no
   GPUs and no internet in notebooks. Check the weekly GPU quota on the same page (the plan assumes 30 h).
2. **API token** for the PC: Settings → API → Create New Token saves `kaggle.json`; put it in
   `%USERPROFILE%\.kaggle\kaggle.json` (Windows) or `~/.kaggle/kaggle.json`. Then `pip install kaggle`.
   The token is a password: it never goes in git.
3. **The shards as a private dataset** (K3's `data/pretrain/fineweb_edu_10bt/`, ~19 GB):

   ```powershell
   kaggle datasets init -p data/pretrain/fineweb_edu_10bt
   # edit data/pretrain/fineweb_edu_10bt/dataset-metadata.json: "title": "vulpes-fineweb-edu-10bt",
   #   "id": "<user>/vulpes-fineweb-edu-10bt"
   kaggle datasets create -p data/pretrain/fineweb_edu_10bt     # private unless --public
   ```

   The upload's time depends on the connection's upload speed; if it is far too slow, decision D4's
   fallback builds the same shards in a Kaggle CPU notebook (no GPU quota).
4. **The notebook.** Kaggle → Code → New Notebook → File → Import Notebook →
   `scripts/kaggle/vulpes-flos-train.ipynb`. Name it `vulpes-flos-train`. Settings: **Accelerator
   GPU T4 ×2** (never the P100), **Internet on**. Add Input → Your Datasets → the shards. If the
   repository is private, add a GitHub token as the secret `GITHUB_TOKEN` (Add-ons → Secrets).
5. **Pin the code.** Set `COMMIT` in the second cell to the commit to train with (a hash, after
   pushing it). Every session of the run should use the same code; each session records the commit
   it ran in `kaggle-sessions.jsonl`.

## Every session

1. Kaggle → the notebook → **Save Version → Save & Run All**. The browser can be closed: the
   session runs in the background for up to 12 hours.
2. From the second session on, the notebook's inputs must include **its own output** (Add Input →
   Your Work → `vulpes-flos-train`), so `session.py` finds the newest checkpoint. With no checkpoint
   in the inputs it starts a new run, so check the first lines of the log: `run: ... is at step N`.
3. On the PC, when the session has ended:

   ```powershell
   python scripts/kaggle/sync.py <user>/vulpes-flos-train
   python scripts/train_status.py --run runs/vulpes-flos-1.0-base     # where the run and its schedule are
   ```

   This copy is the run's insurance: if Kaggle is lost, the run continues anywhere from it.
4. Use all the quota: 2–3 sessions back to back each week.

When to start the decay is decision D2: `train_status.py` shows the `max_steps` that starts it from
the newest checkpoint; pass it once as `EXTRA = "max_steps=N"` in the notebook (and then set it
back to `""`: from then on the saved config has it).

## What `session.py` does, and why

- **Inputs are read-only**, so the run is copied to `/kaggle/working/runs/vulpes-flos-1.0-base/`
  before training: the newest step checkpoint, `best.pt` and the small files (config, logs,
  metrics, TensorBoard). Older checkpoints, half-written `.tmp` files and the previous session's
  hand-over files are not carried over. Only `/kaggle/working` (20 GB) is saved as output.
- **Disk.** The run keeps 2 step checkpoints + `best.pt`, 4.2 GB each; while a third is being written
  (before the oldest is pruned) that is ~17 GB of the 20 GB.
- **Time.** `--stop-after` is 12 h minus the 45-minute `--margin` minus the time the notebook has
  already used (`--started-at`, set in the first cell). Training then finishes the step, saves and
  exits; the margin covers that step, an evaluation that may be running, and the SHA-256s.
- **Exit code.** 0 when the run is in a good state (finished, or stopped and saved), so Kaggle keeps
  the output; 1 after a failure (the checkpoints before it are intact: saves are atomic). Either
  way `SESSION_DONE` says what happened.
- **One run, one writer.** Never run two sessions of the same run at once: both would continue from
  the same checkpoint and the second hand-over would silently drop the first one's steps.

## Still to do in K4, on Kaggle

The plan leaves these to be measured with the account (they need ~1 h of GPU quota):

| What | How | Decides |
|---|---|---|
| The hand-over | a short trial run kept apart from the real one, `EXTRA = "--run k4-trial --hours 0.25"`, twice: the second with the first one's output as input must say `run: ... is at step N` and `resumed from` | whether the notebook's own output is enough, or the checkpoint must go to a private dataset via the API (the plan's other option) |
| Input mount paths | the session log's `data:` and `run:` lines | nothing, if the search finds them (it looks everywhere under `/kaggle/input`) |
| `sync.py` | run it after that session | download time of ~12.6 GB, and whether the CLI keeps the folders (`sync.py` handles both) |
| Upload of the shards | step 3 above | whether D4's CPU-notebook fallback is needed |

Then K5 measures speed (micro-batch, `torch.compile`, checkpointing) and K6 is the two-session pilot.
