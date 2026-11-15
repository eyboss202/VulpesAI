# Vulpes Flos 1.0 — Plan

Status: **approved** (2026-11-14), with one rule above all: get the most out of Kaggle's free tier.
The full plan, in Portuguese, is [`Vulpes-Flos-1.0-Plano.pdf`](Vulpes-Flos-1.0-Plano.pdf); this file is
its working copy in the repository, like [`PLAN.md`](PLAN.md) is for Vulpes 0.1, and it records every
measurement and decision from here on. Until 2026-11-14 this plan was called Aupus 1.
Progress: **K1 done** (this file, 2026-11-14). **K2 done** (2026-11-14, §12): DDP, WSD and the Flos
config, tested on CPU. **K3 half done**: the download finished 2026-11-14 (all 14 files, 28.52 GB in
89 min, SHA-256 OK). **K3's shards done** 2026-11-15: 9,672,101 documents in 3 h 45 min (3 workers,
716 docs/s), 92.9% kept, **9.23B train + 46.1M val tokens** (18 GB, 93 + 1 shards); `k3_check.py`
OK: they start with exactly 0.1's documents and tokens. Left in K3: the upload to Kaggle. **K4's code done** (§13): the notebook and `scripts/kaggle/`, tested on
CPU; what needs the Kaggle account is in [`KAGGLE.md`](KAGGLE.md). Times and results below are
estimates until K5 measures them.

Vulpes Flos 1.0 follows Vulpes 0.1 (125M, trained 2026-11-10) and comes before Vulpes Lusitanus 1.0
(~1B, code and Portuguese; [its own plan](Vulpes-Lusitanus-1.0-Plano.pdf)). Everything that Flos
builds (DDP, WSD, fp16, the Kaggle workflow) carries over to Lusitanus.

## In short

Vulpes 0.1 taken as far as Kaggle's free GPUs allow: a bigger model that reads more text and
remembers more of a conversation. Same repository, plain PyTorch, the same human-written data, the
same tokenizer and the same chat.

| | Vulpes 0.1 | Vulpes Flos 1.0 |
|---|---|---|
| Parameters | 125.3M | **348.4M** (2.8×; still fits a 4 GB GPU for chat) |
| Context | 1,024 | **2,048** tokens |
| Pretraining tokens | 2.5B | **10–15B+**, for as long as the loss keeps improving |
| Cost | ~4 h on a rented A100 | **0 USD**: Kaggle's 2× T4 and, if a trial pays off, its TPUs |

**Honest expectation.** 348M is still small. Flos should write more coherent English and get more
simple questions right than 0.1, but it will still make facts up, can't program and barely speaks
Portuguese. The expected gain is real and modest (~5–10 points of HellaSwag) and takes two to three
months of GPU quota, less if the TPUs deliver.

## 1. Starting point: what Vulpes 0.1 leaves to improve

Measured (`runs/vulpes-0.1-*/eval_best.json`): Base 2.5B tokens, 9,537 steps, ~4 h on an A100 80 GB at
~167K tokens/s (138 TFLOPS); Instruct 600 SFT steps.

| | 0.1 Base | 0.1 Instruct | Random |
|---|---|---|---|
| HellaSwag (acc_norm, 10,042) | 31.3% | 31.2% | 25% |
| ARC-Easy (acc, 2,376) | 52.4% | 52.2% | 25% |
| PIQA (acc_norm, 1,838) | 61.9% | 61.0% | 50% |
| FineWeb-Edu val perplexity | 22.9 | 24.8 | 32,000 |

What more compute fixes, and so what Flos does:

1. **It is small.** 125M is GPT-2 small. At this scale growing is the cheapest gain per FLOP, and 348M
   still runs in fp32 on the desktop's GTX 1650 Super.
2. **It read little.** 2.5B tokens, 0.85 epochs of the 2.94B built from 4 of the 14 `sample-10BT` files.
3. **It forgets fast.** 1,024 tokens fill up after a few exchanges in the chat; 2,048 costs only +8% FLOP per token.
4. **Its end was fixed.** With cosine the length had to be chosen up front; with a weekly quota and
   12-hour sessions it is better to decide mid-way when to stop.

What needs a different model, and so waits for Lusitanus: code and Portuguese (a new tokenizer: today
1.85 and 2.8 characters per token, and new data), tools and an agent (new special tokens, ~1B
parameters), and 1B itself (it doesn't fit a T4, §2).

## 2. Where: what fits in Kaggle's free tier

| What | Value | What it means for Vulpes |
|---|---|---|
| GPU | 2× T4 16 GB (or one P100) | fp16 tensor cores, no bf16: fp16 autocast + GradScaler (0.1's decision 13) |
| Quota | ~30 h per week | not documented, seen in the account; reports say 30–41 h; the sums use 30 h |
| Session | up to 12 h, in the background | "Save & Run All" runs with the browser closed; `--stop-after` stops before the limit |
| Saved disk | 20 GB in `/kaggle/working` | two 4.2 GB checkpoints and `best.pt` fit |
| Data | private datasets, read-only | the ~19 GB of shards come in as a dataset under `/kaggle/input` |

| Option | FLOP | Hours on 2× T4 | Weeks | Verdict |
|---|---|---|---|---|
| Redo 0.1 (125M, 2.5B) | 2.1e18 | 17–23 | < 1 | already exists |
| Continue 0.1 to 10B tokens | 6.2e18 | 50–70 | ~2 | cheap, small gain: plan B |
| **Flos: 348M, 10B, context 2k** | **2.4e19** | **195–270** | **6.5–9** | recommended |
| Flos continued to 15B | 3.7e19 | 290–410 | 10–14 | decided (D2), while the loss still falls |
| Lusitanus Mini (321M, 10B, 4k) | 2.5e19 | 200–280 | 7–9 | needs a new tokenizer |
| Lusitanus 1.0 (1B, 50B) | 3.8e20 | 3,000–4,200 | 100–140 | doesn't fit a T4 |

Assumes 25–35 effective TFLOPS for the two T4s in fp16 with DDP: an estimate (~¼ of the 138 TFLOPS
measured on the A100), not a measurement. K5 measures it and redoes this table. FLOP from
`vulpes.model.config.flops_per_token` (pinned by `tests/test_model_config.py`).

**Why not 1B here.** AdamW in fp32 costs 16 bytes per parameter (weights, gradients, two moments):
~17 GB per GPU for 1B, and a T4 has 16 GB. DDP copies the model to both GPUs, so memory doesn't add
up. Even with sharding it would be two years of quota.

## 3. The other quota: Kaggle's TPUs, a trial that may be worth a lot

Kaggle also has a separate TPU quota (~20 h per week, sessions of ~9 h, a TPU v5e-8 or v6e-8: 8 chips
of 16 GB, bf16, ~1,600 TFLOPS peak). At 13–19% of peak that would be 6–9B Flos tokens per week against
1.1–1.5B on the T4s. The figures come from secondary sources that disagree with each other; they are
checked in the account before K5, and PyTorch/XLA's real throughput with our code is the big unknown.

**The trial, inside K5** (uses only TPU quota, so it never delays the T4s): run Flos with PyTorch/XLA
in bf16 (pretraining has fixed shapes, which XLA wants); measure tokens/s on 1 and 8 chips; train 200
steps and compare the loss with the same 200 steps on the T4s; save on the TPU and resume on a T4, and
the other way round. It stays plain PyTorch (no `transformers`, no `accelerate`); the XLA code lives on
a path of its own and the CUDA path doesn't change.

**If it works,** the run alternates (TPU quota first, the faster, then GPU quota, never both at once).
At that speed the limit becomes data (~9.5B unique tokens), not compute, so before the long run K5
revisits the size (a bigger model, near ~700M) and the data (more FineWeb-Edu files), with the measured
numbers in front of us. **If not,** everything stays on the T4s exactly as in §2, §6 and §7.

## 4. Architecture: the same design, one preset up

The Llama-style family of 0.1 doesn't change. Flos is the `base` preset (`configs/model/base.yaml`),
which exists since Milestone 4 and whose parameter and FLOP counts are tested against the real model.

| | Vulpes 0.1 (`small`) | Vulpes Flos 1.0 (`base`) |
|---|---|---|
| Parameters (non-embedding) | 125.3M (100.7M) | 348.4M (315.7M) |
| Layers × width | 16 × 768 | 28 × 1,024 |
| Query / KV heads (GQA) | 12 / 4 | 16 / 4 |
| Head dimension | 64 | 64 |
| SwiGLU hidden | 2,048 | 2,816 |
| Vocabulary | 32,000 | 32,000 (the same tokenizer) |
| Context | 1,024 | 2,048 |
| RoPE θ | 10,000 | 10,000 |
| FLOP per training token | 0.83 G | 2.44 G |
| Pretraining tokens | 2.5B | 10–15B+ |
| Tokens per parameter | 20 | 29–43+ |

Why this shape:
- **348M, not 125M or 1B.** For ~2.4e19 FLOP, Chinchilla points to ~450M on 9B tokens. A little
  smaller and longer-trained costs almost nothing in quality and runs faster on the PC. If the TPUs
  deliver, K5 revisits the size.
- **Deep and narrow.** 28 layers of 1,024, like MobileLLM-350M: at this size more layers beat more
  width (the same logic that gave 0.1 its 16 × 768).
- **The same tokenizer.** The 2.94B tokens already built are reused as they are, and chat, evaluation,
  SFT and export don't change. Changing the tokenizer is Lusitanus's first step, not this one's.
- **2,048 tokens, θ = 10,000.** θ = 10,000 reaches 2k comfortably (Llama 2 used it at 4k).
- **On the PC.** fp32 weights 1.4 GB, a 2,048-token KV cache ~117 MB: fits the GTX 1650 Super's ~2.9 GB
  of free memory without fp16 (slow on that card, decision 13). Estimated 30–60 tokens/s, to be measured.

## 5. Data (K3): the whole FineWeb-Edu `sample-10BT`, not a quarter

0.1 used the first 4 of the 14 `sample-10BT` files. Flos uses all 14, at the same pinned revision,
with the same pipeline, filters and tokenizer: human-written text with an open license (ODC-By 1.0).

| | Tokens |
|---|---|
| Already built (files 1–4) | 2.94B |
| New (files 5–14) | ~6.6B |
| **Total, one epoch** | **~9.5B** (estimated from the first 4 files; duplicates grow with scale, so a little less) |
| Second pass, while the loss falls (D2) | 15B+ ≈ 1.6 epochs |

How it is built (decision D4), on the desktop, like M7, at below-normal priority on 3 processes:

```bash
python scripts/download_fineweb_edu.py --files 14        # 28.5 GB, ~1.5 h; files already there are skipped
python scripts/prepare_data.py --source parquet --input data/raw/fineweb_edu --out data/pretrain/fineweb_edu_10bt --workers 3
```

~4 h of CPU, ~19 GB of shards; the parquet files can be deleted afterwards. No new pipeline code: Gopher
filters, exact + MinHash dedup, split by hash, `uint16` shards, all as in 0.1. One fix was needed
first (§12): with Python sets the near-dedup state would have taken **~13 GB of RAM** for ~9.7M
documents (the PDF's "~5 GB" was an under-estimate); it now takes ~3.4 GB.

**Comparable with 0.1, by construction.** Files are processed in order and dedup keeps the first copy,
so every decision about a document in files 1–4 is the same as in M7: the new set starts with exactly
0.1's documents. So:
- all of 0.1's train shards but its last (partial) one are byte-identical to the first ones here, a
  cheap check that K3 worked (compare SHA-256s with `data/pretrain/fineweb_edu/`);
- the val split starts with 0.1's whole val set (14.6M tokens), and the training loop's validation
  (`eval_batches` 200 × 2 × 2,048) reads the same 819K tokens 0.1 read (800 × 1,024).

**For Kaggle** the shards go up as a private dataset (D4), read by the GPU and TPU sessions. If the
upload is too slow, the same pipeline runs in a Kaggle CPU notebook, which uses no GPU quota.

**Repeating data.** Up to ~9.5B tokens no window repeats; 15B is ~1.6 epochs, 19B two. Muennighoff et
al. (2023) show that up to ~4 epochs costs almost nothing against fresh data, so going past 10B pays.

## 6. Pretraining (K5–K7): measure first, then train

| Stage | What | Quota | For |
|---|---|---|---|
| K5 · Benchmark | 348M on 1 and 2 T4s; the TPU trial | ~1 h GPU, ~3 h TPU | real tokens/s, micro-batch, checkpointing, `torch.compile`; redo the calendar |
| K6 · Pilot | 348M, 2 sessions of ~2 h | ~4 h | the second session resumes the first and repeats the loss of the same steps, as 0.1's pilot did |
| K7 · Base | 348M, 10–15B+ tokens | 195–410 h | the main run: 2–3 sessions a week, while the loss falls |

The run is `configs/train/flos-kaggle.yaml` (checked against this section by `tests/test_train.py`):

- **Batch** 256 sequences × 2,048 = 524,288 tokens per step (GPT-3 Medium's, 350M): ~19,000 steps
  for 10B, 28,610 for 15B. On 2 T4s each does 128 of the 256 micro-batch slots (`micro_batch` 2 ×
  `grad_accum` 128 to start; K5 picks the fastest that fits).
- **AdamW** as in 0.1 (β 0.9/0.95, decay 0.1 on matrices only, clip 1.0). Peak LR 3e-4, GPT-3
  Medium's (0.1 used 6e-4: bigger models want a lower LR), 500 warmup steps.
- **fp16 + GradScaler** on the T4s, which have no bf16 (bf16 on TPUs). RMSNorm and the loss are
  computed in fp32, and a non-finite loss stops training with a clear message.
- **WSD (warmup, stable, decay) instead of cosine.** On the plateau the LR doesn't depend on
  `max_steps`, so the target (10B or 15B) can change mid-run without starting over. The decay takes
  the last ~15% (shape `1 - sqrt`, Hägele et al. 2024) and only starts when the loss stops paying for
  the quota (D2). Still a pure function of the step (decision 21).
- **Two GPUs with DDP** (`torchrun`), plain PyTorch. Each step sees the same sequences in the same
  order as on one GPU, so the run can move to one GPU, the PC or a pod without changing the data.

| Phase | Quota | Weeks at 30 h |
|---|---|---|
| K4–K6 · Kaggle setup, benchmark, pilot | ~6 h | < 1 |
| K7 · Base, 10B tokens | 195–270 h | 6.5–9 |
| K7 · continued to 15B (D2) | +95–140 h | +3–5 |
| K8–K9 · Evaluation (on the PC) and Instruct | ~1–2 h | < 1 |

T4s only. If the TPUs deliver as estimated, 15B fits in ~2–3 weeks of TPU quota. For comparison: K7 at
10B on a rented A100 would be ~50 h (~60–95 USD).

**Deciding when to decay (D2).** `python scripts/train_status.py --run runs/vulpes-flos-1.0-base`
shows where the schedule is and, while on the plateau, the `max_steps` that starts the decay from the
newest checkpoint (`vulpes.train.optim.max_steps_to_decay_from`). Resuming with a larger `max_steps`
lengthens the plateau; once the decay has begun, a resume that would move the LR is refused.

## 7. The session cycle (K4): twelve hours at a time

No Kaggle session is long enough for the run, and none needs to be: 0.1 already resumed bit for bit
on another machine, and here every session is a resume (~17–36 sessions in all).

One session, step by step:
1. Clone the repository at the pinned commit and install the package (Kaggle's PyTorch is already there).
2. Find the newest checkpoint in the inputs; if there is none, this is the start of the run.
3. `torchrun --nproc_per_node 2 scripts/train.py --resume … --stop-after 11.25`, saving every ~30 min.
4. At the end (or on the stop signal) save the checkpoint to `/kaggle/working` and end.
5. On the PC, `scripts/kaggle/sync.py` (like RunPod's) downloads it with the Kaggle API and checks its SHA-256.

The data stay in a dataset of their own, read by every session. Two sessions never write the same run.

Built in K4 (§13, [`KAGGLE.md`](KAGGLE.md)): `scripts/kaggle/` (the notebook, `session.py` and the
PC's `sync.py`). Still to choose by measuring on Kaggle: how a checkpoint passes to the next session
(the notebook's own output, which `session.py` uses as written, or a private dataset via the API). Then a PyTorch/XLA path for the TPUs (K5), separate from the CUDA path.

**No hour of quota wasted.** GPU quota is only for training (pretraining and SFT); data, evaluation and
copies happen on the PC or in a CPU notebook. All the quota every week (2–3 sessions back to back, TPU
quota first if the trial works). The fastest settings, measured: K5 picks micro-batch, `torch.compile`
and checkpointing before the long run.

## 8. Instruct and evaluation (K8–K9)

SFT repeats 0.1's with a bigger model and conversations twice as long: oasst2 English (5,270
conversations, Apache-2.0), Dolly 15k (15,011, CC BY-SA 3.0) and the identity set (44, repeated 3×,
Apache-2.0), **updated by the user to say it is Vulpes Flos 1.0** (written by a person, decision 36;
the facts are in `docs/IDENTITY.md`). LR 3e-5, loss on replies only, early stopping on val loss. At
2,048 tokens almost no conversation is cut. On one T4: ~1–2 h. Evaluation runs on the PC (the model
fits in fp32), with no quota.

| Test | Measures | 0.1 (measured) | Flos (expected) |
|---|---|---|---|
| HellaSwag (acc_norm) | common sense | 31.3% | 36–41% |
| ARC-Easy (acc) | school science | 52.4% | 56–62% |
| PIQA (acc_norm) | everyday physics | 61.9% | 65–68% |
| Val perplexity | 0.1's val set | 22.9 | 15–18 |

The expectations come from published models of similar size (GPT-2 Medium, Pythia-410M) and
FineWeb-Edu's own comparisons, which raise ARC most. They are not promises: Pythia-410M saw 30× more
tokens. With 15B tokens or more, count on the upper half of the ranges.

**Comparing perplexity fairly.** K8 evaluates on 0.1's val set (`--val-data data/pretrain/fineweb_edu`),
not the bigger new one. Flos reads it in 2,048-token windows and 0.1 in 1,024: a longer context lowers
the loss a little by itself, so also report Flos at `seq_len` 1,024 for the strict comparison.

Should do: more coherent English over more sentences; more simple school questions right; follow a
conversation for more turns; know who it is. Shouldn't: get facts reliably right (it will make things
up), write working code, chat well in Portuguese, reason in several steps.

## 9. Milestones

Ten milestones in four phases, called K1–K10 (K for Kaggle) so they never mix with 0.1's M1–M12 or
Lusitanus's. One at a time, each with tests and measured results recorded here.

| # | Milestone | Where | GPU quota | Status |
|---|---|---|---|---|
| **A · Prepare, on the PC** | | | | |
| K1 | Plan and decisions: the PDF (decided 2026-11-14) and this file | desktop | 0 | ✅ 2026-11-14 |
| K2 | Training v1.1: DDP, WSD, the Flos config; CPU tests as always | desktop | 0 | ✅ 2026-11-14 (§12) |
| K3 | Data: the whole `sample-10BT`, ~9.5B tokens, same pipeline; a private Kaggle dataset | desktop CPU, ~6 h | 0 | 🔄 download ✅ 2026-11-14; shards ✅ 2026-11-15 (9.23B + 46.1M tokens, check OK); Kaggle upload to do |
| K4 | Kaggle: notebook, `scripts/kaggle/`, checkpoint hand-over, checked copy to the PC | desktop + Kaggle | ~1 h | code ✅ 2026-11-14 (§13); on Kaggle: to do ([`KAGGLE.md`](KAGGLE.md)) |
| **B · Trial** | | | | |
| K5 | Benchmark and TPU trial: real tokens/s on 1 and 2 T4s, PyTorch/XLA on the TPU (§3); redo the calendar and maybe the size | Kaggle | ~1 h + ~3 h TPU | |
| K6 | Two-session pilot: the resume between sessions repeats the loss. If it fails, stop here | Kaggle | ~4 h | |
| **C · The model** | | | | |
| K7 | Vulpes Flos 1.0 Base: 348M, context 2,048; 10B, then to 15B or more while the loss falls (WSD) | Kaggle, 6.5–14 weeks | 195–410 h | |
| K8 | Evaluation: HellaSwag, ARC-Easy, PIQA and perplexity, against 0.1 | desktop | 0 | |
| **D · The assistant** | | | | |
| K9 | Vulpes Flos 1.0 Instruct (SFT): oasst2, Dolly and identity; context 2,048 | Kaggle | ~1–2 h | |
| K10 | Release and chat: export, chat on the PC, MODEL_CARD, README and plans updated | desktop | 0 | |

K2 and K3 can run at the same time (one is code, the other the PC's CPU). K7 only starts after K6.

## 10. Risks

| Risk | Answer |
|---|---|
| The T4s are slower than estimated | K5 measures before starting. The target becomes what fits: with WSD, stopping earlier gives a finished model, just less trained |
| The quota drops or Kaggle changes its rules | The checkpoint is always on the PC; the run continues on another machine without starting over (a rented GPU or the desktop) |
| A session ends half-way | `--stop-after` stops before 12 h; checkpoints every ~30 min. An error or a cut costs at most that |
| fp16 becomes unstable | GradScaler; RMSNorm and loss in fp32 (already); a non-finite loss stops training. Resume from the previous checkpoint with a lower LR. The pilot shows it early |
| Two GPUs don't scale well | K5 measures 1 against 2. Gradients are summed once per step. At worst one GPU: half the speed, twice the weeks |
| A P100 instead of the T4s | Always pick "GPU T4 ×2". The P100 has no tensor cores and recent PyTorch may no longer support it |
| The 20 GB disk fills up | ~4.2 GB checkpoints, two plus `best.pt`: ~12.6 GB. Data come in as a dataset and don't count |
| The PC overheats building the data | 3 processes at below-normal priority (decision 38), with temperature logging, as in M7 |
| The PC runs out of RAM building the data | Found in K2 (§12): fixed, ~3.4 GB instead of ~13 GB for the dedup state |
| PyTorch/XLA doesn't deliver on the TPUs | Everything stays on the T4s, as planned; only TPU hours are lost. The XLA path never replaces the CUDA path |
| The gain is below expectations | K8 compares with 0.1 on the same val set. Even a small gain leaves ready what Lusitanus needs: DDP, WSD, fp16, the Kaggle workflow |

## 11. Decisions

Decided on 2026-11-14 as recommended, with one rule above all: get the most out of Kaggle to train
the model as far as it goes. They continue the decision table of [`PLAN.md`](PLAN.md) §10.

| # | Question | Decision |
|---|---|---|
| D1 | A 348M model from scratch, or keep training 0.1 (125M)? | **348M from scratch.** Continuing 0.1 is plan B if the T4s are much slower than expected. If the TPUs deliver, K5 revisits the size upwards |
| D2 | How far to train? | **As far as it goes:** not stopping at 10B. Continue to 15B or more while the validation loss falls enough to pay for the quota; only then the final WSD decay. Repeating data up to ~2 epochs is accepted |
| D3 | Context 2,048 or keep 1,024? | **2,048:** +8% compute per token, and the chat remembers twice as much |
| D4 | Build the data on the PC and upload it, or in a Kaggle CPU notebook? | **On the PC,** as in M7, gently (3 processes, low priority): the pipeline is proven there. If uploading ~19 GB takes too long, the CPU notebook, which uses no GPU quota |
| D5 | Portuguese or code already in Flos? | **No.** They need a new tokenizer, which throws away the 2.94B tokens already made and changes chat and evaluation. That is Lusitanus's job |
| D6 | Are Flos's weights published? | **Open until K10:** as planned for 0.1 (GitHub Releases, Apache-2.0), or private. The data are 0.1's, so there are no new licenses to weigh |
| D7 | Also use Kaggle's TPU quota? | **Yes, if K5's trial justifies it** (§3). The plan doesn't depend on it: without TPUs everything runs on the T4s |
| D8 | Near-dedup state for the full `sample-10BT` (2026-11-14, K2) | **A flat 64-bit open-addressing table (`KeySet`)** instead of Python sets: ~0.35 KB per kept document instead of ~1.34 KB, so ~3.4 GB for ~9.7M documents instead of ~13 GB on a 16 GB PC. The same keep/drop decisions (tested against a set), so 0.1's shards rebuild byte for byte; ~22 µs more per document in the main process (~4 min over K3) |
| D9 | Stopping a DDP session (2026-11-14, K2) | **Every process waits at each save** until process 0 has written the checkpoint. Without it the other process exits first with code 130 and `torchrun` stops process 0 (SIGTERM, then SIGKILL ~30 s later) part-way through writing 4.2 GB: the save is atomic, so nothing breaks, but the session's last ~30 min of training would be lost |

## 12. K2 results: training v1.1 (2026-11-14)

What the training code gained for Flos, all plain PyTorch, all tested on CPU (CI runs the same tests):

- **DDP** (`vulpes/train/distributed.py`, used by `Trainer`). Started by `torchrun`, each process
  takes micro-batches r, r + W, r + 2W, … of every step and skips the others' windows in the shared
  seeded order (`PretrainLoader.skip`), so a step trains on exactly the sequences of a single-GPU
  step. Gradients are all-reduced once per step (`no_sync` on the other micro-batches); validation is
  split between the processes and summed; only process 0 writes logs, metrics, samples and
  checkpoints; stop requests and the `torch.compile` probe are agreed by all. Fine-tuning refuses
  `torchrun` (padded SFT batches don't split this way, and SFT takes 1–2 h on one GPU).
- **WSD** (`vulpes/train/optim.py`, `optim.schedule: wsd`, `decay_fraction`). Resuming with another
  `max_steps` is allowed on the plateau (logged) and refused once the decay has begun.
- **The Flos run** (`configs/train/flos-kaggle.yaml`) and the `base` preset at context 2,048.

Tests added (`tests/test_distributed.py`, `tests/test_train.py`, `tests/test_model_config.py`,
`tests/test_data.py`, `tests/test_train_status.py`):

| Check | Result |
|---|---|
| Two processes (Gloo, CPU) against one, 6 steps | same weights within 1e-5; both processes bit-identical to each other; same loader position and val losses; one log header, one record per step, checkpoints from process 0 only |
| A run moved 2 processes → 1 (1 × 8 instead of 2 × 4) → 2 again | equals one uninterrupted single-process run within 1e-5; same loader position |
| A WSD run lengthened on the plateau (6 → 10 steps at step 2) | **bit-identical** to a run planned at 10 steps from the start |
| A WSD run in its decay, resumed with another `max_steps` | refused, with the `max_steps` to keep |
| `flos-kaggle.yaml` | 524,288 tokens per step, 15.0B tokens, decay from step 24,318, `grad_accum` splits over 2 GPUs, val reads 0.1's 819K tokens |
| `base` | 348,447,744 parameters, 2.44 GFLOP per training token, +8% for 2,048 against 1,024 |
| `KeySet` against a Python set | the same answers and the same dedup decisions on 5,000 documents with shared bands |

Two problems found and fixed on the way (decisions D8, D9): the dedup memory for K3 and the end of a
DDP session. Also cleaned up: the 0.1-only pilot config (`configs/train/small-pilot.yaml`, in git
history), "Vulpes 0.1" hard-coded in the chats' welcome text, and two lint errors that would have
failed CI. Ruff's `extend-exclude = ["data", ...]` (meant for the top-level `data/`) also matched
`vulpes/data/`, so the data pipeline was never linted; the patterns are now anchored to the root
(`./data`, ...) and its 5 files reformatted (layout only). `scripts/evaluate.py --val-seq-len` was
added for K8 (§8), and `scripts/train_status.py` now finds the newest run by itself.

## 13. K4, the code: one Kaggle session (2026-11-14)

Written while K3 builds the data (it needs only the PC's CPU), from the RunPod scripts' design.
How to use it: [`KAGGLE.md`](KAGGLE.md).

- **`scripts/kaggle/vulpes-flos-train.ipynb`**: clones the repository at a pinned `COMMIT` into
  `/tmp` (so the code is not saved as output), installs it with `--no-deps` (Kaggle's PyTorch stays)
  and runs `session.py` with the time the notebook started.
- **`scripts/kaggle/session.py`**: finds the shards (`meta.json` of format `vulpes-shards`) and the
  newest copy of the run anywhere under `/kaggle/input`, copies the newest step checkpoint, `best.pt`
  and the small files to `/kaggle/working` (inputs are read-only), and trains with `torchrun` on
  every GPU until `--margin` (45 min) before the 12-hour limit. Then `SESSION_DONE`, `SHA256SUMS`
  and one line per session in `kaggle-sessions.jsonl` (steps, hours, tokens/s, GPUs, commit). Exit
  code 0 whenever the run is in a good state, so Kaggle keeps the output. `--run k4-trial` keeps
  trial sessions apart from the real run.
- **`scripts/kaggle/sync.py`** (PC): `kaggle kernels status`/`output`, a check against
  `SHA256SUMS` before anything is moved into `runs/`, never over a copy that has trained further,
  newest `--keep` step checkpoints kept.

Tested (`tests/test_kaggle.py`): finding the data and the newest of several copies of a run;
what is carried between sessions (not older checkpoints, `.tmp` files or the old hand-over); the
`torchrun` command and the time budget; `sync.py` refusing a corrupted or incomplete download and a
newer local copy, and accepting an output without its folders; and **two real sessions on CPU**,
the second resuming from the first's output and lengthening the run, then copied to the "PC".

Left for the Kaggle account (~1 h of quota; [`KAGGLE.md`](KAGGLE.md), "Still to do"): whether a
notebook's own output works as the hand-over or a private dataset is needed, where Kaggle mounts
inputs, `sync.py`'s download time, and the 19 GB upload of the shards.

**Next:** K3 finishes (then the dataset goes up to Kaggle), the K4 trial session on Kaggle, then K5.
