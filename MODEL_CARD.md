# Vulpes 0.1 model card

> **Status:** Vulpes 0.1 Base and Instruct were trained on 2026-11-10; the numbers below are
> measured. The weights aren't published yet. The next model, Vulpes Flos 1.0 (348M), gets its
> own card at its release (milestone K10, [`docs/PLAN-FLOS.md`](docs/PLAN-FLOS.md)).

## Models

| | Vulpes 0.1 Base | Vulpes 0.1 Instruct |
|---|---|---|
| What it does | Continues English text | Answers in a chat (one assistant turn at a time) |
| Made by | Pretraining from random weights | Supervised fine-tuning of Base |
| Architecture | Llama-style decoder: RMSNorm, RoPE, grouped-query attention, SwiGLU, tied embeddings, no biases | same weights, same shape |
| Size | 125.3M parameters (`small` preset: 16 layers × 768, 100.7M non-embedding) | same |
| Context window | 1,024 tokens | same |
| Tokenizer | 32,000-id byte-level BPE, trained from scratch on FineWeb-Edu (`data/tokenizer/vulpes-bpe-32k.json`) | same |
| License | Apache-2.0 (weights and code) | Apache-2.0 |

Everything is written from scratch in this repository with plain PyTorch:
tokenizer, model, training loop, fine-tuning, evaluation and inference. No
pretrained weights, no other model's outputs, and no high-level model libraries.

## Intended use

- Learning and research: reading, running and changing every part of a small
  language model, end to end, on one consumer GPU.
- Light experiments: simple factual questions, short explanations, text continuation.

**Not suitable for:** anything where a wrong answer causes harm (medical, legal,
financial, safety advice), decisions about people, or as a source of facts. Vulpes
has no internet access and no memory between conversations.

## Training data

All training text is human-written and openly licensed. Full details, filters
and measured counts: [`docs/DATA.md`](docs/DATA.md).

| Stage | Data | License |
|---|---|---|
| Pretraining | FineWeb-Edu `sample-10BT` (educational English web pages from Common Crawl), first 4 parquet files at a pinned revision, after our own filters and deduplication | ODC-By 1.0 |
| Fine-tuning | OpenAssistant oasst2: English conversations, best-ranked reply at each turn | Apache-2.0 |
| Fine-tuning | Databricks Dolly 15k: instruction/response pairs written by Databricks employees | CC BY-SA 3.0 |
| Fine-tuning | `data/identity.jsonl`: 44 short conversations about what Vulpes is (questions by the author; most replies corrected with an AI assistant's help, the one declared exception in docs/DATA.md) | Apache-2.0 |

Benchmarks (HellaSwag, ARC-Easy, PIQA) are used only to measure the models and
never for training.

## Training procedure

- **Base:** next-token prediction on packed 1,024-token windows, AdamW
  (β = 0.9/0.95, weight decay 0.1 on matmul weights only), peak LR 6e-4 with 190 warmup steps
  then cosine decay to 6e-5, gradient clipping at 1.0. 2.5B tokens (20 per parameter), 9,537
  steps of 262,144 tokens, on one rented A100 80 GB in bf16 with `torch.compile`: ~4 h at
  ~167,000 tokens/s (138 TFLOPS).
- **Instruct:** the same loop on conversations in the Vulpes chat template
  (`<|user|> … <|end|><|assistant|> … <|end|>`), loss only on the assistant's
  replies, learning rate 3e-5, up to 3 epochs (1,808 steps of 32 conversations) with early
  stopping on validation loss: best at step 600, stopped at step 900.

Configs: `configs/train/`, `configs/sft/`. Every run saves its resolved config
with its checkpoints, so each published model can be traced to exact settings.

## Evaluation

Zero-shot, log-likelihood scoring of the answer options, using lm-evaluation-harness's
prompt formats (`vulpes/eval/`, `scripts/evaluate.py`). `acc_norm` divides each
option's log-probability by its length in characters.

| | Base | Instruct | Random guess |
|---|---|---|---|
| HellaSwag (acc_norm, 10,042 questions) | 31.3% | 31.2% | 25% |
| ARC-Easy (acc, 2,376) | 52.4% | 52.2% | 25% |
| PIQA (acc_norm, 1,838) | 61.9% | 61.0% | 50% |
| FineWeb-Edu val perplexity (14.6M tokens) | 22.9 | 24.8 | 32,000 |
| Bits per byte | 0.983 | 1.007 | — |

Raw numbers: `runs/vulpes-0.1-*/eval_best.json`. The plan expected HellaSwag ~30%, ARC-Easy
~40–45% and PIQA ~60–63% (docs/PLAN.md §8): ARC-Easy came out well above it, the gain
FineWeb-Edu's educational text is known for. Fine-tuning costs almost nothing on the benchmarks
and a little perplexity on web text, as expected.

## Limitations and risks

- **It makes things up.** A model this small has read far less than large
  assistants and stores much less. It states wrong facts fluently and
  confidently. Check anything that matters.
- **Weak reasoning.** Multi-step arithmetic, logic and long instructions usually fail.
- **English only.** Other languages are mostly absent from the training data and
  tokenize poorly.
- **It calls itself Aupus.** The 0.1 weights were fine-tuned before the project was renamed;
  `data/identity.jsonl` now says Vulpes, and the next fine-tuning run teaches it.
- **Short memory.** The context window is ~1,000 tokens. In a long chat the
  oldest messages are dropped.
- **Bias.** Web text and volunteer-written conversations carry the biases of
  their authors. FineWeb-Edu's educational filter reduces but does not remove
  toxic or one-sided content. No safety tuning beyond the fine-tuning data was done.
- **No refusals by design.** Vulpes wasn't trained to refuse harmful requests and
  may answer them badly rather than decline.

## How to use

```bash
python scripts/chat_web.py --model release/vulpes-0.1-instruct      # chat in the browser (http://127.0.0.1:8000)
python scripts/chat.py --model release/vulpes-0.1-instruct          # chat in the terminal
python scripts/generate.py --model release/vulpes-0.1-base --prompt "The water cycle is"
python scripts/evaluate.py --model release/vulpes-0.1-instruct       # reproduce the table
```

A release directory holds `model.safetensors` (weights only), `model.yaml`,
`tokenizer.json`, `release.json` (source checkpoint, step, validation loss,
SHA-256 of each file), this card and the license.
