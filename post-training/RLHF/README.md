# Mamba reward model stage

This module trains a scalar preference reward model and independently evaluates it
on HelpSteer2-Preference. It contains no policy optimizer, rollouts, reference
policy, vocabulary head, attention, or MoE. Actual training belongs on cloud GPUs.
The offline development profile uses an **8,513-parameter CPU model**, not the
production model.

## Architecture and exact size

```text
Prompt + Response (existing SFT role-text format)
  -> token embedding
  -> 48 x [RMSNorm -> repository MambaBlock -> residual addition]
  -> final RMSNorm
  -> final nonpadding hidden state
  -> Linear(1024, 1)
  -> reward [B, 1]
```

| Setting | Default |
|---|---:|
| Vocabulary | 32,000 |
| d_model | 1,024 |
| Layers | 48 |
| d_state | 16 |
| d_conv | 4 |
| expand | 2 |
| Internal width | 2,048 |
| dt_rank (repository default: ceil(d_model / 16)) | 64 |
| Embedding parameters | 32,768,000 |
| Mamba parameters | 319,979,520 |
| All normalization parameters | 50,176 |
| Scalar head parameters, including bias | 1,025 |
| **Total / trainable parameters** | **352,798,721 / 352,798,721** |

For this actual `MambaBlock`, one block has
`d_inner * (3*d_model + 2*dt_rank + 3*d_state + d_conv + 3)` parameters:
**6,666,240**. Each residual layer adds 1,024 RMSNorm parameters. Counts are also
computed from actual tensors on the meta device before training; production
configurations outside 300M–400M raise an error. Tokenizer vocabulary is checked
using `utils.dataset.verify_tokenizer_vocab`.

`RewardModel(input_ids=..., attention_mask=...)` returns `[B, 1]`. Inputs must be
nonempty and right padded, matching the repository's Mamba scan contract.
The head gathers the final valid position; changing padding tokens does not change
the score. The same model scores both branches in one concatenated forward.

Weights start with the repository block's initialization, including its specialized
SSM time-step bias and A/D initialization. No compatible pretrained pure-Mamba
checkpoint exists in this repository. `system.initial_checkpoint` can warm-start
from a compatible reward DCP checkpoint; hybrid policy/SFT checkpoints cannot be
silently loaded into this architecture. Starting from random weights is supported,
but this implementation does not claim that preference-only training from random
initialization yields a useful language reward model. A compatible pretrained
backbone would improve that starting point and requires separate preparation.

## Verified dataset schemas

Schema references were checked on 2026-09-08. Two streamed records from each source
were also normalized and tokenized locally using the repository tokenizer.

| Purpose | Hugging Face ID | Config / selection | Relevant fields |
|---|---|---|---|
| Train | `HuggingFaceH4/ultrafeedback_binarized` | `default`, `train_prefs` | `prompt`, chosen/rejected message lists |
| Train | `Anthropic/hh-rlhf` | `default`, `train` | chosen/rejected full Human/Assistant transcripts |
| Periodic validation | Same two sources | `test_prefs` / `test` | Same schemas |
| Held-out evaluation only | `nvidia/HelpSteer2` | `default`, `data_dir="preference"`, HF split `train` | `prompt`, `response_1`, `response_2`, `preference_strength`, row-level `split` |

UltraFeedback provides train/test preference, SFT, and generation splits; only
the preference splits are used. Its binarized release includes the upstream
cleaning/fixes, avoiding raw multi-candidate conversion. See its
[dataset card](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized).
HH-RLHF provides train/test chosen/rejected human comparisons; its shared dialogue
prefix is checked before the final assistant responses are extracted. See
[Anthropic's dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf).

HelpSteer2's preference directory has one HF `train` split containing both row-level
train and validation annotations. **Both remain held out here.** Negative strength
prefers response 1; positive prefers response 2; zero is a tie and is filtered.
We use these direct human preferences, not differences between the five rating
attributes in the default ratings release. Metrics are grouped by original split
and preference strength; categorical quality-dimension labels are not supplied by
this preference schema. See [NVIDIA's dataset card](https://huggingface.co/datasets/nvidia/HelpSteer2).

The configuration preserves `ultrafeedback_weight=35` and `hh_rlhf_weight=75`.
These are **relative sampling weights**, normalized to **31.8181818%** and
**68.1818182%**. HF interleaving uses the configured seed and `all_exhausted`:
exhausted sources are repeated until every source has exhausted at least once.
The proportions describe source draws, not exact retained-pair counts after
filtering, nor token percentages. Changing the seed per epoch is deterministic.

Canonical records contain string `prompt`, `chosen`, and `rejected` values.
An additional `prompt_messages` field preserves multi-turn boundaries for the SFT
tokenizer. Missing/empty text, malformed roles, mismatched prompt prefixes,
identical responses, ties, invalid strengths, and pairs made identical by
tokenization/truncation are filtered. Raw/filtered/valid counts and reason counts
are logged per producer invocation; resume starts a new diagnostics interval.

Default maximum length is **2,048 tokens**. The entire prompt and assistant header
are preserved identically for both branches. Response overflow retains its start
and end independently. The producer reserves up to 128 response tokens (less for
short responses) plus EOS. Prompts that cannot fit that reserve are filtered with
an explicit reason, rather than deleting instructions or the preference signal.
This deliberate filter can reduce long-context coverage; tune length/reserve for
the cloud GPU budget. There is no global cross-source deduplication pass.

## Integration with existing infrastructure

The audit found that `utils/dataset.py` uses a **bounded disk-shard queue**, not a
Python queue or independent tokenization process pool. Its background producer
streams HF rows, tokenizes on demand into a sliding buffer, publishes binary/JSON
shards atomically using temporary files, fsync and rename, and cleans consumed
`.done` shards with bounded retries for Windows file handles. Consumers read
memory maps. Native HF iterable state is preferred; deterministic skip is the
logged compatibility fallback. HF handles transport retries; exhausted errors
propagate instead of dropping a source or changing mixture weights.

**`utils/dataset.py` and `post-training/sft_post_train.py` are unchanged.** Each
training stage keeps its existing implementation and can run independently.
`PreferenceShardProducer` inherits the existing pretraining and SFT producer
interfaces. SFT supplies tokenizer injection, atomic JSON publication, and its
bounded `_flush` implementation. The pretraining producer supplies thread-error
propagation and retrying `.done` cleanup. Preference-specific iteration, HF source
selection, normalization, tokenization, and pair storage live in `rlhf_dataset.py`.
The reference path/name/split/weight configuration schema, streaming/interleaving,
source-state/skip resumption, and producer-consumer design are preserved without
adding hooks or monkeypatches to either original stage.

Whole pairs are stored as uint32 token arrays with offset/category metadata;
packing unrelated examples into an LM window would destroy reward boundaries.
Metadata is published first and `.bin` last, matching SFT's atomic ready-file
contract. No full dataset is materialized.

`PreferenceFeed` inherits `post-training/sft_post_train.py::ShardFeed` and reuses its
shutdown/join lifecycle. Its RLHF-specific setup uses the same rank-zero background
thread and stop event. Its readiness wait follows SFT's timeout/error/broadcast
protocol, with a fixed file bound: SFT retains old shards and grows its queue limit,
whereas RLHF releases consumed shards because source cursors support resume.
That small behavioral adaptation is entirely in the reward training module.
A private UUID cache belongs to each invocation; files are released after producer
shutdown. At most `buffer_size` unconsumed shards plus one bounded pair buffer are
live per feed. No independent worker pool or alternative queue framework is added.

Formatting uses `utils.sft_dataset.tokenize_messages`: BOS, ordinary `role:\n`
headers, assistant EOS, and no newly invented chat tokens. Dynamic padding is
specific to variable-length preference pairs because the old LM/SFT consumers
have fixed packed windows. DataLoader workers only read mmap shards; tokenization
stays in the existing producer thread. `num_workers` and `prefetch_factor` are
configurable, with zero workers the low-overhead default. Training shards use
`DistributedSampler` with deterministic epoch/shard seeding and no overlapping
rank samples; at most `world_size-1` trailing pairs per shard are dropped.

## Objective and training defaults

The Bradley–Terry/logistic objective is
`loss = -logsigmoid(reward_chosen - reward_rejected).mean()`.
Equal rewards give `log(2)` loss; increasing the chosen reward lowers it.
Gradient accumulation weights microbatches by pair count, including partial
windows. Effective full-window batch size is
`batch_size * gradient_accumulation_steps * world_size`.

Defaults: repository AdamW-only optimizer builder, LR **1e-5**, betas **(0.9, 0.95)**,
weight decay **0.01** with repository no-decay rules, epsilon **1e-8**; repository
cosine scheduler with **3% warmup** and **10% minimum LR**. Per-rank batch size is
**1**, accumulation **16**, gradient clip norm **1.0**, one epoch with a **10,000
successful-update cap**. Epoch exhaustion may end training before the cap.
The scheduler horizon is the configured cap, not an expensive full-stream count.
`--stop-after` allows interruption tests without changing this horizon.

Logs include step, epoch, pairwise loss/accuracy, both mean rewards, mean margin,
LR, global gradient norm, pair throughput, optimizer-state entry count, scheduler
step, loss scale, and peak allocated CUDA memory. Rank zero writes `train.log`
through the existing logger and `metrics.jsonl`. Validation uses only UF/HH test
splits, defaults to 16 batches every 100 updates, and is clearly labeled validation.
Its deterministic small prefix is a diagnostic, not a full validation benchmark.

## FSDP2, precision, and checkpointing

The repository cloud baseline pins **torch 2.6.0** in `requirements-fsdp2.txt`;
the inspected development environment has **torch 2.13.0+cu126**, **datasets 5.0.1**,
and **transformers 5.16.1**. The existing `_require_fsdp2`, distributed initializer,
optimizer builder, global DTensor-aware clipping, RNG helpers, and logger are reused.

The old hybrid trainer's `ignored_params` argument is absent in PyTorch 2.6.
This reward model does not use that argument. It applies the supported public
`torch.distributed.fsdp.fully_shard` bottom-up to complete residual layers and
then the root. Non-reentrant checkpoint wrappers own entire layers, ensuring
FSDP gathers weights before replay reads Mamba A/D and RMSNorm parameters. Inner
Mamba checkpoint regions are disabled. The optimizer is constructed after sharding.
Parameters are initialized on CPU and moved/sharded one layer at a time; a full
CPU model initially exists on each rank. Full GPU model copies are not constructed.

FP32 master parameters and reductions follow repository conventions. `auto` uses
BF16 autocast where supported, otherwise FP16. FP16 uses explicit local-gradient
unscaling and the shared global-norm reduction for a synchronized overflow decision;
all ranks skip together and halve the saved loss scale. BF16/FP32 nonfinite gradients
raise. Accumulation uses `set_requires_gradient_sync`, clipping is global, and
gradients are cleared with `zero_grad(set_to_none=True)`. Activation checkpointing
is enabled by default. Install the compatible optional `mamba-ssm` CUDA extension
for practical cloud throughput; the repository's CPU/PyTorch scan remains available.

All ranks participate in **Distributed Checkpoint (DCP)** using the distributed
state-dict API with `full_state_dict=False`. No full model/optimizer gathering is
performed on each rank. Checkpoints contain sharded model and AdamW state, scheduler,
successful step, epoch, intra-shard batch cursor, source boundary/native HF state,
configuration, pinned Hub/tokenizer revisions, per-rank Python/Torch/CUDA RNG,
and FP16 scale state. A `complete` marker commits the small trainer metadata last;
incomplete/incompatible checkpoints are rejected. Checkpoint directories are never
overwritten. Keep output directories on storage shared by all ranks/nodes.

Resume rebuilds from the **consumer's** shard boundary, then restores its batch
offset; it never advances to the background producer's speculative position.
This allows bounded caches to be discarded safely. Native HF state restoration is
used when available; the repository fallback replays/skips raw examples at O(N)
cost. Exact resume requires the same model/data/training settings, world size,
precision, and compatible dependencies. DCP does not promise cross-version
compatibility; use the same environment for training and resume. Evaluation can
load only the model into a new world size, subject to DCP compatibility.

Evaluation uses the same bounded feed and scores identical batches on every rank,
then reduces totals. This avoids uneven FSDP forward collectives and duplicate-row
bias from padded distributed samplers, at the cost of redundant evaluation compute.
The reported sample count divides out rank replication. A single-GPU torchrun is
appropriate for standalone evaluation. `--max-batches` explicitly bounds evaluation.

## Commands

Run these from the repository root. The following development commands are small
and safe for a laptop; no distributed group or CUDA training is initialized:

```bash
python post-training/RLHF/train_reward_model.py --count-only
python -m unittest tests.test_reward_model -v
python post-training/RLHF/train_reward_model.py --smoke --output-dir runs/reward_smoke --stop-after 1
python post-training/RLHF/train_reward_model.py --smoke --resume runs/reward_smoke/checkpoint-00000001
python post-training/RLHF/evaluate_reward_model.py --smoke --checkpoint runs/reward_smoke/checkpoint-00000002 --max-batches 1
```

Use a fresh output directory for a fresh smoke run. The smoke evaluator uses
offline HelpSteer2-shaped fixtures; its metric is a software check, not model quality.
To repeat the tiny live schema check (two rows per source), in PowerShell:

```powershell
$env:RM_LIVE_SCHEMA='1'
python -m unittest tests.test_reward_model.RewardTests.test_live_hf_samples -v
Remove-Item Env:RM_LIVE_SCHEMA
```

Generate and edit the nested JSON configuration before cloud training:

```bash
python post-training/RLHF/train_reward_model.py --write-config reward_config.json
```

**Cloud only**, after installing `requirements-fsdp2.txt` using the appropriate
official CUDA wheel index and a compatible optional `mamba-ssm` build:

```bash
# Single cloud GPU
torchrun --standalone --nproc_per_node=1 post-training/RLHF/train_reward_model.py --config reward_config.json

# Multiple cloud GPUs, existing repository torchrun convention
torchrun --standalone --nproc_per_node=4 post-training/RLHF/train_reward_model.py --config reward_config.json

# Resume: saved config supplies the original scheduler and data settings
torchrun --standalone --nproc_per_node=4 post-training/RLHF/train_reward_model.py --resume runs/reward_model/checkpoint-00000500

# Independent held-out evaluation (omit --max-batches for all preferences)
torchrun --standalone --nproc_per_node=1 post-training/RLHF/evaluate_reward_model.py --checkpoint runs/reward_model/checkpoint-00000500 --max-batches 32
```

No cloud training was run during implementation. CPU smoke/checkpoint tests and
tiny real dataset samples do not validate NCCL collectives, real FSDP2 backward,
fused scans, GPU BF16/FP16 behavior, peak cloud memory, or reward quality. These
require a short cloud validation job before a long run. The local CUDA wheel could
not initialize against the laptop's old NVIDIA driver. No trained production reward
checkpoint is included; only development smoke checkpoints can be produced locally.
The documented cloud launcher targets one node with multiple GPUs. The reused
distributed initializer checks world size against local GPU count, so multi-node
launches require a separate update to that existing utility before they are supported.
