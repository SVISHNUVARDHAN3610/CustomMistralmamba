# Mamba reward model stage

This module trains a scalar preference reward model and independently evaluates it
on UltraFeedback test, HH-RLHF test, and HelpSteer2-Preference. It contains no policy optimizer, rollouts, reference
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
are logged per producer invocation; resume starts a new accounting interval.

Default maximum length is **2,048 tokens**. The entire prompt and assistant header
are preserved identically for both branches. Response overflow retains its start
and end independently (`data.response_truncation_strategy="head_tail"`). The producer reserves up to 128 response tokens (less for
short responses) plus EOS. Prompts that cannot fit that reserve are filtered with
an explicit reason, rather than deleting instructions or the preference signal.
This deliberate filter can reduce long-context coverage; tune length/reserve for
the cloud GPU budget. There is no global cross-source deduplication pass.

### Explicit truncation and sequence accounting

`data.max_length` stays **2048**. Response budget is `max_length - len(prompt with
BOS and role headers) - len(assistant header) - 1 EOS`. Long responses never cause
the shared prompt to be shortened. The existing minimum-response reserve remains
`min(min_response_tokens, max(chosen_length, rejected_length))`; pairs below that
reserve, or with zero response budget, are filtered explicitly. Thus a nearly full
prompt can be rejected even when a few response tokens would technically fit.

Strategies are `head` (first N tokens) and the unchanged default `head_tail`
(ceil(N/2) beginning tokens plus floor(N/2) ending tokens). Each side uses its own
length and the same available budget; the shorter response stays complete. One
terminal EOS is appended after truncation; existing terminal EOS tokens are removed
before budgeting to avoid duplication. Empty tokenized responses and pairs made
identical by truncation are filtered. Padding happens afterwards in the collator.
Truncation is a deliberate data-processing decision that can remove preference
evidence, including reasoning/code in the middle; these diagnostics do not establish
that either strategy is optimal for learning.

Startup logs show **only bounded producer prefetch**, not a full preprocessing pass.
Per-source counters include chosen/rejected/both/pair truncations and insufficient
budget filtering. Rates divide by nonempty tokenized pairs before truncation,
including pairs subsequently rejected for insufficient budget or identical tokens.
Length summaries separate truncated and complete responses and original/retained
lengths (response body only, excluding prompt/EOS). Count, mean and max are exact;
median/p95/p99 use a private seeded reservoir of at most 2048 lengths per group,
and are labeled approximate above that capacity. Final summaries cover each
producer invocation. No entire source or unbounded length list is materialized.

### Effective source mixture accounting

`dataset_accounting.json` and the final `Dataset Accounting` log report configured
35:75 weights, target probabilities, and each source's raw, normalized, filtered,
valid, tokenized and consumed pair counts. `normalized` means schema validation
passed; `valid` and `tokenized` mean the final token pair passed all filters.
Retention is valid/raw; filtering is filtered/raw. Valid and consumed proportions
are reported separately, since filtering, prefetch and distributed sampler tails
can change them. No rebalancing is applied: HF probabilities, seed, source order
and `all_exhausted` behavior remain unchanged.

Raw/normalization/filter/tokenization counts belong to the single rank-zero producer
and include bounded unconsumed prefetch. Consumed counts are incremented only after
a training microbatch's backward pass, and summed across disjoint rank samplers;
DataLoader prefetch and validation do not increment them. Overflow-skipped updates
still consumed their input pairs. Repeated source draws from `all_exhausted` or new
epochs count as repeated training exposures, not unique dataset records.

Accounting is explicitly **current invocation only**. Resume regenerates its shard
boundary and may repeat producer work; resumed consumed counts include only newly
used batches. Historical counts are not reconstructed or added to checkpoint cursors.
This preserves the existing checkpoint format and resume semantics; reports from
different invocations must not be interpreted as a deduplicated lifetime dataset count.

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
through the existing logger and `metrics.jsonl`. Periodic validation uses the fixed
UF/HH diagnostic sets described below; full evaluations have separate reports.

## Diagnostic and full evaluation

Repository-native nested JSON settings under `system`:

```json
{
  "diagnostic_eval_enabled": true,
  "diagnostic_eval_pairs": 512,
  "diagnostic_eval_interval": 100,
  "full_eval_enabled": true,
  "full_eval_at_checkpoints": false
}
```

The diagnostic subset is built **once before the training feed starts** using
independent source feeds: the first 256 valid UF `test_prefs` pairs and 256 valid
HH `test` pairs in pinned HF source order. A private RNG with `cfg.seed` orders each
cached subset. For odd totals UF gets the extra pair. Only these tokenized pairs
are cached (plus the existing bounded feed's transient prefetch); no full source is
materialized. Insufficient valid pairs raise a clear error. The smoke profile uses
four diagnostic pairs. This fixed prefix is reproducible but is not a random
representative sample of the entire benchmark.

Every periodic evaluation uses these exact cached tokens and reports UF and HH
separately, with `subset_sha256` fingerprints. Rebuilding with the checkpoint's
pinned revisions/configuration produces the same membership/order regardless of
training cursor, epoch or RNG state. Python, NumPy, Torch CPU and initialized CUDA
RNG states and model train/eval mode are restored, including on failure. Diagnostic
progress is never recorded in a training checkpoint. The optional interval defaults
to the existing `eval_interval` (100). Legacy `eval_batches` is accepted when reading
old configurations but no longer controls periodic evaluation.

Full evaluation streams each source separately to exhaustion: UF `test_prefs`, HH
`test`, HelpSteer2's preference release. It runs at normal training completion by
default, optionally after saved checkpoints, and independently from the evaluation
entry point. It does **not** run at every diagnostic interval. `--stop-after` skips
end-of-training full evaluation when it is only interrupting a longer job.
HelpSteer2 remains entirely outside training and periodic diagnostic membership.
Its overall results, original-train/original-validation subsets (when present),
and existing split/strength categories each have independent statistics.

Reports are written to `diagnostic_evaluation_<step>.json` or
`full_evaluation_<step>.json`, plus evaluation records in `metrics.jsonl`. Standalone
evaluation defaults to all three sources and writes `full_evaluation_metrics.json`;
`--dataset helpsteer2` retains a HelpSteer2-only report. `--max-batches` is a bounded
sample per source for development, explicitly labeled Sample Evaluation in logs.

### Reward Scale Diagnostics

**Pairwise accuracy remains the primary metric**: the fraction with chosen reward
strictly greater than rejected reward. Exact ties are incorrect rankings and are
also reported separately. Accuracy measures ranking quality; it cannot distinguish
small reward differences from excessively large magnitudes or tails.

Every source/subset report includes `pairs`, `loss`, `pairwise_accuracy`, and:

- `chosen_reward_{mean,std,min,max,p50,p95,p99}`
- `rejected_reward_{mean,std,min,max,p50,p95,p99}`
- `margin_{mean,std,min,max,p50,p95,p99}` for chosen minus rejected
- `fraction_positive_margin`, `fraction_zero_margin`, `fraction_negative_margin`
- Pooled chosen+rejected `reward_{mean,std,min,max,p50,p95,p99}` within that source

Existing mean metric names remain aliases; `mean_chosen_reward`,
`mean_rejected_reward`, `mean_reward_margin`, and `std_reward_margin` are also
provided. The nested `reward_scale_diagnostics` section and labeled log include
both reward means/stds and margin mean/std/p50/p95/p99. The tiny diagnostic set
receives these statistics too, without extra work every training step.

Detached rewards move to CPU and accumulate in float64 using stable Chan/Welford
moments. Standard deviations are population standard deviations (`ddof=0`), including
zero for a singleton. Exact percentiles use NumPy linear interpolation and in-place
partitioning of temporary writable memory maps, avoiding a Python list or GPU tensor
containing all rewards. Disk use is 40 bytes per pair per reported group, released
even on errors; provision OS temporary storage in the cloud (e.g. `TMPDIR` on Linux).
Only the current batch and scalar moments need active working RAM.

These natural reward-scale measurements establish a baseline for later PPO reward
processing; they **do not prove PPO stability**. Outputs remain unrestricted linear
scalars: there is no normalization, clipping, whitening, sigmoid or tanh added to
the model, and no changes to the objective or policy optimization.

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

Evaluation uses the same bounded feed and scores identical batches on every rank.
This avoids uneven FSDP forward collectives and duplicate-row
bias from padded distributed samplers, at the cost of redundant evaluation compute.
Rank zero owns each scored pair exactly once and computes the global moments and
exact percentiles, then broadcasts the small report. Because every rank sees the
entire evaluation stream, no cross-rank reward gathering or averaging of local
percentiles is necessary. Counts are independent of world size. A single-GPU torchrun is
appropriate for standalone evaluation. `--max-batches` explicitly bounds evaluation.

## Commands

Run these from the repository root. The following development commands are small
and safe for a laptop; no distributed group or CUDA training is initialized:

```bash
python post-training/RLHF/train_reward_model.py --count-only
python -m unittest tests.test_reward_model -v
python -m unittest tests.test_reward_diagnostics -v
python post-training/RLHF/train_reward_model.py --smoke --output-dir runs/reward_smoke --stop-after 1
python post-training/RLHF/train_reward_model.py --smoke --resume runs/reward_smoke/checkpoint-00000001
python post-training/RLHF/evaluate_reward_model.py --smoke --checkpoint runs/reward_smoke/checkpoint-00000002 --max-batches 1
```

Use a fresh output directory for a fresh smoke run. The smoke evaluator uses
offline UF/HH/HelpSteer2-shaped fixtures; its metric is a software check, not model quality.
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

# Independent full evaluation of all three sources (cloud only)
torchrun --standalone --nproc_per_node=1 post-training/RLHF/evaluate_reward_model.py --checkpoint runs/reward_model/checkpoint-00000500

# Optional single-source evaluation
torchrun --standalone --nproc_per_node=1 post-training/RLHF/evaluate_reward_model.py --checkpoint runs/reward_model/checkpoint-00000500 --dataset helpsteer2
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

## Kaggle TPU v5e-8 entry point

`reward_model_tpu_train.py` is an opt-in **hybrid Mamba–Attention–MoE** reward
trainer for a single TPU host with eight devices. The GPU trainer above still uses
the pure-Mamba reward model. The TPU model reuses `HybridModel` embeddings,
`HybridDecoderLayer`, normalization, attention, Mamba, fusion and expert weights,
and adds a scalar linear head at the last valid token. It has no vocabulary head.
It uses the existing pairwise logistic loss, AdamW (LR `1e-5`, betas `0.9/0.95`,
constant LR), clipping and configurable gradient accumulation.

Supply a repository-native `HybridMambaMoEConfig` JSON. Set
`use_dual_memory=false`, `use_auxiliary_losses=false`, `capacity_factor=null`
and `use_torch_compile=false`: this entry supports stateless hybrid reward
scoring, not persistent memory or auxiliary policy training. CUDA fused scans
are disabled in the TPU adapter. Model dimensions come from your JSON and the
actual parameter count is logged before training; there is no implicit enormous
policy-model default. `--backbone-weights` optionally loads a **bare
`HybridModel.state_dict()`** on CPU before device placement. It does not accept
a full LM wrapper or a GPU distributed checkpoint directory.

Create an appropriately sized starting configuration, for example in a Kaggle
notebook cell with the repository as the working directory:

```python
from model.core.config import HybridMambaMoEConfig

HybridMambaMoEConfig(
    vocab_size=32000,
    hidden_size=768,
    num_layers=12,
    num_heads=8,
    num_kv_heads=2,
    head_dim=96,
    intermediate_size=2048,
    num_experts=4,
    top_k=2,
    dropout=0.0,
    max_position_embeddings=2048,
    window_size=2048,
    use_dual_memory=False,
    use_auxiliary_losses=False,
    use_fused_mamba_scan=False,
).save_pretrained("reward_tpu_model.json")
```

The Dataset is already preprocessed. A `--dataset-factory module:callable` returns
a deterministic, map-style PyTorch Dataset. Each item contains **unpadded integer
token sequences**, `{"chosen": [...], "rejected": [...]}`, including the shared
formatted prompt and terminal EOS, exactly as `PreferenceShardDataset` returns.
An existing shard can be used directly through that class as the factory. For
multiple shards, your factory can return a PyTorch `ConcatDataset` of them.
No full Dataset is loaded or tokenized by this entry. Keep HelpSteer2 out of the
training Dataset; its contents and source mixture remain the caller's existing
preprocessing responsibility.

`FixedPreferenceCollator` calls the existing `PreferenceCollator`, removes
non-tensor metadata from device transfers, and pads to `--sequence-length`
(default 256). It rejects empty/overlength/out-of-vocabulary records rather than
silently truncating preprocessed preferences. Prepare longer data upstream or
explicitly increase that option within the model's window and RoPE limits.
Global batch size defaults to eight pairs and must be divisible by eight.
The per-epoch incomplete batch is dropped to keep compilation shapes constant;
a partial final gradient-accumulation window is scaled by its actual batch count.
Each pair is visited once per epoch's shuffled index order, apart from that
explicit dropped tail. Preprocessing must be deterministic for exact resume.

Install matching Linux TPU builds of `torch` and `torch_xla` (same major/minor,
2.6 or newer), plus the repository's dataset dependencies, using the official
[XLA installation instructions](https://docs.pytorch.org/xla/master/learn/quickstart.html).
Restart the Kaggle kernel after changing runtime packages, and launch a fresh
Python process once. The script selects PJRT TPU, calls `runtime.use_spmd()`
before `xm.xla_device()`, verifies eight locally addressable devices, and builds
an eight-way `fsdp` mesh. `torch.distributed` is aliased as `xla_dist` for
`init_process_group("gloo", init_method="xla://")`; this only coordinates the host.
The old `torch_xla.distributed.xla_dist` launcher has no SPMD initialization API.

There is no generic `SPMD(model, optimizer)` wrapper in the supported API.
Instead, parameters and Adam moments receive matching `mark_sharding`
annotations, layer activations and input batches are partitioned across the
mesh, and the XLA compiler generates the required TPU gradient collectives.
No DDP sampler, eight Python replicas or extra `xm.optimizer_step` reduction is
used. The existing XLA `ParallelLoader` provides bounded background uploads.
BF16 autocast preserves FP32 optimizer weights/moments and sensitive Mamba
computations; do **not** set `XLA_USE_BF16`, `XLA_DOWNCAST_BF16` or legacy
`XLA_TPU_ENABLE_XRT`. The script rejects these flags. Activation checkpointing
uses XLA's RNG-aware checkpoint helper and can be disabled with
`--no-activation-checkpointing`.

The TPU-local MoE adapter computes experts at fixed shapes and masks their
outputs with the original top-k router weights. This preserves the dropless
mixture and avoids variable-size token indexing, at the cost of evaluating
unselected experts. CPU tests compare outputs and gradients to the original
dispatch. This is a compatibility baseline: fused TPU MoE/scan performance,
compilation time and peak memory still require measurement on Kaggle.

```bash
# Kaggle only: one process controls all eight devices. Use your Dataset factory.
python post-training/RLHF/reward_model_tpu_train.py --model-config reward_tpu_model.json --dataset-factory my_data:make_train_dataset --dataset-kwargs '{}' --output-dir /kaggle/working/reward_tpu

# Resume with the SAME model, Dataset and training settings.
python post-training/RLHF/reward_model_tpu_train.py --model-config reward_tpu_model.json --dataset-factory my_data:make_train_dataset --dataset-kwargs '{}' --output-dir /kaggle/working/reward_tpu --resume /kaggle/working/reward_tpu/latest.pth

# Laptop: only a built-in 6,417-parameter CPU fixture, at most two steps.
python post-training/RLHF/reward_model_tpu_train.py --smoke --output-dir runs/reward_tpu_smoke
python -m unittest tests.test_reward_tpu_train -v
```

`latest.pth` is atomically replaced at save intervals and completion. It contains
the model, Adam state, configuration, step/epoch/next-batch cursor, and Python,
Torch and XLA RNG state. The deterministic sampler resumes from the next index
without fetching preceding examples. To extend a finished run, increase
`--max-steps` and/or `--epochs`. Dataset content must stay identical: factory
arguments and Dataset length are checked, but content is not hashed.
`xm.save` gathers checkpoint tensors to the **sole host**, so checkpointing
requires enough CPU RAM for the model and optimizer state. It does not gather
copies onto each TPU core. This hybrid TPU checkpoint family is distinct from
the pure-Mamba GPU DCP checkpoints and the existing GPU evaluation CLI.
For scoring, construct `TPURewardModel` from the saved `config`, load its `model`
state dict, call `eval()`, and supply token IDs and nonempty right-padding masks.

Local validation uses tiny CPU fixtures only. No TPU compilation, eight-device
execution, BF16 XLA backward, or TPU checkpoint round trip has been validated on
the development laptop. Run a short Kaggle job before committing to long training.
