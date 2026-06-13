# Module 3 — Planned Improvements

Notes from a design discussion (2026-06). Not yet implemented; captured here so the
direction isn't lost. Two themes: **smarter selection** and **hyperparameter coverage**,
which meet at `image_size`.

## 1. Selection is too coarse / not smart enough

Current structured score uses only 4 signals (data_size match, priority-vs-complexity,
`preferred_when` bonus, `few_shot` bonus), then merges 60% structured / 40% vector.

Weaknesses:

- **Near-ties decided by noise.** `data_size` has 3 buckets, `priority` 3 values. Rankings
  sit very close (observed: efficientnet 0.691 vs dinov2 0.67) so small Module-1 wording
  changes flip the pick. Needs a principled tiebreak — e.g. **cost-aware**: when scores are
  within ε, prefer the lighter / cheaper backbone.
- **Signals already collected but unused** (highest ROI — just wire them up):
  - `domain_transfer` (strong/moderate/weak) on backbones
  - `recommended_when` on pretrained_model nodes
  - **Module 2 image statistics** (resolution, colour mode, format) — dropped in
    `merge_modules`, never reach Module 3 (see §3)
  - `preferred_when` edges on loss / pretrained nodes (dead data)
- **No task-character signal.** Fine-grained tasks (e.g. cassava leaf disease) favour
  full-finetune CNNs + higher resolution, but the system can't tell. Source could be
  Module 1 (LLM extracts "fine-grained") or Module 2 (intra-class similarity).
- **Vector weight (40%) is shaky** — cosine of the query against a one-paragraph backbone
  description. Consider demoting it to a tiebreaker, or enriching descriptions.
- **`_select_components` uses `candidates[0]`** for head/optimizer — order-dependent and
  fragile (existing known issue).

## 2. Hyperparameters — principled split, not ad-hoc

Decide each HP by one test: *"would changing it change which model is the right pick, or
just how to run the chosen model on this machine?"*

| Class | Owner | Examples |
|---|---|---|
| Model-coupled / recommendation-level | **Module 3** (the "recipe") | learning_rate (head_only→high, full→low), image_size, scheduler type, augmentation strength, epoch budget |
| Execution / hardware | **Module 4** | batch_size (OOM auto-retry / grad-accum), num_workers, mixed_precision, gradient_clip, checkpoint cadence |
| Needs runtime feedback | **Module 4** (only it sees training dynamics) | early stopping, lr warmup auto, OOM retry |

Module 4 already sets some of these by default — make the split **explicit and principled**,
not accidental.

⚠️ **Provenance**: once HPs come from three places (Module 3 recipe, Module 4 defaults, the
orphan `recommended_epochs` in `pipeline.py`), the final config is hard to reason about. The
merged recipe should record where each value came from.

## 3. Wire Module 2 statistics into selection

`merge_modules` (pipeline.py) currently keeps only `total_images`→data_size, `num_classes`,
and `class_distribution`→class_imbalance. It **drops** the resolution (min/max/avg width &
height), colour `mode_distribution`, and `format_distribution` that Module 2 computed.

Fix, following the existing `derive_*` pattern:

1. Add `derive_*` helpers turning raw stats into semantic signals: `resolution_tier`
   (low/medium/high), `color_mode` (rgb/grayscale), maybe `size_variability`.
2. Extend the Module 3 input schema with these fields.
3. Consume them in the recipe layer — primarily for `image_size`.

## 4. The convergence point: `image_size`

`image_size = f(backbone constraints, data resolution, task fine-grainedness)`
— DINOv2 needs multiples of 14; EfficientNet-B3 wants 300; Module 2 gives the data
resolution; Module 1 gives fine-grainedness. It is both a hyperparameter (§2) and a
consumer of Module 2 data (§3). Getting it right also matters for speed (image decode was
the training bottleneck on Colab).

## 5. Recipe layer — proposed v0

A small **recipe module** that Module 3 calls, mapping
`(backbone family, finetune_strategy, data_size, image stats)` → HPs.

- **v0 scope: just `image_size` + `learning_rate`** — smallest slice with a complete loop
  (Module 2 → Module 3 plumbing + recipe + Module 4 hand-off). Validate the chain, then add
  scheduler / epochs / augmentation.
- Open question: live inside Module 3, or as a standalone module Module 3 calls (preferred —
  isolated, testable, independently iterable).

## 6. Make it measurable — catalog as eval harness

`vision_benchmark_catalog.py` (14 datasets + baselines) is a ready-made evaluation set.
Stand up a harness that runs "Module 3 pick → train → compare to baseline" so every change
to selection or the recipe is judged by **real scores**, not hand-tuning. Do this early so
§1–§5 are data-driven.

## Suggested order

1. Wire unused signals (esp. Module 2 image stats → image_size) + cost-aware tiebreak.
2. Recipe layer v0 (image_size + lr).
3. Catalog eval harness (in parallel, so 1–2 are measurable).
