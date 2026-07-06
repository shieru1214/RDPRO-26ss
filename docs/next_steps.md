# Next steps / open decisions

Planning notes for the accumulating-recommender direction (branch:
`integration-recommender`). The thesis: occupy the corner MLE-STAR structurally can't —
**cheap, explainable, constraint-aware, accumulating** model recommendation — rather than
out-searching it.

## Recently landed

- **Recipe layer v0 is integrated** (`recipe/`, Module 3, Module 4). Natural-language
  classification candidates now receive deterministic, model/data-aware defaults for
  `image_size`, `learning_rate`, `epochs`, and `augmentation`, plus provenance.
  Module 4 consumes those defaults when the incoming config does not explicitly
  override them.
- **Loss-imbalance A/B foundation is implemented** (`experiments/ab_loss_imbalance/`).
  The CE-vs-focal comparison has a frozen matrix, paired stratified 5-fold injection,
  validation-prediction export, resume-safe run driver, and pure verdict collection.
- **Latest local regression status**: `302 passed` with
  `--ignore=ingestion/test_dataset.py`; the skipped collection point requires the
  optional `datasets` package in the current environment.

## Open decision: ensembles (needs discussion)

Transformer/foundation-model era rarely uses explicit ensembles, because finetunes from the
**same pretrained checkpoint are highly correlated** → little diversity → little ensemble
(and model-soup) gain. Remaining ensemble value is in **cross-architecture / cross-pretraining**
diversity, which Module 3's heterogeneous candidate pool already provides.

Three options (undecided):

- **A. Build ensembles as a niche feature** — but framed as *cross-architecture complementary*
  ensembles (greedy selection over KB candidates of different family/pretraining, diversity from
  KB metadata + cheap LogME/probe predictions). Under a single-model deployment budget, collapse
  same-arch members into a soup.
- **B. Don't reinvest in ensembles** — focus on strong single-model selection + near-free recipe
  upgrades (EMA/SWA) + constraint-awareness. Follows where the field is going.
- **C. Ensemble knowledge as a selection rule** — Jiaozi decides *whether a task is worth
  ensembling* (small data / CNN / generous budget → recommend cross-arch ensemble; big model /
  tight budget → single strong model + EMA). The "judgement" itself is the product.

Leaning: **C + B** (Jiaozi knows when to ensemble; defaults to single strong model + EMA).
Model soup is demoted to an optional product of the ensemble mode under tight deployment budgets,
not a headline.

## Recipe layer (hyperparameters)

- **v0 — landed** (`recipe/`). Rules: checkpoint/family image-size defaults, divisor snapping,
  high-resolution fine-grained bumping, learning rate by architecture family and training mode,
  epochs by data size and training mode, and augmentation by data size / image statistics /
  constraints. The layer is now attached in the normal NL retrieval path, not only through the
  old `--use-recipe` side path.
- **v1 — reserved** (`_llm_recipe_proposal` stub): an LLM proposes the soft HPs grounded by the
  outcome memory, validated against the rule recipe's ranges (rules are the guardrailed floor).
  One call, not a search loop. Expectation: a strong "practitioner first-guess" config, improving
  via memory — measure it with the AIDE harness (rules vs LLM-recipe vs AIDE, quality-vs-cost).
- **later**: weight decay, warmup, EMA/SWA switches, and recipe promotion from deterministic
  defaults into learned defaults once enough outcomes exist.

## Pending build items

1. **Run the loss-imbalance A/B for real** — execute both frozen testbeds on Kaggle/GPU,
   collect the paired verdict, and only then decide whether CE or focal should become the KB
   default for class-imbalance classification.
2. **EMA in generated training** (high value, low risk, no cost tension) — add exponential weight
   averaging to the Module 4 train template: single run, single model, near-free quality gain.
   Strictly better than model soup on the cost axis.
3. **Wire LogME to real data** — `logme_score` is a pure metric; extract frozen features via
   `model_utils` on a data sample in the pipeline to produce `{backbone: logme}` for the cold-start
   ranker (and, later, per-model val predictions for ensemble selection).
4. **Cold-start seeding** — seed the outcome memory from public transfer benchmarks (timm / VTAB)
   or a few real `run_and_log` runs.
5. **AIDE comparison harness** — run AIDE on the same tasks under the same budget, log its quality
   + cost into the same table → the quality-vs-cost Pareto plot (the selling-point figure).
6. **Automated KB updating** (companion pillar) — ingest new backbones/checkpoints from HF /
   papers-with-code into the persistent structured graph, so the recommender isn't capped by a
   narrow hand-curated KB.

## Validation still pending
- Full `pytest` once `datasets` is installed for `ingestion/test_dataset.py`.
- Real Kaggle/GPU results for the loss-imbalance A/B, followed by a KB action if the verdict is
  decisive.
- Recommender "improves as used" — needs real logged runs.
- Model soup Phase 0 — does it actually beat the best single on real data? (branch `feature/model-soup`)
- Constraint-aware selection vs ground truth (HW-NAS-Bench style) / vs AIDE.
