# RoBERTa → DINOv2 Translator (BigJump++) — Readme for Humans 🧠➡️🖼️

**Goal (what we optimize):** turn each caption embedding $x\in\mathbb{R}^{1024}$ (from `roberta-large-nli-stsb-mean-tokens`) into a DINOv2 image embedding $\hat y\in\mathbb{R}^{1536}$ so that, for *validation captions vs validation images*, the true image ranks as high as possible by cosine similarity. The scoreboard is **MRR** (Mean Reciprocal Rank).

MRR: $ \text{MRR}=\frac{1}{N}\sum_{i=1}^N \frac{1}{\text{rank}_i} $  
Similarity: cosine on **L2-normalized** vectors (both predicted and gallery).

---

## TL;DR of the Workflow

1) **Load & sanity check data**
   - Read the single `train.npz` + `captions.txt` for train; read test NPZ later for submission.
   - Check the embedding dimensions are exactly **text=1024**, **image=1536** (no wrong encoders).
   - Parse `captions.txt` to match each caption row to the correct image **by name/stem** (robust to prefixes like `Images/`).

2) **Build per-caption targets (supervision)**
   - For each caption row, pick the **corresponding image embedding** $y\in\mathbb{R}^{1536}$ from the image table.  
   - This yields training pairs $(x, y)$ with **one image vector per caption** (even if multiple captions share the same image).

3) **Split by *image id* (no leakage)**
   - Randomly choose a **set of unique images** for validation (default 10%).
   - **Train captions** = all captions whose image is in the train set.  
     **Val captions** = all captions whose image is in the val set.  
   - **Validation gallery** = unique val images only.  
   - For each val caption we pre-compute the **local index** of its true image in that gallery, so ranking is straightforward.

4) **Model architectures (you can swap these)**
   - **Linear / MLP1 / MLP2:** standard heads, randomly initialized (Xavier/He).
   - **Geom**: a *geometry-preserving* linear map $\hat y = A x + b$ initialized **in closed form** (Procrustes-like):
     - **Center** $X$ and $Y$ → **Whiten** both → **Find rotation** with SVD → **Re-color** to image space → build $A,b$.
   - **Geom+Adapter / BigJump (used here):** $\hat y = \underbrace{A x + b}_{\text{GeomLinear (frozen at start)}} \;+\; \underbrace{f_\theta(x)}_{\text{small residual MLP}}$  
     The residual learns the details the closed-form can’t capture.

5) **Training objective (MRR-oriented via contrastive retrieval)**
   - **Primary loss:** multi-positive **InfoNCE** with a big **negatives queue** and **extra positives** when multiple captions share an image.
     - Normalize predictions and targets, compute logits vs a **bank** (in-batch targets + queue; optional hard-mined subset).
     - **Cross-batch positives (XBP):** we store recent predictions per image so a caption sees **more than one positive** for the same image.
     - **Hard mining:** from recent queue slice, pick top-$H$ hard negatives per anchor.
     - Optional **Debiased NT-Xent** (off by default).
   - **Aux losses:**  
     - **Cosine**: average $(1-\cos)$ with the true $y$.  
     - **Moment matching**: align per-dimension means/variances of $\hat y$ and $y$ (stabilizes distribution).  
     - **Caption agreement**: if an image has multiple captions in a batch, make their predictions **agree** (variance penalty).
   - **Temperature curriculum**: if `--tau` is None, we smoothly **decrease $\tau$** from 0.10 → 0.06 across epochs.
   - **Queue warm-up**: first few epochs **exclude** the negatives queue from the loss (but still fill it), then **ramp** the “recent slice” size: 16k → 32k → 65k.

6) **Evaluation (every epoch)**
   - Predict $\hat y$ for **val captions**, L2-normalize, compute cosine vs the **val gallery** (unique val images).
   - Record **MRR (primary)**, **Recall@1/5/10**, and **rank distribution** (median, 75th percentile).

7) **Efficiency tracking**
   - Count params, model size (MB), and report **ms/query** on GPU & CPU with a big dummy batch.

8) **Submission**
   - Load **test captions** only, predict $\hat y$ for each, **L2-normalize**, and write `submission.csv` with the required format.

---

## What Actually Ran in *your* logs

- **Architecture:** `bigjump` → **GeomWithAdapter** (closed-form $A,b$ + residual MLP).
- **Queues & extras:**  
  - Negatives **ImgQueue=65,536** (used after warm-up).  
  - **XBP (cross-batch positives)** = ON (per image 4, global 32k).  
  - **Hard mining**: top-**H=64** from the recent queue slice (after warm-up).  
  - **Debiased NT-Xent (DCL)** = **OFF**.  
  - **PredQueue (dual queue of past predictions)** = **OFF**.  
  - **P×K sampler** = **OFF** (standard shuffled batches).
- **Curriculum:** warm-up epochs = 2 (queue excluded), then recentQ = 16k → 32k → 65,536.  
  **$\tau$** annealed from 0.10 to 0.06.
- **Metrics (best epoch ~24–25):** `MRR ≈ 0.424`, `R@1 ≈ 0.290`, `R@5 ≈ 0.578`, `R@10 ≈ 0.697`, median rank ≈ 4, p75 ≈ 14–15.
- **Efficiency:** ~**4.2M params** (~**16 MB**), **~0.0016 ms/query (GPU)**, **~0.053 ms/query (CPU)** on the synthetic timing.

---

## The Pieces in Plain Language

### A) Data plumbing (safe & deterministic)
- **Why the metadata check?** To guarantee we’re mapping **1024 → 1536**. If numbers differ, it means encoders changed — that would make results incomparable (bad for leaderboard & science).
- **Why match captions via file names?** The NPZ stores image embeddings in one array and captions in another. We use `captions.txt` to **line up** each caption row with the correct image row (building the supervision pairs).
- **Why split by image id?** If the same image appears in both train and val, we’d “peek” at it (data leakage). We *only* allow **different images** in val.

### B) Geometry init (closed-form warm start)
- Before learning anything, we compute a **best-fit linear map** $A,b$ that roughly aligns the *global shape* of $X$ to $Y$:
  1. Remove means (center $X$ and $Y$).
  2. Whiten covariances (so axes have unit variance).
  3. Find a rotation that best aligns the whitened spaces (via SVD).
  4. Re-color back to the image covariance; add the bias $b$.
- This gives a strong, **stable starting point**. Then the **small residual MLP** learns the fine-grained corrections that matter for ranking.

### C) Contrastive training (how we “teach” retrieval)
- For a batch, predict $\hat y$ for each caption.  
  We then build a **bank** of vectors to compare with:
  - **Positives:** all targets belonging to the **same image** (in-batch), plus **recent predictions** for that image (XBP).
  - **Negatives:** many **other** images from the queue (and optionally hard-mined top-$H$).
- We nudge the model so **positives get higher cosine than negatives**, at a temperature $\tau$ that we gradually lower to make the ranking sharper.
- **Aux terms** keep the distribution healthy and make **multiple captions of the same image agree**.

### D) Validation like the official eval
- Queries = **val captions**.  
  Gallery = **unique val images** only.  
  Score = **cosine on L2-normalized vectors**.  
  Track **MRR** as the main metric, plus Recall@K and rank percentiles for intuition.

---

## Default knobs (what to tweak first)

- **Batch size:** `--batch=512` (try 256–1024 depending on GPU).  
- **Epochs:** `--epochs=25` (range 15–40).  
- **Base LR:** `--lr=2e-4` (range 1e-4–5e-4).  
- **Tau curriculum:** `--tau_start=0.10`, `--tau_end=0.06` (a bit steeper helps in later epochs).  
- **Queue warm-up:** `--queue_warmup_epochs=2` (try 1–3).  
- **Recent queue sizes:** `[16000, 32000, 65536]` (keep the last large).  
- **Hard mining H:** `--mine_H=64` (try 32–128).  
- **Agreement loss:** `--lambda_agree=0.05` (try 0.02–0.10).  
- **Moment matching:** `--moment=0.02` (try 0.01–0.05).  
- **XBP:** per-image=4, global=32k (raise per-image to 6–8 if you often batch same-image captions).

---

## Important gotchas (please read)

- **Geometry unfreeze bug:** the code tries to “unfreeze only $A$ at epoch 3” by checking parameter names that **contain `'A'`**.  
  In PyTorch, linear params are named `'weight'` and `'bias'` — so **nothing gets unfrozen**.  
  *Effect:* Geometry stayed **frozen** the whole time; you trained **only the adapter**.  
  **Fix idea:** either unfreeze `geom.fc.weight` (and optionally keep `geom.fc.bias` frozen), or just unfreeze both weight & bias with the tiny LR scale.  
  (This could give you a few extra MRR points.)

- **Pooling is a no-op:** `apply_pooling` returns input unchanged. That’s fine (we don’t have patches here), but don’t expect it to do anything.

- **PK Sampler is off:** If you want **more same-image captions per batch**, enable `--use_pk --P 256 --K_pk 2` (or tune `K_pk` up to 3–4 if memory allows). This can strengthen agreement learning and the multi-positive effect.

- **DCL is off:** `--use_dcl` can help when positives are numerous; test it (start with `dcl_prior=0.01`).

---

## Which functions/classes you *actually used* in the run

**Used (directly or via BigJump path):**
- `seed_all`
- `_assert_metadata_dims`, `_build_image_index`, `_iter_targets_from_captions`, `load_train`
- `procrustes_closed_form`, `_cov_eigh`
- `GeomLinear` (as part of `GeomWithAdapter`)
- `ResidualAdapter`, `GeomWithAdapter`, `make_model`
- `ImgQueue`, `XBPBuffer`, `mine_hard` (because `mine_H=64` and queue active), `info_nce_multi`
- `agreement_loss`, `moment_align`
- `build_image_id_split`
- `validate_retrieval`
- `count_params_mb`, `time_ms_per_query`
- `train_bigjump`
- `main` (CLI entry)
- `generate_submission` (from `challenge.src.common`)

**Defined but **not** used in your logged run (kept for alternatives/legacy):**
- Architectures: `LinearProj`, `MLP1`, `MLP2` (you ran `bigjump` instead)
- Legacy training: `PairDS`, `train_one`, `info_nce` (single-positive), `load_test_npz`
- Queues/extras that were disabled by flags:  
  - `PredQueue` (dual queue) — **not used** (`--queue_pred` was false)  
  - Debiased denominator `_debiased_logZ` — **not used** (`--use_dcl` false)
- `apply_pooling` (is called, but it’s a **no-op**)

---

## Checklist to replicate or iterate

- [ ] Keep **encoders fixed** and check dims: text=1024, image=1536.  
- [ ] Split by **image id**, not by rows; ensure train/val sets are disjoint in images.  
- [ ] Use **Geom+Adapter** init; confirm $A,b$ are actually applied.  
- [ ] **Fix geometry unfreeze** if you want to fine-tune $A$ later with tiny LR (e.g., `geom_lr_scale=0.05`).  
- [ ] Enable **negatives queue** with warm-up and increasing **recent slice**.  
- [ ] Keep **XBP** ON; increase per-image to 6–8 if batches often contain repeated images.  
- [ ] Try **PK sampling** to guarantee multiple captions per image in-batch.  
- [ ] Tune **$\tau$** (final around 0.05–0.07 often works).  
- [ ] Track **MRR** primary + **R@1/5/10** and **rank median/p75**.  
- [ ] Export **L2-normalized** predictions for submission.

---

## Why this should maximize MRR (intuition)

- The **closed-form geometry** gets you near the right *global shape* (good starting ranks).  
- The **residual adapter** and **contrastive loss** then focus on **relative ordering**: positives higher than a *large set* of negatives, which is directly aligned with ranking metrics.  
- **Queues** give you **many diverse negatives**; **hard mining** surfaces the tricky ones; **XBP** gives **multiple positives** when an image has many captions; **agreement** ensures those captions point to a **single** image representation.  
- The **temperature schedule** sharpens decisions late in training.

---

## Repro block from your log (for context)

- `arch=bigjump`, `epochs=25`, `batch=512`, `lr=2e-4`, `wd=1e-4`  
- `tau: 0.10→0.06`, `queue_warmup_epochs=2`, `queue_recent=[16k,32k,65,536]`  
- `mine_H=64`, `lambda_agree=0.05`, `moment=0.02`  
- `use_pk: False`, `use_dcl: False`, `queue_pred: False`  
- **Best MRR ≈ 0.424**; `R@1 ≈ 0.290`; median rank ≈ 4.

---

*If you want, I can write a tiny patch snippet to correctly unfreeze only the geometry weight at epoch 3 and keep its LR scaled down. Just say “patch the unfreeze” and I’ll add the exact code change.*
