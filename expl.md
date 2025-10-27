%%markdown
# **runner.py — Technical Walkthrough (MRR-Focused)**

**Goal.** Map fixed text embeddings (RoBERTa, 1024-D) to fixed image latents (DINOv2, 1536-D) so that cosine retrieval on a *val gallery of images* maximizes **MRR** (primary) and Recall@K (secondary). Accuracy first, then efficiency.

---

## **1) Data, Metadata & Determinism**

**Inputs**
- `train.npz`:  
  - `captions/embeddings ∈ ℝ^{N_text×1024}`  
  - `images/embeddings ∈ ℝ^{N_img×1536}`  
  - optional: `captions/ids`, `images/names`
- `captions.txt`: one line per caption → contains target image filename/stem.

**Checks**
- `_assert_metadata_dims()` enforces 1024→1536.
- `seed_all(s)` fixes PRNGs and disables CuDNN autotune for full determinism.

---

## **2) Caption→Image Targets (Robust Matching)**

The script builds lookup tables (path, basename, stem) for image matching.  
Each caption line (split by `|`, `,`, or tabs) is matched to its image → index `target`.

Resulting paired matrices:
$$
X = \text{captions/embeddings} ∈ ℝ^{N×1024}, \quad
Y = I[\text{targets}] ∈ ℝ^{N×1536}
$$
Each **row of $Y$** is the ground-truth image vector for that caption.

---

## **3) Split by Image ID (No Leakage)**

`build_image_id_split(...)`:
- randomly selects **validation images** (ratio `val_ratio`)
- moves **all captions** of those images to validation
- builds `val_gallery ∈ ℝ^{M×1536}` and index map `cap2gal_local`

All captions of a single image remain in the same split → no leakage.

---

## **4) Model Families**

- **LinearProj:** affine layer $ \hat y = W x + b $
- **MLP1/MLP2:** shallow MLPs with ReLU + dropout
- **GeomLinear:** geometry-preserving closed-form map (Whiten→Procrustes→Recolor)

### **4.1 Geometry-Preserving Linear Map**

Centered data:
$$
X_c = X - μ_x, \quad Y_c = Y - μ_y
$$

Covariances:
$$
C_x = \tfrac{1}{N} X_c^⊤ X_c, \quad C_y = \tfrac{1}{N} Y_c^⊤ Y_c
$$

Eigen decompositions:
$$
C_x = U_x S_x U_x^⊤, \quad C_y = U_y S_y U_y^⊤
$$

Whitening:
$$
C_x^{-1/2} = U_x (S_x + εI)^{-1/2} U_x^⊤, \quad
C_y^{-1/2},\ C_y^{+1/2} \text{ analogously.}
$$

**Whitened data**
$$
X_w = X_c C_x^{-1/2}, \quad Y_w = Y_c C_y^{-1/2}
$$

**Orthogonal Procrustes:**
$$
\max_R \mathrm{Tr}(R^⊤ X_w^⊤ Y_w), \quad \text{s.t. } R^⊤R=I
$$
If $M=X_w^⊤Y_w=UΣV^⊤$, then $R=VU^⊤$.

**Recoloring:**
$$
A = C_y^{+1/2} R C_x^{-1/2}, \quad b = μ_y - A μ_x
$$
Final linear map: $\hat y = A x + b$

Preserves mean + covariance alignment → improves cosine geometry.

---

## **5) Training Objective**

Combined loss:
$$
\mathcal{L} =
α(1 - \cos(\hat y_i, y_i))
+ β\|\hat y_i - y_i\|_2^2
+ λ\mathcal{L}_{moment}
+ γ\mathcal{L}_{InfoNCE}
$$

- **Moment:** $\|\mu(\hat Y)-\mu(Y)\|_2^2 + \|\sigma(\hat Y)-\sigma(Y)\|_2^2$
- **InfoNCE:** logits $L=\hat Y Y^⊤$ with diagonal positives  
  (only if `γ>0` and large batch)

Defaults: `α=1`, `β=1`, `λ=0.05`, `γ=0`.

---

> while beautifully efficient and theoretically elegant, is not the most powerful approach in this challenge.

What Geometry Solves — and What It Doesn’t

It solves:

Global distribution mismatch (mean, variance, orientation).

Aligns second-order statistics (covariances).

Works instantly (no training, no overfitting).

It doesn’t solve:

Non-linear correlations (text semantics ↔ visual patterns).

Fine-grained alignment (e.g., “a cat under a table” vs “a cat on a table”).

Multi-caption per image variability (captions describing same image differently).

Distribution shifts between train/val/test.

So it’s great for geometry, but blind to semantic structure — it’s like aligning the “clouds” but ignoring individual caption–image subtleties.

---

## **6) Validation (MRR & Recall@K)**

Queries = val captions, Gallery = val images.  
Compute cosine similarities on L2-normalized vectors.

For each query:
$$
\text{MRR} = \frac{1}{N_{val}} \sum_i \frac{1}{\text{rank}_i}, \quad
\text{Recall@K} = \frac{1}{N_{val}} \sum_i \mathbb{1}[\text{rank}_i ≤ K]
$$

Also logs median and 75th-percentile rank.

---

## **7) Efficiency Metrics**

- Params & Memory (FP32 = 4 bytes/param)  
- Latency (ms/query) on GPU & CPU using synthetic batch.

---
