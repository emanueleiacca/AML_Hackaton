# 🧭 RoBERTa → DINOv2 “BigJump” Pipeline Explained 

**Goal:** Learn a translator that maps text embeddings (RoBERTa) → image latents (DINOv2)  
and maximizes retrieval MRR. The training process combines geometry initialization,  
multi-positive InfoNCE, and several stabilizers for robust convergence.

---

## 1️⃣ Data Loading & Split (no leakage)

- Load cached embeddings: **RoBERTa (1024-d)** for captions, **DINOv2 (1536-d)** for images.  
- **Split by image ID**: all captions of the same image go to the same fold.  

---

## 2️⃣ Geometry Initialization + Adapter

- Compute **Procrustes closed-form** linear map $(A,b)$:
  $$
  \min_{A,b}\;\lVert A X + b - Y\rVert_2^2
  $$
  with whitening $\varepsilon$ on covariances for stability.  
- Wrap it into `GeomWithAdapter` (hidden=1024, dropout=0.1):
  - Linear base $(A,b)$ for coarse alignment  
  - Nonlinear adapter for fine tuning  
- Start with geometry **frozen** (only adapter trainable).

---

## 3️⃣ Optimizer + Scheduler

- **AdamW** with two parameter groups:
  - Adapter: base LR  
  - Geometry: lower LR (later unfrozen)  
- **LR schedule:** 1-epoch warmup → cosine decay to 0.1×  
- **Grad clip:** 1.0 | **AMP:** on

---

## 4️⃣ Queue Warmup (negatives off at start)

- Initialize an **image FIFO queue** of DINO features (`queue_size = 65,536`).  
- Epochs 1–2: only **enqueue**, not used in loss.  
  *Reason:* prevent unstable negatives before translator aligns.

---

## 5️⃣ Main Training Loop

### Step (a): Forward
Predict DINO features from text embeddings → `pred`.

### Step (b): Multi-Positive InfoNCE
For each caption *i*, positives = all captions of the **same image**;  
negatives = other batch images + recent queue items.

$$
\mathcal{L}_{\text{NCE}}(i)
= -\log
  \frac{
  \sum_{p\in\mathcal{P}_i} \exp\!\left(\frac{s(\mathbf{z}_i,\mathbf{y}_p)}{\tau}\right)
  }{
  \sum_{a\in\mathcal{A}_i} \exp\!\left(\frac{s(\mathbf{z}_i,\mathbf{y}_a)}{\tau}\right)
  },
$$
where $s(\mathbf{z},\mathbf{y}) = \frac{\mathbf{z}\cdot\mathbf{y}}{\|\mathbf{z}\|\|\mathbf{y}\|}$.

---

### Step (c): Add Stabilizers

1. **Cosine alignment**
   $$
   \mathcal{L}_{\cos}
   = \tfrac{1}{B}\sum_i (1 - \cos(\mathbf{z}_i,\mathbf{y}_i))
   $$
2. **Moment alignment**
   $$
   \mathcal{L}_{\text{moment}}
   = \lVert \mu_{\mathbf{z}} - \mu_{\mathbf{y}}\rVert_2^2,\quad
   \mu_{\mathbf{z}}=\tfrac{1}{B}\sum_i \mathbf{z}_i
   $$
3. **Caption agreement**
   $$
   \mathcal{L}_{\text{agree}}
   = \frac{1}{|\mathcal{G}|}
     \sum_{g\in\mathcal{G}}
     \operatorname{mean}\!\bigl(\operatorname{Var}(\mathbf{z}_g,\text{dim}=0)\bigr)
   $$
   where $\mathcal{G}$ are groups of captions referring to the same image.

---

### Step (d): Total Loss
$$
\mathcal{L}
= \mathcal{L}_{\text{NCE}}
+ \alpha\,\mathcal{L}_{\cos}
+ \lambda_{\text{moment}}\,\mathcal{L}_{\text{moment}}
+ \lambda_{\text{agree}}\,\mathcal{L}_{\text{agree}}
$$

Default weights:  
- $\alpha=0.5$ (range 0.3–0.7)  
- $\lambda_{\text{moment}}=0.02$ (range 0–0.05)  
- $\lambda_{\text{agree}}=0.05$ (range 0.02–0.10)

---

## 6️⃣ Temperature Curriculum

If `--tau` is not fixed:
$$
\tau_e = \tau_{\text{end}} + (\tau_{\text{start}} - \tau_{\text{end}})
          \tfrac{1}{2}(1+\cos(\pi\,t_e))
$$
with $t_e = \frac{e-1}{E-1}$  
Default: $\tau_{\text{start}}=0.10 \to \tau_{\text{end}}=0.06$

---

## 7️⃣ Queue Policy (after warmup)

After 2 epochs, include queue negatives with **recent slice** sizes:
16k → 32k → 65k.  
This ensures harder, up-to-date negatives for sharper ranks.

---

## 8️⃣ Micro-Unfreeze Geometry (Epoch 3)

- Unfreeze only matrix **A**, not bias **b**.  
- Apply **tiny LR** scale (×0.05).  
- If validation MRR drops >0.005 → refreeze geometry.

---

## 9️⃣ Validation & Checkpointing

- Evaluate on **val captions → val images** (no leakage).  
- Log:
  - **MRR (primary)**
  - R@1/5/10
  - median, p75
- Save best checkpoint and metrics JSON.

---

## 🔟 Inference & Submission

- L2-normalize outputs → cosine retrieval.  
- Generate submission CSV with test caption embeddings.

---

## ⚙️ Key Defaults

| Hyperparameter | Default | Range / Notes |
|----------------|----------|----------------|
| batch | 512 | scale by VRAM |
| queue size | 65,536 | 32k–262k |
| queue warmup | 2 epochs | exclude from loss |
| $\tau$ start/end | 0.10 → 0.06 | soft→sharp curriculum |
| $\alpha$ (cosine) | 0.5 | 0.3–0.7 |
| $\lambda_{\text{moment}}$ | 0.02 | 0–0.05 |
| $\lambda_{\text{agree}}$ | 0.05 | 0.02–0.10 |
| geom unfreeze epoch | 3 | 0 disables |
| geom lr scale | 0.05 | 0.02–0.10 |

---

## 💡 Why It Works (for MRR)

1. **Geometry init** gives an aligned starting point.  
2. **Adapter-only early training** stabilizes learning.  
3. **Queue curriculum** provides a controlled explosion of negatives.  
4. **Caption agreement** collapses same-image captions, removing ambiguity.  
5. **Temperature decay** sharpens late discrimination.  
6. **Tiny geometry tuning** adds final precision without chaos.  

→ The result: *tight intra-image clusters* + *clean global ranking* → high **MRR**.
