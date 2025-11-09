# Model B — Optimizations for TransReID (Losses, Sampling & Optional CLIP)

> This repo extends **TransReID (DeiT/ViT + SIE + JPM)** primarily with **training-side** improvements: loss functions and sampling.
> **Additionally**, we provide **optional CLIP semantic fusion** that can be enabled **at train and/or test time** without changing the backbone or heads.

---


## 1) Method Overview (aligned with Report §7.2.1)

**Runtime behavior (important).** Using `run_phased_loss.py` will, by default, **train and then automatically run external `test.py`** for each newly saved checkpoint (per-epoch backfill; already tested epochs are skipped).

* **Train-only:** add `--train-only` (alias of `--no-test`); this keeps in-training eval but skips external tests.
* **Re-test a specific epoch:** delete the corresponding `..._test/epoch_X/` folder and rerun the launcher.

### 1.1 SupCon (Supervised Contrastive Learning)

* Operates on **bnneck** features by default (switchable via `LOSS.SUPCON.FEAT_SRC ∈ {pre_bn, bnneck}`).
* **Camera-aware positives:** `POS_RULE='class'` with `CAM_AWARE=True`.
* Default hyper-params: **T=0.07**, **W=0.30** (override via YAML/CLI).
* Trains jointly with Triplet/TripletX to improve intra-class compactness and cross-camera consistency.

### 1.2 Conditional P×K Sampling (Loss-Coupled)

* If **TripletX** or **SupCon** is active → **K=8**; otherwise **K=4**.
* Rationale: increase same-ID density and hard-sample coverage when structured discriminative losses are enabled.

### 1.3 Phase Scheduling (optional)

* Recommended schedule: **A (Triplet) → B (TripletX + SupCon) → C (Triplet + SupCon)**.
* Rebuilds the DataLoader at phase boundaries and switches sampling scale / loss weights
  (see `LOSS.PHASED.*`, including `TRIPLETX_END` to gracefully retire TripletX).

### 1.4 Threshold-Driven Multi-Loss Routing

* Split by grayscale mean threshold **τ**:
  `μ(I) < τ → TripletX` (robust to low illumination), `μ(I) ≥ τ → Triplet`.
* Implemented **on the loss side only**; **constant weights** by default (no learned/adaptive re-weighting).
* Can co-exist with SupCon and Conditional P×K.
* To isolate illumination, **disable phase scheduling** and keep **constant weights**; or disable **TripletX** under matched settings as a control. If SupCon is used, keep **W=0.30**, **T=0.07** (no extra dark-sample up-weighting).

### 1.5 CLIP Semantic Fusion (optional; affects train **and** test)

* Unified switch: `MODEL.USE_CLIP`; implementation: `MODEL.CLIP_IMPL ∈ {open_clip, hf}`; fusion: `MODEL.CLIP_FUSION ∈ {minimal, quickwin}`.

  * **OpenCLIP** (ViT-B/16, LAION2B): `minimal` is lightweight global fusion; `quickwin` can also touch local/JPM.
  * **TinyCLIP (HF)**: compact vision tower with `CLIP_LR_SCALE` (e.g., 0.2) and optional `AFEM/sem_refine`.
* **Note:** CLIP fusion is optional and non-intrusive (no backbone/head changes). You may **train with fusion, test without** (to study transfer), or **enable both** at train and test for end-to-end consistency.

---

## 2) Key Config Mapping (YAML → Behavior)

* **SupCon**
  `LOSS.SUPCON.ENABLE=True`
  `LOSS.SUPCON.{T,W,FEAT_SRC,CAM_AWARE,POS_RULE}`

* **Conditional P×K**
  `DATALOADER.PHASED.ENABLE=True`
  `DATALOADER.PHASED.{K_WHEN_TRIPLETX=8, K_WHEN_SUPCON=8, K_OTHER=4}`

* **Phase Scheduling**
  `LOSS.PHASED.ENABLE=True`
  `LOSS.PHASED.{BOUNDARIES, METRIC_SEQ, W_METRIC_SEQ, W_SUP_SPEC, TRIPLETX_END}`

* **Threshold-Driven Multi-Loss Routing (τ)**
  `LOSS.BRIGHTNESS.ENABLE=True`
  `LOSS.BRIGHTNESS.THRESH=0.35` (example τ)
  `LOSS.BRIGHTNESS.K=0.08`
  *Note:* the historical name of this block is `BRIGHTNESS`; in text we call it **threshold-driven multi-loss routing**.

* **Feature Sources**
  `LOSS.TRIPLET.FEAT_SRC='pre_bn'`
  `LOSS.TRIPLETX.FEAT_SRC='pre_bn'`

* **CLIP (optional; controllable at train and test)**
  `MODEL.USE_CLIP`, `MODEL.CLIP_IMPL={'open_clip'|'hf'}`, `MODEL.CLIP_FUSION={'minimal'|'quickwin'}`
  plus `MODEL.CLIP_BACKBONE`, `MODEL.CLIP_PRETRAIN`, `MODEL.CLIP_HF_ID`, etc.

  * **Train with CLIP, test without:** set `MODEL.USE_CLIP=False` in the test YAML.
  * **Train and test with CLIP:** keep the same CLIP config on both sides.

---

## 3) Environment & Data

- Example layout:

```text
Project Root
├─ configs/
│  └─ VeRi/           # YAMLs (baseline, pk=8, SupCon/TripletX, CLIP, phased)
├─ datasets/          # Dataset loaders (VeRi-776, Market, Duke, VehicleID)
├─ losses/            # Loss implementations (triplet, tripletx, supcon, make_loss)
├─ model/             # TransReID backbones & make_model
├─ processor/         # Training loop, phase scheduling & PK rebuild hooks
├─ solver/            # LR schedulers, optimizers
├─ utils/             # Logger, metrics, reranking, plotting
├─ figs/              # (optional) figures
├─ pretrained/        # (placeholder) ImageNet/CLIP weights
├─ logs/              # training & eval outputs
├─ run_modelB_deit.py # Model B training entry
├─ run_phased_loss.py # Multi-seed launcher + best-epoch summarizer
├─ test.py            # Standalone evaluation
├─ requirements.txt
├─ LICENSE
└─ README.md
````

---

## 4) Environment & Requirements

### Install with `requirements.txt`



```bash
# from the repo root (where requirements.txt lives)
pip install -r requirements.txt
```

**Notes**

* **Apple Silicon (M1/M2/M3, macOS + MPS):** if PyTorch isn’t pinned in `requirements.txt`, install it first:

  ```bash
  pip install torch torchvision torchaudio
  ```
* **NVIDIA GPU (CUDA):** install the matching wheel for your CUDA version, e.g. CUDA 12.1:

  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

**Optional (only if you run CLIP/TinyCLIP configs):**

```bash
# OpenCLIP
pip install open-clip-torch==2.24.0
# TinyCLIP / HF
pip install "transformers>=4.41" "huggingface-hub>=0.23" accelerate
```

**Verify install**

```bash
python - <<'PY'
import torch
print("Torch:", torch.__version__,
      "| CUDA:", torch.cuda.is_available(),
      "| MPS:", torch.backends.mps.is_available())
PY
```


## 4.1 Quick Start (Generic Launcher)

All training runs use the same launcher:

```bash
python run_phased_loss.py \
  --config configs/VeRi/deit_transreid_stride_baseline.yml \
  --seeds 0 --epochs 120 --save-every 3 \
  --output-root ./logs
# Optional overrides:
#  --opts DATASETS.ROOT_DIR ./datasets
```

> **Default behavior:** the launcher **automatically** runs external `test.py` for **each new checkpoint**.
> Use `--train-only` (or `--no-test`) to skip external tests while keeping in-training eval.

**Runtime toggles (do not edit YAML unless you want to):**
`--supcon / --no-supcon`, `--tripletx / --no-tripletx`, `--phased / --baseline`.

---

## 4.2 Triplet + SupCon (pre_bn routing, PK=8, joint losses)

```bash
python run_phased_loss.py \
  --config configs/VeRi/deit_transreid_stride_ce_triplet_pk8_prebn_supcon_prebn.yml \
  --seeds 0 --epochs 120 --save-every 3 \
  --output-root ./logs
# Optional:
#  --opts DATASETS.ROOT_DIR ./datasets
```

**What this config demonstrates**

* **Feature routing**: both **Triplet** and **SupCon** use `FEAT_SRC=pre_bn` (explicitly aligned feature paths).
* **Joint losses**: Triplet × SupCon trained together (weights as in YAML).
* **Conditional P×K**: K auto-switching to **8** when structured losses are active.
* **Default behavior**: per-epoch external tests are run automatically after training.



---

## 4.3 Conditional P×K Sampling (auto K=8 with TripletX/SupCon) & Phase Schedule A→B→C 

```bash
python run_phased_loss.py \
  --config configs/VeRi/deit_transreid_stride_A_ce_triplet_B_ce_tripletx_supcon_C_ce_triplet_supcon.yml \
  --seeds 0 --epochs 120 --save-every 3 \
  --output-root ./logs
# Optional overrides:
#  --opts DATALOADER.PHASED.K_WHEN_TRIPLETX 8 \
#        DATALOADER.PHASED.K_WHEN_SUPCON 8 \
#        DATALOADER.PHASED.K_OTHER 4 \
#        DATASETS.ROOT_DIR ./datasets
```

> When **TripletX** or **SupCon** is active, the loader switches to **K=8**; otherwise **K=4**.
*A: Triplet → B: TripletX + SupCon → C: Triplet + SupCon.*
The DataLoader is rebuilt at phase boundaries to enforce the P×K policy.

---

## 4.4 Threshold-Driven Multi-Loss Routing (Triplet / TripletX)

```bash
python run_phased_loss.py \
  --config configs/VeRi/deit_transreid_stride_ce_triplet_tripletx_supcon_prebn_brightness.yml \
  --seeds 0 --epochs 120 --save-every 3 --tripletx \
  --output-root ./logs
# Optional overrides:
#  --opts LOSS.BRIGHTNESS.THRESH 0.35 \
#        DATASETS.ROOT_DIR ./datasets
```

> A per-image **threshold** routes darker samples → **TripletX** (enhanced Triplet), normal samples → **Triplet**.
> SupCon can co-exist per YAML, or force it on with `--supcon`.

---

## 4.6 CLIP Fusion (OpenCLIP / TinyCLIP, optional; training-side with test-time control)

**OpenCLIP (minimal fusion; global concat + Linear):**

```bash
python run_phased_loss.py \
  --config configs/VeRi/deit_transreid_stride_ce_triplet_openclip_minimal.yml \
  --seeds 0 --epochs 120 --save-every 3 \
  --output-root ./logs
# Optional overrides:
#  --opts MODEL.CLIP_BACKBONE ViT-B-16 MODEL.CLIP_PRETRAIN laion2b_s34b_b88k \
#        DATASETS.ROOT_DIR ./datasets
```

**TinyCLIP (HF; quickwin fusion; optional AFEM / sem_refine):**

```bash
python run_phased_loss.py \
  --config configs/VeiRi/deit_transreid_stride_ce_triplet_320tinyclip0.2_noafem.yml \
  --seeds 0 --epochs 120 --save-every 3 \
  --output-root ./logs
# Optional overrides:
#  --opts MODEL.CLIP_HF_ID "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M" \
#        MODEL.CLIP_INPUT_SIZE '(224,224)' MODEL.CLIP_FINETUNE True \
#        MODEL.CLIP_USE_AFEM True MODEL.CLIP_USE_SEM_REFINE True \
#        DATASETS.ROOT_DIR ./datasets
```

> CLIP fusion is **optional** and leaves the backbone/heads intact.
> If you trained with CLIP but want to test **without** fusion, set `MODEL.USE_CLIP=False` in the test YAML.

---

## 5) Evaluation

```bash
python test.py \
  --config_file configs/VeRi/deit_transreid_stride_ce_tripletx_supcon.yml \
  TEST.WEIGHT ./logs/<your_run>/transformer_120.pth
# Optional overrides:
#  DATASETS.ROOT_DIR ./datasets OUTPUT_DIR ./logs/<your_run>_test
#  # If you trained with CLIP but want to test without fusion:
#  # set MODEL.USE_CLIP=False in the test YAML.
```

---

## 6) Results & Ablations (placeholder)

Mirror Report §7.2 in this order: **Baseline** → **+SupCon** → **+Conditional P×K** →
**+Phase** → **+Threshold-Driven Routing** → **+CLIP**.
Report **mAP / R-1 / R-5 / R-10** (and short- vs long-schedule if applicable).

---

## 7) Repo Layout (minimal)

```text
configs/VeRi/
├─ deit_transreid_stride_baseline.yml
├─ deit_transreid_stride_ce_triplet_supcon.yml
├─ deit_transreid_stride_ce_triplet_pk8.yml
├─ deit_transreid_stride_A_ce_triplet_B_ce_tripletx_supcon_C_ce_triplet_supcon.yml
├─ deit_transreid_stride_ce_triplet_tripletx.yml
├─ deit_transreid_stride_ce_triplet_openclip_minimal.yml
└─ deit_transreid_stride_ce_triplet_tinyclip_quickwin.yml
losses/*    processor/*    models/*   # training-side changes (losses / PK rebuild / phase hooks / CLIP)
```

---


## 8) License & Acknowledgements

* Base license: **MIT License** (TransReID).
* Addendum for training-side additions (put in `LICENSE-ADDENDUM` or file headers):

```text
Copyright (c) 2025
Hang Zhang <kongkongercat@gmail.com>
Additions and modifications to training-side losses, sampling, phase control, and CLIP fusion.
```

Thanks to **TransReID** (He et al., ICCV 2021), **reid-strong-baseline**, **timm**, **OpenCLIP**, and related open-source projects.

---

## 9) Citation

```bibtex
@InProceedings{He_2021_ICCV,
  author    = {He, Shuting and Luo, Hao and Wang, Pichao and Wang, Fan and Li, Hao and Jiang, Wei},
  title     = {TransReID: Transformer-Based Object Re-Identification},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2021},
  pages     = {15013-15022}
}
```
