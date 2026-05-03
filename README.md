# Scene-XAI

**Interpretable urban scene classification via CLIP + Matryoshka Sparse Autoencoders + Fuzzy Inference**

Bachelor's thesis project 

---

## Overview

End-to-end interpretable pipeline for scene image classification:

```
Places365 images → CLIP (ViT-L/14) → MSAE (Matryoshka Sparse Autoencoder) → FIS (Fuzzy Inference System)
```

CLIP encodes visual semantics; the MSAE decomposes them into sparse, human-interpretable concepts (SUN attributes); the FIS produces transparent classification rules. The goal is to explain *why* a model classifies a scene, not just *what* it predicts.

Based on: [Zaigrajew, Baniecki, Biecek — *Interpreting CLIP with Hierarchical Sparse Autoencoders* (arXiv:2502.20578, ICML 2025)](https://arxiv.org/abs/2502.20578)

---

## Dataset

**Places365-Standard** (Zhou et al., MIT CSAIL)
- 1.8M images · 365 scene categories · 256×256 px
- Split: `train/` (alphabetical subfolders) + `val/val_256/`
- Semantic attributes: 102 SUN attributes (Patterson & Hays, CVPR 2012)

Dataset not included in this repo — see [Places365](http://places2.csail.mit.edu/).

---


## Repository Structure

```
scene-xai/
├── scripts/
│   ├── 1-metadata_EDA.py          # Phase 1: metadata analysis
│   ├── 2-image_quality.py         # Phase 2: image quality metrics (cv2 + Lanczos)
│   ├── 3-clip_text_embeddings.py  # Phase 3: CLIP text encoding
│   ├── 4-clustering.py            # Phase 4: K-Means + UMAP
│   └── 5-subset_selection.py      # Phase 5: balanced subset for MSAE training
├── outputs/                       # Generated CSVs and plots
├── output_subset/                 # Balanced subset (cluster_<id>/<category>/*.jpg)
├── categories_places365.txt       # 365 category names
├── IO_places365.txt               # Indoor/outdoor labels
├── labels_sunattribute.txt        # 102 SUN semantic attributes
├── W_sceneattribute_wideresnet18.npy  # Pre-trained attribute classifier
├── requirements.txt               # Python dependencies
└── CLAUDE.md                      # Project context & documentation
```

---

## Setup

```bash
# Clone
git clone https://github.com/<your-username>/scene-xai.git
cd scene-xai

# Install dependencies
pip install torch numpy pandas matplotlib scikit-learn umap-learn pillow tqdm

# CLIP backend (choose one)
pip install git+https://github.com/openai/CLIP.git
# or: pip install transformers  (HuggingFace fallback)
```

### Run EDA pipeline

```bash
python scripts/1-metadata_EDA.py
python scripts/2-image_quality.py
python scripts/3-clip_text_embeddings.py
python scripts/4-clustering.py
```

Outputs are saved to `outputs/`.

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Visual encoder | CLIP ViT-L/14 (`openai/clip`) |
| Sparse autoencoder | PyTorch (custom MSAE) |
| Clustering | scikit-learn (K-Means) |
| Visualization | UMAP + matplotlib |
| Data | pandas, numpy |

---

## References

- Zhou et al. — *Places: A 10 Million Image Database for Scene Recognition*, TPAMI 2018
- Patterson & Hays — *SUN Attribute Database*, CVPR 2012
- Zaigrajew, Baniecki, Biecek — *Interpreting CLIP with Hierarchical Sparse Autoencoders*, arXiv:2502.20578
