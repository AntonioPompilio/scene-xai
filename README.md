# Scene-XAI

**Interpretable urban scene classification via CLIP + Matryoshka Sparse Autoencoders + Fuzzy Inference**

Bachelor's thesis project

---

## Overview

End-to-end interpretable pipeline for scene image classification:

```
Places365 images → CLIP (ViT-L/14) → MSAE (Matryoshka Sparse Autoencoder) → FIS (Fuzzy Inference System)
```

CLIP encodes visual semantics; the MSAE decomposes them into sparse, human-interpretable concepts (SUN attributes + visual attributes); the FIS produces transparent classification rules. The goal is to explain *why* a model classifies a scene, not just *what* it predicts.

Based on: [Zaigrajew, Baniecki, Biecek — *Interpreting CLIP with Hierarchical Sparse Autoencoders* (arXiv:2502.20578, ICML 2025)](https://arxiv.org/abs/2502.20578)

---

## Dataset

**Places365-Standard** (Zhou et al., MIT CSAIL)
- 1.8M images · 365 scene categories · 256×256 px
- Split: `train/` (alphabetical subfolders) + `val/val_256/`
- Semantic attributes: 102 SUN attributes (Patterson & Hays, CVPR 2012)

Dataset not included in this repo — see [Places365](http://places2.csail.mit.edu/).

---

## Pipeline

| Phase | Script | Description |
|-------|--------|-------------|
| 1 | `scripts/1-metadata_EDA.py` | Metadata analysis (365 categories, indoor/outdoor split) |
| 2 | `scripts/2-image_quality.py` | Image quality metrics (brightness, contrast, saturation) |
| 3 | `scripts/3-clip_text_embeddings.py` | CLIP text encoding of category names (768D) |
| 4 | `scripts/4-clustering.py` | K-Means + UMAP on CLIP embeddings (k=13 optimal) |
| 5 | `scripts/5-subset_selection.py` | Balanced subset: 3 categories × 13 clusters × 750 images |
| 6 | `scripts/6-embedding_extraction.ipynb` | CLIP image embeddings for subset (29,250 imgs → memmap) |
| 7 | `scripts/7-sae_training.ipynb` | MSAE_RW training on Colab (expansion factor 8 or 16) |
| 8 | `scripts/8-vocab_embedding_extraction.ipynb` | CLIP text embeddings for combined vocabulary |
| 9 | `scripts/9-concept_naming.ipynb` | Concept naming: decoder features ↔ vocabulary similarity |
| 10 | `scripts/10-sae_activations_test.ipynb` | SAE activations on test set |
| 11 | `scripts/11_concept_analysis.ipynb` | Concept analysis: distributions, frequencies, visualizations |

Phases 1–5 run locally; phases 6–11 run on **Google Colab** with outputs stored on Google Drive.

---

## Repository Structure

```
scene-xai/
├── scripts/
│   ├── 1-metadata_EDA.py
│   ├── 2-image_quality.py
│   ├── 3-clip_text_embeddings.py
│   ├── 4-clustering.py
│   ├── 5-subset_selection.py
│   ├── 6-embedding_extraction.ipynb      # CLIP image embeddings (Colab)
│   ├── 7-sae_training.ipynb              # MSAE_RW training (Colab)
│   ├── 8-vocab_embedding_extraction.ipynb # vocabulary embeddings (Colab)
│   ├── 9-concept_naming.ipynb            # concept naming (Colab)
│   ├── 10-sae_activations_test.ipynb     # SAE inference on test set (Colab)
│   └── 11_concept_analysis.ipynb         # concept analysis (Colab)
├── MSAE/                                 # SAE module (patched from WolodjaZ/MSAE)
│   ├── sae.py                            # Autoencoder, MatryoshkaAutoencoder, SAE
│   ├── train.py                          # Training pipeline (early stopping, 3 splits)
│   ├── config.py                         # Model/training/loss configs
│   ├── loss.py                           # SAELoss (MSE/cosine + L1/L0)
│   ├── sae_naming.py                     # Concept naming / similarity matrix
│   ├── precompute_activations.py         # CLIP embedding extraction
│   ├── extract_sae_embeddings.py         # Representation extraction + metrics
│   ├── score_topk_sae_embeddings.py      # Progressive TopK scoring
│   ├── linear_eval.py                    # Linear probe evaluation
│   ├── metrics.py                        # FVU, CKNNA, MAE, L0
│   ├── utils.py                          # SAEDataset, EarlyStopping, schedulers
│   ├── aligment_metric.py               # Alignment metrics
│   ├── requirements.txt
│   └── vocab/
│       ├── clip_disect_20k.txt
│       ├── laion_400_unigram.txt
│       ├── laion_400_bigram.txt
│       └── mscoco_unigram.txt
├── outputs/                              # EDA outputs (CSVs + plots)
├── output_subset/                        # Balanced subset (cluster_<id>/<category>/*.jpg)
├── categories_places365.txt
├── IO_places365.txt
├── labels_sunattribute.txt
├── W_sceneattribute_wideresnet18.npy     # Pre-trained SUN attribute classifier
├── requirements.txt
├── README.md
└── CLAUDE.md                             # Project context & documentation
```

---

## Setup

```bash
# Clone
git clone https://github.com/<your-username>/scene-xai.git
cd scene-xai

# Install dependencies
pip install torch numpy pandas matplotlib scikit-learn umap-learn pillow tqdm

# CLIP backend (phases 1–5)
pip install git+https://github.com/openai/CLIP.git

# Open CLIP backend (phases 6–11, Colab)
pip install open_clip_torch
```

### Run EDA pipeline (phases 1–5, local)

```bash
python scripts/1-metadata_EDA.py
python scripts/2-image_quality.py
python scripts/3-clip_text_embeddings.py
python scripts/4-clustering.py
python scripts/5-subset_selection.py
```

### Run MSAE pipeline (phases 6–11, Google Colab)

Upload the subset to Google Drive, then run notebooks 6–11 in order on Colab.
Set `EXPANSION_FACTOR = 8` (latent 6144, as in paper) or `16` (latent 12288) at the top of each notebook.

---

## MSAE Architecture

The **Matryoshka Sparse Autoencoder** (MSAE_RW) uses:
- **Input**: CLIP ViT-L/14 embeddings (768D), L2-normalized
- **Activation**: TopKReLU (k=64 per nesting level)
- **K-list**: [64, 128, 256, ..., LATENT_DIM] — nested hierarchy of feature subsets
- **Weighting**: Reverse Weighting (RW) — higher weight for smaller k values
- **Loss**: MSE reconstruction + no explicit sparsity term (enforced by TopK)
- **Training**: AdamW, cosine LR decay, early stopping (patience=20)

### Concept Naming

Each decoder neuron is named by its maximum cosine similarity against a combined vocabulary:
- **SUN Attributes** (102 terms): scene materials, lighting, layout, activities
- **attribute_index.json** (~620 terms): colors, materials, shapes, textures, actions
- ~697 deduplicated terms total

Validation filters (Appendix A of paper): `top1_sim > 0.42` AND `top1_sim / top2_sim > 2.0`.

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Visual encoder (phases 1–5) | CLIP ViT-L/14 (`openai/clip`) |
| Visual encoder (phases 6–11) | CLIP ViT-L/14 CommonPool XL (`open_clip_torch`) |
| Sparse autoencoder | PyTorch (custom MSAE, patched from WolodjaZ/MSAE) |
| Clustering | scikit-learn (K-Means) |
| Visualization | UMAP + matplotlib |
| Training infrastructure | Google Colab (T4 GPU) + Google Drive |
| Data | pandas, numpy |

---

## References

- Zhou et al. — *Places: A 10 Million Image Database for Scene Recognition*, TPAMI 2018
- Patterson & Hays — *SUN Attribute Database*, CVPR 2012
- Zaigrajew, Baniecki, Biecek — *Interpreting CLIP with Hierarchical Sparse Autoencoders*, arXiv:2502.20578, ICML 2025
