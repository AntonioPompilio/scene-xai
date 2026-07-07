# Tesi — Classificazione interpretabile di scene urbane

## Obiettivo
Pipeline interpretabile per classificazione di immagini:
CLIP → MSAE (Matryoshka Sparse Autoencoder) → FIS (Fuzzy Inference System)

Paper di riferimento: Zaigrajew, Baniecki, Biecek — "Interpreting CLIP with
Hierarchical Sparse Autoencoders" (arXiv 2502.20578, ICML 2025)

## Dataset
- Places365-Standard (Zhou et al., MIT CSAIL)
- 1.8M immagini, 365 categorie di scene, 256×256 px
- Split: train/ (diviso per lettera alfabetica) + val/val_256
- File label: categories_places365.txt, IO_places365.txt
- File attributi: labels_sunattribute.txt (102 attributi SUN)
- Classificatore attributi pre-addestrato: W_sceneattribute_wideresnet18.npy

## Stato attuale
- Dataset scaricato (versione small 256×256, train + val)
- **EDA Phase 1 ✓**: metadata analysis completata
- **EDA Phase 2b ✓**: image quality analysis completata
- **EDA Phase 3 ✓**: CLIP text embeddings completata
- **EDA Phase 4 ✓**: clustering & visualization completata — k ottimale = 13
- **Phase 5 ✓**: subset selection completata (3 categorie × 13 cluster, 750 img/cat)
- **Phase 6 ✓**: estrazione embedding CLIP immagini su subset (29.250 img → memmap)
- **Phase 7 ✓**: training MSAE_RW su Google Colab (expansion 8 o 16)
- **Phase 8 ✓**: estrazione embedding testuali vocabolario combinato (SUN + attribute_index)
- **Phase 9 ✓**: concept naming (similarity matrix + concept map paper/adattiva)
- **Phase 10 ✓**: calcolo attivazioni SAE sul test set
- **Phase 11 ✓**: concept analysis (distribuzione concetti, frequenze, visualizzazioni)
- Prossimi step: FIS (Fuzzy Inference System)

## EDA Completata

### Phase 1 — Metadata Analysis
**Cosa**: parsing e statistiche da file di metadati senza caricare immagini.
**Output** (`outputs/metadata.csv`):
- 365 categorie con indice, path completo, label indoor/outdoor, conteggio immagini
- Indoor: 161 (44.1%), Outdoor: 204 (55.9%) — leggero bias verso outdoor
- Immagini train: 1,803,460 totali (min=3,068, max=5,000, media=4,941)
- Dataset quasi perfettamente bilanciato (la quasi totalità ha 5.000 immagini)

**Grafici generati**:
- `plot_io_distribution.png` — torta indoor/outdoor
- `plot_image_counts.png` — istogramma bilanciamento

### Phase 2b — Image Quality Analysis
**Cosa**: analisi visiva su campione rappresentativo per identificare categorie ambigue e outlier.
**Campionamento**: 200 immagini per categoria (4.4% del dataset, ~73.000 immagini totali), campionate casualmente con seed=42 per riproducibilità.
**Preprocessing**: Load → RGB conversion → resize 64×64 → normalizzazione [0.0, 1.0].

**Metriche estratte per immagine** (aggregate per categoria con media):
1. **Luminanza** (ITU-R BT.709): L = 0.2126R + 0.7152G + 0.0722B
2. **Contrasto**: σ(L) = std della luminanza
3. **Saturazione**: S = (max(R,G,B) − min(R,G,B)) / max(R,G,B)

**Output** (`outputs/image_quality_stats.csv`):
- Per ogni categoria: mean_brightness, std_brightness, mean_saturation

**Grafici generati**:
- `plot_overview_montage.png` — griglia 25×15 con 1 thumbnail per categoria (365 immagini)
- `plot_brightness_per_category.png` — bar chart luminosità media per categoria, colorata indoor/outdoor
- `plot_variance_ranking.png` — top/bottom 15 categorie per varianza visiva intra-categoria

**Insights principali**:
- **Top varianza visiva**: pagoda (0.260), arena/hockey (0.255), ice_skating_rink/indoor (0.250)
- **Più scure**: orchestra_pit (0.198), stage/indoor (0.205), movie_theater/indoor (0.225)
- **Assenza correlazione luminosità ↔ indoor/outdoor**: il fattore dominante è il tipo di scena

### Phase 3 — CLIP Text Embeddings
**Cosa**: encoding dei 365 nomi di categorie con CLIP (ViT-L/14, 768D) usando prompt `"a photo of {category}"`.
**Modello**: openai/clip-vit-large-patch14 (ViT-L/14), backend openai/CLIP con fallback HuggingFace transformers.
**Processing**: encoding in batch da 64 → L2-normalizzazione → matrice cosine similarity (365×365).

**Output**:
- `clip_text_embeddings.npy` — matrice (365, 768) float32, L2-normalizzata
- `clip_cosine_similarity.npy` — matrice (365, 365) float32
- `clip_category_metadata.csv` — idx, full_name, io, prompt per ogni categoria

**Grafici generati**:
- `plot_similarity_heatmap.png` — heatmap cosine similarity 365×365
- `plot_top_similar_pairs.png` — 20 coppie di categorie più simili semanticamente
- `plot_top_dissimilar_pairs.png` — 20 coppie meno simili (outlier: oast_house appare in molte coppie dissimili)

**Insights principali**:
- oast_house è l'outlier più marcato: appare sistematicamente nelle coppie a bassa similarità
- CLIP cattura similarità visiva, non solo linguistica (es. canyon e butte vicini, forest e forest path distanti)

### Phase 4 — Clustering & Visualization
**Cosa**: K-Means sugli embeddings CLIP (365, 768) per k ∈ [4, 35], UMAP 2D per visualizzazione.
**Parametri**: init=k-means++, n_init=20, max_iter=500, seed=42. UMAP: n_neighbors=15, min_dist=0.1, metric=cosine.

**Risultati Elbow/Silhouette** (range completo k=4-35):
- Elbow: nessun gomito visibile — curva decresce uniformemente da 111.8 a 86.3
- Silhouette: massimo a **k=13 (score=0.077)** → k ottimale
- Tutti i silhouette < 0.1: spazio CLIP continuo, cluster sfumati (atteso per scene urbane)

**Output**:
- `clustering_results.csv` — assignment cluster per ogni k (colonne cluster_k4 ... cluster_k35)
- `plot_elbow_silhouette.png` — curve Elbow e Silhouette (range 4-35)
- `plot_umap_by_io.png` — UMAP colorato indoor/outdoor
- `plot_umap_by_cluster_k{k}.png` — UMAP colorato per cluster (k ∈ [8, 13, 20, 25, 30])
- `plot_cluster_sizes_k{k}.png` — distribuzione dimensioni cluster (k ∈ [8, 13, 20, 25, 30])
- `plot_cluster_labels_k13.png` — categorie per cluster per k=13

**Cluster k=13** (dettaglio):
| Cluster | n | Indoor/Outdoor | Tema semantico |
|---------|---|----------------|----------------|
| C0 | 23 | 100% indoor | Spazi pubblici coperti (cinema, museo, piscina, chiesa) |
| C1 | 33 | 97% indoor | Negozi e servizi commerciali (ristorante, bakery, pharmacy) |
| C2 | 14 | 100% outdoor | Giardini e ambienti naturali curati |
| C3 | 31 | 97% outdoor | Paesaggi naturali estremi (glacier, volcano, ocean, mountain) |
| C4 | 18 | 94% indoor | Spazi funzionali interni (bathroom, corridor, locker room) |
| C5 | 7 | 100% outdoor | Strutture verticali (lighthouse, tower, skyscraper) |
| C6 | 30 | 73% outdoor | Edifici e architetture (castle, house, palace, temple) |
| C7 | 42 | 95% indoor | Spazi abitativi e lavorativi (bedroom, kitchen, office) |
| C8 | 59 | 63% outdoor | Ambienti urbani e infrastrutture (airport, campus, street) |
| C9 | 31 | 100% outdoor | Esterni di edifici (courtyard, patio, porch) |
| C10 | 50 | 92% outdoor | Paesaggi aperti e rurali (park, river, harbor, canyon) |
| C11 | 12 | 75% outdoor | Impianti sportivi (stadium, baseball field, soccer field) |
| C12 | 15 | 73% indoor | Spazi transizionali (lobby, staircase, car interior) |

**Insights principali**:
- CLIP separa indoor/outdoor in modo naturale: la maggioranza dei cluster è quasi pura
- Cluster misti (C6, C8, C12) riflettono ambiguità semantica reale delle scene
- C8 è il più grande (n=59): zona "urbana generica" dello spazio CLIP poco differenziata
- C5 è il più piccolo (n=7): strutture verticali semanticamente molto coese

### Phase 5 — Subset Selection
**Cosa**: selezione casuale di 3 categorie per cluster (k=13), campionamento di 750 immagini per categoria, copia fisica in cartella di output strutturata.
**Campionamento**: per ogni cluster, `random.sample(3)` sulle categorie; per ogni categoria, `random.sample(750)` sulle immagini disponibili. Seed=42.
**Output** (`outputs/subset_manifest.csv`):
- 13 cluster × 3 categorie × 750 immagini = ~29.250 file
- Colonne: cluster_id, category, image_path_source, image_path_dest
**Struttura output** (`output_subset/`):
- `cluster_<cluster_id>/<category_name>/<image_file>.jpg`
- Preserva categorie nidificate (es. `cluster_3/field/cultivated/img.jpg`)
**Script**: `scripts/5-subset_selection.py` — argomenti `--base-dir`, `--output-dir`, `--clustering-csv` con sensible defaults.

## Pipeline MSAE

### Phase 6 — Estrazione Embedding CLIP Immagini
**Script**: `scripts/6-embedding_extraction.ipynb` (Google Colab)
**Cosa**: forward pass CLIP su tutte le 29.250 immagini del subset → embedding 768D L2-normalizzati.
**Modello**: `laion/CLIP-ViT-L-14-CommonPool.XL.clip-s13B-b90K` via `open_clip` (non openai/CLIP; stessa architettura ViT-L/14, pre-addestrato su CommonPool XL 13B).
**Split**: stratificato per cluster, 70/20/10 → train/val/test.
**Output** (raw memmap float32, salvati su Google Drive `MyDrive/tesi/`):
- `places365_ViT-L~14_train_image_20475_768.npy` (~20.475 immagini)
- `places365_ViT-L~14_val_image_5850_768.npy` (~5.850 immagini)
- `places365_ViT-L~14_test_image_2925_768.npy` (~2.938 immagini, N_TEST=2938)
- `clip_image_embedding_manifest_{train,val,test}.csv` — mapping embedding_idx → cluster/categoria/path

### Phase 7 — Training MSAE
**Script**: `scripts/7-sae_training.ipynb` (Google Colab)
**Cosa**: addestramento MSAE_RW (Matryoshka SAE con reverse weighting) su embedding CLIP del subset.
**Configurazione**:
- `EXPANSION_FACTOR`: 8 → latente 6144 (come paper); 16 → latente 12288 (capacità raddoppiata)
- K-list Matryoshka (generata automaticamente da nesting_list=64): [64, 128, 256, 512, 1024, 2048, 4096, 6144] (exp 8) o [64,...,12288] (exp 16)
- Attivazione: TopKReLU; max_nesting patchato in config.py = LATENT_DIM; batch_size patchato = 256
- Early stopping: patience=20 epoche senza miglioramento su val_loss
- Epoche max: 200
**Nota implementativa**: `train.py` e `utils.py` sono versioni patchate rispetto alla repo ufficiale WolodjaZ/MSAE (aggiunta early stopping + argomento `-dv/--dataset_val`). I file patch si trovano in `MSAE/`.
**Output** (su Drive `MyDrive/tesi/models/`):
- `{LATENT_DIM}_768_TopKReLU_64_RW_False_False_0.0_places365_ViT-L~14_train_image_20475_768.pth`
- Contiene: `model` state_dict, `mean_center`, `scaling_factor`, `target_norm`

### Phase 8 — Estrazione Embedding Vocabolario
**Script**: `scripts/8-vocab_embedding_extraction.ipynb` (Google Colab)
**Cosa**: embedding CLIP testuali per il vocabolario combinato di attributi visivi.
**Vocabolario**: SUN Attributes (102 termini) + `attribute_index.json` (620 attributi visivi: colori, materiali, forme, texture, azioni) → ~697 termini dopo deduplicazione.
**Nota**: indipendente dal modello MSAE (dipende solo da CLIP); il file prodotto è condiviso da entrambi i modelli (expansion 8 e 16).
**Output** (su Drive `MyDrive/tesi/`):
- `sun_attrindex_ViT-L~14_text_{N}_768.npy` — raw memmap embedding (L2-normalizzati)
- `sun_attrindex_ViT-L~14_text_{N}.txt` — termini bare in ordine allineato agli embedding

### Phase 9 — Concept Naming
**Script**: `scripts/9-concept_naming.ipynb` (Google Colab)
**Cosa**: assegna a ogni neurone del decoder MSAE il termine del vocabolario con massima cosine similarity. Applica i filtri di validazione dell'Appendice A del paper.
**Metodo** (da `MSAE/sae_naming.py`):
- Le colonne del decoder sono riportate nello spazio CLIP con `postprocess()` (`decoder + pre_bias` → `/scaling_factor + mean`)
- `patch_diff=True`: sottrae la ricostruzione del vettore latente nullo (rimuove componente di background comune)
- Il vocabolario è caricato grezzo (non preprocessato), coerente con lo spazio CLIP originale
**Filtri**:
- Paper (Appendice A): top1_sim > 0.42 AND ratio > 2.0
- Adattivi (fallback): top1_sim > mean + 3σ AND ratio > 1.10
**Output** (su Drive `MyDrive/tesi/`):
- `similarity_matrix_{LATENT_DIM}.npy` — matrice float32 [n_vocab, LATENT_DIM]
- `concept_map_paper_{LATENT_DIM}.csv` — neuron_id, concept, top1_sim, top2_sim, ratio, valid
- `concept_map_adaptive_{LATENT_DIM}.csv` — stessa struttura con filtri adattivi
- Charts in `MyDrive/tesi/charts/`: `plot_top1_sim_all.png`, `plot_top1_sim_valid.png`, `plot_top20_concepts.png`, `plot_threshold_sensitivity.png`
**Nota sul checkpoint corrente**: con expansion=16, 26 neuroni su 12288 passano i filtri adattivi. I concetti nominati includono termini non puramente visivi (termini geografici da attribute_index.json), segnale che il modello è sotto-addestrato su un subset di 20k immagini.

### Phase 10 — Attivazioni SAE sul Test Set
**Script**: `scripts/10-sae_activations_test.ipynb` (Google Colab)
**Cosa**: inferenza MSAE sul test set, salvataggio delle attivazioni latenti full (ReLU senza TopK constraint) per analisi downstream.
**Preprocessing**: embeddings CLIP caricati grezzi; `SAE.encode()` applica internamente `(x - mean) * scaling_factor`.
**Output**:
- `sae_activations_test_{N_TEST}_{LATENT_DIM}.npy` — raw memmap float32 [N_TEST, LATENT_DIM]
- Sparsity attesa: ~0.80-0.85 per MSAE_RW

### Phase 11 — Concept Analysis
**Script**: `scripts/11_concept_analysis.ipynb` (Google Colab)
**Cosa**: analisi delle attivazioni MSAE sui neuroni "validi" (concept map adattiva).
**Soglia magnitudo**: 0.5 (>0 include floor di ricostruzione non semantico; soglie 0.5–1.0 filtrano la coda semantica).
**Analisi**:
- Distribuzione concetti attivi per immagine (istogramma skewed, ~2-3 concetti/img a soglia 0.5)
- Frequenza dei concetti nel test set → `concept_stats_{LATENT_DIM}.csv`
- Visualizzazione 8 immagini campione con top-5 concetti attivi per magnitudine
**Output** (su Drive `MyDrive/tesi/`):
- `concept_stats_{LATENT_DIM}.csv` — concept, neuron_id, n_images_active, pct_images_active, mean_activation_when_active
- Charts: `plot_concepts_per_image.png`, `plot_top20_concept_frequency.png`, `plot_image_concepts_grid.png`

## Vocabolario semantico
Vocabolario combinato usato per concept naming:
- **SUN Attributes** (Patterson & Hays, CVPR 2012): 102 attributi di scena (materiali, illuminazione, layout, attività)
- **attribute_index.json**: 620 attributi visivi (colori, materiali, forme, texture, azioni)
- Deduplicazione: ~697 termini totali (termini esatti in comune contati una sola volta)
- File SUN: `labels_sunattribute.txt`
- File JSON: `attribute_index.json` (su Google Drive `MyDrive/tesi/`)

## Modulo MSAE (`MSAE/`)
Versione patchata della repo WolodjaZ/MSAE (arXiv:2502.20578).

| File | Descrizione |
|------|-------------|
| `sae.py` | `Autoencoder`, `MatryoshkaAutoencoder`, `SAE` (wrapper con preprocessing), `load_model` |
| `train.py` | Pipeline di training completa con early stopping, 3 split (train/val/test) |
| `config.py` | `TrainConfig`, `LossConfig`, `ModelConfig`; preset per ReLUSAE, TopKSAE, BatchTopKSAE, MSAE_UW, MSAE_RW |
| `loss.py` | `SAELoss`: MSE/NMSE/cosine + L1/L0 sparsity |
| `sae_naming.py` | Concept naming: similarity matrix decoder↔vocab, `compute_similarities`, `patch_diff` |
| `precompute_activations.py` | Estrazione embedding CLIP (ImageNet, CC3M, CelebA, vocabolari) |
| `extract_sae_embeddings.py` | Estrazione rappresentazioni SAE + metriche (FVU, MAE, CKNNA, L0) |
| `score_topk_sae_embeddings.py` | Scoring progressivo a diversi livelli TopK |
| `linear_eval.py` | Linear probe per valutare la preservazione semantica |
| `metrics.py` | FVU, CKNNA, MAE, L0, orthogonal_decoder |
| `utils.py` | `SAEDataset`, `EarlyStopping`, `set_seed`, `get_device`, schedulers |
| `aligment_metric.py` | Metriche di allineamento tra spazi |
| `vocab/` | Vocabolari: `clip_disect_20k.txt`, `laion_400_{unigram,bigram}.txt`, `mscoco_unigram.txt` |

## Stack tecnico
- CLIP immagini: `laion/CLIP-ViT-L-14-CommonPool.XL.clip-s13B-b90K` via `open_clip` (fase 6+)
- CLIP testo (EDA): openai/clip-vit-large-patch14 via openai/CLIP
- SAE: PyTorch — custom MSAE (patchato da WolodjaZ/MSAE)
- Training: Google Colab (GPU T4), Google Drive per storage
- Librerie analisi: pandas, numpy, matplotlib, scikit-learn, umap-learn
- Embedding extraction: open_clip_torch

## Struttura cartelle
```
train/                  # immagini training divise per lettera
val/val_256             # immagini validation
outputs/                # grafici e risultati analisi EDA + clustering
  metadata.csv
  image_quality_stats.csv
  clustering_results.csv
  subset_manifest.csv
output_subset/          # subset finale per training MSAE (Phase 5)
  cluster_0/ ... cluster_12/
scripts/                # script Python e notebook Colab
  1-metadata_EDA.py
  2-image_quality.py
  3-clip_text_embeddings.py
  4-clustering.py
  5-subset_selection.py
  6-embedding_extraction.ipynb      # estrazione embedding CLIP immagini
  7-sae_training.ipynb              # training MSAE_RW su Colab
  8-vocab_embedding_extraction.ipynb # embedding vocabolario (SUN + attr_index)
  9-concept_naming.ipynb            # concept naming + matrice similarità
  10-sae_activations_test.ipynb     # attivazioni SAE sul test set
  11_concept_analysis.ipynb         # analisi concetti attivi + visualizzazioni
MSAE/                   # modulo SAE (patchato da WolodjaZ/MSAE)
  sae.py
  train.py
  config.py
  loss.py
  sae_naming.py
  precompute_activations.py
  extract_sae_embeddings.py
  score_topk_sae_embeddings.py
  linear_eval.py
  metrics.py
  utils.py
  aligment_metric.py
  requirements.txt
  vocab/
    clip_disect_20k.txt
    laion_400_unigram.txt
    laion_400_bigram.txt
    mscoco_unigram.txt
categories_places365.txt
IO_places365.txt
labels_sunattribute.txt
W_sceneattribute_wideresnet18.npy
requirements.txt
```

## Note su Google Drive
I file di grandi dimensioni (embedding .npy, checkpoint .pth) sono su Google Drive `MyDrive/tesi/`:
- `places365_ViT-L~14_{train,val,test}_image_*_768.npy` — embedding immagini (memmap)
- `models/{LATENT_DIM}_768_TopKReLU_64_RW_*.pth` — checkpoint MSAE
- `sun_attrindex_ViT-L~14_text_*_768.npy` — embedding vocabolario
- `sun_attrindex_ViT-L~14_text_*.txt` — termini vocabolario
- `similarity_matrix_{LATENT_DIM}.npy` — matrice similarità decoder↔vocab
- `concept_map_{paper,adaptive}_{LATENT_DIM}.csv` — concept map
- `sae_activations_test_*_{LATENT_DIM}.npy` — attivazioni test set
- `concept_stats_{LATENT_DIM}.csv` — statistiche concetti
- `clip_image_embedding_manifest_{train,val,test}.csv` — manifest split
- `output_subset.zip` + `subset_manifest.zip` — subset immagini compressi
