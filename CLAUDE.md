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
- Prossimi step: Phase 5 — Subset Selection (1 categoria per cluster)

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

## Step Successivi

### Phase 5 — Subset Selection
**Cosa**: selezionare 3 categorie per cluster in maniera casuale(dal clustering con k = 13), di queste classi estrarre 750 immagini in maniera casuale 
**Input**: clustering_results.csv + embeddings CLIP
**Output**: subset su cui poi verrà addestrato lo sparse autoencoder

## Vocabolario semantico
SUN Attributes (Patterson & Hays, CVPR 2012)
- 102 attributi semantici per scene (materiali, illuminazione, layout)
- Usati per concept naming dell'MSAE
- File: labels_sunattribute.txt

## Stack tecnico
- CLIP: openai/clip-vit-large-patch14
- StreetCLIP (alternativa): geolocal/StreetCLIP
- Librerie analisi: pandas, numpy, matplotlib, scikit-learn, umap-learn

## Struttura cartelle
```
train/          # immagini training divise per lettera
val/val_256     # immagini validation
outputs/        # grafici e risultati analisi
scripts/        # script Python
  1-metadata_EDA.py
  2-image_quality.py
  3-clip_text_embeddings.py
  4-clustering.py
categories_places365.txt
IO_places365.txt
labels_sunattribute.txt
W_sceneattribute_wideresnet18.npy
```
