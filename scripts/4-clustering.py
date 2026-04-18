"""
Phase 4 — Clustering & Visualization
======================================
K-Means sugli embeddings CLIP (365, 768) per k in [10, 15, 20, 25, 30],
con UMAP 2D per visualizzazione e analisi Elbow/Silhouette per il k ottimale.

OBIETTIVI:
  1. Clusterizzare le 365 categorie Places365 in gruppi semanticamente coerenti
  2. Identificare il k ottimale tramite Elbow curve e Silhouette score
  3. Visualizzare i cluster in 2D con UMAP
  4. Produrre materiale visivo e quantitativo per la scelta del subset 

PERCHÉ K-MEANS su embeddings CLIP:
  Gli embeddings L2-normalizzati vivono su un'ipersfera(cioè su una sfera di raggio 1, dato che sono L2-normalizzati). K-Means con
  inizializzazione k-means++(per scegliere i punti k in maniera "intelligente") e distanza euclidea converge bene in questo
  spazio, trovando cluster semanticamente coerenti. La normalizzazione
  garantisce che la distanza euclidea approssimi la dissimilarità coseno(avranno un fattore di proporzione rad2 , in parole povere
  saranno direttamente proporzionali e il loro comportamento è simile).

uso UMAP per visualizzare lo spazio degli embeddings in 2D, dato che di base sono in uno spazio n-dimensionale
PERCHÉ UMAP:
  UMAP preserva la struttura locale E globale dello spazio high-dimensional
  molto meglio di t-SNE a parità di costo computazionale. La visualizzazione
  2D mostra quanto i cluster siano separati e coerenti semanticamente.

Output generati in outputs/:
  clustering_results.csv              — assignment cluster per ogni k
  plot_elbow_silhouette.png           — curve Elbow e Silhouette
  plot_umap_by_cluster_k{k}.png       — UMAP colorato per cluster (per ogni k)
  plot_umap_by_io.png                 — UMAP colorato indoor/outdoor (unico)
  plot_cluster_sizes_k{k}.png         — distribuzione dimensioni cluster (per ogni k)
  plot_cluster_labels_k{k}.png        — categorie per cluster, top k ottimale

Esecuzione:
  python scripts/4-clustering.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

# ── Parametri ─────────────────────────────────────────────────────────────────
# k 4-35: curva elbow/silhouette leggibile su range ampio
K_VALUES_ANALYSIS = list(range(4, 36))          # per elbow/silhouette (range completo)
K_VALUES_PLOT     = [8, 13, 20, 25, 30]         # per UMAP e cluster sizes
K_LABELS          = 13                           # cluster labels (nomi categorie) solo per questo k
K_VALUES          = sorted(set(K_VALUES_ANALYSIS) | set(K_VALUES_PLOT))  # tutti i k da eseguire
SEED       = 42
N_INIT     = 20   # n. inizializzazioni K-Means (per stabilità), quante volte inizializzo k punti casuali che saranno i centroidi iniziali dei cluster
MAX_ITER   = 500  # massimo iterazioni K-Means, numero max di iterazioni in cui vengono spostati i centroidi verso il baricentro del cluster

# UMAP params: n_neighbors controlla struttura locale vs globale
# 15 è il default, buon compromesso per 365 punti
UMAP_N_NEIGHBORS  = 15 # quanti vicini considerare per ogni punto nello spazio n dimensionale(in questo caso 768) per costruire il grafo di vicinato che UMAP usa per la riduzione dimensionale. Valori più bassi preservano meglio la struttura locale, valori più alti catturano meglio la struttura globale.
UMAP_MIN_DIST     = 0.1 # distanza minima tra i punti nello spazio 2D, valori più bassi creano cluster più compatti, valori più alti distribuiscono i punti più uniformemente. 0.1 è un buon punto di partenza per visualizzazioni di cluster.
UMAP_METRIC       = "cosine"  # coerente con la similarity calcolata in Phase 3

plt.rcParams.update({
    "figure.dpi":       150,
    "figure.facecolor": "white",
    "axes.spines.top":  False,
    "axes.spines.right": False,
})


# ==============================================================================
# STEP 1 — Caricamento dati Phase 3
# ==============================================================================
# carico gli embeddings del CLIP dei prompt(controllo inoltre che siano normalizzati) e i metadati per ogni categoria
def load_phase3_outputs() -> tuple[np.ndarray, pd.DataFrame]:
    """
    Carica gli embeddings L2-normalizzati e i metadati dalla Phase 3.
    Ritorna (embeddings (365, 768), df con full_name e io).
    """
    print("\n[1/5] Caricamento output Phase 3 ...")

    emb_path  = OUTPUTS / "clip_text_embeddings.npy"
    meta_path = OUTPUTS / "clip_category_metadata.csv"

    if not emb_path.exists():
        raise FileNotFoundError(
            "clip_text_embeddings.npy non trovato. "
            "Esegui prima scripts/3-clip_text_embeddings.py"
        )

    embeddings = np.load(emb_path)               # (365, 768) float32
    df         = pd.read_csv(meta_path)           # idx, full_name, io, prompt

    print(f"  Embeddings shape : {embeddings.shape}")
    print(f"  Categorie        : {len(df)}")
    print(f"  Indoor           : {(df['io'] == 'indoor').sum()}")
    print(f"  Outdoor          : {(df['io'] == 'outdoor').sum()}")

    # Verifica normalizzazione
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  Norms — min: {norms.min():.4f}, max: {norms.max():.4f} (atteso ~1.0)") # controllo che gli embedding siano normalizzati 

    return embeddings, df


# ==============================================================================
# STEP 2 — K-Means per tutti i valori di k
# ==============================================================================
#fa il k-means per ogni k in K_VALUES, calcola :
# inertia(Somma delle distanze quadrate di ogni punto dal centroide del suo cluster, + è basso e più il cluster è compatto) 
# silhouette score(misura quanto bene ogni punto appartiene al suo cluster vs ai cluster vicini)
# ha range tra -1 e 1, più è alto e più i cluster sono ben separati e coerenti(se vale 1 è ben clusterizzato , -1 vuol dire che il punto è nel cluster sbagliato )
#lo score globale è la MEDIA dei silhouette di tutti i punti , più è alto e più i cluster sono ben separati e coerenti(>0.5 è considerato buono)
# salva i risultati in un dizionario

#ricorda! elbow method =  graficare inertia al variare di k e cercare un "gomito" (punto dove la diminuzione rallenta, li trovo il numero di k cluster ideale)
def run_kmeans_all_k(embeddings: np.ndarray) -> dict:
    """
    Esegue K-Means per ogni k in K_VALUES.
    Ritorna dict: k → {labels, inertia, silhouette, model}

    NOTA sulla metrica:
    K-Means usa distanza euclidea internamente. Su embeddings L2-normalizzati
    la distanza euclidea è monotona con la distanza coseno:
        ||a - b||² = 2 - 2·cos(a,b)   ( ||a||=||b||=1, essendo gli embeddings normalizzati)
    Quindi minimizzare inertia euclidea ≡ massimizzare cosine similarity(perchè facendo a-b , quando a e b sono simili il risultato è un vettore vicino a zero, quindi la distanza euclidea è bassa, e la similarità coseno è alta).
    """
    print(f"\n[2/5] K-Means per k in {K_VALUES} (totale: {len(K_VALUES)}) ...")
    results = {}

    for k in K_VALUES:
        print(f"  k={k:2d} ", end="", flush=True)
        km = KMeans(
            n_clusters=k,
            init="k-means++",  # inizializzazione intelligente per convergenza rapida
            n_init=N_INIT,     # 20 run con init diverse, prende la migliore
            max_iter=MAX_ITER,
            random_state=SEED,
        )
        labels    = km.fit_predict(embeddings)  # (365,) con valori in [0, k-1]
        inertia   = km.inertia_                 # somma distanze quadratiche dai centroidi
        sil_score = silhouette_score(embeddings, labels, metric="cosine")

        results[k] = {
            "labels":     labels,
            "inertia":    inertia,
            "silhouette": sil_score,
            "model":      km,
        }

        print(f"→ inertia={inertia:.2f}, silhouette={sil_score:.4f}")

    return results


# ==============================================================================
# STEP 3 — UMAP 2D
# ==============================================================================
#Funzione che schiaccia il cluster per visualizzarlo su un piano bidimensionale e visualizzare la distribuzione dei cluster nei grafici. 
def compute_umap(embeddings: np.ndarray) -> np.ndarray:
    """
    Riduce gli embeddings (365, 768) a 2D con UMAP.
    Usa metrica cosine 
    Ritorna array (365, 2).

    PERCHÉ un solo UMAP per tutti i k(per efficienza computazionale)
    La proiezione 2D è una proprietà degli embeddings, non dei cluster.
    I cluster vengono visualizzati come layer di colore sulla stessa mappa.
    """
    print("\n[3/5] Calcolo UMAP 2D ...")
    reducer = umap.UMAP( 
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=SEED,
        n_components=2,
    )
    embedding_2d = reducer.fit_transform(embeddings)  # (365, 2), questa funzione1   
    print(f"  UMAP shape: {embedding_2d.shape}")
    return embedding_2d


# ==============================================================================
# STEP 4 — Salvataggio risultati
# ==============================================================================
#Questa funzione salva i risultati della divisione in cluster e mostra i risultati di inertia e silhouette per ogni k
#restituisce un dataframe con le categorie e i cluster assegnati per ogni k, che sarà usato per le visualizzazioni successive
def save_clustering_results(df: pd.DataFrame, results: dict) -> pd.DataFrame:
    """
    Salva i cluster assignment per ogni k in un unico CSV.
    Ogni colonna 'cluster_k{k}' contiene l'ID del cluster per quella categoria.
    """
    print("\n[4/5] Salvataggio risultati ...")

    out_df = df[["idx", "full_name", "io"]].copy()
    for k, res in results.items():
        out_df[f"cluster_k{k}"] = res["labels"]

    out_path = OUTPUTS / "clustering_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  Salvato: {out_path.name}")

    # Stampa riassunto statistiche
    print("\n  k   | inertia    | silhouette")
    print("  " + "-" * 35)
    for k, res in results.items():
        print(f"  k={k:2d} | {res['inertia']:10.2f} | {res['silhouette']:.4f}")

    return out_df


# ==============================================================================
# STEP 5 — Visualizzazioni
# ==============================================================================
#stampa i grafici che confrontano l'andamento di inertia e silhouette al variare di k, per identificare il k ottimale.
#inoltre la funzione restituisce il k ottimale
def plot_elbow_silhouette(results: dict, out_dir: Path):
    """
    Due grafici affiancati:
    - Elbow curve: inertia vs k (cerca il "gomito" dove la curva si appiattisce)
    - Silhouette score vs k (massimo = k ottimale)

    PERCHÉ:
    - Elbow individua il k dove aggiungere un cluster non riduce più
      significativamente l'inertia (rendimento decrescente)
    - Silhouette misura quanto ogni punto è simile al suo cluster rispetto
      agli altri cluster (range [-1, 1], più alto = meglio separato)
    """
    # Usa solo il range completo (4-35) per leggibilità della curva
    ks         = [k for k in K_VALUES_ANALYSIS if k in results]
    inertias   = [results[k]["inertia"]    for k in ks]
    silhouettes= [results[k]["silhouette"] for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Tick sull'asse x ogni 5 valori (es. 5, 10, 15, ...) + il primo
    tick_positions = [k for k in ks if k % 5 == 0 or k == ks[0]]

    # ── Elbow ────────────────────────────────────────────────────────────────
    ax1.plot(ks, inertias, "o-", color="#4C72B0", linewidth=2, markersize=6)
    ax1.set_xlabel("k (numero di cluster)", fontsize=12)
    ax1.set_ylabel("Inertia (somma distanze²)", fontsize=12)
    ax1.set_title("Elbow Curve", fontsize=13)
    ax1.set_xticks(tick_positions)
    ax1.tick_params(axis="x", labelsize=10)
    # Annota ogni punto con il valore
    for k, v in zip(ks, inertias):
        ax1.annotate(f"{v:.1f}", (k, v), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=7)

    # ── Silhouette ───────────────────────────────────────────────────────────
    ax2.plot(ks, silhouettes, "o-", color="#DD8452", linewidth=2, markersize=6)
    ax2.set_xlabel("k (numero di cluster)", fontsize=12)
    ax2.set_ylabel("Silhouette score (cosine)", fontsize=12)
    ax2.set_title("Silhouette Score", fontsize=13)
    ax2.set_xticks(tick_positions)
    ax2.tick_params(axis="x", labelsize=10)
    # Annota ogni punto
    for k, v in zip(ks, silhouettes):
        ax2.annotate(f"{v:.3f}", (k, v), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=7)
    # Evidenzia il k con silhouette massima
    best_k = ks[int(np.argmax(silhouettes))]
    best_s = max(silhouettes)
    ax2.axvline(best_k, color="green", linestyle="--", alpha=0.6,
                label=f"k ottimale = {best_k}")
    ax2.legend(fontsize=10)

    fig.suptitle("CLIP Clustering — Scelta del k ottimale", fontsize=14)
    fig.tight_layout()
    out = out_dir / "plot_elbow_silhouette.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvato: {out.name}")
    print(f"  k ottimale (silhouette): {best_k} (score={best_s:.4f})")
    return best_k

#questa funzione serve per assegnare un  colore diverso a ogni cluster nei grafici UMAP
def _get_cluster_colors(k: int) -> list:
    """Palette discreta di k colori ben distinguibili."""
    cmap = plt.get_cmap("tab20" if k <= 20 else "hsv")
    return [cmap(i / k) for i in range(k)]

#questa funzione serve per visualizzare i cluster sui grafici UMAP, con marker diversi per indoor/outdoor e colori diversi per ogni cluster.
def plot_umap_by_cluster(embedding_2d: np.ndarray, df: pd.DataFrame,
                         results: dict, out_dir: Path):
    """
    Per ogni k: scatter plot UMAP colorato per cluster.
    Ogni punto = una categoria (365 totali).
    Forma del marker: ● indoor, ▲ outdoor (per distinguere le due categorie
    indipendentemente dal colore del cluster).

    PERCHÉ: mostra visivamente la coerenza semantica dei cluster.
    Cluster "belli" sono regioni compatte nello spazio UMAP.
    """
    markers = {"indoor": "o", "outdoor": "^"}

    for k, res in {k: results[k] for k in K_VALUES_PLOT if k in results}.items():
        labels = res["labels"]
        colors = _get_cluster_colors(k)

        fig, ax = plt.subplots(figsize=(12, 9))

        for io_type, marker in markers.items():
            mask = df["io"] == io_type
            for cluster_id in range(k):
                cmask = (labels == cluster_id) & mask.values
                if cmask.sum() == 0:
                    continue
                ax.scatter(
                    embedding_2d[cmask, 0],
                    embedding_2d[cmask, 1],
                    c=[colors[cluster_id]],
                    marker=marker,
                    s=70,
                    alpha=0.85,
                    edgecolors="white",
                    linewidths=0.5,
                    label=f"C{cluster_id}" if io_type == "indoor" else None,
                )

        # Etichette dei centroidi (numero cluster) sul plot
        for cluster_id in range(k):
            cmask = labels == cluster_id
            cx = embedding_2d[cmask, 0].mean()
            cy = embedding_2d[cmask, 1].mean()
            ax.text(cx, cy, str(cluster_id), fontsize=8, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, ec="none"))

        # Legenda: tipo marker per indoor/outdoor
        legend_io = [
            mpatches.Patch(color="gray", label="● indoor"),
            mpatches.Patch(color="gray", label="▲ outdoor"),
        ]
        ax.legend(handles=legend_io, fontsize=10, loc="lower right")

        ax.set_title(f"UMAP — Cluster CLIP (k={k})\n"
                     f"● = indoor, ▲ = outdoor", fontsize=12)
        ax.set_xlabel("UMAP dim 1", fontsize=10)
        ax.set_ylabel("UMAP dim 2", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()

        out = out_dir / f"plot_umap_by_cluster_k{k}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  Salvato: {out.name}")

#questa funzione serve per visualizzare la distribuzione di indoor e outdoor sui grafici UMAP, con colori diversi per indoor e outdoor, indipendentemente dai cluster.
def plot_umap_by_io(embedding_2d: np.ndarray, df: pd.DataFrame, out_dir: Path):
    """
    Scatter UMAP colorato per indoor/outdoor.
    Un solo plot (non dipende da k).
    Utile per verificare se CLIP separa naturalmente indoor da outdoor.

    PERCHÉ: se indoor e outdoor formano zone separate nel plot UMAP,
    significa che CLIP cattura questa distinzione fondamentale del dataset.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    colors_io = {"indoor": "#4C72B0", "outdoor": "#DD8452"}

    for io_type, color in colors_io.items():
        mask = df["io"] == io_type
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=color,
            label=io_type,
            s=70,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
        )

    # Annota le categorie più isolate (oast house era outlier!)
    for _, row in df.iterrows():
        if "oast_house" in row["full_name"]:
            idx = row.name
            ax.annotate(
                row["full_name"].replace("_", " "),
                (embedding_2d[idx, 0], embedding_2d[idx, 1]),
                fontsize=7, ha="center",
                xytext=(0, 12), textcoords="offset points",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
            )

    ax.legend(fontsize=12)
    ax.set_title("UMAP — Distribuzione Indoor / Outdoor (CLIP embeddings)", fontsize=13)
    ax.set_xlabel("UMAP dim 1", fontsize=10)
    ax.set_ylabel("UMAP dim 2", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    out = out_dir / "plot_umap_by_io.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvato: {out.name}")

#questa funzione stampa un istogramma per come varia la distribuzione delle categorie nei cluster coi vari k
def plot_cluster_sizes(df: pd.DataFrame, results: dict, out_dir: Path):
    """
    Per ogni k: istogramma delle dimensioni dei cluster (n. categorie per cluster).
    Colorato per composizione indoor/outdoor.

    PERCHÉ: cluster molto sbilanciati (1 categoria vs 50) indicano che k è
    troppo alto o troppo basso. La distribuzione ideale è relativamente uniforme.
    """
    for k, res in {k: results[k] for k in K_VALUES_PLOT if k in results}.items():
        labels = res["labels"]
        df_k   = df.copy()
        df_k["cluster"] = labels

        # Conta indoor/outdoor per cluster
        counts = df_k.groupby(["cluster", "io"]).size().unstack(fill_value=0)
        # Assicura entrambe le colonne esistano
        for col in ["indoor", "outdoor"]:
            if col not in counts.columns:
                counts[col] = 0
        counts = counts.sort_index()

        fig, ax = plt.subplots(figsize=(max(8, k * 0.6), 5))
        x = np.arange(k)
        w = 0.4
        ax.bar(x - w/2, counts.get("indoor",  0), width=w,
               color="#4C72B0", label="indoor",  edgecolor="white")
        ax.bar(x + w/2, counts.get("outdoor", 0), width=w,
               color="#DD8452", label="outdoor", edgecolor="white")

        # Etichetta totale sopra ogni coppia di barre
        totals = [counts.iloc[i].sum() for i in range(k)]
        max_total = max(totals)
        for i, total in enumerate(totals):
            ax.text(i, total + max_total * 0.01, str(total), ha="center", fontsize=8)

        # Margine verticale per non tagliare le etichette
        ax.set_ylim(0, max_total * 1.15)

        ax.set_xlabel("Cluster ID", fontsize=11)
        ax.set_ylabel("N. categorie", fontsize=11)
        ax.set_title(f"Dimensione cluster (k={k}) — indoor vs outdoor", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"C{i}" for i in range(k)], fontsize=8)
        ax.legend(fontsize=10)
        fig.tight_layout()

        out = out_dir / f"plot_cluster_sizes_k{k}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  Salvato: {out.name}")

#questa funzione serve per visualizzare le categorie assegnate a ogni cluster, in questo caso è stampata per il k ottimale
def plot_cluster_labels(df: pd.DataFrame, results: dict,
                        best_k: int, out_dir: Path):
    """
    Per il k ottimale: visualizza le categorie assegnate a ogni cluster.
    Un subplot per cluster con lista nomi.

    PERCHÉ: permette di dare un "nome semantico" a ogni cluster
    (es. cluster 3 = ambienti sportivi, cluster 7 = interni domestici).
    Fondamentale per la discussione con i relatori nella Phase 5.
    """
    k      = best_k
    labels = results[k]["labels"]
    df_k   = df.copy()
    df_k["cluster"] = labels

    # Calcola altezza in base al cluster più grande
    sizes      = [len(df_k[df_k["cluster"] == i]) for i in range(k)]
    max_items  = max(sizes)
    n_cols     = 3
    n_rows     = int(np.ceil(k / n_cols))
    row_height = max(5.0, max_items * 0.30)   # almeno 5 inch, poi scala con il cluster più grande
    fig_width  = n_cols * 7                   # 7 inch per colonna
    fig_height = n_rows * row_height + 1      # +1 per il titolo

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(fig_width, fig_height))
    axes = axes.flatten()

    for cluster_id in range(k):
        ax = axes[cluster_id]
        members = df_k[df_k["cluster"] == cluster_id].sort_values("io")

        text_lines = []
        for _, row in members.iterrows():
            icon = "●" if row["io"] == "indoor" else "▲"
            name = row["full_name"].replace("_", " ").replace("/", " /")
            text_lines.append(f"{icon} {name}")

        ax.axis("off")
        ax.text(0.03, 0.98, f"Cluster {cluster_id}  (n={len(members)})",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                va="top", ha="left")
        ax.text(0.03, 0.92, "\n".join(text_lines),
                transform=ax.transAxes, fontsize=8.5, va="top", ha="left",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="#f8f8f8", ec="#cccccc"))

    # Nascondi assi extra
    for i in range(k, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"CLIP Clustering k={k} — Categorie per cluster\n"
                 f"● indoor  ▲ outdoor", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = out_dir / f"plot_cluster_labels_k{k}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Salvato: {out.name}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("Phase 4 — Clustering & Visualization")
    print("=" * 60)

    # Step 1: carica dati Phase 3
    embeddings, df = load_phase3_outputs()

    # Step 2: K-Means per tutti i k(faccio la divisione in cluster per ogni k, salvo i risultati in un dizionario)
    results = run_kmeans_all_k(embeddings)

    # Step 3: UMAP
    embedding_2d = compute_umap(embeddings)

    # Step 4: salva CSV clustering
    out_df = save_clustering_results(df, results)

    # Step 5: visualizzazioni
    print("\n[5/5] Generazione grafici ...")
    best_k = plot_elbow_silhouette(results, OUTPUTS)
    plot_umap_by_io(embedding_2d, df, OUTPUTS)
    plot_umap_by_cluster(embedding_2d, df, results, OUTPUTS)
    plot_cluster_sizes(out_df, results, OUTPUTS)
    plot_cluster_labels(out_df, results, K_LABELS, OUTPUTS)

    # Stampa analisi semantica per k=K_LABELS
    print(f"\n{'─'*60}")
    print(f"ANALISI CLUSTER (k={K_LABELS})")
    print(f"{'─'*60}")
    labels = results[K_LABELS]["labels"]
    df_k   = df.copy()
    df_k["cluster"] = labels

    for cluster_id in range(K_LABELS):
        members = df_k[df_k["cluster"] == cluster_id]
        n_in  = (members["io"] == "indoor").sum()
        n_out = (members["io"] == "outdoor").sum()
        names = [n.replace("_", " ") for n in members["full_name"].tolist()]
        print(f"\n  Cluster {cluster_id:2d} (n={len(members):2d}, "
              f"in={n_in}, out={n_out}):")
        for name in names:
            print(f"    - {name}")

    print(f"\nPhase 4 completata.")
    print(f"Output in: {OUTPUTS}")
    print("\nFile generati:")
    for f in sorted(OUTPUTS.glob("*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
