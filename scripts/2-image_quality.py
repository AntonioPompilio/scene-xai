"""
Fase 2 — Image Quality & Visual Exploration
==============================================
Analisi visiva e qualitativa su un campione di immagini del dataset.

OBIETTIVI:
  1. Verificare la coerenza visiva delle categorie (montage)
  2. Rilevare categorie con alta varianza visiva (potenzialmente ambigue)
  3. Identificare outlier di luminosità/contrasto per categoria
  4. Produrre materiale visivo per la tesi

NON carica l'intero dataset: usa campioni ridotti per categoria.

Output generati in outputs/:
  plot_overview_montage.png       — 1 immagine per categoria (griglia 365)
  plot_brightness_per_category.png — luminosità media e varianza per categoria
  plot_saturation_per_category.png — saturazione media per categoria
  plot_quality_scatter.png         — scatter luminosità vs varianza intra-cat.
  image_quality_stats.csv          — statistiche qualitative per categoria

Esecuzione:
  python scripts/phase2b_image_quality.py
"""

from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")#usa backend "Agg" per generare immagini senza display (headless)
import matplotlib.pyplot as plt #importa pyplot per la creazione di grafici
import matplotlib.gridspec as gridspec #gridspec permette di creare layout di subplot più complessi e personalizzati
from PIL import Image #PIL (Pillow) per la manipolazione delle immagini (caricamento, ridimensionamento, conversione)
from tqdm import tqdm #tqdm fornisce barre di progresso per iterazioni lunghe (es. loop su categorie o immagini)

# ── Riproducibilità ───────────────────────────────────────────────────────────
SEED = 42 # fisso il seed a 42 (perchè è la risposta a tutto) per riproducibilità dei campionamenti casuali
random.seed(SEED)
np.random.seed(SEED)

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
TRAIN   = ROOT / "train"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

# ── Parametri ─────────────────────────────────────────────────────────────────
N_QUALITY_SAMPLES = 200  # immagini per categoria per le statistiche qualitative
MONTAGE_SIZE      = 64   # pixel per lato di ogni thumbnail nel montage overview
THUMB_QUALITY     = 20   

plt.rcParams.update({
    "figure.dpi": 150,#dpi più alto per immagini più nitide
    "figure.facecolor": "white",#sfondo bianco per tutte le figure
})


# ==============================================================================
# UTILITÀ
# ==============================================================================
# restituisce i path delle immagini .jpg per categoria specificata in scene_path 
def get_image_paths(train_root: Path, scene_path: str) -> list:
    parts  = scene_path.strip("/").split("/")
    folder = train_root.joinpath(*parts)
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.jpg"))#restituisce una lista ordinata di tutti i file con estensione .jpg nella cartella specificata (e nelle sue sottocartelle, se presenti)

#prende n immagini casuali dalla categoria in modo riproducibile (usando un seed fisso)
def sample_paths(paths: list, n: int, seed: int = SEED) -> list:
    rng = random.Random(seed)
    return rng.sample(paths, min(n, len(paths)))

#carica un'immagine, la ridimensiona a 128 x 128 (non 256 x 256 per velocità e per mantenere abbastanza dettaglio per le statistiche qualitative)
#restituisce un array numpy con i valori RGB (float32 per i calcoli, ma in scala 0-255, che poi verrà normalizzato in [0.0, 1.0] dividendo per 255.0)
def load_image_rgb(path: Path, size: int ) -> np.ndarray:

    with Image.open(path) as img: #apre l'immagine usando PIL, with garantisce che il file venga chiuso dopo l'uso
        img = img.convert("RGB").resize((size, size), Image.BILINEAR) # scompone la matrice originale dell'immagine in 3 matrici(R, G , B) e le combina in un'unica matrice tridimensionale, poi l'immagine viene ridimensionata a 128 x 128 usando l'interpolazione bilineare (Image.BILINEAR), una "compressione intelligente" dell'immagine che è un buon compromesso tra qualità e velocità per questo scopo
        return np.array(img, dtype=np.float32) / 255.0  # normalizzato in [0.0, 1.0]

#prende un'immagine, calcola la luminosità percettiva, il contrasto (deviazione standard della luminosità) e la saturazione media (calcolata con la formula HSV) e restituisce un dizionario con queste statistiche. Tutti i valori sono normalizzati in scala [0.0, 1.0].
def compute_image_stats(img_array: np.ndarray) -> dict:
    """
    Calcola statistiche visive da un array RGB (H, W, 3) float32 in [0.0, 1.0].
    Tutti i valori restituiti sono in scala [0.0, 1.0].

    LUMINOSITÀ: percezione umana della "chiarezza". Formula standard
    ponderata per i canali RGB (il verde pesa di più perché l'occhio
    umano è più sensibile ad esso).

    CONTRASTO: deviazione standard del canale di luminosità. Alta std
    = immagine con zone molto chiare e molto scure (es. scena con sole
    e ombre). Bassa std = immagine piatta (es. cielo uniforme).

    SATURAZIONE: quanto i colori sono "vividi". Calcolata direttamente
    con la formula matematica dello spazio HSV tramite NumPy.
    Formula: S = (max(R,G,B) - min(R,G,B)) / max(R,G,B)
    I pixel completamente neri (max=0) hanno saturazione 0 per definizione.(la divisione per 0 creerebbe problemi)
    """
    # Vista lungo l'asse dei canali: shape (H, W) ciascuna
    # np.max/min con keepdims=False evita copie inutili dell'array
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

    # Luminosità percettiva ITU-R BT.709, scala [0.0, 1.0]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

    # Contrasto = deviazione standard della luminosità
    contrast = float(luminance.std())

    # Saturazione HSV 
    #per ogni singolo pixel ,prendo il valore massimo e minimo tra i 3 canali RGB, calcolo la differenza (delta) e poi divido per il massimo per ottenere la saturazione. Se il massimo è 0 (pixel nero), la saturazione è definita come 0.
    c_max = np.max(img_array, axis=2)   # per ogni pixel dei 3 canali, prendo il valore massimo, shape (H, W) 
    c_min = np.min(img_array, axis=2)   # per-pixel min dei 3 canali,prendo il minimo, shape (H, W)
    delta = c_max - c_min               # "chroma": ampiezza dell'intervallo cromatico

    # Sostituiamo i denominatori zero con 1.0 PRIMA della divisione:
    # np.where valuta entrambi i rami, quindi delta/c_max verrebbe calcolato
    # anche dove c_max==0, generando RuntimeWarning. safe_denom evita il problema.
    safe_denom = np.where(c_max > 0.0, c_max, 1.0)#sostituisce i valori di c_max con 1 dove c_max è 0, altrimenti lascia i valori di c_max invariati. 
    saturation_map = np.where(c_max > 0.0, delta / safe_denom, 0.0)#calcola la saturazione pixel-wise, evitando divisioni per zero. 
    saturation = float(saturation_map.mean())#calcolo la saturazione media su tutta l'immagine

    return {
        "mean_brightness": float(luminance.mean()),
        "std_brightness":  contrast,
        "mean_saturation": saturation,
    }


# ==============================================================================
# STEP 1 — Statistiche qualitative per categoria
# ==============================================================================
# Per ogni categoria campioniamo N_QUALITY_SAMPLES immagini e calcoliamo
# le statistiche aggregate (media e std di luminosità, contrasto, saturazione).
#
# PERCHÉ: categorie con alta std_brightness (alta varianza intra-categoria)
# sono visivamente eterogenee → più difficili da classificare e interpretare.
# Saperlo prima di scegliere il subset è utile.

def compute_quality_stats(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[1/3] Calcolo statistiche qualitative per categoria ...")
    print(f"  ({N_QUALITY_SAMPLES} immagini per categoria, ~{len(df) * N_QUALITY_SAMPLES} immagini totali)")

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), ascii=True, desc="  categorie"):#itero sulle righe del DataFrame con una barra di progresso, total indica il numero totale di iterazioni per la barra, ascii=True usa caratteri ASCII per la barra (compatibile con tutti i terminali), desc è una descrizione da mostrare accanto alla barra
        paths = get_image_paths(TRAIN, row["scene_path"]) #ottengo i path delle immagini per la categoria specificata in scene_path 
        if not paths:#se non ci sono immagini per questa categoria, aggiunge nan alle statistiche e va avanti alla prossima categoria
            records.append({
                "idx": row["idx"], "full_name": row["full_name"],
                "io": row["io"],
                "mean_brightness": np.nan, "std_brightness": np.nan,
                "mean_saturation": np.nan, "n_images": 0,
            })
            continue

        sampled = sample_paths(paths, N_QUALITY_SAMPLES)
        stats_list = []
        for p in sampled:
            try:
                arr = load_image_rgb(p, size=64)   # 64px: veloce(ancora più veloce di 128px) ma abbastanza dettagliato per le statistiche qualitative
                stats_list.append(compute_image_stats(arr))
            except Exception:
                continue

        if not stats_list:
            continue

        mean_bright = np.mean([s["mean_brightness"] for s in stats_list])
        std_bright  = np.mean([s["std_brightness"]  for s in stats_list])
        mean_sat    = np.mean([s["mean_saturation"]  for s in stats_list])

        records.append({
            "idx":             row["idx"],
            "full_name":       row["full_name"],
            "io":              row["io"],
            "mean_brightness": mean_bright,
            "std_brightness":  std_bright,
            "mean_saturation": mean_sat,
            "n_images":        len(paths),
        })

    return pd.DataFrame(records)


# ==============================================================================
# STEP 2 — Overview montage (1 immagine per categoria)
# ==============================================================================
# Una griglia con una thumbnail per ciascuna delle 365 categorie.
# Ordinate per indice → corrispondono all'ordine in metadata.csv.
#
# PERCHÉ: mostra in un colpo d'occhio la varietà del dataset. Utile nella
# presentazione della tesi per giustificare la scelta del dataset.

def build_overview_montage(df: pd.DataFrame, out_dir: Path):
    print("\n[2/3] Costruzione overview montage (1 img/categoria) ...")

    n_cols = 25
    n_rows = int(np.ceil(len(df) / n_cols))   # 365 / 25 = 15 righe
    thumb  = MONTAGE_SIZE

    canvas = np.ones((n_rows * thumb, n_cols * thumb, 3), dtype=np.uint8) * 200

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), ascii=True, desc="  thumbnails")):
        paths = get_image_paths(TRAIN, row["scene_path"])
        if not paths:
            continue
        path = random.choice(paths)
        try:
            with Image.open(path) as img:
                img = img.convert("RGB").resize((thumb, thumb), Image.BILINEAR)
                arr = np.array(img)
        except Exception:
            continue

        r = i // n_cols
        c = i  % n_cols
        canvas[r*thumb:(r+1)*thumb, c*thumb:(c+1)*thumb] = arr

    # Aggiungi una griglia leggera per separare visivamente le thumbnail
    fig, ax = plt.subplots(figsize=(n_cols * thumb / 100, n_rows * thumb / 100))
    ax.imshow(canvas)
    ax.axis("off")
    ax.set_title(f"Places365 — Overview (365 categorie, 1 immagine per categoria)", fontsize=8)
    fig.tight_layout(pad=0.1)
    out = out_dir / "plot_overview_montage.png"
    fig.savefig(out, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"  Salvato: {out.name}")


# ==============================================================================
# STEP 3 — Plot statistiche qualitative
# ==============================================================================

def plot_quality_stats(stats_df: pd.DataFrame, out_dir: Path):
    print("\n[3/3] Generazione grafici statistiche qualitative ...")

    stats_df = stats_df.sort_values("mean_brightness")

    # ── Luminosità per categoria ──────────────────────────────────────────────
    # Ogni barra = una categoria. Colore = indoor (blu) / outdoor (arancio).
    # Le categorie a sinistra sono le più scure (es. cave, nightclub),
    # quelle a destra le più chiare (es. snowfield, beach).
    fig, ax = plt.subplots(figsize=(20, 5))
    colors = ["#4C72B0" if io == "indoor" else "#DD8452"
              for io in stats_df["io"]]
    ax.bar(range(len(stats_df)), stats_df["mean_brightness"], color=colors,
           width=1.0, edgecolor="none")
    ax.set_xlabel("Categoria (ordinata per luminosità crescente)", fontsize=11)
    ax.set_ylabel("Luminosità media (0–1)", fontsize=11)
    ax.set_title("Places365 — Luminosità media per categoria", fontsize=13)
    ax.set_xticks([])
    # Legenda manuale
    from matplotlib.patches import Patch
    legend = [Patch(color="#4C72B0", label="indoor"),
              Patch(color="#DD8452", label="outdoor")]
    ax.legend(handles=legend, fontsize=10)
    fig.tight_layout()
    out = out_dir / "plot_brightness_per_category.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvato: {out.name}")

    # ── Top/Bottom categorie per varianza visiva ──────────────────────────────
    # Mostra le 15 categorie con MAGGIORE e MINORE varianza intra-categoria.
    # Alta varianza → categoria visivamente eterogenea (attenzione nella
    # selezione del subset).
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    top15 = stats_df.nlargest(15, "std_brightness")
    bot15 = stats_df.nsmallest(15, "std_brightness")

    for ax, sub_df, title in [
        (axes[0], top15, "Top 15 — Maggiore varianza visiva (più ambigue)"),
        (axes[1], bot15, "Bottom 15 — Minore varianza visiva (più omogenee)"),
    ]:
        # Per categorie annidate (es. "apartment_building/outdoor") mostriamo
        # il path completo con "/" come separatore, non solo l'ultimo componente.
        labels = [n.replace("_", " ").replace("/", " / ")[:35] for n in sub_df["full_name"]]
        colors_bar = ["#4C72B0" if io == "indoor" else "#DD8452"
                      for io in sub_df["io"]]
        ax.barh(range(len(sub_df)), sub_df["std_brightness"],
                color=colors_bar, edgecolor="white")
        ax.set_yticks(range(len(sub_df)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Contrasto medio (std luminosità)", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(handles=legend, fontsize=9)

    fig.suptitle("Places365 — Categorie per varianza visiva intra-categoria",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    out = out_dir / "plot_variance_ranking.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvato: {out.name}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("Phase 2b — Image Quality & Visual Exploration")
    print("=" * 60)

    # Carica il DataFrame prodotto dalla Phase 1
    metadata_path = OUTPUTS / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            "outputs/metadata.csv non trovato. "
            "Esegui prima scripts/phase1_metadata.py"
        )
    df = pd.read_csv(metadata_path)
    print(f"\nCaricate {len(df)} categorie da metadata.csv")

    # Step 1: statistiche qualitative
    stats_df = compute_quality_stats(df)
    out_csv = OUTPUTS / "image_quality_stats.csv"
    stats_df.to_csv(out_csv, index=False)
    print(f"\n  Statistiche salvate: {out_csv.name}")

    # Stampa le categorie più "problematiche"
    print("\n─── Top 10 categorie con maggiore varianza visiva ───")
    top10 = stats_df.nlargest(10, "std_brightness")[["full_name", "io", "std_brightness", "mean_brightness"]]
    print(top10.to_string(index=False))

    print("\n─── Top 10 categorie più scure (bassa luminosità) ───")
    dark10 = stats_df.nsmallest(10, "mean_brightness")[["full_name", "io", "mean_brightness", "std_brightness"]]
    print(dark10.to_string(index=False))

    # Step 2: overview montage
    build_overview_montage(df, OUTPUTS)

    # Step 3: plot statistiche
    plot_quality_stats(stats_df, OUTPUTS)

    print("\nPhase 2b completata.")
    print(f"Output in: {OUTPUTS}")
    print("\nFile generati:")
    for f in sorted(OUTPUTS.glob("*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
