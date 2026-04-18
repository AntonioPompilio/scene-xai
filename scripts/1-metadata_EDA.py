"""
Fase 1 — Metadata EDA
======================
Analisi esplorativa del dataset Places365 basata esclusivamente sui file
di metadati (.txt) e sulla struttura delle cartelle.

NON carica immagini

Output generati in outputs/:
    metadata.csv                   — DataFrame master con tutte le categorie
    plot_io_distribution.png       — torta indoor vs outdoor
    plot_letter_distribution.png   — n. categorie per lettera iniziale
    plot_image_counts.png          — conferma bilanciamento del dataset

Esecuzione:
    python scripts/phase1_metadata.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # salva i plot su file senza aprire finestre GUI

# ── Path setup ────────────────────────────────────────────────────────────────
# Path(__file__) è il path di questo script.
# .parent.parent risale di due livelli → root del progetto.
ROOT    = Path(__file__).parent.parent # prende la root del progetto
TRAIN   = ROOT / "train"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)# Crea la cartella outputs/ se non esiste già

# ── Stile grafici ─────────────────────────────────────────────────────────────
plt.rcParams.update({# stile base per i grafici: font più grande, colori più vivaci, sfondo bianco
    "figure.dpi":      150,# risoluzione più alta per immagini più nitide
    "figure.facecolor": "white",
    "axes.spines.top":  False,
    "axes.spines.right": False,
})


# ==============================================================================
# STEP 1 — Parsing di categories_places365.txt
# ==============================================================================
# Il file ha 365 righe, una per categoria.
# Formato: /lettera/nome_categoria indice
# Alcune categorie hanno path annidati: /a/apartment_building/outdoor
#
# PERCHÉ lo facciamo: costruiamo il DataFrame "master" che useremo in
# tutte le fasi successive. Ogni riga = una categoria con il suo indice
# numerico, il nome leggibile, la lettera iniziale e il path completo.

def parse_categories(filepath: Path) -> pd.DataFrame:
    records = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()#strip rimuove spazi bianchi all'inizio e alla fine della riga
            if not line:#salta righe vuote
                continue
            parts = line.split()#splitta le singole righe in due parti: il path e l'indice numerico(perchè divise da uno spazio)
            scene_path = parts[0]          # es. /a/apartment_building/outdoor, prendo la prima parte
            idx        = int(parts[1])     # indice numerico (0–364), 2nda parte

            path_parts  = scene_path.strip("/").split("/")#strip("/") rimuove eventuali slash iniziali/finali
            letter      = path_parts[0]    # prima lettera dell'alfabeto
            # Il nome visualizzabile è l'ultimo componente del path,
            # con underscore sostituiti da spazi
            display     = path_parts[-1].replace("_", " ")#path_parts[-1] prende l'ultimo componente del path, replace sostituisce gli underscore con spazi
            
            full_name   = "/".join(path_parts[1:])#full name senza la lettera iniziale

            records.append({
                "idx":         idx,
                "scene_path":  scene_path,
                "full_name":   full_name,
                "display":     display,
                "letter":      letter,
            })

    df = pd.DataFrame(records).sort_values("idx").reset_index(drop=True)
    return df #l'output sarà un DataFrame con le colonne: idx, scene_path, full_name, display, letter


# ==============================================================================
# STEP 2 — Parsing di IO_places365.txt (indoor / outdoor)
# ==============================================================================
# Formato: /lettera/nome_categoria 1_o_2
# Convenzione Places365: 1 = indoor, 2 = outdoor
#
# PERCHÉ ci interessa: indoor/outdoor è la principale suddivisione semantica
# del dataset.

def parse_io(filepath: Path) -> dict:
    io_map = {}#dizionario che mappa scene_path a "indoor" o "outdoor"
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()#splitto le 2 righe della stringa con lo spazio
            scene_path = parts[0] 
            io_map[scene_path] = "indoor" if int(parts[1]) == 1 else "outdoor"#assegno indoor se 1 , outdoor se 2
    return io_map


# ==============================================================================
# STEP 3 — Conteggio immagini per categoria (scan del filesystem)
# ==============================================================================
# Percorriamo la cartella train/ e contiamo i .jpg per ogni categoria.
# Non carichiamo le immagini — usiamo solo os.listdir() via Path.iterdir().
#
# ATTENZIONE ai path annidati: la categoria /a/apartment_building/outdoor
# si trova in train/a/apartment_building/outdoor/
#
# PERCHÉ lo facciamo:  verificare che il dataset sia bilanciato
# (stesso numero di immagini per ogni categoria). 

def count_images(train_root: Path, scene_path: str) -> int:
    """Conta i file .jpg nella cartella corrispondente a scene_path."""
    parts  = scene_path.strip("/").split("/")   # ['a', 'airfield']
    folder = train_root.joinpath(*parts)        # train/a/airfield/, ricostruisce il path completo
    if not folder.is_dir(): #se la cartella non esiste, (quindi se non è true folder.is_dir())
        return 0
    return sum(1 for f in folder.iterdir() if f.suffix.lower() == ".jpg")#iterdir() restituisce un iteratore sui file nella cartella, conto quanti hanno estensione .jpg (case-insensitive)


# ==============================================================================
# STEP 4 — Visualizzazioni
# ==============================================================================

def plot_io_distribution(df: pd.DataFrame, out_dir: Path):
    """
    Grafico a torta: quante categorie sono indoor vs outdoor.
    PERCHÉ: dà un'idea immediata dello sbilanciamento semantico del dataset.
    """
    counts = df["io"].value_counts()
    colors = ["#4C72B0", "#DD8452"]   # blu=indoor, arancio=outdoor

    fig, ax = plt.subplots(figsize=(6, 6)) #crea una figura di 6x6 pollici
    wedges, texts, autotexts = ax.pie(# crea un grafico a torta con i valori di counts, le etichette corrispondenti, percentuali, colori e stile
        counts, # I dati da visualizzare (es. conteggi per categoria)
        labels=counts.index,     # Le etichette (nomi delle categorie)
        autopct="%1.1f%%",   # Formato per le percentuali (es. "25.0%")
        colors=colors,  # Colori per le fette (blu e arancio)
        startangle=90,  # Ruota il grafico in modo che la prima fetta parta dalle 12(in alto)
        wedgeprops={"edgecolor": "white", "linewidth": 2},  # Stile dei bordi delle fette (bianco, spessore 2)
    )
    for t in autotexts:#stile per le etichette delle percentuali: font più grande e grassetto
        t.set_fontsize(13)
        t.set_fontweight("bold")

    ax.set_title("Places365 — Distribuzione Indoor / Outdoor\n(365 categorie)", fontsize=14)
    fig.tight_layout()
    out = out_dir / "plot_io_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvato: {out.name}")


def plot_letter_distribution(df: pd.DataFrame, out_dir: Path):
    """
    Bar chart: quante categorie per ogni lettera iniziale.
    PERCHÉ: il dataset è organizzato alfabeticamente su disco; questo
    grafico mostra la distribuzione e aiuta a capire la struttura dei path.
    (questo grafico potrebbe essere potenzialmente inutile)
    """
    counts = df["letter"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(14, 4))
    bars = ax.bar(counts.index, counts.values, color="#4C72B0", edgecolor="white")
    ax.bar_label(bars, fontsize=8, padding=2)
    ax.set_xlabel("Lettera iniziale della categoria", fontsize=12)
    ax.set_ylabel("N. categorie", fontsize=12)
    ax.set_title("Places365 — Categorie per lettera iniziale", fontsize=14)
    fig.tight_layout()
    out = out_dir / "plot_letter_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvato: {out.name}")


def plot_image_counts(df: pd.DataFrame, out_dir: Path):
    """
    Istogramma del numero di immagini per categoria.
    PERCHÉ: verifica il bilanciamento. Se tutte le barre sono a 5000,
    il dataset è perfettamente bilanciato → nessun over/under-sampling
    necessario nel subset.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["train_count"], bins=30, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Immagini per categoria (train)", fontsize=12)
    ax.set_ylabel("N. categorie", fontsize=12)
    ax.set_title("Places365 — Distribuzione immagini per categoria", fontsize=14)
    ax.axvline(df["train_count"].mean(), color="#DD8452", linestyle="--",
               label=f"Media: {df['train_count'].mean():.0f}")
    ax.legend()
    fig.tight_layout()
    out = out_dir / "plot_image_counts.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvato: {out.name}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("Fase 1 — Metadata EDA")
    print("=" * 60)

    # -- Step 1: categorie --
    print("\n[1/4] Parsing categories_places365.txt ...")
    df = parse_categories(ROOT / "categories_places365.txt")
    print(f"  Categorie trovate: {len(df)}")

    # -- Step 2: indoor/outdoor --
    print("\n[2/4] Parsing IO_places365.txt ...")
    io_map = parse_io(ROOT / "IO_places365.txt")
    df["io"] = df["scene_path"].map(io_map)# aggiungo la categoria indoor/outdoor al DataFrame, mappando scene_path con io_map
    n_missing = df["io"].isna().sum()#conto se ci sono celle vuote(che non hanno indoor/outdoor) nella colonna io
    if n_missing > 0:
        print(f"  ATTENZIONE: {n_missing} categorie senza label IO")
    else:
        print("  Tutte le categorie hanno il label IO.")

    # -- Step 3: conteggio immagini --
    print("\n[3/4] Conteggio immagini nel train/ ...")
    df["train_count"] = df["scene_path"].apply(
        lambda p: count_images(TRAIN, p) #su ogni riga di scene_path applico la funzione count_images
    )
    missing_dirs = (df["train_count"] == 0).sum()
    if missing_dirs > 0:
        print(f"  ATTENZIONE: {missing_dirs} categorie con 0 immagini trovate.")

    # -- Statistiche riassuntive --
    print("\n" + "─" * 60)
    print("STATISTICHE DATASET")
    print("─" * 60)
    print(f"  Totale categorie     : {len(df)}")
    io_counts = df["io"].value_counts()
    for label, n in io_counts.items():
        pct = n / len(df) * 100
        print(f"  {label:10s}         : {n} ({pct:.1f}%)")
    print(f"  Immagini train totali: {df['train_count'].sum():,}")
    print(f"  Min img per categoria: {df['train_count'].min():,}")
    print(f"  Max img per categoria: {df['train_count'].max():,}")
    print(f"  Media img/categoria  : {df['train_count'].mean():.0f}")
    print("─" * 60)

    # -- Salva metadata.csv --
    out_csv = OUTPUTS / "metadata.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  DataFrame salvato: {out_csv.name}")
    print(f"  Colonne: {list(df.columns)}")

    # -- Step 4: visualizzazioni --
    print("\n[4/4] Generazione grafici ...")
    plot_io_distribution(df, OUTPUTS)
    plot_letter_distribution(df, OUTPUTS)
    plot_image_counts(df, OUTPUTS)

    print(f"Output in: {OUTPUTS}")


if __name__ == "__main__":
    main()
