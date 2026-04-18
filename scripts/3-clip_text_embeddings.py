"""
Fase 3 — CLIP Text Embeddings
================================
Codifica i 365 nomi di categorie Places365 con CLIP (ViT-L/14, 768D)
usando il prompt "a photo of {category}".

OBIETTIVI:
  1. Estrarre embeddings semantici per ciascuna categoria
  2. L2-normalizzare(faccio la norma euclidea[che mi da la lunghezza] e lo normalizzo) gli embeddings (per cosine similarity via dot product)
  3. Calcolare la matrice di cosine similarity (365×365), una sorta di matrice "di vicinanza" semantica tra le categorie
  4. Visualizzare le categorie più/meno simili tra loro
  5. Preparare l'input per il clustering 

PERCHÉ CLIP text embeddings:
  CLIP è addestrato su coppie immagine-testo: il suo encoder testuale
  produce embeddings che catturano la semantica visiva delle scene, non
  solo la semantica linguistica. Due categorie con nomi simili ma aspetto
  diverso (es. "forest" vs "forest path") saranno distanti; categorie
  visivamente simili ma con nomi diversi (es. "canyon" vs "butte") saranno
  vicine. Questo rende gli embeddings CLIP ideali come base per il clustering
  semanticamente significativo del dataset Places365.

Output generati in outputs/:
  clip_text_embeddings.npy        — matrice (365, 768) float32, L2-normalizzata
  clip_cosine_similarity.npy      — matrice (365, 365) float32
  clip_category_metadata.csv      — DataFrame con idx, full_name, io + colonne embedding
  plot_similarity_heatmap.png     — heatmap cosine similarity (365×365)
  plot_top_similar_pairs.png      — bar chart delle 20 coppie più simili
  plot_top_dissimilar_pairs.png   — bar chart delle 20 coppie meno simili

Esecuzione:
  python scripts/3-clip_text_embeddings.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")#è un backend di matplotlib che genera i grafici senza mostrare il popup
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch

# ── CLIP import ───────────────────────────────────────────────────────────────
# Usiamo il pacchetto ufficiale OpenAI CLIP via pip:
#   pip install git+https://github.com/openai/CLIP.git
# oppure transformers di HuggingFace come fallback.
try:
    import clip  # openai/CLIP
    CLIP_BACKEND = "openai"
except ImportError:
    try:
        from transformers import CLIPTokenizer, CLIPTextModel
        CLIP_BACKEND = "transformers"
    except ImportError:
        raise ImportError(
            "Nessun backend CLIP trovato.\n"
            "Installa con:\n"
            "  pip install git+https://github.com/openai/CLIP.git\n"
            "oppure:\n"
            "  pip install transformers"
        )

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

# ── Parametri ─────────────────────────────────────────────────────────────────
MODEL_NAME   = "ViT-L/14"        # CLIP model (768D text embeddings)
PROMPT_TMPL  = "a photo of {}"   # prompt template
BATCH_SIZE   = 64                # token batch per encoding(CPU friendly, dato che lo faccio girare in locale), gli embedding ottenuti dal prompt messo su clip per ogni categoria verranno processati 64 alla volta

plt.rcParams.update({
    "figure.dpi":       150,
    "figure.facecolor": "white",
    "axes.spines.top":  False,
    "axes.spines.right": False,
})


# ==============================================================================
# STEP 1 — Caricamento modello CLIP
# ==============================================================================

def load_clip_model(device: str):
    """
    Carica il modello CLIP e il tokenizer.
    Ritorna (model, tokenize_fn, embed_dim).
    """
    print(f"\n[1/5] Caricamento modello CLIP ({MODEL_NAME}) su {device} ...")
    print(f"  Backend: {CLIP_BACKEND}")

    if CLIP_BACKEND == "openai":
        model, preprocess = clip.load(MODEL_NAME, device=device)#scarica il modello CLIP e  lo mette in model, preprocess è una funzione di preprocessing per le immagini, ma noi ci interessano solo i text embeddings, quindi non lo useremo
        model.eval()#mette il modello in modalità di valutazione(cioè il modello che mi serve per fare inferenza ), non addestramento
        embed_dim = model.token_embedding.embedding_dim  # 768 per ViT-L/14
        # Wrap tokenizer per interfaccia uniforme
        tokenize_fn = clip.tokenize # faccio aliasing della funzione di tokenizzazione del CLIP
        print(f"  Embedding dim: {embed_dim}")
        return model, tokenize_fn, embed_dim, preprocess

    else:  # transformers, questa branch non dovrebbe essere usata ma fa da fallback, che però riporta allo stesso modello di base (ViT-L/14) quindi non cambia nulla a livello di embedding
        hf_name = "openai/clip-vit-large-patch14"
        tokenizer = CLIPTokenizer.from_pretrained(hf_name)
        text_model = CLIPTextModel.from_pretrained(hf_name).to(device)
        text_model.eval()
        embed_dim = text_model.config.hidden_size  # 768
        print(f"  Embedding dim: {embed_dim}")
        return text_model, tokenizer, embed_dim, None


# ==============================================================================
# STEP 2 — Costruzione prompt e encoding
# ==============================================================================

def build_prompts(df: pd.DataFrame) -> list[str]:
    """
    Costruisce il prompt per ogni categoria.
    Usa full_name (es. "apartment_building/outdoor") ma riformattato:
      - underscore → spazi
      - "/" → " " (rimuoviamo la gerarchia, manteniamo la leggibilità)
    Esempio: "apartment_building/outdoor" → "a photo of apartment building outdoor"
    """
    prompts = []
    for full_name in df["full_name"]:
        readable = full_name.replace("_", " ").replace("/", " ")
        prompt = PROMPT_TMPL.format(readable)#formatto i prompt cosi da non avere / o _ che erano presenti nei titoli e nelle categorie
        prompts.append(prompt)
    return prompts

#funzione che prende i prompt e sputa fuori gli embedding del clip
@torch.no_grad()# disabilita il calcolo dei gradienti, dato che sto facendo inferenza e non training
def encode_texts_openai(model, tokenize_fn, prompts: list[str],
                        device: str, batch_size: int) -> np.ndarray:
    """
    Encoding con backend openai/CLIP.
    Ritorna array (N, 768) float32 NON normalizzato.
    """
    all_feats = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start: start + batch_size]
        tokens = tokenize_fn(batch, truncate=True).to(device)  # (B, 77), dato il prompt, questa funzione sputa fuori per ognuno 77 token, truncate= True taglia i prompt + lunghi di 77 token, se il prompt è più piccolo , viene paddato
        feats  = model.encode_text(tokens)                      # (B, 768), qua dai token sputo fuori gli embeddings dei {batch_size} prompt
        all_feats.append(feats.cpu().float().numpy()) #sposto su CPU e converto il tensore PyTorch in un array NumPy float32, e lo aggiungo alla lista di tutti gli embedding
    return np.concatenate(all_feats, axis=0)  # (N, 768),aggiungo tutti gli embedding ottenuti in un unico array NumPy di forma (N, 768)

#stessa funzione di prima che prende i prompt e sputa gli embedding del clip
@torch.no_grad()
def encode_texts_transformers(text_model, tokenizer, prompts: list[str],
                               device: str, batch_size: int) -> np.ndarray:
    """
    Encoding con backend transformers.
    Usa il pooled output (EOS token) del CLIP text encoder.
    Ritorna array (N, 768) float32 NON normalizzato.
    """
    all_feats = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start: start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=77).to(device)
        outputs = text_model(**inputs)
        # pooler_output = rappresentazione aggregata dell'intera sequenza
        feats = outputs.pooler_output  # (B, 768)
        all_feats.append(feats.cpu().float().numpy())
    return np.concatenate(all_feats, axis=0)

#in base al modello che sto usando, chiama la funzione corrispondente corretta tra le due prima
def encode_prompts(model, tokenize_fn, prompts: list[str],
                   device: str) -> np.ndarray:
    """
    Dispatch al backend corretto. Ritorna (N, 768) float32, NON normalizzato.
    """
    print(f"\n[2/5] Encoding {len(prompts)} prompt con CLIP ...")
    if CLIP_BACKEND == "openai":
        feats = encode_texts_openai(model, tokenize_fn, prompts, device, BATCH_SIZE)
    else:
        feats = encode_texts_transformers(model, tokenize_fn, prompts, device, BATCH_SIZE)
    print(f"  Shape raw embeddings: {feats.shape}")
    return feats


# ==============================================================================
# STEP 3 — L2-normalizzazione e cosine similarity
# ==============================================================================
#prende gli embedding e ne fa la normalizzazione, cosicchè ogni embedding abbia norma 1 , in questo modo la cosine similarity tra due embedding è semplicemente il loro prodotto scalare, dato che la norma è 1
#in questo modo con la cosine similiarity tra due embedding considera la vicinanza tra essi solo in base alla direzione, non alla lunghezza, e questo è importante per gli embedding di CLIP che possono avere magnitudini diverse ma essere comunque semanticamente vicini
def l2_normalize(x: np.ndarray) -> np.ndarray:
    """
    L2-normalizza ogni riga di x: ||x_i|| = 1.
    Dopo la normalizzazione: cosine_sim(i, j) = dot(x_i, x_j)
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True)#calcolo la norma L2 degli embedding per ogni riga, in quanto nella matriciona degli embedding ogni riga è un prompt
    # Evita divisione per zero (embedding nullo, non dovrebbe accadere con CLIP)
    norms = np.where(norms > 0, norms, 1.0)#se la norma è maggiore di 0, mantieni la norma, altrimenti sostituisci con 1.0 per evitare divisione per zero
    return x / norms#applico la formula della normalizzazione L2, divido ogni riga per la sua norma, in questo modo ottengo un embedding normalizzato con norma 1

#calcola la matrice di similiarità tra gli embedding, capisco cosi quelli che sono vicini tra loro 
def compute_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """
    Matrice (N, N) di cosine similarity.
    Assumendo embeddings già L2-normalizzati: sim = E @ E.T
    """
    return embeddings @ embeddings.T  # (365, 365), valori in [-1, 1]


def process_embeddings(raw_feats: np.ndarray):
    """
    Normalizza e calcola la matrice di similarity.
    Ritorna (embeddings_norm, sim_matrix).
    """
    print("\n[3/5] Normalizzazione L2 e calcolo cosine similarity ...")
    emb_norm = l2_normalize(raw_feats)#normalizzo gli embedding
    sim_mat  = compute_cosine_similarity(emb_norm)#calcolo la matrice di cosine similarity tra gli embedding normalizzati

    # Statistiche sulla similarity (esclusa la diagonale = 1.0)
    mask = ~np.eye(len(sim_mat), dtype=bool)#rimuove la diagonale della matrice di similarità,inutile perchè chiaramente avrà tutti 1 
    off_diag = sim_mat[mask]
    print(f"  Cosine similarity — min: {off_diag.min():.4f}, "
          f"max: {off_diag.max():.4f}, "
          f"media: {off_diag.mean():.4f}, "
          f"std: {off_diag.std():.4f}")
    return emb_norm, sim_mat


# ==============================================================================
# STEP 4 — Salvataggio output
# ==============================================================================

def save_outputs(df: pd.DataFrame, emb_norm: np.ndarray,
                 sim_mat: np.ndarray, prompts: list[str]):
    print("\n[4/5] Salvataggio output ...")

    # Embeddings (365, 768)
    emb_path = OUTPUTS / "clip_text_embeddings.npy"
    np.save(emb_path, emb_norm)
    print(f"  Salvato: {emb_path.name}  shape={emb_norm.shape}")

    # Matrice cosine similarity (365, 365)
    sim_path = OUTPUTS / "clip_cosine_similarity.npy"
    np.save(sim_path, sim_mat)
    print(f"  Salvato: {sim_path.name}  shape={sim_mat.shape}")

    # Metadata CSV arricchito
    meta = df[["idx", "full_name", "io"]].copy()
    meta["prompt"] = prompts
    meta_path = OUTPUTS / "clip_category_metadata.csv"
    meta.to_csv(meta_path, index=False)
    print(f"  Salvato: {meta_path.name}")

    return meta


# ==============================================================================
# STEP 5 — Visualizzazioni
# ==============================================================================




def plot_top_pairs(sim_mat: np.ndarray, df: pd.DataFrame,
                   out_dir: Path, top_n: int = 20):
    """
    Due grafici:
      - Le top_n coppie più simili (cosine similarity più alta, esclusa diagonale)
      - Le top_n coppie meno simili (cosine similarity più bassa)

    PERCHÉ: identifica categorie semanticamente vicine per CLIP (utile per
    capire dove il modello potrebbe confondersi) e quelle massimamente
    distanti (estremi dello spazio semantico).
    """
    n = len(sim_mat)
    # Estrai triangolo superiore (senza diagonale) per evitare duplicati
    rows, cols = np.triu_indices(n, k=1)
    sims = sim_mat[rows, cols]
    names = df["full_name"].tolist()

    # Ordina decrescente per le più simili
    sorted_idx = np.argsort(sims)[::-1]

    def get_pairs(indices):
        return [(names[rows[i]].replace("_", " "),
                 names[cols[i]].replace("_", " "),
                 float(sims[i])) for i in indices]

    top_pairs  = get_pairs(sorted_idx[:top_n])
    bot_pairs  = get_pairs(sorted_idx[-top_n:][::-1])

    for pairs, title, fname, color in [
        (top_pairs,  f"Top {top_n} coppie più simili (CLIP)",
         "plot_top_similar_pairs.png",    "#2ecc71"),
        (bot_pairs,  f"Top {top_n} coppie meno simili (CLIP)",
         "plot_top_dissimilar_pairs.png", "#e74c3c"),
    ]:
        labels = [f"{a}  ↔  {b}" for a, b, _ in pairs]
        values = [v for _, _, v in pairs]

        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.barh(range(len(labels)), values, color=color, edgecolor="white")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Cosine similarity", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        # Etichette valori sulle barre
        for bar, val in zip(bars, values):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=7)
        fig.tight_layout()
        out = out_dir / fname
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  Salvato: {out.name}")


def plot_category_self_similarity(sim_mat: np.ndarray, df: pd.DataFrame,
                                  out_dir: Path, top_n: int = 20):
    """
    Per ogni categoria calcola la similarity media con tutte le altre.
    Mostra top_n categorie più "centrali" (alta sim media = vicine a tutto)
    e top_n più "isolate" (bassa sim media = outlier semantici).

    PERCHÉ: le categorie centrali sono candidate ideali per rappresentare
    cluster nel subset selection (Phase 5).
    """
    # Sim media per ogni categoria (esclusa la diagonale)
    n = len(sim_mat)
    sim_sum  = sim_mat.sum(axis=1) - 1.0  # sottrai la diagonale (=1.0)
    sim_mean = sim_sum / (n - 1)

    df_sim = df[["full_name", "io"]].copy()
    df_sim["mean_sim_to_others"] = sim_mean

    top = df_sim.nlargest(top_n, "mean_sim_to_others")
    bot = df_sim.nsmallest(top_n, "mean_sim_to_others")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, sub, title in [
        (axes[0], top, f"Top {top_n} — più 'centrali' (alta sim media)"),
        (axes[1], bot, f"Top {top_n} — più 'isolate' (bassa sim media)"),
    ]:
        labels = [n.replace("_", " ")[:35] for n in sub["full_name"]]
        colors = ["#4C72B0" if io == "indoor" else "#DD8452"
                  for io in sub["io"]]
        ax.barh(range(len(sub)), sub["mean_sim_to_others"],
                color=colors, edgecolor="white")
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Cosine similarity media verso tutte le altre", fontsize=10)
        ax.set_title(title, fontsize=11)
        from matplotlib.patches import Patch
        legend = [Patch(color="#4C72B0", label="indoor"),
                  Patch(color="#DD8452", label="outdoor")]
        ax.legend(handles=legend, fontsize=9)

    fig.suptitle("CLIP Text Embeddings — Centralità semantica delle categorie",
                 fontsize=13)
    fig.tight_layout()
    out = out_dir / "plot_category_centrality.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvato: {out.name}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("Phase 3 — CLIP Text Embeddings")
    print("=" * 60)

    # Carica metadata prodotto dalla Phase 1
    meta_path = OUTPUTS / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(
            "outputs/metadata.csv non trovato. "
            "Esegui prima scripts/1-metadata_EDA.py"
        )
    df = pd.read_csv(meta_path)
    print(f"\nCaricate {len(df)} categorie da metadata.csv")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Step 1: carica modello
    model, tokenize_fn, embed_dim, _ = load_clip_model(device)

    # Step 2: costruisci prompt e codifica
    prompts = build_prompts(df)
    print(f"\nEsempi di prompt:")
    for p in prompts[:5]:
        print(f"  → {p}")

    raw_feats = encode_prompts(model, tokenize_fn, prompts, device)

    # Step 3: normalizza e calcola similarity
    emb_norm, sim_mat = process_embeddings(raw_feats)

    # Step 4: salva output
    meta = save_outputs(df, emb_norm, sim_mat, prompts)

    # Step 5: visualizzazioni
    print("\n[5/5] Generazione grafici ...")
    plot_top_pairs(sim_mat, df, OUTPUTS, top_n=20)
    plot_category_self_similarity(sim_mat, df, OUTPUTS, top_n=20)

    # Stampa riassunto testuale
    print("\n" + "─" * 60)
    print("RIEPILOGO — Top 10 coppie più simili")
    print("─" * 60)
    names = df["full_name"].tolist()
    rows_idx, cols_idx = np.triu_indices(len(sim_mat), k=1)
    sims = sim_mat[rows_idx, cols_idx]
    top10_idx = np.argsort(sims)[::-1][:10]
    for i in top10_idx:
        a = names[rows_idx[i]].replace("_", " ")
        b = names[cols_idx[i]].replace("_", " ")
        print(f"  {sims[i]:.4f}  {a}  ↔  {b}")

    print("\n" + "─" * 60)
    print("RIEPILOGO — Top 10 coppie meno simili")
    print("─" * 60)
    bot10_idx = np.argsort(sims)[:10]
    for i in bot10_idx:
        a = names[rows_idx[i]].replace("_", " ")
        b = names[cols_idx[i]].replace("_", " ")
        print(f"  {sims[i]:.4f}  {a}  ↔  {b}")

    
    print(f"Output in: {OUTPUTS}")
    print("\nFile generati:")
    for f in sorted(OUTPUTS.glob("*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
