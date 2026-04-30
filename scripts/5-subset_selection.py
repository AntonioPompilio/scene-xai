"""
Fase 5 — Subset Selection
==============================================
Selezione casuale di un subset di immagini dal dataset Places365 a partire dal
clustering ottimale (k=13) ottenuto in Fase 4.

Per ciascuno dei 13 cluster:
  - 3 categorie scelte casualmente
  - 750 immagini per categoria, campionate casualmente

Le immagini vengono fisicamente copiate in OUTPUT_DIR mantenendo la struttura:
  output_subset/cluster_<cluster_id>/<category_name>/<image_file>

Output:
  subset_manifest.csv  — traccia di ogni file copiato (cluster, categoria, src, dst)

Esecuzione:
  python scripts/5-subset_selection.py
  python scripts/5-subset_selection.py --base-dir <root> --output-dir <out>
"""

from pathlib import Path
import argparse #per gestire gli argomenti da riga di comando
import random
import shutil #contiene il metodo copy2 per copiare i file da una cartella a un'altra mantenendo i metadati
import sys #per leggere gli argomenti passati da riga di comando
import pandas as pd
from tqdm import tqdm

# ── Riproducibilità ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)

# ── Parametri ─────────────────────────────────────────────────────────────────
K              = 13       # clustering ottimale (Phase 4)
N_CATS_PER_CL  = 3        # categorie da estrarre per cluster
N_IMGS_PER_CAT = 750      # immagini da campionare per categoria
CLUSTER_COL    = f"cluster_k{K}"
IMG_EXTS       = {".jpg", ".jpeg", ".png"}

# ── Path di default ───────────────────────────────────────────────────────────
ROOT             = Path(__file__).parent.parent # per risalire alla root originale del progetto
DEFAULT_BASE_DIR = ROOT
DEFAULT_OUT_DIR  = ROOT / "output_subset"
CLUSTERING_CSV   = ROOT / "outputs" / "clustering_results.csv"

#metodo per gestire gli argomenti da riga di comando, con valori di default e descrizioni
#con questo modo posso runnare lo script con i valori di default o posso specificarli es:
# Specificando tutto
#python script.py --base-dir /path/dataset --output-dir /path/output --clustering-csv /path/clustering_result.csv
def parse_args() -> argparse.Namespace:
    #inizializzo il parser 
    p = argparse.ArgumentParser(description="Estrae un subset bilanciato dal dataset Places365.")
    #assegna il nome --base-dir , di tipo path, il cui valore di default è il path base del progetto
    p.add_argument( 
        "--base-dir", type=Path, default=DEFAULT_BASE_DIR,
        help="Root del progetto dove c'è il dataset Places365 (deve contenere la cartella 'train/'). "
             f"Default: {DEFAULT_BASE_DIR}",
    )
    p.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=f"Cartella di destinazione del subset. Default: {DEFAULT_OUT_DIR}",
    )
    p.add_argument(
        "--clustering-csv", type=Path, default=CLUSTERING_CSV,
        help=f"Path al CSV con i risultati del clustering. Default: {CLUSTERING_CSV}",
    )
    return p.parse_args() #restituisce un oggetto con gli argomenti parsati

#returna il folder per ogni categoria, in modo tale da risolvere i casi di categorie nidificate (es. field/cultivated) 
def category_folder(train_root: Path, full_name: str) -> Path:
    """
    Risolve la cartella di una categoria sul filesystem.
    Le categorie Places365 sono organizzate come train/<lettera>/<full_name>,
    dove full_name può contenere '/' per categorie nidificate (es. field/cultivated).
    """
    letter = full_name[0]
    return train_root / letter / full_name

#preso il patb di una categoria, restituisce la lista dei file immagine al suo interno 
def list_images(folder: Path) -> list[Path]:
    """Restituisce tutti i file immagine nella cartella (non ricorsivo)."""
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def main() -> int:
    args = parse_args()
    base_dir: Path  = args.base_dir.resolve()
    out_dir: Path   = args.output_dir.resolve()
    clust_csv: Path = args.clustering_csv.resolve()
    train_root      = base_dir / "train"

    # ── Validazione input ─────────────────────────────────────────────────────
    #nel caso in cui non venissero trovate le cartelle
    if not train_root.is_dir():
        sys.exit(f"[ERROR] Cartella train non trovata: {train_root}")
    if not clust_csv.is_file():
        sys.exit(f"[ERROR] CSV clustering non trovato: {clust_csv}")

    print(f"[INFO] BASE_DIR     = {base_dir}")
    print(f"[INFO] OUTPUT_DIR   = {out_dir}")
    print(f"[INFO] CLUSTERING   = {clust_csv}")
    print(f"[INFO] k            = {K}  ({CLUSTER_COL})")
    print(f"[INFO] cats/cluster = {N_CATS_PER_CL}, imgs/cat = {N_IMGS_PER_CAT}, seed = {SEED}\n")

    # ── Lettura clustering ────────────────────────────────────────────────────
    df = pd.read_csv(clust_csv, usecols=["full_name", CLUSTER_COL])
    cluster_ids = sorted(df[CLUSTER_COL].unique())
    print(f"[INFO] Trovati {len(cluster_ids)} cluster, {len(df)} categorie totali.")

    # ── Selezione categorie + campionamento + copia ───────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []
    summary: list[tuple[int, list[str], int]] = []
    #ciclo per ogni cluster
    for cid in cluster_ids:
        cats_in_cluster = df.loc[df[CLUSTER_COL] == cid, "full_name"].tolist()#ottengo una lista delle categorie nell' i-esimo cluster
        n_pick = min(N_CATS_PER_CL, len(cats_in_cluster))#prendo il numero di categorie da estrarre, se il cluster è troppo piccolo prenderò il numero di categorie del cluster
        chosen = random.sample(cats_in_cluster, n_pick)#prendo 3 categorie a caso

        copied_in_cluster = 0
        #ciclo per ogni categoria delle scelte dal cluster
        for cat in chosen:
            src_folder = category_folder(train_root, cat)#ottengo il path della cartella della categoria
            if not src_folder.is_dir():
                print(f"  [WARN] Cartella mancante per '{cat}': {src_folder} — skip")
                continue

            all_imgs = list_images(src_folder)#ricavo le immagini per ogni categoria
            if len(all_imgs) < N_IMGS_PER_CAT:#controlo che abbia abbastanza immagini
                print(f"  [WARN] '{cat}' ha solo {len(all_imgs)} immagini "
                      f"(< {N_IMGS_PER_CAT}): le prendo tutte.")
            n_sample = min(N_IMGS_PER_CAT, len(all_imgs))
            sampled = random.sample(all_imgs, n_sample)#samplo le immagini da copiare

            dst_folder = out_dir / f"cluster_{cid}" / cat
            dst_folder.mkdir(parents=True, exist_ok=True)#creo la cartella dove finiranno le immagini
            #ciclo per ogni immagine campionata, la copio nella cartella di destinazione e aggiorno il manifest
            for src in tqdm(sampled, desc=f"cluster {cid:02d} | {cat}", leave=False):
                dst = dst_folder / src.name
                shutil.copy2(src, dst)
                manifest_rows.append({
                    "cluster_id":        cid,
                    "category":          cat,
                    "image_path_source": str(src),
                    "image_path_dest":   str(dst),
                })
                copied_in_cluster += 1

        summary.append((cid, chosen, copied_in_cluster))

    # ── Salvataggio manifest ──────────────────────────────────────────────────
    manifest_path = out_dir / "subset_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"\n[INFO] Manifest salvato: {manifest_path}  ({len(manifest_rows)} righe)")

    # ── Riepilogo finale ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'RIEPILOGO SUBSET':^70}")
    print("=" * 70)
    total = 0
    for cid, chosen, n_copied in summary:
        print(f"  Cluster {cid:2d} | {n_copied:>5d} img | categorie: {', '.join(chosen)}")
        total += n_copied
    print("-" * 70)
    print(f"  TOTALE: {total} immagini copiate in {out_dir}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
