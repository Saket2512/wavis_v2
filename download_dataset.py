"""
WAVIS v2 - Dataset Download & Preparation (BirdCLEF 2023 Edition)
==================================================================
Downloads and prepares datasets:

1. ESC-50        — 50 environmental sound classes (animals subset) ~600MB
2. BirdCLEF-2023 — 264 bird species, ~16K clips (via Kaggle) ~4GB  ← NEW
3. Xeno-Canto    — Curated bird recordings (free, no auth)

BirdCLEF 2023 is much smaller than 2021 (~4GB vs ~35GB) and covers
264 species from soundscapes recorded in Africa, Americas, and Europe.

Usage:
    python download_dataset.py --birdclef2023          # BirdCLEF-2023 + ESC-50 (recommended)
    python download_dataset.py --quick                 # ESC-50 + Xeno-Canto only (~700MB)
    python download_dataset.py --esc50                 # ESC-50 only (fastest, ~600MB)

Kaggle setup (one-time, needed for --birdclef2023):
    1. https://kaggle.com → Account → API → Create New Token → kaggle.json
    2. Place at  Windows: %USERPROFILE%\\.kaggle\\kaggle.json
                 Linux/Mac: ~/.kaggle/kaggle.json
    3. Run: python download_dataset.py --birdclef2023
"""

import os
import sys
import json
import shutil
import zipfile
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ─── Species Taxonomy ─────────────────────────────────────────────────────────

ESC50_ANIMAL_CLASSES = [
    'dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen',
    'insects', 'sheep', 'crow', 'crickets',
]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def download_file(url: str, dest: str, desc: str = "Downloading"):
    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get('content-length', 0))
    with open(dest, 'wb') as f, tqdm(desc=desc, total=total, unit='iB', unit_scale=True) as bar:
        for chunk in response.iter_content(8192):
            f.write(chunk)
            bar.update(len(chunk))


def extract_zip(zip_path: str, extract_to: str):
    print(f"📦 Extracting {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
    os.remove(zip_path)


# ─── Dataset 1: ESC-50 ────────────────────────────────────────────────────────

def download_esc50(data_dir: str = 'data') -> pd.DataFrame:
    extract_path = os.path.join(data_dir, "ESC-50-master")

    if not os.path.exists(extract_path):
        zip_path = os.path.join(data_dir, "esc50.zip")
        download_file(
            "https://github.com/karoldvl/ESC-50/archive/master.zip",
            zip_path, "ESC-50"
        )
        extract_zip(zip_path, data_dir)
    else:
        print("✅ ESC-50 already downloaded.")

    meta = pd.read_csv(os.path.join(extract_path, 'meta', 'esc50.csv'))
    audio_dir = os.path.join(extract_path, 'audio')

    df = meta[meta['category'].isin(ESC50_ANIMAL_CLASSES)].copy()
    df['audio_path'] = df['filename'].apply(lambda f: os.path.join(audio_dir, f))
    df['species'] = df['category']
    df['common_name'] = df['category'].str.replace('_', ' ').str.title()
    df['dataset'] = 'esc50'
    df['sample_rate'] = 44100

    print(f"  ESC-50: {len(df)} clips, {df['species'].nunique()} animal classes")
    return df[['audio_path', 'species', 'common_name', 'fold', 'dataset', 'sample_rate']]


# ─── Dataset 2: BirdCLEF 2023 (via Kaggle) ───────────────────────────────────

def check_kaggle_setup() -> bool:
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(kaggle_json):
        return True
    # Also check Windows USERPROFILE path
    win_path = os.path.join(os.environ.get('USERPROFILE', ''), '.kaggle', 'kaggle.json')
    return os.path.exists(win_path)


def setup_kaggle_instructions():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          HOW TO SET UP KAGGLE API (one-time)                ║
╠══════════════════════════════════════════════════════════════╣
║  1. Go to https://www.kaggle.com → Account → API            ║
║  2. Click "Create New API Token" → downloads kaggle.json    ║
║  3. Move it:                                                 ║
║     Windows:  %USERPROFILE%\\.kaggle\\kaggle.json             ║
║     Linux/Mac: ~/.kaggle/kaggle.json                         ║
║  4. chmod 600 ~/.kaggle/kaggle.json  (Linux/Mac only)        ║
║  5. Re-run: python download_dataset.py --birdclef2023        ║
╚══════════════════════════════════════════════════════════════╝
""")


def download_birdclef2023(data_dir: str = 'data') -> pd.DataFrame:
    """
    Download BirdCLEF 2023 from Kaggle (~4GB).

    Dataset structure:
        train_audio/{species_code}/{recording}.ogg
        train_metadata.csv  — filename, primary_label, common_name, etc.

    Competition: https://www.kaggle.com/competitions/birdclef-2023
    """
    birdclef_dir = os.path.join(data_dir, 'birdclef-2023')
    audio_root   = os.path.join(birdclef_dir, 'train_audio')
    meta_path    = os.path.join(birdclef_dir, 'train_metadata.csv')

    # ── Already downloaded? ──────────────────────────────────────────────────
    existing_oggs = list(Path(birdclef_dir).rglob('*.ogg')) if os.path.exists(birdclef_dir) else []
    if len(existing_oggs) > 100:
        print(f"✅ BirdCLEF-2023 already downloaded ({len(existing_oggs):,} files found).")
    else:
        if not check_kaggle_setup():
            setup_kaggle_instructions()
            return pd.DataFrame()

        print("📥 Downloading BirdCLEF-2023 (~4GB via Kaggle API)...")
        os.makedirs(birdclef_dir, exist_ok=True)

        # Use kaggle CLI
        ret = os.system(
            f'kaggle competitions download -c birdclef-2023 -p "{birdclef_dir}"'
        )
        if ret != 0:
            print("❌ Kaggle download failed. Check your kaggle.json credentials.")
            print("   Also make sure you have accepted the competition rules at:")
            print("   https://www.kaggle.com/competitions/birdclef-2023/rules")
            return pd.DataFrame()

        # Extract all zips
        for zf in sorted(Path(birdclef_dir).glob('*.zip')):
            print(f"📦 Extracting {zf.name}...")
            with zipfile.ZipFile(str(zf), 'r') as z:
                z.extractall(birdclef_dir)
            os.remove(str(zf))
        print("✅ BirdCLEF-2023 extracted successfully.")

    # ── Build metadata DataFrame ─────────────────────────────────────────────
    if not os.path.exists(audio_root):
        print(f"⚠️  Expected train_audio/ not found at: {audio_root}")
        print("   Check extraction was successful.")
        return pd.DataFrame()

    # Load official metadata (has common_name, latitude, longitude, rating etc.)
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path)
        print(f"  Loaded metadata: {len(meta):,} rows, columns: {list(meta.columns)}")
    else:
        print("⚠️  train_metadata.csv not found — will build from folder structure only.")
        meta = pd.DataFrame()

    records = []
    species_dirs = [d for d in Path(audio_root).iterdir() if d.is_dir()]
    print(f"  Indexing {len(species_dirs)} species directories...")

    for sp_dir in tqdm(species_dirs, desc="BirdCLEF-2023"):
        species_code = sp_dir.name  # e.g. "afrsil1"

        # Get common name from metadata
        common_name = species_code  # fallback
        if not meta.empty:
            sp_rows = meta[meta['primary_label'] == species_code]
            if len(sp_rows) > 0 and 'common_name' in sp_rows.columns:
                common_name = sp_rows['common_name'].iloc[0]

        for audio_file in sp_dir.glob('*.ogg'):
            # Get per-file quality rating if available (A=best … E=worst)
            rating = 'A'
            if not meta.empty and 'filename' in meta.columns and 'rating' in meta.columns:
                fname_key = f"train_audio/{species_code}/{audio_file.name}"
                row = meta[meta['filename'] == fname_key]
                if len(row) > 0:
                    rating = str(row['rating'].iloc[0])

            records.append({
                'audio_path':  str(audio_file),
                'species':     species_code,
                'common_name': common_name,
                'fold':        hash(audio_file.name) % 5 + 1,  # deterministic 5-fold
                'dataset':     'birdclef2023',
                'sample_rate': 32000,
                'rating':      rating,
            })

    df = pd.DataFrame(records)
    if df.empty:
        print("⚠️  No .ogg files found inside train_audio/")
        return df

    # Optional: filter to higher-quality recordings only (A and B ratings)
    if 'rating' in df.columns:
        before = len(df)
        df_hq = df[df['rating'].isin(['A', 'B'])].copy()
        if len(df_hq) > 500:  # only filter if we still have enough
            df = df_hq
            print(f"  Quality filter (A/B ratings): {before:,} → {len(df):,} clips")

    print(f"  BirdCLEF-2023: {len(df):,} clips, {df['species'].nunique()} bird species")
    return df[['audio_path', 'species', 'common_name', 'fold', 'dataset', 'sample_rate']]


# ─── Dataset 3: Xeno-Canto Subset (free, no auth) ────────────────────────────

def download_xenocanto_subset(data_dir: str = 'data', n_species: int = 20) -> pd.DataFrame:
    xc_dir    = os.path.join(data_dir, 'xenocanto')
    os.makedirs(xc_dir, exist_ok=True)
    meta_path = os.path.join(xc_dir, 'metadata.csv')

    if os.path.exists(meta_path):
        df = pd.read_csv(meta_path)
        if len(df) > 0:
            print(f"✅ Xeno-Canto subset already downloaded ({len(df)} clips).")
            return df

    target_species = [
        "Corvus corax", "Bubo bubo", "Turdus merula", "Picus viridis",
        "Cuculus canorus", "Passer domesticus", "Hirundo rustica",
        "Luscinia megarhynchos", "Motacilla alba", "Upupa epops",
        "Alcedo atthis", "Garrulus glandarius", "Erithacus rubecula",
        "Sylvia atricapilla", "Troglodytes troglodytes", "Ardea cinerea",
        "Ciconia ciconia", "Milvus migrans", "Buteo buteo", "Falco tinnunculus",
    ][:n_species]

    records = []
    base_url = "https://xeno-canto.org/api/2/recordings"
    print(f"📥 Downloading Xeno-Canto recordings for {len(target_species)} species...")

    for sp_name in tqdm(target_species, desc="Xeno-Canto"):
        try:
            query = sp_name.replace(' ', '+')
            r = requests.get(f"{base_url}?query={query}+q:A&page=1", timeout=15)
            if r.status_code != 200:
                continue
            recordings = r.json().get('recordings', [])[:10]
            sp_dir = os.path.join(xc_dir, sp_name.replace(' ', '_'))
            os.makedirs(sp_dir, exist_ok=True)

            for rec in recordings:
                file_url = 'https:' + rec.get('file', '')
                filename = f"{rec.get('id', 'unknown')}.mp3"
                dest = os.path.join(sp_dir, filename)
                if not os.path.exists(dest) and file_url.startswith('https:'):
                    try:
                        r2 = requests.get(file_url, timeout=30, stream=True)
                        if r2.status_code == 200:
                            with open(dest, 'wb') as f:
                                for chunk in r2.iter_content(8192):
                                    f.write(chunk)
                    except Exception:
                        continue
                if os.path.exists(dest):
                    records.append({
                        'audio_path':  dest,
                        'species':     sp_name.replace(' ', '_').lower(),
                        'common_name': rec.get('en', sp_name),
                        'fold':        hash(filename) % 5 + 1,
                        'dataset':     'xenocanto',
                        'sample_rate': 44100,
                    })
        except Exception as e:
            print(f"  ⚠️ Failed {sp_name}: {e}")

    df = pd.DataFrame(records)
    if len(df) > 0:
        df.to_csv(meta_path, index=False)
        print(f"  Xeno-Canto: {len(df)} clips, {df['species'].nunique()} species")
    return df


# ─── Combine & Finalize ───────────────────────────────────────────────────────

def build_combined_dataset(dfs: list, data_dir: str = 'data',
                            min_samples: int = 10, max_samples: int = 500) -> pd.DataFrame:
    combined = pd.concat([df for df in dfs if len(df) > 0], ignore_index=True)
    combined = combined.dropna(subset=['audio_path', 'species'])

    # Remove species with too few samples
    counts = combined['species'].value_counts()
    valid_species = counts[counts >= min_samples].index
    dropped = len(counts) - len(valid_species)
    if dropped > 0:
        print(f"  ℹ️  Dropped {dropped} species with fewer than {min_samples} clips")
    combined = combined[combined['species'].isin(valid_species)].copy()

    # Cap overrepresented classes
    balanced = []
    for sp, grp in combined.groupby('species'):
        if len(grp) > max_samples:
            grp = grp.sample(max_samples, random_state=42)
        balanced.append(grp)
    combined = pd.concat(balanced, ignore_index=True)

    # Assign integer labels
    species_sorted = sorted(combined['species'].unique())
    sp_to_idx      = {sp: i for i, sp in enumerate(species_sorted)}
    combined['label'] = combined['species'].map(sp_to_idx)

    # Save
    os.makedirs(data_dir, exist_ok=True)
    out_csv = os.path.join(data_dir, 'combined_dataset.csv')
    combined.to_csv(out_csv, index=False)

    # Class map  {str(idx): species_code}
    class_map = {str(i): sp for sp, i in sp_to_idx.items()}
    with open(os.path.join(data_dir, 'class_map.json'), 'w') as f:
        json.dump(class_map, f, indent=2)

    # Display names  {species_code: common_name}
    display_map = {}
    for _, row in combined.drop_duplicates('species').iterrows():
        display_map[row['species']] = row['common_name']
    with open(os.path.join(data_dir, 'display_names.json'), 'w') as f:
        json.dump(display_map, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  📊 Combined Dataset Summary")
    print(f"{'='*55}")
    print(f"  Total clips    : {len(combined):,}")
    print(f"  Total species  : {combined['species'].nunique()}")
    print(f"  Datasets used  : {combined['dataset'].unique().tolist()}")
    print(f"\n  Top 10 species by count:")
    top10 = combined['species'].value_counts().head(10)
    for sp, cnt in top10.items():
        name = display_map.get(sp, sp)
        print(f"    {name[:35]:<35} {cnt:>4} clips")
    print(f"{'='*55}")
    print(f"\n✅ Dataset saved to {out_csv}")
    print(f"✅ Class map saved to {data_dir}/class_map.json")
    return combined


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="WAVIS v2 Dataset Downloader — BirdCLEF 2023 Edition"
    )
    parser.add_argument('--birdclef2023', action='store_true',
                        help='ESC-50 + BirdCLEF-2023 via Kaggle (~4GB) — RECOMMENDED')
    parser.add_argument('--quick',        action='store_true',
                        help='ESC-50 + Xeno-Canto subset (~700MB, no Kaggle needed)')
    parser.add_argument('--esc50',        action='store_true',
                        help='ESC-50 only (~600MB, fastest)')
    parser.add_argument('--data-dir',     default='data')
    args = parser.parse_args()

    # Default = birdclef2023 mode
    if not any([args.birdclef2023, args.quick, args.esc50]):
        args.birdclef2023 = True

    os.makedirs(args.data_dir, exist_ok=True)
    dfs = []

    print("\n🌿 WAVIS v2 — Dataset Downloader (BirdCLEF 2023 Edition)")
    print("=" * 55)

    # Step 1: ESC-50 baseline (always included)
    print("\n[1] ESC-50 Animal Sounds (~600MB)")
    dfs.append(download_esc50(args.data_dir))

    if args.quick:
        print("\n[2] Xeno-Canto Bird Recordings (free, ~100MB)")
        dfs.append(download_xenocanto_subset(args.data_dir, n_species=20))

    if args.birdclef2023:
        print("\n[2] BirdCLEF-2023 (Kaggle, ~4GB, 264 species)")
        df_bc23 = download_birdclef2023(args.data_dir)
        if len(df_bc23) == 0:
            print("⚠️  BirdCLEF-2023 not available — continuing with ESC-50 only.")
            print("    Tip: Set up kaggle.json and re-run with --birdclef2023")
        else:
            dfs.append(df_bc23)

    print("\n🔧 Building combined dataset...")
    # Use lower min_samples for BirdCLEF which has uneven species counts
    min_s = 5 if args.birdclef2023 else 20
    build_combined_dataset(dfs, args.data_dir, min_samples=min_s)

    print(f"\n🎉 Ready! Next step:")
    print(f"   python train.py")


if __name__ == '__main__':
    main()