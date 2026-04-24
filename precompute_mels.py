import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from utils.audio_features import load_audio, normalize_audio, audio_to_mel_image, SR

# ─── CONFIG ─────────────────────────────────────
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "combined_dataset.csv")
CACHE_DIR = os.path.join(DATA_DIR, "mel_cache")

os.makedirs(CACHE_DIR, exist_ok=True)

# ─── LOAD DATA ──────────────────────────────────
df = pd.read_csv(CSV_PATH)

print(f"🚀 Precomputing mel spectrograms for {len(df)} files...")
print(f"📁 Saving to: {CACHE_DIR}\n")

# ─── PROCESS ────────────────────────────────────
success, failed = 0, 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    audio_path = row["audio_path"]

    # Unique filename based on audio file name
    fname = Path(audio_path).stem + ".npy"
    save_path = os.path.join(CACHE_DIR, fname)

    # Skip if already processed
    if os.path.exists(save_path):
        success += 1
        continue

    try:
        # Load + process audio
        y = load_audio(audio_path, sr=SR, duration=5.0)
        y = normalize_audio(y)

        # Convert to mel image
        img = audio_to_mel_image(y, sr=SR, augment=False)

        # Save as numpy
        np.save(save_path, img.astype(np.float32))
        success += 1

    except Exception as e:
        # Save dummy if error
        dummy = np.zeros((3, 224, 224), dtype=np.float32)
        np.save(save_path, dummy)
        failed += 1

print("\n✅ Precomputation complete!")
print(f"✔ Success: {success}")
print(f"❌ Failed: {failed}")