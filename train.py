"""
WAVIS v2 - High Accuracy Training Script
==========================================
Two-phase training strategy to hit 95%+ accuracy:

  Phase 1 (epochs 1-10):  Freeze backbone, train head only
                           → Fast convergence, no catastrophic forgetting
  Phase 2 (epochs 11-50): Unfreeze backbone, fine-tune everything
                           → Backbone adapts to wildlife sounds

Additional techniques:
  - MixUp augmentation (alpha=0.4)
  - SpecAugment (time/freq masking)
  - Cosine LR with warmup
  - Label smoothing (0.05)
  - Automatic Mixed Precision (AMP) for GPU speed
  - Gradient accumulation (handles larger effective batch size)

Usage:
    python train.py                     # default (50 epochs, GPU if available)
    python train.py --epochs 30         # quick training
    python train.py --backbone efficientnet_b5   # more accurate, slower
    python train.py --resume            # resume from checkpoint
"""

import os
import sys
import json
import time
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import classification_report, top_k_accuracy_score
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(__file__))
from utils.audio_features import load_audio, normalize_audio, audio_to_mel_image, SR
from models.efficientnet_model import WildlifeEfficientNet, mixup_data, mixup_criterion

try:
    import audiomentations as A
    HAS_AUDIOMENTATIONS = True
except ImportError:
    HAS_AUDIOMENTATIONS = False
    print("⚠️  audiomentations not installed — using basic augmentation")


# ─── Dataset ──────────────────────────────────────────────────────────────────

class WildlifeDataset(Dataset):
    """
    PyTorch Dataset for high-accuracy wildlife sound classification.
    Applies aggressive augmentation during training.
    """

    def __init__(self, df: pd.DataFrame, cfg: dict, mode: str = 'train'):
        self.df     = df.reset_index(drop=True)
        self.cfg    = cfg
        self.mode   = mode  # 'train' | 'val' | 'test'
        self.augment = (mode == 'train')

        # Audio augmentation pipeline (time domain)
        if self.augment and HAS_AUDIOMENTATIONS:
            self.audio_aug = A.Compose([
                A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.4),
                A.TimeStretch(min_rate=0.8, max_rate=1.2, p=0.3),
                A.PitchShift(min_semitones=-3, max_semitones=3, p=0.3),
                A.Shift(min_shift=-0.2, max_shift=0.2, p=0.5),
                A.Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
                A.LowPassFilter(min_cutoff_freq=4000, max_cutoff_freq=16000, p=0.2),
            ])
        else:
            self.audio_aug = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = int(row['label'])

        try:
            y = load_audio(str(row['audio_path']), sr=SR, duration=5.0)
            y = normalize_audio(y)
        except Exception:
            y = np.zeros(int(SR * 5.0), dtype=np.float32)

        # Time-domain augmentation
        if self.augment:
            if self.audio_aug is not None:
                y = self.audio_aug(samples=y, sample_rate=SR)
            else:
                y = self._basic_augment(y)

        # Convert to 3-channel mel image with SpecAugment
        img = audio_to_mel_image(y, sr=SR, augment=self.augment)  # (3, 224, 224)

        # ImageNet normalization (since we use pretrained EfficientNet)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        img  = (img - mean) / std

        return torch.FloatTensor(img), label

    def _basic_augment(self, y: np.ndarray) -> np.ndarray:
        """Fallback augmentation without audiomentations."""
        # Time shift
        if np.random.random() < 0.5:
            shift = int(np.random.uniform(-0.2, 0.2) * len(y))
            y = np.roll(y, shift)
        # Noise
        if np.random.random() < 0.4:
            y += np.random.randn(len(y)).astype(np.float32) * 0.005
        # Gain
        if np.random.random() < 0.5:
            y *= np.random.uniform(0.7, 1.3)
        return y


# ─── Trainer ──────────────────────────────────────────────────────────────────

class WAVISTrainer:
    def __init__(self, model: WildlifeEfficientNet, cfg: dict, device: str):
        self.model  = model.to(device)
        self.cfg    = cfg
        self.device = device
        self.scaler = GradScaler(enabled=(cfg['training']['amp'] and device != 'cpu'))
        self.history = {k: [] for k in
                        ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'val_top5']}
        self.best_val_acc = 0.0

    def _get_optimizer(self, phase: int):
        """
        Phase 1: only head params (high LR)
        Phase 2: all params (backbone at 0.1x LR, head at 1x LR)
        """
        lr   = self.cfg['training']['learning_rate']
        wd   = self.cfg['training']['weight_decay']

        if phase == 1:
            params = [{'params': self.model.head.parameters(), 'lr': lr}]
        else:
            params = [
                {'params': self.model.backbone.parameters(), 'lr': lr * 0.1},
                {'params': self.model.head.parameters(),     'lr': lr},
            ]
        return optim.AdamW(params, weight_decay=wd)

    def _get_scheduler(self, optimizer, n_epochs: int, warmup: int):
        def lr_lambda(epoch):
            if epoch < warmup:
                return epoch / max(warmup, 1)
            progress = (epoch - warmup) / max(n_epochs - warmup, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def train_epoch(self, loader, optimizer, criterion, use_mixup: bool = True):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        acc_steps = self.cfg['training']['accumulate_grad']

        optimizer.zero_grad()
        for step, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            # MixUp augmentation
            if use_mixup and np.random.random() < 0.5:
                imgs, y_a, y_b, lam = mixup_data(
                    imgs, labels, self.cfg['training']['mixup_alpha']
                )
                with autocast(enabled=self.scaler.is_enabled()):
                    outputs = self.model(imgs)
                    loss = mixup_criterion(criterion, outputs, y_a, y_b, lam) / acc_steps
            else:
                with autocast(enabled=self.scaler.is_enabled()):
                    outputs = self.model(imgs)
                    loss = criterion(outputs, labels) / acc_steps

            self.scaler.scale(loss).backward()

            # Gradient accumulation
            if (step + 1) % acc_steps == 0:
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.cfg['training']['grad_clip'])
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * acc_steps * imgs.size(0)
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total   += imgs.size(0)

        return total_loss / total, 100.0 * correct / total

    @torch.no_grad()
    def eval_epoch(self, loader, criterion):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_probs, all_labels = [], []

        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with autocast(enabled=self.scaler.is_enabled()):
                outputs = self.model(imgs)
                loss    = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, pred = probs.max(1)
            correct += pred.eq(labels).sum().item()
            total   += imgs.size(0)
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        all_probs  = np.vstack(all_probs)
        all_labels = np.array(all_labels)
        top5 = top_k_accuracy_score(all_labels, all_probs,
                                     k=min(5, all_probs.shape[1]),
                                     labels=list(range(all_probs.shape[1]))) * 100

        return total_loss / total, 100.0 * correct / total, top5, all_probs, all_labels

    def train(self, train_loader, val_loader, total_epochs: int,
              phase1_epochs: int = 10,
              save_path: str = 'models/wavis_v2_best.pth'):

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        criterion = nn.CrossEntropyLoss(
            label_smoothing=self.cfg['model']['label_smoothing']
        )

        print(f"\n{'='*65}")
        print(f"  WAVIS v2 — EfficientNet Training")
        print(f"  Backbone   : {self.cfg['model']['backbone']}")
        print(f"  Device     : {self.device}")
        print(f"  AMP        : {self.cfg['training']['amp']}")
        print(f"  Total epochs  : {total_epochs}")
        print(f"  Phase 1 (frozen backbone): epochs 1–{phase1_epochs}")
        print(f"  Phase 2 (full fine-tune) : epochs {phase1_epochs+1}–{total_epochs}")
        print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
        print(f"{'='*65}\n")

        current_phase = 0

        for epoch in range(1, total_epochs + 1):

            # ── Phase switching ───────────────────────────────────────────────
            if epoch == 1:
                print("🔒 Phase 1: Training head only (backbone frozen)")
                self.model.freeze_backbone()
                optimizer  = self._get_optimizer(phase=1)
                scheduler  = self._get_scheduler(optimizer, phase1_epochs,
                                                  warmup=2)
                current_phase = 1

            elif epoch == phase1_epochs + 1:
                print(f"\n🔓 Phase 2: Full fine-tuning (epoch {epoch}/{total_epochs})")
                self.model.unfreeze_backbone()
                optimizer  = self._get_optimizer(phase=2)
                remaining  = total_epochs - phase1_epochs
                scheduler  = self._get_scheduler(optimizer, remaining,
                                                  warmup=self.cfg['training']['warmup_epochs'])
                current_phase = 2

            # ── Train + Eval ──────────────────────────────────────────────────
            t0 = time.time()
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, criterion,
                use_mixup=(current_phase == 2)  # MixUp only in phase 2
            )
            val_loss, val_acc, val_top5, _, _ = self.eval_epoch(val_loader, criterion)
            scheduler.step()
            elapsed = time.time() - t0

            # Record
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['val_top5'].append(val_top5)

            # Save best
            star = ""
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_acc': val_acc,
                    'val_top5': val_top5,
                    'num_classes': self.model.num_classes,
                    'backbone': self.cfg['model']['backbone'],
                    'config': self.cfg,
                }, save_path)
                star = " ⭐"

            lr_now = optimizer.param_groups[-1]['lr']
            print(f"  [{epoch:2d}/{total_epochs}] "
                  f"Train {train_acc:.1f}% | "
                  f"Val {val_acc:.1f}% (Top5: {val_top5:.1f}%) | "
                  f"Loss {val_loss:.4f} | "
                  f"LR {lr_now:.2e} | "
                  f"{elapsed:.0f}s{star}")

        print(f"\n✅ Training complete!")
        print(f"   Best Val Accuracy (Top-1): {self.best_val_acc:.2f}%")
        return self.history


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',   default='configs/config.yaml')
    parser.add_argument('--epochs',   type=int, default=None)
    parser.add_argument('--backbone', default=None,
                        help='efficientnet_b3 | efficientnet_b5 | vit_small_patch16_224')
    parser.add_argument('--resume',   action='store_true')
    parser.add_argument('--data-dir', default='data')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.epochs:   cfg['training']['epochs']          = args.epochs
    if args.backbone: cfg['model']['backbone']           = args.backbone

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  Running on CPU — training will be slow. Consider Google Colab for GPU.")

    # Load dataset
    csv_path  = os.path.join(args.data_dir, 'combined_dataset.csv')
    map_path  = os.path.join(args.data_dir, 'class_map.json')

    if not os.path.exists(csv_path):
        print("❌ Dataset not found! Run: python download_dataset.py first")
        return

    df = pd.read_csv(csv_path)
    with open(map_path) as f:
        class_map = json.load(f)
    num_classes = len(class_map)

    print(f"\n📊 Dataset: {len(df):,} clips | {num_classes} species")

    # Train/val/test split by fold
    train_df = df[~df['fold'].isin([4, 5])].copy()
    val_df   = df[df['fold'] == 4].copy()
    test_df  = df[df['fold'] == 5].copy()
    print(f"   Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # Datasets
    train_ds = WildlifeDataset(train_df, cfg, mode='train')
    val_ds   = WildlifeDataset(val_df,   cfg, mode='val')
    test_ds  = WildlifeDataset(test_df,  cfg, mode='test')

    nw = cfg['training']['num_workers'] if device == 'cuda' else 0
    bs = cfg['training']['batch_size']

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                               num_workers=nw, pin_memory=(device=='cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                               num_workers=nw, pin_memory=(device=='cuda'))
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                               num_workers=0)

    # Model
    model = WildlifeEfficientNet(
        num_classes=num_classes,
        backbone=cfg['model']['backbone'],
        pretrained=cfg['model']['pretrained'],
        dropout=cfg['model']['dropout'],
    )
    print(f"\n🧠 Model: {cfg['model']['backbone']}")
    print(f"   Total params:     {model.count_parameters(False):,}")
    print(f"   Trainable params: {model.count_parameters(True):,}")

    save_path = os.path.join(cfg['paths']['model_dir'], 'wavis_v2_best.pth')

    # Resume if requested
    if args.resume and os.path.exists(save_path):
        print(f"📂 Resuming from {save_path}")
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

    # Train
    trainer = WAVISTrainer(model, cfg, device)
    history = trainer.train(
        train_loader, val_loader,
        total_epochs=cfg['training']['epochs'],
        phase1_epochs=10,
        save_path=save_path,
    )

    # Final test evaluation
    print("\n📊 Final evaluation on held-out test set...")
    from models.efficientnet_model import load_model
    best_model = load_model(save_path, num_classes, cfg['model']['backbone'], device)
    trainer_eval = WAVISTrainer(best_model, cfg, device)
    criterion = nn.CrossEntropyLoss()
    _, test_acc, test_top5, all_probs, all_labels = trainer_eval.eval_epoch(
        test_loader, criterion
    )

    print(f"\n{'='*55}")
    print(f"  🎯 Test Top-1 Accuracy : {test_acc:.2f}%")
    print(f"  🎯 Test Top-5 Accuracy : {test_top5:.2f}%")
    print(f"{'='*55}")

    class_names = [class_map[str(i)] for i in range(num_classes)]
    preds = np.argmax(all_probs, axis=1)
    print("\nPer-class Report (top 20 classes):")
    report = classification_report(all_labels, preds, target_names=class_names,
                                    zero_division=0, output_dict=True)
    sorted_classes = sorted([(k, v['f1-score']) for k, v in report.items()
                              if isinstance(v, dict) and 'f1-score' in v],
                             key=lambda x: x[1], reverse=True)
    for cls, f1 in sorted_classes[:20]:
        acc_row = report[cls]
        print(f"  {cls[:30]:<30} F1={f1:.3f}  Prec={acc_row['precision']:.3f}  Rec={acc_row['recall']:.3f}")

    # Save history
    os.makedirs('models', exist_ok=True)
    with open('models/training_history.json', 'w') as f:
        json.dump({'history': history, 'test_top1': test_acc, 'test_top5': test_top5}, f)

    print(f"\n🎉 Done! Launch demo: streamlit run app.py")


if __name__ == '__main__':
    main()
