"""
WAVIS v2 - Multi-Agent Pipeline
================================
5 agents with species ID, distance, AND direction:

  Agent 1: SoundAcquisitionAgent   — validate, load, normalize
  Agent 2: SoundSeparationAgent    — denoise, segment, extract features
  Agent 3: ClassificationAgent     — EfficientNet species ID (95%+)
  Agent 4: LocalizationAgent       — distance + direction estimation
  Agent 5: DecisionFusionAgent     — combine, format, output
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.audio_features import (
    load_audio, normalize_audio, audio_to_mel_image,
    extract_acoustic_features, estimate_distance, estimate_direction,
    SR, IMG_SIZE
)
from models.efficientnet_model import WildlifeEfficientNet


# ─── Animal Facts & Emoji Map ─────────────────────────────────────────────────

ANIMAL_EMOJIS = {
    # Birds
    'crow': '🐦‍⬛', 'rooster': '🐓', 'hen': '🐔',
    'corvus_corax': '🪶', 'bubo_bubo': '🦉',
    'turdus_merula': '🐦', 'cuculus_canorus': '🐦',
    # Mammals
    'dog': '🐕', 'cat': '🐈', 'cow': '🐄',
    'pig': '🐖', 'sheep': '🐑',
    # Amphibians
    'frog': '🐸',
    # Insects
    'insects': '🦗', 'crickets': '🦗',
}

DEFAULT_EMOJI = '🦁'

ANIMAL_FACTS = {
    'crow':    "Crows can recognize human faces and hold grudges. They communicate in regional dialects.",
    'rooster': "Roosters crow to establish territory. Their crow is timed by an internal circadian clock.",
    'frog':    "Only male frogs call. Each species has a unique call frequency — like a biological fingerprint.",
    'dog':     "Dogs have 18 muscles controlling each ear, allowing precise sound localization.",
    'cat':     "Cats meow almost exclusively at humans — in the wild, adult cats rarely vocalize.",
    'cow':     "Cows have distinct 'voices'. Researchers can identify individual cows by their moo.",
    'pig':     "Pigs have 20+ distinct vocalizations — different sounds for hunger, stress, happiness.",
    'sheep':   "Sheep mothers and lambs recognize each other's unique bleats within hours of birth.",
    'insects': "Insect sounds (stridulation) are produced by rubbing body parts together.",
    'crickets': "Cricket chirp rate is directly correlated with ambient temperature (Dolbear's Law).",
}

DIRECTION_TIPS = {
    'Left':             "🔊 Sound appears to come from your left. Turn left to face the animal.",
    'Right':            "🔊 Sound appears to come from your right. Turn right to face the animal.",
    'Center / Ahead':   "🔊 Sound appears to be directly ahead of you.",
}

ELEVATION_TIPS = {
    'Above (tree canopy / flying)': "🌳 Animal may be perched high in the canopy or flying.",
    'Ground level':                  "🌿 Animal appears to be at ground level.",
    'Eye level':                     "👁️ Animal likely at eye level — check shrubs and mid-height.",
}


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class AudioSignal:
    audio:      np.ndarray
    sr:         int
    duration:   float
    is_valid:   bool
    error:      Optional[str] = None


@dataclass
class ProcessedSignal:
    audio:          np.ndarray
    sr:             int
    mel_image:      Optional[np.ndarray]      # (3, 224, 224) for CNN
    mel_matrix:     Optional[np.ndarray]      # (128, T) for display
    features:       dict = field(default_factory=dict)
    snr_db:         float = 0.0


@dataclass
class ClassificationResult:
    species:            str
    common_name:        str
    emoji:              str
    confidence_pct:     float
    top_k:              List[dict]            # [{species, name, emoji, conf}, ...]
    fact:               str = ""


@dataclass
class LocalizationResult:
    # Distance
    dist_category:      str
    dist_range:         str
    dist_emoji:         str
    dist_color:         str
    dist_confidence:    float
    dist_warning:       str
    proximity_score:    float
    # Direction
    horizontal:         str
    h_arrow:            str
    h_confidence:       float
    vertical:           str
    v_arrow:            str
    v_confidence:       float
    direction_tip:      str
    elevation_tip:      str
    direction_disclaimer: str
    # Raw features
    rms_energy:         float
    spectral_rolloff:   float


@dataclass
class FinalOutput:
    # Species
    species:            str
    common_name:        str
    emoji:              str
    confidence:         float
    top_k:              List[dict]
    fact:               str
    # Distance
    dist_category:      str
    dist_range:         str
    dist_emoji:         str
    dist_color:         str
    dist_confidence:    float
    dist_warning:       str
    proximity_score:    float
    # Direction
    horizontal:         str
    h_arrow:            str
    h_confidence:       float
    vertical:           str
    v_arrow:            str
    v_confidence:       float
    direction_tip:      str
    elevation_tip:      str
    direction_disclaimer: str
    # Raw acoustic features (for Details tab)
    rms_energy:         float
    spectral_rolloff:   float
    # Meta
    processing_ms:      float
    agents_status:      dict
    snr_db:             float


# ─── Agent 1: Sound Acquisition ───────────────────────────────────────────────

class SoundAcquisitionAgent:
    name = "SoundAcquisitionAgent"

    def process(self, audio_input) -> AudioSignal:
        try:
            if isinstance(audio_input, (str, os.PathLike)):
                y = load_audio(str(audio_input), sr=SR, duration=5.0)
            elif isinstance(audio_input, np.ndarray):
                y = audio_input.astype(np.float32)
                target = int(SR * 5.0)
                y = y[:target] if len(y) >= target else np.pad(y, (0, target - len(y)))
            elif isinstance(audio_input, bytes):
                import soundfile as sf, io
                y, orig_sr = sf.read(io.BytesIO(audio_input))
                if y.ndim > 1: y = y.mean(axis=1)
                import librosa
                y = librosa.resample(y.astype(np.float32), orig_sr=orig_sr, target_sr=SR)
                target = int(SR * 5.0)
                y = y[:target] if len(y) >= target else np.pad(y, (0, target - len(y)))
            else:
                raise ValueError(f"Unsupported input type: {type(audio_input)}")

            if np.max(np.abs(y)) < 1e-7:
                return AudioSignal(y, SR, 5.0, False, "Audio is silent. Check mic input.")

            y = normalize_audio(y)
            return AudioSignal(y, SR, len(y) / SR, True)

        except Exception as e:
            return AudioSignal(np.zeros(int(SR * 5.0), dtype=np.float32),
                               SR, 5.0, False, str(e))


# ─── Agent 2: Sound Separation / Preprocessing ───────────────────────────────

class SoundSeparationAgent:
    name = "SoundSeparationAgent"

    def process(self, sig: AudioSignal) -> ProcessedSignal:
        from utils.audio_features import get_mel_matrix
        y, sr = sig.audio.copy(), sig.sr

        # ── Lightweight noise reduction (spectral subtraction) ────────────────
        noise_len = min(int(0.5 * sr), len(y) // 5)
        if noise_len > 100:
            import scipy.signal as scipy_sig
            _, _, Zxx = scipy_sig.stft(y, fs=sr, nperseg=512)
            noise_profile = np.mean(np.abs(Zxx[:, :noise_len // 256 + 1]), axis=1, keepdims=True)
            mag = np.abs(Zxx)
            phase = np.angle(Zxx)
            mag_clean = np.maximum(mag - 1.2 * noise_profile, 0.05 * mag)
            _, y_clean = scipy_sig.istft(mag_clean * np.exp(1j * phase), fs=sr, nperseg=512)
            y_clean = y_clean[:len(y)].astype(np.float32)
            if len(y_clean) < len(y):
                y_clean = np.pad(y_clean, (0, len(y) - len(y_clean)))
        else:
            y_clean = y

        # ── Best 5s window selection ──────────────────────────────────────────
        target_len = int(5.0 * sr)
        if len(y_clean) > target_len:
            step = sr // 4
            best_start, best_e = 0, -1
            for start in range(0, len(y_clean) - target_len, step):
                e = np.sum(y_clean[start:start + target_len] ** 2)
                if e > best_e:
                    best_e, best_start = e, start
            y_clean = y_clean[best_start:best_start + target_len]

        # ── SNR estimate ──────────────────────────────────────────────────────
        sig_pwr   = np.mean(y_clean ** 2)
        noise_pwr = np.mean(y[:noise_len] ** 2) if noise_len > 0 else 1e-8
        snr       = 10 * np.log10(sig_pwr / (noise_pwr + 1e-8))

        # ── Features + Images ──────────────────────────────────────────────────
        mel_img    = audio_to_mel_image(y_clean, sr=sr, augment=False)  # (3, 224, 224)
        mel_matrix = get_mel_matrix(y_clean, sr=sr)                     # (128, T) for display
        features   = extract_acoustic_features(y_clean, sr)

        return ProcessedSignal(
            audio=y_clean, sr=sr,
            mel_image=mel_img, mel_matrix=mel_matrix,
            features=features, snr_db=float(snr)
        )


# ─── Agent 3: Species Classification ─────────────────────────────────────────

class ClassificationAgent:
    name = "ClassificationAgent"

    def __init__(self, model_path: str, class_map_path: str,
                 display_names_path: str, device: str = 'cpu'):
        self.device       = device
        self.model        = None
        self.class_map    = {}
        self.display_names = {}
        self.model_loaded = False
        self._load(model_path, class_map_path, display_names_path)

    def _load(self, model_path, class_map_path, display_names_path):
        try:
            with open(class_map_path)     as f: self.class_map     = json.load(f)
            with open(display_names_path) as f: self.display_names = json.load(f)
        except FileNotFoundError:
            pass

        if os.path.exists(model_path) and self.class_map:
            try:
                ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
                backbone = ckpt.get('backbone', 'efficientnet_b3')
                self.model = WildlifeEfficientNet(
                    num_classes=len(self.class_map),
                    backbone=backbone, pretrained=False
                )
                self.model.load_state_dict(ckpt['model_state_dict'])
                self.model.eval().to(self.device)
                self.model_loaded = True
                print(f"✅ Model loaded: {backbone} ({len(self.class_map)} classes)")
            except Exception as e:
                print(f"⚠️  Model load failed: {e}")

    def process(self, sig: ProcessedSignal) -> ClassificationResult:
        if not self.model_loaded:
            return self._demo_result()

        # Normalize with ImageNet stats
        img = sig.mel_image.copy()   # (3, 224, 224)
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img  = (img - mean) / std

        tensor = torch.FloatTensor(img).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)

        with torch.no_grad():
            probs = self.model.predict_proba(tensor)[0].cpu().numpy()

        top_idx = np.argsort(probs)[::-1][:10]

        def make_entry(i):
            sp   = self.class_map[str(i)]
            name = self.display_names.get(sp, sp.replace('_', ' ').title())
            emoji = ANIMAL_EMOJIS.get(sp, DEFAULT_EMOJI)
            return {'species': sp, 'name': name, 'emoji': emoji, 'conf': float(probs[i] * 100)}

        top1   = make_entry(top_idx[0])
        top_k  = [make_entry(i) for i in top_idx]

        return ClassificationResult(
            species=top1['species'],
            common_name=top1['name'],
            emoji=top1['emoji'],
            confidence_pct=top1['conf'],
            top_k=top_k,
            fact=ANIMAL_FACTS.get(top1['species'], ""),
        )

    def _demo_result(self) -> ClassificationResult:
        sp = 'crow'
        return ClassificationResult(
            species=sp,
            common_name='Common Crow',
            emoji='🐦‍⬛',
            confidence_pct=72.0,
            top_k=[
                {'species': sp,     'name': 'Common Crow',  'emoji': '🐦‍⬛', 'conf': 72.0},
                {'species': 'frog', 'name': 'Common Frog',  'emoji': '🐸',   'conf': 14.0},
                {'species': 'dog',  'name': 'Dog',          'emoji': '🐕',   'conf': 8.0},
            ],
            fact="⚠️ Demo mode — train the model for real predictions!",
        )


# ─── Agent 4: Localization (Distance + Direction) ─────────────────────────────

class LocalizationAgent:
    name = "LocalizationAgent"

    def process(self, sig: ProcessedSignal) -> LocalizationResult:
        dist = estimate_distance(sig.features)
        dirn = estimate_direction(sig.features)

        return LocalizationResult(
            # Distance
            dist_category=dist['category'],
            dist_range=dist['range'],
            dist_emoji=dist['emoji'],
            dist_color=dist['color'],
            dist_confidence=dist['confidence'],
            dist_warning=dist['warning'],
            proximity_score=dist['proximity_score'],
            # Direction
            horizontal=dirn['horizontal'],
            h_arrow=dirn['h_arrow'],
            h_confidence=dirn['h_confidence'],
            vertical=dirn['vertical'],
            v_arrow=dirn['v_arrow'],
            v_confidence=dirn['v_confidence'],
            direction_tip=DIRECTION_TIPS.get(dirn['horizontal'],
                          "🔊 Face the direction the sound is loudest."),
            elevation_tip=ELEVATION_TIPS.get(dirn['vertical'], ""),
            direction_disclaimer=dirn['disclaimer'],
            # Raw
            rms_energy=dist['rms_energy'],
            spectral_rolloff=dist['spectral_rolloff_hz'],
        )


# ─── Agent 5: Decision Fusion ─────────────────────────────────────────────────

class DecisionFusionAgent:
    name = "DecisionFusionAgent"

    def process(self, cls: ClassificationResult, loc: LocalizationResult,
                sig: ProcessedSignal, agents_status: dict,
                t_ms: float) -> FinalOutput:
        return FinalOutput(
            # Species
            species=cls.species,
            common_name=cls.common_name,
            emoji=cls.emoji,
            confidence=cls.confidence_pct,
            top_k=cls.top_k,
            fact=cls.fact,
            # Distance
            dist_category=loc.dist_category,
            dist_range=loc.dist_range,
            dist_emoji=loc.dist_emoji,
            dist_color=loc.dist_color,
            dist_confidence=loc.dist_confidence,
            dist_warning=loc.dist_warning,
            proximity_score=loc.proximity_score,
            # Direction
            horizontal=loc.horizontal,
            h_arrow=loc.h_arrow,
            h_confidence=loc.h_confidence,
            vertical=loc.vertical,
            v_arrow=loc.v_arrow,
            v_confidence=loc.v_confidence,
            direction_tip=loc.direction_tip,
            elevation_tip=loc.elevation_tip,
            direction_disclaimer=loc.direction_disclaimer,
            # Raw acoustic features
            rms_energy=loc.rms_energy,
            spectral_rolloff=loc.spectral_rolloff,
            # Meta
            processing_ms=t_ms,
            agents_status=agents_status,
            snr_db=sig.snr_db,
        )


# ─── Pipeline Orchestrator ────────────────────────────────────────────────────

class WAVISPipeline:
    """
    Orchestrates all 5 agents.
    Usage:
        pipeline = WAVISPipeline()
        result   = pipeline.run("path/to/audio.wav")
        print(result.common_name, result.dist_range, result.horizontal)
    """

    def __init__(self,
                 model_path:         str = 'models/wavis_v2_best.pth',
                 class_map_path:     str = 'data/class_map.json',
                 display_names_path: str = 'data/display_names.json',
                 device:             str = None):

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.agent1 = SoundAcquisitionAgent()
        self.agent2 = SoundSeparationAgent()
        self.agent3 = ClassificationAgent(
            model_path, class_map_path, display_names_path, device
        )
        self.agent4 = LocalizationAgent()
        self.agent5 = DecisionFusionAgent()
        self.model_loaded = self.agent3.model_loaded

    def run(self, audio_input) -> FinalOutput:
        t0     = time.time()
        status = {}

        sig = self.agent1.process(audio_input)
        status['acquisition'] = '✅' if sig.is_valid else '❌'
        if not sig.is_valid:
            raise ValueError(sig.error)

        proc = self.agent2.process(sig)
        status['separation'] = '✅'

        cls = self.agent3.process(proc)
        status['classification'] = '✅'

        loc = self.agent4.process(proc)
        status['localization'] = '✅'

        result = self.agent5.process(cls, loc, proc, status, (time.time() - t0) * 1000)
        status['fusion'] = '✅'
        result.agents_status = status

        return result