import os
import sys
import time
import warnings
import math
import random
import argparse
import json
import shutil
from collections import defaultdict
from contextlib import contextmanager, nullcontext

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from acoustic_features import AcousticFeatureExtractor
from augmentation import SeldAugmentor
from baseline_paths import FRAMES_BASE, LABELS_BASE, MIC_BASE, UPLAM_CHECKPOINT
from evaluate import DEFAULT_ETH, DEFAULT_SIGMA_DEG, evaluate_prediction_dir
from model import (
    get_sequence_infos,
    EnergySegDataset,
    collate_fn,
    worker_init_fn,
    EnergyInstanceModel,
)
from run_inference import DEFAULT_ENERGY_TOP_K, DEFAULT_MAX_JSON_MB, run_inference_for_sequences
from speech_prior import SpeechPriorExtractor, get_speech_prior_embedding_dim


def env_int(name: str, default=None):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def env_float(name: str, default=None):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def env_bool(name: str, default=False):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


CLASS_NAME_TO_ID = {
    "female speech": 0,
    "male speech": 1,
    "clapping": 2,
    "telephone": 3,
    "laughter": 4,
    "domestic sounds": 5,
    "walk / footsteps": 6,
    "walk/footsteps": 6,
    "door open/close": 7,
    "music": 8,
    "musical instrument": 9,
    "water tap": 10,
    "water tap/shower": 10,
    "bell": 11,
    "knock": 12,
}


def env_class_set(name: str) -> set[int] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    classes: set[int] = set()
    for item in raw.split(","):
        key = item.strip()
        if not key:
            continue
        lowered = key.lower()
        if lowered in CLASS_NAME_TO_ID:
            classes.add(CLASS_NAME_TO_ID[lowered])
            continue
        cid = int(key)
        if not 0 <= cid <= 12:
            raise ValueError(f"{name} class id must be in [0, 12], got {cid}")
        classes.add(cid)
    return classes


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def sync_processes():
    if is_distributed():
        dist.barrier()


def init_distributed_mode():
    world_size = env_int("WORLD_SIZE", 1) or 1
    if world_size <= 1:
        return False
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requested but CUDA is unavailable.")

    local_rank = env_int("LOCAL_RANK", 0) or 0
    torch.cuda.set_device(local_rank)

    global DEVICE, _gpu_note
    DEVICE = torch.device(f"cuda:{local_rank}")
    _gpu_note = f"  [{torch.cuda.get_device_name(local_rank)} | rank {env_int('RANK', 0)}/{world_size}]"

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return True


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def sync_metric_sums(metric_dict: dict, count: int, device: torch.device):
    keys = sorted(metric_dict.keys())
    payload = [float(metric_dict[k]) for k in keys] + [float(count)]
    tensor = torch.tensor(payload, dtype=torch.float64, device=device)
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    reduced = {k: float(tensor[i].item()) for i, k in enumerate(keys)}
    return reduced, int(round(float(tensor[-1].item())))


def filter_sequence_infos(sequence_infos: list, prefix: str) -> list:
    """Optional sequence-level filtering for smoke runs."""
    regex = os.environ.get(f"DCASE2026_{prefix}_SEQ_REGEX", "").strip()
    limit = env_int(f"DCASE2026_{prefix}_MAX_SEQS", None)
    seed = env_int(f"DCASE2026_{prefix}_SEQ_SEED", 20260601) or 20260601

    filtered = list(sequence_infos)
    if regex:
        import re
        pat = re.compile(regex)
        filtered = [item for item in filtered if pat.search(item[2])]
    if limit is not None and limit > 0 and len(filtered) > limit:
        rng = random.Random(seed)
        filtered = rng.sample(filtered, limit)
        filtered.sort(key=lambda item: item[2])
    return filtered


class StageTimer:
    """Small helper for optional per-step timing without changing training logic."""

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)
        self._starts = {}
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)

    def start(self, key: str):
        if self.enabled:
            self._starts[key] = time.perf_counter()

    def stop(self, key: str):
        if self.enabled and key in self._starts:
            self.totals[key] += time.perf_counter() - self._starts.pop(key)
            self.counts[key] += 1

    def summary(self) -> dict:
        if not self.enabled:
            return {}
        out = {}
        for key in sorted(self.totals):
            count = max(self.counts.get(key, 0), 1)
            out[key] = {
                "total_sec": self.totals[key],
                "avg_sec": self.totals[key] / count,
                "count": count,
            }
        return out

# ── 0. MLOps UTILITIES ────────────────────────────────────────────────────────
class Logger(object):
    """Routes stdout/stderr to both the console and a specified log file."""
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def set_seed(seed=42):
    """Ensures deterministic runs for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── 1. AUTO-DETECT HPC RESOURCES ─────────────────────────────────────────────
def detect_resources():
    import psutil

    total_ram_gb = psutil.virtual_memory().total / 1e9
    cpu_count    = os.cpu_count() or 1

    gpu_vram_gb = 0.0
    if torch.cuda.is_available():
        props       = torch.cuda.get_device_properties(0)
        gpu_vram_gb = props.total_memory / 1e9

    num_workers  = min(max(cpu_count - 1, 0), 8)
    cache_per_ds = min(500, max(100, int(total_ram_gb * 5)))

    if   gpu_vram_gb >= 75: batch_size = 256
    elif gpu_vram_gb >= 40: batch_size = 128
    elif gpu_vram_gb >= 24: batch_size = 32
    elif gpu_vram_gb >= 16: batch_size = 16
    elif gpu_vram_gb >=  8: batch_size = 8
    else:                   batch_size = 4

    pin = env_bool("DCASE2026_PIN_MEMORY", torch.cuda.is_available())

    batch_size = env_int("DCASE2026_BATCH_SIZE", batch_size)
    num_workers = env_int("DCASE2026_NUM_WORKERS", num_workers)
    cache_per_ds = env_int("DCASE2026_CACHE_MAX", cache_per_ds)

    print(f"\n[TUNING] BATCH_SIZE  = {batch_size}")
    print(f"[TUNING] NUM_WORKERS = {num_workers}")
    print(f"[TUNING] CACHE_MAX   = {cache_per_ds} frames per dataset")
    print(f"[TUNING] RAM         = {total_ram_gb:.1f} GB")
    print(f"[TUNING] pin_memory  = {pin}\n")

    return batch_size, num_workers, cache_per_ds, pin


IMG_W, IMG_H     = 360, 180
N_CHANNELS       = 12
N_ACOUSTIC       = 9
NUM_CLASSES      = 14
NUM_EPOCHS       = env_int("DCASE2026_NUM_EPOCHS", 200)
PATIENCE         = env_int("DCASE2026_PATIENCE", 6) or 6

TRAIN_FRAMES_PER_EPOCH = env_int("DCASE2026_TRAIN_FRAMES_PER_EPOCH", None)
VAL_FRAMES_PER_EPOCH   = env_int("DCASE2026_VAL_FRAMES_PER_EPOCH", None)
FIXED_TRAIN_SUBSET     = env_bool("DCASE2026_FIXED_TRAIN_SUBSET", False)
TRAIN_SUBSET_SEED      = env_int("DCASE2026_TRAIN_SUBSET_SEED", 20260426) or 20260426
FIXED_VAL_SUBSET       = env_bool("DCASE2026_FIXED_VAL_SUBSET", True)
VAL_SUBSET_SEED        = env_int("DCASE2026_VAL_SUBSET_SEED", 20260426) or 20260426

LR               = 3e-4
BB_STEM_LR_MULT  = env_float("DCASE2026_BB_STEM_LR_MULT", 0.10)
BB_L3_LR_MULT    = env_float("DCASE2026_BB_L3_LR_MULT", 0.005)
BB_L4_LR_MULT    = env_float("DCASE2026_BB_L4_LR_MULT", 0.01)
FPN_LR_MULT      = env_float("DCASE2026_FPN_LR_MULT", 0.05)
RPN_LR_MULT      = env_float("DCASE2026_RPN_LR_MULT", 0.02)
BOX_HEAD_LR_MULT = env_float("DCASE2026_BOX_HEAD_LR_MULT", 0.02)
BOX_PRED_LR_MULT = env_float("DCASE2026_BOX_PRED_LR_MULT", 0.50)
ROI_FUSION_LR_MULT = env_float("DCASE2026_ROI_FUSION_LR_MULT", 0.50)
RESNET_CONFORMER_LR_MULT = env_float("DCASE2026_RESNET_CONFORMER_LR_MULT", 0.50)
SPEECH_PRIOR_LR_MULT = env_float("DCASE2026_SPEECH_PRIOR_LR_MULT", 1.00)
CUSTOM_LR_MULT   = env_float("DCASE2026_CUSTOM_LR_MULT", 1.00)
FREEZE_RPN       = env_bool("DCASE2026_FREEZE_RPN", False)
SCORE_THR        = 0.05
ENERGY_ANNOT_W   = env_float("DCASE2026_ENERGY_ANNOT_W", 5.0)
FULLMAP_ANNOT_W  = env_float("DCASE2026_FULLMAP_ANNOT_W", 10.0)
ENERGY_MASK_LOSS_W = env_float("DCASE2026_ENERGY_MASK_LOSS_WEIGHT", 1.0)
FULLMAP_LOSS_W   = env_float("DCASE2026_FULLMAP_LOSS_WEIGHT", 1.0)
DISTANCE_LOSS_W  = env_float("DCASE2026_DISTANCE_LOSS_WEIGHT", 1.0)
ENERGY_MASK_IOU_W = env_float("DCASE2026_ENERGY_MASK_IOU_W", 0.25)
GT_BOX_JITTER_FRAC = env_float("DCASE2026_GT_BOX_JITTER_FRAC", 0.10)
DIST_NORM        = 500.0
ENERGY_EXPORT_THR= 0.10
INFERENCE_HZ     = 10
INFERENCE_SEC    = 5

# ── Backbone progressive-unfreezing schedule ──────────────────────────────────
UNFREEZE_LAYER4_EPOCH = env_int("DCASE2026_UNFREEZE_LAYER4_EPOCH", 15) or 15
UNFREEZE_LAYER3_EPOCH = env_int("DCASE2026_UNFREEZE_LAYER3_EPOCH", 28) or 28

PHASE_GRACE = env_int("DCASE2026_PHASE_GRACE", 4) or 4


# ── Device Setup ──────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE    = torch.device("cuda")
    _gpu_note = f"  [{torch.cuda.get_device_name(0)}]"
    torch.backends.cudnn.benchmark = True
    _use_tf32 = env_bool("DCASE2026_TF32", True)
    torch.backends.cuda.matmul.allow_tf32 = _use_tf32
    torch.backends.cudnn.allow_tf32 = _use_tf32
    if _use_tf32:
        torch.set_float32_matmul_precision("high")
else:
    DEVICE        = torch.device("cpu")
    _mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    _gpu_note     = "  [Apple Silicon — CPU only]" if _mps_available else ""
    _use_tf32 = False

print(f"[INFO] Device: {DEVICE}{_gpu_note}")
if torch.cuda.is_available():
    print(f"[INFO] TF32: {'ENABLED' if _use_tf32 else 'DISABLED'}")


# ── 3. EVAL BEHAVIOR CONTEXT MANAGER (BN + DROPOUT) ──────────────────────────
def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, (nn.DataParallel, DDP)) else model


def reduce_loss_value(v):
    if isinstance(v, torch.Tensor) and v.ndim > 0:
        return v.mean()
    return v


def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj


class DetectionDataParallel(nn.DataParallel):
    """Scatter detection-style batches by sample instead of splitting image channels."""

    def scatter(self, inputs, kwargs, device_ids):
        if not inputs:
            return super().scatter(inputs, kwargs, device_ids)

        images = inputs[0]
        targets = inputs[1] if len(inputs) > 1 else None
        speech_priors = inputs[2] if len(inputs) > 2 else None
        if not isinstance(images, (list, tuple)) or len(images) == 0:
            return super().scatter(inputs, kwargs, device_ids)

        n_used = min(len(device_ids), len(images))
        device_ids = list(device_ids[:n_used])

        image_chunks = [[] for _ in range(n_used)]
        target_chunks = [[] for _ in range(n_used)] if targets is not None else None
        prior_chunks = [[] for _ in range(n_used)] if speech_priors is not None else None

        for idx, image in enumerate(images):
            bucket = idx % n_used
            image_chunks[bucket].append(image)
            if targets is not None:
                target_chunks[bucket].append(targets[idx])
            if speech_priors is not None:
                prior_chunks[bucket].append(speech_priors[idx])

        scattered_inputs = []
        for device_idx, chunk_images in zip(device_ids, image_chunks):
            device = torch.device(f"cuda:{device_idx}")
            args = [move_to_device(chunk_images, device)]
            if targets is not None:
                args.append(move_to_device(target_chunks[device_ids.index(device_idx)], device))
            if speech_priors is not None:
                prior_chunk = prior_chunks[device_ids.index(device_idx)]
                if isinstance(speech_priors, torch.Tensor):
                    prior_chunk = torch.stack(prior_chunk, dim=0)
                args.append(move_to_device(prior_chunk, device))
            scattered_inputs.append(tuple(args))

        scattered_kwargs = tuple({} for _ in range(len(scattered_inputs)))
        return tuple(scattered_inputs), scattered_kwargs


@contextmanager
def eval_behavior_for_loss(model):
    """
    Forces BatchNorm to use running stats and disables Dropout locally,
    without changing the global model.train() state, preserving the loss dict
    output from Torchvision's Faster R-CNN forward pass.
    """
    target_types = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.Dropout, nn.Dropout2d, nn.Dropout3d,
    )
    modules     = [m for m in model.modules() if isinstance(m, target_types)]
    orig_states = {m: m.training for m in modules}
    for m in modules:
        m.eval()
    try:
        yield
    finally:
        for m, state in orig_states.items():
            m.train(mode=state)


# ── 4. BACKBONE PROGRESSIVE UNFREEZING ───────────────────────────────────────
def apply_backbone_freeze(model: nn.Module, epoch: int):
    """
    Progressively unfreeze ResNet50 backbone layers as training matures.
    """
    model = unwrap_model(model)
    body = model.det.backbone.body

    # Keep the RGB stem trainable; the acoustic branch is separate and trained
    # through the custom parameter groups below.
    for module in [body.layer1, body.layer2]:
        for p in module.parameters():
            p.requires_grad = False

    layer4_active = epoch >= UNFREEZE_LAYER4_EPOCH
    for p in body.layer4.parameters():
        p.requires_grad = layer4_active

    layer3_active = epoch >= UNFREEZE_LAYER3_EPOCH
    for p in body.layer3.parameters():
        p.requires_grad = layer3_active

    for p in model.det.backbone.fpn.parameters():
        p.requires_grad = True

    phase = "heads+FPN only"
    if layer3_active:
        phase = "layer3+4 + heads+FPN"
    elif layer4_active:
        phase = "layer4 + heads+FPN"

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    return phase, n_trainable, n_total


# ── 5. OPTIMIZER HELPERS ──────────────────────────────────────────────────────
def _get_pg(optimizer, name):
    """Helper to cleanly extract optimizer parameter groups by name."""
    return next(g for g in optimizer.param_groups if g.get("name") == name)

def _get_pg_lr(optimizer, name, default: float = 0.0):
    for group in optimizer.param_groups:
        if group.get("name") == name:
            return float(group.get("lr", default))
    return default

def freeze_backbone_bn(model: nn.Module):
    """
    After model.train() resets all modules to training mode, re-freeze the
    BatchNorm running_mean/running_var for backbone layers that are never
    unfrozen (layer1, layer2). 
    
    conv1 and bn1 are explicitly EXCLUDED from this list so the visual stem can
    continue adapting to the dataset while the deeper RGB backbone stays mostly
    frozen early in training.
    """
    model = unwrap_model(model)
    body = model.det.backbone.body
    for module in [body.layer1, body.layer2]:
        for m in module.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()


def freeze_bn_dropout_for_loss(model: nn.Module):
    """
    Keep stochastic normalization/dropout layers in eval behavior while leaving
    the model in train mode, so detection-style forwards still return losses.
    """
    target_types = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.Dropout, nn.Dropout2d, nn.Dropout3d,
    )
    for module in model.modules():
        if isinstance(module, target_types):
            module.eval()


def set_localization_train_scope(model: nn.Module, scope: str) -> str:
    """
    Narrow trainable parameters for localization warmup experiments.
    Optimizer groups are already built; toggling requires_grad here controls
    which tensors receive gradients without touching checkpoint structure.
    """
    model = unwrap_model(model)
    scope = str(scope or "all").strip().lower()
    if scope in {"", "all", "default"}:
        return "all"

    for param in model.parameters():
        param.requires_grad = False

    modules = []
    if scope in {"energy_heads", "heads"}:
        modules = [model.energy_head, model.energy_decoder]
    elif scope in {"roi_energy_heads", "roi_heads", "class_energy_heads"}:
        roi_heads = model.det.roi_heads
        modules = [
            roi_heads.box_head,
            roi_heads.box_predictor,
            model.energy_head,
            model.energy_decoder,
        ]
        if getattr(roi_heads, "roi_fusion_head", None) is not None:
            modules.append(roi_heads.roi_fusion_head)
        if getattr(model, "resnet_conformer", None) is not None:
            modules.append(model.resnet_conformer)
    elif scope in {"energy_path", "path"}:
        modules = [
            model.energy_head,
            model.energy_decoder,
            model.audio_backbone,
            model.fpn_fusion,
            model.det.backbone.fpn,
        ]
        if getattr(model, "resnet_conformer", None) is not None:
            modules.append(model.resnet_conformer)
    else:
        raise ValueError(
            f"Unsupported DCASE2026_TRAIN_SCOPE={scope!r}; "
            "expected all, energy_heads, roi_energy_heads, or energy_path."
        )

    for module in modules:
        for param in module.parameters():
            param.requires_grad = True
    return scope


def compute_foreground_class_counts(dataset, num_classes: int) -> np.ndarray:
    n_fg = max(1, num_classes - 1)
    counts = np.zeros(n_fg, dtype=np.float64)
    for cid, sample_indices in dataset.class_to_sample_indices.items():
        if 0 <= cid < n_fg:
            counts[cid] = float(len(sample_indices))
    return counts


def compute_foreground_class_priors(dataset, num_classes: int) -> np.ndarray:
    counts = compute_foreground_class_counts(dataset, num_classes)
    if counts.sum() <= 0:
        counts.fill(1.0)
    return counts / counts.sum()


def compute_foreground_class_weights(
    class_counts: np.ndarray,
    mode: str = "inverse_sqrt",
    beta: float = 0.9999,
    max_weight: float = 4.0,
) -> np.ndarray:
    counts = np.asarray(class_counts, dtype=np.float64).copy()
    counts[counts <= 0] = 1.0
    mode = str(mode or "none").strip().lower()

    if mode in {"none", "uniform", "off"}:
        weights = np.ones_like(counts)
    elif mode == "inverse":
        weights = 1.0 / counts
    elif mode in {"inverse_sqrt", "inv_sqrt", "sqrt_inv"}:
        weights = 1.0 / np.sqrt(counts)
    elif mode in {"effective_num", "effective"}:
        beta = float(np.clip(beta, 0.0, 0.999999))
        effective_num = 1.0 - np.power(beta, counts)
        effective_num = np.clip(effective_num, 1e-9, None)
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(
            f"Unsupported DCASE2026_ROI_CLASS_WEIGHT_MODE={mode!r}; "
            "expected one of none, inverse, inverse_sqrt, effective_num."
        )

    weights = weights / np.mean(weights)
    if max_weight is not None and max_weight > 0:
        weights = np.clip(weights, None, float(max_weight))
        weights = weights / np.mean(weights)
    return weights.astype(np.float32)


def initialize_box_predictor_bias(model: nn.Module, class_priors: np.ndarray, background_prior: float = 0.5):
    model = unwrap_model(model)
    cls_score = model.det.roi_heads.box_predictor.cls_score
    if cls_score.bias is None:
        return None

    priors = np.asarray(class_priors, dtype=np.float64)
    n_fg = cls_score.out_features - 1
    if priors.size != n_fg:
        resized = np.full(n_fg, 1.0 / max(n_fg, 1), dtype=np.float64)
        n_copy = min(priors.size, n_fg)
        if n_copy > 0:
            resized[:n_copy] = priors[:n_copy]
        priors = resized

    priors = np.clip(priors, 1e-6, None)
    priors = priors / priors.sum()
    bg = float(np.clip(background_prior, 0.05, 0.95))

    probs = np.full(cls_score.out_features, 1e-6, dtype=np.float64)
    probs[0] = bg
    probs[1:] = (1.0 - bg) * priors
    probs = probs / probs.sum()

    bias = torch.as_tensor(
        np.log(np.clip(probs, 1e-6, None)),
        dtype=cls_score.bias.dtype,
        device=cls_score.bias.device,
    )
    with torch.no_grad():
        cls_score.bias.copy_(bias)
    return probs


# ── 6. PLOTTING UTILITY ───────────────────────────────────────────────────────
def plot_losses(history: dict, save_path: str = "training_loss.png"):
    base_keys = [k for k in history.keys() if k.startswith("loss_")]
    all_keys  = ["total"] + sorted(base_keys)
    epochs    = range(1, len(history.get("total", [])) + 1)

    n_plots = len(all_keys)
    cols    = 4
    rows    = math.ceil(n_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
    axes      = np.array(axes).flatten()
    fig.suptitle("Training & Validation Loss Curves", fontsize=15, y=1.02)

    def smooth(v, k=7):
        if len(v) < k:
            return v, 0
        k = max(3, min(k, (len(v) // 3) * 2 + 1) | 1)
        return np.convolve(v, np.ones(k) / k, mode="valid"), k // 2

    for i, key in enumerate(all_keys):
        ax    = axes[i]
        vals  = history.get(key, [])
        title = "Total Loss" if key == "total" else key.replace("loss_", "").replace("_", " ").capitalize()

        ax.plot(list(epochs), vals, lw=1, color="steelblue", alpha=0.4, label="train (raw)")
        if len(vals) >= 5:
            s, pad = smooth(vals)
            ax.plot(list(range(pad + 1, len(vals) - pad + 1)), s,
                    lw=2, color="orangered", label="train (smoothed)")

        val_key = "val_total" if key == "total" else f"val_{key}"
        if val_key in history and len(history[val_key]) == len(epochs):
            ax.plot(list(epochs), history[val_key], lw=2, color="green", label="validation")

        ax.legend(fontsize=7)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Loss curves → {save_path}")


@contextmanager
def temporary_eval_mode(module: nn.Module):
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)


def is_finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def broadcast_scalar_payload(values, device: torch.device):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if is_distributed():
        dist.broadcast(tensor, src=0)
    return [float(x) for x in tensor.cpu().tolist()]


def run_devtest_eval(
    model: nn.Module,
    sequence_infos: list,
    exp_dir: str,
    epoch: int,
    cfg: dict,
    acoustic_extractor=None,
    speech_prior_extractor=None,
):
    epoch_dir = os.path.join(exp_dir, "devtest_epoch_eval", f"epoch_{epoch:03d}")
    pred_dir = os.path.join(epoch_dir, "inference_outputs")
    report_path = os.path.join(epoch_dir, "eval_summary.txt")
    metrics_path = os.path.join(epoch_dir, "metrics.json")
    os.makedirs(epoch_dir, exist_ok=True)

    eval_sequence_infos = list(sequence_infos)
    num_seqs = int(cfg.get("num_seqs") or 0)
    if 0 < num_seqs < len(eval_sequence_infos):
        rng = random.Random(int(cfg.get("num_seqs_seed") or 0))
        eval_sequence_infos = rng.sample(eval_sequence_infos, num_seqs)
        print(
            f"[DevTest] Fixed sequence subset: {num_seqs}/{len(sequence_infos)} "
            f"(seed={cfg.get('num_seqs_seed')})"
        )

    if cfg["gpu_acoustic"] and acoustic_extractor is None:
        acoustic_extractor = AcousticFeatureExtractor(
            uplam_checkpoint=UPLAM_CHECKPOINT,
            device=DEVICE,
            num_bands=N_ACOUSTIC,
            fast_precision=cfg["fast_uplam"],
        )

    inference_info = run_inference_for_sequences(
        model=model,
        sequence_infos=eval_sequence_infos,
        out_dir=pred_dir,
        score_thr=cfg["score_thr"],
        nms_iou_thr=cfg["nms_iou_thr"],
        max_dets_per_class=cfg["max_dets_per_class"],
        min_hits=cfg["min_hits"],
        track_mean_score_thr=cfg["track_mean_score_thr"],
        track_peak_score_thr=cfg["track_peak_score_thr"],
        track_mean_score_thr_by_class=cfg["track_mean_score_thr_by_class"],
        track_peak_score_thr_by_class=cfg["track_peak_score_thr_by_class"],
        max_total_dets_per_frame=cfg["max_total_dets_per_frame"],
        max_total_dets_per_frame_by_class=cfg["max_total_dets_per_frame_by_class"],
        energy_top_k=cfg["energy_top_k"],
        energy_export_thr=cfg["energy_export_thr"],
        energy_source=cfg["energy_source"],
        energy_export_mode=cfg["energy_export_mode"],
        energy_region_rel_thr=cfg["energy_region_rel_thr"],
        energy_region_pad=cfg["energy_region_pad"],
        max_json_mb=cfg["max_json_mb"],
        iou_thr=cfg["iou_thr"],
        max_age=cfg["max_age"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        coast_decay=cfg["coast_decay"],
        gpu_acoustic=cfg["gpu_acoustic"],
        acoustic_extractor=acoustic_extractor,
        speech_prior_extractor=speech_prior_extractor,
    )
    metrics = evaluate_prediction_dir(
        pred_dir=pred_dir,
        gt_root=LABELS_BASE,
        split=cfg["split"],
        sigma=cfg["sigma"],
        energy_thr=cfg["energy_thr"],
        save_path=report_path,
    )
    payload = {
        "epoch": epoch,
        "epoch_dir": epoch_dir,
        "pred_dir": pred_dir,
        "config": cfg,
        "num_eval_sequences": len(eval_sequence_infos),
        "eval_sequences": [seq_name for _, _, seq_name in eval_sequence_infos],
        "metrics": metrics,
        "inference": inference_info,
    }
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[DevTest] Metrics JSON → {metrics_path}")
    return payload


# ── 7. TRAINING LOOP ──────────────────────────────────────────────────────────
def train(train_infos: list, val_infos: list, exp_dir: str):

    best_model_path = os.path.join(exp_dir, "energy_seg_best.pth")
    save_every_epoch = env_bool("DCASE2026_SAVE_EVERY_EPOCH", True)
    epoch_ckpt_dir = os.environ.get(
        "DCASE2026_EPOCH_CKPT_DIR",
        os.path.join(exp_dir, "epoch_checkpoints"),
    )

    batch_size, num_workers, cache_max, pin_memory = detect_resources()
    distributed = is_distributed()
    world_size = get_world_size()
    rank = get_rank()
    profile_steps = env_int("DCASE2026_PROFILE_STEPS", 0) or 0
    channels_last = env_bool("DCASE2026_CHANNELS_LAST", False)
    train_scope_cfg = os.environ.get("DCASE2026_TRAIN_SCOPE", "all")
    train_eval_behavior = env_bool("DCASE2026_TRAIN_EVAL_BEHAVIOR", False)
    if distributed and num_workers > 0:
        original_workers = num_workers
        num_workers = max(1, num_workers // world_size)
        if is_main_process():
            print(
                f"[TRAIN] DDP worker rebalance: {original_workers} total hint -> "
                f"{num_workers} workers per rank ({world_size} ranks)"
            )
    print(f"[TRAIN] NUM_CLASSES = {NUM_CLASSES}")
    gpu_acoustic = env_bool("DCASE2026_GPU_ACOUSTIC", torch.cuda.is_available())
    gpu_rgb_prep = env_bool("DCASE2026_GPU_RGB_PREP", gpu_acoustic)
    fast_uplam = env_bool("DCASE2026_FAST_UPLAM", gpu_acoustic)
    use_speech_prior = env_bool("DCASE2026_USE_SPEECH_PRIOR", False)
    speech_prior_bundle = os.environ.get("DCASE2026_SPEECH_PRIOR_BUNDLE", "WAVLM_BASE_PLUS").strip()
    speech_prior_context_sec = env_float("DCASE2026_SPEECH_PRIOR_CONTEXT_SEC", 1.0)
    speech_prior_hidden_dim = env_int("DCASE2026_SPEECH_PRIOR_HIDDEN_DIM", 512) or 512
    speech_prior_dropout = env_float("DCASE2026_SPEECH_PRIOR_DROPOUT", 0.10)
    speech_prior_scale_init = env_float("DCASE2026_SPEECH_PRIOR_SCALE_INIT", -2.0)
    speech_prior_cache_size = env_int("DCASE2026_SPEECH_PRIOR_CACHE_SIZE", 512) or 512
    speech_prior_dim = (
        get_speech_prior_embedding_dim(speech_prior_bundle)
        if use_speech_prior else 0
    )
    if gpu_acoustic and not torch.cuda.is_available():
        print("[TRAIN] GPU acoustic extraction requested but CUDA is unavailable; falling back to CPU acoustic path.")
        gpu_acoustic = False
    if gpu_rgb_prep and not torch.cuda.is_available():
        print("[TRAIN] GPU RGB preprocessing requested but CUDA is unavailable; falling back to CPU RGB path.")
        gpu_rgb_prep = False
    if fast_uplam and not torch.cuda.is_available():
        fast_uplam = False
    if use_speech_prior and not speech_prior_bundle:
        raise ValueError("DCASE2026_SPEECH_PRIOR_BUNDLE is empty.")

    use_amp = torch.cuda.is_available()
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("[TRAIN] Mixed precision (AMP) ENABLED")

    acoustic_device = DEVICE if gpu_acoustic else torch.device("cpu")
    acoustic_extractor = AcousticFeatureExtractor(
        uplam_checkpoint = UPLAM_CHECKPOINT,
        device           = acoustic_device,
        num_bands        = N_ACOUSTIC,
        fast_precision   = fast_uplam,
    )
    acoustic_extractor.clear_cache()
    print(
        f"[TRAIN] Acoustic extraction: {'GPU-batched in main process' if gpu_acoustic else 'dataset-side CPU path'} "
        f"(device={acoustic_device})"
    )
    print(f"[TRAIN] UpLAM precision: {'fast float32/complex64' if fast_uplam else 'full float64/complex128'}")
    speech_prior_extractor = None
    if use_speech_prior:
        speech_prior_device_name = os.environ.get("DCASE2026_SPEECH_PRIOR_DEVICE", "cuda").strip().lower()
        speech_prior_device = DEVICE if speech_prior_device_name != "cpu" and torch.cuda.is_available() else torch.device("cpu")
        if distributed and not is_main_process():
            sync_processes()
        speech_prior_extractor = SpeechPriorExtractor(
            bundle_name=speech_prior_bundle,
            device=speech_prior_device,
            context_sec=speech_prior_context_sec,
            cache_size=speech_prior_cache_size,
            use_amp=use_amp,
        )
        if distributed and is_main_process():
            sync_processes()
        print(
            f"[TRAIN] Speech prior: {speech_prior_bundle} | dim={speech_prior_dim} | "
            f"context={speech_prior_context_sec:.2f}s | cache={speech_prior_cache_size}"
        )
    else:
        print("[TRAIN] Speech prior: DISABLED")
    print(
        f"[TRAIN] RGB frame preprocessing: "
        f"{'GPU resize/float conversion in main process' if gpu_rgb_prep else 'dataset-side CPU path'}"
    )
    print(f"[TRAIN] Memory format: {'channels_last' if channels_last else 'contiguous/NCHW'}")
    if profile_steps > 0:
        print(f"[TRAIN] Perf profiling enabled for first {profile_steps} train steps")

    # ── Augmentor (train only) ────────────────────────────────────────────────
    disable_augmentation = env_bool("DCASE2026_DISABLE_AUGMENTATION", False)
    train_augmentor = None
    if not disable_augmentation:
        train_augmentor = SeldAugmentor(
            img_w                 = IMG_W,
            img_h                 = IMG_H,
            n_acoustic            = N_ACOUSTIC,
            azimuth_rotate        = True,
            hflip_prob            = 0.5,
            max_bands_masked      = 4,
            intensity_scale_range = (0.6, 1.4),
            acoustic_noise_std    = 0.03,
            rgb_jitter_prob       = 0.8,
        )
        print(f"[TRAIN] Augmentation: azimuth_rotate=True | hflip_prob=0.5 | rgb_jitter=0.8 | "
              f"max_bands_masked=4 | intensity_scale=(0.6,1.4) | acoustic_noise_std=0.03")
    else:
        print("[TRAIN] Augmentation: DISABLED")

    # ── Datasets ──────────────────────────────────────────────────────────────
    annotation_target_mode = (
        os.environ.get("DCASE2026_ANNOTATION_TARGET_MODE", "sparse") or "sparse"
    ).strip().lower()
    teacher23_csv_dir = os.environ.get("DCASE2026_TEACHER23_CSV_DIR", "").strip()
    teacher23_pos_loss_w = env_float("DCASE2026_TEACHER23_POS_LOSS_W", 0.00)
    teacher23_sigma_px = env_float("DCASE2026_TEACHER23_SIGMA_PX", 4.00)
    teacher23_fallback_gt_peak = env_bool("DCASE2026_TEACHER23_FALLBACK_GT_PEAK", False)
    class_balanced_env = os.environ.get("DCASE2026_CLASS_BALANCED_SAMPLING")
    class_balanced_sampling = (
        None
        if class_balanced_env is None
        else env_bool("DCASE2026_CLASS_BALANCED_SAMPLING", False)
    )
    class_balanced_fraction = env_float("DCASE2026_BALANCED_SAMPLE_FRACTION", 0.50)
    class_balanced_classes = env_class_set("DCASE2026_BALANCED_CLASSES")
    train_dataset = EnergySegDataset(
        sequence_infos     = train_infos,
        frames_base        = FRAMES_BASE,
        mic_base           = MIC_BASE,
        acoustic_extractor = None if gpu_acoustic else acoustic_extractor,
        frames_per_epoch   = TRAIN_FRAMES_PER_EPOCH,
        img_w=IMG_W, img_h=IMG_H,
        dist_norm          = DIST_NORM,
        cache_max_size     = cache_max,
        augmentor          = None if gpu_acoustic else train_augmentor,
        gpu_rgb_pipeline   = gpu_rgb_prep,
        fixed_subset       = FIXED_TRAIN_SUBSET,
        subset_seed        = TRAIN_SUBSET_SEED,
        return_meta        = gpu_acoustic or use_speech_prior,
        annotation_target_mode = annotation_target_mode,
        teacher23_csv_dir = teacher23_csv_dir,
        teacher23_sigma_px = teacher23_sigma_px,
        teacher23_fallback_gt_peak = teacher23_fallback_gt_peak,
        class_balanced_sampling = class_balanced_sampling,
        class_balanced_fraction = class_balanced_fraction,
        class_balanced_classes = class_balanced_classes,
    )
    
    val_dataset = EnergySegDataset(
        sequence_infos     = val_infos,
        frames_base        = FRAMES_BASE,
        mic_base           = MIC_BASE,
        acoustic_extractor = None if gpu_acoustic else acoustic_extractor,
        frames_per_epoch   = VAL_FRAMES_PER_EPOCH,
        img_w=IMG_W, img_h=IMG_H,
        dist_norm          = DIST_NORM,
        cache_max_size     = max(1, cache_max // 2),
        augmentor          = None,
        gpu_rgb_pipeline   = gpu_rgb_prep,
        fixed_subset       = FIXED_VAL_SUBSET,
        subset_seed        = VAL_SUBSET_SEED,
        return_meta        = gpu_acoustic or use_speech_prior,
        annotation_target_mode = annotation_target_mode,
        teacher23_csv_dir = teacher23_csv_dir,
        teacher23_sigma_px = teacher23_sigma_px,
        teacher23_fallback_gt_peak = teacher23_fallback_gt_peak,
        class_balanced_sampling = False,
    )

    train_loader_kwargs = dict(
        collate_fn         = collate_fn,
        num_workers        = num_workers,
        pin_memory         = pin_memory,
        prefetch_factor    = 4 if num_workers > 0 else None,
        worker_init_fn     = worker_init_fn if num_workers > 0 else None,
        persistent_workers = num_workers > 0,
    )
    val_num_workers = min(2, num_workers)
    val_loader_kwargs = dict(
        collate_fn         = collate_fn,
        num_workers        = val_num_workers,
        pin_memory         = pin_memory,
        prefetch_factor    = 2 if val_num_workers > 0 else None,
        worker_init_fn     = worker_init_fn if val_num_workers > 0 else None,
        persistent_workers = False,
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed and len(val_dataset) > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **train_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        **val_loader_kwargs,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    det_fusion_mode = (os.environ.get("DCASE2026_DET_FUSION_MODE", "hybrid") or "hybrid").strip().lower()
    fusion_scale_init = env_float("DCASE2026_FUSION_SCALE_INIT", -4.0)
    cls_bg_prior = env_float("DCASE2026_CLS_BG_PRIOR", 0.50)
    roi_class_weight_mode = (os.environ.get("DCASE2026_ROI_CLASS_WEIGHT_MODE", "none") or "none").strip().lower()
    roi_class_beta = env_float("DCASE2026_ROI_CLASS_BETA", 0.9999)
    roi_class_max_weight = env_float("DCASE2026_ROI_CLASS_MAX_WEIGHT", 1.0)
    roi_bg_loss_weight = env_float("DCASE2026_ROI_BG_LOSS_WEIGHT", 1.00)
    roi_logit_adj_tau = env_float("DCASE2026_ROI_LOGIT_ADJ_TAU", 0.00)
    roi_infer_logit_adj_tau = env_float("DCASE2026_ROI_INFER_LOGIT_ADJ_TAU", 0.00)
    roi_label_smoothing = env_float("DCASE2026_ROI_LABEL_SMOOTHING", 0.00)
    roi_cls_loss_weight = env_float("DCASE2026_ROI_CLS_LOSS_WEIGHT", 1.00)
    roi_gt_aux_loss_weight = env_float("DCASE2026_ROI_GT_AUX_LOSS_WEIGHT", 0.00)
    energy_mask_loss_w = env_float("DCASE2026_ENERGY_MASK_LOSS_WEIGHT", ENERGY_MASK_LOSS_W)
    fullmap_loss_w = env_float("DCASE2026_FULLMAP_LOSS_WEIGHT", FULLMAP_LOSS_W)
    distance_loss_w = env_float("DCASE2026_DISTANCE_LOSS_WEIGHT", DISTANCE_LOSS_W)
    energy_mask_iou_w = env_float("DCASE2026_ENERGY_MASK_IOU_W", ENERGY_MASK_IOU_W)
    energy_mask_bce_w = env_float("DCASE2026_ENERGY_MASK_BCE_W", 0.00)
    energy_mask_bce_pos_w = env_float("DCASE2026_ENERGY_MASK_BCE_POS_W", 4.00)
    energy_mask_target_mode = (
        os.environ.get("DCASE2026_ENERGY_MASK_TARGET_MODE", "support") or "support"
    ).strip().lower()
    energy_mask_core_thr = env_float("DCASE2026_ENERGY_MASK_CORE_THR", 0.50)
    energy_mask_focal_w = env_float("DCASE2026_ENERGY_MASK_FOCAL_W", 0.00)
    energy_mask_focal_gamma = env_float("DCASE2026_ENERGY_MASK_FOCAL_GAMMA", 2.00)
    energy_mask_tversky_w = env_float("DCASE2026_ENERGY_MASK_TVERSKY_W", 0.00)
    energy_mask_tversky_alpha = env_float("DCASE2026_ENERGY_MASK_TVERSKY_ALPHA", 0.30)
    energy_mask_tversky_beta = env_float("DCASE2026_ENERGY_MASK_TVERSKY_BETA", 0.70)
    energy_mask_lovasz_w = env_float("DCASE2026_ENERGY_MASK_LOVASZ_W", 0.00)
    energy_mask_boundary_w = env_float("DCASE2026_ENERGY_MASK_BOUNDARY_W", 0.00)
    energy_mask_boundary_kernel = env_int("DCASE2026_ENERGY_MASK_BOUNDARY_KERNEL", 3) or 3
    energy_dist_w = env_float("DCASE2026_ENERGY_DIST_W", 0.00)
    energy_dist_gamma = env_float("DCASE2026_ENERGY_DIST_GAMMA", 1.00)
    fullmap_dist_w = env_float("DCASE2026_FULLMAP_DIST_W", 0.00)
    fullmap_dist_gamma = env_float("DCASE2026_FULLMAP_DIST_GAMMA", 1.00)
    fullmap_mask_iou_w = env_float("DCASE2026_FULLMAP_MASK_IOU_W", 0.00)
    fullmap_mask_bce_w = env_float("DCASE2026_FULLMAP_MASK_BCE_W", 0.00)
    fullmap_mask_bce_pos_w = env_float("DCASE2026_FULLMAP_MASK_BCE_POS_W", 8.00)
    fullmap_mask_focal_w = env_float("DCASE2026_FULLMAP_MASK_FOCAL_W", 0.00)
    fullmap_mask_focal_gamma = env_float("DCASE2026_FULLMAP_MASK_FOCAL_GAMMA", 2.00)
    fullmap_mask_tversky_w = env_float("DCASE2026_FULLMAP_MASK_TVERSKY_W", 0.00)
    fullmap_mask_tversky_alpha = env_float("DCASE2026_FULLMAP_MASK_TVERSKY_ALPHA", 0.30)
    fullmap_mask_tversky_beta = env_float("DCASE2026_FULLMAP_MASK_TVERSKY_BETA", 0.70)
    fullmap_mask_lovasz_w = env_float("DCASE2026_FULLMAP_MASK_LOVASZ_W", 0.00)
    fullmap_mask_boundary_w = env_float("DCASE2026_FULLMAP_MASK_BOUNDARY_W", 0.00)
    fullmap_mask_boundary_kernel = env_int("DCASE2026_FULLMAP_MASK_BOUNDARY_KERNEL", 5) or 5
    det_loss_weight = env_float("DCASE2026_DET_LOSS_WEIGHT", 1.00)
    gt_box_jitter_frac = env_float("DCASE2026_GT_BOX_JITTER_FRAC", GT_BOX_JITTER_FRAC)
    roi_infer_score_temperature = env_float("DCASE2026_ROI_INFER_SCORE_TEMPERATURE", 1.00)
    roi_infer_fg_bias = env_float("DCASE2026_ROI_INFER_FG_BIAS", 0.00)
    roi_infer_score_power = env_float("DCASE2026_ROI_INFER_SCORE_POWER", 1.00)
    use_roi_cmaf = env_bool("DCASE2026_USE_ROI_CMAF", True)
    roi_cmaf_scale_init = env_float("DCASE2026_ROI_CMAF_SCALE_INIT", -3.0)
    use_resnet_conformer = env_bool("DCASE2026_USE_RESNET_CONFORMER", False)
    resnet_conformer_layers = env_int("DCASE2026_RESNET_CONFORMER_LAYERS", 2) or 2
    resnet_conformer_ffn_dim = env_int("DCASE2026_RESNET_CONFORMER_FFN_DIM", 1024) or 1024
    resnet_conformer_kernel_size = env_int("DCASE2026_RESNET_CONFORMER_KERNEL_SIZE", 31) or 31
    resnet_conformer_dropout = env_float("DCASE2026_RESNET_CONFORMER_DROPOUT", 0.10)
    resnet_conformer_scale_init = env_float("DCASE2026_RESNET_CONFORMER_SCALE_INIT", -4.0)
    resnet_conformer_target = (
        os.environ.get("DCASE2026_RESNET_CONFORMER_TARGET", "roi") or "roi"
    ).strip().lower()

    fg_class_counts = compute_foreground_class_counts(train_dataset, NUM_CLASSES)
    fg_class_priors = compute_foreground_class_priors(train_dataset, NUM_CLASSES)
    fg_class_weights = compute_foreground_class_weights(
        fg_class_counts,
        mode=roi_class_weight_mode,
        beta=roi_class_beta,
        max_weight=roi_class_max_weight,
    )

    model = EnergyInstanceModel(
        num_classes     = NUM_CLASSES,
        n_channels      = N_CHANNELS,
        img_w=IMG_W, img_h=IMG_H,
        energy_annot_w  = ENERGY_ANNOT_W,
        fullmap_annot_w = FULLMAP_ANNOT_W,
        det_fusion_mode = det_fusion_mode,
        fusion_scale_init = fusion_scale_init,
        roi_fg_class_priors = fg_class_priors,
        roi_fg_class_weights = fg_class_weights,
        roi_bg_loss_weight = roi_bg_loss_weight,
        roi_logit_adj_tau = roi_logit_adj_tau,
        roi_infer_logit_adj_tau = roi_infer_logit_adj_tau,
        roi_label_smoothing = roi_label_smoothing,
        roi_cls_loss_weight = roi_cls_loss_weight,
        roi_gt_aux_loss_weight = roi_gt_aux_loss_weight,
        energy_mask_loss_weight = energy_mask_loss_w,
        fullmap_loss_weight = fullmap_loss_w,
        distance_loss_weight = distance_loss_w,
        energy_mask_iou_w = energy_mask_iou_w,
        energy_mask_bce_w = energy_mask_bce_w,
        energy_mask_bce_pos_w = energy_mask_bce_pos_w,
        energy_mask_target_mode = energy_mask_target_mode,
        energy_mask_core_thr = energy_mask_core_thr,
        energy_mask_focal_w = energy_mask_focal_w,
        energy_mask_focal_gamma = energy_mask_focal_gamma,
        energy_mask_tversky_w = energy_mask_tversky_w,
        energy_mask_tversky_alpha = energy_mask_tversky_alpha,
        energy_mask_tversky_beta = energy_mask_tversky_beta,
        energy_mask_lovasz_w = energy_mask_lovasz_w,
        energy_mask_boundary_w = energy_mask_boundary_w,
        energy_mask_boundary_kernel = energy_mask_boundary_kernel,
        energy_dist_w = energy_dist_w,
        energy_dist_gamma = energy_dist_gamma,
        fullmap_dist_w = fullmap_dist_w,
        fullmap_dist_gamma = fullmap_dist_gamma,
        fullmap_mask_iou_w = fullmap_mask_iou_w,
        fullmap_mask_bce_w = fullmap_mask_bce_w,
        fullmap_mask_bce_pos_w = fullmap_mask_bce_pos_w,
        fullmap_mask_focal_w = fullmap_mask_focal_w,
        fullmap_mask_focal_gamma = fullmap_mask_focal_gamma,
        fullmap_mask_tversky_w = fullmap_mask_tversky_w,
        fullmap_mask_tversky_alpha = fullmap_mask_tversky_alpha,
        fullmap_mask_tversky_beta = fullmap_mask_tversky_beta,
        fullmap_mask_lovasz_w = fullmap_mask_lovasz_w,
        fullmap_mask_boundary_w = fullmap_mask_boundary_w,
        fullmap_mask_boundary_kernel = fullmap_mask_boundary_kernel,
        teacher23_pos_loss_w = teacher23_pos_loss_w,
        det_loss_weight = det_loss_weight,
        gt_box_jitter_frac = gt_box_jitter_frac,
        roi_infer_score_temperature = roi_infer_score_temperature,
        roi_infer_fg_bias = roi_infer_fg_bias,
        roi_infer_score_power = roi_infer_score_power,
        use_roi_cmaf = use_roi_cmaf,
        roi_cmaf_scale_init = roi_cmaf_scale_init,
        use_resnet_conformer = use_resnet_conformer,
        resnet_conformer_layers = resnet_conformer_layers,
        resnet_conformer_ffn_dim = resnet_conformer_ffn_dim,
        resnet_conformer_kernel_size = resnet_conformer_kernel_size,
        resnet_conformer_dropout = resnet_conformer_dropout,
        resnet_conformer_scale_init = resnet_conformer_scale_init,
        resnet_conformer_target = resnet_conformer_target,
        use_speech_prior = use_speech_prior,
        speech_prior_dim = speech_prior_dim,
        speech_prior_hidden_dim = speech_prior_hidden_dim,
        speech_prior_dropout = speech_prior_dropout,
        speech_prior_scale_init = speech_prior_scale_init,
        speech_prior_bundle = speech_prior_bundle,
        speech_prior_context_sec = speech_prior_context_sec,
    ).to(DEVICE)
    if distributed:
        model = DDP(
            model,
            device_ids=[DEVICE.index],
            output_device=DEVICE.index,
            broadcast_buffers=False,
            # Torchvision detection losses do not always touch every branch on
            # every step, so strict bucket rebuilding can trip here under DDP.
            find_unused_parameters=True,
        )
        print(f"[TRAIN] DDP ENABLED on rank {rank}/{world_size} (device={DEVICE})")
    elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
        visible_devices = list(range(torch.cuda.device_count()))
        model = DetectionDataParallel(model, device_ids=visible_devices)
        print(f"[TRAIN] DataParallel ENABLED on {len(visible_devices)} visible GPUs")
    core_model = unwrap_model(model)
    bias_probs = initialize_box_predictor_bias(model, fg_class_priors, background_prior=cls_bg_prior)
    init_checkpoint = os.environ.get("DCASE2026_INIT_CHECKPOINT", "").strip()
    if init_checkpoint:
        if not os.path.exists(init_checkpoint):
            raise FileNotFoundError(f"DCASE2026_INIT_CHECKPOINT not found: {init_checkpoint}")
        state = torch.load(init_checkpoint, map_location=DEVICE, weights_only=True)
        load_result = core_model.load_state_dict(state, strict=False)
        if is_main_process():
            print(
                f"[TRAIN] Warm-started model weights from {init_checkpoint} "
                f"(missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)})"
            )
    model_config = core_model.export_config()
    model_config.update(
        {
            "cls_background_prior": float(np.clip(cls_bg_prior, 0.05, 0.95)),
            "foreground_class_counts": fg_class_counts.tolist(),
            "foreground_class_priors": fg_class_priors.tolist(),
            "roi_class_weight_mode": roi_class_weight_mode,
            "roi_class_beta": float(roi_class_beta),
            "roi_class_max_weight": float(roi_class_max_weight),
            "train_scope": train_scope_cfg,
            "train_eval_behavior": bool(train_eval_behavior),
            "disable_augmentation": bool(disable_augmentation),
            "fixed_train_subset": bool(FIXED_TRAIN_SUBSET),
            "train_subset_seed": int(TRAIN_SUBSET_SEED),
            "annotation_target_mode": annotation_target_mode,
            "teacher23_csv_dir": teacher23_csv_dir,
            "teacher23_pos_loss_w": float(teacher23_pos_loss_w),
            "teacher23_sigma_px": float(teacher23_sigma_px),
            "teacher23_fallback_gt_peak": bool(teacher23_fallback_gt_peak),
            "class_balanced_sampling": (
                None if class_balanced_sampling is None else bool(class_balanced_sampling)
            ),
            "class_balanced_fraction": float(class_balanced_fraction),
            "class_balanced_classes": (
                None if class_balanced_classes is None else sorted(class_balanced_classes)
            ),
        }
    )
    if is_main_process():
        config_path = os.path.join(exp_dir, "model_config.json")
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(model_config, fh, indent=2)
        print(
            f"[TRAIN] Detection fusion mode: {det_fusion_mode} | "
            f"initial fusion scale sigmoid(logit)={1.0 / (1.0 + math.exp(-fusion_scale_init)):.4f}"
        )
        print(
            f"[TRAIN] ROI fusion: {'CMAF enabled' if use_roi_cmaf else 'visual-only box head'} | "
            f"scale sigmoid(logit)={1.0 / (1.0 + math.exp(-roi_cmaf_scale_init)):.4f}"
        )
        if use_resnet_conformer:
            print(
                "[TRAIN] ResNet-Conformer FPN adapter ENABLED: "
                f"layers={resnet_conformer_layers} | ffn={resnet_conformer_ffn_dim} | "
                f"kernel={resnet_conformer_kernel_size} | target={resnet_conformer_target} | "
                f"scale sigmoid(logit)={1.0 / (1.0 + math.exp(-resnet_conformer_scale_init)):.4f}"
            )
        else:
            print("[TRAIN] ResNet-Conformer FPN adapter DISABLED")
        if use_speech_prior:
            print(
                "[TRAIN] WavLM speech prior ENABLED: "
                f"bundle={speech_prior_bundle} | dim={speech_prior_dim} | "
                f"hidden={speech_prior_hidden_dim} | context={speech_prior_context_sec:.2f}s | "
                f"scale sigmoid(logit)={1.0 / (1.0 + math.exp(-speech_prior_scale_init)):.4f}"
            )
        else:
            print("[TRAIN] WavLM speech prior DISABLED")
        print(f"[TRAIN] Annotation target mode: {annotation_target_mode}")
        print(
            f"[TRAIN] ROI cls rebalance: mode={roi_class_weight_mode} | "
            f"bg_weight={roi_bg_loss_weight:.2f} | train_tau={roi_logit_adj_tau:.2f} | "
            f"infer_tau={roi_infer_logit_adj_tau:.2f} | label_smoothing={roi_label_smoothing:.3f}"
        )
        print(f"[TRAIN] ROI base cls loss weight: {roi_cls_loss_weight:.2f}")
        print(f"[TRAIN] ROI GT-box aux cls loss weight: {roi_gt_aux_loss_weight:.2f}")
        print(
            f"[TRAIN] Energy losses: annot_w={ENERGY_ANNOT_W:.2f} | "
            f"mask_target={energy_mask_target_mode}/core{energy_mask_core_thr:.2f} | "
            f"mask_w={energy_mask_loss_w:.2f} | soft-IoU_w={energy_mask_iou_w:.2f} | "
            f"mask_bce_w={energy_mask_bce_w:.2f}/{energy_mask_bce_pos_w:.1f} | "
            f"mask_focal_w={energy_mask_focal_w:.2f}/g{energy_mask_focal_gamma:.1f} | "
            f"mask_tversky_w={energy_mask_tversky_w:.2f}/a{energy_mask_tversky_alpha:.2f}/b{energy_mask_tversky_beta:.2f} | "
            f"mask_lovasz_w={energy_mask_lovasz_w:.2f} | "
            f"mask_boundary_w={energy_mask_boundary_w:.2f}/k{energy_mask_boundary_kernel:d} | "
            f"energy_dist_w={energy_dist_w:.2f}/g{energy_dist_gamma:.2f} | "
            f"fullmap_annot_w={FULLMAP_ANNOT_W:.2f} | fullmap_w={fullmap_loss_w:.2f} | "
            f"fullmap_dist_w={fullmap_dist_w:.2f}/g{fullmap_dist_gamma:.2f} | "
            f"fullmap_iou_w={fullmap_mask_iou_w:.2f} | "
            f"fullmap_bce_w={fullmap_mask_bce_w:.2f}/{fullmap_mask_bce_pos_w:.1f} | "
            f"fullmap_focal_w={fullmap_mask_focal_w:.2f}/g{fullmap_mask_focal_gamma:.1f} | "
            f"fullmap_tversky_w={fullmap_mask_tversky_w:.2f}/a{fullmap_mask_tversky_alpha:.2f}/b{fullmap_mask_tversky_beta:.2f} | "
            f"fullmap_lovasz_w={fullmap_mask_lovasz_w:.2f} | "
            f"fullmap_boundary_w={fullmap_mask_boundary_w:.2f}/k{fullmap_mask_boundary_kernel:d} | "
            f"teacher23_pos_w={teacher23_pos_loss_w:.2f}/sigma{teacher23_sigma_px:.1f}"
            f"/fallback_gt_peak={int(teacher23_fallback_gt_peak)} | "
            f"distance_w={distance_loss_w:.2f} | det_w={det_loss_weight:.2f} | "
            f"gt_jitter={gt_box_jitter_frac:.3f}"
        )
        print(
            f"[TRAIN] ROI score calibration: temp={roi_infer_score_temperature:.2f} | "
            f"fg_bias={roi_infer_fg_bias:.2f} | score_power={roi_infer_score_power:.2f}"
        )
        if bias_probs is not None:
            fg_probs = bias_probs[1:]
            top_k = np.argsort(fg_probs)[::-1][:3]
            summary = ", ".join(
                f"class{idx}:{fg_probs[idx]:.3f}" for idx in top_k
            )
            print(
                f"[TRAIN] Box predictor bias initialized with bg_prior={model_config['cls_background_prior']:.2f} "
                f"| top fg priors → {summary}"
            )
        top_weight_idx = np.argsort(fg_class_weights)[::-1][:3]
        weight_summary = ", ".join(
            f"class{idx}:{fg_class_weights[idx]:.2f}" for idx in top_weight_idx
        )
        print(f"[TRAIN] ROI class weights (top-3) → {weight_summary}")
        print(f"[TRAIN] Model config saved → {config_path}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    body = core_model.det.backbone.body
    
    # RGB stem gets a dedicated learning rate and remains trainable.
    backbone_stem_params = list(body.conv1.parameters()) + list(body.bn1.parameters())
    
    # Layer 1 and 2 are totally frozen. They are completely REMOVED from the optimizer.
    backbone_l3_params = list(body.layer3.parameters())
    backbone_l4_params = list(body.layer4.parameters())
    fpn_params         = list(core_model.det.backbone.fpn.parameters())

    # Pretrained COCO components
    rpn_params       = list(core_model.det.rpn.parameters())
    if FREEZE_RPN:
        for p in rpn_params:
            p.requires_grad = False
        rpn_params = []
    box_head_params  = list(core_model.det.roi_heads.box_head.parameters())
    roi_fusion_module = getattr(core_model.det.roi_heads, "roi_fusion_head", None)
    roi_fusion_params = list(roi_fusion_module.parameters()) if roi_fusion_module is not None else []
    resnet_conformer_module = getattr(core_model, "resnet_conformer", None)
    resnet_conformer_params = (
        list(resnet_conformer_module.parameters())
        if resnet_conformer_module is not None else []
    )
    speech_prior_adapter = getattr(core_model, "speech_prior_adapter", None)
    speech_prior_params = (
        list(speech_prior_adapter.parameters())
        if speech_prior_adapter is not None else []
    )
    speech_prior_scale = getattr(core_model, "speech_prior_scale_logit", None)
    if speech_prior_scale is not None:
        speech_prior_params.append(speech_prior_scale)

    # Fresh custom components
    audio_backbone_params = list(core_model.audio_backbone.parameters())
    fusion_params    = list(core_model.fpn_fusion.parameters())
    box_pred_params  = list(core_model.det.roi_heads.box_predictor.parameters())
    custom_params    = (
        audio_backbone_params +
        fusion_params +
        list(core_model.energy_head.parameters()) +
        list(core_model.energy_decoder.parameters()) +
        list(core_model.distance_head.parameters())
    )

    param_groups = []

    def add_param_group(params, lr_mult: float, name: str):
        params = [p for p in params if p.requires_grad]
        if params:
            param_groups.append({
                "params": params,
                "lr": LR * float(lr_mult),
                "weight_decay": 1e-4,
                "name": name,
            })

    add_param_group(backbone_stem_params, BB_STEM_LR_MULT, "bb_stem")
    add_param_group(backbone_l3_params, BB_L3_LR_MULT, "bb_l3")
    add_param_group(backbone_l4_params, BB_L4_LR_MULT, "bb_l4")
    add_param_group(fpn_params, FPN_LR_MULT, "fpn")
    add_param_group(rpn_params, RPN_LR_MULT, "rpn")
    add_param_group(box_head_params, BOX_HEAD_LR_MULT, "box_head")
    add_param_group(box_pred_params, BOX_PRED_LR_MULT, "box_pred")
    add_param_group(roi_fusion_params, ROI_FUSION_LR_MULT, "roi_fusion")
    add_param_group(resnet_conformer_params, RESNET_CONFORMER_LR_MULT, "resnet_conformer")
    add_param_group(speech_prior_params, SPEECH_PRIOR_LR_MULT, "speech_prior")
    add_param_group(custom_params, CUSTOM_LR_MULT, "custom")

    train_scope = set_localization_train_scope(model, train_scope_cfg)
    trainable_after_scope = sum(p.numel() for p in core_model.parameters() if p.requires_grad)

    if is_main_process():
        rpn_msg = "frozen" if FREEZE_RPN else f"{RPN_LR_MULT:.3f}x"
        print(
            "[TRAIN] LR multipliers: "
            f"rpn={rpn_msg} | box_head={BOX_HEAD_LR_MULT:.3f}x | "
            f"box_pred={BOX_PRED_LR_MULT:.3f}x | roi_fusion={ROI_FUSION_LR_MULT:.3f}x | "
            f"resnet_conformer={RESNET_CONFORMER_LR_MULT:.3f}x | "
            f"speech_prior={SPEECH_PRIOR_LR_MULT:.3f}x | "
            f"custom={CUSTOM_LR_MULT:.3f}x"
        )
        print(
            f"[TRAIN] Training subset: "
            f"{'fixed' if FIXED_TRAIN_SUBSET else 'reshuffled'} | "
            f"frames_per_epoch={TRAIN_FRAMES_PER_EPOCH or 'all'} | seed={TRAIN_SUBSET_SEED} | "
            f"class_balanced={train_dataset.class_balanced_sampling} "
            f"(fraction={train_dataset.class_balanced_fraction:.2f}, "
            f"classes={sorted(class_balanced_classes) if class_balanced_classes else 'all'})"
        )
        print(
            f"[TRAIN] Validation subset: "
            f"{'fixed' if FIXED_VAL_SUBSET else 'reshuffled'} | "
            f"frames_per_epoch={VAL_FRAMES_PER_EPOCH or 'all'} | seed={VAL_SUBSET_SEED}"
        )
        print(
            f"[TRAIN] Localization train scope: {train_scope} | "
            f"trainable_after_scope={trainable_after_scope/1e6:.2f}M | "
            f"train_eval_behavior={'ON' if train_eval_behavior else 'OFF'}"
        )

    optimizer = optim.AdamW(param_groups)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        factor   = 0.5,
        patience = 3,         
        min_lr   = 1e-7,
    )

    history           = defaultdict(list)
    t0                = time.time()
    best_loss         = float("inf")
    best_map          = float("-inf")
    best_map_epoch    = 0
    best_epoch_eval_dir = None
    epochs_no_improve = 0
    prev_phase        = None

    def to_float(v):
        return v.item() if isinstance(v, torch.Tensor) else float(v or 0)

    def abbreviate_loss_name(name):
        return (name.replace("loss_", "")
                    .replace("classifier", "cls")
                    .replace("objectness", "obj")
                    .replace("rpn_box_reg", "rpn_reg")
                    .replace("distance", "dist")
                    .replace("energy_mask", "mask")
                    .replace("fullmap", "fmap")[:7])

    effective_batch = batch_size * world_size
    print(
        f"[TRAIN] Max {NUM_EPOCHS} epochs × {len(train_dataset)} frames | "
        f"batch/rank={batch_size} | effective_batch={effective_batch} | "
        f"workers/rank={num_workers} | device={DEVICE}"
    )

    devtest_enabled = bool(val_infos) and env_bool("DCASE2026_DEVTEST_EVAL", True)
    devtest_cfg = {
        "split": os.environ.get("DCASE2026_DEVTEST_SPLIT", "test"),
        "batch_size": env_int("DCASE2026_DEVTEST_BATCH_SIZE", batch_size),
        "num_workers": env_int("DCASE2026_DEVTEST_NUM_WORKERS", val_num_workers),
        "score_thr": env_float("DCASE2026_DEVTEST_SCORE_THR", 0.35),
        "nms_iou_thr": env_float("DCASE2026_DEVTEST_NMS_IOU_THR", 0.30),
        "max_dets_per_class": env_int("DCASE2026_DEVTEST_MAX_DETS_PER_CLASS", 3),
        "min_hits": env_int("DCASE2026_DEVTEST_MIN_HITS", 2),
        "track_mean_score_thr": env_float("DCASE2026_DEVTEST_TRACK_MEAN_SCORE_THR", 0.0),
        "track_peak_score_thr": env_float("DCASE2026_DEVTEST_TRACK_PEAK_SCORE_THR", 0.0),
        "track_mean_score_thr_by_class": os.environ.get("DCASE2026_DEVTEST_TRACK_MEAN_SCORE_THR_BY_CLASS", "").strip(),
        "track_peak_score_thr_by_class": os.environ.get("DCASE2026_DEVTEST_TRACK_PEAK_SCORE_THR_BY_CLASS", "").strip(),
        "max_total_dets_per_frame": env_int("DCASE2026_DEVTEST_MAX_TOTAL_DETS_PER_FRAME", 0),
        "max_total_dets_per_frame_by_class": os.environ.get("DCASE2026_DEVTEST_MAX_TOTAL_DETS_PER_FRAME_BY_CLASS", "").strip(),
        "energy_top_k": env_int("DCASE2026_DEVTEST_ENERGY_TOP_K", DEFAULT_ENERGY_TOP_K),
        "energy_export_thr": env_float("DCASE2026_DEVTEST_ENERGY_EXPORT_THR", ENERGY_EXPORT_THR),
        "energy_source": os.environ.get("DCASE2026_DEVTEST_ENERGY_SOURCE", "roi").strip() or "roi",
        "energy_export_mode": os.environ.get("DCASE2026_DEVTEST_ENERGY_EXPORT_MODE", "topk").strip() or "topk",
        "energy_region_rel_thr": env_float("DCASE2026_DEVTEST_ENERGY_REGION_REL_THR", 0.35),
        "energy_region_pad": env_int("DCASE2026_DEVTEST_ENERGY_REGION_PAD", 0),
        "max_json_mb": env_float("DCASE2026_DEVTEST_MAX_JSON_MB", DEFAULT_MAX_JSON_MB),
        "iou_thr": env_float("DCASE2026_DEVTEST_TRACK_IOU_THR", 0.3),
        "max_age": env_int("DCASE2026_DEVTEST_MAX_AGE", 5),
        "coast_decay": env_float("DCASE2026_DEVTEST_COAST_DECAY", 0.9),
        "sigma": env_float("DCASE2026_DEVTEST_SIGMA", DEFAULT_SIGMA_DEG),
        "energy_thr": env_float("DCASE2026_DEVTEST_ENERGY_THR", DEFAULT_ETH),
        "gpu_acoustic": env_bool("DCASE2026_DEVTEST_GPU_ACOUSTIC", gpu_acoustic),
        "fast_uplam": env_bool("DCASE2026_DEVTEST_FAST_UPLAM", fast_uplam),
        "eval_every": max(1, env_int("DCASE2026_DEVTEST_EVAL_EVERY", 1) or 1),
        "keep_all_epoch_evals": env_bool("DCASE2026_KEEP_ALL_EPOCH_EVALS", False),
        "num_seqs": env_int("DCASE2026_DEVTEST_NUM_SEQS", 0) or 0,
        "num_seqs_seed": env_int("DCASE2026_DEVTEST_NUM_SEQS_SEED", 20260511) or 20260511,
    }
    if devtest_cfg["gpu_acoustic"] and not torch.cuda.is_available():
        devtest_cfg["gpu_acoustic"] = False
    if devtest_cfg["fast_uplam"] and not torch.cuda.is_available():
        devtest_cfg["fast_uplam"] = False

    if is_main_process():
        if devtest_enabled:
            print(
                "[TRAIN] Dev-test eval ENABLED: every "
                f"{devtest_cfg['eval_every']} epoch(s), checkpoint selection by Mask mAP"
            )
            print(
                "[TRAIN] Dev-test inference: "
                f"batch={devtest_cfg['batch_size']} | workers={devtest_cfg['num_workers']} | "
                f"score_thr={devtest_cfg['score_thr']:.2f} | nms={devtest_cfg['nms_iou_thr']:.2f} | "
                f"min_hits={devtest_cfg['min_hits']} | "
                f"track_mean>={devtest_cfg['track_mean_score_thr']:.2f} | "
                f"track_peak>={devtest_cfg['track_peak_score_thr']:.2f} | "
                f"mean_by_class={devtest_cfg['track_mean_score_thr_by_class'] or 'off'} | "
                f"peak_by_class={devtest_cfg['track_peak_score_thr_by_class'] or 'off'} | "
                f"max_total/frame={devtest_cfg['max_total_dets_per_frame'] or 'off'} | "
                f"cap_by_class={devtest_cfg['max_total_dets_per_frame_by_class'] or 'off'} | "
                f"top_k={devtest_cfg['energy_top_k']} | "
                f"max_json={devtest_cfg['max_json_mb']:.2f}MB | "
                f"seqs={devtest_cfg['num_seqs'] or 'all'}"
            )
        else:
            print("[TRAIN] Dev-test eval DISABLED — checkpoint selection falls back to validation loss.")
        if save_every_epoch:
            os.makedirs(epoch_ckpt_dir, exist_ok=True)
            print(f"[TRAIN] Every-epoch checkpoints ENABLED → {epoch_ckpt_dir}")

    def prepare_rgb_images_on_device(images):
        moved = [img.to(DEVICE, non_blocking=pin_memory) for img in images]

        same_shape = bool(moved) and all(img.shape == moved[0].shape for img in moved)
        if same_shape:
            batch = torch.stack(moved, dim=0)
            if batch.dtype == torch.uint8:
                batch = batch.float().div_(255.0)
            else:
                batch = batch.float()
            if batch.shape[-2:] != (IMG_H, IMG_W):
                batch = F.interpolate(batch, size=(IMG_H, IMG_W), mode="bilinear", align_corners=False)
            return [img.contiguous() for img in batch]

        prepared = []
        for image in moved:
            if image.dtype == torch.uint8:
                image = image.float().div_(255.0)
            else:
                image = image.float()
            if image.shape[-2:] != (IMG_H, IMG_W):
                image = F.interpolate(
                    image.unsqueeze(0),
                    size=(IMG_H, IMG_W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            prepared.append(image.contiguous())
        return prepared

    def prepare_batch(images, targets, metas=None, augmentor=None):
        speech_prior_features = None
        if use_speech_prior:
            if metas is None:
                raise RuntimeError("Speech prior requires frame metadata; set return_meta=True on datasets.")
            speech_prior_features = speech_prior_extractor.get_frame_embeddings_batch(
                [meta["wav_path"] for meta in metas],
                [int(meta["frame_idx"]) for meta in metas],
                output_device=DEVICE,
            )
        if gpu_acoustic:
            rgb_images = prepare_rgb_images_on_device(images) if gpu_rgb_prep else [
                img.to(DEVICE, non_blocking=pin_memory) for img in images
            ]
            acoustic_images = acoustic_extractor.get_frame_bands_batch(
                [meta["wav_path"] for meta in metas],
                [int(meta["frame_idx"]) for meta in metas],
                output_device=DEVICE,
            )
            images = [
                torch.cat([rgb, acoustic], dim=0)
                for rgb, acoustic in zip(rgb_images, acoustic_images)
            ]
            targets = [
                {k: v.to(DEVICE, non_blocking=pin_memory) if isinstance(v, torch.Tensor) else v
                 for k, v in t.items()}
                for t in targets
            ]
            if augmentor is not None:
                augmented = [augmentor(img, tgt) for img, tgt in zip(images, targets)]
                images = [img for img, _ in augmented]
                targets = [tgt for _, tgt in augmented]
            return images, targets, speech_prior_features

        if isinstance(model, nn.DataParallel):
            return images, targets, speech_prior_features
        images = prepare_rgb_images_on_device(images) if gpu_rgb_prep else [
            img.to(DEVICE, non_blocking=pin_memory) for img in images
        ]
        targets = [
            {k: v.to(DEVICE, non_blocking=pin_memory) if isinstance(v, torch.Tensor) else v
             for k, v in t.items()}
            for t in targets
        ]
        return images, targets, speech_prior_features

    for epoch in range(1, NUM_EPOCHS + 1):
        train_dataset.reset_epoch()
        val_dataset.reset_epoch()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        if epoch % 10 == 1:
            acoustic_extractor.clear_cache()
            if speech_prior_extractor is not None:
                speech_prior_extractor.clear_cache()

        # ── Apply (or update) backbone freeze schedule ─────────────────────────
        phase, n_trainable, n_total = apply_backbone_freeze(model, epoch)
        if train_scope != "all":
            set_localization_train_scope(model, train_scope)
            n_trainable = sum(p.numel() for p in core_model.parameters() if p.requires_grad)
        if phase != prev_phase:
            print(f"\n[FREEZE] Epoch {epoch}: training phase → '{phase}' "
                  f"({n_trainable/1e6:.1f}M / {n_total/1e6:.1f}M params active)")

            epochs_no_improve = max(0, epochs_no_improve - PHASE_GRACE)
            scheduler.num_bad_epochs = max(0, scheduler.num_bad_epochs - PHASE_GRACE)
            
            print(f"[FREEZE] Phase grace applied: patience counter → {epochs_no_improve}/{PATIENCE}")

            if epoch == UNFREEZE_LAYER4_EPOCH:
                print(f"[FREEZE] layer4 active. (head LR = {_get_pg(optimizer, 'custom')['lr']:.2e})")

            if epoch == UNFREEZE_LAYER3_EPOCH:
                print(f"[FREEZE] layer3 active. (head LR = {_get_pg(optimizer, 'custom')['lr']:.2e})")

            prev_phase = phase

        # ─── TRAIN ────────────────────────────────────────────────────────────
        model.train()
        freeze_backbone_bn(model)
        if train_eval_behavior:
            freeze_bn_dropout_for_loss(model)
        
        ep_losses = defaultdict(float)
        n = 0
        step_timer = StageTimer(profile_steps > 0 and epoch == 1 and is_main_process())

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:3d}/{NUM_EPOCHS} [Train]",
            leave=False,
            file=sys.stdout,
            disable=not is_main_process(),
        )
        data_wait_start = time.perf_counter()
        for batch in pbar:
            if step_timer.enabled and n < profile_steps:
                step_timer.totals["data_wait"] += time.perf_counter() - data_wait_start
                step_timer.counts["data_wait"] += 1
                step_timer.start("prepare_batch")
            if len(batch) == 3:
                images, targets, metas = batch
            else:
                images, targets = batch
                metas = None
            images, targets, speech_prior_features = prepare_batch(
                images,
                targets,
                metas,
                augmentor=train_augmentor if gpu_acoustic else None,
            )
            if step_timer.enabled and n < profile_steps:
                if torch.cuda.is_available():
                    torch.cuda.synchronize(DEVICE)
                step_timer.stop("prepare_batch")
                step_timer.start("forward_backward_step")

            optimizer.zero_grad(set_to_none=True)

            train_loss_context = eval_behavior_for_loss(model) if train_eval_behavior else nullcontext()
            with train_loss_context:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss_dict = model(images, targets, speech_prior_features)
                    loss_dict = {k: reduce_loss_value(v) for k, v in loss_dict.items()}
                    total     = sum(loss_dict.values())

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            if step_timer.enabled and n < profile_steps:
                if torch.cuda.is_available():
                    torch.cuda.synchronize(DEVICE)
                step_timer.stop("forward_backward_step")

            ep_losses["total"] += total.item()
            for k, v in loss_dict.items():
                ep_losses[k] += to_float(v)
            n += 1

            postfix = {"Tot": f"{total.item():.3f}", "phase": phase[:6]}
            for k, v in loss_dict.items():
                postfix[abbreviate_loss_name(k)] = f"{to_float(v):.3f}"
            pbar.set_postfix(postfix)
            data_wait_start = time.perf_counter()

        ep_losses, n = sync_metric_sums(ep_losses, n, DEVICE)
        for k, v in ep_losses.items():
            history[k].append(v / max(n, 1))
        if step_timer.enabled:
            summary = step_timer.summary()
            print("[PROFILE] Train step timing summary:")
            for key, stats in summary.items():
                print(
                    f"  - {key:<22} avg={stats['avg_sec']:.4f}s "
                    f"total={stats['total_sec']:.4f}s count={stats['count']}"
                )

        # ─── VALIDATION ───────────────────────────────────────────────────────
        val_ep_losses = defaultdict(float)
        val_n         = 0

        if len(val_loader) > 0:
            pbar_val = tqdm(
                val_loader,
                desc=f"Epoch {epoch:3d}/{NUM_EPOCHS} [Val]",
                leave=False,
                file=sys.stdout,
                disable=not is_main_process(),
            )
            with torch.no_grad(), eval_behavior_for_loss(model):
                for batch in pbar_val:
                    if len(batch) == 3:
                        images, targets, metas = batch
                    else:
                        images, targets = batch
                        metas = None
                    images, targets, speech_prior_features = prepare_batch(images, targets, metas)

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        loss_dict = model(images, targets, speech_prior_features)
                        loss_dict = {k: reduce_loss_value(v) for k, v in loss_dict.items()}
                        val_total = sum(loss_dict.values())

                    val_ep_losses["total"] += val_total.item()
                    for k, v in loss_dict.items():
                        val_ep_losses[k] += to_float(v)
                    val_n += 1

                    postfix_val = {"Tot": f"{val_total.item():.3f}"}
                    for k, v in loss_dict.items():
                        postfix_val[abbreviate_loss_name(k)] = f"{to_float(v):.3f}"
                    pbar_val.set_postfix(postfix_val)

            val_ep_losses, val_n = sync_metric_sums(val_ep_losses, val_n, DEVICE)
            for k, v in val_ep_losses.items():
                history[f"val_{k}"].append(v / max(val_n, 1))        

        # ─── EPOCH SUMMARY ────────────────────────────────────────────────────
        print(f"\n[{time.strftime('%H:%M:%S')}] Epoch {epoch}/{NUM_EPOCHS} Completed "
              f"({time.time()-t0:.1f}s) | phase={phase}")

        train_tot = history["total"][-1]
        train_sum = sum(history[k][-1] for k in ep_losses if k != "total")
        print(f"  [Train] Verification: Sum({train_sum:.4f}) == Total({train_tot:.4f})")
        for k in sorted(ep_losses):
            if k != "total":
                print(f"          - {k:<28}: {history[k][-1]:.4f}")

        val_avg_total = float("inf")
        if val_n > 0:
            val_avg_total = history["val_total"][-1]
            val_sum       = sum(history[f"val_{k}"][-1] for k in val_ep_losses if k != "total")
            print(f"  [Val]   Verification: Sum({val_sum:.4f}) == Total({val_avg_total:.4f})")
            for k in sorted(val_ep_losses):
                if k != "total":
                    print(f"          - {k:<28}: {history[f'val_{k}'][-1]:.4f}")

        devtest_mask_ap = float("nan")
        devtest_mask_ap50 = float("nan")
        devtest_pearson = float("nan")
        devtest_distance = float("nan")
        devtest_ran = bool(devtest_enabled and epoch % devtest_cfg["eval_every"] == 0)
        improved = False
        should_stop = False
        devtest_payload = None

        if devtest_ran and is_main_process():
            print("  [DevTest] Running full dev-test inference + Mask mAP evaluation ...")
            reuse_acoustic_extractor = acoustic_extractor if (
                devtest_cfg["gpu_acoustic"] and gpu_acoustic
            ) else None
            try:
                with torch.no_grad(), temporary_eval_mode(core_model):
                    devtest_payload = run_devtest_eval(
                        model=core_model,
                        sequence_infos=val_infos,
                        exp_dir=exp_dir,
                        epoch=epoch,
                        cfg=devtest_cfg,
                        acoustic_extractor=reuse_acoustic_extractor,
                        speech_prior_extractor=speech_prior_extractor,
                    )
                devtest_metrics = devtest_payload["metrics"]
                devtest_mask_ap = float(devtest_metrics.get("mask_ap", float("nan")))
                devtest_mask_ap50 = float(devtest_metrics.get("mask_ap50", float("nan")))
                devtest_pearson = float(devtest_metrics.get("pearson_r", float("nan")))
                devtest_distance = float(devtest_metrics.get("distance_mape", float("nan")))
                print(
                    "  [DevTest] "
                    f"mAP={devtest_mask_ap:.4f} | AP50={devtest_mask_ap50:.4f} | "
                    f"Pearson={devtest_pearson:.4f} | DistMAPE={devtest_distance:.2f}%"
                )
            except Exception as exc:
                devtest_enabled = False
                print(
                    "  [WARN] Dev-test evaluation failed and will be disabled for the rest "
                    f"of this run: {type(exc).__name__}: {exc}"
                )
                devtest_payload = None
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Explicit LR Logging
        head_lr = _get_pg(optimizer, "custom")["lr"]
        fpn_lr  = _get_pg(optimizer, "fpn")["lr"]
        rpn_lr = _get_pg_lr(optimizer, "rpn", 0.0)
        box_pred_lr = _get_pg_lr(optimizer, "box_pred", 0.0)
        roi_fusion_lr = _get_pg_lr(optimizer, "roi_fusion", 0.0)
        resnet_conformer_lr = _get_pg_lr(optimizer, "resnet_conformer", 0.0)
        speech_prior_lr = _get_pg_lr(optimizer, "speech_prior", 0.0)
        print(
            f"  [LR] heads={head_lr:.2e} | FPN={fpn_lr:.2e} | "
            f"RPN={rpn_lr:.2e} | box_pred={box_pred_lr:.2e} | "
            f"roi_fusion={roi_fusion_lr:.2e} | resnet_conformer={resnet_conformer_lr:.2e} | "
            f"speech_prior={speech_prior_lr:.2e}"
        )

        # ─── SCHEDULING ───────────────────────────────────────────────────────
        track_raw = val_avg_total if val_n > 0 else train_tot
        track_list = history["val_total"] if val_n > 0 else history["total"]
        recent_smoothed = np.mean(track_list[-5:]) if len(track_list) >= 5 else track_raw
        scheduler.step(recent_smoothed)

        if is_main_process():
            if save_every_epoch:
                epoch_model_path = os.path.join(epoch_ckpt_dir, f"energy_seg_epoch_{epoch:03d}.pth")
                torch.save(core_model.state_dict(), epoch_model_path)
                print(f"  [CKPT] Epoch checkpoint saved → {epoch_model_path}")

            if devtest_ran and is_finite_number(devtest_mask_ap):
                if devtest_mask_ap > best_map:
                    previous_best_eval_dir = best_epoch_eval_dir
                    best_map = devtest_mask_ap
                    best_map_epoch = epoch
                    epochs_no_improve = 0
                    improved = True
                    torch.save(core_model.state_dict(), best_model_path)

                    best_epoch_eval_dir = devtest_payload["epoch_dir"]
                    best_report_src = os.path.join(best_epoch_eval_dir, "eval_summary.txt")
                    best_report_dst = os.path.join(exp_dir, "best_devtest_eval_summary.txt")
                    if os.path.isfile(best_report_src):
                        shutil.copyfile(best_report_src, best_report_dst)

                    best_metrics_payload = {
                        "best_epoch": best_map_epoch,
                        "best_mask_ap": best_map,
                        "best_checkpoint": best_model_path,
                        "epoch_dir": best_epoch_eval_dir,
                        "metrics": devtest_payload["metrics"],
                    }
                    with open(os.path.join(exp_dir, "best_devtest_metrics.json"), "w", encoding="utf-8") as fh:
                        json.dump(best_metrics_payload, fh, indent=2)

                    if (
                        previous_best_eval_dir
                        and previous_best_eval_dir != best_epoch_eval_dir
                        and not devtest_cfg["keep_all_epoch_evals"]
                    ):
                        old_pred_dir = os.path.join(previous_best_eval_dir, "inference_outputs")
                        if os.path.isdir(old_pred_dir):
                            shutil.rmtree(old_pred_dir, ignore_errors=True)

                    print(
                        f"  [*] Dev-test Mask mAP improved to {best_map:.4f} at epoch {best_map_epoch}. "
                        f"Model saved → {best_model_path}"
                    )
                else:
                    epochs_no_improve += 1
                    print(
                        f"  [!] No dev-test mAP improvement for {epochs_no_improve}/{PATIENCE} epochs. "
                        f"(Best: {best_map:.4f} @ epoch {best_map_epoch}, Curr: {devtest_mask_ap:.4f})"
                    )
            else:
                if track_raw < best_loss:
                    best_loss = track_raw
                    epochs_no_improve = 0
                    improved = True
                    torch.save(core_model.state_dict(), best_model_path)
                    print(
                        f"  [*] Validation loss improved to {best_loss:.4f}. "
                        f"Fallback checkpoint saved → {best_model_path}"
                    )
                else:
                    epochs_no_improve += 1
                    print(
                        f"  [!] No validation-loss improvement for {epochs_no_improve}/{PATIENCE} epochs. "
                        f"(Best: {best_loss:.4f}, Curr: {track_raw:.4f})"
                    )

            if devtest_ran and devtest_payload is not None and not improved and not devtest_cfg["keep_all_epoch_evals"]:
                shutil.rmtree(devtest_payload["pred_dir"], ignore_errors=True)
                print(f"  [DevTest] Removed non-best inference outputs → {devtest_payload['pred_dir']}")

            should_stop = epochs_no_improve >= PATIENCE
            state_payload = broadcast_scalar_payload(
                [
                    best_loss,
                    best_map,
                    float(best_map_epoch),
                    float(epochs_no_improve),
                    1.0 if improved else 0.0,
                    1.0 if should_stop else 0.0,
                    1.0 if devtest_enabled else 0.0,
                    devtest_mask_ap,
                    devtest_mask_ap50,
                    devtest_pearson,
                    devtest_distance,
                    1.0 if devtest_ran else 0.0,
                ],
                DEVICE,
            )
        else:
            state_payload = broadcast_scalar_payload([0.0] * 12, DEVICE)

        (
            best_loss,
            best_map,
            best_map_epoch_raw,
            epochs_no_improve_raw,
            improved_raw,
            should_stop_raw,
            devtest_enabled_raw,
            devtest_mask_ap,
            devtest_mask_ap50,
            devtest_pearson,
            devtest_distance,
            devtest_ran_raw,
        ) = state_payload
        best_map_epoch = int(round(best_map_epoch_raw))
        epochs_no_improve = int(round(epochs_no_improve_raw))
        improved = improved_raw >= 0.5
        should_stop = should_stop_raw >= 0.5
        devtest_enabled = devtest_enabled_raw >= 0.5
        devtest_ran = devtest_ran_raw >= 0.5

        history["devtest_mask_ap"].append(devtest_mask_ap if devtest_ran else float("nan"))
        history["devtest_mask_ap50"].append(devtest_mask_ap50 if devtest_ran else float("nan"))
        history["devtest_pearson_r"].append(devtest_pearson if devtest_ran else float("nan"))
        history["devtest_distance_mape"].append(devtest_distance if devtest_ran else float("nan"))

        if should_stop:
            print(f"\n[!] Early stopping triggered after {epoch} epochs.")
            break

    print(f"\n[TRAIN] Finished in {time.time()-t0:.1f}s")
    sync_processes()
    if is_main_process() and os.path.exists(best_model_path):
        core_model.load_state_dict(
            torch.load(best_model_path, map_location=DEVICE, weights_only=True)
        )
    return core_model, dict(history)


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Train Energy Instance Model")
    parser.add_argument("--exp_name", type=str, default="baseline", help="Name of the experiment run")
    parser.add_argument("--seed",     type=int, default=42,         help="Random seed for reproducibility")
    args = parser.parse_args()
    distributed = init_distributed_mode()

    timestamp = os.environ.get("DCASE2026_RUN_STAMP") or time.strftime("%Y%m%d_%H%M%S")
    exp_dir   = os.path.join(os.getcwd(), "experiments", f"{args.exp_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    sync_processes()

    log_prefix = "" if is_main_process() else f"rank{get_rank()}_"
    sys.stdout = Logger(os.path.join(exp_dir, f"{log_prefix}train.log"), sys.stdout)
    sys.stderr = Logger(os.path.join(exp_dir, f"{log_prefix}error.log"), sys.stderr)

    print(f"[MAIN] Starting experiment: {args.exp_name}")
    print(f"[MAIN] Output directory:    {exp_dir}")
    print(f"[MAIN] Rank/World:          {get_rank()}/{get_world_size()}")

    try:
        set_seed(args.seed + get_rank())

        train_infos = filter_sequence_infos(
            get_sequence_infos("train", LABELS_BASE, FRAMES_BASE),
            "TRAIN",
        )
        test_infos = filter_sequence_infos(
            get_sequence_infos("test", LABELS_BASE, FRAMES_BASE),
            "TEST",
        )

        if not train_infos:
            raise FileNotFoundError(f"No train directories found in {LABELS_BASE}")
        if not test_infos:
            print(f"[WARN] No val directories found in {LABELS_BASE}")

        train_on_all_dev = env_bool("DCASE2026_TRAIN_ON_ALL_DEV", False)
        if train_on_all_dev:
            # Intended only for final-evaluation style training, where the
            # challenge rules allow using all development data.  Do not use
            # these runs as fair dev-test validation comparisons.
            train_infos = list(train_infos) + list(test_infos)
            print(
                "\n[MAIN] DCASE2026_TRAIN_ON_ALL_DEV=1: "
                "training on dev-train + dev-test labels."
            )
            print(
                "[MAIN] WARNING: dev-test metrics from this run are label-leaked "
                "and should only guide final-submission fine-tuning."
            )

        print(f"\n[MAIN] {len(train_infos)} training sequences | {len(test_infos)} val sequences")

        model, history = train(train_infos, test_infos, exp_dir)
        sync_processes()
        if is_main_process():
            plot_losses(history, os.path.join(exp_dir, "training_loss.png"))

            print("\n[DONE]")
            print(f"  All artifacts (logs, checkpoints, loss curves) saved to: {exp_dir}")
    finally:
        cleanup_distributed()
