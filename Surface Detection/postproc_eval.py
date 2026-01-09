#!/usr/bin/env python3
"""
Vesuvius 2025 (Surface Detection) - local post-processing evaluation script (parallel).

Per volume:
- baseline binarization
- legacy post-process (3D spherical closing + remove_small_objects)
- advanced post-process (topology-friendly hysteresis + cleaning)

Defaults to a fallback metric implementation (good for A/B testing).
Runs in parallel via ProcessPoolExecutor (safe on macOS/Windows due to __main__ guard).
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- Global Constants -------------------------------------------------------

PRED_DIR = Path("data/validate-post-process/for_val_pp")
GT_DIR = Path("data/validate-post-process/labels")

PARALLEL = True
NUM_WORKERS = max(1, (os.cpu_count() or 4) - 1)

BASE_THR = 0.50

# Advanced post-processing (tune)
T_LOW = 0.45
T_HIGH = 0.80
PRUNE_BELOW = None
CLOSE2D_RADIUS = 1
MIN_SIZE = 500
HOLE_SIZE = 1000

# Legacy post-processing (from notebook)
LEGACY_MIN_SIZE = 1000
LEGACY_CLOSING_RADIUS = 5

# Metric params
TAU = 2.0
SPACING = (1.0, 1.0, 1.0)
VOI_ALPHA = 0.3

# Leaderboard weights
TOPO_WEIGHT = 0.30
SURFACE_WEIGHT = 0.35
VOI_WEIGHT = 0.35

SAVE_CSV = Path("postproc_metrics.csv")


# --- IO helpers -------------------------------------------------------------

def _read_volume(path: Path) -> np.ndarray:
    suf = path.suffix.lower()
    if suf in [".tif", ".tiff"]:
        try:
            import tifffile
        except ImportError as e:
            raise RuntimeError("tifffile is required to read .tif/.tiff volumes. Install: pip install tifffile") from e
        return tifffile.imread(str(path))
    if suf == ".npy":
        return np.load(str(path))
    if suf == ".npz":
        z = np.load(str(path))
        if "arr_0" in z:
            return z["arr_0"]
        return z[z.files[0]]
    raise ValueError(f"Unsupported volume extension: {path.name}")


def _list_volumes(dir_path: Path) -> Dict[str, Path]:
    exts = (".tif", ".tiff", ".npy", ".npz")
    out: Dict[str, Path] = {}
    for p in sorted(dir_path.glob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            out[p.stem] = p
    return out


def _pair_pred_gt(pred_dir: Path, gt_dir: Path) -> List[Tuple[str, Path, Path]]:
    preds = _list_volumes(pred_dir)
    gts = _list_volumes(gt_dir)
    common = sorted(set(preds.keys()) & set(gts.keys()))
    if not common:
        raise FileNotFoundError(
            f"No matching volume ids found.\n"
            f"pred_dir={pred_dir} contains {len(preds)} volumes\n"
            f"gt_dir={gt_dir} contains {len(gts)} volumes\n"
            f"Expected matching stems (filenames without extension)."
        )
    missing_pred = sorted(set(gts.keys()) - set(preds.keys()))
    missing_gt = sorted(set(preds.keys()) - set(gts.keys()))
    if missing_pred:
        warnings.warn(f"{len(missing_pred)} GT volumes have no prediction match (ignored): {missing_pred[:10]}")
    if missing_gt:
        warnings.warn(f"{len(missing_gt)} predictions have no GT match (ignored): {missing_gt[:10]}")
    return [(vid, preds[vid], gts[vid]) for vid in common]


def _as_probability(pred: np.ndarray) -> Tuple[np.ndarray, str]:
    arr = np.asarray(pred)
    if arr.dtype == np.bool_:
        return arr.astype(np.float32), "binary"

    if arr.dtype.kind in "iu":
        vmax = int(arr.max()) if arr.size else 0
        vmin = int(arr.min()) if arr.size else 0
        if vmax <= 2:
            return (arr > 0).astype(np.float32), "binary"
        if 0 <= vmin and vmax <= 255:
            return (arr.astype(np.float32) / 255.0), "prob"
        arr_f = arr.astype(np.float32)
        if arr_f.max() > arr_f.min():
            arr_f = (arr_f - arr_f.min()) / (arr_f.max() - arr_f.min())
        return arr_f, "prob"

    if arr.dtype.kind == "f":
        arr_f = arr.astype(np.float32)
        if arr_f.min() < 0.0 or arr_f.max() > 1.0:
            arr_f = 1.0 / (1.0 + np.exp(-arr_f))
        return np.clip(arr_f, 0.0, 1.0), "prob"

    arr_f = np.clip(arr.astype(np.float32), 0.0, 1.0)
    return arr_f, "prob"


# --- Post-processing --------------------------------------------------------

def _label_3d(mask: np.ndarray, connectivity: int = 26) -> Tuple[np.ndarray, int]:
    from scipy.ndimage import label, generate_binary_structure
    mask = mask.astype(bool, copy=False)
    conn = 3 if connectivity == 26 else 1
    structure = generate_binary_structure(rank=3, connectivity=conn)  # 1->6, 3->26
    lbl, n = label(mask, structure=structure)
    return lbl, int(n)


def post_process_from_probs(
    prob: np.ndarray,
    ignore_mask: Optional[np.ndarray] = None,
    t_low: float = T_LOW,
    t_high: float = T_HIGH,
    prune_below: Optional[float] = PRUNE_BELOW,
    close2d_radius: int = CLOSE2D_RADIUS,
    min_size: int = MIN_SIZE,
    hole_size: int = HOLE_SIZE,
) -> np.ndarray:
    from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing, disk

    prob = prob.astype(np.float32, copy=False)
    if ignore_mask is None:
        ignore_mask = np.zeros_like(prob, dtype=bool)
    else:
        ignore_mask = ignore_mask.astype(bool, copy=False)

    weak = (prob >= float(t_low)) & (~ignore_mask)
    strong = (prob >= float(t_high)) & (~ignore_mask)

    if weak.sum() == 0:
        return np.zeros_like(prob, dtype=np.uint8)

    lbl, n = _label_3d(weak, connectivity=26)
    if n == 0:
        return np.zeros_like(prob, dtype=np.uint8)

    strong_labels = np.unique(lbl[strong])
    strong_labels = strong_labels[strong_labels != 0]
    keep = np.isin(lbl, strong_labels)

    if prune_below is not None:
        keep &= (prob >= float(prune_below))

    if min_size and min_size > 0:
        keep = remove_small_objects(keep, min_size=int(min_size), connectivity=3)

    if hole_size and hole_size > 0:
        keep = remove_small_holes(keep, area_threshold=int(hole_size), connectivity=3)

    if close2d_radius and close2d_radius > 0:
        se = disk(int(close2d_radius))
        out = keep.copy()
        for z in range(out.shape[0]):
            out[z] = binary_closing(out[z], se)
        keep = out

    keep[ignore_mask] = False
    return keep.astype(np.uint8)


def post_process_legacy_3d(
    volume: np.ndarray,
    min_size: int = LEGACY_MIN_SIZE,
    closing_radius: int = LEGACY_CLOSING_RADIUS,
) -> np.ndarray:
    from skimage.morphology import ball, remove_small_objects
    from scipy.ndimage import binary_dilation, binary_erosion

    binary = np.asarray(volume) > 0
    if closing_radius > 0:
        selem = ball(int(closing_radius))
        closed = binary_dilation(binary, selem)
        closed = binary_erosion(closed, selem)
    else:
        closed = binary

    cleaned = remove_small_objects(closed, min_size=int(min_size), connectivity=3)
    return cleaned.astype(np.uint8)


# --- Fallback metric implementation ----------------------------------------

@dataclass
class MetricBreakdown:
    topo: float
    surface_dice: float
    voi: float
    score: float


def surface_dice_at_tau(
    pred: np.ndarray,
    gt: np.ndarray,
    tau: float = TAU,
    spacing: Tuple[float, float, float] = SPACING,
) -> float:
    from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

    pred = pred.astype(bool, copy=False)
    gt = gt.astype(bool, copy=False)

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0

    struct = generate_binary_structure(3, 1)
    pred_er = binary_erosion(pred, structure=struct, border_value=0)
    gt_er = binary_erosion(gt, structure=struct, border_value=0)
    pred_s = pred & (~pred_er)
    gt_s = gt & (~gt_er)

    if pred_s.sum() == 0 and gt_s.sum() == 0:
        return 1.0
    if pred_s.sum() == 0 or gt_s.sum() == 0:
        return 0.0

    dt_to_gt = distance_transform_edt(~gt_s, sampling=spacing)
    dt_to_pr = distance_transform_edt(~pred_s, sampling=spacing)

    pred_match = float((dt_to_gt[pred_s] <= tau).mean())
    gt_match = float((dt_to_pr[gt_s] <= tau).mean())
    return float(0.5 * (pred_match + gt_match))


def voi_score(
    pred: np.ndarray,
    gt: np.ndarray,
    alpha: float = VOI_ALPHA,
    connectivity: int = 26,
) -> float:
    pred = pred.astype(bool, copy=False)
    gt = gt.astype(bool, copy=False)

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0

    union = pred | gt
    N = int(union.sum())
    if N == 0:
        return 1.0

    gt_lbl, _ = _label_3d(gt, connectivity=connectivity)
    pr_lbl, _ = _label_3d(pred, connectivity=connectivity)

    a = gt_lbl[union].astype(np.int64, copy=False)
    b = pr_lbl[union].astype(np.int64, copy=False)

    bmax = int(b.max())
    key = a * (bmax + 1) + b
    keys, counts = np.unique(key, return_counts=True)

    a_keys = keys // (bmax + 1)
    b_keys = keys % (bmax + 1)

    p_ij = counts.astype(np.float64) / float(N)

    p_i: Dict[int, float] = {}
    q_j: Dict[int, float] = {}
    for ai, bj, pij in zip(a_keys, b_keys, p_ij):
        p_i[int(ai)] = p_i.get(int(ai), 0.0) + float(pij)
        q_j[int(bj)] = q_j.get(int(bj), 0.0) + float(pij)

    eps = 1e-12
    H_gt_given_pr = 0.0
    H_pr_given_gt = 0.0
    for ai, bj, pij in zip(a_keys, b_keys, p_ij):
        q = max(q_j[int(bj)], eps)
        p = max(p_i[int(ai)], eps)
        H_gt_given_pr -= float(pij) * math.log2(max(float(pij), eps) / q)
        H_pr_given_gt -= float(pij) * math.log2(max(float(pij), eps) / p)

    voi_total = H_gt_given_pr + H_pr_given_gt
    return float(1.0 / (1.0 + alpha * voi_total))


def _betti_numbers_approx(fg: np.ndarray) -> Tuple[int, int, int, int]:
    from scipy.ndimage import label, generate_binary_structure
    from skimage.measure import euler_number

    fg = fg.astype(bool, copy=False)

    struct26 = generate_binary_structure(3, 3)
    _, b0 = label(fg, structure=struct26)

    bg = ~fg
    struct6 = generate_binary_structure(3, 1)
    bg_lbl, nb = label(bg, structure=struct6)
    if nb == 0:
        b2 = 0
    else:
        border = np.zeros_like(bg, dtype=bool)
        border[0, :, :] = True
        border[-1, :, :] = True
        border[:, 0, :] = True
        border[:, -1, :] = True
        border[:, :, 0] = True
        border[:, :, -1] = True
        outside_ids = np.unique(bg_lbl[border])
        outside_ids = outside_ids[outside_ids != 0]
        all_ids = np.arange(1, nb + 1, dtype=np.int32)
        cavity_ids = np.setdiff1d(all_ids, outside_ids, assume_unique=False)
        b2 = int(len(cavity_ids))

    chi = int(euler_number(fg, connectivity=3))
    b1 = int(b0 + b2 - chi)
    if b1 < 0:
        b1 = 0
    return int(b0), int(b1), int(b2), int(chi)


def topo_score_approx(
    pred: np.ndarray,
    gt: np.ndarray,
    w0: float = 0.34,
    w1: float = 0.33,
    w2: float = 0.33,
) -> float:
    pred = pred.astype(bool, copy=False)
    gt = gt.astype(bool, copy=False)

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0

    b0_p, b1_p, b2_p, _ = _betti_numbers_approx(pred)
    b0_g, b1_g, b2_g, _ = _betti_numbers_approx(gt)

    def f1_count(a: int, b: int) -> Optional[float]:
        if a == 0 and b == 0:
            return None
        m = min(a, b)
        prec = m / a if a > 0 else 0.0
        rec = m / b if b > 0 else 0.0
        if prec + rec == 0.0:
            return 0.0
        return 2.0 * prec * rec / (prec + rec)

    f0 = f1_count(b0_p, b0_g)
    f1v = f1_count(b1_p, b1_g)
    f2 = f1_count(b2_p, b2_g)

    parts = []
    weights = []
    if f0 is not None:
        parts.append(f0); weights.append(w0)
    if f1v is not None:
        parts.append(f1v); weights.append(w1)
    if f2 is not None:
        parts.append(f2); weights.append(w2)
    if not parts:
        return 1.0

    w = np.array(weights, dtype=np.float32)
    w /= w.sum()
    return float((np.array(parts, dtype=np.float32) * w).sum())


def compute_fallback_metrics(pred_bin: np.ndarray, gt_label: np.ndarray) -> MetricBreakdown:
    ignore = (gt_label == 2)
    gt_fg = (gt_label == 1)

    pred = pred_bin.astype(bool, copy=False)
    pred[ignore] = False
    gt_fg = gt_fg & (~ignore)

    s = surface_dice_at_tau(pred, gt_fg, tau=TAU, spacing=SPACING)
    v = voi_score(pred, gt_fg, alpha=VOI_ALPHA, connectivity=26)
    t = topo_score_approx(pred, gt_fg)

    score = TOPO_WEIGHT * t + SURFACE_WEIGHT * s + VOI_WEIGHT * v
    return MetricBreakdown(topo=float(t), surface_dice=float(s), voi=float(v), score=float(score))


@dataclass
class VolumeReport:
    volume_id: str
    baseline: MetricBreakdown
    legacy: MetricBreakdown
    advanced: MetricBreakdown


def _process_one(args: Tuple[str, str, str]) -> VolumeReport:
    vid, pred_path_s, gt_path_s = args
    pred_raw = _read_volume(Path(pred_path_s))
    gt = _read_volume(Path(gt_path_s))

    if pred_raw.shape != gt.shape:
        raise ValueError(f"Shape mismatch for {vid}: pred={pred_raw.shape}, gt={gt.shape}")

    prob, mode = _as_probability(pred_raw)
    ignore = (gt == 2)

    base_pred = (prob > 0.0) if mode == "binary" else (prob >= float(BASE_THR))
    base_pred = base_pred & (~ignore)

    legacy_u8 = post_process_legacy_3d(base_pred, min_size=LEGACY_MIN_SIZE, closing_radius=LEGACY_CLOSING_RADIUS)
    legacy_pred = (legacy_u8 > 0) & (~ignore)

    adv_u8 = post_process_from_probs(
        prob=prob,
        ignore_mask=ignore,
        t_low=float(T_LOW),
        t_high=float(T_HIGH),
        prune_below=PRUNE_BELOW,
        close2d_radius=int(CLOSE2D_RADIUS),
        min_size=int(MIN_SIZE),
        hole_size=int(HOLE_SIZE),
    )
    adv_pred = (adv_u8 > 0) & (~ignore)

    return VolumeReport(
        volume_id=vid,
        baseline=compute_fallback_metrics(base_pred, gt),
        legacy=compute_fallback_metrics(legacy_pred, gt),
        advanced=compute_fallback_metrics(adv_pred, gt),
    )


def main() -> int:
    if not PRED_DIR.exists():
        raise FileNotFoundError(f"pred_dir not found: {PRED_DIR.resolve()}")
    if not GT_DIR.exists():
        raise FileNotFoundError(f"gt_dir not found: {GT_DIR.resolve()}")

    pairs = _pair_pred_gt(PRED_DIR, GT_DIR)
    print(f"[info] Found {len(pairs)} paired volumes.")
    print(f"[info] Parallel={PARALLEL} workers={NUM_WORKERS}")

    tasks = [(vid, str(pred_path), str(gt_path)) for vid, pred_path, gt_path in pairs]

    reports: List[VolumeReport] = []

    if PARALLEL and NUM_WORKERS > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from tqdm import tqdm

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futs = [ex.submit(_process_one, t) for t in tasks]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Scoring volumes"):
                rep = fut.result()
                reports.append(rep)
                print(
                    f"{rep.volume_id}: "
                    f"baseline={rep.baseline.score:.4f} | "
                    f"legacy={rep.legacy.score:.4f} | "
                    f"advanced={rep.advanced.score:.4f} | "
                    f"Δ adv-base={rep.advanced.score - rep.baseline.score:+.4f}"
                )
    else:
        for t in tasks:
            rep = _process_one(t)
            reports.append(rep)
            print(
                f"{rep.volume_id}: "
                f"baseline={rep.baseline.score:.4f} | "
                f"legacy={rep.legacy.score:.4f} | "
                f"advanced={rep.advanced.score:.4f} | "
                f"Δ adv-base={rep.advanced.score - rep.baseline.score:+.4f}"
            )

    reports.sort(key=lambda r: r.volume_id)

    def avg(vals: List[float]) -> float:
        return float(np.mean(np.array(vals, dtype=np.float64))) if vals else float("nan")

    base_scores = [r.baseline.score for r in reports]
    legacy_scores = [r.legacy.score for r in reports]
    advanced_scores = [r.advanced.score for r in reports]

    print("\n=== Averages over volumes ===")
    print(f"Baseline : {avg(base_scores):.5f}")
    print(f"Legacy   : {avg(legacy_scores):.5f}  (Δ {avg(legacy_scores) - avg(base_scores):+.5f})")
    print(f"Advanced : {avg(advanced_scores):.5f}  (Δ {avg(advanced_scores) - avg(base_scores):+.5f})")

    import csv
    SAVE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SAVE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "volume_id",
            "baseline_score", "baseline_topo", "baseline_surface", "baseline_voi",
            "legacy_score", "legacy_topo", "legacy_surface", "legacy_voi",
            "advanced_score", "advanced_topo", "advanced_surface", "advanced_voi",
        ])
        for r in reports:
            w.writerow([
                r.volume_id,
                f"{r.baseline.score:.6f}", f"{r.baseline.topo:.6f}",
                f"{r.baseline.surface_dice:.6f}", f"{r.baseline.voi:.6f}",
                f"{r.legacy.score:.6f}", f"{r.legacy.topo:.6f}",
                f"{r.legacy.surface_dice:.6f}", f"{r.legacy.voi:.6f}",
                f"{r.advanced.score:.6f}", f"{r.advanced.topo:.6f}",
                f"{r.advanced.surface_dice:.6f}", f"{r.advanced.voi:.6f}",
            ])

    print(f"[info] Saved per-volume metrics to: {SAVE_CSV.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
