#!/usr/bin/env python3
"""
Compare post-processing on:
  1) binary predictions (0/1)  in data/validate-post-process/for_val_pp
  2) prob predictions   (0..255) in data/validate-post-process/val_probs_u8_no_post

Metrics: fallback approximation (SurfaceDice@tau + VOI_score + TopoScore approx)
Post-proc:
  - legacy 3D closing + remove_small_objects
  - advanced hysteresis-from-probs + cleaning (exactly as provided)
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ------------------------
# PATHS (EDIT IF NEEDED)
# ------------------------
GT_DIR = Path("data/validate-post-process/labels")

PRED_DIRS = [
    # ("bin_0_1", Path("data/validate-post-process/for_val_pp")),
    ("prob_u8", Path("data/validate-post-process/val_probs_u8_no_post")),
]

# ------------------------
# SETTINGS
# ------------------------
PARALLEL = True
NUM_WORKERS = min(16, (os.cpu_count() or 8))

PRINT_PER_VOLUME = True  # set False if you want less spam

# Baseline threshold (only for prob mode)
BASE_THR = 0.50

# Advanced post-processing params (same defaults you used)
T_LOW = 0.45
T_HIGH = 0.80
PRUNE_BELOW = 0.55
CLOSE2D_RADIUS = 1
MIN_SIZE = 500
HOLE_SIZE = 1000

# Legacy params
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


# ------------------------
# IO
# ------------------------
def _read_volume(path: Path) -> np.ndarray:
    suf = path.suffix.lower()
    if suf in [".tif", ".tiff"]:
        try:
            import tifffile
        except ImportError as e:
            raise RuntimeError("tifffile required: pip install tifffile") from e
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
            f"No matching ids.\n"
            f"pred_dir={pred_dir} has {len(preds)} vols\n"
            f"gt_dir={gt_dir} has {len(gts)} vols\n"
        )
    missing_gt = sorted(set(preds.keys()) - set(gts.keys()))
    missing_pr = sorted(set(gts.keys()) - set(preds.keys()))
    if missing_gt:
        warnings.warn(f"{len(missing_gt)} preds have no GT match (ignored): {missing_gt[:10]}")
    if missing_pr:
        warnings.warn(f"{len(missing_pr)} GT have no pred match (ignored): {missing_pr[:10]}")
    return [(vid, preds[vid], gts[vid]) for vid in common]


def _as_probability(pred: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Returns (prob_float32_in_[0..1], mode) where mode in {"binary","prob"}.
    Handles:
      - bool
      - int masks {0,1} or {0,255}
      - uint8 probs 0..255
      - float probs 0..1
    """
    arr = np.asarray(pred)

    if arr.dtype == np.bool_:
        return arr.astype(np.float32), "binary"

    if arr.dtype.kind in "iu":
        vmax = int(arr.max()) if arr.size else 0
        vmin = int(arr.min()) if arr.size else 0
        if vmax <= 2:
            return (arr > 0).astype(np.float32), "binary"
        if 0 <= vmin and vmax <= 255:
            # treat as prob_u8 or mask_u8
            p = arr.astype(np.float32) / 255.0
            # if it's essentially {0,1} after scaling, still ok
            return p, "prob"
        arr_f = arr.astype(np.float32)
        if arr_f.max() > arr_f.min():
            arr_f = (arr_f - arr_f.min()) / (arr_f.max() - arr_f.min())
        return arr_f, "prob"

    if arr.dtype.kind == "f":
        arr_f = arr.astype(np.float32)
        if arr_f.min() < 0.0 or arr_f.max() > 1.0:
            # assume logits -> sigmoid
            arr_f = 1.0 / (1.0 + np.exp(-arr_f))
        return np.clip(arr_f, 0.0, 1.0), "prob"

    arr_f = np.clip(arr.astype(np.float32), 0.0, 1.0)
    return arr_f, "prob"


# ------------------------
# Post-processing (exactly your functions)
# ------------------------
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

    # p = prob.astype(np.float32)
    # print("frac>=0.80", (p >= 0.80).mean())
    # print("frac>=0.70", (p >= 0.70).mean())
    # print("frac>=0.60", (p >= 0.60).mean())
    # print("max", p.max(), "p99", np.quantile(p, 0.99), "p999", np.quantile(p, 0.999))

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


# ------------------------
# Metrics (fallback)
# ------------------------
@dataclass
class MetricBreakdown:
    topo: float
    surface_dice: float
    voi: float
    score: float


def surface_dice_at_tau(pred: np.ndarray, gt: np.ndarray, tau: float = TAU, spacing: Tuple[float, float, float] = SPACING) -> float:
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


def voi_score(pred: np.ndarray, gt: np.ndarray, alpha: float = VOI_ALPHA, connectivity: int = 26) -> float:
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


def topo_score_approx(pred: np.ndarray, gt: np.ndarray, w0: float = 0.34, w1: float = 0.33, w2: float = 0.33) -> float:
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


def compute_metrics(pred_bin: np.ndarray, gt_label: np.ndarray) -> MetricBreakdown:
    ignore = (gt_label == 2)
    gt_fg = (gt_label == 1) & (~ignore)

    pred = pred_bin.astype(bool, copy=False)
    pred[ignore] = False

    s = surface_dice_at_tau(pred, gt_fg, tau=TAU, spacing=SPACING)
    v = voi_score(pred, gt_fg, alpha=VOI_ALPHA, connectivity=26)
    t = topo_score_approx(pred, gt_fg)

    score = TOPO_WEIGHT * t + SURFACE_WEIGHT * s + VOI_WEIGHT * v
    return MetricBreakdown(float(t), float(s), float(v), float(score))


# ------------------------
# Eval per-dir
# ------------------------
@dataclass
class VolumeResult:
    vid: str
    baseline: MetricBreakdown
    legacy: MetricBreakdown
    advanced: MetricBreakdown

def _process_one_with_cfg(task: Tuple[str, str, str], cfg: Tuple[float, float, Optional[float]]) -> VolumeResult:
    """
    Same as _process_one, but advanced postproc uses cfg=(t_low, t_high, prune_below).
    Baseline/legacy are still computed in the same way (baseline thresholding at BASE_THR).
    """
    vid, pred_path_s, gt_path_s = task
    pred_raw = _read_volume(Path(pred_path_s))
    gt = _read_volume(Path(gt_path_s)).astype(np.uint8, copy=False)

    if pred_raw.shape != gt.shape:
        raise ValueError(f"Shape mismatch for {vid}: pred={pred_raw.shape}, gt={gt.shape}")

    prob, mode = _as_probability(pred_raw)
    ignore = (gt == 2)

    # baseline
    if mode == "binary":
        base_pred = (prob > 0.0)
    else:
        base_pred = (prob >= float(BASE_THR))
    base_pred = base_pred & (~ignore)

    # legacy
    legacy_u8 = post_process_legacy_3d(base_pred, min_size=LEGACY_MIN_SIZE, closing_radius=LEGACY_CLOSING_RADIUS)
    legacy_pred = (legacy_u8 > 0) & (~ignore)

    # advanced with cfg
    t_low, t_high, prune_below = cfg
    adv_u8 = post_process_from_probs(
        prob=prob,
        ignore_mask=ignore,
        t_low=float(t_low),
        t_high=float(t_high),
        prune_below=prune_below,
        close2d_radius=int(CLOSE2D_RADIUS),
        min_size=int(MIN_SIZE),
        hole_size=int(HOLE_SIZE),
    )
    adv_pred = (adv_u8 > 0) & (~ignore)

    return VolumeResult(
        vid=vid,
        baseline=compute_metrics(base_pred, gt),
        legacy=compute_metrics(legacy_pred, gt),
        advanced=compute_metrics(adv_pred, gt),
    )

def _process_one(task: Tuple[str, str, str]) -> VolumeResult:
    vid, pred_path_s, gt_path_s = task
    pred_raw = _read_volume(Path(pred_path_s))
    gt = _read_volume(Path(gt_path_s)).astype(np.uint8, copy=False)

    if pred_raw.shape != gt.shape:
        raise ValueError(f"Shape mismatch for {vid}: pred={pred_raw.shape}, gt={gt.shape}")

    prob, mode = _as_probability(pred_raw)
    ignore = (gt == 2)

    # baseline
    if mode == "binary":
        base_pred = (prob > 0.0)
    else:
        base_pred = (prob >= float(BASE_THR))
    base_pred = base_pred & (~ignore)

    # legacy on baseline binary
    legacy_u8 = post_process_legacy_3d(base_pred, min_size=LEGACY_MIN_SIZE, closing_radius=LEGACY_CLOSING_RADIUS)
    legacy_pred = (legacy_u8 > 0) & (~ignore)

    # advanced on probs
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

    return VolumeResult(
        vid=vid,
        baseline=compute_metrics(base_pred, gt),
        legacy=compute_metrics(legacy_pred, gt),
        advanced=compute_metrics(adv_pred, gt),
    )


def _avg(xs: List[float]) -> float:
    return float(np.mean(np.array(xs, dtype=np.float64))) if xs else float("nan")


CAND = [
  (0.35, 0.70, 0.50),
  (0.35, 0.70, 0.55),
  (0.35, 0.70, 0.60),
  (0.30, 0.65, 0.55),
  (0.40, 0.75, 0.60),
  (0.35, 0.60, 0.50),
]

def evaluate_pred_dir(tag: str, pred_dir: Path) -> None:
    pairs = _pair_pred_gt(pred_dir, GT_DIR)
    print(f"\n====================")
    print(f"[eval] {tag} | pred_dir={pred_dir}")
    print(f"[eval] Found {len(pairs)} paired volumes.")
    print(f"[eval] Parallel={PARALLEL} workers={NUM_WORKERS}")
    print(f"[params] BASE_THR={BASE_THR} | "
          f"T_LOW={T_LOW} T_HIGH={T_HIGH} PRUNE={PRUNE_BELOW} | "
          f"MIN={MIN_SIZE} HOLE={HOLE_SIZE} CLOSE2D={CLOSE2D_RADIUS} | "
          f"LEGACY min={LEGACY_MIN_SIZE} closeR={LEGACY_CLOSING_RADIUS}")

    tasks = [(vid, str(pred_path), str(gt_path)) for vid, pred_path, gt_path in pairs]

    # ---------- Default single-run (baseline/legacy/advanced with global params) ----------
    def run_single() -> List[VolumeResult]:
        results: List[VolumeResult] = []
        if PARALLEL and NUM_WORKERS > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from tqdm import tqdm
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
                futs = [ex.submit(_process_one, t) for t in tasks]
                for fut in tqdm(as_completed(futs), total=len(futs), desc=f"Scoring {tag}"):
                    r = fut.result()
                    results.append(r)
                    if PRINT_PER_VOLUME:
                        print(
                            f"{r.vid}: base={r.baseline.score:.4f} | "
                            f"legacy={r.legacy.score:.4f} | "
                            f"adv={r.advanced.score:.4f} | "
                            f"Δadv-base={r.advanced.score - r.baseline.score:+.4f}"
                        )
        else:
            for t in tasks:
                r = _process_one(t)
                results.append(r)
                if PRINT_PER_VOLUME:
                    print(
                        f"{r.vid}: base={r.baseline.score:.4f} | "
                        f"legacy={r.legacy.score:.4f} | "
                        f"adv={r.advanced.score:.4f} | "
                        f"Δadv-base={r.advanced.score - r.baseline.score:+.4f}"
                    )
        return results

    # ---------- Mini sweep for prob_u8 ----------
    def run_cand_sweep() -> None:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        # We'll compute baseline once (cfg doesn't affect baseline/legacy)
        base_results = run_single()  # prints per-volume if enabled
        base_results.sort(key=lambda x: x.vid)

        base_avg = _avg([r.baseline.score for r in base_results])
        legacy_avg = _avg([r.legacy.score for r in base_results])

        print("\n--- Mini sweep on Advanced (prob_u8) ---")
        print("Candidates (t_low, t_high, prune_below):")
        for c in CAND:
            print(" ", c)

        sweep_rows = []
        # Do NOT spam per-volume during sweep
        global PRINT_PER_VOLUME
        old_ppv = PRINT_PER_VOLUME
        PRINT_PER_VOLUME = False

        try:
            for cfg in CAND:
                results_cfg: List[VolumeResult] = []
                if PARALLEL and NUM_WORKERS > 1:
                    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
                        futs = [ex.submit(_process_one_with_cfg, t, cfg) for t in tasks]
                        for fut in tqdm(as_completed(futs), total=len(futs),
                                        desc=f"cfg tl={cfg[0]:.2f} th={cfg[1]:.2f} pr={cfg[2]}"):
                            results_cfg.append(fut.result())
                else:
                    for t in tasks:
                        results_cfg.append(_process_one_with_cfg(t, cfg))

                adv_avg = _avg([r.advanced.score for r in results_cfg])
                sweep_rows.append((adv_avg, cfg))
                print(f"cfg={cfg} | adv_avg={adv_avg:.5f} | Δadv-base={adv_avg - base_avg:+.5f}")

        finally:
            PRINT_PER_VOLUME = old_ppv

        sweep_rows.sort(key=lambda x: x[0], reverse=True)
        print("\n=== Averages (fixed baseline/legacy + best advanced cfg) ===")
        print(f"Baseline : {base_avg:.5f}")
        print(f"Legacy   : {legacy_avg:.5f}  (Δ {legacy_avg - base_avg:+.5f})")

        best_adv, best_cfg = sweep_rows[0]
        print(f"Advanced(best cfg={best_cfg}) : {best_adv:.5f}  (Δ {best_adv - base_avg:+.5f})")

        print("\nTop-3 cfg:")
        for i, (adv_avg, cfg) in enumerate(sweep_rows[:3], 1):
            print(f" {i}. cfg={cfg} | adv_avg={adv_avg:.5f} | Δ {adv_avg - base_avg:+.5f}")

    # If prob_u8 -> run sweep, else just single
    if tag == "prob_u8":
        run_cand_sweep()
        return

    # default behavior for other tags
    results = run_single()
    results.sort(key=lambda x: x.vid)

    base_scores = [r.baseline.score for r in results]
    legacy_scores = [r.legacy.score for r in results]
    adv_scores = [r.advanced.score for r in results]

    print("\n=== Averages ===")
    print(f"Baseline : {_avg(base_scores):.5f}")
    print(f"Legacy   : {_avg(legacy_scores):.5f}  (Δ {_avg(legacy_scores) - _avg(base_scores):+.5f})")
    print(f"Advanced : {_avg(adv_scores):.5f}  (Δ {_avg(adv_scores) - _avg(base_scores):+.5f})")


def main() -> int:
    if not GT_DIR.exists():
        raise FileNotFoundError(f"GT_DIR not found: {GT_DIR.resolve()}")

    for tag, pdir in PRED_DIRS:
        if not pdir.exists():
            raise FileNotFoundError(f"Pred dir not found: {pdir.resolve()}")
        evaluate_pred_dir(tag, pdir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
