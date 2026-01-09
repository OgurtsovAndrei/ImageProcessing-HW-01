#!/usr/bin/env python3
"""postproc_sweep_cpu_full.py

Parallelizes across parameter sets (configs) using ProcessPoolExecutor so CPU stays busy even
when PH1_EVAL_N_VOLUMES is small.

- Each worker process loads + caches volumes once (initializer).
- Each task evaluates 1 config over K volumes sequentially.
- Avoids "only 8 tasks" issue from per-volume threading.

⚠️ RAM note: volumes/caches are duplicated per process. If volumes are big, start with NUM_WORKERS=6..8.

Edit CONSTANTS below (no CLI params).
"""

from __future__ import annotations

import csv
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

# Prevent oversubscription inside SciPy/NumPy kernels when using many processes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# =========================
# CONSTANTS (EDIT THESE)
# =========================
PRED_DIR = Path("data/validate-post-process/val_probs_u8_no_post")
GT_DIR   = Path("data/validate-post-process/labels")

TAU = 2.0
SPACING = (1.0, 1.0, 1.0)

VOI_ALPHA = 0.3

W_TOPO = 0.30
W_SURF = 0.35
W_VOI  = 0.35

PH1_EVAL_N_VOLUMES = 8
PH2_EVAL_N_VOLUMES = 8

# Processes (not threads). Start 6-8 if volumes are large.
NUM_WORKERS = 16

PH1_T_LOW  = [0.20, 0.25, 0.30, 0.35, 0.40]
PH1_T_GAP  = [0.25, 0.35, 0.45]
PH1_PRUNE  = ["none", "t_low"]
PH1_CLOSE2D = [0, 1, 2]
PH1_MIN_SIZE = [200, 500, 1000]
PH1_HOLE_SIZE = [0, 500, 1000, 2000]
PH1_TOPK_CC = [0, 1, 2]

TOPK_PH1 = 20

REFINE_T_STEP = 0.05
REFINE_CLOSE2D_DELTA = [-1, 0, 1]
REFINE_SCALE = [0.5, 1.0, 2.0]
REFINE_EXTRA = [0, 200]

TOPN_PRINT = 20
SAVE_CSV = Path("postproc_sweep_results.csv")

# =========================
# IO helpers
# =========================
def _read_volume(path: Path) -> np.ndarray:
    suf = path.suffix.lower()
    if suf in [".tif", ".tiff"]:
        try:
            import tifffile
        except ImportError as e:
            raise RuntimeError("tifffile is required: pip install tifffile") from e
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
            f"No matching volume ids found. pred={len(preds)} gt={len(gts)}; expected same stems."
        )
    missing_pred = sorted(set(gts.keys()) - set(preds.keys()))
    missing_gt = sorted(set(preds.keys()) - set(gts.keys()))
    if missing_pred:
        warnings.warn(f"{len(missing_pred)} GT volumes have no prediction match (ignored): {missing_pred[:10]}")
    if missing_gt:
        warnings.warn(f"{len(missing_gt)} predictions have no GT match (ignored): {missing_gt[:10]}")
    return [(vid, preds[vid], gts[vid]) for vid in common]

# =========================
# Core ops
# =========================
def _label_3d(mask: np.ndarray, connectivity: int = 26) -> Tuple[np.ndarray, int]:
    from scipy.ndimage import label, generate_binary_structure
    mask = mask.astype(bool, copy=False)
    conn = 3 if connectivity == 26 else 1
    structure = generate_binary_structure(rank=3, connectivity=conn)
    lbl, n = label(mask, structure=structure)
    return lbl, int(n)

def _extract_surface_6n(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion, generate_binary_structure
    mask = mask.astype(bool, copy=False)
    struct = generate_binary_structure(3, 1)
    er = binary_erosion(mask, structure=struct, border_value=0)
    return mask & (~er)

def _dt_to_surface(surface: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    return distance_transform_edt(~surface, sampling=SPACING)

def hysteresis_from_probs(prob_u8: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    lo = int(round(max(0.0, min(1.0, t_low)) * 255.0))
    hi = int(round(max(0.0, min(1.0, t_high)) * 255.0))

    weak = prob_u8 >= lo
    strong = prob_u8 >= hi
    if not weak.any() or not strong.any():
        return np.zeros_like(prob_u8, dtype=bool)

    lbl, n = _label_3d(weak, connectivity=26)
    if n == 0:
        return np.zeros_like(prob_u8, dtype=bool)

    strong_ids = np.unique(lbl[strong])
    strong_ids = strong_ids[strong_ids != 0]
    if strong_ids.size == 0:
        return np.zeros_like(prob_u8, dtype=bool)

    return np.isin(lbl, strong_ids)

def post_process_from_prob_u8(
    prob_u8: np.ndarray,
    ignore_mask: Optional[np.ndarray],
    t_low: float,
    t_high: float,
    prune_mode: str,
    close2d_radius: int,
    min_size: int,
    hole_size: int,
    topk_components: int,
) -> np.ndarray:
    from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing, disk

    ign = ignore_mask.astype(bool, copy=False) if ignore_mask is not None else None
    m = hysteresis_from_probs(prob_u8, t_low=t_low, t_high=t_high)

    if ign is not None:
        m = m & (~ign)

    if prune_mode != "none":
        if prune_mode == "t_low":
            pr = int(round(max(0.0, min(1.0, t_low)) * 255.0))
        else:
            raise ValueError(f"Unknown prune_mode: {prune_mode}")
        above = (prob_u8 >= pr)
        lbl, n = _label_3d(m, connectivity=26)
        if n > 0:
            keep_ids = np.unique(lbl[above & m])
            keep_ids = keep_ids[keep_ids != 0]
            m = np.isin(lbl, keep_ids)

    if close2d_radius and close2d_radius > 0:
        se = disk(int(close2d_radius))
        out = m.copy()
        for z in range(out.shape[0]):
            out[z] = binary_closing(out[z], se)
        m = out

    if min_size and min_size > 0:
        m = remove_small_objects(m, min_size=int(min_size), connectivity=3)

    if hole_size and hole_size > 0:
        m = remove_small_holes(m, area_threshold=int(hole_size), connectivity=3)

    if topk_components and topk_components > 0:
        lbl, n = _label_3d(m, connectivity=26)
        if n > 0:
            counts = np.bincount(lbl.reshape(-1))
            comp_ids = np.argsort(counts[1:])[::-1] + 1
            keep_ids = comp_ids[: int(topk_components)]
            m = np.isin(lbl, keep_ids)

    if ign is not None:
        m[ign] = False

    return m.astype(np.uint8)

# =========================
# Metrics (with caching)
# =========================
@dataclass
class MetricBreakdown:
    topo: float
    surface_dice: float
    voi: float
    score: float

def surface_dice_cached(pred: np.ndarray, gt_surface: np.ndarray, dt_to_gt_surface: np.ndarray, tau: float = TAU) -> float:
    from scipy.ndimage import distance_transform_edt

    pred = pred.astype(bool, copy=False)

    if pred.sum() == 0 and gt_surface.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt_surface.sum() == 0:
        return 0.0

    pred_surface = _extract_surface_6n(pred)
    if pred_surface.sum() == 0 and gt_surface.sum() == 0:
        return 1.0
    if pred_surface.sum() == 0 or gt_surface.sum() == 0:
        return 0.0

    pred_match = float((dt_to_gt_surface[pred_surface] <= tau).mean())

    dt_to_pred = distance_transform_edt(~pred_surface, sampling=SPACING)
    gt_match = float((dt_to_pred[gt_surface] <= tau).mean())

    return float(0.5 * (pred_match + gt_match))

def voi_score_fast(pred: np.ndarray, gt_fg: np.ndarray, gt_lbl: np.ndarray, alpha: float = VOI_ALPHA) -> float:
    pred = pred.astype(bool, copy=False)
    gt_fg = gt_fg.astype(bool, copy=False)

    if pred.sum() == 0 and gt_fg.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt_fg.sum() == 0:
        return 0.0

    union = pred | gt_fg
    N = int(union.sum())
    if N == 0:
        return 1.0

    pr_lbl, _ = _label_3d(pred, connectivity=26)

    a = gt_lbl[union].astype(np.int64, copy=False)
    b = pr_lbl[union].astype(np.int64, copy=False)

    bmax = int(b.max())
    key = a * (bmax + 1) + b
    keys, counts = np.unique(key, return_counts=True)

    a_keys = keys // (bmax + 1)
    b_keys = keys % (bmax + 1)

    p_ij = counts.astype(np.float64) / float(N)

    max_a = int(a.max())
    max_b = int(b.max())
    p_i = np.bincount(a_keys, weights=p_ij, minlength=max_a + 1)
    q_j = np.bincount(b_keys, weights=p_ij, minlength=max_b + 1)

    eps = 1e-12
    pij = np.maximum(p_ij, eps)
    q = np.maximum(q_j[b_keys], eps)
    p = np.maximum(p_i[a_keys], eps)

    H_gt_given_pr = -float(np.sum(p_ij * np.log2(pij / q)))
    H_pr_given_gt = -float(np.sum(p_ij * np.log2(pij / p)))

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

def topo_score_approx(pred: np.ndarray, gt_fg: np.ndarray) -> float:
    pred = pred.astype(bool, copy=False)
    gt_fg = gt_fg.astype(bool, copy=False)

    if pred.sum() == 0 and gt_fg.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt_fg.sum() == 0:
        return 0.0

    b0_p, b1_p, b2_p, _ = _betti_numbers_approx(pred)
    b0_g, b1_g, b2_g, _ = _betti_numbers_approx(gt_fg)

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
        parts.append(f0); weights.append(0.34)
    if f1v is not None:
        parts.append(f1v); weights.append(0.33)
    if f2 is not None:
        parts.append(f2); weights.append(0.33)

    if not parts:
        return 1.0

    w = np.array(weights, dtype=np.float32)
    w /= w.sum()
    return float((np.array(parts, dtype=np.float32) * w).sum())

@dataclass
class VolumeCache:
    volume_id: str
    prob_u8: np.ndarray
    gt: np.ndarray
    ignore: np.ndarray
    gt_fg: np.ndarray
    gt_surface: np.ndarray
    dt_to_gt_surface: np.ndarray
    gt_lbl: np.ndarray

def compute_metrics_cached(pred_bin: np.ndarray, vc: VolumeCache) -> MetricBreakdown:
    pred = pred_bin.astype(bool, copy=False)
    pred[vc.ignore] = False

    s = surface_dice_cached(pred, vc.gt_surface, vc.dt_to_gt_surface, tau=TAU)
    v = voi_score_fast(pred, vc.gt_fg, vc.gt_lbl, alpha=VOI_ALPHA)
    t = topo_score_approx(pred, vc.gt_fg)
    score = W_TOPO * t + W_SURF * s + W_VOI * v
    return MetricBreakdown(topo=float(t), surface_dice=float(s), voi=float(v), score=float(score))

def _load_all(pred_dir: Path, gt_dir: Path) -> List[VolumeCache]:
    pairs = _pair_pred_gt(pred_dir, gt_dir)
    out: List[VolumeCache] = []
    seen_vals: Set[int] = set()

    for vid, pred_path, gt_path in pairs:
        pred_raw = _read_volume(pred_path)
        gt = _read_volume(gt_path).astype(np.uint8, copy=False)
        if pred_raw.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {vid}: pred={pred_raw.shape}, gt={gt.shape}")

        if pred_raw.dtype != np.uint8:
            if np.issubdtype(pred_raw.dtype, np.floating):
                pred_u8 = (np.clip(pred_raw, 0, 1) * 255.0).round().astype(np.uint8)
            else:
                pred_u8 = pred_raw.astype(np.uint8)
        else:
            pred_u8 = pred_raw

        u = np.unique(pred_u8)
        head = u[:8]
        tail = u[-8:] if u.size >= 8 else u
        for x in np.unique(np.concatenate([head, tail])):
            seen_vals.add(int(x))

        ignore = (gt == 2)
        gt_fg = (gt == 1) & (~ignore)

        gt_surface = _extract_surface_6n(gt_fg)
        dt_to_gt_surface = _dt_to_surface(gt_surface)
        gt_lbl, _ = _label_3d(gt_fg, connectivity=26)

        out.append(VolumeCache(
            volume_id=vid,
            prob_u8=pred_u8,
            gt=gt,
            ignore=ignore,
            gt_fg=gt_fg,
            gt_surface=gt_surface,
            dt_to_gt_surface=dt_to_gt_surface,
            gt_lbl=gt_lbl,
        ))

    out.sort(key=lambda x: x.volume_id)
    print(f"[info] Example pred unique values (head+tail): {sorted(list(seen_vals))[:24]} ...")
    return out

# =========================
# Sweep types + worker globals
# =========================
@dataclass(frozen=True)
class Config:
    t_low: float
    t_high: float
    prune_mode: str
    close2d_radius: int
    min_size: int
    hole_size: int
    topk_components: int

@dataclass
class SweepRow:
    avg_score: float
    avg_topo: float
    avg_surface: float
    avg_voi: float
    cfg: Config

def _fmt_cfg(cfg: Config) -> str:
    return (f"t_low={cfg.t_low:.2f}, t_high={cfg.t_high:.2f}, prune={cfg.prune_mode}, "
            f"close2d={cfg.close2d_radius}, min={cfg.min_size}, hole={cfg.hole_size}, topk_cc={cfg.topk_components}")

_G_VOLS_PH1: List[VolumeCache] = []
_G_VOLS_PH2: List[VolumeCache] = []

def _worker_init(pred_dir_str: str, gt_dir_str: str, ph1_n: int, ph2_n_or_neg1: int) -> None:
    global _G_VOLS_PH1, _G_VOLS_PH2
    vols = _load_all(Path(pred_dir_str), Path(gt_dir_str))
    _G_VOLS_PH1 = vols[: min(len(vols), ph1_n)] if ph1_n > 0 else vols
    _G_VOLS_PH2 = vols if ph2_n_or_neg1 < 0 else vols[: min(len(vols), ph2_n_or_neg1)]

def _eval_config_in_worker(args: Tuple[int, Config]) -> SweepRow:
    phase, cfg = args
    vols = _G_VOLS_PH1 if phase == 1 else _G_VOLS_PH2

    mets: List[MetricBreakdown] = []
    for vc in vols:
        mask_u8 = post_process_from_prob_u8(
            vc.prob_u8,
            ignore_mask=vc.ignore,
            t_low=cfg.t_low,
            t_high=cfg.t_high,
            prune_mode=cfg.prune_mode,
            close2d_radius=cfg.close2d_radius,
            min_size=cfg.min_size,
            hole_size=cfg.hole_size,
            topk_components=cfg.topk_components,
        )
        mets.append(compute_metrics_cached(mask_u8 > 0, vc))

    avg_score = float(np.mean([m.score for m in mets]))
    avg_topo  = float(np.mean([m.topo for m in mets]))
    avg_surf  = float(np.mean([m.surface_dice for m in mets]))
    avg_voi   = float(np.mean([m.voi for m in mets]))
    return SweepRow(avg_score=avg_score, avg_topo=avg_topo, avg_surface=avg_surf, avg_voi=avg_voi, cfg=cfg)

# =========================
# Config generation
# =========================
def _baseline(vols: List[VolumeCache]) -> SweepRow:
    mets = []
    for vc in vols:
        pred = (vc.prob_u8 >= 128)
        mets.append(compute_metrics_cached(pred, vc))
    return SweepRow(
        avg_score=float(np.mean([m.score for m in mets])),
        avg_topo=float(np.mean([m.topo for m in mets])),
        avg_surface=float(np.mean([m.surface_dice for m in mets])),
        avg_voi=float(np.mean([m.voi for m in mets])),
        cfg=Config(0.5, 0.5, "none", 0, 0, 0, 0),
    )

def _phase1_configs() -> List[Config]:
    cfgs: List[Config] = []
    for tl in PH1_T_LOW:
        for gap in PH1_T_GAP:
            th = min(0.95, tl + gap)
            for pr in PH1_PRUNE:
                for c2 in PH1_CLOSE2D:
                    for ms in PH1_MIN_SIZE:
                        for hs in PH1_HOLE_SIZE:
                            for k in PH1_TOPK_CC:
                                cfgs.append(Config(float(tl), float(th), pr, int(c2), int(ms), int(hs), int(k)))
    return cfgs

def _neighbors(cfg: Config) -> Set[Config]:
    def clamp01(x: float) -> float:
        return float(max(0.01, min(0.95, x)))

    tls = [clamp01(cfg.t_low + d) for d in (-REFINE_T_STEP, 0.0, REFINE_T_STEP)]
    ths = [clamp01(cfg.t_high + d) for d in (-REFINE_T_STEP, 0.0, REFINE_T_STEP)]
    prs = {cfg.prune_mode, "none", "t_low"}

    c2s = {max(0, cfg.close2d_radius + d) for d in REFINE_CLOSE2D_DELTA}
    c2s = {min(4, x) for x in c2s}

    ms_vals: Set[int] = set()
    hs_vals: Set[int] = set()
    for s in REFINE_SCALE:
        for a in REFINE_EXTRA:
            ms_vals.add(max(0, int(round(cfg.min_size * s)) + a))
            ms_vals.add(max(0, int(round(cfg.min_size * s)) - a))
            hs_vals.add(max(0, int(round(cfg.hole_size * s)) + a))
            hs_vals.add(max(0, int(round(cfg.hole_size * s)) - a))

    ms_vals = {min(50000, x) for x in ms_vals}
    hs_vals = {min(50000, x) for x in hs_vals}

    ks = {cfg.topk_components, 0, 1, 2, 3, 5}
    ks = {min(20, max(0, int(x))) for x in ks}

    out: Set[Config] = set()
    for tl in tls:
        for th in ths:
            if th <= tl:
                continue
            for pr in prs:
                for c2 in c2s:
                    for ms in ms_vals:
                        for hs in hs_vals:
                            for k in ks:
                                out.add(Config(float(tl), float(th), pr, int(c2), int(ms), int(hs), int(k)))
    return out

# =========================
# Main
# =========================
def main() -> int:
    t0 = time.time()
    if not PRED_DIR.exists():
        raise FileNotFoundError(f"pred_dir not found: {PRED_DIR.resolve()}")
    if not GT_DIR.exists():
        raise FileNotFoundError(f"gt_dir not found: {GT_DIR.resolve()}")
    print(f"[info] Workers (processes): {NUM_WORKERS}")

    vols_main = _load_all(PRED_DIR, GT_DIR)
    print(f"[info] Loaded {len(vols_main)} volumes.")
    base = _baseline(vols_main)
    print(f"[baseline avg] score={base.avg_score:.5f}, topo={base.avg_topo:.5f}, surf={base.avg_surface:.5f}, voi={base.avg_voi:.5f}")

    cfgs1 = _phase1_configs()
    print(f"\n[phase 1] Evaluating {len(cfgs1)} configs on {min(PH1_EVAL_N_VOLUMES, len(vols_main))} volumes...")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    ph2_n = -1 if (PH2_EVAL_N_VOLUMES is None) else int(PH2_EVAL_N_VOLUMES)

    rows1: List[SweepRow] = []
    best1: Optional[SweepRow] = None

    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=_worker_init,
        initargs=(str(PRED_DIR), str(GT_DIR), int(PH1_EVAL_N_VOLUMES), int(ph2_n)),
    ) as ex:
        futs = [ex.submit(_eval_config_in_worker, (1, cfg)) for cfg in cfgs1]
        done = 0
        for fut in as_completed(futs):
            row = fut.result()
            rows1.append(row)
            if best1 is None or row.avg_score >= best1.avg_score:
                best1 = row
            done += 1
            if done % 50 == 0 or done == len(cfgs1):
                assert best1 is not None
                print(f"  {done:>4}/{len(cfgs1)} best score={best1.avg_score:.5f} with {_fmt_cfg(best1.cfg)}")

    rows1.sort(key=lambda r: r.avg_score, reverse=True)
    top1 = rows1[:TOPK_PH1]
    print(f"\n[phase 1] Top {len(top1)}:")
    for i, r in enumerate(top1, 1):
        print(f"  {i:>2}. score={r.avg_score:.5f} | {_fmt_cfg(r.cfg)}")

    cand2: Set[Config] = set()
    for r in top1:
        cand2 |= _neighbors(r.cfg)
    cand2 |= {r.cfg for r in top1}
    cfgs2 = list(cand2)

    n_ph2 = len(vols_main) if (PH2_EVAL_N_VOLUMES is None) else min(len(vols_main), int(PH2_EVAL_N_VOLUMES))
    print(f"\n[phase 2] Evaluating {len(cfgs2)} refined configs on {n_ph2} volumes...")

    rows2: List[SweepRow] = []
    best2: Optional[SweepRow] = None

    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=_worker_init,
        initargs=(str(PRED_DIR), str(GT_DIR), int(PH1_EVAL_N_VOLUMES), int(ph2_n)),
    ) as ex:
        futs = [ex.submit(_eval_config_in_worker, (2, cfg)) for cfg in cfgs2]
        done = 0
        for fut in as_completed(futs):
            row = fut.result()
            rows2.append(row)
            if best2 is None or row.avg_score >= best2.avg_score:
                best2 = row
            done += 1
            if done % 100 == 0 or done == len(cfgs2):
                assert best2 is not None
                print(f"  {done:>4}/{len(cfgs2)} best score={best2.avg_score:.5f} (Δ {best2.avg_score - base.avg_score:+.5f}) with {_fmt_cfg(best2.cfg)}")

    rows2.sort(key=lambda r: r.avg_score, reverse=True)

    print(f"\n=== Top {TOPN_PRINT} configs (phase 2) ===")
    for i, r in enumerate(rows2[:TOPN_PRINT], 1):
        print(
            f"{i:>2}. score={r.avg_score:.5f} (Δ {r.avg_score - base.avg_score:+.5f}) | "
            f"topo={r.avg_topo:.5f}, surf={r.avg_surface:.5f}, voi={r.avg_voi:.5f} | "
            f"{_fmt_cfg(r.cfg)}"
        )

    SAVE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SAVE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "rank",
            "avg_score", "delta_vs_baseline",
            "avg_topo", "avg_surface_dice", "avg_voi",
            "t_low", "t_high", "prune_mode", "close2d_radius", "min_size", "hole_size", "topk_components",
        ])
        for i, r in enumerate(rows2, 1):
            w.writerow([
                i,
                f"{r.avg_score:.6f}", f"{(r.avg_score - base.avg_score):+.6f}",
                f"{r.avg_topo:.6f}", f"{r.avg_surface:.6f}", f"{r.avg_voi:.6f}",
                f"{r.cfg.t_low:.2f}", f"{r.cfg.t_high:.2f}", r.cfg.prune_mode,
                r.cfg.close2d_radius, r.cfg.min_size, r.cfg.hole_size, r.cfg.topk_components,
            ])

    print(f"\n[info] Saved sweep results to: {SAVE_CSV.resolve()}")
    print(f"[info] Total time: {time.time() - t0:.1f}s")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
