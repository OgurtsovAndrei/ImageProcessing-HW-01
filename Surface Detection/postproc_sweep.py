#!/usr/bin/env python3
"""
postproc_sweep_threads.py

Memory-safe sweep:
- Loads volumes once (shared memory)
- Parallelizes per-config using ThreadPoolExecutor across volumes
- Adds progress bars (tqdm)
- Avoids macOS spawn/pickle issues and avoids OOM from ProcessPool duplicating caches

Expected folders:
  PRED_DIR: uint8 prob volumes (.tif/.tiff/.npy/.npz), values 0..255
  GT_DIR:   GT volumes with labels {0=bg, 1=fg, 2=ignore}
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

# prevent oversubscription with BLAS when using ThreadPool
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# =========================
# CONSTANTS (EDIT THESE)
# =========================
PRED_DIR = Path("data/validate-post-process/val_probs_u8_no_post")
GT_DIR = Path("data/validate-post-process/labels")

TAU = 2.0
SPACING = (1.0, 1.0, 1.0)
VOI_ALPHA = 0.3

W_TOPO = 0.30
W_SURF = 0.35
W_VOI  = 0.35

PH1_EVAL_N_VOLUMES = 8
PH2_EVAL_N_VOLUMES = None  # None => all

NUM_WORKERS = 16  # threads

# Phase 1: 128 configs
PH1_T_LOW = [0.20, 0.25, 0.30, 0.35]
PH1_T_GAP = [0.25, 0.35]
PH1_PRUNE = ["none", "t_low"]
PH1_CLOSE2D = [0, 1]
PH1_MIN_SIZE = [200, 500]
PH1_HOLE_SIZE = [0, 1000]
PH1_TOPK_CC = [0]

TOPK_PH1 = 5
MAX_PHASE2 = 160

TOPN_PRINT = 20
SAVE_CSV = Path("postproc_sweep_results.csv")


# =========================
# IO
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
            f"No matching volume ids found.\n"
            f"pred_dir={pred_dir} contains {len(preds)} volumes\n"
            f"gt_dir={gt_dir} contains {len(gts)} volumes\n"
            f"Expected matching stems."
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
    dt = distance_transform_edt(~surface, sampling=SPACING)
    return dt.astype(np.float32, copy=False)


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
# Metrics
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
    return MetricBreakdown(float(t), float(s), float(v), float(score))


# =========================
# Config generation
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
    return (
        f"t_low={cfg.t_low:.2f}, t_high={cfg.t_high:.2f}, prune={cfg.prune_mode}, "
        f"close2d={cfg.close2d_radius}, min={cfg.min_size}, hole={cfg.hole_size}, topk_cc={cfg.topk_components}"
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

    tls = [clamp01(cfg.t_low + d) for d in (-0.05, -0.025, 0.0, 0.025)]
    ths = [clamp01(cfg.t_high + d) for d in (-0.05, 0.0, 0.05)]

    prs = [cfg.prune_mode]
    c2s = [cfg.close2d_radius]
    ms_vals = sorted({max(0, int(cfg.min_size)), max(0, int(round(cfg.min_size * 0.5)))})
    hs_vals = sorted({max(0, int(cfg.hole_size)), max(0, int(round(cfg.hole_size * 2.0)))})
    ks = [0, 1]

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
# Eval
# =========================
def _eval_one_volume(vc: VolumeCache, cfg: Config) -> MetricBreakdown:
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
    return compute_metrics_cached(mask_u8 > 0, vc)


def _eval_config(vols: List[VolumeCache], cfg: Config, ex) -> SweepRow:
    mets = list(ex.map(lambda v: _eval_one_volume(v, cfg), vols))
    return SweepRow(
        avg_score=float(np.mean([m.score for m in mets])),
        avg_topo=float(np.mean([m.topo for m in mets])),
        avg_surface=float(np.mean([m.surface_dice for m in mets])),
        avg_voi=float(np.mean([m.voi for m in mets])),
        cfg=cfg,
    )


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
                prob_u8 = (np.clip(pred_raw, 0, 1) * 255.0).round().astype(np.uint8)
            else:
                prob_u8 = pred_raw.astype(np.uint8)
        else:
            prob_u8 = pred_raw

        u = np.unique(prob_u8)
        head = u[:8]
        tail = u[-8:] if u.size >= 8 else u
        for x in np.unique(np.concatenate([head, tail])):
            seen_vals.add(int(x))

        ignore = (gt == 2)
        gt_fg = (gt == 1) & (~ignore)

        gt_surface = _extract_surface_6n(gt_fg)
        dt_to_gt_surface = _dt_to_surface(gt_surface)          # float32
        gt_lbl, _ = _label_3d(gt_fg, connectivity=26)
        gt_lbl = gt_lbl.astype(np.int32, copy=False)           # int32 saves memory

        out.append(
            VolumeCache(
                volume_id=vid,
                prob_u8=prob_u8.astype(np.uint8, copy=False),
                ignore=ignore.astype(bool, copy=False),
                gt_fg=gt_fg.astype(bool, copy=False),
                gt_surface=gt_surface.astype(bool, copy=False),
                dt_to_gt_surface=dt_to_gt_surface,
                gt_lbl=gt_lbl,
            )
        )

    out.sort(key=lambda x: x.volume_id)
    print(f"[info] Example pred unique values (head+tail): {sorted(list(seen_vals))[:24]} ...")
    return out


def main() -> int:
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    t0 = time.time()

    if not PRED_DIR.exists():
        raise FileNotFoundError(f"pred_dir not found: {PRED_DIR.resolve()}")
    if not GT_DIR.exists():
        raise FileNotFoundError(f"gt_dir not found: {GT_DIR.resolve()}")

    vols_all = _load_all(PRED_DIR, GT_DIR)
    print(f"[info] Loaded {len(vols_all)} volumes.")
    base = _baseline(vols_all)
    print(f"[baseline avg] score={base.avg_score:.5f}, topo={base.avg_topo:.5f}, surf={base.avg_surface:.5f}, voi={base.avg_voi:.5f}")

    vols_ph1 = vols_all[: min(len(vols_all), PH1_EVAL_N_VOLUMES)] if PH1_EVAL_N_VOLUMES else vols_all
    vols_ph2 = vols_all if (PH2_EVAL_N_VOLUMES is None) else vols_all[: min(len(vols_all), int(PH2_EVAL_N_VOLUMES))]

    cfgs1 = _phase1_configs()
    print(f"\n[phase 1] Evaluating {len(cfgs1)} configs on {len(vols_ph1)} volumes...")

    rows1: List[SweepRow] = []
    best1: Optional[SweepRow] = None

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        for cfg in tqdm(cfgs1, desc="Phase 1 configs", total=len(cfgs1)):
            row = _eval_config(vols_ph1, cfg, ex)
            rows1.append(row)
            if best1 is None or row.avg_score >= best1.avg_score:
                best1 = row

    rows1.sort(key=lambda r: r.avg_score, reverse=True)
    top1 = rows1[: min(TOPK_PH1, len(rows1))]

    print(f"\n[phase 1] Top {len(top1)}:")
    for i, r in enumerate(top1, 1):
        print(f"  {i:>2}. score={r.avg_score:.5f} | {_fmt_cfg(r.cfg)}")

    cand2: Set[Config] = set(r.cfg for r in top1)
    for r in top1:
        cand2 |= _neighbors(r.cfg)

    cfgs2 = list(cand2)

    if len(cfgs2) > MAX_PHASE2:
        cfgs2 = sorted(
            cfgs2,
            key=lambda c: (round(c.t_low, 3), round(c.t_high, 3), c.min_size, c.hole_size, c.topk_components),
        )[:MAX_PHASE2]

    print(f"\n[phase 2] Evaluating {len(cfgs2)} refined configs on {len(vols_ph2)} volumes...")

    rows2: List[SweepRow] = []
    best2: Optional[SweepRow] = None

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        for cfg in tqdm(cfgs2, desc="Phase 2 configs", total=len(cfgs2)):
            row = _eval_config(vols_ph2, cfg, ex)
            rows2.append(row)
            if best2 is None or row.avg_score >= best2.avg_score:
                best2 = row

    rows2.sort(key=lambda r: r.avg_score, reverse=True)

    print(f"\n=== Top {TOPN_PRINT} configs (phase 2) ===")
    for i, r in enumerate(rows2[:TOPN_PRINT], 1):
        print(
            f"{i:>2}. score={r.avg_score:.5f} (Δ {r.avg_score - base.avg_score:+.5f}) | "
            f"topo={r.avg_topo:.5f}, surf={r.avg_surface:.5f}, voi={r.avg_voi:.5f} | {_fmt_cfg(r.cfg)}"
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
