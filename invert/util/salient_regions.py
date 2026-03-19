from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr


def _get_row_value(row: Mapping | object, key: str, default=None):
    if isinstance(row, Mapping):
        return row.get(key, default)
    return getattr(row, key, default)


def _as_float_array(values) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.astype(float, copy=False)


def _as_index_list(values) -> list[np.ndarray]:
    if values is None:
        return []
    if isinstance(values, np.ndarray) and values.ndim == 1:
        return [values.astype(int, copy=False)]
    if not isinstance(values, Sequence):
        return []
    out: list[np.ndarray] = []
    for item in values:
        arr = np.asarray(item, dtype=int)
        if arr.size == 0:
            continue
        out.append(arr.reshape(-1).astype(int, copy=False))
    return out


def normalize_saliency(values: np.ndarray, *, mode: str = "sum") -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = np.clip(arr, a_min=0.0, a_max=None)
    if arr.size == 0:
        return arr
    if mode == "sum":
        total = float(arr.sum())
        return arr / total if total > 0 else np.zeros_like(arr)
    if mode == "max":
        peak = float(arr.max())
        return arr / peak if peak > 0 else np.zeros_like(arr)
    raise ValueError(f"Unknown normalization mode: {mode!r}")


def dominant_source_indices(
    sim_row,
    *,
    amplitude_ratio: float = 0.6,
    cumulative_mass: float = 0.8,
) -> np.ndarray:
    amps = _as_float_array(_get_row_value(sim_row, "amplitudes"))
    if amps.size == 0:
        return np.asarray([], dtype=int)

    amps = np.clip(amps, a_min=0.0, a_max=None)
    order = np.argsort(amps)[::-1]
    if order.size == 0:
        return np.asarray([], dtype=int)

    max_amp = float(amps[order[0]])
    total_amp = float(amps.sum())
    keep: list[int] = []
    mass = 0.0

    for idx in order:
        amp = float(amps[idx])
        enough_mass = (mass / total_amp) >= float(cumulative_mass) if total_amp > 0 else True
        salient_enough = amp >= float(amplitude_ratio) * max(max_amp, 1e-12)
        if not keep or salient_enough or not enough_mass:
            keep.append(int(idx))
            mass += amp
        else:
            break

    return np.asarray(keep, dtype=int)


def dominant_centers(
    sim_row,
    *,
    amplitude_ratio: float = 0.6,
    cumulative_mass: float = 0.8,
) -> np.ndarray:
    centers = _as_float_array(_get_row_value(sim_row, "centers")).astype(int, copy=False)
    if centers.size == 0:
        return np.asarray([], dtype=int)
    keep = dominant_source_indices(
        sim_row,
        amplitude_ratio=amplitude_ratio,
        cumulative_mass=cumulative_mass,
    )
    keep = keep[keep < centers.size]
    return centers[keep]


def salient_patch_target(
    sim_row,
    n_vertices: int,
    *,
    amplitude_ratio: float = 0.6,
    cumulative_mass: float = 0.8,
) -> np.ndarray:
    target = np.zeros(int(n_vertices), dtype=float)
    amplitudes = _as_float_array(_get_row_value(sim_row, "amplitudes"))
    source_vertices = _as_index_list(_get_row_value(sim_row, "source_vertices"))

    keep = dominant_source_indices(
        sim_row,
        amplitude_ratio=amplitude_ratio,
        cumulative_mass=cumulative_mass,
    )
    if keep.size == 0 or len(source_vertices) == 0:
        return target

    weights = amplitudes[keep] if amplitudes.size else np.ones(keep.size, dtype=float)
    weights = normalize_saliency(weights, mode="sum")
    for weight, idx in zip(weights, keep, strict=False):
        if idx >= len(source_vertices):
            continue
        region = source_vertices[int(idx)]
        if region.size == 0:
            continue
        target[region] += float(weight)
    return normalize_saliency(target, mode="sum")


def salient_center_target(
    sim_row,
    n_vertices: int,
    *,
    amplitude_ratio: float = 0.6,
    cumulative_mass: float = 0.8,
) -> np.ndarray:
    target = np.zeros(int(n_vertices), dtype=float)
    centers = _as_float_array(_get_row_value(sim_row, "centers")).astype(int, copy=False)
    amplitudes = _as_float_array(_get_row_value(sim_row, "amplitudes"))
    keep = dominant_source_indices(
        sim_row,
        amplitude_ratio=amplitude_ratio,
        cumulative_mass=cumulative_mass,
    )
    if keep.size == 0 or centers.size == 0:
        return target

    keep = keep[keep < centers.size]
    if keep.size == 0:
        return target
    weights = amplitudes[keep] if amplitudes.size else np.ones(keep.size, dtype=float)
    weights = normalize_saliency(weights, mode="sum")
    target[centers[keep]] = weights
    return normalize_saliency(target, mode="sum")


def salient_region_metrics(
    sim_row,
    saliency: np.ndarray,
    pos: np.ndarray,
    *,
    amplitude_ratio: float = 0.6,
    cumulative_mass: float = 0.8,
    center_hit_radius: float = 0.02,
) -> dict[str, float]:
    saliency = normalize_saliency(np.asarray(saliency, dtype=float), mode="sum")
    n_vertices = int(saliency.shape[0])

    patch_target = salient_patch_target(
        sim_row,
        n_vertices,
        amplitude_ratio=amplitude_ratio,
        cumulative_mass=cumulative_mass,
    )
    true_centers = dominant_centers(
        sim_row,
        amplitude_ratio=amplitude_ratio,
        cumulative_mass=cumulative_mass,
    )

    target_mask = patch_target > 0
    target_idx = np.flatnonzero(target_mask)
    if target_idx.size == 0:
        return {
            "unsigned_corr": float("nan"),
            "mass_in_target": float("nan"),
            "topk_recall": float("nan"),
            "mean_center_error": float("nan"),
            "center_hit_20mm": float("nan"),
        }

    if np.allclose(saliency, saliency[0]) or np.allclose(patch_target, patch_target[0]):
        unsigned_corr = float("nan")
    else:
        unsigned_corr = abs(float(pearsonr(patch_target, saliency)[0]))

    total_mass = float(saliency.sum())
    mass_in_target = float(saliency[target_mask].sum() / total_mass) if total_mass > 0 else 0.0

    k = max(1, int(target_idx.size))
    top_idx = np.argsort(saliency)[::-1][:k]
    topk_recall = float(np.intersect1d(top_idx, target_idx).size / float(target_idx.size))

    if true_centers.size == 0:
        mean_center_error = float("nan")
        center_hit = float("nan")
    else:
        pred_centers = np.argsort(saliency)[::-1][: int(true_centers.size)]
        dist = cdist(pos[true_centers], pos[pred_centers])
        row_ind, col_ind = linear_sum_assignment(dist)
        matched = dist[row_ind, col_ind]
        mean_center_error = float(np.mean(matched))
        center_hit = float(np.mean(matched <= float(center_hit_radius)))

    return {
        "unsigned_corr": unsigned_corr,
        "mass_in_target": mass_in_target,
        "topk_recall": topk_recall,
        "mean_center_error": mean_center_error,
        "center_hit_20mm": center_hit,
    }
