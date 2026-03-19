"""Cache for optimized global parameters (conductivity, offset, stretch, rotation).

Stores and retrieves optimization results keyed by a hash of the sensor
covariance matrix.  This avoids re-running the expensive multi-minute
optimization when the same data is used again.

Usage::

    from invert.cortical_folding.cache import save_params, load_params

    # After optimization
    save_params(C_data, sigma_skull=ss, sigma_scalp=sc,
                offset=off, stretch=stretch, rotation=rot,
                co_deform=True, cost=cost)

    # In a new session
    params = load_params(C_data)
    if params is not None:
        ss = params["sigma_skull"]
        sc = params["sigma_scalp"]
        off = params["offset"]
        stretch = params["stretch"]
        rot = params["rotation"]
        co_deform = params["co_deform"]
"""

import hashlib
import os

import numpy as np

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def _cov_hash(C_data: np.ndarray) -> str:
    """Deterministic hash of a covariance matrix."""
    # Use the raw bytes of the float64 array, rounded to avoid noise
    rounded = np.round(C_data, decimals=12)
    return hashlib.sha256(rounded.tobytes()).hexdigest()[:16]


def save_params(
    C_data: np.ndarray,
    *,
    sigma_skull: float,
    offset: np.ndarray,
    stretch: np.ndarray,
    cost: float = np.nan,
    sigma_scalp: float = 0.3,
    rotation: np.ndarray | None = None,
    co_deform: bool = False,
    k: int = 10,
    extra: dict | None = None,
) -> str:
    """Save optimized parameters to the cache.

    Parameters
    ----------
    C_data : (n_ch, n_ch) array
        Sensor covariance used for optimization (used as cache key).
    sigma_skull : float
        Skull conductivity (S/m).
    offset : (3,) array
        Brain offset [x, y, z] in meters.
    stretch : (3,) array
        Per-axis stretch factors [x, y, z].
    cost : float
        Final subspace alignment cost.
    sigma_scalp : float
        Scalp conductivity (S/m). Default 0.3.
    rotation : (3,) array or None
        Brain rotation [rx, ry, rz] in degrees (XYZ extrinsic Euler).
        None or zeros means no rotation.
    co_deform : bool
        Whether the inner skull BEM surface was co-deformed with sources.
    k : int
        Number of eigenvectors used in subspace alignment.
    extra : dict or None
        Additional metadata to store.

    Returns
    -------
    path : str
        Path to the saved cache file.
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    h = _cov_hash(C_data)
    path = os.path.join(_CACHE_DIR, f"global_params_{h}.npz")
    if rotation is None:
        rotation = np.zeros(3)
    save_dict = dict(
        sigma_skull=np.float64(sigma_skull),
        sigma_scalp=np.float64(sigma_scalp),
        offset=np.asarray(offset, dtype=np.float64),
        stretch=np.asarray(stretch, dtype=np.float64),
        rotation=np.asarray(rotation, dtype=np.float64),
        co_deform=np.bool_(co_deform),
        cost=np.float64(cost),
        k=np.int64(k),
        cov_shape=np.array(C_data.shape),
    )
    if extra:
        for key, val in extra.items():
            save_dict[f"extra_{key}"] = val
    np.savez(path, **save_dict)
    return path


def load_params(C_data: np.ndarray) -> dict | None:
    """Load cached parameters for a given sensor covariance.

    Returns
    -------
    dict or None
        Dictionary with keys ``sigma_skull``, ``sigma_scalp``, ``offset``,
        ``stretch``, ``rotation``, ``co_deform``, ``cost``, ``k``.
        Returns ``None`` if no cache entry exists.

        For backwards compatibility with old cache files that only stored
        ``sigma``, the ``sigma`` field is mapped to ``sigma_skull``.
    """
    h = _cov_hash(C_data)
    path = os.path.join(_CACHE_DIR, f"global_params_{h}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    # Backwards compatibility: old caches stored "sigma" not "sigma_skull"
    if "sigma_skull" in data:
        sigma_skull = float(data["sigma_skull"])
    else:
        sigma_skull = float(data["sigma"])
    return {
        "sigma_skull": sigma_skull,
        "sigma_scalp": float(data["sigma_scalp"]) if "sigma_scalp" in data else 0.3,
        "offset": data["offset"],
        "stretch": data["stretch"],
        "rotation": data["rotation"] if "rotation" in data else np.zeros(3),
        "co_deform": bool(data["co_deform"]) if "co_deform" in data else False,
        "cost": float(data["cost"]),
        "k": int(data["k"]),
    }


def list_cached() -> list[str]:
    """List all cached parameter files."""
    if not os.path.isdir(_CACHE_DIR):
        return []
    return [f for f in os.listdir(_CACHE_DIR) if f.endswith(".npz")]
