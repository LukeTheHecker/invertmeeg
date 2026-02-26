import mne
import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta


def _batch_score(Gram, Q, idx_array):
    """Vectorised ML score for all subsets: tr(Gram_S^{-1} Q_S)."""
    rows = idx_array[:, :, None]
    cols = idx_array[:, None, :]
    G_batch = Gram[rows, cols]
    Q_batch = Q[rows, cols]
    X = np.linalg.solve(G_batch, Q_batch)
    return np.trace(X, axis1=-2, axis2=-1)


def _score_single(Gram, Q, subset):
    """ML score for a single subset."""
    idx = np.array(subset, dtype=np.intp)
    G_sub = Gram[np.ix_(idx, idx)]
    Q_sub = Q[np.ix_(idx, idx)]
    return np.trace(np.linalg.solve(G_sub, Q_sub))


def greedy_ml_search(Gram, Q, total_var, n_eff, T, n_ext, k_max,
                     n_starts, n_refine_iters, penalty_mode,
                     excluded_fn=None):
    """Multi-start greedy forward ML search with coordinate-wise refinement.

    Parameters
    ----------
    Gram : (n_ext, n_ext) — leadfield Gram matrix
    Q : (n_ext, n_ext) — data projection matrix
    total_var : float — total data variance
    n_eff, T : int — effective channels and timepoints
    n_ext : int — number of columns in (extended) dictionary
    k_max : int — maximum number of sources
    n_starts : int — number of greedy restarts
    n_refine_iters : int — coordinate-wise refinement iterations
    penalty_mode : str — BIC variant
    excluded_fn : callable or None — given a set of selected extended indices,
        returns a set of additional indices to exclude (for mutual exclusivity).
        If None, no extra exclusions.

    Returns
    -------
    selected : list[int] — selected extended dictionary indices
    """

    def _penalty(k):
        if penalty_mode == "bic_per_timepoint":
            return k * T * np.log(n_eff)
        elif penalty_mode == "aic":
            return 2 * k * T
        elif penalty_mode == "bic_stacked":
            return k * (T + 1) * np.log(n_eff * T)
        elif penalty_mode == "bic_support_only":
            return k * np.log(n_eff * T)
        else:
            raise ValueError(f"Unknown penalty_mode: {penalty_mode}")

    def _bic(score, k):
        residual = max(total_var - score, 1e-30)
        return n_eff * T * np.log(residual / (n_eff * T)) + _penalty(k)

    def _get_remaining(selected):
        """Get indices not selected and not excluded."""
        blocked = set(selected)
        if excluded_fn is not None:
            blocked |= excluded_fn(selected)
        return np.array(
            [j for j in range(n_ext) if j not in blocked], dtype=np.intp
        )

    null_bic = _bic(0.0, 0)

    # Score all individual entries for multi-start seeds
    single_idx = np.arange(n_ext, dtype=np.intp).reshape(-1, 1)
    single_scores = _batch_score(Gram, Q, single_idx)
    seed_order = np.argsort(single_scores)[::-1]
    n_starts_eff = min(n_starts, n_ext)

    global_best_bic = null_bic
    global_best_selected = None

    for start_i in range(n_starts_eff):
        seed = int(seed_order[start_i])
        selected = [seed]
        current_bic = _bic(single_scores[seed], 1)

        if current_bic >= null_bic:
            continue

        for k in range(2, min(k_max, n_ext) + 1):
            remaining = _get_remaining(selected)
            if len(remaining) == 0:
                break

            candidates = np.array(
                [sorted(selected + [j]) for j in remaining], dtype=np.intp
            )
            scores = _batch_score(Gram, Q, candidates)

            best_idx = np.argmax(scores)
            new_bic = _bic(scores[best_idx], k)

            if new_bic < current_bic:
                current_bic = new_bic
                selected = list(candidates[best_idx])
            else:
                break

        # Coordinate-wise refinement
        current_score = _score_single(Gram, Q, selected)

        for _ in range(n_refine_iters):
            improved = False
            for pos_idx in range(len(selected)):
                base = [s for i, s in enumerate(selected) if i != pos_idx]
                remaining = _get_remaining(base)

                candidates = np.array(
                    [sorted(base + [j]) for j in remaining], dtype=np.intp
                )
                scores = _batch_score(Gram, Q, candidates)

                best_idx = np.argmax(scores)
                if scores[best_idx] > current_score + 1e-10:
                    selected = list(candidates[best_idx])
                    current_score = scores[best_idx]
                    improved = True

            if not improved:
                break

        final_bic = _bic(current_score, len(selected))
        if final_bic < global_best_bic:
            global_best_bic = final_bic
            global_best_selected = list(selected)

    if global_best_selected is None:
        global_best_selected = [int(seed_order[0])]

    return global_best_selected


class SolverGreedyML(BaseSolver):
    """Source localization via multi-start greedy forward selection with
    ML objective, BIC model order selection, and coordinate-wise refinement.

    Scales as O(n_starts * n_sources * k_max) — works for any source space.

    References
    ----------
    [1] Wax, M., & Kailath, T. (1985). Detection of signals by information
        theoretic criteria. IEEE Trans. ASSP, 33(2), 387-392.
    """

    meta = SolverMeta(
        acronym="GreedyML",
        full_name="Greedy Maximum-Likelihood",
        category="Subspace Methods",
        description=(
            "Source localization via multi-start greedy forward selection "
            "with ML objective, BIC model order selection, and coordinate-wise "
            "refinement. Scales to large source spaces."
        ),
        references=[
            "Wax, M., & Kailath, T. (1985). Detection of signals by information theoretic criteria. IEEE Trans. ASSP, 33(2), 387-392.",
        ],
    )

    def __init__(self, name="GreedyML", **kwargs):
        self.name = name
        super().__init__(**kwargs)

    def make_inverse_operator(
        self,
        forward,
        mne_obj=None,
        *args,
        alpha="auto",
        noise_cov: mne.Covariance | None = None,
        k_max=5,
        n_starts=50,
        n_refine_iters=3,
        penalty_mode="bic_per_timepoint",
        **kwargs,
    ):
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        wf = self.prepare_whitened_forward(noise_cov)

        data = self.unpack_data_obj(mne_obj)
        data_w = wf.sensor_transform @ data
        G_w = wf.G_white

        n_eff, n_sources = G_w.shape
        T = data_w.shape[1]

        Gram = G_w.T @ G_w
        # Ridge for numerical stability (rank-deficient when n_sources >> n_eff)
        eps_gram = 1e-10 * np.trace(Gram) / max(n_sources, 1)
        Gram[np.diag_indices_from(Gram)] += eps_gram
        R = G_w.T @ data_w
        Q = R @ R.T
        total_var = np.sum(data_w ** 2)

        selected = greedy_ml_search(
            Gram, Q, total_var, n_eff, T, n_sources,
            k_max, n_starts, n_refine_iters, penalty_mode,
        )
        selected_sources = np.array(selected, dtype=np.intp)

        # Construct ML inverse operator
        G_sel = G_w[:, selected_sources]
        inv_w = np.linalg.lstsq(G_sel, np.eye(n_eff), rcond=None)[0]

        inverse_operator_w = np.zeros((n_sources, n_eff))
        inverse_operator_w[selected_sources, :] = inv_w

        self.inverse_operators = [
            InverseOperator(inverse_operator_w @ wf.sensor_transform, self.name),
        ]
        return self
