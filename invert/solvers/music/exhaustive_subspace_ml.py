from itertools import combinations

import mne
import numpy as np

from ..base import BaseSolver, InverseOperator, SolverMeta


def _batch_score(Gram, Q, idx_array):
    """Vectorised score for all subsets: tr(Gram_S^{-1} Q_S)."""
    rows = idx_array[:, :, None]  # (n_subsets, k, 1)
    cols = idx_array[:, None, :]  # (n_subsets, 1, k)
    G_batch = Gram[rows, cols]  # (n_subsets, k, k)
    Q_batch = Q[rows, cols]  # (n_subsets, k, k)
    # solve G_batch @ X = Q_batch  =>  tr(G^{-1} Q) = tr(X)
    X = np.linalg.solve(G_batch, Q_batch)
    return np.trace(X, axis1=-2, axis2=-1)


def _score_single(Gram, Q, subset):
    """Score a single subset."""
    idx = np.array(subset, dtype=np.intp)
    G_sub = Gram[np.ix_(idx, idx)]
    Q_sub = Q[np.ix_(idx, idx)]
    return np.trace(np.linalg.solve(G_sub, Q_sub))


class SolverExhaustiveSubspaceML(BaseSolver):
    """Source localization via exhaustive maximum-likelihood subset search
    with BIC model order selection.

    For k <= k_exhaustive (default 3), evaluates ALL C(n_sources, k) subsets.
    For k > k_exhaustive, uses beam search extending top-B solutions from k-1.

    References
    ----------
    [1] Wax, M., & Kailath, T. (1985). Detection of signals by information
        theoretic criteria. IEEE Trans. ASSP, 33(2), 387-392.
    """

    meta = SolverMeta(
        acronym="ExhaustiveML",
        full_name="Exhaustive Subspace Maximum-Likelihood",
        category="Subspace Methods",
        description=(
            "Source localization via exhaustive maximum-likelihood subset "
            "search (k<=3) with beam search extension (k>3) and BIC model "
            "order selection."
        ),
        references=[
            "Wax, M., & Kailath, T. (1985). Detection of signals by information theoretic criteria. IEEE Trans. ASSP, 33(2), 387-392.",
        ],
    )

    def __init__(self, name="ExhaustiveML", **kwargs):
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
        k_exhaustive=3,
        beam_width=50,
        penalty_mode="bic_per_timepoint",
        **kwargs,
    ):
        super().make_inverse_operator(forward, mne_obj, *args, alpha=alpha, **kwargs)
        wf = self.prepare_whitened_forward(noise_cov)

        data = self.unpack_data_obj(mne_obj)
        data_w = wf.sensor_transform @ data  # (n_eff, T)
        G_w = wf.G_white  # (n_eff, n_sources)

        n_eff, n_sources = G_w.shape
        T = data_w.shape[1]

        Gram = G_w.T @ G_w
        # Ridge for numerical stability (rank-deficient when n_sources >> n_eff)
        eps_gram = 1e-10 * np.trace(Gram) / max(n_sources, 1)
        Gram[np.diag_indices_from(Gram)] += eps_gram
        R = G_w.T @ data_w
        Q = R @ R.T
        total_var = np.sum(data_w**2)

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

        def _compute_bic(scores, k):
            residuals = np.maximum(total_var - scores, 1e-30)
            return n_eff * T * np.log(residuals / (n_eff * T)) + _penalty(k)

        best_bic = np.inf
        best_subset = None
        chunk_size = 500_000

        top_subsets: list[tuple[int, ...]] = []

        # Limit exhaustive search to feasible sizes (< 2M subsets).
        from math import comb as _comb

        _max_exhaustive_subsets = 2_000_000

        for k in range(1, min(k_max, n_sources) + 1):
            if k <= k_exhaustive and _comb(n_sources, k) <= _max_exhaustive_subsets:
                idx_list = list(combinations(range(n_sources), k))
                idx_array = np.array(idx_list, dtype=np.intp)

                all_score_chunks = []
                for start in range(0, len(idx_array), chunk_size):
                    chunk = idx_array[start : start + chunk_size]
                    scores = _batch_score(Gram, Q, chunk)
                    all_score_chunks.append(scores)

                all_scores = np.concatenate(all_score_chunks)
                bics = _compute_bic(all_scores, k)
                local_best = np.argmin(bics)

                if bics[local_best] < best_bic:
                    best_bic = bics[local_best]
                    best_subset = idx_list[local_best]

                top_indices = np.argsort(all_scores)[-beam_width:]
                top_subsets = [idx_list[i] for i in top_indices]

            else:
                new_subsets: dict[tuple[int, ...], float] = {}
                for subset in top_subsets:
                    subset_set = set(subset)
                    remaining = [j for j in range(n_sources) if j not in subset_set]
                    if not remaining:
                        continue
                    extended = [tuple(sorted(subset + (j,))) for j in remaining]
                    ext_array = np.array(extended, dtype=np.intp)

                    for start in range(0, len(ext_array), chunk_size):
                        chunk = ext_array[start : start + chunk_size]
                        scores = _batch_score(Gram, Q, chunk)
                        for idx, sc in enumerate(scores):
                            sub = extended[start + idx]
                            if sub not in new_subsets or sc > new_subsets[sub]:
                                new_subsets[sub] = sc

                if not new_subsets:
                    break

                ext_list = list(new_subsets.keys())
                ext_scores = np.array(list(new_subsets.values()))
                bics = _compute_bic(ext_scores, k)
                local_best = np.argmin(bics)

                if bics[local_best] < best_bic:
                    best_bic = bics[local_best]
                    best_subset = ext_list[local_best]

                top_indices = np.argsort(ext_scores)[-beam_width:]
                top_subsets = [ext_list[i] for i in top_indices]

        selected_sources = np.array(best_subset, dtype=np.intp)

        G_sel = G_w[:, selected_sources]
        inv_w = np.linalg.lstsq(G_sel, np.eye(n_eff), rcond=None)[0]

        inverse_operator_w = np.zeros((n_sources, n_eff))
        inverse_operator_w[selected_sources, :] = inv_w

        self.inverse_operators = [
            InverseOperator(inverse_operator_w @ wf.sensor_transform, self.name),
        ]
        return self


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
        data_w = wf.sensor_transform @ data  # (n_eff, T)
        G_w = wf.G_white  # (n_eff, n_sources)

        n_eff, n_sources = G_w.shape
        T = data_w.shape[1]

        Gram = G_w.T @ G_w
        # Ridge for numerical stability (rank-deficient when n_sources >> n_eff)
        eps_gram = 1e-10 * np.trace(Gram) / max(n_sources, 1)
        Gram[np.diag_indices_from(Gram)] += eps_gram
        R = G_w.T @ data_w
        Q = R @ R.T
        total_var = np.sum(data_w**2)

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

        null_bic = _bic(0.0, 0)

        # --- Score all individual sources for multi-start seeds ---
        single_idx = np.arange(n_sources, dtype=np.intp).reshape(-1, 1)
        single_scores = _batch_score(Gram, Q, single_idx)
        seed_order = np.argsort(single_scores)[::-1]
        n_starts_eff = min(n_starts, n_sources)

        # --- Multi-start greedy forward selection ---
        global_best_bic = null_bic
        global_best_selected = None

        for start_i in range(n_starts_eff):
            seed = int(seed_order[start_i])
            selected = [seed]
            current_bic = _bic(single_scores[seed], 1)

            if current_bic >= null_bic:
                # This seed isn't even better than null model
                continue

            for k in range(2, min(k_max, n_sources) + 1):
                selected_set = set(selected)
                remaining = np.array(
                    [j for j in range(n_sources) if j not in selected_set],
                    dtype=np.intp,
                )
                if len(remaining) == 0:
                    break

                candidates = np.array(
                    [tuple(sorted(selected + [j])) for j in remaining],
                    dtype=np.intp,
                )
                scores = _batch_score(Gram, Q, candidates)

                best_idx = np.argmax(scores)
                new_bic = _bic(scores[best_idx], k)

                if new_bic < current_bic:
                    current_bic = new_bic
                    selected = list(candidates[best_idx])
                else:
                    break

            # --- Coordinate-wise refinement ---
            current_score = _score_single(Gram, Q, selected)

            for _ in range(n_refine_iters):
                improved = False
                for pos_idx in range(len(selected)):
                    base = [s for i, s in enumerate(selected) if i != pos_idx]
                    base_set = set(base)
                    remaining = np.array(
                        [j for j in range(n_sources) if j not in base_set],
                        dtype=np.intp,
                    )

                    candidates = np.array(
                        [tuple(sorted(base + [j])) for j in remaining],
                        dtype=np.intp,
                    )
                    scores = _batch_score(Gram, Q, candidates)

                    best_idx = np.argmax(scores)
                    if scores[best_idx] > current_score + 1e-10:
                        selected = list(candidates[best_idx])
                        current_score = scores[best_idx]
                        improved = True

                if not improved:
                    break

            # Recompute BIC after refinement (K might be same but score changed)
            final_bic = _bic(current_score, len(selected))
            if final_bic < global_best_bic:
                global_best_bic = final_bic
                global_best_selected = list(selected)

        if global_best_selected is None:
            # Fallback: best individual source
            global_best_selected = [int(seed_order[0])]

        selected_sources = np.array(global_best_selected, dtype=np.intp)

        # --- Construct ML inverse operator ---
        G_sel = G_w[:, selected_sources]
        inv_w = np.linalg.lstsq(G_sel, np.eye(n_eff), rcond=None)[0]

        inverse_operator_w = np.zeros((n_sources, n_eff))
        inverse_operator_w[selected_sources, :] = inv_w

        self.inverse_operators = [
            InverseOperator(inverse_operator_w @ wf.sensor_transform, self.name),
        ]
        return self
