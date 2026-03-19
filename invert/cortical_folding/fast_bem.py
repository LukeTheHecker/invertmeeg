"""
Fast BEM solution computation via K/gamma factorization.

MNE's make_bem_solution() computes two things:
1. K matrix (geometry-dependent coefficient matrix) — expensive, O(n_vert * n_tri)
2. M = I - K*gamma/(2pi), then inv(M) — gamma depends only on conductivity

For conductivity sweeps (e.g. optimizing skull conductivity), we compute K once
and only redo the cheap inversion for each new conductivity.

For geometry sweeps (e.g. co-deforming the inner skull), use partial_K_update()
to recompute only the blocks involving the changed surface (~1.8x speedup).

Benchmarks (fsaverage):
    ico3 (1926 verts): K=2s, inv=0.07s → 30x speedup per sweep step
    ico3 partial K (inner skull only): 0.9s → 1.8x vs full K
    ico4 (7686 verts): K=25s, inv=3.7s → 7x speedup per sweep step
    ico2 vs ico4 leadfield correlation: 0.9933 (viable for coarse search)

Usage:
    # Initial solve (same cost as MNE)
    bem, coeff, coeff_h = make_bem_solution_factored(model)

    # Sweep conductivity (only pays inversion cost)
    for sigma in sigma_list:
        rebuild_bem_from_coeff(bem, coeff, coeff_h, sigma)
        fwd = mne.make_forward_solution(..., bem=bem, ...)

    # Sweep inner skull geometry (partial K + conductivity)
    for new_rr in skull_candidates:
        bem["surfs"][2]["rr"] = new_rr
        _check_complete_surface(bem["surfs"][2])
        coeff = partial_K_update(bem["surfs"], coeff_base, changed_idx=2)
        rebuild_bem_from_coeff(bem, coeff, coeff_h, conductivity)
"""

import numpy as np

from mne.bem import (
    _add_gamma_multipliers,
    _check_complete_surface,
    _correct_auto_elements,
    _ensure_bem_surfaces,
    _fwd_bem_ip_modify_solution,
    _fwd_bem_lin_pot_coeff,
    _fwd_bem_multi_solution,
    _lin_pot_coeff,
)

_BEM_IP_APPROACH_LIMIT = 0.321


def partial_K_update(surfs, coeff_base, changed_idx=2):
    """Recompute only the K blocks involving a changed BEM surface.

    When a single surface (e.g. inner skull) is deformed, only the blocks
    that involve that surface need recomputation. For a 3-layer model this
    is 5 of 9 blocks (~1.8x speedup over full recomputation).

    Parameters
    ----------
    surfs : list of dict
        BEM surfaces with the changed surface already updated
        (new ``rr``, and ``_check_complete_surface`` already called).
    coeff_base : (n_tot, n_tot) array
        K matrix from the original (undeformed) geometry.
    changed_idx : int
        Index of the surface that changed (default 2 = inner skull).

    Returns
    -------
    coeff : (n_tot, n_tot) array
        Updated K matrix (copy — coeff_base is not modified).
    """
    coeff = coeff_base.copy()
    nps = [s["np"] for s in surfs]
    offsets = np.cumsum(np.concatenate(([0], nps)))

    for si_1 in range(len(surfs)):
        for si_2 in range(len(surfs)):
            if si_1 != changed_idx and si_2 != changed_idx:
                continue
            surf1, surf2 = surfs[si_1], surfs[si_2]
            rr_ord = np.arange(nps[si_1])
            tri_rr = surf2["rr"][surf2["tris"]]
            tri_nn = surf2["tri_nn"]
            tri_area = surf2["tri_area"]
            submat = coeff[
                offsets[si_1]:offsets[si_1 + 1],
                offsets[si_2]:offsets[si_2 + 1],
            ]
            submat[:] = 0
            for k in range(surf2["ntri"]):
                tri = surf2["tris"][k]
                if si_1 == si_2:
                    skip_idx = (
                        (rr_ord == tri[0]) | (rr_ord == tri[1]) | (rr_ord == tri[2])
                    )
                else:
                    skip_idx = []
                coeffs = _lin_pot_coeff(
                    fros=surf1["rr"],
                    tri_rr=tri_rr[k],
                    tri_nn=tri_nn[k],
                    tri_area=tri_area[k],
                )
                coeffs[skip_idx] = 0.0
                submat[:, tri] -= coeffs
            if si_1 == si_2:
                _correct_auto_elements(surf1, submat)
    return coeff


def make_bem_solution_factored(surfs):
    """Compute BEM solution and return the reusable K coefficient matrix.

    Parameters
    ----------
    surfs : list of dict
        BEM surfaces from mne.make_bem_model().

    Returns
    -------
    bem : ConductorModel
        Complete BEM solution (usable with mne.make_forward_solution).
    coeff : (n_tot, n_tot) array
        Geometry-dependent coefficient matrix K.
    coeff_h : (n_last, n_last) array or None
        Homogeneous K for IP approach (3-layer models with low skull/scalp ratio).
    """
    bem = _ensure_bem_surfaces(surfs)
    _add_gamma_multipliers(bem)
    for surf in bem["surfs"]:
        _check_complete_surface(surf)

    coeff = _fwd_bem_lin_pot_coeff(bem["surfs"])
    coeff_saved = coeff.copy()  # _fwd_bem_multi_solution destroys input

    nps = [surf["np"] for surf in bem["surfs"]]
    bem["nsol"] = len(coeff)
    bem["solution"] = _fwd_bem_multi_solution(coeff, bem["gamma"], nps)

    coeff_h_saved = None
    if len(bem["surfs"]) == 3:
        ip_mult = bem["sigma"][1] / bem["sigma"][2]
        if ip_mult <= _BEM_IP_APPROACH_LIMIT:
            coeff_h = _fwd_bem_lin_pot_coeff([bem["surfs"][-1]])
            coeff_h_saved = coeff_h.copy()
            ip_solution = _fwd_bem_multi_solution(
                coeff_h, None, [bem["surfs"][-1]["np"]]
            )
            _fwd_bem_ip_modify_solution(bem["solution"], ip_solution, ip_mult, nps)

    bem["bem_method"] = 2  # FIFFV_BEM_APPROX_LINEAR
    bem["solver"] = "mne"
    return bem, coeff_saved, coeff_h_saved


def rebuild_bem_from_coeff(bem, coeff, coeff_h, conductivity):
    """Rebuild BEM solution from precomputed K with new conductivity.

    Cost: matrix copy + assembly + inversion only (~0.07s ico3, ~3.7s ico4).

    Parameters
    ----------
    bem : ConductorModel
        BEM to update in-place.
    coeff : (n_tot, n_tot) array
        Precomputed K from make_bem_solution_factored.
    coeff_h : (n_last, n_last) array or None
        Precomputed homogeneous K.
    conductivity : tuple
        New conductivity values, e.g. (0.3, 0.006, 0.3).

    Returns
    -------
    bem : ConductorModel
        Updated in-place.
    """
    for surf, sigma in zip(bem["surfs"], conductivity):
        surf["sigma"] = sigma
    _add_gamma_multipliers(bem)

    nps = [surf["np"] for surf in bem["surfs"]]
    coeff_work = coeff.copy()
    bem["solution"] = _fwd_bem_multi_solution(coeff_work, bem["gamma"], nps)

    if len(bem["surfs"]) == 3:
        ip_mult = bem["sigma"][1] / bem["sigma"][2]
        if ip_mult <= _BEM_IP_APPROACH_LIMIT and coeff_h is not None:
            coeff_h_work = coeff_h.copy()
            ip_solution = _fwd_bem_multi_solution(
                coeff_h_work, None, [bem["surfs"][-1]["np"]]
            )
            _fwd_bem_ip_modify_solution(
                bem["solution"], ip_solution, ip_mult, nps
            )

    return bem
