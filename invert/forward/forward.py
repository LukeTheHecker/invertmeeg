import os

import mne


def _get_fsaverage_dir(*, verbose=0) -> str:
    """Return a local fsaverage directory without forcing downloads.

    MNE's ``fetch_fsaverage`` always performs a manifest check and will attempt
    to download missing files. In sandboxed/offline environments this can fail
    even when the required forward-model assets are already present.

    This helper prefers local installations (via env vars) and falls back to
    ``fetch_fsaverage`` only when necessary.
    """
    env_candidates = []
    for key in ("INVERT_FSAVERAGE_DIR", "MNE_FSAVERAGE_DIR"):
        value = os.environ.get(key)
        if value:
            env_candidates.append(value)

    subjects_dir = os.environ.get("SUBJECTS_DIR")
    if subjects_dir:
        env_candidates.append(os.path.join(subjects_dir, "fsaverage"))

    mne_data = os.environ.get("MNE_DATA")
    if mne_data:
        env_candidates.append(os.path.join(mne_data, "MNE-fsaverage-data", "fsaverage"))

    # Common default (real user home, not necessarily the process HOME).
    user_home = os.path.expanduser("~")
    env_candidates.append(
        os.path.join(user_home, "mne_data", "MNE-fsaverage-data", "fsaverage")
    )

    required = [
        os.path.join("bem", "fsaverage-trans.fif"),
        os.path.join("bem", "fsaverage-5120-5120-5120-bem-sol.fif"),
    ]
    for cand in env_candidates:
        if not cand:
            continue
        if all(os.path.exists(os.path.join(cand, rel)) for rel in required):
            return cand

    # Fall back to MNE (may download in non-sandboxed environments).
    fs_dir = mne.datasets.fetch_fsaverage(verbose=verbose)
    return str(fs_dir)


def create_forward_model(
    sampling="ico3", info=None, fixed_ori=True, conductivity=None, n_jobs=1, verbose=0
):
    """Create a forward model using the fsaverage template form freesurfer.

    Parameters:
    ----------
    sampling : str
        the downsampling of the forward model.
        Recommended are 'ico3' (small), 'ico4' (medium) or
        'ico5' (large).
    info : mne.Info
        info instance which contains the desired
        electrode names and positions.
        This can be obtained e.g. from your processed mne.Raw.info,
        mne.Epochs.info or mne.Evoked.info
        If info is None the Info instance is created where
        electrodes are chosen automatically from the easycap-M10
        layout.
    fixed_ori : bool
        Whether orientation of dipoles shall be fixed (set to True)
        or free (set to False).

    Return
    ------
    fwd : mne.Forward
        The forward model object
    """

    # Fetch the template files for our forward model
    fs_dir = _get_fsaverage_dir(verbose=verbose)
    subjects_dir = os.path.dirname(fs_dir)

    # The files live in:
    subject = "fsaverage"
    trans = os.path.join(fs_dir, "bem", "fsaverage-trans.fif")
    src = os.path.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
    if conductivity is None:
        bem = os.path.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
    else:
        if "ico" not in sampling:
            msg = f"sampling must be ico# but is {sampling}"
            raise AttributeError(msg)
        if not isinstance(conductivity, (tuple,)):
            msg = f"conductivity must be None or a tuple of three conductivity values but is {conductivity}"
            raise AttributeError(msg)

        bem = mne.make_bem_model(
            subject,
            ico=int(sampling[-1]),
            conductivity=conductivity,
            subjects_dir=subjects_dir,
            verbose=verbose,
        )

    # Create our own info object, see e.g.:
    if info is None:
        info = get_info()

    # Create and save Source Model
    src = mne.setup_source_space(
        subject,
        spacing=sampling,
        surface="white",
        subjects_dir=subjects_dir,
        add_dist=False,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    # Forward Model
    fwd = mne.make_forward_solution(
        info,
        trans=trans,
        src=src,
        bem=bem,
        eeg=True,
        mindist=5.0,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    if fixed_ori:
        # Fixed Orientations
        fwd = mne.convert_forward_solution(
            fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose=verbose
        )

    return fwd


def get_info(kind="easycap-M10", sfreq=1000):
    """Create some generic mne.Info object.

    Parameters
    ----------
    kind : str
        kind, for examples see:
            https://mne.tools/stable/generated/mne.channels.make_standard_montage.html#mne.channels.make_standard_montage

    Return
    ------
    info : mne.Info
        The mne.Info object

    https://mne.tools/stable/generated/mne.create_info.html#mne.create_info
    https://mne.tools/stable/auto_tutorials/simulation/plot_creating_data_structures.html
    """
    montage = mne.channels.make_standard_montage(kind)
    info = mne.create_info(
        montage.ch_names, sfreq, ch_types=["eeg"] * len(montage.ch_names), verbose=0
    )
    info.set_montage(kind)
    return info


def create_muse_montage(sfreq=256):
    """Create a montage for the Muse headband (4 electrodes).

    The Muse headband uses 4 dry EEG electrodes positioned at:
    - TP9 (left temporal)
    - AF7 (left frontal)
    - AF8 (right frontal)
    - TP10 (right temporal)

    The positions are extracted from the standard_1020 montage.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz (default: 256 Hz for Muse)

    Return
    ------
    info : mne.Info
        The mne.Info object with Muse montage
    montage : mne.channels.DigMontage
        The montage object (useful for visualization)

    Example
    -------
    >>> info, montage = create_muse_montage()
    >>> montage.plot()  # Visualize electrode positions
    >>> fwd = create_forward_model(info=info, sampling='ico3')
    """
    # Muse electrode names
    ch_names = ["TP9", "AF7", "AF8", "TP10", "Fpz"]  # Fpz is the reference electrode

    # Load standard 10-20 montage which contains these positions
    standard_montage = mne.channels.make_standard_montage("standard_1020")

    # Extract positions for Muse channels
    standard_pos = standard_montage.get_positions()
    ch_pos = {ch: standard_pos["ch_pos"][ch] for ch in ch_names}

    # Create custom montage with Muse positions
    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        nasion=standard_pos["nasion"],
        lpa=standard_pos["lpa"],
        rpa=standard_pos["rpa"],
        coord_frame="head",
    )

    # Create Info object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    info.set_montage(montage)

    return info, montage


def create_8channel_montage(sfreq=1000):
    """Create an evenly spaced 8-channel EEG montage.

    This montage uses 8 electrodes evenly distributed across the scalp:
    - Fp1, Fp2 (frontal)
    - C3, Cz, C4 (central)
    - P3, P4 (parietal)
    - Oz (occipital)

    The positions are extracted from the standard_1020 montage.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz (default: 1000 Hz)

    Return
    ------
    info : mne.Info
        The mne.Info object with 8-channel montage
    montage : mne.channels.DigMontage
        The montage object (useful for visualization)

    Example
    -------
    >>> info, montage = create_8channel_montage()
    >>> montage.plot()  # Visualize electrode positions
    >>> fwd = create_forward_model(info=info, sampling='ico3')
    """
    # Select 8 evenly distributed channels
    ch_names = ["Fp1", "Fp2", "C3", "Cz", "C4", "P3", "P4", "Oz"]

    # Load standard 10-20 montage
    standard_montage = mne.channels.make_standard_montage("standard_1020")

    # Extract positions for selected channels
    standard_pos = standard_montage.get_positions()
    ch_pos = {ch: standard_pos["ch_pos"][ch] for ch in ch_names}

    # Create custom montage
    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        nasion=standard_pos["nasion"],
        lpa=standard_pos["lpa"],
        rpa=standard_pos["rpa"],
        coord_frame="head",
    )

    # Create Info object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    info.set_montage(montage)

    return info, montage
