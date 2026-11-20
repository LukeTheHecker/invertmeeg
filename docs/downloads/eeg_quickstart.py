"""Recover a simulated dipole from 32-channel EEG using minimum norm."""

import mne
import numpy as np

from invert import Solver

montage = mne.channels.make_standard_montage("biosemi32")
info = mne.create_info(montage.ch_names, sfreq=200, ch_types="eeg")
info.set_montage(montage)
sphere = mne.make_sphere_model(r0=(0, 0, 0.043), head_radius=0.09, verbose=False)
src = mne.setup_volume_source_space(pos=20, sphere=(0, 0, 0.045, 0.065), verbose=False)
fwd = mne.make_forward_solution(
    info, trans=None, src=src, bem=sphere, meg=False, verbose=False
)

# Place a single x-oriented dipole near the specified head coordinates (metres).
locations = fwd["source_rr"]
target = np.argmin(np.linalg.norm(locations - [0.04, 0.02, 0.08], axis=1))
topography = fwd["sol"]["data"][:, 3 * target]
times = np.arange(40) / info["sfreq"]
pulse = 10e-9 * np.exp(-0.5 * ((times - 0.1) / 0.02) ** 2)
signal = topography[:, None] * pulse
noise = np.random.default_rng(42).normal(scale=0.1 * signal.std(), size=signal.shape)
evoked = mne.EvokedArray(signal + noise, info, verbose=False)
evoked.set_eeg_reference("average", projection=True, verbose=False)
evoked.apply_proj(verbose=False)

solver = Solver("MNE")
solver.make_inverse_operator(fwd, alpha=0.01)
estimate = solver.apply_inverse_operator(evoked)
if not np.isfinite(estimate.data).all() or not np.any(estimate.data):
    raise ValueError("Expected a finite, nonzero source estimate")
peak = np.argmax(np.abs(estimate.data[:, pulse.argmax()]))
error_mm = 1000 * np.linalg.norm(locations[peak] - locations[target])
print(f"Peak localization error: {error_mm:.1f} mm")
# The seeded example should recover the source to within one grid step.
assert error_mm <= 20.01
