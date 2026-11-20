"""Compare focal EEG scans on nested montages using a surface EEG forward."""

import argparse
import csv
import sys

import mne
import numpy as np


def probe(forward_path, cases):
    fwd = mne.read_forward_solution(forward_path, verbose=False)
    with mne.utils.use_log_level("ERROR"):
        fwd = mne.pick_types_forward(fwd, eeg=True, meg=False)
    fwd = mne.convert_forward_solution(fwd, force_fixed=True, verbose=False)
    gain = fwd["sol"]["data"].astype(float)
    positions = np.array([ch["loc"][:3] for ch in fwd["info"]["chs"]])
    # Choose a nested spatially spread subset, starting at the top electrode.
    selected = [int(positions[:, 2].argmax())]
    for _ in range(31):
        distances = np.linalg.norm(positions[:, None] - positions[selected], axis=2)
        selected.append(int(distances.min(axis=1).argmax()))
    if len(set(selected)) != 32:
        raise ValueError("Supply a forward with at least 32 distinct EEG positions")
    gain = gain[selected]
    scan = {}
    for count in (8, 16, 32):
        centered = gain[:count] - gain[:count].mean(axis=0)
        scan[count] = (centered / np.linalg.norm(centered, axis=0)).T

    rng = np.random.default_rng(20261120)
    samples = np.arange(100)
    writer = csv.writer(sys.stdout)
    writer.writerow(["case", "channels", "snr_db", "method", "error_mm"])
    for case in range(cases):
        target = rng.integers(gain.shape[1])
        pulse = np.exp(-0.5 * ((samples - rng.uniform(40, 60)) / 8) ** 2)
        signal = gain[:, target, None] * pulse
        noise = rng.standard_normal(signal.shape)
        # Set RMS SNR after average referencing on the full 32-channel montage.
        noise *= (signal - signal.mean(axis=0)).std() / (
            noise - noise.mean(axis=0)
        ).std()
        for snr in (-10, -5, 0, 5):
            data = signal + noise * 10 ** (-snr / 20)
            for count, filters in scan.items():
                projected = filters @ (data[:count] - data[:count].mean(axis=0))
                # Predefined event window, independent of the true peak latency.
                scores = {
                    "full_power": np.mean(projected**2, axis=1),
                    "event_mean": np.abs(projected[:, 35:66].mean(axis=1)),
                }
                for method, score in scores.items():
                    delta = fwd["source_rr"][score.argmax()] - fwd["source_rr"][target]
                    writer.writerow(
                        [case, count, snr, method, 1000 * np.linalg.norm(delta)]
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forward", help="EEG surface forward solution in FIF format")
    parser.add_argument("--cases", type=int, default=100)
    args = parser.parse_args()
    probe(args.forward, args.cases)
