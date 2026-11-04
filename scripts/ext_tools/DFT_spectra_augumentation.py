from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_PEAKS_CSV = Path(r"D:\OneDrive - ichf.edu.pl\PycharmProjects\DFT_database_gen\data\dft_peaks.csv")
OUTPUT_DIR = Path(r"D:\OneDrive - ichf.edu.pl\PycharmProjects\DFT_database_gen\augmented_spectra")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_AUGMENTED = 50
RANDOM_SEED = 42

# spectral axis
X_MIN = 200.0
X_MAX = 2000.0
X_STEP = 1.0

# plotting / save options
SAVE_PLOT = True
SAVE_BASE_SPECTRUM = True

# optional preprocessing
NORMALIZE_EACH_SPECTRUM_TO_MAX = True
CLIP_NEGATIVE_TO_ZERO = False


# ============================================================
# Augmentation parameters
# ============================================================

@dataclass
class AugmentParams:
    # global shift applied to all peaks in one augmented sample
    global_shift_min: float = -5.0
    global_shift_max: float = 5.0

    # local shift applied independently to each peak
    local_shift_std: float = 1.5

    # intensity jitter: multiplicative, per peak
    intensity_jitter_std: float = 0.12

    # global multiplicative scaling for whole spectrum
    global_scale_min: float = 0.7
    global_scale_max: float = 1.3

    # FWHM range for each augmented spectrum
    fwhm_min: float = 10.0
    fwhm_max: float = 18.0

    # baseline
    baseline_offset_max: float = 0.03
    baseline_slope_max: float = 2.0e-4
    baseline_quad_max: float = 2.0e-7

    # broad fluorescence-like hump
    hump_probability: float = 0.5
    hump_amp_max: float = 0.08
    hump_width_min: float = 200.0
    hump_width_max: float = 700.0

    # additive gaussian noise relative to spectrum max
    noise_std_fraction_min: float = 0.005
    noise_std_fraction_max: float = 0.02

    # optional cosmic spikes
    spike_probability: float = 0.15
    n_spikes_min: int = 1
    n_spikes_max: int = 4
    spike_height_fraction_min: float = 0.03
    spike_height_fraction_max: float = 0.10


# ============================================================
# Core spectrum utilities
# ============================================================

def fwhm_to_sigma(fwhm: float) -> float:
    return fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def gaussian_spectrum(
    x: np.ndarray,
    peak_positions: np.ndarray,
    peak_intensities: np.ndarray,
    fwhm: float,
) -> np.ndarray:
    sigma = fwhm_to_sigma(fwhm)
    y = np.zeros_like(x, dtype=float)

    for mu, amp in zip(peak_positions, peak_intensities):
        y += amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    return y


def normalize_to_max(y: np.ndarray) -> np.ndarray:
    ymax = float(np.max(y))
    if ymax <= 0:
        return y.copy()
    return y / ymax


def add_polynomial_baseline(
    x: np.ndarray,
    rng: np.random.Generator,
    params: AugmentParams,
) -> np.ndarray:
    x0 = x - np.mean(x)

    offset = rng.uniform(0.0, params.baseline_offset_max)
    slope = rng.uniform(-params.baseline_slope_max, params.baseline_slope_max)
    quad = rng.uniform(-params.baseline_quad_max, params.baseline_quad_max)

    baseline = offset + slope * x0 + quad * (x0 ** 2)
    baseline -= baseline.min()  # keep baseline mostly non-negative
    return baseline


def add_fluorescence_hump(
    x: np.ndarray,
    rng: np.random.Generator,
    params: AugmentParams,
) -> np.ndarray:
    if rng.random() > params.hump_probability:
        return np.zeros_like(x)

    center = rng.uniform(x.min(), x.max())
    width = rng.uniform(params.hump_width_min, params.hump_width_max)
    amp = rng.uniform(0.0, params.hump_amp_max)

    return amp * np.exp(-0.5 * ((x - center) / width) ** 2)


def add_noise(
    y: np.ndarray,
    rng: np.random.Generator,
    params: AugmentParams,
) -> np.ndarray:
    ymax = max(float(np.max(y)), 1e-12)
    noise_std = rng.uniform(
        params.noise_std_fraction_min,
        params.noise_std_fraction_max,
    ) * ymax
    return y + rng.normal(loc=0.0, scale=noise_std, size=y.shape)


def add_spikes(
    y: np.ndarray,
    rng: np.random.Generator,
    params: AugmentParams,
) -> np.ndarray:
    out = y.copy()

    if rng.random() > params.spike_probability:
        return out

    ymax = max(float(np.max(out)), 1e-12)
    n_spikes = int(rng.integers(params.n_spikes_min, params.n_spikes_max + 1))
    indices = rng.choice(len(out), size=n_spikes, replace=False)

    for idx in indices:
        spike_height = rng.uniform(
            params.spike_height_fraction_min,
            params.spike_height_fraction_max,
        ) * ymax
        out[idx] += spike_height

    return out


# ============================================================
# Augmentation logic
# ============================================================

def augment_peaks(
    peak_positions: np.ndarray,
    peak_intensities: np.ndarray,
    rng: np.random.Generator,
    params: AugmentParams,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    global_shift = rng.uniform(params.global_shift_min, params.global_shift_max)
    local_shifts = rng.normal(0.0, params.local_shift_std, size=peak_positions.shape)

    new_positions = peak_positions + global_shift + local_shifts

    intensity_factors = 1.0 + rng.normal(
        0.0,
        params.intensity_jitter_std,
        size=peak_intensities.shape,
    )
    intensity_factors = np.clip(intensity_factors, 0.05, None)

    global_scale = rng.uniform(params.global_scale_min, params.global_scale_max)
    new_intensities = peak_intensities * intensity_factors

    fwhm = rng.uniform(params.fwhm_min, params.fwhm_max)
    return new_positions, new_intensities, fwhm, global_scale


def generate_augmented_spectrum(
    x: np.ndarray,
    peak_positions: np.ndarray,
    peak_intensities: np.ndarray,
    rng: np.random.Generator,
    params: AugmentParams,
) -> tuple[np.ndarray, dict]:
    aug_positions, aug_intensities, fwhm, global_scale = augment_peaks(
        peak_positions=peak_positions,
        peak_intensities=peak_intensities,
        rng=rng,
        params=params,
    )

    y = gaussian_spectrum(
        x=x,
        peak_positions=aug_positions,
        peak_intensities=aug_intensities,
        fwhm=fwhm,
    )

    y *= global_scale

    baseline = add_polynomial_baseline(x=x, rng=rng, params=params)
    hump = add_fluorescence_hump(x=x, rng=rng, params=params)

    spectrum = y + baseline * max(float(np.max(y)), 1.0) + hump * max(float(np.max(y)), 1.0)
    spectrum = add_noise(spectrum, rng=rng, params=params)
    spectrum = add_spikes(spectrum, rng=rng, params=params)

    if CLIP_NEGATIVE_TO_ZERO:
        spectrum = np.clip(spectrum, 0.0, None)

    if NORMALIZE_EACH_SPECTRUM_TO_MAX:
        spectrum = normalize_to_max(spectrum)

    metadata = {
        "fwhm": fwhm,
        "global_scale": global_scale,
        "mean_peak_shift": float(np.mean(aug_positions - peak_positions)),
        "std_peak_shift": float(np.std(aug_positions - peak_positions)),
    }
    return spectrum, metadata


# ============================================================
# I/O
# ============================================================

def load_peaks_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)

    required = {"wavenumber", "intensity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {sorted(missing)}. "
            f"Expected columns: {sorted(required)}"
        )

    df = df.copy()
    df = df.sort_values("wavenumber")

    peak_positions = df["wavenumber"].to_numpy(dtype=float)
    peak_intensities = df["intensity"].to_numpy(dtype=float)

    valid = np.isfinite(peak_positions) & np.isfinite(peak_intensities)
    peak_positions = peak_positions[valid]
    peak_intensities = peak_intensities[valid]

    if len(peak_positions) == 0:
        raise ValueError("No valid peaks found in input CSV.")

    return peak_positions, peak_intensities


def save_spectra_long_format(
    x: np.ndarray,
    base_spectrum: np.ndarray,
    augmented_spectra: list[np.ndarray],
    output_csv: Path,
) -> None:
    rows = []

    for xi, yi in zip(x, base_spectrum):
        rows.append(
            {
                "sample_id": "base",
                "wavenumber": xi,
                "intensity": yi,
            }
        )

    for i, spectrum in enumerate(augmented_spectra, start=1):
        sample_id = f"aug_{i:03d}"
        for xi, yi in zip(x, spectrum):
            rows.append(
                {
                    "sample_id": sample_id,
                    "wavenumber": xi,
                    "intensity": yi,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)


def save_spectra_wide_format(
    x: np.ndarray,
    base_spectrum: np.ndarray,
    augmented_spectra: list[np.ndarray],
    output_csv: Path,
) -> None:
    data = {"wavenumber": x, "base": base_spectrum}
    for i, spectrum in enumerate(augmented_spectra, start=1):
        data[f"aug_{i:03d}"] = spectrum

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)


def save_metadata(metadata_rows: list[dict], output_csv: Path) -> None:
    df = pd.DataFrame(metadata_rows)
    df.to_csv(output_csv, index=False)


def plot_example_spectra(
    x: np.ndarray,
    base_spectrum: np.ndarray,
    augmented_spectra: list[np.ndarray],
    output_png: Optional[Path] = None,
    n_examples: int = 8,
) -> None:
    plt.figure(figsize=(12, 7))
    plt.plot(x, base_spectrum, linewidth=2, label="base")

    for i, spectrum in enumerate(augmented_spectra[:n_examples], start=1):
        plt.plot(x, spectrum, alpha=0.8, label=f"aug_{i:03d}")

    plt.xlabel("Raman shift (cm$^{-1}$)")
    plt.ylabel("Intensity (a.u.)")
    plt.title("Base DFT spectrum and augmented variants")
    plt.legend()
    plt.tight_layout()

    if output_png is not None:
        plt.savefig(output_png, dpi=200)

    plt.show()


# ============================================================
# Main
# ============================================================

def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    params = AugmentParams()

    peak_positions, peak_intensities = load_peaks_csv(INPUT_PEAKS_CSV)

    x = np.arange(X_MIN, X_MAX + X_STEP, X_STEP, dtype=float)

    # Base spectrum
    base_fwhm = 12.0
    base_spectrum = gaussian_spectrum(
        x=x,
        peak_positions=peak_positions,
        peak_intensities=peak_intensities,
        fwhm=base_fwhm,
    )

    if NORMALIZE_EACH_SPECTRUM_TO_MAX:
        base_spectrum = normalize_to_max(base_spectrum)

    augmented_spectra: list[np.ndarray] = []
    metadata_rows: list[dict] = []

    for i in range(N_AUGMENTED):
        spectrum, meta = generate_augmented_spectrum(
            x=x,
            peak_positions=peak_positions,
            peak_intensities=peak_intensities,
            rng=rng,
            params=params,
        )
        augmented_spectra.append(spectrum)

        meta_row = {"sample_id": f"aug_{i + 1:03d}"}
        meta_row.update(meta)
        metadata_rows.append(meta_row)

    if SAVE_BASE_SPECTRUM:
        pd.DataFrame(
            {
                "wavenumber": x,
                "intensity": base_spectrum,
            }
        ).to_csv(OUTPUT_DIR / "base_spectrum.csv", index=False)

    save_spectra_long_format(
        x=x,
        base_spectrum=base_spectrum,
        augmented_spectra=augmented_spectra,
        output_csv=OUTPUT_DIR / "augmented_spectra_long.csv",
    )

    save_spectra_wide_format(
        x=x,
        base_spectrum=base_spectrum,
        augmented_spectra=augmented_spectra,
        output_csv=OUTPUT_DIR / "augmented_spectra_wide.csv",
    )

    save_metadata(
        metadata_rows=metadata_rows,
        output_csv=OUTPUT_DIR / "augmentation_metadata.csv",
    )

    if SAVE_PLOT:
        plot_example_spectra(
            x=x,
            base_spectrum=base_spectrum,
            augmented_spectra=augmented_spectra,
            output_png=OUTPUT_DIR / "augmented_spectra_examples.png",
            n_examples=8,
        )

    print("Done.")
    print(f"Input peaks: {INPUT_PEAKS_CSV}")
    print(f"Output dir:  {OUTPUT_DIR}")
    print(f"Generated augmented spectra: {len(augmented_spectra)}")


if __name__ == "__main__":
    main()