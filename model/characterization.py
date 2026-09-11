from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt

from tools.ts_params import TSP_F_HZ
from tools.utils import moving_average, stable_id, timed
from tools.utils import adc_id as get_adc_id
from adc.adc import RES_GEN_TYPE

@dataclass
class ADCCharacterization:
    W: np.ndarray
    m: dict
    n: dict
    slew_rates: np.ndarray
    adc_name: str | None = None
    adc_id: str | None = None
    metadata: dict | None = None
    simulation_parameters_id: str | None = None
    characterization_parameters_id: str | None = None

    @property
    def id(self):
        return stable_id({
            "adc_id": self.adc_id,
            "characterization_parameters_id": getattr(self, "characterization_parameters_id", None),
            "W": self.W,
            "m": self.m,
            "slew_rates": self.slew_rates,
            "metadata": self.metadata,
        })

    @property
    def x(self):
        return np.asarray(self.m["x"])

    @property
    def y(self):
        return np.asarray(self.m["y"])

    @property
    def w(self):
        return np.asarray(self.m["z"])


CharacterizationResult = ADCCharacterization


def coerce_characterization(value, m=None, n=None):
    if isinstance(value, ADCCharacterization):
        return value

    if m is not None:
        return ADCCharacterization(
            W=np.asarray(value),
            m=m,
            n={} if n is None else n,
            slew_rates=np.asarray(m.get("y", [])),
        )

    raise TypeError("Expected ADCCharacterization or legacy W together with m.")


# =============================================================================
# Ramp-file handling
# =============================================================================

def find_ramp_files(signal_dir):
    files = []
    slew_rates = []

    for path in Path(signal_dir).glob("ramp_*"):
        match = re.fullmatch(r"ramp_sr_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)_V_s(?:\..*)?", path.name)

        if match:
            files.append(str(path))
            slew_rates.append(float(match.group(1)))

    pairs = list(zip(slew_rates, files))
    pairs.sort()

    if not pairs:
        return [], []

    slew_rates, files = map(list, zip(*pairs))

    return slew_rates, files


@timed("run_ramp_characterization")
def run_ramp_characterization(adc, signal_dir, output_dir, run=True, progress=True, fs_Hz=None):
    slew_rates, files = find_ramp_files(signal_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    produced = []

    for i, (sr, filename) in enumerate(zip(slew_rates, files)):
        with open(filename, "rb") as f:
            series = pickle.load(f)

        run_fs_Hz = series.params[TSP_F_HZ] if fs_Hz is None else fs_Hz

        adc.reset()
        adc.load_input_signal(series)

        if run:
            adc.run(fs_Hz=run_fs_Hz, progress=progress)
            adc.clean_up()

            output_file = output_dir / f"{adc.name}_run_{i:02d}.pkl"

            with output_file.open("wb") as f:
                pickle.dump(adc, f)

            produced.append(output_file)

        if progress:
            print(f"Ramp {i + 1}/{len(files)} | {sr:g} V/s")

    return slew_rates, files, produced


# =============================================================================
# Characterization-run indexing
# =============================================================================

def get_characterization_run_index(adc_name, output_dir):
    """
    Obtain the list of saved characterization runs.

    Only lightweight metadata is retained. The complete ADC run objects are
    loaded one at a time and immediately released.
    """

    files = []
    pattern = re.compile(rf"^{re.escape(adc_name)}_run_(\d+)\.pkl$")

    for path in Path(output_dir).glob(f"{adc_name}_run_*.pkl"):
        match = pattern.fullmatch(path.name)

        if match:
            files.append((int(match.group(1)), str(path)))

    files.sort(key=lambda x: x[0])

    runs = []
    seen_slew_rates = set()

    for run_idx, path in files:
        with open(path, "rb") as f:
            adc_run = pickle.load(f)

        match = re.search(r"slew:([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", adc_run.input_signal.name)

        if match is None:
            print(f"⚠️ Could not extract slew rate from {path}")
            continue

        slew_rate = float(match.group(1))

        if slew_rate in seen_slew_rates:
            print(f"⚠️ Repeated slew rate {slew_rate:g} V/s in run {run_idx:02d}, ignoring it")
            continue

        seen_slew_rates.add(slew_rate)

        runs.append({
            "slew_rate": slew_rate,
            "run_idx": run_idx,
            "path": path,
        })

    runs.sort(key=lambda x: x["slew_rate"])

    return runs


def load_characterization_runs(adc_name, output_dir):
    """
    Compatibility helper returning the complete ADC run objects.

    Prefer get_characterization_run_index() internally when possible because
    it does not keep all characterization runs in memory.
    """

    run_index = get_characterization_run_index(adc_name, output_dir)

    runs = []

    for run in run_index:
        with open(run["path"], "rb") as f:
            adc_run = pickle.load(f)

        runs.append((
            run["slew_rate"],
            run["run_idx"],
            run["path"],
            adc_run,
        ))

    return runs


# =============================================================================
# Level-width extraction
# =============================================================================

def get_run_level_widths(adc_run, slew_rate):
    """
    Extract crossing input voltage, measured level width and direction from one
    empirical ramp run.
    """

    vin_tx_V = np.interp(
        x=adc_run.eb_txs_s,
        xp=adc_run.sim_input_signal.time,
        fp=adc_run.sim_input_signal.data,
    )[1:]

    dtx_tx_s = adc_run.get_eb_dtx_s()

    lvlw_tx_V = dtx_tx_s * slew_rate

    dirs = adc_run.get_eb_dirs()[1:]

    return vin_tx_V, lvlw_tx_V, dirs


# =============================================================================
# First pass: determine observed width range
# =============================================================================

def get_observed_width_range(adc_name, output_dir, progress=True):
    """
    Scan every characterization run and obtain the global minimum and maximum
    observed level width.

    The ADC simulations are not rerun. This function only reads the already
    saved characterization runs.
    """

    run_index = get_characterization_run_index(adc_name, output_dir)

    if len(run_index) == 0:
        raise RuntimeError(f"No characterization runs found for {adc_name} in {output_dir}")

    width_min_V = np.inf
    width_max_V = -np.inf
    width_count = 0

    for i, run in enumerate(run_index):
        with open(run["path"], "rb") as f:
            adc_run = pickle.load(f)

        _, lvlw_tx_V, _ = get_run_level_widths(adc_run, run["slew_rate"])

        lvlw_tx_V = np.asarray(lvlw_tx_V)
        lvlw_tx_V = lvlw_tx_V[np.isfinite(lvlw_tx_V)]

        if len(lvlw_tx_V) > 0:
            width_min_V = min(width_min_V, float(np.min(lvlw_tx_V)))
            width_max_V = max(width_max_V, float(np.max(lvlw_tx_V)))
            width_count += len(lvlw_tx_V)

        if progress:
            print(f"\rWidth-range pass: {i + 1}/{len(run_index)} ({100 * (i + 1) / len(run_index):5.1f}%) | SR={run['slew_rate']:g} V/s", end="")

    if progress:
        print()

    if width_count == 0:
        raise RuntimeError("No valid level widths were found in the characterization runs")

    if width_max_V <= width_min_V:
        raise RuntimeError(f"Observed level-width range is degenerate: {width_min_V:g} ... {width_max_V:g} V")

    # Expand by the smallest possible floating-point amount so that values
    # numerically equal to the observed extrema are guaranteed to fall inside
    # the histogram support.
    width_min_edge_V = np.nextafter(width_min_V, -np.inf)
    width_max_edge_V = np.nextafter(width_max_V, np.inf)

    if progress:
        print()
        print("Observed level-width range")
        print(f"  min : {width_min_V * 1e3:.6f} mV")
        print(f"  max : {width_max_V * 1e3:.6f} mV")
        print(f"  N   : {width_count}")

    return width_min_edge_V, width_max_edge_V


# =============================================================================
# Second pass: build W using automatically selected width bins
# =============================================================================

@timed("build_W_from_runs")
def build_W_from_runs(adc, output_dir, bins_n=127, simulation_parameters_id=None, characterization_parameters_id=None, characterization_fs_Hz=None, progress=True):
    """
    Build W(V,V',w) from saved ramp characterization runs.

    Two-pass procedure:

        1. Scan every saved ramp run to obtain the global observed minimum and
           maximum level width.

        2. Construct one common width grid using `bins_n` bins and scan the
           runs again to populate W(V,V',w).

    `bins_n` is the number of actual PMF bins. Therefore the histogram uses
    `bins_n + 1` bin edges.
    """

    if bins_n < 1:
        raise ValueError("bins_n must be >= 1")

    run_index = get_characterization_run_index(adc.name, output_dir)

    if len(run_index) == 0:
        raise RuntimeError(f"No characterization runs found for {adc.name} in {output_dir}")

    slew_rates = np.array([run["slew_rate"] for run in run_index], dtype=float)
    n_sr = len(run_index)

    codes = np.asarray(adc.comps.dac.codes_V)

    # -------------------------------------------------------------------------
    # Pass 1: determine width support
    # -------------------------------------------------------------------------

    width_min_V, width_max_V = get_observed_width_range(
        adc_name=adc.name,
        output_dir=output_dir,
        progress=progress,
    )

    width_edges_V = np.linspace(
        width_min_V,
        width_max_V,
        bins_n + 1,
    )

    width_centers_V = 0.5 * (
        width_edges_V[:-1] +
        width_edges_V[1:]
    )

    # -------------------------------------------------------------------------
    # Allocate W
    # -------------------------------------------------------------------------

    x_n = len(codes)
    y_n = 2 * n_sr
    z_n = bins_n

    W = np.zeros(
        (x_n, y_n, z_n),
        dtype=float,
    )
    capture_probability_VDV = np.zeros((x_n, y_n), dtype=float)
    crossing_error_pos_V = []
    crossing_error_neg_V = []
    dw_step_V = float(np.mean(np.diff(width_centers_V))) if len(width_centers_V) > 1 else max(float(width_centers_V[0]), 1e-12)
    delta_w_axis_V = np.arange(-(z_n - 1), z_n) * dw_step_V
    delta_w_edges_V = np.concatenate(([delta_w_axis_V[0] - 0.5 * dw_step_V], 0.5 * (delta_w_axis_V[:-1] + delta_w_axis_V[1:]), [delta_w_axis_V[-1] + 0.5 * dw_step_V]))
    delta_w_characterization_VDV = np.zeros((x_n, y_n, len(delta_w_axis_V)), dtype=float)

    # -------------------------------------------------------------------------
    # Pass 2: histogram each amplitude/slope population
    # -------------------------------------------------------------------------

    for sr, run in enumerate(run_index):
        slew_rate = run["slew_rate"]

        with open(run["path"], "rb") as f:
            adc_run = pickle.load(f)

        periods_match = re.search(r"x(\d+)\s+periods", adc_run.input_signal.name)

        if periods_match is None:
            raise RuntimeError(f"Could not extract number of ramp periods from '{adc_run.input_signal.name}'")

        periods_n = int(periods_match.group(1))

        vin_tx_V, lvlw_tx_V, dirs = get_run_level_widths(
            adc_run,
            slew_rate,
        )

        lvlw_pos = lvlw_tx_V[dirs > 0]
        vin_pos = vin_tx_V[dirs > 0]

        lvlw_neg = lvlw_tx_V[dirs < 0]
        vin_neg = vin_tx_V[dirs < 0]

        # Crossing-amplitude error relative to the nearest ideal DAC code.
        # This captures threshold/noise/delay uncertainty already present in the
        # ramp simulations and is later used to broaden reversal recapture error.
        ideal_pos = adc_run.comps.dac.Vss + np.clip(np.round((vin_pos - adc_run.comps.dac.Vss) / adc_run.comps.dac.lsb_V), 0, len(codes) - 1) * adc_run.comps.dac.lsb_V
        ideal_neg = adc_run.comps.dac.Vss + np.clip(np.round((vin_neg - adc_run.comps.dac.Vss) / adc_run.comps.dac.lsb_V), 0, len(codes) - 1) * adc_run.comps.dac.lsb_V
        crossing_error_pos_V.extend((vin_pos - ideal_pos).tolist())
        crossing_error_neg_V.extend((vin_neg - ideal_neg).tolist())

        lvlw_per_code_pos = [[] for _ in range(len(codes))]
        lvlw_per_code_neg = [[] for _ in range(len(codes))]
        delta_w_per_code_pos = [[] for _ in range(len(codes))]
        delta_w_per_code_neg = [[] for _ in range(len(codes))]

        # Paired consecutive widths from the same ramp realization retain the
        # spatial/DNL and stochastic correlation that is lost in marginal W.
        for vv, ww, out in ((vin_pos, lvlw_pos, delta_w_per_code_pos), (vin_neg, lvlw_neg, delta_w_per_code_neg)):
            for q in range(len(ww) - 1):
                c0 = int(round((vv[q] - adc_run.comps.dac.Vss) / adc_run.comps.dac.lsb_V))
                if 0 <= c0 < len(codes):
                    out[c0].append(float(ww[q + 1] - ww[q]))

        # Positive-going crossings
        for vin, lvlw in zip(vin_pos, lvlw_pos):
            lvl = int(round((vin - adc_run.comps.dac.Vss) / adc_run.comps.dac.lsb_V))

            if 0 <= lvl < len(codes):
                lvlw_per_code_pos[lvl].append(lvlw)

        # Negative-going crossings
        for vin, lvlw in zip(vin_neg, lvlw_neg):
            lvl = int(round((vin - adc_run.comps.dac.Vss) / adc_run.comps.dac.lsb_V))

            if 0 <= lvl < len(codes):
                lvlw_per_code_neg[lvl].append(lvlw)

        # Histogram each level independently
        for c in range(len(codes)):
            for lvlws, direction in [
                (lvlw_per_code_pos, "pos"),
                (lvlw_per_code_neg, "neg"),
            ]:

                if len(lvlws[c]) == 0:
                    continue

                counts, _ = np.histogram(
                    lvlws[c],
                    bins=width_edges_V,
                )

                if np.sum(counts) == 0:
                    continue

                captured = min(
                    1.0,
                    np.sum(counts) / periods_n,
                )

                counts = counts / np.sum(counts)

                if direction == "neg":
                    y = n_sr - 1 - sr
                else:
                    y = n_sr + sr

                W[c, y, :] = counts
                capture_probability_VDV[c, y] = captured

                delta_values = delta_w_per_code_neg[c] if direction == "neg" else delta_w_per_code_pos[c]
                if len(delta_values):
                    dcounts, _ = np.histogram(delta_values, bins=delta_w_edges_V)
                    if np.sum(dcounts) > 0:
                        delta_w_characterization_VDV[c, y, :] = dcounts / np.sum(dcounts)

        if progress:
            print(f"\rHistogram pass: {sr + 1}/{n_sr} ({100 * (sr + 1) / n_sr:5.1f}%) | SR={slew_rate:g} V/s", end="")

    if progress:
        print()

    # -------------------------------------------------------------------------
    # Axes
    # -------------------------------------------------------------------------

    m = {
        "x": codes,
        "y": np.concatenate((-slew_rates[::-1], slew_rates)),
        "z": width_centers_V,
    }

    n = {
        "x": "V (V)",
        "y": "V' (V/s)",
        "z": "Δtx.V'~LSB (V)",
    }

    metadata = {
        "width_bins_n": int(bins_n),
        "width_min_observed_V": float(width_min_V),
        "width_max_observed_V": float(width_max_V),
        "width_edges_V": width_edges_V,
        "characterization_fs_Hz": characterization_fs_Hz,
        "nominal_level_width_V": float(adc.dp.lsb_V),
        "adc_levels_n": int(adc.dp.lsb_n),
        "adc_amplitude_range_V": (
            float(adc.dp.Vss_V),
            float(adc.dp.Vdd_V),
        ),
        "emits_reversal_events": bool(getattr(adc.dp, "emits_reversal_events", True)),
        "capture_probability_VDV": capture_probability_VDV,
        "delta_w_axis_V": delta_w_axis_V,
        "delta_w_characterization_VDV": delta_w_characterization_VDV,
        "crossing_error_pos_mean_V": float(np.mean(crossing_error_pos_V)) if crossing_error_pos_V else 0.0,
        "crossing_error_pos_sigma_V": float(np.std(crossing_error_pos_V)) if crossing_error_pos_V else 0.0,
        "crossing_error_neg_mean_V": float(np.mean(crossing_error_neg_V)) if crossing_error_neg_V else 0.0,
        "crossing_error_neg_sigma_V": float(np.std(crossing_error_neg_V)) if crossing_error_neg_V else 0.0,
        "crossing_error_difference_sigma_V": float(np.hypot(np.std(crossing_error_pos_V), np.std(crossing_error_neg_V))) if crossing_error_pos_V and crossing_error_neg_V else 0.0,
        "dac_dnl_seed": getattr(adc.dp, "dac_dnl_seed", None),
        "noise_seed": getattr(adc.dp, "noise_seed", None),
    }

    return ADCCharacterization(
        W=W,
        m=m,
        n=n,
        slew_rates=slew_rates,
        adc_name=adc.name,
        adc_id=get_adc_id(adc),
        metadata=metadata,
        simulation_parameters_id=simulation_parameters_id,
        characterization_parameters_id=characterization_parameters_id,
    )


# =============================================================================
# Save characterization
# =============================================================================

def save_characterization(result, directory, adc_name, adc=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    files = {
        "W": directory / f"{adc_name}_W.pkl",
        "M_legacy": directory / f"{adc_name}_M.pkl",
        "m": directory / f"{adc_name}_m.pkl",
        "n": directory / f"{adc_name}_n.pkl",
    }

    for key, filename in files.items():
        if key in ("W", "M_legacy"):
            obj = result.W
        elif key == "m":
            obj = result.m
        else:
            obj = result.n

        with filename.open("wb") as f:
            pickle.dump(obj, f)

    if adc is not None:
        adc_file = directory / f"{adc_name}_adc.pkl"

        with adc_file.open("wb") as f:
            pickle.dump(adc, f)

        files["adc"] = adc_file

    return files

def overload_slope(min_level_width_V, update_time_s):
    """Slope for which the narrowest level is traversed in one update time."""
    return min_level_width_V / update_time_s


def overload_reached_by_signal(series, overload_slope_V_s):
    dt = np.diff(series.time)
    slope = np.diff(series.data) / dt
    max_abs_slope = np.max(np.abs(slope))
    return max_abs_slope >= overload_slope_V_s, max_abs_slope

def plot_characterization_ridgeline(adc, result, output_dir=None):
    """Plot the characterized W PMFs. Saved ramp-run files are not required."""
    slew_rates = np.asarray(result.slew_rates)
    n_sr = len(slew_rates)
    codes = adc.comps.dac.codes_V
    w = result.m["z"]
    ridge_spacing = 1.0
    ridge_height = 0.75 * ridge_spacing

    fig, ax = plt.subplots(figsize=(10, 6))
    for sr, slew_rate in enumerate(slew_rates):
        offset = sr * ridge_spacing
        for c in range(len(codes)):
            for y, color in [(n_sr + sr, "green"), (n_sr - 1 - sr, "red")]:
                counts = result.W[c, y]
                valid = counts > 0
                if not np.any(valid):
                    continue
                plot_bins = w[valid] + codes[c] - codes[1]
                plot_counts = moving_average(counts[valid], w=2)
                counts_max = np.max(plot_counts)
                if counts_max == 0:
                    continue
                ax.plot(plot_bins, offset + (plot_counts / counts_max) * ridge_height, color=color, linewidth=1)
        ax.axhline(offset, color="gray", linewidth=0.5)

    for code in codes:
        ax.axvline(code, color="black", linestyle="--", linewidth=0.5, alpha=0.4)

    ridge_positions = np.arange(n_sr) * ridge_spacing
    ax.set_yticks(ridge_positions)
    ax.set_yticklabels([f"{sr:g}" for sr in slew_rates])
    ax.set_ylim(-0.1 * ridge_spacing, (n_sr - 1) * ridge_spacing + ridge_spacing)
    ax.set_title("PMF ridgeline")
    ax.set_xlabel("Expected code + Effective LSB (Δtx·SR) [V]")
    ax.set_ylabel("Slew rate [V/s]")
    ax.set_xlim(-0.01, 1.01)
    fig.tight_layout()
    return fig, ax


def plot_dac_nonlinearity(adc):
    if adc.dp.res_gen_type != RES_GEN_TYPE.DAC_BASED_RES:
        return None

    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(6, 3), sharex=True)
    axs[0].vlines(range(adc.comps.dac.ncodes - 1), ymin=0, ymax=adc.comps.dac.dnl_LSB, color="k")
    axs[1].step(range(adc.comps.dac.ncodes), adc.comps.dac.inl_LSB, color="k")
    axs[1].set_xlabel("Code")
    axs[0].set_title("DAC non-linearity")
    fig.tight_layout()
    return fig, axs

def plot_characterization_deviation_heatmap(adc, result, dispersion_percentile=95.0):
    """Plot W(V,V') mean-width deviation with dispersion encoded as whiteness.

    Hue:
        negative mean deviation -> red
        zero deviation          -> green
        positive mean deviation -> blue

    Whiteness:
        0 sigma                  -> fully saturated hue
        sigma ~= mean width      -> essentially white / catastrophic dispersion

    The hue scale is deliberately expanded to at least +/- the global mean width.
    Therefore the colorbar directly shows what a deviation comparable with one
    complete level width would look like, even when the observed deviations are
    much smaller.
    """
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    from tools.utils import centers_to_edges

    W = np.asarray(result.W, dtype=float)
    w = np.asarray(result.m["z"], dtype=float)
    V = np.asarray(result.m["x"], dtype=float)
    DV = np.asarray(result.m["y"], dtype=float)

    mass = np.sum(W, axis=2)
    mean_w = np.divide(np.sum(W * w[None, None, :], axis=2), mass, out=np.full_like(mass, np.nan), where=mass > 0)
    variance_w = np.divide(np.sum(W * (w[None, None, :] - mean_w[:, :, None])**2, axis=2), mass, out=np.full_like(mass, np.nan), where=mass > 0)
    sigma_w = np.sqrt(np.maximum(variance_w, 0.0))

    valid = np.isfinite(mean_w) & (mass > 0)
    if not np.any(valid):
        raise RuntimeError("Characterization contains no populated W(V,V') cells.")

    # Weight all populated characterization states equally. This is a property of
    # W itself and intentionally does not introduce signal-dependent D2 weighting.
    global_mean_w = float(np.mean(mean_w[valid]))
    deviation_w = mean_w - global_mean_w
    max_abs_deviation = float(np.max(np.abs(deviation_w[valid])))

    # Keep +/- one complete mean level on the hue colorbar. This makes the plot
    # interpretable in absolute ADC terms rather than auto-scaling tiny deviations
    # to catastrophic-looking saturated colors.
    deviation_reference_V = max(max_abs_deviation, global_mean_w, 1e-12)

    sigma_valid = sigma_w[valid]
    sigma_percentile_V = float(np.percentile(sigma_valid, dispersion_percentile)) if len(sigma_valid) else 0.0
    # sigma ~= mean width is the physically useful "catastrophic" reference.
    sigma_white_V = max(sigma_percentile_V, global_mean_w, 1e-12)

    cmap = LinearSegmentedColormap.from_list("width_deviation_rgb", [(0.0, "red"), (0.5, "green"), (1.0, "blue")])
    norm = Normalize(vmin=-deviation_reference_V, vmax=deviation_reference_V)
    base_rgba = cmap(norm(np.nan_to_num(deviation_w, nan=0.0)))

    whiteness = np.clip(sigma_w / sigma_white_V, 0.0, 1.0)
    whiteness = np.nan_to_num(whiteness, nan=1.0)
    rgb = base_rgba[..., :3] * (1.0 - whiteness[..., None]) + whiteness[..., None]
    rgba = np.concatenate((rgb, np.ones((*rgb.shape[:2], 1))), axis=2)
    rgba[~valid, :3] = 0.92

    x_edges = centers_to_edges(V)
    y_edges = centers_to_edges(DV)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(np.transpose(rgba, (1, 0, 2)), origin="lower", aspect="auto", interpolation="nearest", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])

    # Hue colorbar: explicitly mark +/- one average level width.
    sm_mean = ScalarMappable(norm=norm, cmap=cmap)
    sm_mean.set_array([])
    cbar_mean = fig.colorbar(sm_mean, ax=ax, pad=0.02, label="Mean width deviation from characterized average")
    cbar_mean.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x*1e3:.1f} mV"))
    for value, label in ((-global_mean_w, "-mean w"), (global_mean_w, "+mean w")):
        cbar_mean.ax.axhline(value, color="black", linewidth=1.2, linestyle="--")
    cbar_mean.ax.text(1.15, 0.02, f"mean w = {global_mean_w*1e3:.2f} mV", transform=cbar_mean.ax.transAxes, rotation=90, va="bottom", ha="left", fontsize=8)

    # Whiteness legend: sigma == mean width is explicitly marked.
    sigma_cmap = LinearSegmentedColormap.from_list("dispersion_whiteness", ["black", "white"])
    sigma_norm = Normalize(vmin=0.0, vmax=sigma_white_V)
    sm_sigma = ScalarMappable(norm=sigma_norm, cmap=sigma_cmap)
    sm_sigma.set_array([])
    cbar_sigma = fig.colorbar(sm_sigma, ax=ax, pad=0.10, label=r"Whiteness scale: local $\sigma(w)$")
    cbar_sigma.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x*1e3:.1f} mV"))
    cbar_sigma.ax.axhline(global_mean_w, color="red", linewidth=1.5, linestyle="--")
    cbar_sigma.ax.text(1.15, min(0.98, global_mean_w / sigma_white_V), r"$\sigma=\bar{w}$", transform=cbar_sigma.ax.transAxes, rotation=90, va="top", ha="left", fontsize=8, color="red")

    ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("Input amplitude V (V)")
    ax.set_ylabel("Slew rate V' (V/s)")
    ax.set_title("Level-width mean deviation; whiteness encodes dispersion")
    ax.text(0.01, 0.01, f"mean w = {global_mean_w*1e3:.2f} mV\n95th percentile sigma(w) = {sigma_percentile_V*1e3:.2f} mV\nwhite reference = max(mean w, percentile sigma)", transform=ax.transAxes, ha="left", va="bottom", fontsize=8, bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=2))
    fig.tight_layout()
    return fig, ax
