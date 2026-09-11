from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tools.utils import DistributionStatistics, CrossingModeRateResults, get_distribution_statistics, adc_id, signal_id, stable_id, timed
from model.characterization import ADCCharacterization


# =============================================================================
# simulation.py
# =============================================================================

@dataclass
class EmpiricalRun:
    adc: object
    adc_id: str | None = None
    signal_id: str | None = None
    signal_name: str | None = None
    metadata: dict = field(default_factory=dict)

    def __getattr__(self, name):
        adc = self.__dict__.get("adc", None)
        if adc is None:
            raise AttributeError(name)
        return getattr(adc, name)

    @property
    def id(self):
        metadata = dict(self.metadata or {})
        metadata.pop("simulation_parameters_id", None)
        return stable_id({"adc_id": self.adc_id, "signal_id": self.signal_id, "metadata": metadata, "eb_txs_s": getattr(self.adc, "eb_txs_s", []), "eb_dirs": getattr(self.adc, "eb_dirs", [])})


def coerce_empirical_run(value):
    if isinstance(value, EmpiricalRun):
        return value
    if hasattr(value, "get_eb_tx_s") and hasattr(value, "input_signal"):
        return EmpiricalRun(adc=value, adc_id=adc_id(value), signal_id=signal_id(value.input_signal), signal_name=value.input_signal.name)
    raise TypeError("Expected EmpiricalRun or a legacy LC_ADC run object.")


@timed("run_adc")
def run_adc(adc, signal, fs_Hz=None, tf_s=None, save_to=None, progress=True, simulation_parameters_id=None, empirical_parameters_id=None):
    adc.reset()
    adc.load_input_signal(signal)
    adc.run(tf_s=tf_s, fs_Hz=fs_Hz, progress=progress)
    adc.clean_up()

    result = EmpiricalRun(adc=adc, adc_id=adc_id(adc), signal_id=signal_id(signal), signal_name=signal.name, metadata={"fs_Hz": fs_Hz, "tf_s": tf_s, "simulation_parameters_id": simulation_parameters_id, "empirical_parameters_id": empirical_parameters_id})

    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        with save_to.open("wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    return result


def load_run(filename):
    with open(filename, "rb") as f:
        return coerce_empirical_run(pickle.load(f))

# =============================================================================
# dynamic_range.py
# =============================================================================

@dataclass
class EmpiricalDynamicRangeResult:
    lvlw_max: float
    dynamic_range: float
    dynamic_range_dB: float
    ENOB: float
    min_x: float
    max_x: float


def _coverage_value(values, coverage):
    values = np.sort(np.asarray(values, dtype=float))
    if len(values) == 0:
        return np.nan
    index = min(len(values) - 1, int(np.ceil(coverage * len(values))) - 1)
    return values[max(index, 0)]


def get_empirical_max_lsb_and_range(widths, width_amplitudes_V, coverage, min_x, max_x):
    widths = np.asarray(widths, dtype=float)
    amplitudes = np.asarray(width_amplitudes_V, dtype=float)
    valid = (amplitudes >= min_x) & (amplitudes <= max_x) & np.isfinite(widths) & (widths > 0)
    selected = widths[valid]

    if len(selected) == 0:
        return {"lvlw_max": np.nan, "dynamic_range": np.nan, "dynamic_range_dB": np.nan, "ENOB": np.nan}

    lvlw_max = _coverage_value(selected, coverage)
    dynamic_range = (max_x - min_x) / lvlw_max
    return {"lvlw_max": lvlw_max, "dynamic_range": dynamic_range, "dynamic_range_dB": 20 * np.log10(dynamic_range), "ENOB": np.log2(dynamic_range)}


def find_empirical_maximum_dynamic_range(widths, width_amplitudes_V, range_min_x, range_max_x, coverage=0.997):
    range_min_x = np.asarray(range_min_x, dtype=float)
    range_max_x = np.asarray(range_max_x, dtype=float)
    DR_matrix = np.full((len(range_min_x), len(range_max_x)), np.nan, dtype=float)

    for i, min_x in enumerate(range_min_x):
        for j, max_x in enumerate(range_max_x):
            if max_x <= min_x:
                continue
            result = get_empirical_max_lsb_and_range(widths, width_amplitudes_V, coverage, min_x, max_x)
            DR_matrix[i, j] = result["dynamic_range_dB"]

    if not np.any(np.isfinite(DR_matrix)):
        return None, DR_matrix

    max_i, max_j = np.unravel_index(np.nanargmax(DR_matrix), DR_matrix.shape)
    best_min_x = range_min_x[max_i]
    best_max_x = range_max_x[max_j]
    best = get_empirical_max_lsb_and_range(widths, width_amplitudes_V, coverage, best_min_x, best_max_x)
    return EmpiricalDynamicRangeResult(best["lvlw_max"], best["dynamic_range"], best["dynamic_range_dB"], best["ENOB"], best_min_x, best_max_x), DR_matrix

# =============================================================================
# drift.py
# =============================================================================

@dataclass
class EmpiricalDriftResult:
    reference_width_V: float
    reconstructed_crossing_V: np.ndarray
    reconstruction_error_V: np.ndarray
    total_drift_V: float
    per_crossing_drift_V: float
    per_crossing_ratio: float
    total_ratio: float
    crossings_n: int
    same_direction_error_per_crossing_V: float
    reversal_error_per_crossing_V: float
    drift_rate_V_s: float | None = None
    duration_s: float | None = None
    expected_total_drift_V: float | None = None
    expected_total_ratio: float | None = None
    mode: str = "reversals_reported"


def compute_empirical_drift(vin, event_directions=None, average_width_V=None, reference_width_V=None, crossing_rate_Hz=None, duration_s=None, mode="reversals_reported"):
    """Compute signed reconstruction drift from crossing amplitudes.

    Pass-12 drift definition
    ------------------------
    The reconstruction assumes that every *full-level* event advances by the
    measured average same-direction level width ``average_width_V``.

    For a same-direction transition::

        error_increment = direction * (average_width - actual_width)

    For a reported reversal/recapture event the ideal amplitude displacement is
    zero.  Therefore its complete measured signed recapture displacement is an
    accumulated reconstruction error::

        error_increment = -actual_recapture_displacement

    With full-level hysteresis (``mode='reversals_suppressed'``) recapture events
    have already been removed from ``vin``.  Each remaining reported transition
    is consequently treated as a full-level output transition, including a
    direction change after a suppressed recapture.

    ``reference_width_V`` is retained only for backward API compatibility.  It
    no longer controls drift; the reconstruction width is always the empirical
    average same-direction width, as required by the drift definition.
    """
    # Legacy positional pattern: compute_empirical_drift(vin, average_width, reference_width)
    if event_directions is not None and np.ndim(event_directions) == 0 and reference_width_V is None and average_width_V is not None:
        reference_width_V = float(average_width_V)
        average_width_V = float(event_directions)
        event_directions = None

    vin = np.asarray(vin, dtype=float)
    if average_width_V is None:
        raise TypeError("average_width_V is required")
    reconstruction_width_V = float(average_width_V)

    if mode not in ("reversals_reported", "reversals_suppressed"):
        raise ValueError("mode must be 'reversals_reported' or 'reversals_suppressed'")

    if event_directions is None:
        if len(vin) <= 1:
            event_directions = np.ones(len(vin), dtype=float)
        else:
            interval_direction = np.sign(np.diff(vin))
            first = interval_direction[0] if interval_direction[0] != 0 else 1.0
            event_directions = np.concatenate(([first], interval_direction))

    event_directions = np.asarray(event_directions, dtype=float)
    crossings_n = len(vin)

    if crossings_n == 0:
        return EmpiricalDriftResult(reconstruction_width_V, np.array([]), np.array([]), np.nan, np.nan, np.nan, np.nan, 0, np.nan, np.nan, np.nan, duration_s, np.nan, np.nan, mode)

    if crossings_n == 1:
        rec = vin.copy()
        return EmpiricalDriftResult(reconstruction_width_V, rec, np.zeros_like(vin), 0.0, 0.0, 0.0, 0.0, 1, np.nan, np.nan, 0.0 if crossing_rate_Hz is not None else None, duration_s, 0.0 if duration_s is not None else None, 0.0 if duration_s is not None else None, mode)

    if len(event_directions) != crossings_n:
        raise ValueError("event_directions must have one entry per crossing")

    actual_steps = np.diff(vin)
    next_direction = np.sign(event_directions[1:])
    previous_direction = np.sign(event_directions[:-1])
    same = previous_direction * next_direction > 0
    reversal = previous_direction * next_direction < 0

    if mode == "reversals_reported":
        # Full levels advance by the average level width.  A reported recapture
        # should ideally return to exactly the previous crossing amplitude, so
        # its expected reconstruction displacement is zero.
        expected_steps = next_direction * reconstruction_width_V
        expected_steps[reversal] = 0.0
    else:
        # Recapture events are absent from this already-filtered sequence.  Every
        # remaining output transition represents a full-level reported event.
        expected_steps = next_direction * reconstruction_width_V

    transition_error = expected_steps - actual_steps
    reconstructed = vin[0] + np.concatenate(([0.0], np.cumsum(expected_steps)))
    error = reconstructed - vin
    total_drift_V = float(np.sum(transition_error))

    same_error = float(np.mean(transition_error[same])) if np.any(same) else np.nan
    if mode == "reversals_reported" and np.any(reversal):
        # Equivalent to mean(-actual_steps[reversal]), written explicitly to
        # emphasize that the ideal reversal displacement is zero.
        reversal_error = float(np.mean(-actual_steps[reversal]))
    else:
        reversal_error = np.nan

    drift_events_n = len(transition_error)
    per_crossing_drift_V = float(total_drift_V / drift_events_n) if drift_events_n else 0.0
    denominator = reconstruction_width_V
    per_crossing_ratio = per_crossing_drift_V / denominator if denominator != 0 else np.nan
    total_ratio = total_drift_V / denominator if denominator != 0 else np.nan

    # Empirical drift is known exactly from the measured sequence: accumulate
    # first, then divide by duration.  Do not reconstruct the total by
    # multiplying by an independently estimated crossing rate.
    drift_rate_V_s = None
    expected_total_drift_V = None
    expected_total_ratio = None
    if duration_s is not None and np.isfinite(duration_s) and duration_s > 0:
        drift_rate_V_s = float(total_drift_V / duration_s)
        expected_total_drift_V = total_drift_V
        expected_total_ratio = total_ratio
    elif crossing_rate_Hz is not None and np.isfinite(crossing_rate_Hz):
        # Compatibility fallback when a duration is unavailable.
        drift_rate_V_s = float(per_crossing_drift_V * crossing_rate_Hz)

    return EmpiricalDriftResult(
        reference_width_V=reconstruction_width_V,
        reconstructed_crossing_V=reconstructed,
        reconstruction_error_V=error,
        total_drift_V=total_drift_V,
        per_crossing_drift_V=per_crossing_drift_V,
        per_crossing_ratio=per_crossing_ratio,
        total_ratio=total_ratio,
        crossings_n=crossings_n,
        same_direction_error_per_crossing_V=same_error,
        reversal_error_per_crossing_V=reversal_error,
        drift_rate_V_s=drift_rate_V_s,
        duration_s=None if duration_s is None else float(duration_s),
        expected_total_drift_V=expected_total_drift_V,
        expected_total_ratio=expected_total_ratio,
        mode=mode,
    )

# =============================================================================
# metrics.py
# =============================================================================

@dataclass
class EmpiricalMetrics:
    crossings: int
    widths: np.ndarray
    delta_w: np.ndarray
    reversal_error_V: np.ndarray
    crossing_rate_Hz: np.ndarray
    average_crossing_rate_Hz: float
    run_average_crossing_rate_Hz: float
    interval_mean_crossing_rate_Hz: float
    sigma_crossing_rate_Hz: float
    width_statistics: DistributionStatistics
    delta_w_statistics: DistributionStatistics
    reversal_recapture_statistics: DistributionStatistics
    same_direction_fraction: float
    reversal_fraction: float
    same_direction_rate_Hz: float
    reversal_rate_Hz: float
    global_distortion_ratio: float
    local_sigma_ratio: float
    local_rms_ratio: float
    reversal_recapture_sigma_ratio: float
    reversal_recapture_rms_ratio: float
    dynamic_range: EmpiricalDynamicRangeResult | None
    drift: EmpiricalDriftResult
    crossing_rate_reversals_reported: CrossingModeRateResults
    crossing_rate_reversals_suppressed: CrossingModeRateResults
    drift_reversals_reported: EmpiricalDriftResult
    drift_reversals_suppressed: EmpiricalDriftResult
    suppressed_event_indices: np.ndarray

    def summary(self):
        return empirical_summary(self)


def extract_crossing_data(run):
    run = coerce_empirical_run(run)
    tx_s = np.asarray(run.get_eb_tx_s(), dtype=float)
    dtx_s = np.diff(tx_s)
    vin = np.asarray(run.get_eb_vin(), dtype=float)
    event_direction = np.asarray(run.get_eb_dirs(), dtype=float)
    if len(event_direction) != len(vin):
        event_direction = np.sign(np.concatenate(([0.0], np.diff(vin))))
        if len(event_direction) > 1 and event_direction[0] == 0:
            event_direction[0] = event_direction[1]

    dvin = np.diff(vin)
    interval_width = np.abs(dvin)
    same_direction = event_direction[:-1] * event_direction[1:] > 0
    reversal = event_direction[:-1] * event_direction[1:] < 0

    widths_same = interval_width[same_direction]
    width_amplitude_V = vin[1:][same_direction]
    reversal_error_V = dvin[reversal]

    # Delta-w is defined only when three successive reported crossing directions
    # are the same, so both adjacent intervals are genuine level traversals.
    delta_w_all = np.diff(interval_width)
    same_triplet = (event_direction[:-2] * event_direction[1:-1] > 0) & (event_direction[1:-1] * event_direction[2:] > 0)
    delta_w_same = delta_w_all[same_triplet]

    # Full-level hysteresis suppresses the first event after each raw direction
    # reversal. Keeping only non-recapture events merges the elapsed time across
    # those suppressed events without requiring the original input trajectory.
    reversal_event = np.zeros(len(vin), dtype=bool)
    if len(vin) > 1:
        reversal_event[1:] = reversal
    suppressed_event_indices = np.flatnonzero(~reversal_event)

    return {
        "tx_s": tx_s,
        "dtx_s": dtx_s,
        "vin": vin,
        "dvin": dvin,
        "event_direction": event_direction,
        "interval_width": interval_width,
        "same_direction": same_direction,
        "reversal": reversal,
        "widths_same": widths_same,
        "width_amplitude_V": width_amplitude_V,
        "reversal_error_V": reversal_error_V,
        "delta_w_same": delta_w_same,
        "suppressed_event_indices": suppressed_event_indices,
    }


def _empty_statistics():
    return get_distribution_statistics(np.array([0.0]), np.array([0.0]))


def _stats(values):
    values = np.asarray(values, dtype=float)
    return get_distribution_statistics(values, np.ones_like(values)) if len(values) else _empty_statistics()


def _rate_mode(name, tx_s, duration_s):
    tx_s = np.asarray(tx_s, dtype=float)
    dtx = np.diff(tx_s)
    dtx = dtx[np.isfinite(dtx) & (dtx > 0)]
    rates = np.divide(1.0, dtx, out=np.full_like(dtx, np.nan), where=dtx > 0)
    rates = rates[np.isfinite(rates)]

    mean_rate = float(len(tx_s) / duration_s) if duration_s > 0 else np.nan
    interval_mean = float(np.mean(rates)) if len(rates) else np.nan
    sigma_rate = float(np.std(rates)) if len(rates) else np.nan
    min_rate = float(np.min(rates)) if len(rates) else np.nan
    max_rate = float(np.max(rates)) if len(rates) else np.nan
    mean_interval = float(np.mean(dtx)) if len(dtx) else np.nan
    sigma_interval = float(np.std(dtx)) if len(dtx) else np.nan

    return CrossingModeRateResults(
        name=name,
        mean_rate_Hz=mean_rate,
        interval_mean_rate_Hz=interval_mean,
        sigma_rate_Hz=sigma_rate,
        min_rate_Hz=min_rate,
        max_rate_Hz=max_rate,
        mean_interval_s=mean_interval,
        sigma_interval_s=sigma_interval,
        rate_samples_Hz=rates,
        events_n=float(len(tx_s)),
    )


@timed("compute_empirical_metrics")
def compute_empirical_metrics(run, range_min_x=None, range_max_x=None, coverage=0.997, reference_width_V=None):
    run = coerce_empirical_run(run)
    data = extract_crossing_data(run)
    widths = data["widths_same"]
    delta_w = data["delta_w_same"]
    reversal_error = data["reversal_error_V"]

    sim_time = np.asarray(run.adc.sim_input_signal.time, dtype=float)
    duration_s = float(sim_time[-1] - sim_time[0]) if len(sim_time) > 1 else np.nan

    reported_mode = _rate_mode("reversals_reported", data["tx_s"], duration_s)
    suppressed_idx = data["suppressed_event_indices"]
    suppressed_mode = _rate_mode("reversals_suppressed", data["tx_s"][suppressed_idx], duration_s)

    width_stats = _stats(widths)
    delta_stats = _stats(delta_w)
    reversal_stats = _stats(reversal_error)

    denominator = width_stats.mean
    global_distortion_ratio = width_stats.sigma / denominator if denominator != 0 else np.nan
    local_sigma_ratio = delta_stats.sigma / denominator if denominator != 0 else np.nan
    local_rms_ratio = np.sqrt(delta_stats.mean**2 + delta_stats.sigma**2) / denominator if denominator != 0 else np.nan

    if reference_width_V is None:
        reference_width_V = float(run.adc.dp.lsb_V)
    reversal_recapture_sigma_ratio = reversal_stats.sigma / reference_width_V if reference_width_V else np.nan
    reversal_recapture_rms_ratio = np.sqrt(reversal_stats.mean**2 + reversal_stats.sigma**2) / reference_width_V if reference_width_V else np.nan

    transitions_n = len(data["dvin"])
    same_fraction = float(np.sum(data["same_direction"]) / transitions_n) if transitions_n else np.nan
    reversal_fraction = float(np.sum(data["reversal"]) / transitions_n) if transitions_n else np.nan
    same_direction_rate_Hz = reported_mode.mean_rate_Hz * same_fraction if np.isfinite(reported_mode.mean_rate_Hz) else np.nan
    reversal_rate_Hz = reported_mode.mean_rate_Hz * reversal_fraction if np.isfinite(reported_mode.mean_rate_Hz) else np.nan

    dynamic_range = None
    if range_min_x is not None and range_max_x is not None and len(widths):
        dynamic_range, _ = find_empirical_maximum_dynamic_range(widths, data["width_amplitude_V"], np.asarray(range_min_x), np.asarray(range_max_x), coverage=coverage)

    drift_reported = compute_empirical_drift(data["vin"], data["event_direction"], width_stats.mean, reference_width_V, crossing_rate_Hz=reported_mode.mean_rate_Hz, duration_s=duration_s, mode="reversals_reported")
    drift_suppressed = compute_empirical_drift(data["vin"][suppressed_idx], data["event_direction"][suppressed_idx], width_stats.mean, reference_width_V, crossing_rate_Hz=suppressed_mode.mean_rate_Hz, duration_s=duration_s, mode="reversals_suppressed")

    return EmpiricalMetrics(
        crossings=len(data["vin"]),
        widths=widths,
        delta_w=delta_w,
        reversal_error_V=reversal_error,
        crossing_rate_Hz=np.asarray(reported_mode.rate_samples_Hz),
        average_crossing_rate_Hz=reported_mode.mean_rate_Hz,
        run_average_crossing_rate_Hz=reported_mode.mean_rate_Hz,
        interval_mean_crossing_rate_Hz=reported_mode.interval_mean_rate_Hz,
        sigma_crossing_rate_Hz=reported_mode.sigma_rate_Hz,
        width_statistics=width_stats,
        delta_w_statistics=delta_stats,
        reversal_recapture_statistics=reversal_stats,
        same_direction_fraction=same_fraction,
        reversal_fraction=reversal_fraction,
        same_direction_rate_Hz=same_direction_rate_Hz,
        reversal_rate_Hz=reversal_rate_Hz,
        global_distortion_ratio=global_distortion_ratio,
        local_sigma_ratio=local_sigma_ratio,
        local_rms_ratio=local_rms_ratio,
        reversal_recapture_sigma_ratio=reversal_recapture_sigma_ratio,
        reversal_recapture_rms_ratio=reversal_recapture_rms_ratio,
        dynamic_range=dynamic_range,
        drift=drift_reported,
        crossing_rate_reversals_reported=reported_mode,
        crossing_rate_reversals_suppressed=suppressed_mode,
        drift_reversals_reported=drift_reported,
        drift_reversals_suppressed=drift_suppressed,
        suppressed_event_indices=suppressed_idx,
    )


def _mode_summary(prefix, mode):
    return {
        f"{prefix}_average_crossing_rate_Hz": float(mode.mean_rate_Hz),
        f"{prefix}_interval_mean_crossing_rate_Hz": float(mode.interval_mean_rate_Hz),
        f"{prefix}_sigma_crossing_rate_Hz": float(mode.sigma_rate_Hz),
        f"{prefix}_minimum_crossing_rate_Hz": float(mode.min_rate_Hz),
        f"{prefix}_maximum_crossing_rate_Hz": float(mode.max_rate_Hz),
        f"{prefix}_mean_crossing_interval_s": float(mode.mean_interval_s),
        f"{prefix}_sigma_crossing_interval_s": float(mode.sigma_interval_s),
    }


def empirical_summary(metrics):
    summary = {
        "crossings": metrics.crossings,
        "average_crossing_rate_Hz": float(metrics.crossing_rate_reversals_reported.mean_rate_Hz),
        "run_average_crossing_rate_Hz": float(metrics.crossing_rate_reversals_reported.mean_rate_Hz),
        "interval_mean_crossing_rate_Hz": float(metrics.crossing_rate_reversals_reported.interval_mean_rate_Hz),
        "sigma_crossing_rate_Hz": float(metrics.crossing_rate_reversals_reported.sigma_rate_Hz),
        "minimum_crossing_rate_Hz": float(metrics.crossing_rate_reversals_reported.min_rate_Hz),
        "maximum_crossing_rate_Hz": float(metrics.crossing_rate_reversals_reported.max_rate_Hz),
        "same_direction_fraction": float(metrics.same_direction_fraction),
        "reversal_fraction": float(metrics.reversal_fraction),
        "same_direction_rate_Hz": float(metrics.same_direction_rate_Hz),
        "reversal_rate_Hz": float(metrics.reversal_rate_Hz),
        "average_level_width_V": float(metrics.width_statistics.mean),
        "sigma_level_width_V": float(metrics.width_statistics.sigma),
        "global_distortion_ratio": float(metrics.global_distortion_ratio),
        "local_distortion_sigma_ratio": float(metrics.local_sigma_ratio),
        "local_distortion_rms_ratio": float(metrics.local_rms_ratio),
        "reversal_recapture_mean_V": float(metrics.reversal_recapture_statistics.mean),
        "reversal_recapture_sigma_V": float(metrics.reversal_recapture_statistics.sigma),
        "reversal_recapture_sigma_ratio": float(metrics.reversal_recapture_sigma_ratio),
        "reversal_recapture_rms_ratio": float(metrics.reversal_recapture_rms_ratio),
        "drift_per_crossing_ratio": float(metrics.drift_reversals_reported.per_crossing_ratio),
        "drift_rate_V_s": None if metrics.drift_reversals_reported.drift_rate_V_s is None else float(metrics.drift_reversals_reported.drift_rate_V_s),
        "drift_total_ratio": float(metrics.drift_reversals_reported.total_ratio),
        "drift_total_V": float(metrics.drift_reversals_reported.total_drift_V),
        "drift_reference_width_V": float(metrics.drift_reversals_reported.reference_width_V),
    }
    summary.update(_mode_summary("reversals_reported", metrics.crossing_rate_reversals_reported))
    summary.update(_mode_summary("reversals_suppressed", metrics.crossing_rate_reversals_suppressed))

    for prefix, drift in (("reversals_reported", metrics.drift_reversals_reported), ("reversals_suppressed", metrics.drift_reversals_suppressed)):
        summary.update({
            f"{prefix}_drift_per_crossing_V": float(drift.per_crossing_drift_V),
            f"{prefix}_drift_per_crossing_ratio": float(drift.per_crossing_ratio),
            f"{prefix}_drift_rate_V_s": None if drift.drift_rate_V_s is None else float(drift.drift_rate_V_s),
            f"{prefix}_expected_total_drift_V": None if drift.expected_total_drift_V is None else float(drift.expected_total_drift_V),
            f"{prefix}_expected_total_ratio": None if drift.expected_total_ratio is None else float(drift.expected_total_ratio),
        })

    if metrics.dynamic_range is not None:
        summary.update({
            "maximum_dynamic_range": float(metrics.dynamic_range.dynamic_range),
            "maximum_dynamic_range_dB": float(metrics.dynamic_range.dynamic_range_dB),
            "maximum_dynamic_range_ENOB": float(metrics.dynamic_range.ENOB),
            "dynamic_range_min_x_V": float(metrics.dynamic_range.min_x),
            "dynamic_range_max_x_V": float(metrics.dynamic_range.max_x),
            "dynamic_range_lvlw_max_V": float(metrics.dynamic_range.lvlw_max),
        })
    return summary

# =============================================================================
# diagnostics.py
# =============================================================================

def summarize_run(run):
    data = extract_crossing_data(run)
    widths = np.asarray(data["widths_same"])
    dtx_s = np.asarray(data["dtx_s"])
    return {
        "crossings": int(len(data["vin"])),
        "measured_widths": int(len(widths)),
        "width_min_V": float(np.min(widths)) if len(widths) else np.nan,
        "width_max_V": float(np.max(widths)) if len(widths) else np.nan,
        "dtx_min_s": float(np.min(dtx_s)) if len(dtx_s) else np.nan,
        "dtx_max_s": float(np.max(dtx_s)) if len(dtx_s) else np.nan,
    }


def summarize_characterization(characterization):
    W = np.asarray(characterization.W)
    occupied = np.sum(W, axis=2) > 0
    return {
        "shape": W.shape,
        "occupied_V_DV_cells": int(np.sum(occupied)),
        "total_V_DV_cells": int(occupied.size),
        "occupied_fraction": float(np.mean(occupied)),
        "W_min_nonzero": float(np.min(W[W > 0])) if np.any(W > 0) else np.nan,
        "W_max": float(np.max(W)) if W.size else np.nan,
    }

def plot_empirical_reconstruction(run, metrics=None):
    """Plot the empirical constant-width reconstruction against the ADC input.

    Two reconstruction policies are shown when ``metrics`` is available:
      * reversals reported: reversal/recapture events are output events whose
        ideal reconstruction displacement is zero;
      * reversals suppressed: recapture events are removed and the reconstruction
        advances only on the remaining full-level events.

    A second figure shows the accumulated reconstruction error at reported events.
    This makes drift, missed levels, wrong reversal handling and long-term offsets
    visually apparent.
    """
    run = coerce_empirical_run(run)
    data = extract_crossing_data(run)

    if metrics is None:
        metrics = compute_empirical_metrics(run)

    tx_all = np.asarray(data["tx_s"], dtype=float)
    vin_all = np.asarray(data["vin"], dtype=float)

    input_series = run.adc.sim_input_signal
    input_t = np.asarray(input_series.time, dtype=float)
    input_v = np.asarray(input_series.data, dtype=float)

    drift_reported = metrics.drift_reversals_reported
    drift_suppressed = metrics.drift_reversals_suppressed
    suppressed_idx = np.asarray(metrics.suppressed_event_indices, dtype=int)

    tx_suppressed = tx_all[suppressed_idx]
    vin_suppressed = vin_all[suppressed_idx]

    # Reconstruction figure.  The input is shown continuously; the reconstructions
    # are sample-and-hold trajectories defined only at reported output events.
    fig_rec, ax_rec = plt.subplots(figsize=(10, 4))
    ax_rec.plot(input_t, input_v, linewidth=1.0, label="ADC input")
    ax_rec.step(tx_all, drift_reported.reconstructed_crossing_V, where="post", linewidth=1.2, label=f"Reconstruction — reversals reported ({drift_reported.reference_width_V*1e3:.2f} mV step)")
    ax_rec.step(tx_suppressed, drift_suppressed.reconstructed_crossing_V, where="post", linewidth=1.2, label=f"Reconstruction — reversals suppressed ({drift_suppressed.reference_width_V*1e3:.2f} mV step)")
    ax_rec.scatter(tx_all, vin_all, s=8, alpha=0.35, label="Measured crossing input")
    ax_rec.set_xlabel("Time (s)")
    ax_rec.set_ylabel("Amplitude (V)")
    ax_rec.set_title("Empirical input and constant-width reconstruction")
    ax_rec.legend(loc="best")
    ax_rec.grid(alpha=0.3)
    fig_rec.tight_layout()

    # Accumulated reconstruction-error figure.
    fig_err, ax_err = plt.subplots(figsize=(10, 4))
    ax_err.plot(tx_all, drift_reported.reconstruction_error_V, linewidth=1.2, label="Error — reversals reported")
    ax_err.plot(tx_suppressed, drift_suppressed.reconstruction_error_V, linewidth=1.2, label="Error — reversals suppressed")
    ax_err.axhline(0.0, linewidth=1.0, label="Ideal = 0 V")
    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Reconstruction error (V)")
    ax_err.set_title("Accumulated empirical reconstruction error")
    ax_err.legend(loc="best")
    ax_err.grid(alpha=0.3)
    fig_err.tight_layout()

    return (fig_rec, ax_rec), (fig_err, ax_err)


def plot_empirical_crossings(run):
    """Plot empirical widths and signed drift increments without mixing units/meaning.

    Two separate figures are produced:
      1. Same-direction physical crossing widths (always positive).
      2. Signed reconstruction-error increment for every transition.  Same-direction
         events use direction*(mean_width-width); reported reversals use -dVin because
         their ideal recapture displacement is zero.
    """
    data = extract_crossing_data(run)
    t = np.asarray(data["tx_s"][1:], dtype=float)
    same = np.asarray(data["same_direction"], dtype=bool)
    reversal = np.asarray(data["reversal"], dtype=bool)
    event_direction = np.asarray(data["event_direction"], dtype=float)
    interval_width = np.asarray(data["interval_width"], dtype=float)
    dvin = np.asarray(data["dvin"], dtype=float)

    mean_width = float(np.mean(data["widths_same"])) if len(data["widths_same"]) else np.nan

    # Figure 1: physical level widths only. Reversals are deliberately excluded,
    # because a recapture displacement is not a level width.
    fig_width, ax_width = plt.subplots(figsize=(9, 4))
    ax_width.scatter(t[same], interval_width[same], s=8, label="Empirical same-direction width")
    if np.isfinite(mean_width):
        ax_width.axhline(mean_width, linestyle="--", linewidth=1.2, label=f"Empirical mean width = {mean_width*1e3:.2f} mV")
    ax_width.set_xlabel("Time (s)")
    ax_width.set_ylabel("Level width (V)")
    ax_width.set_title("Empirical same-direction level widths")
    ax_width.legend(loc="upper left")
    ax_width.grid(alpha=0.3)
    fig_width.tight_layout()

    # Figure 2: quantities that actually accumulate into drift. Both same-direction
    # and reversal contributions are signed and have the same physical meaning.
    next_direction = np.sign(event_direction[1:])
    drift_increment = np.full_like(dvin, np.nan, dtype=float)
    if np.isfinite(mean_width):
        drift_increment[same] = next_direction[same] * (mean_width - interval_width[same])
    drift_increment[reversal] = -dvin[reversal]

    fig_drift, ax_drift = plt.subplots(figsize=(9, 4))
    ax_drift.scatter(t[same], drift_increment[same], s=8, label="Same-direction drift increment")
    ax_drift.scatter(t[reversal], drift_increment[reversal], s=12, marker="x", label="Reversal drift increment")
    ax_drift.axhline(0.0, linewidth=1.0, label="Ideal = 0 V")
    ax_drift.set_xlabel("Time (s)")
    ax_drift.set_ylabel("Signed reconstruction-error increment (V)")
    ax_drift.set_title("Empirical signed drift contribution per reported transition")
    ax_drift.legend(loc="upper left")
    ax_drift.grid(alpha=0.3)
    fig_drift.tight_layout()

    return (fig_width, ax_width), (fig_drift, ax_drift)

