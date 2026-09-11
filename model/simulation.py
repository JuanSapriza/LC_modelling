from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
import numpy as np

from tools.ts_params import TSP_F_HZ
from tools.utils import stable_id

@dataclass(frozen=True)
class CharacterizationParameters:
    """Numerical settings used to generate ramps and characterize W(V,V',w)."""

    amplitude_low_V: float | None = None
    amplitude_high_V: float | None = None
    periods_n: int = 20
    ramp_half_intervals_n: int = 1000

    slew_range_low_V_s: float = 1.0
    slew_range_high_V_s: float = 400.0
    slew_range_steps_n: int = 100
    slew_range_type: str = "linear"

    sampling_frequency_Hz: float | None = None
    width_pmf_bins_n: int = 127

    def amplitude_range_for_adc(self, adc):
        low = adc.dp.Vss_V if self.amplitude_low_V is None else self.amplitude_low_V
        high = adc.dp.Vdd_V if self.amplitude_high_V is None else self.amplitude_high_V
        return float(low), float(high)

    def validate(self):
        if self.periods_n < 1:
            raise ValueError("characterization.periods_n must be >= 1")
        if self.ramp_half_intervals_n < 2:
            raise ValueError("characterization.ramp_half_intervals_n must be >= 2")
        if self.slew_range_low_V_s <= 0 or self.slew_range_high_V_s <= self.slew_range_low_V_s:
            raise ValueError("Characterization slew range must satisfy 0 < low < high")
        if self.slew_range_steps_n < 1:
            raise ValueError("characterization.slew_range_steps_n must be >= 1")
        if self.slew_range_type != "linear":
            raise ValueError("characterization.slew_range_type must be 'linear'. Characterization slopes are intentionally independent from the signal and linearly spaced.")
        if self.sampling_frequency_Hz is not None and self.sampling_frequency_Hz <= 0:
            raise ValueError("characterization.sampling_frequency_Hz must be > 0 or None")
        if self.width_pmf_bins_n < 2:
            raise ValueError("characterization.width_pmf_bins_n must be >= 2")


@dataclass(frozen=True)
class SignalStatisticsParameters:
    """Independent linear discretization used for D2(V,V',V'')."""

    amplitude_bins_n: int | None = None
    amplitude_min_V: float | None = None
    amplitude_max_V: float | None = None

    derivative_bins_n: int = 150
    derivative_min_V_s: float | None = None
    derivative_max_V_s: float | None = None

    second_derivative_bins_n: int = 64
    second_derivative_min_V_s2: float | None = None
    second_derivative_max_V_s2: float | None = None

    def validate(self, model_order="D2"):
        order = str(model_order).upper()
        rank = {"W": -1, "D0": 0, "D1": 1, "D2": 2}.get(order)
        if rank is None:
            raise ValueError("model_order must be one of: W, D0, D1, D2")
        if self.amplitude_bins_n is not None and self.amplitude_bins_n < 1:
            raise ValueError("signal_statistics.amplitude_bins_n must be >= 1 or None")
        if self.amplitude_min_V is not None and self.amplitude_max_V is not None and self.amplitude_max_V <= self.amplitude_min_V:
            raise ValueError("Signal-statistics amplitude range must satisfy min < max")
        if rank >= 1:
            if self.derivative_bins_n < 2:
                raise ValueError("signal_statistics.derivative_bins_n must be >= 2")
            if self.derivative_bins_n % 2 != 0:
                raise ValueError("signal_statistics.derivative_bins_n must be EVEN so V'=0 is a bin edge.")
            if self.derivative_min_V_s is not None and self.derivative_max_V_s is not None and self.derivative_max_V_s <= self.derivative_min_V_s:
                raise ValueError("Signal-statistics derivative range must satisfy min < max")
        if rank >= 2:
            if self.second_derivative_bins_n < 2:
                raise ValueError("signal_statistics.second_derivative_bins_n must be >= 2")
            if self.second_derivative_bins_n % 2 != 0:
                raise ValueError("signal_statistics.second_derivative_bins_n must be EVEN so V''=0 is a bin edge.")
            if self.second_derivative_min_V_s2 is not None and self.second_derivative_max_V_s2 is not None and self.second_derivative_max_V_s2 <= self.second_derivative_min_V_s2:
                raise ValueError("Signal-statistics second-derivative range must satisfy min < max")


@dataclass(frozen=True)
class StatisticalSimulationParameters:
    """Numerical resolution and maximum statistical model order.

    model_order selects the deepest model evaluated: W, D0, D1 or D2.
    Lower-order models are also reported when available.
    """

    model_order: str = "D2"
    headstart_bins_n: int | None = 21
    reversal_error_bins_n: int = 127
    crossing_time_bins_n: int = 192
    transition_probability_floor: float = 1e-6
    max_state_transitions_n: int = 64

    def validate(self):
        order = str(self.model_order).upper()
        if order not in {"W", "D0", "D1", "D2"}:
            raise ValueError("statistical.model_order must be one of: W, D0, D1, D2")
        if self.headstart_bins_n is not None and self.headstart_bins_n < 2:
            raise ValueError("statistical.headstart_bins_n must be >= 2 or None")
        if self.reversal_error_bins_n < 3:
            raise ValueError("statistical.reversal_error_bins_n must be >= 3")
        if self.crossing_time_bins_n < 8:
            raise ValueError("statistical.crossing_time_bins_n must be >= 8")
        if not 0 < self.transition_probability_floor < 1:
            raise ValueError("statistical.transition_probability_floor must be in (0, 1)")
        if self.max_state_transitions_n < 2:
            raise ValueError("statistical.max_state_transitions_n must be >= 2")


@dataclass(frozen=True)
class EmpiricalSimulationParameters:
    """Timing resolution and duration of the direct ADC simulation."""

    sampling_frequency_Hz: float | None = None
    sampling_frequency_multiplier: float = 1.0
    duration_s: float | None = None

    def sampling_frequency_for_signal(self, signal):
        source_fs = float(signal.params[TSP_F_HZ])
        if self.sampling_frequency_Hz is not None:
            return float(self.sampling_frequency_Hz)
        return source_fs * float(self.sampling_frequency_multiplier)

    def validate(self):
        if self.sampling_frequency_Hz is not None and self.sampling_frequency_Hz <= 0:
            raise ValueError("empirical.sampling_frequency_Hz must be > 0 or None")
        if self.sampling_frequency_multiplier <= 0:
            raise ValueError("empirical.sampling_frequency_multiplier must be > 0")
        if self.duration_s is not None and self.duration_s <= 0:
            raise ValueError("empirical.duration_s must be > 0 or None")


@dataclass(frozen=True)
class DynamicRangeParameters:
    """Candidate input ranges and coverage used by both model DR calculations."""

    coverage: float = 0.997

    min_start_V: float = 0.00
    min_stop_V: float = 0.40
    min_step_V: float = 0.05

    max_start_V: float = 0.60
    max_stop_V: float = 1.05
    max_step_V: float = 0.05

    def minimum_candidates_V(self):
        return np.arange(self.min_start_V, self.min_stop_V, self.min_step_V)

    def maximum_candidates_V(self):
        return np.arange(self.max_start_V, self.max_stop_V, self.max_step_V)

    def validate(self):
        if not 0 < self.coverage <= 1:
            raise ValueError("dynamic_range.coverage must be in (0, 1]")
        if self.min_step_V <= 0 or self.max_step_V <= 0:
            raise ValueError("Dynamic-range candidate steps must be > 0")
        if self.min_stop_V <= self.min_start_V or self.max_stop_V <= self.max_start_V:
            raise ValueError("Dynamic-range candidate stops must be greater than starts")


@dataclass(frozen=True)
class MetricParameters:
    dynamic_range: DynamicRangeParameters = field(default_factory=DynamicRangeParameters)
    drift_reference_width_V: float | None = None
    overload_update_time_s: float | None = None

    def validate(self):
        self.dynamic_range.validate()
        if self.drift_reference_width_V is not None and self.drift_reference_width_V <= 0:
            raise ValueError("metrics.drift_reference_width_V must be > 0 or None")
        if self.overload_update_time_s is not None and self.overload_update_time_s <= 0:
            raise ValueError("metrics.overload_update_time_s must be > 0 or None")


@dataclass(frozen=True)
class RuntimeParameters:
    # 0 = compact results only, 1 = normal summaries, 2 = verbose progress/cache/timing
    verbosity: int = 1
    progress: bool = True
    keep_statistical_debug: bool = False
    save_statistical_details: bool = False
    plot_intermediate: bool = True

    def validate(self):
        if int(self.verbosity) not in (0, 1, 2):
            raise ValueError("runtime.verbosity must be 0, 1 or 2")


@dataclass(frozen=True)
class SimulationParameters:
    """Complete set of numerical simulation settings for a reproducible comparison.

    Physical ADC design values remain in ``DesignParameters``. This object contains
    only how that ADC is characterized, simulated, discretized and evaluated.
    """

    name: str = "default"
    characterization: CharacterizationParameters = field(default_factory=CharacterizationParameters)
    signal_statistics: SignalStatisticsParameters = field(default_factory=SignalStatisticsParameters)
    statistical: StatisticalSimulationParameters = field(default_factory=StatisticalSimulationParameters)
    empirical: EmpiricalSimulationParameters = field(default_factory=EmpiricalSimulationParameters)
    metrics: MetricParameters = field(default_factory=MetricParameters)
    runtime: RuntimeParameters = field(default_factory=RuntimeParameters)

    @property
    def characterization_id(self):
        return stable_id(self.characterization)

    @property
    def signal_statistics_id(self):
        return self.signal_statistics_id_for_order(self.statistical.model_order)

    def signal_statistics_id_for_order(self, order=None):
        order = str(self.statistical.model_order if order is None else order).upper()
        p = self.signal_statistics
        payload = {
            "order": order,
            "amplitude_bins_n": p.amplitude_bins_n,
            "amplitude_min_V": p.amplitude_min_V,
            "amplitude_max_V": p.amplitude_max_V,
        }
        if order in {"D1", "D2"}:
            payload.update({
                "derivative_bins_n": p.derivative_bins_n,
                "derivative_min_V_s": p.derivative_min_V_s,
                "derivative_max_V_s": p.derivative_max_V_s,
            })
        if order == "D2":
            payload.update({
                "second_derivative_bins_n": p.second_derivative_bins_n,
                "second_derivative_min_V_s2": p.second_derivative_min_V_s2,
                "second_derivative_max_V_s2": p.second_derivative_max_V_s2,
            })
        return stable_id(payload)

    @property
    def statistical_id(self):
        return self.statistical_id_for_order(self.statistical.model_order)

    def statistical_id_for_order(self, order=None):
        order = str(self.statistical.model_order if order is None else order).upper()
        p = self.statistical
        if order != "D2":
            return stable_id({"model_order": order})
        return stable_id(p)

    @property
    def empirical_id(self):
        return stable_id(self.empirical)

    @property
    def metrics_id(self):
        return stable_id(self.metrics)

    @property
    def id(self):
        # Human-readable name and runtime/debug switches do not change numerical results.
        return stable_id({
            "characterization": self.characterization,
            "signal_statistics_id": self.signal_statistics_id,
            "statistical_id": self.statistical_id,
            "empirical": self.empirical,
            "metrics": self.metrics,
        })

    def validate(self):
        self.characterization.validate()
        self.statistical.validate()
        self.signal_statistics.validate(self.statistical.model_order)
        self.empirical.validate()
        self.metrics.validate()
        self.runtime.validate()
        return self

    def dump(self, filename):
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        with filename.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return filename

    @classmethod
    def load(cls, filename):
        with Path(filename).open("rb") as f:
            value = pickle.load(f)
        if not isinstance(value, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(value).__name__}")
        return value


# =============================================================================
# Experiment orchestration
# =============================================================================

from copy import deepcopy
from tempfile import TemporaryDirectory

from tools.utils import stable_id, print_matrix_plan, print_experiment_summary, print_approximation_recommendations, print_timing_summary, reset_timings, set_verbosity, get_verbosity, vprint
from signals.generator import generate_individual_ramps
from model.characterization import build_W_from_runs, run_ramp_characterization, overload_slope
from model.empirical import run_adc, compute_empirical_metrics
from model.statistical import compute_signal_statistics, compute_statistical_metrics, compute_first_order_metrics, compute_zeroth_order_metrics, compute_characterization_only_metrics
from model.comparison import compare_models, print_comparison
from model.storage import RunStore


def _metric_arguments(simulation_parameters):
    return (
        simulation_parameters.metrics.dynamic_range.minimum_candidates_V(),
        simulation_parameters.metrics.dynamic_range.maximum_candidates_V(),
        simulation_parameters.metrics.dynamic_range.coverage,
    )


def _signal_duration(signal, simulation_parameters):
    duration_s = float(signal.time[-1] - signal.time[0]) if len(signal.time) > 1 else 0.0
    if simulation_parameters.empirical.duration_s is not None:
        duration_s = min(duration_s, float(simulation_parameters.empirical.duration_s))
    return duration_s


def characterize_adc(adc_name, simulation_parameters, root=".", force=False):
    """Characterize one ADC. Reuses the exact characterization when already saved."""
    simulation_parameters.validate()
    set_verbosity(simulation_parameters.runtime.verbosity)
    store = RunStore(root)
    run_path, record = store.get_adc_run(adc_name, create=True)
    key = simulation_parameters.characterization_id
    if key in record["characterizations"] and not force:
        vprint(f"[CACHE] characterization: {adc_name} [{key}]", level=2)
        return run_path, record["characterizations"][key]["characterization"]

    adc = deepcopy(record["adc"])
    adc.verbose = False
    char = simulation_parameters.characterization
    amplitude_low_V, amplitude_high_V = char.amplitude_range_for_adc(adc)

    with TemporaryDirectory(prefix="lc_matrix_char_") as td:
        td = Path(td)
        ramp_dir = td / "ramps"
        run_dir = td / "runs"
        generate_individual_ramps(
            output_dir=ramp_dir,
            amplitude_low_V=amplitude_low_V,
            amplitude_high_V=amplitude_high_V,
            periods_n=char.periods_n,
            slew_range_low_V_s=char.slew_range_low_V_s,
            slew_range_high_V_s=char.slew_range_high_V_s,
            slew_range_steps_n=char.slew_range_steps_n,
            slew_range_type=char.slew_range_type,
            half_intervals_n=char.ramp_half_intervals_n,
        )
        run_ramp_characterization(adc, signal_dir=ramp_dir, output_dir=run_dir, run=True, progress=(simulation_parameters.runtime.progress and get_verbosity() >= 2), fs_Hz=char.sampling_frequency_Hz)
        characterization = build_W_from_runs(
            adc,
            output_dir=run_dir,
            bins_n=char.width_pmf_bins_n,
            simulation_parameters_id=simulation_parameters.id,
            characterization_parameters_id=simulation_parameters.characterization_id,
            characterization_fs_Hz=char.sampling_frequency_Hz,
            progress=(simulation_parameters.runtime.progress and get_verbosity() >= 2),
        )

    record["characterizations"][key] = {
        "parameters": char,
        "simulation_parameters_id": simulation_parameters.id,
        "characterization": characterization,
    }
    store.save(run_path, record)
    vprint(f"[SAVE] characterization -> {run_path.name}", level=2)
    return run_path, characterization


def _require_characterization(record, simulation_parameters):
    key = simulation_parameters.characterization_id
    if key not in record.get("characterizations", {}):
        raise RuntimeError(
            f"ADC '{record['adc_name']}' has not been characterized with characterization configuration {key}. "
            f"Run characterize_adc('{record['adc_name']}', simulation_parameters) first."
        )
    return record["characterizations"][key]["characterization"]


def run_empirical_test(adc_name, signal, simulation_parameters, root=".", force=False):
    """Run/load one empirical test signal. Characterization is not required."""
    simulation_parameters.validate()
    set_verbosity(simulation_parameters.runtime.verbosity)
    store = RunStore(root)
    run_path, record = store.get_adc_run(adc_name, create=True)
    signal_key, signal_entry, signal = store.get_signal_entry(record, signal, create=True)
    signal_name = signal_entry["name"]
    key = simulation_parameters.empirical_id
    if key in signal_entry["empirical_runs"] and not force:
        vprint(f"[CACHE] empirical: {adc_name} × {signal_name} [{key}]", level=2)
        return run_path, signal_entry["empirical_runs"][key]

    adc = deepcopy(record["adc"])
    adc.verbose = False
    fs_Hz = simulation_parameters.empirical.sampling_frequency_for_signal(signal)
    empirical_run = run_adc(adc, signal, fs_Hz=fs_Hz, tf_s=simulation_parameters.empirical.duration_s, progress=(simulation_parameters.runtime.progress and get_verbosity() >= 2), simulation_parameters_id=simulation_parameters.id, empirical_parameters_id=simulation_parameters.empirical_id)
    range_min_x, range_max_x, coverage = _metric_arguments(simulation_parameters)
    empirical_metrics = compute_empirical_metrics(empirical_run, range_min_x=range_min_x, range_max_x=range_max_x, coverage=coverage, reference_width_V=simulation_parameters.metrics.drift_reference_width_V)

    entry = {
        "parameters": simulation_parameters.empirical,
        "simulation_parameters_id": simulation_parameters.id,
        "run": empirical_run,
        "metrics": empirical_metrics,
    }
    signal_entry["empirical_runs"][key] = entry
    store.save(run_path, record)
    vprint(f"[SAVE] empirical -> {run_path.name}", level=2)
    return run_path, entry


def run_statistical_test(adc_name, signal, simulation_parameters, root=".", force=False):
    """Run/load W, D0, D1 and/or D2 up to the selected model order."""
    simulation_parameters.validate()
    set_verbosity(simulation_parameters.runtime.verbosity)
    store = RunStore(root)
    run_path, record = store.get_adc_run(adc_name, create=True)
    characterization = _require_characterization(record, simulation_parameters)
    signal_key, signal_entry, signal = store.get_signal_entry(record, signal, create=True)
    signal_name = signal_entry["name"]

    order = str(simulation_parameters.statistical.model_order).upper()
    rank = {"W": -1, "D0": 0, "D1": 1, "D2": 2}[order]
    signal_statistics = None
    stats_key = "W_ONLY"

    if rank >= 0:
        stats_key = stable_id({"characterization_id": characterization.id, "signal_statistics_id": simulation_parameters.signal_statistics_id, "model_order": order})
        if stats_key not in signal_entry["signal_statistics"] or force:
            p = simulation_parameters.signal_statistics
            signal_statistics = compute_signal_statistics(
                signal,
                characterization,
                order=order,
                bins_x_n=p.amplitude_bins_n,
                bins_x_min=p.amplitude_min_V,
                bins_x_max=p.amplitude_max_V,
                bins_y_n=p.derivative_bins_n,
                bins_y_min=p.derivative_min_V_s,
                bins_y_max=p.derivative_max_V_s,
                bins_z_n=p.second_derivative_bins_n,
                bins_z_min=p.second_derivative_min_V_s2,
                bins_z_max=p.second_derivative_max_V_s2,
                simulation_parameters_id=simulation_parameters.id,
                signal_statistics_parameters_id=simulation_parameters.signal_statistics_id,
            )
            signal_entry["signal_statistics"][stats_key] = {"parameters": p, "model_order": order, "statistics": signal_statistics}
        else:
            signal_statistics = signal_entry["signal_statistics"][stats_key]["statistics"]
            vprint(f"[CACHE] {order}: {signal_name} [{stats_key[:10]}]", level=2)

    stat_key = stable_id({
        "characterization_id": characterization.id,
        "signal_statistics_id": None if signal_statistics is None else signal_statistics.id,
        "statistical_id": simulation_parameters.statistical_id,
        "metrics_id": simulation_parameters.metrics_id,
        "model_order": order,
    })
    if stat_key in signal_entry["statistical_runs"] and not force:
        vprint(f"[CACHE] statistical: {adc_name} × {signal_name} [{stat_key[:10]}]", level=2)
        store.save(run_path, record)
        return run_path, signal_entry["statistical_runs"][stat_key]

    range_min_x, range_max_x, coverage = _metric_arguments(simulation_parameters)
    overload_slope_V_s = None
    if simulation_parameters.metrics.overload_update_time_s is not None:
        width_mass = np.sum(characterization.W, axis=(0, 1))
        observed_widths = characterization.w[width_mass > 0]
        if len(observed_widths):
            overload_slope_V_s = overload_slope(np.min(observed_widths), simulation_parameters.metrics.overload_update_time_s)
    duration_s = _signal_duration(signal, simulation_parameters)

    w_metrics = compute_characterization_only_metrics(characterization, range_min_x=range_min_x, range_max_x=range_max_x, coverage=coverage)
    d0_metrics = None
    d1_metrics = None
    d2_metrics = None

    if rank >= 0:
        d0_metrics = compute_zeroth_order_metrics(characterization, signal_statistics, range_min_x=range_min_x, range_max_x=range_max_x, coverage=coverage)
    if rank >= 1:
        d1_metrics = compute_first_order_metrics(characterization, signal_statistics, range_min_x=range_min_x, range_max_x=range_max_x, coverage=coverage, reference_width_V=simulation_parameters.metrics.drift_reference_width_V, signal_duration_s=duration_s, overload_slope_V_s=overload_slope_V_s)
    if rank >= 2:
        d2_metrics = compute_statistical_metrics(
            characterization,
            signal_statistics,
            range_min_x=range_min_x,
            range_max_x=range_max_x,
            coverage=coverage,
            overload_slope_V_s=overload_slope_V_s,
            progress=(simulation_parameters.runtime.progress and get_verbosity() >= 2),
            keep_debug=simulation_parameters.runtime.keep_statistical_debug,
            reference_width_V=simulation_parameters.metrics.drift_reference_width_V,
            signal_duration_s=duration_s,
            headstart_bins_n=simulation_parameters.statistical.headstart_bins_n,
            reversal_error_bins_n=simulation_parameters.statistical.reversal_error_bins_n,
            crossing_time_bins_n=simulation_parameters.statistical.crossing_time_bins_n,
            transition_probability_floor=simulation_parameters.statistical.transition_probability_floor,
            max_state_transitions_n=simulation_parameters.statistical.max_state_transitions_n,
        )

    entry = {
        "parameters": simulation_parameters.statistical,
        "model_order": order,
        "simulation_parameters_id": simulation_parameters.id,
        "signal_statistics_id": None if signal_statistics is None else signal_statistics.id,
        "signal_statistics": signal_statistics,
        "W_only": w_metrics,
        "D0": d0_metrics,
        "D1": d1_metrics,
        "D2": d2_metrics,
    }
    signal_entry["statistical_runs"][stat_key] = entry
    store.save(run_path, record)
    vprint(f"[SAVE] statistical -> {run_path.name}", level=2)
    return run_path, entry

def compare_saved_models(adc_name, signal, simulation_parameters, root=".", print_table=True):
    """Compare cached empirical/statistical results at the requested maximum order."""
    simulation_parameters.validate()
    set_verbosity(simulation_parameters.runtime.verbosity)
    store = RunStore(root)
    run_path, record = store.get_adc_run(adc_name, create=False)
    characterization = _require_characterization(record, simulation_parameters)
    _, signal_entry, signal = store.get_signal_entry(record, signal, create=False)
    signal_name = signal_entry["name"]
    empirical_key = simulation_parameters.empirical_id
    if empirical_key not in signal_entry["empirical_runs"]:
        raise RuntimeError(f"No empirical test exists for '{adc_name}' × '{signal_name}' with empirical configuration {empirical_key}. Run run_empirical_test(...) first.")

    order = str(simulation_parameters.statistical.model_order).upper()
    rank = {"W": -1, "D0": 0, "D1": 1, "D2": 2}[order]
    signal_statistics = None
    if rank >= 0:
        stats_key = stable_id({"characterization_id": characterization.id, "signal_statistics_id": simulation_parameters.signal_statistics_id, "model_order": order})
        if stats_key not in signal_entry["signal_statistics"]:
            raise RuntimeError(f"No {order} signal statistics exist for '{signal_name}'. Run run_statistical_test(...) first.")
        signal_statistics = signal_entry["signal_statistics"][stats_key]["statistics"]

    stat_key = stable_id({
        "characterization_id": characterization.id,
        "signal_statistics_id": None if signal_statistics is None else signal_statistics.id,
        "statistical_id": simulation_parameters.statistical_id,
        "metrics_id": simulation_parameters.metrics_id,
        "model_order": order,
    })
    if stat_key not in signal_entry["statistical_runs"]:
        raise RuntimeError(f"No statistical metrics exist for '{adc_name}' × '{signal_name}' with requested model order {order}. Run run_statistical_test(...) first.")

    emp = signal_entry["empirical_runs"][empirical_key]["metrics"]
    stat = signal_entry["statistical_runs"][stat_key]
    comparison = compare_models(
        emp,
        stat.get("D2"),
        stat.get("D1"),
        stat.get("D0"),
        stat.get("W_only"),
        selected_order=order,
    )
    comparison_key = stable_id({"empirical": empirical_key, "statistical": stat_key, "model_order": order})
    signal_entry["comparisons"][comparison_key] = comparison
    store.save(run_path, record)
    if print_table:
        print_comparison(comparison)
    return run_path, comparison

def run_experiment(adc_name, signal, simulation_parameters, root=".", characterize=False, empirical=True, statistical=True, compare=True, force=False):
    """Convenience wrapper. Missing dependencies are never silently generated."""
    simulation_parameters.validate()
    set_verbosity(simulation_parameters.runtime.verbosity)
    reset_timings()
    store = RunStore(root)
    run_path, record = store.get_adc_run(adc_name, create=True)
    adc = record["adc"]
    _, signal_entry, signal = store.get_signal_entry(record, signal, create=True)
    signal_name = signal_entry["name"]
    store.save(run_path, record)

    vprint(f"\nEXPERIMENT: ADC={adc_name} | SIGNAL={signal_name} | SIM={simulation_parameters.name} [{simulation_parameters.id}] | ORDER={simulation_parameters.statistical.model_order.upper()}", level=1)
    if get_verbosity() >= 1:
        print_matrix_plan(adc, simulation_parameters)

    if characterize:
        characterize_adc(adc_name, simulation_parameters, root=root, force=force)
    if empirical:
        run_empirical_test(adc_name, signal, simulation_parameters, root=root, force=force)
    if statistical:
        run_statistical_test(adc_name, signal, simulation_parameters, root=root, force=force)
    comparison = None
    if compare:
        _, comparison = compare_saved_models(adc_name, signal, simulation_parameters, root=root, print_table=True)

    if get_verbosity() >= 1:
        run_path, record = store.get_adc_run(adc_name, create=False)
        _, signal_entry, signal = store.get_signal_entry(record, signal, create=False)
        characterization = record.get("characterizations", {}).get(simulation_parameters.characterization_id, {}).get("characterization")
        signal_statistics = None
        if characterization is not None and simulation_parameters.statistical.model_order.upper() != "W":
            stats_key = stable_id({"characterization_id": characterization.id, "signal_statistics_id": simulation_parameters.signal_statistics_id, "model_order": simulation_parameters.statistical.model_order.upper()})
            signal_statistics = signal_entry.get("signal_statistics", {}).get(stats_key, {}).get("statistics")
        print_experiment_summary(record["adc"], signal, simulation_parameters, characterization=characterization, signal_statistics=signal_statistics, signal_duration_s=_signal_duration(signal, simulation_parameters))
        if simulation_parameters.statistical.model_order.upper() == "D2":
            print_approximation_recommendations(simulation_parameters)
        print_timing_summary()
    return run_path, comparison

def plot_saved_experiment(adc_name, signal, simulation_parameters, root="."):
    """Plot cached results and overlay empirical golden truth where comparable."""
    import matplotlib.pyplot as plt
    from model.characterization import plot_characterization_ridgeline, plot_characterization_deviation_heatmap, plot_dac_nonlinearity
    from model.empirical import plot_empirical_crossings, plot_empirical_reconstruction
    from model.statistical import plot_D2, plot_crossing_probability, plot_crossing_source, plot_crossing_time, plot_delta_w, plot_delta_w_vs_amplitude, plot_level_width_distributions, plot_reversal_recapture, plot_model_hierarchy_widths, plot_model_hierarchy_delta_w

    store = RunStore(root)
    _, record = store.get_adc_run(adc_name, create=False)
    adc = record["adc"]
    _, signal_entry, signal = store.get_signal_entry(record, signal, create=False)

    emp_entry = signal_entry.get("empirical_runs", {}).get(simulation_parameters.empirical_id)
    empirical_metrics = None if emp_entry is None else emp_entry.get("metrics")
    empirical_run = None if emp_entry is None else emp_entry.get("run")

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(signal.time, signal.data)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(signal.name)
    fig.tight_layout()
    plot_dac_nonlinearity(adc)

    characterization = record.get("characterizations", {}).get(simulation_parameters.characterization_id, {}).get("characterization")
    if characterization is not None:
        # plot_characterization_ridgeline(adc, characterization)
        plot_characterization_deviation_heatmap(adc, characterization)

        order = simulation_parameters.statistical.model_order.upper()
        rank = {"W": -1, "D0": 0, "D1": 1, "D2": 2}[order]
        stat_obj = None
        if rank >= 0:
            stats_key = stable_id({"characterization_id": characterization.id, "signal_statistics_id": simulation_parameters.signal_statistics_id, "model_order": order})
            stat_obj = signal_entry.get("signal_statistics", {}).get(stats_key, {}).get("statistics")

        if stat_obj is not None:
            if order == "D2":
                plot_D2(stat_obj)
            stat_key = stable_id({
                "characterization_id": characterization.id,
                "signal_statistics_id": stat_obj.id,
                "statistical_id": simulation_parameters.statistical_id,
                "metrics_id": simulation_parameters.metrics_id,
                "model_order": order,
            })
            stat_entry = signal_entry.get("statistical_runs", {}).get(stat_key)
            if stat_entry is not None:
                d2_metrics = stat_entry.get("D2")
                d1_metrics = stat_entry.get("D1")
                d0_metrics = stat_entry.get("D0")
                w_metrics = stat_entry.get("W_only")

                if d2_metrics is not None:
                    plot_model_hierarchy_widths(d2_metrics.state, w_metrics, first_order_metrics=d1_metrics, second_order_metrics=d2_metrics, zeroth_order_metrics=d0_metrics, empirical_metrics=empirical_metrics)
                    if d1_metrics is not None:
                        plot_model_hierarchy_delta_w(d1_metrics, d2_metrics, empirical_metrics=empirical_metrics)
                    plot_crossing_probability(d2_metrics.state, headstart_bins_n=simulation_parameters.statistical.headstart_bins_n)
                    plot_crossing_source(d2_metrics.state, d2_metrics.crossing_source, empirical_run=empirical_run)
                    plot_crossing_time(d2_metrics.local_distortion, empirical_metrics=empirical_metrics)
                    plot_level_width_distributions(d2_metrics.state, d2_metrics.level_width, empirical_metrics=empirical_metrics)
                    plot_delta_w(d2_metrics.state, d2_metrics.local_distortion, empirical_metrics=empirical_metrics)
                    plot_reversal_recapture(d2_metrics.local_distortion, empirical_metrics=empirical_metrics)
                    plot_delta_w_vs_amplitude(d2_metrics.state, d2_metrics.local_distortion, empirical_run=empirical_run)
                elif d1_metrics is not None:
                    # D1 has no reversal model, but its width distribution can still be
                    # compared directly with empirical same-direction widths.
                    plot_model_hierarchy_widths(d1_metrics.state, w_metrics, first_order_metrics=d1_metrics, zeroth_order_metrics=d0_metrics, empirical_metrics=empirical_metrics)

    if empirical_run is not None:
        plot_empirical_crossings(empirical_run)
        plot_empirical_reconstruction(empirical_run, metrics=empirical_metrics)
    plt.show()

