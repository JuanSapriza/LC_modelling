from __future__ import annotations

from pathlib import Path
import numpy as np

from tools.timeseries import Timeseries
from tools.utils import timed

SIGNAL_SOURCE_KEY = "LC-matrix signal source"


def _set_source(series, generator, parameters):
    series.params[SIGNAL_SOURCE_KEY] = {
        "type": "generated",
        "generator": generator,
        "parameters": dict(parameters),
    }
    return series


def generate_sine(frequency_Hz=10.0, sampling_frequency_Hz=50_000.0, duration_s=5.0, amplitude_V=0.35, offset_V=0.5, phase_deg=0.0, name=None):
    samples_n = int(round(float(sampling_frequency_Hz) * float(duration_s))) + 1
    time_s = np.arange(samples_n, dtype=float) / float(sampling_frequency_Hz)
    data_V = float(offset_V) + float(amplitude_V) * np.sin(2 * np.pi * float(frequency_Hz) * time_s + np.deg2rad(float(phase_deg)))
    series = Timeseries(name or f"sine_{frequency_Hz:g}Hz", data=data_V, time=time_s, f_Hz=float(sampling_frequency_Hz))
    return _set_source(series, "generate_sine", {
        "frequency_Hz": float(frequency_Hz),
        "sampling_frequency_Hz": float(sampling_frequency_Hz),
        "duration_s": float(duration_s),
        "amplitude_V": float(amplitude_V),
        "offset_V": float(offset_V),
        "phase_deg": float(phase_deg),
    })


def generate_sinc(sampling_frequency_Hz=50_000.0, duration_s=5.0, width_s=0.1, amplitude_V=0.35, offset_V=0.5, center_s=None, name=None):
    center_s = float(duration_s) / 2 if center_s is None else float(center_s)
    samples_n = int(round(float(sampling_frequency_Hz) * float(duration_s))) + 1
    time_s = np.arange(samples_n, dtype=float) / float(sampling_frequency_Hz)
    data_V = float(offset_V) + float(amplitude_V) * np.sinc((time_s - center_s) / float(width_s))
    series = Timeseries(name or "sinc", data=data_V, time=time_s, f_Hz=float(sampling_frequency_Hz))
    return _set_source(series, "generate_sinc", {
        "sampling_frequency_Hz": float(sampling_frequency_Hz),
        "duration_s": float(duration_s),
        "width_s": float(width_s),
        "amplitude_V": float(amplitude_V),
        "offset_V": float(offset_V),
        "center_s": center_s,
    })


def make_slew_range(slew_range_low_V_s=1, slew_range_high_V_s=4096, slew_range_steps_n=100, slew_range_type="linear"):
    if slew_range_type == "linear":
        return np.linspace(start=slew_range_low_V_s, stop=slew_range_high_V_s, num=slew_range_steps_n)
    if slew_range_type == "log":
        return np.logspace(start=np.log2(slew_range_low_V_s), stop=np.log2(slew_range_high_V_s), num=slew_range_steps_n, base=2)
    raise ValueError("slew_range_type must be 'linear' or 'log'")


def generate_ramp_chirp(amplitude_low_V=0.0, amplitude_high_V=1.0, periods_n=5, slew_range_low_V_s=1e1, slew_range_high_V_s=5e4, slew_range_steps_n=5, slew_range_type="linear", fsim_Hz=None):
    slew_range_V_s = make_slew_range(slew_range_low_V_s, slew_range_high_V_s, slew_range_steps_n, slew_range_type)
    fsim_Hz = slew_range_high_V_s * 2 if fsim_Hz is None else fsim_Hz
    amplitude_range_V = amplitude_high_V - amplitude_low_V
    if amplitude_range_V <= 0:
        raise ValueError("amplitude_high_V must be larger than amplitude_low_V")

    single_period_segments = []
    for slew_V_s in slew_range_V_s:
        half_intervals_n = max(1, int(np.ceil(amplitude_range_V / slew_V_s * fsim_Hz)))
        ramp_up_V = np.linspace(amplitude_low_V, amplitude_high_V, half_intervals_n + 1)
        ramp_down_V = np.linspace(amplitude_high_V, amplitude_low_V, half_intervals_n + 1)[1:]
        triangle_V = np.concatenate((ramp_up_V, ramp_down_V))
        if single_period_segments:
            triangle_V = triangle_V[1:]
        single_period_segments.append(triangle_V)

    single_period_V = np.concatenate(single_period_segments)
    signal_segments = [single_period_V]
    for _ in range(1, periods_n):
        signal_segments.append(single_period_V[1:])

    signal_V = np.concatenate(signal_segments)
    time_s = np.arange(signal_V.size) / fsim_Hz
    series = Timeseries("ramp_chirp", data=signal_V, time=time_s, f_Hz=fsim_Hz)
    return _set_source(series, "generate_ramp_chirp", {
        "amplitude_low_V": amplitude_low_V,
        "amplitude_high_V": amplitude_high_V,
        "periods_n": periods_n,
        "slew_range_low_V_s": slew_range_low_V_s,
        "slew_range_high_V_s": slew_range_high_V_s,
        "slew_range_steps_n": slew_range_steps_n,
        "slew_range_type": slew_range_type,
        "fsim_Hz": fsim_Hz,
    }), slew_range_V_s


def generate_ramp(slew_V_s, amplitude_low_V=0.0, amplitude_high_V=1.0, periods_n=20, half_intervals_n=1000):
    amplitude_range_V = amplitude_high_V - amplitude_low_V
    if amplitude_range_V <= 0:
        raise ValueError("amplitude_high_V must be larger than amplitude_low_V")

    fsim_Hz = half_intervals_n * slew_V_s / amplitude_range_V
    single_period_segments = []
    for _ in range(periods_n):
        ramp_up_V = np.linspace(amplitude_low_V, amplitude_high_V, num=half_intervals_n + 1)
        ramp_down_V = np.linspace(amplitude_high_V, amplitude_low_V, half_intervals_n + 1)[1:]
        triangle_V = np.concatenate((ramp_up_V, ramp_down_V))
        if single_period_segments:
            triangle_V = triangle_V[1:]
        single_period_segments.append(triangle_V)

    signal_V = np.concatenate(single_period_segments)
    time_s = np.arange(signal_V.size) / fsim_Hz
    name = f"Ramp (slew:{slew_V_s:.12g} V/s, ampl m/M {amplitude_low_V}/{amplitude_high_V} V, x{periods_n} periods,  n={signal_V.size} samples)"
    series = Timeseries(name, data=signal_V, time=time_s, f_Hz=fsim_Hz)
    return _set_source(series, "generate_ramp", {
        "slew_V_s": float(slew_V_s),
        "amplitude_low_V": float(amplitude_low_V),
        "amplitude_high_V": float(amplitude_high_V),
        "periods_n": int(periods_n),
        "half_intervals_n": int(half_intervals_n),
    })


@timed("generate_individual_ramps")
def generate_individual_ramps(output_dir, amplitude_low_V=0.0, amplitude_high_V=1.0, periods_n=20, slew_range_low_V_s=1, slew_range_high_V_s=4096, slew_range_steps_n=100, slew_range_type="linear", half_intervals_n=1000):
    from signals.loader import save_signal

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slew_range_V_s = make_slew_range(slew_range_low_V_s, slew_range_high_V_s, slew_range_steps_n, slew_range_type)
    files = []

    for slew_V_s in slew_range_V_s:
        series = generate_ramp(slew_V_s, amplitude_low_V, amplitude_high_V, periods_n, half_intervals_n)
        filename = output_dir / f"ramp_sr_{slew_V_s:.12g}_V_s.pkl"
        save_signal(series, filename=filename, portable=False)
        files.append(filename)

    return slew_range_V_s, files
