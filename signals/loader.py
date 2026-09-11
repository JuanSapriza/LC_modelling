from __future__ import annotations

from pathlib import Path
import hashlib
import pickle
import numpy as np
from scipy.interpolate import interp1d

from tools.timeseries import Timeseries
from tools.ts_params import TSP_F_HZ, TSP_LENGTH_S

SIGNAL_SOURCE_KEY = "LC-matrix signal source"
PORTABLE_FORMAT = "lc_matrix_timeseries_v1"
DATA_DIR = Path(__file__).resolve().parent / "data"


def _resolve_file(name, data_dir=None):
    data_dir = DATA_DIR if data_dir is None else Path(data_dir)
    path = Path(name)
    if path.suffix == "":
        path = path.with_suffix(".pkl")
    if not path.is_absolute() and path.parent == Path("."):
        path = data_dir / path.name
    if not path.exists():
        available = sorted(p.stem for p in data_dir.glob("*.pkl"))
        raise FileNotFoundError(f"Signal pickle '{name}' not found. Available: {available}")
    return path


def _portable_payload(series):
    params = {}
    for key, value in getattr(series, "params", {}).items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, np.ndarray):
            value = value.tolist()
        try:
            pickle.dumps(value)
        except Exception:
            continue
        params[key] = value
    return {
        "format": PORTABLE_FORMAT,
        "name": str(series.name),
        "data": np.asarray(series.data, dtype=float).tolist(),
        "time": np.asarray(series.time, dtype=float).tolist(),
        "params": params,
    }


def save_signal(series, name=None, filename=None, data_dir=None, portable=True):
    if filename is None:
        if name is None:
            name = str(series.name).replace(" ", "_")
        filename = _resolve_output_file(name, data_dir=data_dir)
    else:
        filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    payload = series.copy() if hasattr(series, "copy") else series
    with filename.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return filename


def _resolve_output_file(name, data_dir=None):
    data_dir = DATA_DIR if data_dir is None else Path(data_dir)
    path = Path(name)
    if path.suffix == "":
        path = path.with_suffix(".pkl")
    if not path.is_absolute() and path.parent == Path("."):
        path = data_dir / path.name
    return path


def _load_raw(path):
    with Path(path).open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, Timeseries):
        return obj.copy()
    if isinstance(obj, dict) and obj.get("format") == PORTABLE_FORMAT:
        params = dict(obj.get("params", {}))
        fs = float(params.get(TSP_F_HZ, 0.0))
        series = Timeseries(obj.get("name", Path(path).stem), data=obj["data"], time=obj["time"], f_Hz=fs)
        series.params.update(params)
        return series
    if isinstance(obj, dict) and "data" in obj and "time" in obj:
        fs = float(obj.get("f_Hz", obj.get("params", {}).get(TSP_F_HZ, 0.0)))
        series = Timeseries(obj.get("name", Path(path).stem), data=obj["data"], time=obj["time"], f_Hz=fs)
        series.params.update(obj.get("params", {}))
        return series
    raise TypeError(f"{path} does not contain a Timeseries-compatible pickle")


def load_signal(name, start_s=None, end_s=None, sampling_frequency_Hz=None, offset_V=None, amplitude_Vpp=None, scale=1.0, interpolation="linear", output_name=None, data_dir=None):
    """Load a stored Timeseries pickle and return a transformed copy.

    ``start_s``/``end_s`` crop relative to the stored signal time axis.
    ``sampling_frequency_Hz`` resamples the cropped signal.
    ``scale`` multiplies the signal around its midpoint.
    ``amplitude_Vpp`` optionally forces an exact peak-to-peak amplitude.
    ``offset_V`` optionally sets the midpoint of the signal to that voltage.
    """
    path = _resolve_file(name, data_dir=data_dir)
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    series = _load_raw(path)

    time = np.asarray(series.time, dtype=float)
    data = np.asarray(series.data, dtype=float)
    if len(time) < 2:
        raise ValueError(f"Stored signal '{path.name}' must contain at least two samples")

    t0 = float(time[0]) if start_s is None else float(start_s)
    t1 = float(time[-1]) if end_s is None else float(end_s)
    source_dt = float(np.median(np.diff(time)))
    endpoint_tolerance_s = max(1e-3, 1.5 * source_dt)
    if t0 < time[0] and t0 >= time[0] - endpoint_tolerance_s:
        t0 = float(time[0])
    if t1 > time[-1] and t1 <= time[-1] + endpoint_tolerance_s:
        t1 = float(time[-1])
    if t1 <= t0:
        raise ValueError("end_s must be larger than start_s")
    if t0 < time[0] or t1 > time[-1] + 1e-12:
        raise ValueError(f"Requested interval [{t0}, {t1}] s is outside stored range [{time[0]}, {time[-1]}] s")

    mask = (time >= t0) & (time <= t1)
    cropped_t = time[mask]
    cropped_y = data[mask]
    if len(cropped_t) < 2:
        raise ValueError("Requested crop contains fewer than two samples")

    # Include exact crop boundaries by interpolation when they do not coincide with samples.
    if cropped_t[0] > t0:
        cropped_t = np.insert(cropped_t, 0, t0)
        cropped_y = np.insert(cropped_y, 0, np.interp(t0, time, data))
    if cropped_t[-1] < t1:
        cropped_t = np.append(cropped_t, t1)
        cropped_y = np.append(cropped_y, np.interp(t1, time, data))

    target_fs = float(series.params.get(TSP_F_HZ, 1.0 / np.median(np.diff(time)))) if sampling_frequency_Hz is None else float(sampling_frequency_Hz)
    if target_fs <= 0:
        raise ValueError("sampling_frequency_Hz must be > 0")

    duration_s = t1 - t0
    samples_n = int(round(duration_s * target_fs)) + 1
    new_t = np.arange(samples_n, dtype=float) / target_fs
    source_t = cropped_t - t0
    if sampling_frequency_Hz is None and np.allclose(np.diff(cropped_t), np.median(np.diff(cropped_t)), rtol=1e-5, atol=1e-12):
        new_t = source_t.copy()
        new_y = cropped_y.copy()
        target_fs = float(1.0 / np.median(np.diff(new_t)))
    else:
        f = interp1d(source_t, cropped_y, kind=interpolation, bounds_error=False, fill_value="extrapolate")
        new_y = np.asarray(f(new_t), dtype=float)

    midpoint = 0.5 * (float(np.max(new_y)) + float(np.min(new_y)))
    new_y = midpoint + float(scale) * (new_y - midpoint)
    if amplitude_Vpp is not None:
        current_vpp = float(np.max(new_y) - np.min(new_y))
        if current_vpp <= 0:
            raise ValueError("Cannot set amplitude_Vpp on a constant signal")
        midpoint = 0.5 * (float(np.max(new_y)) + float(np.min(new_y)))
        new_y = midpoint + (new_y - midpoint) * (float(amplitude_Vpp) / current_vpp)
    if offset_V is not None:
        midpoint = 0.5 * (float(np.max(new_y)) + float(np.min(new_y)))
        new_y = new_y + (float(offset_V) - midpoint)

    out = Timeseries(output_name or Path(path).stem, data=new_y, time=new_t, f_Hz=target_fs)
    out.params[SIGNAL_SOURCE_KEY] = {
        "type": "stored",
        "file": path.name,
        "file_sha256": source_sha256,
        "parameters": {
            "start_s": start_s,
            "end_s": end_s,
            "sampling_frequency_Hz": sampling_frequency_Hz,
            "offset_V": offset_V,
            "amplitude_Vpp": amplitude_Vpp,
            "scale": float(scale),
            "interpolation": interpolation,
        },
    }
    out.params[TSP_LENGTH_S] = float(new_t[-1] - new_t[0]) if len(new_t) > 1 else 0.0
    return out


def list_signals(data_dir=None):
    data_dir = DATA_DIR if data_dir is None else Path(data_dir)
    return sorted(p.stem for p in data_dir.glob("*.pkl"))
