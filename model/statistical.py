from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from tools.utils import centers_to_edges, DistributionStatistics, get_distribution_statistics, CrossingModeRateResults, PROGRESS_WIDTH, timed, stable_id
from tools.utils import signal_id as get_signal_id
from tools.ts_params import TSP_F_HZ
from model.characterization import ADCCharacterization, coerce_characterization




# =============================================================================
# Signal statistics: D2(V,V',V'')
# =============================================================================

@dataclass
class SignalStatistics:
    D2: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    source_name: str | None = None
    signal_id: str | None = None
    characterization_id: str | None = None
    metadata: dict = field(default_factory=dict)
    simulation_parameters_id: str | None = None
    signal_statistics_parameters_id: str | None = None

    def __getitem__(self, key):
        return getattr(self, key)

    def as_dict(self):
        return {"D2": self.D2, "x": self.x, "y": self.y, "z": self.z}

    @property
    def id(self):
        return stable_id({
            "signal_id": self.signal_id,
            "characterization_id": self.characterization_id,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "D2": self.D2,
            "signal_statistics_parameters_id": getattr(self, "signal_statistics_parameters_id", None),
        })


def coerce_signal_statistics(value):
    if isinstance(value, SignalStatistics):
        return value
    if isinstance(value, dict) and all(key in value for key in ("D2", "x", "y", "z")):
        return SignalStatistics(D2=np.asarray(value["D2"]), x=np.asarray(value["x"]), y=np.asarray(value["y"]), z=np.asarray(value["z"]))
    raise TypeError("Expected SignalStatistics or a legacy D2 dictionary.")


def _axes_from_characterization(characterization_or_m):
    if hasattr(characterization_or_m, "m"):
        return characterization_or_m.m
    return characterization_or_m


def _linear_edges(values, bins_n, value_min=None, value_max=None, name="axis"):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        raise ValueError(f"No finite values available for {name}")
    low = float(np.min(finite)) if value_min is None else float(value_min)
    high = float(np.max(finite)) if value_max is None else float(value_max)
    if high <= low:
        eps = max(abs(low), 1.0) * 1e-12
        low -= eps
        high += eps
    return np.linspace(low, high, int(bins_n) + 1)


@timed("compute_signal_statistics")
def compute_signal_statistics(series, characterization_or_m, order="D2", bins_x_n=None, bins_x_min=None, bins_x_max=None, bins_y_n=101, bins_y_min=None, bins_y_max=None, bins_z_n=128, bins_z_min=None, bins_z_max=None, signal_key=None, characterization_key=None, simulation_parameters_id=None, signal_statistics_parameters_id=None):
    """Compute the signal histogram only up to the requested model order.

    The returned object keeps the historical ``D2`` storage field for compatibility:
    D0 is stored as (V,1,1), D1 as (V,V',1), and D2 as (V,V',V'').
    """
    order = str(order).upper()
    rank = {"D0": 0, "D1": 1, "D2": 2}.get(order)
    if rank is None:
        raise ValueError("order must be D0, D1 or D2")

    m = _axes_from_characterization(characterization_or_m)
    if signal_key is None:
        signal_key = get_signal_id(series)
    if characterization_key is None and hasattr(characterization_or_m, "id"):
        characterization_key = characterization_or_m.id

    fs = float(series.params[TSP_F_HZ])
    data = np.asarray(series.data, dtype=float)
    characterized_x = np.asarray(m["x"], dtype=float)

    # Use exactly the samples required by the requested derivative order.
    x = data if rank == 0 else data[:-rank]
    if bins_x_n is None:
        bins_x = characterized_x
        bins_x_edges = centers_to_edges(bins_x)
    else:
        x_low = float(characterized_x[0]) if bins_x_min is None else float(bins_x_min)
        x_high = float(characterized_x[-1]) if bins_x_max is None else float(bins_x_max)
        bins_x_edges = _linear_edges(x, bins_x_n, x_low, x_high, name="V")
        bins_x = 0.5 * (bins_x_edges[:-1] + bins_x_edges[1:])

    if rank == 0:
        counts_x, _ = np.histogram(x, bins=bins_x_edges)
        total = counts_x.sum()
        if total <= 0:
            raise RuntimeError("D0 histogram contains no samples. Check the configured V range.")
        histogram = (counts_x / total)[:, None, None]
        bins_y = np.array([0.0])
        bins_z = np.array([0.0])
        bins_y_edges = np.array([-0.5, 0.5])
        bins_z_edges = np.array([-0.5, 0.5])

    else:
        y_full = np.diff(data) * fs
        y = y_full if rank == 1 else y_full[:-1]
        if bins_y_min is None and bins_y_max is None:
            y_lim = float(np.max(np.abs(y[np.isfinite(y)])))
            bins_y_min, bins_y_max = -y_lim, y_lim
        bins_y_edges = _linear_edges(y, bins_y_n, bins_y_min, bins_y_max, name="V'")
        if bins_y_edges[0] < 0 < bins_y_edges[-1]:
            zero_edge_error = float(np.min(np.abs(bins_y_edges)))
            edge_step = float(np.min(np.diff(bins_y_edges))) if len(bins_y_edges) > 1 else 1.0
            if zero_edge_error > max(1e-12, 1e-9 * edge_step):
                raise ValueError("The V' grid crosses zero but V'=0 is not a bin edge. Use an even bin count with a symmetric range.")
        bins_y = 0.5 * (bins_y_edges[:-1] + bins_y_edges[1:])

        if rank == 1:
            counts_xy, _, _ = np.histogram2d(x, y, bins=[bins_x_edges, bins_y_edges])
            total = counts_xy.sum()
            if total <= 0:
                raise RuntimeError("D1 histogram contains no samples. Check the configured V/V' ranges.")
            histogram = (counts_xy / total)[:, :, None]
            bins_z = np.array([0.0])
            bins_z_edges = np.array([-0.5, 0.5])
        else:
            z = np.diff(y_full) * fs
            if bins_z_min is None and bins_z_max is None:
                z_lim = float(np.max(np.abs(z[np.isfinite(z)])))
                bins_z_min, bins_z_max = -z_lim, z_lim
            bins_z_edges = _linear_edges(z, bins_z_n, bins_z_min, bins_z_max, name="V''")
            if bins_z_edges[0] < 0 < bins_z_edges[-1]:
                zero_edge_error = float(np.min(np.abs(bins_z_edges)))
                edge_step = float(np.min(np.diff(bins_z_edges))) if len(bins_z_edges) > 1 else 1.0
                if zero_edge_error > max(1e-12, 1e-9 * edge_step):
                    raise ValueError("The V'' grid crosses zero but V''=0 is not a bin edge. Use an even bin count with a symmetric range.")
            bins_z = 0.5 * (bins_z_edges[:-1] + bins_z_edges[1:])
            counts_xyz, _ = np.histogramdd(np.column_stack((x, y, z)), bins=[bins_x_edges, bins_y_edges, bins_z_edges])
            total = counts_xyz.sum()
            if total <= 0:
                raise RuntimeError("D2 histogram contains no samples. Check the configured V/V'/V'' ranges.")
            histogram = counts_xyz / total

    metadata = {
        "model_order": order,
        "bins_x_n": int(len(bins_x)),
        "bins_x_min": float(bins_x_edges[0]),
        "bins_x_max": float(bins_x_edges[-1]),
        "bins_y_n": int(len(bins_y)),
        "bins_y_min": float(bins_y_edges[0]),
        "bins_y_max": float(bins_y_edges[-1]),
        "bins_z_n": int(len(bins_z)),
        "bins_z_min": float(bins_z_edges[0]),
        "bins_z_max": float(bins_z_edges[-1]),
        "duration_s": float(series.time[-1] - series.time[0]) if len(series.time) > 1 else 0.0,
        "sampling_frequency_Hz": fs,
        "samples_n": int(len(series.data)),
        "grid_type": "linear",
    }

    return SignalStatistics(D2=histogram, x=bins_x, y=bins_y, z=bins_z, source_name=series.name, signal_id=signal_key, characterization_id=characterization_key, metadata=metadata, simulation_parameters_id=simulation_parameters_id, signal_statistics_parameters_id=signal_statistics_parameters_id)


@timed("compute_D2")
def compute_D2(series, characterization_or_m, bins_x_n=None, bins_x_min=None, bins_x_max=None, bins_y_n=101, bins_y_min=None, bins_y_max=None, bins_z_n=128, bins_z_min=None, bins_z_max=None, signal_key=None, characterization_key=None, simulation_parameters_id=None, signal_statistics_parameters_id=None):
    """Backward-compatible D2 wrapper."""
    return compute_signal_statistics(series, characterization_or_m, order="D2", bins_x_n=bins_x_n, bins_x_min=bins_x_min, bins_x_max=bins_x_max, bins_y_n=bins_y_n, bins_y_min=bins_y_min, bins_y_max=bins_y_max, bins_z_n=bins_z_n, bins_z_min=bins_z_min, bins_z_max=bins_z_max, signal_key=signal_key, characterization_key=characterization_key, simulation_parameters_id=simulation_parameters_id, signal_statistics_parameters_id=signal_statistics_parameters_id)

def make_homogeneous_signal_statistics(signal_statistics, source_name_suffix=" [homogeneous D2]"):
    source = coerce_signal_statistics(signal_statistics)
    shape = tuple(int(v) for v in source.D2.shape)
    total_bins = int(np.prod(shape))
    if total_bins <= 0:
        raise ValueError("Cannot homogenize an empty D2 grid")
    D2 = np.full(shape, 1.0 / total_bins, dtype=float)
    metadata = dict(source.metadata or {})
    metadata.update({"homogeneous_D2": True, "homogeneous_D2_bins_n": total_bins, "original_signal_statistics_id": source.id})
    return SignalStatistics(D2=D2, x=np.asarray(source.x, dtype=float).copy(), y=np.asarray(source.y, dtype=float).copy(), z=np.asarray(source.z, dtype=float).copy(), source_name=(source.source_name or "signal") + source_name_suffix, signal_id=source.signal_id, characterization_id=source.characterization_id, metadata=metadata, simulation_parameters_id=source.simulation_parameters_id, signal_statistics_parameters_id=source.signal_statistics_parameters_id)


def plot_D2(D2_data, figsize=(9, 7)):
    from matplotlib.colors import LogNorm
    D2 = D2_data["D2"]
    bins_x = D2_data["x"]
    bins_y = D2_data["y"]
    bins_z = D2_data["z"]
    ix, iy, iz = np.nonzero(D2)
    probability = D2[ix, iy, iz]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(bins_x[ix], bins_y[iy], bins_z[iz], c=probability, cmap="viridis", norm=LogNorm(vmin=probability.min(), vmax=probability.max()))
    ax.set_xlabel("V")
    ax.set_ylabel("V'")
    ax.set_zlabel("V''")
    fig.colorbar(scatter, ax=ax, pad=0.12, label="P(V, V', V'')")
    fig.tight_layout()
    return fig, ax

# =============================================================================
# prepare.py
# =============================================================================

@dataclass
class StatisticalState:
    characterization: ADCCharacterization
    signal_statistics: SignalStatistics
    W: np.ndarray
    m: dict
    D2_data: SignalStatistics
    w: np.ndarray
    Vs: np.ndarray
    DVs: np.ndarray
    D2Vs: np.ndarray
    V_edges: np.ndarray
    DV_edges: np.ndarray
    D2s: np.ndarray
    D0s: np.ndarray
    D1s: np.ndarray
    D2_conditional_VDV: np.ndarray
    Wn: np.ndarray
    W_VDV: np.ndarray
    V_to_W: np.ndarray
    DV_to_W: np.ndarray

    @property
    def NVS(self):
        return len(self.Vs)

    @property
    def NDVS(self):
        return len(self.DVs)

    @property
    def ND2VS(self):
        return len(self.D2Vs)

    @property
    def LW(self):
        return len(self.w)


def _coerce_inputs(characterization_or_W, signal_or_m, D2_data=None):
    if isinstance(characterization_or_W, ADCCharacterization):
        if D2_data is not None:
            raise TypeError("When passing ADCCharacterization, pass SignalStatistics as the second argument only.")
        characterization = characterization_or_W
        signal_statistics = coerce_signal_statistics(signal_or_m)
    else:
        if D2_data is None:
            raise TypeError("Legacy usage requires prepare_statistical_state(W, m, D2_data).")
        characterization = coerce_characterization(characterization_or_W, m=signal_or_m)
        signal_statistics = coerce_signal_statistics(D2_data)
    return characterization, signal_statistics


def _linear_indices(axis, values):
    axis = np.asarray(axis, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(axis)
    axis_sorted = axis[order]

    pos = np.searchsorted(axis_sorted, values, side="right")
    i1s = np.clip(pos, 1, len(axis_sorted) - 1)
    i0s = i1s - 1

    below = values <= axis_sorted[0]
    above = values >= axis_sorted[-1]
    i0s[below] = 0
    i1s[below] = 0
    i0s[above] = len(axis_sorted) - 1
    i1s[above] = len(axis_sorted) - 1

    x0 = axis_sorted[i0s]
    x1 = axis_sorted[i1s]
    alpha = np.divide(values - x0, x1 - x0, out=np.zeros_like(values), where=x1 != x0)
    alpha = np.clip(alpha, 0.0, 1.0)
    return order[i0s], order[i1s], alpha


def _signed_linear_indices(axis, values):
    """Interpolate V' only within the same sign; never bridge the uncharacterized zero gap."""
    axis = np.asarray(axis, dtype=float)
    values = np.asarray(values, dtype=float)
    i0 = np.zeros(len(values), dtype=int)
    i1 = np.zeros(len(values), dtype=int)
    alpha = np.zeros(len(values), dtype=float)

    neg_indices = np.flatnonzero(axis < 0)
    pos_indices = np.flatnonzero(axis > 0)
    if len(neg_indices) == 0 or len(pos_indices) == 0:
        return _linear_indices(axis, values)

    for mask, indices in ((values < 0, neg_indices), (values > 0, pos_indices)):
        if not np.any(mask):
            continue
        a0, a1, aa = _linear_indices(axis[indices], values[mask])
        i0[mask] = indices[a0]
        i1[mask] = indices[a1]
        alpha[mask] = aa

    zero = values == 0
    if np.any(zero):
        neg_near = neg_indices[np.argmax(axis[neg_indices])]
        pos_near = pos_indices[np.argmin(axis[pos_indices])]
        i0[zero] = neg_near
        i1[zero] = pos_near
        alpha[zero] = 0.5

    return i0, i1, alpha


def _interpolate_W_to_signal_grid(Wn, x_char, dv_char, Vs, DVs):
    vx0, vx1, av = _linear_indices(x_char, Vs)
    dy0, dy1, ad = _signed_linear_indices(dv_char, DVs)

    out = np.zeros((len(Vs), len(DVs), Wn.shape[2]), dtype=float)
    for iv in range(len(Vs)):
        Wv0 = (1.0 - av[iv]) * Wn[vx0[iv]] + av[iv] * Wn[vx1[iv]]
        out[iv] = (1.0 - ad)[:, None] * Wv0[dy0] + ad[:, None] * Wv0[dy1]

    total = np.sum(out, axis=2, keepdims=True)
    out = np.divide(out, total, out=np.zeros_like(out), where=total > 0)
    return out


def prepare_statistical_state(characterization_or_W, signal_or_m, D2_data=None):
    """Prepare W and D2 on independent grids.

    W is linearly interpolated from the sparse characterization grid onto the D2
    V/V' grid. V' interpolation is sign-preserving, so positive and negative
    characterization branches are never blended through zero.
    """
    characterization, signal_statistics = _coerce_inputs(characterization_or_W, signal_or_m, D2_data)

    W = np.asarray(characterization.W)
    m = characterization.m
    V = np.asarray(signal_statistics.x)
    DV = np.asarray(signal_statistics.y)
    D2V = np.asarray(signal_statistics.z)
    D2 = np.asarray(signal_statistics.D2)

    W_valid = np.where(np.isfinite(W), W, 0.0)
    w_full = np.asarray(m["z"])
    w_nonzero = np.any(W_valid > 0, axis=(0, 1))
    w_idx = np.flatnonzero(w_nonzero)
    if len(w_idx) == 0:
        raise RuntimeError("W contains no non-zero probability.")

    lw0, lw1 = w_idx[0], w_idx[-1]
    W = W[:, :, lw0:lw1 + 1]
    w = w_full[lw0:lw1 + 1]

    D2_nonzero = D2 > 0
    V_nonzero = np.flatnonzero(np.any(D2_nonzero, axis=(1, 2)))
    DV_nonzero = np.flatnonzero(np.any(D2_nonzero, axis=(0, 2)))
    D2V_nonzero = np.flatnonzero(np.any(D2_nonzero, axis=(0, 1)))
    if len(V_nonzero) == 0 or len(DV_nonzero) == 0 or len(D2V_nonzero) == 0:
        raise RuntimeError("D2 contains no non-zero probability.")

    i0, i1 = V_nonzero[0], V_nonzero[-1]
    j0, j1 = DV_nonzero[0], DV_nonzero[-1]
    k0, k1 = D2V_nonzero[0], D2V_nonzero[-1]

    Vs = V[i0:i1 + 1]
    V_full_edges = np.array([-np.inf, np.inf]) if len(V) == 1 else centers_to_edges(V)
    V_edges = V_full_edges[i0:i1 + 2]
    DVs = DV[j0:j1 + 1]
    DV_full_edges = np.array([-np.inf, np.inf]) if len(DV) == 1 else centers_to_edges(DV)
    DV_edges = DV_full_edges[j0:j1 + 2]
    D2Vs = D2V[k0:k1 + 1]
    D2s = D2[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1]
    D0s = np.sum(D2s, axis=(1, 2))
    D1s = np.sum(D2s, axis=2)
    D2_conditional_VDV = np.divide(D2s, D1s[:, :, None], out=np.zeros_like(D2s, dtype=float), where=D1s[:, :, None] > 0)

    W_clean = np.where(np.isfinite(W), W, 0.0)
    W_sum = np.sum(W_clean, axis=2, keepdims=True)
    Wn = np.divide(W_clean, W_sum, out=np.zeros_like(W_clean), where=W_sum > 0)

    x_char = np.asarray(m["x"], dtype=float)
    dv_char = np.asarray(m["y"], dtype=float)
    W_VDV = _interpolate_W_to_signal_grid(Wn, x_char, dv_char, Vs, DVs)

    V_to_W = np.argmin(np.abs(x_char[:, None] - Vs[None, :]), axis=0)
    DV_to_W = np.argmin(np.abs(dv_char[:, None] - DVs[None, :]), axis=0)

    return StatisticalState(characterization=characterization, signal_statistics=signal_statistics, W=W, m=m, D2_data=signal_statistics, w=w, Vs=Vs, DVs=DVs, D2Vs=D2Vs, V_edges=V_edges, DV_edges=DV_edges, D2s=D2s, D0s=D0s, D1s=D1s, D2_conditional_VDV=D2_conditional_VDV, Wn=Wn, W_VDV=W_VDV, V_to_W=V_to_W, DV_to_W=DV_to_W)

# =============================================================================
# kinematics.py
# =============================================================================

def get_cross_time_matrix(remaining, vp, vpp):
    remaining = np.asarray(remaining, dtype=np.float64)
    tx = np.full(remaining.shape, np.inf, dtype=np.float64)
    tx[remaining <= 0] = 0.0

    s = np.sign(vp)
    if s == 0:
        return tx

    A = 0.5 * s * vpp
    B = s * vp
    active = remaining > 0

    if np.abs(A) < 1e-30:
        tx[active] = remaining[active] / B
        return tx

    discriminant = B**2 + 4 * A * remaining
    valid = active & (discriminant >= 0)
    sqrt_discriminant = np.zeros(remaining.shape, dtype=np.float64)
    sqrt_discriminant[valid] = np.sqrt(discriminant[valid])

    t1 = np.full(remaining.shape, np.inf, dtype=np.float64)
    t2 = np.full(remaining.shape, np.inf, dtype=np.float64)
    t1[valid] = (-B + sqrt_discriminant[valid]) / (2 * A)
    t2[valid] = (-B - sqrt_discriminant[valid]) / (2 * A)
    t1[t1 < 0] = np.inf
    t2[t2 < 0] = np.inf
    tx[valid] = np.minimum(t1[valid], t2[valid])
    return tx


def get_exit_time(j, vpp, DVs, DV_edges=None):
    DVs = np.asarray(DVs, dtype=float)
    if DV_edges is None:
        DV_edges = centers_to_edges(DVs)

    if vpp > 0:
        t_exit = (DV_edges[j + 1] - DVs[j]) / vpp
        j_next = j + 1 if j < len(DVs) - 1 else -1
        return t_exit, j_next

    if vpp < 0:
        t_exit = (DV_edges[j] - DVs[j]) / vpp
        j_next = j - 1 if j > 0 else -1
        return t_exit, j_next

    return np.inf, -1

# =============================================================================
# markov.py
# =============================================================================

@dataclass
class MarkovTransitionResults:
    source_event_PMF_VDV: np.ndarray
    endpoint_event_PMF_VDV: np.ndarray
    approximate_intensity_Hz: float
    stationary_error_L1: float
    stationary_iterations_n: int
    successful_transition_mass: float
    directional_crossing_mass: float
    direction_change_mass: float
    same_direction_mass: float
    reversal_mass: float
    domain_escape_mass: float
    stuck_mass: float
    time_sum_s: float
    time2_sum_s2: float
    rate_sum_Hz: float
    rate2_sum_Hz2: float
    min_rate_Hz: float
    max_rate_Hz: float
    crossing_time_s: np.ndarray
    crossing_time_PMF: np.ndarray
    same_width_distribution_VW: np.ndarray
    DW: np.ndarray
    P_DW_same_V: np.ndarray
    reversal_error_V: np.ndarray
    P_reversal_error_V: np.ndarray
    output_direction_sum: float
    actual_displacement_sum_V: float
    same_output_direction_sum: float
    same_actual_displacement_sum_V: float
    reversal_output_direction_sum: float
    reversal_actual_displacement_sum_V: float
    signal_duration_s: float = 0.0


@dataclass
class _MomentState:
    mass: np.ndarray
    time1: np.ndarray
    time2: np.ndarray
    disp1: np.ndarray


def _merge_state(dst, src):
    if dst is None:
        return _MomentState(src.mass.copy(), src.time1.copy(), src.time2.copy(), src.disp1.copy())
    dst.mass += src.mass
    dst.time1 += src.time1
    dst.time2 += src.time2
    dst.disp1 += src.disp1
    return dst


def _positive_root(x0, v0, a, boundary):
    c = x0 - boundary
    if abs(a) < 1e-30:
        if abs(v0) < 1e-30:
            return np.inf
        t = -c / v0
        return t if t > 1e-15 else np.inf
    disc = v0 * v0 - 2.0 * a * c
    if disc < 0:
        return np.inf
    sd = np.sqrt(disc)
    roots = [(-v0 + sd) / a, (-v0 - sd) / a]
    roots = [t for t in roots if np.isfinite(t) and t > 1e-15]
    return min(roots) if roots else np.inf


def _state_exit(i, j, k, state):
    x = float(state.Vs[i])
    v = float(state.DVs[j])
    a = float(state.D2Vs[k])

    t_x0 = _positive_root(x, v, a, state.V_edges[i]) if i > 0 else np.inf
    t_x1 = _positive_root(x, v, a, state.V_edges[i + 1]) if i < state.NVS - 1 else np.inf
    t_x = min(t_x0, t_x1)

    if a > 0 and j < state.NDVS - 1:
        t_v = (state.DV_edges[j + 1] - v) / a
    elif a < 0 and j > 0:
        t_v = (state.DV_edges[j] - v) / a
    else:
        t_v = np.inf
    if not np.isfinite(t_v) or t_v <= 1e-15:
        t_v = np.inf

    t_exit = min(t_x, t_v)
    if not np.isfinite(t_exit):
        return np.inf, -1, -1, 0.0

    x_end = x + v * t_exit + 0.5 * a * t_exit**2
    v_end = v + a * t_exit
    dx = float(v * t_exit + 0.5 * a * t_exit**2)

    # Step just across the boundary/boundaries that were reached.
    finite_x_edges = state.V_edges[np.isfinite(state.V_edges)]
    finite_v_edges = state.DV_edges[np.isfinite(state.DV_edges)]
    eps_x = (np.min(np.diff(finite_x_edges)) if len(finite_x_edges) > 1 else 1.0) * 1e-9
    eps_v = (np.min(np.diff(finite_v_edges)) if len(finite_v_edges) > 1 else 1.0) * 1e-9
    if t_x <= t_v:
        direction_x = np.sign(v_end if abs(v_end) > 1e-30 else v)
        x_end += direction_x * eps_x
    if t_v <= t_x:
        v_end += np.sign(a) * eps_v

    i_next = int(np.searchsorted(state.V_edges, x_end, side='right') - 1)
    j_next = int(np.searchsorted(state.DV_edges, v_end, side='right') - 1)
    if not (0 <= i_next < state.NVS and 0 <= j_next < state.NDVS):
        return t_exit, -1, -1, dx
    return t_exit, i_next, j_next, dx


def _advance_for_time(i, j, k, dt, state):
    x = float(state.Vs[i])
    v = float(state.DVs[j])
    a = float(state.D2Vs[k])
    dx = float(v * dt + 0.5 * a * dt**2)
    x_end = x + dx
    v_end = v + a * dt
    i_next = int(np.searchsorted(state.V_edges, x_end, side="right") - 1)
    j_next = int(np.searchsorted(state.DV_edges, v_end, side="right") - 1)
    if not (0 <= i_next < state.NVS and 0 <= j_next < state.NDVS):
        return -1, -1, dx
    return i_next, j_next, dx


def _cross_times_same(hs, w, source_sign, vp, vpp):
    # hs is the signed excursion magnitude in the source-event direction.
    remaining = w[None, :] - hs[:, None]
    tx = np.full(remaining.shape, np.inf, dtype=float)
    tx[remaining <= 0] = 0.0
    u0 = source_sign * vp
    a = source_sign * vpp
    active = remaining > 0
    if u0 <= 0:
        return tx
    if abs(a) < 1e-30:
        tx[active] = remaining[active] / u0
        return tx
    disc = u0 * u0 + 2.0 * a * remaining
    valid = active & (disc >= 0)
    sd = np.zeros_like(remaining)
    sd[valid] = np.sqrt(disc[valid])
    t1 = np.full_like(remaining, np.inf)
    t2 = np.full_like(remaining, np.inf)
    t1[valid] = (-u0 + sd[valid]) / a
    t2[valid] = (-u0 - sd[valid]) / a
    t1[t1 < 0] = np.inf
    t2[t2 < 0] = np.inf
    tx[valid] = np.minimum(t1[valid], t2[valid])
    return tx


def _cross_times_reversal(hs, source_sign, vp, vpp):
    u0 = source_sign * vp
    a = source_sign * vpp
    tx = np.full(len(hs), np.inf, dtype=float)
    # hs is the remaining signed distance back to the previous reported crossing.
    # Moment rebinning can legitimately place probability exactly at hs=0 while
    # crossing a V/V' state boundary. That mass has already recaptured the previous
    # threshold and must be absorbed immediately; ignoring it systematically loses
    # reversal events on fine multi-state paths.
    at_threshold = hs <= 1e-15
    tx[at_threshold] = 0.0
    active = hs > 1e-15
    if not np.any(active):
        return tx
    if abs(a) < 1e-30:
        if u0 < 0:
            tx[active] = hs[active] / (-u0)
        return tx
    disc = u0 * u0 - 2.0 * a * hs
    valid = active & (disc >= 0)
    sd = np.zeros_like(hs)
    sd[valid] = np.sqrt(disc[valid])
    t1 = np.full_like(hs, np.inf)
    t2 = np.full_like(hs, np.inf)
    t1[valid] = (-u0 + sd[valid]) / a
    t2[valid] = (-u0 - sd[valid]) / a
    t1[t1 <= 1e-15] = np.inf
    t2[t2 <= 1e-15] = np.inf
    tx[valid] = np.minimum(t1[valid], t2[valid])
    return tx


def _linear_bins(values, axis):
    values = np.clip(np.asarray(values, dtype=float), axis[0], axis[-1])
    p = (values - axis[0]) * (len(axis) - 1) / max(axis[-1] - axis[0], 1e-30)
    i0 = np.floor(p).astype(int)
    i1 = np.minimum(i0 + 1, len(axis) - 1)
    a = p - i0
    return i0, i1, a


def _rebin_moments(moment, values, axis, probability=None, add_time=0.0, add_disp=0.0):
    if probability is None:
        probability = np.ones_like(moment.mass)
    probability = np.asarray(probability, dtype=float)
    m = moment.mass * probability
    t1 = moment.time1 * probability + m * add_time
    t2 = moment.time2 * probability + 2.0 * add_time * moment.time1 * probability + m * add_time**2
    x1 = moment.disp1 * probability + m * add_disp
    i0, i1, a = _linear_bins(values, axis)
    out = _MomentState(np.zeros_like(axis), np.zeros_like(axis), np.zeros_like(axis), np.zeros_like(axis))
    np.add.at(out.mass, i0, m * (1.0 - a)); np.add.at(out.mass, i1, m * a)
    np.add.at(out.time1, i0, t1 * (1.0 - a)); np.add.at(out.time1, i1, t1 * a)
    np.add.at(out.time2, i0, t2 * (1.0 - a)); np.add.at(out.time2, i1, t2 * a)
    np.add.at(out.disp1, i0, x1 * (1.0 - a)); np.add.at(out.disp1, i1, x1 * a)
    return out


def initial_event_source(state):
    mean_w = np.sum(state.W_VDV * state.w[None, None, :], axis=2)
    local_rate = np.divide(np.abs(state.DVs)[None, :], mean_w, out=np.zeros_like(mean_w), where=mean_w > 0)
    raw = state.D1s * local_rate
    intensity = float(np.sum(raw))
    if intensity <= 0:
        raise RuntimeError('Could not construct the initial event source from D2 and W.')
    return raw / intensity, intensity


def _prune_and_normalize(source, floor):
    source = np.where(source >= floor, source, 0.0)
    total = float(np.sum(source))
    if total <= 0:
        i, j = np.unravel_index(np.argmax(source), source.shape)
        source[i, j] = 1.0
        return source
    return source / total


def _scale_moment(moment, scale):
    return _MomentState(moment.mass * scale, moment.time1 * scale, moment.time2 * scale, moment.disp1 * scale)


def propagate_event_distribution(state, source_event_PMF_VDV, headstart_bins_n=None, crossing_time_bins_n=128, reversal_error_bins_n=127, transition_probability_floor=1e-12, max_state_transitions_n=96, reference_width_V=None, collect_distributions=True, progress=False, stop_on_direction_change=False, report_reversals=True):
    """Propagate the histogram-only next-event model.

    V'' is conditionally resampled from D2 whenever a path enters an occupied
    (V,V') state. Instead of materializing one frontier branch per V'' bin, the
    conditional acceleration distribution is integrated inside the state. This
    is exactly the same Markov closure but avoids an ND2V-fold state explosion,
    which is essential for the homogeneous-D2 diagnostic.

    ``k_lock >= 0`` is used only when a path enters a D2-empty state (ballistic
    continuation with the previous acceleration) or after an internally
    suppressed reversal recapture, where the current acceleration must continue
    until the next state transition. Otherwise ``k_lock == -1`` and V'' is
    integrated from P(V''|V,V').
    """
    NHS = state.LW if headstart_bins_n is None else int(headstart_bins_n)
    hs_axis = np.linspace(0.0, float(np.max(state.w)), NHS)
    width_edges = centers_to_edges(state.w)
    width_edges[0] = np.nextafter(width_edges[0], -np.inf)
    width_edges[-1] = np.nextafter(width_edges[-1], np.inf)
    dw_step = float(np.mean(np.diff(state.w)))
    DW = np.arange(-(state.LW - 1), state.LW) * dw_step
    reversal_limit = max(2.0 * float(np.max(state.w)), 1e-12)
    reversal_edges = np.linspace(-reversal_limit, reversal_limit, int(reversal_error_bins_n) + 1)
    reversal_axis = 0.5 * (reversal_edges[:-1] + reversal_edges[1:])

    duration_s = float((state.signal_statistics.metadata or {}).get("duration_s", 1.0))
    min_w = float(np.min(state.w[state.w > 0]))
    max_dv = max(float(np.max(np.abs(state.DVs))), 1e-12)
    time_edges = np.geomspace(max(1e-12, min_w / max_dv / 100.0), max(duration_s, min_w / max_dv * 10.0), int(crossing_time_bins_n) + 1)
    time_axis = np.sqrt(time_edges[:-1] * time_edges[1:])
    time_hist = np.zeros(len(time_axis), dtype=float)

    endpoint = np.zeros_like(source_event_PMF_VDV, dtype=float)
    P_same_width_V = np.zeros((state.NVS, state.LW), dtype=float)
    P_reversal_V = np.zeros((state.NVS, len(reversal_axis)), dtype=float)

    success = directional_success = direction_change_mass = same_mass = reversal_mass = escaped = stuck = 0.0
    time_sum = time2_sum = rate_sum = rate2_sum = 0.0
    min_rate = np.inf
    max_rate = 0.0
    out_dir_sum = act_disp_sum = 0.0
    same_out_dir_sum = same_act_disp_sum = 0.0
    rev_out_dir_sum = rev_act_disp_sum = 0.0

    # key = (V index, V' index, k_lock, active target sign,
    #        previous reported-event sign, direction_changed_since_source)
    # Prune numerically negligible source states before starting the expensive
    # multi-state propagation. Renormalization preserves the conditional event
    # distribution while runtime scales with relevant states rather than every
    # occupied D2 cell.
    source_event_PMF_VDV = np.asarray(source_event_PMF_VDV, dtype=float)
    source_event_PMF_VDV = np.where(source_event_PMF_VDV >= transition_probability_floor, source_event_PMF_VDV, 0.0)
    source_total = float(np.sum(source_event_PMF_VDV))
    if source_total <= 0:
        source_event_PMF_VDV = np.asarray(source_event_PMF_VDV, dtype=float)
        raise RuntimeError("All event-source states were pruned; lower transition_probability_floor.")
    source_event_PMF_VDV = source_event_PMF_VDV / source_total

    frontier = {}
    for i, j in np.argwhere(source_event_PMF_VDV > 0):
        p_source = float(source_event_PMF_VDV[i, j])
        source_sign = int(np.sign(state.DVs[j]))
        if source_sign == 0 or p_source <= 0:
            continue
        vec = np.zeros(NHS, dtype=float)
        vec[0] = p_source
        key = (int(i), int(j), -1, source_sign, source_sign, False)
        frontier[key] = _merge_state(frontier.get(key), _MomentState(vec, np.zeros(NHS), np.zeros(NHS), np.zeros(NHS)))

    for depth in range(int(max_state_transitions_n)):
        if not frontier:
            break
        next_frontier = {}

        for (i, j, k_lock, target_sign, reported_sign, direction_changed), moment_total in frontier.items():
            total_mass = float(np.sum(moment_total.mass))
            if total_mass <= transition_probability_floor:
                stuck += total_mass
                continue

            if k_lock >= 0:
                k_options = ((int(k_lock), 1.0),)
            else:
                pk = np.asarray(state.D2_conditional_VDV[i, j], dtype=float)
                active_k = np.flatnonzero((pk > 0) & (pk * total_mass > transition_probability_floor))
                if len(active_k) == 0:
                    active_k = np.flatnonzero(pk > 0)
                if len(active_k) == 0:
                    # This should only occur for a source bug; D2-supported source
                    # states are occupied and empty destination states are k-locked.
                    stuck += total_mass
                    continue
                k_options = tuple((int(k), float(pk[k])) for k in active_k)

            for k, p_k in k_options:
                if p_k <= 0:
                    continue
                moment = moment_total if p_k == 1.0 else _scale_moment(moment_total, p_k)
                branch_mass_total = float(np.sum(moment.mass))
                if branch_mass_total <= transition_probability_floor:
                    continue

                vp = float(state.DVs[j])
                vpp = float(state.D2Vs[k])
                current_sign = int(np.sign(vp))
                Pw = state.W_VDV[i, j]
                if np.sum(Pw) <= 0:
                    stuck += branch_mass_total
                    continue

                t_exit, i_next, j_next, dx_exit = _state_exit(i, j, k, state)

                mass_h = moment.mass
                active_h = mass_h > 0
                t_before = np.divide(moment.time1, mass_h, out=np.zeros_like(mass_h), where=active_h)
                t2_before = np.divide(moment.time2, mass_h, out=np.zeros_like(mass_h), where=active_h)
                x_before = np.divide(moment.disp1, mass_h, out=np.zeros_like(mass_h), where=active_h)

                if target_sign * current_sign > 0:
                    tx = _cross_times_same(hs_axis, state.w, target_sign, vp, vpp)
                    cross_mask = np.isfinite(tx) & ((tx <= t_exit) if np.isfinite(t_exit) else True)
                    M = mass_h[:, None] * Pw[None, :] * cross_mask
                    cross_total = float(np.sum(M))

                    if cross_total > 0:
                        dt = np.where(cross_mask, tx, 0.0)
                        TT = t_before[:, None] + dt
                        local_dx = vp * dt + 0.5 * vpp * dt**2
                        XX = x_before[:, None] + local_dx
                        T2 = t2_before[:, None] + 2.0 * dt * t_before[:, None] + dt**2

                        success += cross_total
                        endpoint[i, j] += cross_total
                        if not direction_changed:
                            directional_success += cross_total
                        time_sum += float(np.sum(M * TT))
                        time2_sum += float(np.sum(M * T2))

                        valid_rate = (M > 0) & np.isfinite(TT) & (TT > 0)
                        if np.any(valid_rate):
                            rates = 1.0 / TT[valid_rate]
                            weights = M[valid_rate]
                            rate_sum += float(np.sum(weights * rates))
                            rate2_sum += float(np.sum(weights * rates**2))
                            min_rate = min(min_rate, float(np.min(rates)))
                            max_rate = max(max_rate, float(np.max(rates)))
                            time_hist += np.histogram(TT[valid_rate], bins=time_edges, weights=weights)[0]

                        final_sign = current_sign
                        out_dir_sum += final_sign * cross_total
                        act_disp_sum += float(np.sum(M * XX))
                        if final_sign * reported_sign > 0:
                            same_mass += cross_total
                            same_out_dir_sum += final_sign * cross_total
                            same_act_disp_sum += float(np.sum(M * XX))
                        else:
                            reversal_mass += cross_total
                            rev_out_dir_sum += final_sign * cross_total
                            rev_act_disp_sum += float(np.sum(M * XX))

                        if collect_distributions:
                            valid_width = M > 0
                            P_same_width_V[i] += np.histogram(np.abs(XX[valid_width]), bins=width_edges, weights=M[valid_width])[0]

                    exit_weight_hw = (~cross_mask) * Pw[None, :]
                    noncross = np.clip(np.sum(exit_weight_hw, axis=1), 0.0, 1.0)

                else:
                    tx = _cross_times_reversal(hs_axis, target_sign, vp, vpp)
                    crossed = np.isfinite(tx) & ((tx <= t_exit) if np.isfinite(t_exit) else True)
                    M = mass_h * crossed
                    cross_total = float(np.sum(M))

                    if cross_total > 0:
                        dt = np.where(crossed, tx, 0.0)
                        TT = t_before + dt
                        local_dx = vp * dt + 0.5 * vpp * dt**2
                        XX = x_before + local_dx
                        T2 = t2_before + 2.0 * dt * t_before + dt**2

                        if report_reversals:
                            success += cross_total
                            reversal_mass += cross_total
                            endpoint[i, j] += cross_total
                            time_sum += float(np.sum(M * TT))
                            time2_sum += float(np.sum(M * T2))

                            valid_rate = (M > 0) & np.isfinite(TT) & (TT > 0)
                            if np.any(valid_rate):
                                rates = 1.0 / TT[valid_rate]
                                weights = M[valid_rate]
                                rate_sum += float(np.sum(weights * rates))
                                rate2_sum += float(np.sum(weights * rates**2))
                                min_rate = min(min_rate, float(np.min(rates)))
                                max_rate = max(max_rate, float(np.max(rates)))
                                time_hist += np.histogram(TT[valid_rate], bins=time_edges, weights=weights)[0]

                            final_sign = current_sign
                            # A reversal/recapture should ideally occur at exactly
                            # the previous crossing amplitude.  Therefore the
                            # propagated signed displacement XX is itself the
                            # physical recapture error.  Do not add a nominal-LSB
                            # or mean-width correction here; Pass 12 accounts for
                            # the complete recapture displacement directly in drift.
                            recapture = XX
                            out_dir_sum += final_sign * cross_total
                            act_disp_sum += float(np.sum(M * recapture))
                            rev_out_dir_sum += final_sign * cross_total
                            rev_act_disp_sum += float(np.sum(M * recapture))
                            if collect_distributions:
                                valid_rev = M > 0
                                P_reversal_V[i] += np.histogram(recapture[valid_rev], bins=reversal_edges, weights=M[valid_rev])[0]
                        else:
                            # Internal recapture: all crossed headstart states reset to
                            # zero excursion and can therefore be merged exactly.
                            branch = _MomentState(np.zeros(NHS), np.zeros(NHS), np.zeros(NHS), np.zeros(NHS))
                            branch.mass[0] = cross_total
                            branch.time1[0] = float(np.sum((moment.time1 + mass_h * dt) * crossed))
                            branch.time2[0] = float(np.sum((moment.time2 + 2.0 * dt * moment.time1 + mass_h * dt**2) * crossed))
                            branch.disp1[0] = float(np.sum((moment.disp1 + mass_h * local_dx) * crossed))
                            key = (i, j, k, current_sign, reported_sign, True)
                            next_frontier[key] = _merge_state(next_frontier.get(key), branch)

                    noncross = (~crossed).astype(float)

                continue_mass = float(np.sum(moment.mass * noncross))
                if continue_mass <= transition_probability_floor:
                    continue
                if not np.isfinite(t_exit) or i_next < 0 or j_next < 0:
                    escaped += continue_mass
                    continue

                next_dv_sign = int(np.sign(state.DVs[j_next]))
                if (not direction_changed) and reported_sign * next_dv_sign <= 0:
                    direction_change_mass += continue_mass
                if stop_on_direction_change and reported_sign * next_dv_sign <= 0:
                    stuck += continue_mass
                    continue

                if target_sign * current_sign > 0 and target_sign * next_dv_sign < 0:
                    # Distance back to the previous crossing after a reversal is the
                    # accumulated outward excursion itself. Width only controls which
                    # paths survive without already completing the next full level.
                    h_new = np.clip(hs_axis + target_sign * dx_exit, hs_axis[0], hs_axis[-1])
                    advanced = _rebin_moments(moment, h_new, hs_axis, probability=noncross, add_time=t_exit, add_disp=dx_exit)
                else:
                    hs_new = np.clip(hs_axis + target_sign * dx_exit, hs_axis[0], hs_axis[-1])
                    advanced = _rebin_moments(moment, hs_new, hs_axis, probability=noncross, add_time=t_exit, add_disp=dx_exit)

                pk_next = np.asarray(state.D2_conditional_VDV[i_next, j_next], dtype=float)
                next_lock = -1 if np.any(pk_next > 0) else k
                changed_next = bool(direction_changed or (reported_sign * next_dv_sign <= 0))
                key = (i_next, j_next, int(next_lock), target_sign, reported_sign, changed_next)
                next_frontier[key] = _merge_state(next_frontier.get(key), advanced)

        frontier = next_frontier
        if progress:
            active_mass = sum(float(np.sum(v.mass)) for v in frontier.values())
            print(f"\rProbabilistic propagation step {depth+1:3d}: active states={len(frontier):5d}, active mass={active_mass:.6f}", end="", flush=True)

    if frontier:
        stuck += sum(float(np.sum(v.mass)) for v in frontier.values())
    if progress:
        print()

    endpoint_total = float(np.sum(endpoint))
    endpoint_PMF = endpoint / endpoint_total if endpoint_total > 0 else endpoint
    time_total = float(np.sum(time_hist))
    time_PMF = time_hist / time_total if time_total > 0 else time_hist
    if not np.isfinite(min_rate):
        min_rate = 0.0

    P_DW_same_V = np.zeros((state.NVS, len(DW)), dtype=float)
    if collect_distributions:
        source_width_VW = np.sum(source_event_PMF_VDV[:, :, None] * state.W_VDV, axis=1)
        for i in range(state.NVS):
            sm = float(np.sum(source_width_VW[i]))
            nm = float(np.sum(P_same_width_V[i]))
            if sm <= 0 or nm <= 0:
                continue
            ps = source_width_VW[i] / sm
            pn = P_same_width_V[i] / nm
            joint = nm * ps[:, None] * pn[None, :]
            for delta in range(-(state.LW - 1), state.LW):
                P_DW_same_V[i, delta + state.LW - 1] = np.sum(np.diag(joint, k=delta))

    return MarkovTransitionResults(
        source_event_PMF_VDV=np.asarray(source_event_PMF_VDV), endpoint_event_PMF_VDV=endpoint_PMF, approximate_intensity_Hz=np.nan,
        stationary_error_L1=np.nan, stationary_iterations_n=0, successful_transition_mass=success, directional_crossing_mass=directional_success, direction_change_mass=direction_change_mass, same_direction_mass=same_mass,
        reversal_mass=reversal_mass, domain_escape_mass=escaped, stuck_mass=stuck, time_sum_s=time_sum, time2_sum_s2=time2_sum,
        rate_sum_Hz=rate_sum, rate2_sum_Hz2=rate2_sum, min_rate_Hz=float(min_rate), max_rate_Hz=float(max_rate),
        crossing_time_s=time_axis, crossing_time_PMF=time_PMF, same_width_distribution_VW=P_same_width_V, DW=DW,
        P_DW_same_V=P_DW_same_V, reversal_error_V=reversal_axis, P_reversal_error_V=P_reversal_V,
        output_direction_sum=out_dir_sum, actual_displacement_sum_V=act_disp_sum,
        same_output_direction_sum=same_out_dir_sum, same_actual_displacement_sum_V=same_act_disp_sum,
        reversal_output_direction_sum=rev_out_dir_sum, reversal_actual_displacement_sum_V=rev_act_disp_sum,
        signal_duration_s=duration_s,
    )


def solve_histogram_event_model(state, headstart_bins_n=None, crossing_time_bins_n=128, reversal_error_bins_n=127, transition_probability_floor=1e-8, max_state_transitions_n=64, reference_width_V=None, progress=False, report_reversals=True, collect_distributions=True, source_event_PMF_VDV=None, approximate_intensity_Hz=None):
    """Histogram-only probabilistic next-crossing model.

    D2 remains the authoritative long-recording occupancy. The raw candidate
    crossing flux is D2-weighted |V'|/w. Multi-state first-passage propagation
    then estimates how much of that directed path-length flux reaches a level
    before the derivative reverses. This directional survival probability is
    the correction used for the absolute crossing rate. The same propagation
    also continues beyond reversals to obtain same/reversal transition metrics,
    widths, Delta-w, recapture error and drift terms.
    """
    if source_event_PMF_VDV is None:
        source, approximate_intensity = initial_event_source(state)
    else:
        source = np.asarray(source_event_PMF_VDV, dtype=float)
        total = float(np.sum(source))
        if total <= 0:
            raise RuntimeError("Provided second-order event source contains no probability mass.")
        source = source / total
        approximate_intensity = float(approximate_intensity_Hz) if approximate_intensity_Hz is not None else np.nan
    result = propagate_event_distribution(
        state, source, headstart_bins_n=headstart_bins_n, crossing_time_bins_n=crossing_time_bins_n,
        reversal_error_bins_n=reversal_error_bins_n, transition_probability_floor=transition_probability_floor,
        max_state_transitions_n=max_state_transitions_n, reference_width_V=reference_width_V,
        collect_distributions=collect_distributions, progress=progress, stop_on_direction_change=False, report_reversals=report_reversals,
    )
    result.approximate_intensity_Hz = approximate_intensity
    return result

def solve_stationary_event_model(state, headstart_bins_n=None, crossing_time_bins_n=128, reversal_error_bins_n=127, stationary_iterations_n=8, stationary_tolerance=2e-3, stationary_damping=0.7, source_probability_floor=1e-9, transition_probability_floor=1e-12, max_state_transitions_n=96, reference_width_V=None, progress=False):
    source, approximate_intensity = initial_event_source(state)
    source = _prune_and_normalize(source, source_probability_floor)
    error = np.inf
    iterations_done = 0

    for iteration in range(int(stationary_iterations_n)):
        trial = propagate_event_distribution(state, source, headstart_bins_n=headstart_bins_n, crossing_time_bins_n=max(32, crossing_time_bins_n // 2), reversal_error_bins_n=max(31, reversal_error_bins_n // 2), transition_probability_floor=transition_probability_floor, max_state_transitions_n=max_state_transitions_n, reference_width_V=reference_width_V, collect_distributions=False, progress=False)
        endpoint = trial.endpoint_event_PMF_VDV
        if np.sum(endpoint) <= 0:
            raise RuntimeError('Probabilistic transition model produced no next crossings.')
        error = float(np.sum(np.abs(endpoint - source)))
        iterations_done = iteration + 1
        if progress:
            print(f'Stationary event PMF {iteration+1:2d}: L1={error:.4e}, crossing={trial.successful_transition_mass:.6f}, escape={trial.domain_escape_mass:.3e}, stuck={trial.stuck_mass:.3e}')
        if error <= stationary_tolerance:
            source = endpoint
            break
        source = stationary_damping * endpoint + (1.0 - stationary_damping) * source
        source = _prune_and_normalize(source, source_probability_floor)

    final = propagate_event_distribution(state, source, headstart_bins_n=headstart_bins_n, crossing_time_bins_n=crossing_time_bins_n, reversal_error_bins_n=reversal_error_bins_n, transition_probability_floor=transition_probability_floor, max_state_transitions_n=max_state_transitions_n, reference_width_V=reference_width_V, collect_distributions=True, progress=progress)
    final.approximate_intensity_Hz = approximate_intensity
    final.stationary_error_L1 = error
    final.stationary_iterations_n = iterations_done
    return final

# =============================================================================
# second_order.py
# =============================================================================

"""Second-order D2 corrections that remain fully histogram based.

The first-order path-length flux D1*|V'|/E[w] is corrected with the local
constant-acceleration turning distance.  W remains the ADC-only level-width
model and D2 supplies the probability of (V,V',V'').
"""



@dataclass
class SecondOrderSource:
    local_rate_VDV: np.ndarray
    survival_probability_VDVK: np.ndarray
    raw_rate_VDVK: np.ndarray
    rate_Hz: float
    source_PMF_VDVK: np.ndarray
    source_PMF_VDV: np.ndarray
    source_PMF_V: np.ndarray
    width_raw_VDVW: np.ndarray
    width_PMF_VDVW: np.ndarray


def _turn_survival_for_widths(vp, vpp, widths):
    """Whether a full same-direction width is reached before local turnaround."""
    widths = np.asarray(widths, dtype=float)
    if vp == 0:
        return np.zeros_like(widths, dtype=bool)
    if vp * vpp >= 0 or abs(vpp) < 1e-30:
        return widths > 0
    d_turn = vp * vp / (2.0 * abs(vpp))
    return (widths > 0) & (widths <= d_turn)


@timed("compute_second_order_source")
def compute_second_order_source(state):
    """D2/W same-direction crossing flux using the second-order turning test.

    Candidate renewal flux at each (V,V') is |V'|/E[w]. For each V'' state the
    full W PMF is retained and only widths whose crossing distance is reached
    before the local derivative turns around contribute to same-direction events.
    """
    mean_w = np.sum(state.W_VDV * state.w[None, None, :], axis=2)
    local_rate = np.divide(np.abs(state.DVs)[None, :], mean_w, out=np.zeros_like(mean_w), where=mean_w > 0)

    survival = np.zeros((state.NVS, state.NDVS, state.ND2VS), dtype=float)
    width_raw = np.zeros((state.NVS, state.NDVS, state.LW), dtype=float)
    raw = np.zeros_like(state.D2s, dtype=float)

    for j, vp in enumerate(state.DVs):
        if vp == 0:
            continue
        for k, vpp in enumerate(state.D2Vs):
            mask_w = _turn_survival_for_widths(float(vp), float(vpp), state.w)
            if not np.any(mask_w):
                continue
            p_survive_v = np.sum(state.W_VDV[:, j, :] * mask_w[None, :], axis=1)
            survival[:, j, k] = p_survive_v
            base = state.D2s[:, j, k] * local_rate[:, j]
            raw[:, j, k] = base * p_survive_v
            width_raw[:, j, :] += base[:, None] * state.W_VDV[:, j, :] * mask_w[None, :]

    rate = float(np.sum(raw))
    if rate <= 0:
        raise RuntimeError("Second-order D2/W turning model produced no same-direction crossing flux.")
    pmf = raw / rate
    width_total = float(np.sum(width_raw))
    width_pmf = width_raw / width_total if width_total > 0 else np.zeros_like(width_raw)
    return SecondOrderSource(
        local_rate_VDV=local_rate,
        survival_probability_VDVK=survival,
        raw_rate_VDVK=raw,
        rate_Hz=rate,
        source_PMF_VDVK=pmf,
        source_PMF_VDV=np.sum(pmf, axis=2),
        source_PMF_V=np.sum(pmf, axis=(1, 2)),
        width_raw_VDVW=width_raw,
        width_PMF_VDVW=width_pmf,
    )


def _axis_interp(axis, value, same_sign=False):
    axis = np.asarray(axis, dtype=float)
    if same_sign and value != 0:
        ids = np.flatnonzero(axis * value > 0)
        if len(ids):
            sub = axis[ids]
            if value <= sub[0]:
                return int(ids[0]), int(ids[0]), 0.0
            if value >= sub[-1]:
                return int(ids[-1]), int(ids[-1]), 0.0
            q = int(np.searchsorted(sub, value))
            a0, a1 = int(ids[q - 1]), int(ids[q])
            alpha = (value - axis[a0]) / (axis[a1] - axis[a0])
            return a0, a1, float(alpha)
    if len(axis) == 1:
        return 0, 0, 0.0
    if value <= axis[0]:
        return 0, 0, 0.0
    if value >= axis[-1]:
        return len(axis) - 1, len(axis) - 1, 0.0
    q = int(np.searchsorted(axis, value))
    i0, i1 = q - 1, q
    alpha = (value - axis[i0]) / (axis[i1] - axis[i0])
    return i0, i1, float(alpha)


def interpolate_W(state, V, DV):
    """Bilinearly interpolate W on the already prepared D2 V/V' grid."""
    i0, i1, av = _axis_interp(state.Vs, V, same_sign=False)
    j0, j1, ad = _axis_interp(state.DVs, DV, same_sign=True)
    W00 = state.W_VDV[i0, j0]
    W01 = state.W_VDV[i0, j1]
    W10 = state.W_VDV[i1, j0]
    W11 = state.W_VDV[i1, j1]
    W0 = (1.0 - ad) * W00 + ad * W01
    W1 = (1.0 - ad) * W10 + ad * W11
    out = (1.0 - av) * W0 + av * W1
    total = float(np.sum(out))
    return out / total if total > 0 else np.zeros_like(out)


@timed("compute_second_order_width_delta")
def compute_second_order_width_delta(state, source):
    """Crossing-conditioned same-direction W and consecutive Delta-w.

    When available, the paired Delta-w characterization extracted from the same
    ramp runs is used directly. This retains deterministic adjacent-code DNL and
    stochastic correlation that cannot be recovered from two independent W
    marginals. It is weighted by the second-order D2 crossing source, so no extra
    ADC simulation is required. Legacy characterizations fall back to the
    endpoint-derivative W construction.
    """
    LW = state.LW
    width_VW = np.sum(source.width_raw_VDVW, axis=1)

    # Signed same-direction drift moments come directly from the successful
    # crossing-conditioned width distribution.
    raw_width = source.width_raw_VDVW
    signs = np.sign(state.DVs)[None, :, None]
    signed_direction_sum = float(np.sum(raw_width * signs))
    signed_displacement_sum = float(np.sum(raw_width * signs * state.w[None, None, :]))
    total_flux = float(np.sum(raw_width))

    metadata = state.characterization.metadata or {}
    Q_char = metadata.get("delta_w_characterization_VDV")
    DW_char = metadata.get("delta_w_axis_V")
    if Q_char is not None and DW_char is not None:
        Q_char = np.asarray(Q_char, dtype=float)
        DW = np.asarray(DW_char, dtype=float)
        if Q_char.ndim == 3 and Q_char.shape[:2] == state.characterization.W.shape[:2]:
            Q_signal = _interpolate_W_to_signal_grid(Q_char, np.asarray(state.characterization.m["x"], dtype=float), np.asarray(state.characterization.m["y"], dtype=float), state.Vs, state.DVs)
            P_DW_V = np.sum(source.source_PMF_VDV[:, :, None] * Q_signal, axis=1)
            if np.sum(P_DW_V) > 0:
                return {
                    "DW": DW,
                    "P_DW_V": P_DW_V,
                    "same_width_distribution_VW": width_VW,
                    "signed_direction_sum": signed_direction_sum,
                    "signed_displacement_sum_V": signed_displacement_sum,
                    "total_flux_Hz": total_flux,
                    "delta_w_source": "paired_characterization",
                }

    dw_step = float(np.mean(np.diff(state.w))) if LW > 1 else 1.0
    DW = np.arange(-(LW - 1), LW) * dw_step
    P_DW_V = np.zeros((state.NVS, len(DW)), dtype=float)

    mean_w = np.sum(state.W_VDV * state.w[None, None, :], axis=2)
    local_rate = np.divide(np.abs(state.DVs)[None, :], mean_w, out=np.zeros_like(mean_w), where=mean_w > 0)
    active_states = np.argwhere(state.D2s > 0)
    for i, j, k in active_states:
        vp = float(state.DVs[j])
        vpp = float(state.D2Vs[k])
        s = float(np.sign(vp))
        if s == 0:
            continue
        base = float(state.D2s[i, j, k] * local_rate[i, j])
        if base <= 0:
            continue
        Pw = state.W_VDV[i, j]
        for wi, pwi in enumerate(Pw):
            if pwi <= 0:
                continue
            w0 = float(state.w[wi])
            if not _turn_survival_for_widths(vp, vpp, np.array([w0]))[0]:
                continue
            flux = base * float(pwi)
            rad = vp * vp + 2.0 * s * vpp * w0
            if rad < -1e-14:
                continue
            vf = s * np.sqrt(max(0.0, rad))
            Vnext = float(state.Vs[i] + s * w0)
            Pnext = interpolate_W(state, Vnext, vf)
            if np.sum(Pnext) <= 0:
                continue
            for wf, pwf in enumerate(Pnext):
                if pwf > 0:
                    P_DW_V[i, (wf - wi) + LW - 1] += flux * float(pwf)

    return {
        "DW": DW,
        "P_DW_V": P_DW_V,
        "same_width_distribution_VW": width_VW,
        "signed_direction_sum": signed_direction_sum,
        "signed_displacement_sum_V": signed_displacement_sum,
        "total_flux_Hz": total_flux,
        "delta_w_source": "independent_endpoint_W",
    }

# =============================================================================
# crossings.py
# =============================================================================

@dataclass
class CrossingSourceResults:
    local_rate_DVW: np.ndarray
    approximate_intensity_Hz: float
    cross_source_raw: np.ndarray
    cross_source_PMF: np.ndarray
    cross_source_VDV: np.ndarray
    cross_source_V: np.ndarray
    source_width_raw_VDVW: np.ndarray
    source_width_PMF_VDVW: np.ndarray


@dataclass
class CrossingRateResults:
    # Backward-compatible aliases use the "reversals reported" output policy.
    mean_rate_Hz: float
    interval_mean_rate_Hz: float
    sigma_rate_Hz: float
    min_rate_Hz: float
    max_rate_Hz: float
    mean_interval_s: float
    sigma_interval_s: float
    same_direction_fraction: float
    reversal_fraction: float
    same_direction_rate_Hz: float
    reversal_rate_Hz: float
    successful_transition_mass: float
    cross_source_raw: np.ndarray
    cross_source_PMF: np.ndarray
    cross_source_VDV: np.ndarray
    cross_source_V: np.ndarray
    reversals_reported: CrossingModeRateResults
    reversals_suppressed: CrossingModeRateResults


def compute_crossing_source(state):
    """Build the second-order D2/W same-direction crossing source.

    The D1 renewal flux ``|V'|/E[w]`` is corrected per D2 state with the
    constant-acceleration turning distance ``V'^2/(2|V''|)``.  The complete W
    PMF is retained so broad/narrow levels have the correct probability of
    surviving a local turnaround.
    """

    source = compute_second_order_source(state)

    valid_w = state.w > 0
    local_rate_DVW = np.zeros((state.NDVS, state.LW), dtype=np.float64)
    local_rate_DVW[:, valid_w] = np.abs(state.DVs)[:, None] / state.w[None, valid_w]

    return CrossingSourceResults(
        local_rate_DVW=local_rate_DVW,
        approximate_intensity_Hz=float(source.rate_Hz),
        cross_source_raw=source.raw_rate_VDVK,
        cross_source_PMF=source.source_PMF_VDVK,
        cross_source_VDV=source.source_PMF_VDV,
        cross_source_V=source.source_PMF_V,
        source_width_raw_VDVW=source.width_raw_VDVW,
        source_width_PMF_VDVW=source.width_PMF_VDVW,
    )


def compute_crossing_probability(state, headstart_bins_n=None):
    """Diagnostic: P(cross before leaving ONE derivative bin | V',V'',w)."""
    NHS = state.LW if headstart_bins_n is None else int(headstart_bins_n)
    if NHS < 2:
        raise ValueError("headstart_bins_n must be >= 2")

    fractions = np.linspace(0.0, 1.0, NHS, endpoint=False) + 0.5 / NHS
    crossing_probability = np.zeros((state.NDVS, state.ND2VS, state.LW), dtype=np.float64)

    for j, vp in enumerate(state.DVs):
        if vp == 0:
            continue
        hs = fractions[:, None] * state.w[None, :]
        remaining = state.w[None, :] - hs
        for k, vpp in enumerate(state.D2Vs):
            tx = get_cross_time_matrix(remaining, vp, vpp)
            t_exit, _ = get_exit_time(j, vpp, state.DVs, state.DV_edges)
            crossed = np.isfinite(tx) & ((tx <= t_exit) if np.isfinite(t_exit) else True)
            crossing_probability[j, k, :] = np.mean(crossed, axis=0)

    return crossing_probability


def _weighted_quantile(values, weights, q):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    if len(values) == 0:
        return np.nan
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    return float(values[min(np.searchsorted(cdf, q), len(values) - 1)])


def _mode_rate_statistics(name, transition_results, mean_rate_Hz, duration_s):
    time_axis = np.asarray(transition_results.crossing_time_s, dtype=float)
    time_pmf = np.asarray(transition_results.crossing_time_PMF, dtype=float)
    valid = np.isfinite(time_axis) & (time_axis > 0) & np.isfinite(time_pmf) & (time_pmf > 0)

    if np.any(valid):
        t = time_axis[valid]
        p = time_pmf[valid]
        p = p / np.sum(p)
        rates = 1.0 / t
        interval_mean_rate_Hz = float(np.sum(p * rates))
        sigma_rate_Hz = float(np.sqrt(max(0.0, np.sum(p * (rates - interval_mean_rate_Hz) ** 2))))
        raw_mean_interval_s = float(np.sum(p * t))
        sigma_interval_s = float(np.sqrt(max(0.0, np.sum(p * (t - raw_mean_interval_s) ** 2))))

        expected_n = max(float(mean_rate_Hz) * float(duration_s), 1.0) if duration_s > 0 else 1.0
        tail_q = min(0.25, 1.0 / (expected_n + 1.0))
        min_rate_Hz = _weighted_quantile(rates, p, tail_q)
        max_rate_Hz = _weighted_quantile(rates, p, 1.0 - tail_q)
    else:
        interval_mean_rate_Hz = np.nan
        sigma_rate_Hz = np.nan
        sigma_interval_s = np.nan
        min_rate_Hz = np.nan
        max_rate_Hz = np.nan

    mean_interval_s = 1.0 / mean_rate_Hz if mean_rate_Hz > 0 else np.nan
    return CrossingModeRateResults(
        name=name,
        mean_rate_Hz=float(mean_rate_Hz),
        interval_mean_rate_Hz=interval_mean_rate_Hz,
        sigma_rate_Hz=sigma_rate_Hz,
        min_rate_Hz=min_rate_Hz,
        max_rate_Hz=max_rate_Hz,
        mean_interval_s=mean_interval_s,
        sigma_interval_s=sigma_interval_s,
        crossing_time_s=time_axis,
        crossing_time_PMF=time_pmf,
        events_n=float(mean_rate_Hz * duration_s) if duration_s > 0 else None,
    )


def compute_crossing_rates_from_transitions(source_results, transition_results_reported, transition_results_suppressed=None):
    """Calculate output rates for both hysteresis policies.

    The D2/W candidate flux represents full-level directional progress. The
    probability of completing that progress before the first direction reversal
    gives the full-level crossing rate. This is the output rate when reversal
    recaptures are suppressed.

    With reversal reporting enabled, the same full-level crossing rate is the
    same-direction component of a larger event stream. The total output rate is
    therefore obtained from the modeled same/reversal event fraction. The two
    first-passage PMFs provide their own interval tails and sigma values.
    """
    if transition_results_suppressed is None:
        transition_results_suppressed = transition_results_reported

    # ``source_results.approximate_intensity_Hz`` is already the second-order
    # same-direction full-level crossing flux after the D2 turning correction.
    # Do not multiply by another first-passage survival factor here.
    full_level_rate_Hz = float(source_results.approximate_intensity_Hz)

    same_mass = float(getattr(transition_results_reported, "same_direction_mass", 0.0))
    reversal_mass = float(getattr(transition_results_reported, "reversal_mass", 0.0))
    class_mass = same_mass + reversal_mass
    if class_mass > 0:
        same_fraction = same_mass / class_mass
        reversal_fraction = reversal_mass / class_mass
    else:
        same_fraction = 1.0
        reversal_fraction = 0.0

    reported_rate_Hz = full_level_rate_Hz / same_fraction if same_fraction > 0 else np.nan
    reversal_rate_Hz = max(0.0, reported_rate_Hz - full_level_rate_Hz) if np.isfinite(reported_rate_Hz) else np.nan

    duration_s = float(getattr(transition_results_reported, "signal_duration_s", 0.0))
    if duration_s <= 0:
        duration_s = float(getattr(transition_results_suppressed, "signal_duration_s", 0.0))

    reported_mode = _mode_rate_statistics("reversals_reported", transition_results_reported, reported_rate_Hz, duration_s)
    suppressed_mode = _mode_rate_statistics("reversals_suppressed", transition_results_suppressed, full_level_rate_Hz, duration_s)

    return CrossingRateResults(
        mean_rate_Hz=reported_mode.mean_rate_Hz,
        interval_mean_rate_Hz=reported_mode.interval_mean_rate_Hz,
        sigma_rate_Hz=reported_mode.sigma_rate_Hz,
        min_rate_Hz=reported_mode.min_rate_Hz,
        max_rate_Hz=reported_mode.max_rate_Hz,
        mean_interval_s=reported_mode.mean_interval_s,
        sigma_interval_s=reported_mode.sigma_interval_s,
        same_direction_fraction=float(same_fraction),
        reversal_fraction=float(reversal_fraction),
        same_direction_rate_Hz=float(full_level_rate_Hz),
        reversal_rate_Hz=float(reversal_rate_Hz),
        successful_transition_mass=float(transition_results_reported.successful_transition_mass),
        cross_source_raw=source_results.cross_source_raw,
        cross_source_PMF=source_results.cross_source_PMF,
        cross_source_VDV=source_results.cross_source_VDV,
        cross_source_V=source_results.cross_source_V,
        reversals_reported=reported_mode,
        reversals_suppressed=suppressed_mode,
    )


def compute_crossing_rates(state, headstart_bins_n=None):
    raise RuntimeError("compute_crossing_rates(state) was replaced. Use compute_crossing_source(state), transition propagation, then compute_crossing_rates_from_transitions(...).")

# =============================================================================
# level_width.py
# =============================================================================

@dataclass
class LevelWidthResults:
    occupancy_distribution: np.ndarray
    occupancy_statistics: DistributionStatistics
    crossing_distribution: np.ndarray
    crossing_statistics: DistributionStatistics


def get_maximum_lvlw(P, m, coverage=0.997):
    P = np.asarray(P, dtype=float)
    z = np.asarray(m["z"], dtype=float)
    total_prob = np.sum(P, axis=2, keepdims=True)
    probability_z = np.divide(P, total_prob, out=np.zeros_like(P), where=total_prob > 0)
    cum_prob = np.cumsum(probability_z, axis=2)
    indices = np.argmax(cum_prob >= coverage, axis=2)
    lsb_max = z[indices]
    lsb_max[total_prob[:, :, 0] == 0] = 0
    return lsb_max


def get_mean_lvlw(P, m):
    P = np.asarray(P, dtype=float)
    z = np.asarray(m["z"], dtype=float)
    total_prob = np.sum(P, axis=2, keepdims=True)
    prob_z = np.divide(P, total_prob, out=np.zeros_like(P), where=total_prob > 0)
    lvlw_mean = np.sum(prob_z * z[None, None, :], axis=2)
    lvlw_mean[total_prob[:, :, 0] == 0] = 0
    return lvlw_mean


def get_quant_error_power(P, m):
    P = np.asarray(P, dtype=float)
    z2 = np.asarray(m["z"], dtype=float) ** 2
    P_sum = np.sum(P, axis=2)
    valid = P_sum > 0
    return np.divide(np.sum(P * z2[None, None, :], axis=2), P_sum, out=np.zeros_like(P_sum), where=valid) / 12


def compute_level_width_results(state, local_results):
    """Level width is defined only for same-direction next crossings.

    Occupancy statistics are retained as a characterization diagnostic. The
    crossing statistics used in the model comparison come from the actual signed
    transition displacement generated by the same propagation as Delta-w.
    """
    P_W_D2 = np.sum(state.D1s[:, :, None] * state.W_VDV, axis=(0, 1))
    occupancy_statistics = get_distribution_statistics(state.w, P_W_D2)

    P_W_same = np.sum(local_results.same_width_distribution_VW, axis=0)
    crossing_statistics = get_distribution_statistics(state.w, P_W_same)

    return LevelWidthResults(
        occupancy_distribution=P_W_D2,
        occupancy_statistics=occupancy_statistics,
        crossing_distribution=P_W_same,
        crossing_statistics=crossing_statistics,
    )

# =============================================================================
# dynamic_range.py
# =============================================================================

@dataclass
class DynamicRangeResult:
    lvlw_max: float
    dynamic_range: float
    dynamic_range_dB: float
    ENOB: float
    min_x: float
    max_x: float


def _weighted_coverage_value(values, weights, coverage):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    if len(values) == 0:
        return np.nan
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    return float(values[min(np.searchsorted(cdf, coverage), len(values) - 1)])


def get_crossing_max_lsb_and_range(width_axis, crossing_P_VW, amplitudes_V, coverage, min_x, max_x):
    width_axis = np.asarray(width_axis, dtype=float)
    P = np.asarray(crossing_P_VW, dtype=float)
    amplitudes_V = np.asarray(amplitudes_V, dtype=float)
    amplitude_mask = (amplitudes_V >= min_x) & (amplitudes_V <= max_x)
    selected = P[amplitude_mask]
    if selected.size == 0 or np.sum(selected) <= 0:
        return {"lvlw_max": np.nan, "dynamic_range": np.nan, "dynamic_range_dB": np.nan, "ENOB": np.nan}
    weights_w = np.sum(selected, axis=0)
    lvlw_max = _weighted_coverage_value(width_axis, weights_w, coverage)
    if not np.isfinite(lvlw_max) or lvlw_max <= 0:
        return {"lvlw_max": np.nan, "dynamic_range": np.nan, "dynamic_range_dB": np.nan, "ENOB": np.nan}
    dynamic_range = (max_x - min_x) / lvlw_max
    return {"lvlw_max": lvlw_max, "dynamic_range": dynamic_range, "dynamic_range_dB": 20 * np.log10(dynamic_range), "ENOB": np.log2(dynamic_range)}


def find_maximum_dynamic_range_from_crossings(width_axis, crossing_P_VW, amplitudes_V, range_min_x, range_max_x, coverage=0.997):
    DR_matrix = np.full((len(range_min_x), len(range_max_x)), np.nan, dtype=float)
    ENOB_matrix = np.full_like(DR_matrix, np.nan)
    for i, min_x in enumerate(range_min_x):
        for j, max_x in enumerate(range_max_x):
            if max_x <= min_x:
                continue
            result = get_crossing_max_lsb_and_range(width_axis, crossing_P_VW, amplitudes_V, coverage, min_x, max_x)
            DR_matrix[i, j] = result["dynamic_range_dB"]
            ENOB_matrix[i, j] = result["ENOB"]
    if not np.any(np.isfinite(DR_matrix)):
        return None, DR_matrix, ENOB_matrix
    max_i, max_j = np.unravel_index(np.nanargmax(DR_matrix), DR_matrix.shape)
    best_min_x = float(range_min_x[max_i])
    best_max_x = float(range_max_x[max_j])
    best = get_crossing_max_lsb_and_range(width_axis, crossing_P_VW, amplitudes_V, coverage, best_min_x, best_max_x)
    return DynamicRangeResult(best["lvlw_max"], best["dynamic_range"], best["dynamic_range_dB"], best["ENOB"], best_min_x, best_max_x), DR_matrix, ENOB_matrix


# Legacy functions retained for older notebooks/tests.
def get_max_lsb_and_range(W, D, m, coverage, min_x, max_x):
    x = np.asarray(m["x"], dtype=float)
    R = get_maximum_lvlw(W, m, coverage)
    weights = D * np.sum(W, axis=2)
    valid = (x[:, None] >= min_x) & (x[:, None] <= max_x) & (weights > 0)
    R_valid = R[valid]
    weights_valid = weights[valid]
    if len(R_valid) == 0 or np.sum(weights_valid) == 0:
        return {"lvlw_max": np.nan, "dynamic_range": np.nan, "dynamic_range_dB": np.nan, "ENOB": np.nan}
    lvlw_max = _weighted_coverage_value(R_valid, weights_valid, coverage)
    dynamic_range = (max_x - min_x) / lvlw_max
    return {"lvlw_max": lvlw_max, "dynamic_range": dynamic_range, "dynamic_range_dB": 20 * np.log10(dynamic_range), "ENOB": np.log2(dynamic_range)}


def find_maximum_dynamic_range(W, D, m, range_min_x, range_max_x, coverage=0.997):
    DR_matrix = np.full((len(range_min_x), len(range_max_x)), np.nan)
    ENOB_matrix = np.full_like(DR_matrix, np.nan)
    for i, min_x in enumerate(range_min_x):
        for j, max_x in enumerate(range_max_x):
            if max_x <= min_x:
                continue
            results = get_max_lsb_and_range(W, D, m, coverage, min_x, max_x)
            DR_matrix[i, j] = results["dynamic_range_dB"]
            ENOB_matrix[i, j] = results["ENOB"]
    max_i, max_j = np.unravel_index(np.nanargmax(DR_matrix), DR_matrix.shape)
    best_min_x = range_min_x[max_i]
    best_max_x = range_max_x[max_j]
    best = get_max_lsb_and_range(W, D, m, coverage, best_min_x, best_max_x)
    return DynamicRangeResult(best["lvlw_max"], best["dynamic_range"], best["dynamic_range_dB"], best["ENOB"], best_min_x, best_max_x), DR_matrix, ENOB_matrix

# =============================================================================
# drift.py
# =============================================================================

@dataclass
class StatisticalDriftResult:
    mode: str
    reference_width_V: float
    mean_error_per_crossing_V: float
    per_crossing_ratio: float
    crossing_rate_Hz: float
    drift_rate_V_s: float
    expected_crossings_n: float | None
    total_drift_V: float | None
    total_ratio: float | None
    same_direction_error_per_crossing_V: float
    reversal_error_per_crossing_V: float
    duration_s: float | None = None


@dataclass
class StatisticalDriftModes:
    reversals_reported: StatisticalDriftResult
    reversals_suppressed: StatisticalDriftResult

    @property
    def reference_width_V(self): return self.reversals_reported.reference_width_V
    @property
    def mean_error_per_crossing_V(self): return self.reversals_reported.mean_error_per_crossing_V
    @property
    def per_crossing_ratio(self): return self.reversals_reported.per_crossing_ratio
    @property
    def expected_crossings_n(self): return self.reversals_reported.expected_crossings_n
    @property
    def total_drift_V(self): return self.reversals_reported.total_drift_V
    @property
    def total_ratio(self): return self.reversals_reported.total_ratio
    @property
    def same_direction_error_per_crossing_V(self): return self.reversals_reported.same_direction_error_per_crossing_V
    @property
    def reversal_error_per_crossing_V(self): return self.reversals_reported.reversal_error_per_crossing_V
    @property
    def drift_rate_V_s(self): return self.reversals_reported.drift_rate_V_s


def _mean_full_level_error(mean_width_V, direction_sum, displacement_sum_V, mass):
    """Mean signed reconstruction error for full-level reported events."""
    if mass <= 0:
        return np.nan
    return float((mean_width_V * direction_sum - displacement_sum_V) / mass)


def _result(mode_name, mean_error, same_error, reversal_error, rate_mode, average_width_V, signal_duration_s):
    denominator = float(average_width_V)
    per_crossing_ratio = mean_error / denominator if denominator and np.isfinite(mean_error) else np.nan
    crossing_rate_Hz = float(rate_mode.mean_rate_Hz)
    drift_rate_V_s = float(mean_error * crossing_rate_Hz) if np.isfinite(mean_error) and np.isfinite(crossing_rate_Hz) else np.nan
    expected_crossings_n = total_drift_V = total_ratio = None
    if signal_duration_s is not None and np.isfinite(signal_duration_s):
        expected_crossings_n = float(crossing_rate_Hz * signal_duration_s)
        total_drift_V = float(drift_rate_V_s * signal_duration_s)
        total_ratio = total_drift_V / denominator if denominator else np.nan
    return StatisticalDriftResult(
        mode=mode_name,
        reference_width_V=denominator,
        mean_error_per_crossing_V=float(mean_error),
        per_crossing_ratio=float(per_crossing_ratio),
        crossing_rate_Hz=crossing_rate_Hz,
        drift_rate_V_s=drift_rate_V_s,
        expected_crossings_n=expected_crossings_n,
        total_drift_V=total_drift_V,
        total_ratio=total_ratio,
        same_direction_error_per_crossing_V=float(same_error),
        reversal_error_per_crossing_V=float(reversal_error),
        duration_s=None if signal_duration_s is None else float(signal_duration_s),
    )


def compute_statistical_drift(local_results, crossing_rates, average_width_V, reference_width_V=None, signal_duration_s=None):
    """Expected signed drift using the Pass-12 reconstruction definition.

    The statistical reconstruction step is the D2 crossing-conditioned average
    same-direction level width.  For a full-level event the signed increment is

        direction * (mean_width - actual_width).

    A *reported* reversal/recapture ideally has zero amplitude displacement, so
    its complete signed recapture displacement is reconstruction error:

        reversal_error = -E[Delta V_recapture].

    In full-level-hysteresis mode reversals are internal transitions.  Their
    displacement and elapsed time remain in the Markov path, and drift is
    evaluated only when the next full-level event is reported.  This avoids
    counting a suppressed reversal as a separate output event.

    ``reference_width_V`` is retained for API compatibility but intentionally
    does not control drift in Pass 12.
    """
    reported_model = local_results.mode_reversals_reported
    suppressed_model = local_results.mode_reversals_suppressed
    if reported_model is None or suppressed_model is None:
        raise RuntimeError("Drift requires both reversal-policy transition models.")

    mean_width_V = float(average_width_V)

    # Same-direction crossings use the successful crossing-conditioned width
    # moments. Direction is retained explicitly, so opposite directions cancel
    # only when their width biases truly balance.
    same_error = _mean_full_level_error(
        mean_width_V,
        reported_model.same_output_direction_sum,
        reported_model.same_actual_displacement_sum_V,
        reported_model.same_direction_mass,
    )

    # For a reported recapture the ideal displacement is zero.  The modeled
    # recapture PMF is signed, so its mean already contains the crossing
    # direction and any amplitude asymmetry.
    reversal_mean_displacement = float(local_results.reversal_recapture_statistics.mean)
    reversal_error = -reversal_mean_displacement if np.isfinite(reversal_mean_displacement) else np.nan

    same_fraction = float(crossing_rates.same_direction_fraction)
    reversal_fraction = float(crossing_rates.reversal_fraction)
    reported_mean = 0.0
    valid_reported = False
    if same_fraction > 0:
        if not np.isfinite(same_error):
            reported_mean = np.nan
        else:
            reported_mean += same_fraction * same_error
            valid_reported = True
    if reversal_fraction > 0:
        if not np.isfinite(reversal_error):
            reported_mean = np.nan
        elif np.isfinite(reported_mean):
            reported_mean += reversal_fraction * reversal_error
            valid_reported = True
    if not valid_reported and np.isfinite(reported_mean):
        reported_mean = np.nan

    # With full-level hysteresis the reversal is not a reported event.  The
    # suppressed Markov model has already merged the reversal path into the next
    # full-level event, so use its complete signed displacement moments.
    suppressed_mean = _mean_full_level_error(
        mean_width_V,
        suppressed_model.output_direction_sum,
        suppressed_model.actual_displacement_sum_V,
        suppressed_model.successful_transition_mass,
    )

    reported = _result(
        "reversals_reported",
        reported_mean,
        same_error,
        reversal_error,
        crossing_rates.reversals_reported,
        mean_width_V,
        signal_duration_s,
    )
    suppressed = _result(
        "reversals_suppressed",
        suppressed_mean,
        same_error,
        np.nan,
        crossing_rates.reversals_suppressed,
        mean_width_V,
        signal_duration_s,
    )
    return StatisticalDriftModes(reversals_reported=reported, reversals_suppressed=suppressed)

# =============================================================================
# distortion.py
# =============================================================================

@dataclass
class LocalDistortionResults:
    DW: np.ndarray
    P_DW_V: np.ndarray
    P_DW_same_V: np.ndarray
    statistics: DistributionStatistics
    same_direction_statistics: DistributionStatistics
    same_width_distribution_VW: np.ndarray
    same_width_statistics: DistributionStatistics
    reversal_error_V: np.ndarray
    P_reversal_error_V: np.ndarray
    reversal_recapture_statistics: DistributionStatistics
    J_all: np.ndarray | None
    cross_map_all: np.ndarray | None
    crossing_mass_V: np.ndarray
    same_direction_mass_V: np.ndarray
    reversal_mass_V: np.ndarray
    domain_escape_mass_V: np.ndarray
    stuck_mass_V: np.ndarray
    conservation_error_V: np.ndarray
    successful_transition_mass: float
    directional_crossing_mass: float
    direction_change_mass: float
    candidate_intensity_Hz: float
    signal_duration_s: float
    same_direction_mass: float
    reversal_mass: float
    time_sum_s: float
    time2_sum_s2: float
    rate_sum_Hz: float
    rate2_sum_Hz2: float
    min_rate_Hz: float
    max_rate_Hz: float
    crossing_time_s: np.ndarray
    crossing_time_PMF: np.ndarray
    output_direction_sum: float
    actual_displacement_sum_V: float
    same_output_direction_sum: float
    same_actual_displacement_sum_V: float
    reversal_output_direction_sum: float
    reversal_actual_displacement_sum_V: float
    mode_reversals_reported: object | None = None
    mode_reversals_suppressed: object | None = None

    # Backward-compatible names. Reversal Delta-w is intentionally no longer defined.
    @property
    def P_DW_reversal_V(self):
        return np.zeros_like(self.P_DW_same_V)

    @property
    def reversal_statistics(self):
        return self.reversal_recapture_statistics


@dataclass
class DistortionSummary:
    global_ratio_occupancy: float
    global_ratio_crossings: float
    local_sigma_ratio: float
    local_rms_ratio: float
    reversal_recapture_sigma_ratio: float
    reversal_recapture_rms_ratio: float


def _get_hs_linear_bins(values, hs):
    NHS = len(hs)
    values = np.clip(np.asarray(values), hs[0], hs[-1])
    if hs[-1] == hs[0]:
        return np.zeros_like(values, dtype=int), np.zeros_like(values, dtype=int), np.zeros_like(values, dtype=float)
    position = (values - hs[0]) * (NHS - 1) / (hs[-1] - hs[0])
    index0 = np.floor(position).astype(int)
    index1 = np.minimum(index0 + 1, NHS - 1)
    alpha = position - index0
    return index0, index1, alpha


def _kinematics_in_bin(Pw, j, vpp, state, hs):
    vp = state.DVs[j]
    t_exit, j_next = get_exit_time(j, vpp, state.DVs, state.DV_edges)
    remaining = state.w[None, :] - hs[:, None]
    tx = get_cross_time_matrix(remaining, vp, vpp)
    crossing_mask = np.isfinite(tx) & ((tx <= t_exit) if np.isfinite(t_exit) else True)
    return tx, crossing_mask, t_exit, j_next


def _advance_non_crossing_mass(H, Pw, j, vpp, state, hs, crossing_mask, t_exit, j_next):
    H_next = np.zeros_like(H)
    if not np.any(H):
        return H_next, np.zeros(H.shape[0]), np.zeros(H.shape[0]), 0.0
    if np.sum(Pw) <= 0:
        return H_next, np.zeros(H.shape[0]), np.sum(H, axis=1), 0.0

    exit_mask = ~crossing_mask
    exit_probability_hs = exit_mask @ Pw

    if not np.isfinite(t_exit):
        return H_next, np.zeros(H.shape[0]), H @ exit_probability_hs, 0.0

    if j_next < 0:
        return H_next, H @ exit_probability_hs, np.zeros(H.shape[0]), 0.0

    vp = state.DVs[j]
    travelled_signed = float(vp * t_exit + 0.5 * vpp * t_exit**2)
    travelled = abs(travelled_signed)
    current_sign = np.sign(vp)
    next_sign = np.sign(state.DVs[j_next])

    if current_sign == next_sign or current_sign == 0 or next_sign == 0:
        hs_new = hs + travelled
        index0, index1, alpha = _get_hs_linear_bins(hs_new, hs)
        for ih in np.flatnonzero(exit_probability_hs > 0):
            source_mass = H[:, ih] * exit_probability_hs[ih]
            H_next[:, index0[ih]] += source_mass * (1.0 - alpha[ih])
            H_next[:, index1[ih]] += source_mass * alpha[ih]
        return H_next, np.zeros(H.shape[0]), np.zeros(H.shape[0]), travelled_signed

    # On a derivative-sign reversal the opposite threshold is the threshold that
    # was just crossed. If x has been travelled away from it, only x remains to
    # return. Expressed with the same headstart convention this is hs_new=w-x.
    NHS = len(hs)
    transition_matrix = np.zeros((NHS, NHS), dtype=np.float64)
    probability_exit_hw = exit_mask * Pw[None, :]
    hs_new_hw = state.w[None, :] - (hs[:, None] + travelled)
    ih_grid = np.broadcast_to(np.arange(NHS)[:, None], probability_exit_hw.shape)
    valid = probability_exit_hw > 0
    if np.any(valid):
        ih_flat = ih_grid[valid]
        hs_new_flat = hs_new_hw[valid]
        probability_flat = probability_exit_hw[valid]
        index0, index1, alpha = _get_hs_linear_bins(hs_new_flat, hs)
        np.add.at(transition_matrix, (ih_flat, index0), probability_flat * (1.0 - alpha))
        np.add.at(transition_matrix, (ih_flat, index1), probability_flat * alpha)
    H_next = H @ transition_matrix
    return H_next, np.zeros(H.shape[0]), np.zeros(H.shape[0]), travelled_signed


def _accumulate_histogram(hist, edges, values, weights):
    values = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if np.any(valid):
        hist += np.histogram(values[valid], bins=edges, weights=weights[valid])[0]


def _difference_pmf_from_joint(joint):
    LW = joint.shape[0]
    out = np.zeros(2 * LW - 1, dtype=float)
    for delta_index in range(-(LW - 1), LW):
        out[delta_index + LW - 1] = np.sum(np.diag(joint, k=delta_index))
    return out


def _print_progress(stage, run, total, ii, state, kk=None, detail=""):
    percentage = 100 * run / total if total > 0 else 100.0
    v_text = "V %3d/%3d = %+12.5e" % (ii + 1, state.NVS, state.Vs[ii])
    vpp_text = "V'' ---/--- = ------------" if kk is None else "V'' %3d/%3d = %+12.5e" % (kk + 1, state.ND2VS, state.D2Vs[kk])
    line = "%-31s | %5d/%5d | %6.2f%% | %-27s | %-29s | %-48s" % (stage, run, total, percentage, v_text, vpp_text, detail)
    print("\r" + line.ljust(PROGRESS_WIDTH), end="", flush=True)


@timed("compute_local_distortion")
def compute_local_distortion(state, source_results, progress=True, keep_debug=False, headstart_bins_n=None, reversal_error_bins_n=127, crossing_time_bins_n=128):
    """Unified next-crossing transition propagation.

    The same propagation now produces:
      * next-crossing time -> min/mean/sigma/max crossing rate,
      * same-direction crossing displacement -> level width,
      * same-direction consecutive width difference -> local distortion,
      * reversal signed displacement -> amplitude recapture error (ideal 0 V),
      * signed output-step and actual displacement sums -> drift.

    A reversal is deliberately NOT assigned a level width or Delta-w.
    """
    if state.LW < 2:
        raise RuntimeError("At least two w bins are required.")

    dw_step = float(np.mean(np.diff(state.w)))
    if not np.allclose(np.diff(state.w), dw_step, rtol=1e-6, atol=max(1e-15, abs(dw_step) * 1e-9)):
        raise ValueError("The Delta-w calculation assumes an evenly spaced w axis.")

    DW = np.arange(-(state.LW - 1), state.LW) * dw_step
    NDW = len(DW)
    NHS = state.LW if headstart_bins_n is None else int(headstart_bins_n)
    if NHS < 2:
        raise ValueError("headstart_bins_n must be >= 2")
    hs = np.linspace(0.0, float(np.max(state.w)), NHS)

    width_edges = centers_to_edges(state.w)
    width_edges[0] = np.nextafter(width_edges[0], -np.inf)
    width_edges[-1] = np.nextafter(width_edges[-1], np.inf)

    reversal_limit = max(2.0 * float(np.max(state.w)), 1e-12)
    reversal_edges = np.linspace(-reversal_limit, reversal_limit, int(reversal_error_bins_n) + 1)
    reversal_axis = 0.5 * (reversal_edges[:-1] + reversal_edges[1:])

    duration_s = float((getattr(state.signal_statistics, "metadata", None) or {}).get("duration_s", 1.0))
    dt_min = max(1e-12, float(np.min(state.w[state.w > 0])) / max(float(np.max(np.abs(state.DVs))), 1e-12) / 100.0)
    dt_max = max(duration_s, dt_min * 10.0)
    time_edges = np.geomspace(dt_min, dt_max, int(crossing_time_bins_n) + 1)
    time_axis = np.sqrt(time_edges[:-1] * time_edges[1:])
    time_hist = np.zeros(len(time_axis), dtype=float)

    J_all = np.zeros((state.NVS, state.NDVS, state.NDVS), dtype=np.float32) if keep_debug else None
    cross_map_all = np.zeros((state.NVS, state.ND2VS, state.NDVS), dtype=np.float32) if keep_debug else None
    crossing_mass_V = np.zeros(state.NVS, dtype=float)
    same_direction_mass_V = np.zeros(state.NVS, dtype=float)
    reversal_mass_V = np.zeros(state.NVS, dtype=float)
    domain_escape_mass_V = np.zeros(state.NVS, dtype=float)
    stuck_mass_V = np.zeros(state.NVS, dtype=float)
    conservation_error_V = np.zeros(state.NVS, dtype=float)

    P_same_width_V = np.zeros((state.NVS, state.LW), dtype=float)
    P_DW_same_V = np.zeros((state.NVS, NDW), dtype=float)
    P_reversal_error_V = np.zeros((state.NVS, len(reversal_axis)), dtype=float)

    time_sum_s = 0.0
    time2_sum_s2 = 0.0
    rate_sum_Hz = 0.0
    rate2_sum_Hz2 = 0.0
    min_rate_Hz = np.inf
    max_rate_Hz = 0.0
    output_direction_sum = 0.0
    actual_displacement_sum_V = 0.0
    same_output_direction_sum = 0.0
    same_actual_displacement_sum_V = 0.0
    reversal_output_direction_sum = 0.0
    reversal_actual_displacement_sum_V = 0.0

    active_VK = np.argwhere(np.sum(source_results.cross_source_PMF, axis=1) > 0)
    total_propagations = len(active_VK)
    propagation_run = 0

    for ii in range(state.NVS):
        J = np.zeros((state.NDVS, state.NDVS), dtype=float)
        crossing_map = np.zeros((state.ND2VS, state.NDVS), dtype=float)
        next_same_source_DVW = np.zeros((state.NDVS, state.LW), dtype=float)
        domain_escape_mass = 0.0
        stuck_mass = 0.0
        active_k = np.flatnonzero(np.sum(source_results.cross_source_PMF[ii], axis=0) > 0)

        for kk in active_k:
            propagation_run += 1
            vpp = state.D2Vs[kk]
            source_probability = source_results.cross_source_PMF[ii, :, kk]
            active_sources = np.flatnonzero(source_probability > 0)
            if len(active_sources) == 0:
                continue

            if progress:
                _print_progress("Transition propagation", propagation_run, total_propagations, ii, state, kk, "active V'i = %3d" % len(active_sources))

            source_to_row = np.full(state.NDVS, -1, dtype=int)
            source_to_row[active_sources] = np.arange(len(active_sources))
            H = np.zeros((len(active_sources), NHS), dtype=float)
            elapsed_s = np.zeros(len(active_sources), dtype=float)
            displacement_V = np.zeros(len(active_sources), dtype=float)

            derivative_order = range(active_sources[0], state.NDVS) if vpp >= 0 else range(active_sources[-1], -1, -1)

            for j in derivative_order:
                source_row = source_to_row[j]
                if source_row >= 0:
                    H[source_row, 0] += source_probability[j]
                if not np.any(H):
                    continue

                Pw = state.W_VDV[ii, j]
                if np.sum(Pw) <= 0:
                    stuck_mass += float(np.sum(H))
                    H[:] = 0.0
                    continue

                tx, crossing_mask, t_exit, j_next = _kinematics_in_bin(Pw, j, vpp, state, hs)
                crossing_weight_hw = crossing_mask * Pw[None, :]
                crossing_mass_rhw = H[:, :, None] * crossing_weight_hw[None, :, :]
                crossing_source = np.sum(crossing_mass_rhw, axis=(1, 2))
                J[active_sources, j] += crossing_source
                crossing_map[kk, j] += float(np.sum(crossing_source))

                if np.any(crossing_mass_rhw > 0):
                    tx_valid = np.where(crossing_mask, tx, 0.0)
                    local_disp_hw = state.DVs[j] * tx_valid + 0.5 * vpp * tx_valid**2
                    total_time = elapsed_s[:, None, None] + tx_valid[None, :, :]
                    total_disp = displacement_V[:, None, None] + local_disp_hw[None, :, :]
                    weights = crossing_mass_rhw
                    valid = (weights > 0) & (total_time > 0) & np.isfinite(total_time)

                    if np.any(valid):
                        vv = weights[valid]
                        tt = total_time[valid]
                        rr = 1.0 / tt
                        time_sum_s += float(np.sum(vv * tt))
                        time2_sum_s2 += float(np.sum(vv * tt**2))
                        rate_sum_Hz += float(np.sum(vv * rr))
                        rate2_sum_Hz2 += float(np.sum(vv * rr**2))
                        min_rate_Hz = min(min_rate_Hz, float(np.min(rr)))
                        max_rate_Hz = max(max_rate_Hz, float(np.max(rr)))
                        _accumulate_histogram(time_hist, time_edges, tt, vv)

                    final_sign = float(np.sign(state.DVs[j]))
                    source_signs = np.sign(state.DVs[active_sources])
                    same_rows = np.flatnonzero(source_signs * final_sign > 0)
                    reversal_rows = np.flatnonzero(source_signs * final_sign < 0)

                    for row in same_rows:
                        row_weights = weights[row]
                        row_disp = total_disp[row]
                        row_mass = float(np.sum(row_weights))
                        if row_mass <= 0:
                            continue
                        same_direction_mass_V[ii] += row_mass
                        width_hist = np.zeros(state.LW, dtype=float)
                        _accumulate_histogram(width_hist, width_edges, np.abs(row_disp), row_weights)
                        P_same_width_V[ii] += width_hist
                        next_same_source_DVW[active_sources[row]] += width_hist
                        same_output_direction_sum += final_sign * row_mass
                        same_actual_displacement_sum_V += float(np.sum(row_weights * row_disp))

                    for row in reversal_rows:
                        row_weights = weights[row]
                        row_disp = total_disp[row]
                        row_mass = float(np.sum(row_weights))
                        if row_mass <= 0:
                            continue
                        reversal_mass_V[ii] += row_mass
                        _accumulate_histogram(P_reversal_error_V[ii], reversal_edges, row_disp, row_weights)
                        reversal_output_direction_sum += final_sign * row_mass
                        reversal_actual_displacement_sum_V += float(np.sum(row_weights * row_disp))

                    total_mass = float(np.sum(weights))
                    output_direction_sum += final_sign * total_mass
                    actual_displacement_sum_V += float(np.sum(weights * total_disp))

                H_next, domain_escape_source, stuck_source, travelled_signed = _advance_non_crossing_mass(H, Pw, j, vpp, state, hs, crossing_mask, t_exit, j_next)
                domain_escape_mass += float(np.sum(domain_escape_source))
                stuck_mass += float(np.sum(stuck_source))

                continuing = np.sum(H_next, axis=1) > 0
                if np.isfinite(t_exit) and np.any(continuing):
                    elapsed_s[continuing] += t_exit
                    displacement_V[continuing] += travelled_signed
                H = H_next

        source_mass = float(np.sum(source_results.cross_source_PMF[ii]))
        crossing_mass = float(np.sum(J))
        crossing_mass_V[ii] = crossing_mass
        domain_escape_mass_V[ii] = domain_escape_mass
        stuck_mass_V[ii] = stuck_mass
        conservation_error_V[ii] = crossing_mass + domain_escape_mass + stuck_mass - source_mass
        if keep_debug:
            J_all[ii] = J.astype(np.float32)
            cross_map_all[ii] = crossing_map.astype(np.float32)

        # Same-direction local Delta-w: previous source width versus the actual
        # next same-direction displacement generated by the transition path.
        for j in range(state.NDVS):
            next_width = next_same_source_DVW[j]
            same_mass_j = float(np.sum(next_width))
            if same_mass_j <= 0:
                continue
            source_width = source_results.source_width_PMF_VDVW[ii, j]
            source_width_mass = float(np.sum(source_width))
            if source_width_mass <= 0:
                continue
            source_cond = source_width / source_width_mass
            next_cond = next_width / same_mass_j
            joint = same_mass_j * source_cond[:, None] * next_cond[None, :]
            P_DW_same_V[ii] += _difference_pmf_from_joint(joint)

    if progress:
        print("\r" + "Transition, timing and distortion computation complete.".ljust(PROGRESS_WIDTH))

    successful_transition_mass = float(np.sum(crossing_mass_V))
    same_direction_mass = float(np.sum(same_direction_mass_V))
    reversal_mass = float(np.sum(reversal_mass_V))

    if not np.isfinite(min_rate_Hz):
        min_rate_Hz = 0.0

    P_same_width = np.sum(P_same_width_V, axis=0)
    P_DW_same = np.sum(P_DW_same_V, axis=0)
    P_reversal_error = np.sum(P_reversal_error_V, axis=0)
    time_total = float(np.sum(time_hist))
    crossing_time_PMF = time_hist / time_total if time_total > 0 else time_hist

    same_width_statistics = get_distribution_statistics(state.w, P_same_width)
    same_dw_statistics = get_distribution_statistics(DW, P_DW_same)
    reversal_statistics = get_distribution_statistics(reversal_axis, P_reversal_error)

    return LocalDistortionResults(
        DW=DW,
        P_DW_V=P_DW_same_V,
        P_DW_same_V=P_DW_same_V,
        statistics=same_dw_statistics,
        same_direction_statistics=same_dw_statistics,
        same_width_distribution_VW=P_same_width_V,
        same_width_statistics=same_width_statistics,
        reversal_error_V=reversal_axis,
        P_reversal_error_V=P_reversal_error_V,
        reversal_recapture_statistics=reversal_statistics,
        J_all=J_all,
        cross_map_all=cross_map_all,
        crossing_mass_V=crossing_mass_V,
        same_direction_mass_V=same_direction_mass_V,
        reversal_mass_V=reversal_mass_V,
        domain_escape_mass_V=domain_escape_mass_V,
        stuck_mass_V=stuck_mass_V,
        conservation_error_V=conservation_error_V,
        successful_transition_mass=successful_transition_mass,
        same_direction_mass=same_direction_mass,
        reversal_mass=reversal_mass,
        time_sum_s=time_sum_s,
        time2_sum_s2=time2_sum_s2,
        rate_sum_Hz=rate_sum_Hz,
        rate2_sum_Hz2=rate2_sum_Hz2,
        min_rate_Hz=min_rate_Hz,
        max_rate_Hz=max_rate_Hz,
        crossing_time_s=time_axis,
        crossing_time_PMF=crossing_time_PMF,
        output_direction_sum=output_direction_sum,
        actual_displacement_sum_V=actual_displacement_sum_V,
        same_output_direction_sum=same_output_direction_sum,
        same_actual_displacement_sum_V=same_actual_displacement_sum_V,
        reversal_output_direction_sum=reversal_output_direction_sum,
        reversal_actual_displacement_sum_V=reversal_actual_displacement_sum_V,
    )


def summarize_distortion(level_width_results, local_results, reference_width_V=None):
    w_occ = level_width_results.occupancy_statistics
    w_same = level_width_results.crossing_statistics
    dw = local_results.same_direction_statistics
    rev = local_results.reversal_recapture_statistics
    denom = w_same.mean
    ref = denom if reference_width_V is None else float(reference_width_V)

    return DistortionSummary(
        global_ratio_occupancy=w_occ.sigma / w_occ.mean if w_occ.mean else np.nan,
        global_ratio_crossings=w_same.sigma / denom if denom else np.nan,
        local_sigma_ratio=dw.sigma / denom if denom else np.nan,
        local_rms_ratio=np.sqrt(dw.mean**2 + dw.sigma**2) / denom if denom else np.nan,
        reversal_recapture_sigma_ratio=rev.sigma / ref if ref else np.nan,
        reversal_recapture_rms_ratio=np.sqrt(rev.mean**2 + rev.sigma**2) / ref if ref else np.nan,
    )


# =============================================================================
# Pass 7 histogram-only probabilistic transition model
# =============================================================================

def compute_local_distortion(state, source_results, progress=True, keep_debug=False, headstart_bins_n=None, reversal_error_bins_n=127, crossing_time_bins_n=128, transition_probability_floor=1e-8, max_state_transitions_n=64, reference_width_V=None):
    """Compute transition statistics from D2 and W only.

    Two output policies are propagated from the same histogram state model:

    - reversals reported: recapture of the previous crossing amplitude is an event;
    - reversals suppressed: recapture is internal and propagation continues until
      the next full-level event.

    Width, Delta-w and reversal-recapture distributions use the reported-reversal
    model. Crossing-rate tails and drift are retained for both policies.
    """

    if progress:
        print("Statistical transition model: reversals reported")
    reported = solve_histogram_event_model(
        state,
        headstart_bins_n=headstart_bins_n,
        crossing_time_bins_n=crossing_time_bins_n,
        reversal_error_bins_n=reversal_error_bins_n,
        transition_probability_floor=transition_probability_floor,
        max_state_transitions_n=max_state_transitions_n,
        reference_width_V=reference_width_V,
        progress=progress,
        report_reversals=True,
        collect_distributions=True,
        source_event_PMF_VDV=source_results.cross_source_VDV,
        approximate_intensity_Hz=source_results.approximate_intensity_Hz,
    )

    if progress:
        print("Statistical transition model: reversals suppressed (full-level hysteresis)")
    suppressed = solve_histogram_event_model(
        state,
        headstart_bins_n=headstart_bins_n,
        crossing_time_bins_n=crossing_time_bins_n,
        reversal_error_bins_n=reversal_error_bins_n,
        transition_probability_floor=transition_probability_floor,
        max_state_transitions_n=max_state_transitions_n,
        reference_width_V=reference_width_V,
        progress=progress,
        report_reversals=False,
        collect_distributions=False,
        source_event_PMF_VDV=source_results.cross_source_VDV,
        approximate_intensity_Hz=source_results.approximate_intensity_Hz,
    )

    # Same-direction width and Delta-w are evaluated directly from the
    # crossing-conditioned W distribution.  A physical level width belongs to
    # the crossing transition and is not resampled while moving through D2 bins.
    so_source = compute_second_order_source(state)
    so_width = compute_second_order_width_delta(state, so_source)
    reported.same_width_distribution_VW = so_width["same_width_distribution_VW"]
    reported.DW = so_width["DW"]
    reported.P_DW_same_V = so_width["P_DW_V"]

    # Keep the Markov reversal paths, but add the independent opposite-threshold
    # crossing-position uncertainty measured during ramp characterization.
    threshold_sigma = float((state.characterization.metadata or {}).get("crossing_error_difference_sigma_V", 0.0) or 0.0)
    if threshold_sigma > 0 and len(reported.reversal_error_V) > 1:
        step = float(np.mean(np.diff(reported.reversal_error_V)))
        sigma_bins = threshold_sigma / max(abs(step), 1e-30)
        radius = max(1, int(np.ceil(4.0 * sigma_bins)))
        xk = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (xk / max(sigma_bins, 1e-12)) ** 2)
        kernel /= np.sum(kernel)
        broadened = np.zeros_like(reported.P_reversal_error_V)
        for iv in range(len(broadened)):
            broadened[iv] = np.convolve(reported.P_reversal_error_V[iv], kernel, mode="same")
        reported.P_reversal_error_V = broadened

    # Correct same-direction signed displacement moments used by drift.  Reversal
    # moments remain those of the probabilistic first-passage model.
    if so_width["total_flux_Hz"] > 0 and reported.same_direction_mass > 0:
        mean_dir = so_width["signed_direction_sum"] / so_width["total_flux_Hz"]
        mean_disp = so_width["signed_displacement_sum_V"] / so_width["total_flux_Hz"]
        reported.same_output_direction_sum = reported.same_direction_mass * mean_dir
        reported.same_actual_displacement_sum_V = reported.same_direction_mass * mean_disp
        reported.output_direction_sum = reported.same_output_direction_sum + reported.reversal_output_direction_sum
        reported.actual_displacement_sum_V = reported.same_actual_displacement_sum_V + reported.reversal_actual_displacement_sum_V

    P_same_width = np.sum(reported.same_width_distribution_VW, axis=0)
    P_DW_same = np.sum(reported.P_DW_same_V, axis=0)
    P_reversal = np.sum(reported.P_reversal_error_V, axis=0)
    same_width_statistics = get_distribution_statistics(state.w, P_same_width)
    same_dw_statistics = get_distribution_statistics(reported.DW, P_DW_same)
    reversal_statistics = get_distribution_statistics(reported.reversal_error_V, P_reversal)

    same_mass_V = np.sum(reported.same_width_distribution_VW, axis=1)
    reversal_mass_V = np.sum(reported.P_reversal_error_V, axis=1)
    crossing_mass_V = same_mass_V + reversal_mass_V
    domain_escape_mass_V = np.zeros(state.NVS, dtype=float)
    stuck_mass_V = np.zeros(state.NVS, dtype=float)
    conservation_error_V = np.zeros(state.NVS, dtype=float)
    if state.NVS:
        domain_escape_mass_V[0] = reported.domain_escape_mass
        stuck_mass_V[0] = reported.stuck_mass
        conservation_error_V[0] = reported.successful_transition_mass + reported.domain_escape_mass + reported.stuck_mass - 1.0

    duration_s = float((state.signal_statistics.metadata or {}).get("duration_s", 0.0))

    return LocalDistortionResults(
        DW=reported.DW,
        P_DW_V=reported.P_DW_same_V,
        P_DW_same_V=reported.P_DW_same_V,
        statistics=same_dw_statistics,
        same_direction_statistics=same_dw_statistics,
        same_width_distribution_VW=reported.same_width_distribution_VW,
        same_width_statistics=same_width_statistics,
        reversal_error_V=reported.reversal_error_V,
        P_reversal_error_V=reported.P_reversal_error_V,
        reversal_recapture_statistics=reversal_statistics,
        J_all=None,
        cross_map_all=None,
        crossing_mass_V=crossing_mass_V,
        same_direction_mass_V=same_mass_V,
        reversal_mass_V=reversal_mass_V,
        domain_escape_mass_V=domain_escape_mass_V,
        stuck_mass_V=stuck_mass_V,
        conservation_error_V=conservation_error_V,
        successful_transition_mass=float(reported.successful_transition_mass),
        directional_crossing_mass=float(reported.directional_crossing_mass),
        direction_change_mass=float(reported.direction_change_mass),
        candidate_intensity_Hz=float(reported.approximate_intensity_Hz),
        signal_duration_s=duration_s,
        same_direction_mass=float(reported.same_direction_mass),
        reversal_mass=float(reported.reversal_mass),
        time_sum_s=float(reported.time_sum_s),
        time2_sum_s2=float(reported.time2_sum_s2),
        rate_sum_Hz=float(reported.rate_sum_Hz),
        rate2_sum_Hz2=float(reported.rate2_sum_Hz2),
        min_rate_Hz=float(reported.min_rate_Hz),
        max_rate_Hz=float(reported.max_rate_Hz),
        crossing_time_s=reported.crossing_time_s,
        crossing_time_PMF=reported.crossing_time_PMF,
        output_direction_sum=float(reported.output_direction_sum),
        actual_displacement_sum_V=float(reported.actual_displacement_sum_V),
        same_output_direction_sum=float(reported.same_output_direction_sum),
        same_actual_displacement_sum_V=float(reported.same_actual_displacement_sum_V),
        reversal_output_direction_sum=float(reported.reversal_output_direction_sum),
        reversal_actual_displacement_sum_V=float(reported.reversal_actual_displacement_sum_V),
        mode_reversals_reported=reported,
        mode_reversals_suppressed=suppressed,
    )

# =============================================================================
# characterization_only.py
# =============================================================================

@dataclass
class CharacterizationOnlyMetrics:
    width_statistics: object
    global_distortion_ratio: float
    dynamic_range: object | None
    drift: object

    def summary(self):
        return {
            "average_level_width_V": float(self.width_statistics.mean),
            "sigma_level_width_V": float(self.width_statistics.sigma),
            "global_distortion_ratio": float(self.global_distortion_ratio),
            "drift_per_crossing_V": float(self.drift.mean_error_per_crossing_V),
            "drift_per_crossing_ratio": float(self.drift.per_crossing_ratio),
        }


@timed("compute_characterization_only_metrics")
def compute_characterization_only_metrics(characterization, range_min_x=None, range_max_x=None, coverage=0.997):
    """ADC-only W summary with equal weight for each characterized (V,V') state."""
    W = np.asarray(characterization.W, dtype=float)
    w = np.asarray(characterization.w, dtype=float)
    x = np.asarray(characterization.x, dtype=float)
    W = np.where(np.isfinite(W), W, 0.0)
    totals = np.sum(W, axis=2, keepdims=True)
    Wn = np.divide(W, totals, out=np.zeros_like(W), where=totals > 0)
    valid_state = totals[:, :, 0] > 0
    weights = valid_state.astype(float)
    weights /= np.sum(weights) if np.sum(weights) > 0 else 1.0
    P_W = np.sum(weights[:, :, None] * Wn, axis=(0, 1))
    stats = get_distribution_statistics(w, P_W)
    global_ratio = stats.sigma / stats.mean if stats.mean else np.nan

    dynamic_range = None
    if range_min_x is not None and range_max_x is not None:
        # Equal V' weighting within each V, no signal occupancy.
        P_VW = np.sum(Wn, axis=1)
        dynamic_range, _, _ = find_maximum_dynamic_range_from_crossings(w, P_VW, x, np.asarray(range_min_x), np.asarray(range_max_x), coverage=coverage)

    # W-only drift: use the global mean width as the reconstruction step and
    # compare it with the mean width of every characterized (V,V') state.  The
    # derivative sign converts width mismatch into signed reconstruction error.
    y = np.asarray(characterization.y, dtype=float)
    mean_w_state = np.sum(Wn * w[None, None, :], axis=2)
    state_weights = valid_state.astype(float)
    state_weights /= np.sum(state_weights) if np.sum(state_weights) > 0 else 1.0
    signed_error_state = np.sign(y)[None, :] * (stats.mean - mean_w_state)
    mean_error = float(np.sum(state_weights * signed_error_state))
    pos_mask = valid_state & (y[None, :] > 0)
    neg_mask = valid_state & (y[None, :] < 0)
    mean_pos = float(np.mean(mean_w_state[pos_mask])) if np.any(pos_mask) else np.nan
    mean_neg = float(np.mean(mean_w_state[neg_mask])) if np.any(neg_mask) else np.nan
    drift = SimpleNamespace(
        reference_width_V=float(stats.mean),
        mean_error_per_crossing_V=mean_error,
        per_crossing_drift_V=mean_error,
        per_crossing_ratio=(mean_error / stats.mean if stats.mean else np.nan),
        mean_width_positive_V=mean_pos,
        mean_width_negative_V=mean_neg,
        drift_rate_V_s=np.nan,
        total_drift_V=None,
    )
    return CharacterizationOnlyMetrics(stats, global_ratio, dynamic_range, drift)


# =============================================================================
# zeroth_order.py
# =============================================================================

@dataclass
class ZerothOrderMetrics:
    D0: np.ndarray
    x: np.ndarray
    width_distribution_VW: np.ndarray
    width_statistics: object
    global_distortion_ratio: float
    dynamic_range: object | None

    def summary(self):
        return {
            "average_level_width_V": float(self.width_statistics.mean),
            "sigma_level_width_V": float(self.width_statistics.sigma),
            "global_distortion_ratio": float(self.global_distortion_ratio),
        }


@timed("compute_zeroth_order_metrics")
def compute_zeroth_order_metrics(characterization, signal_statistics, range_min_x=None, range_max_x=None, coverage=0.997):
    """D0(V) model: amplitude occupancy only.

    V' is unknown, so the ADC width PMF is averaged equally over all valid
    characterized derivative states at each amplitude. Crossing rate, local
    distortion, reversals and signed drift are intentionally undefined.
    """
    D0 = np.sum(np.asarray(signal_statistics.D2, dtype=float), axis=(1, 2))
    if np.sum(D0) <= 0:
        raise RuntimeError("D0 contains no probability mass")
    D0 = D0 / np.sum(D0)
    x0 = np.asarray(signal_statistics.x, dtype=float)

    W = np.asarray(characterization.W, dtype=float)
    w = np.asarray(characterization.w, dtype=float)
    xc = np.asarray(characterization.x, dtype=float)
    totals = np.sum(W, axis=2, keepdims=True)
    Wn = np.divide(W, totals, out=np.zeros_like(W), where=totals > 0)
    valid = totals[:, :, 0] > 0
    count_v = np.sum(valid, axis=1)
    Wavg_c = np.divide(np.sum(Wn, axis=1), count_v[:, None], out=np.zeros((len(xc), len(w))), where=count_v[:, None] > 0)

    Wavg = np.zeros((len(x0), len(w)), dtype=float)
    for iw in range(len(w)):
        Wavg[:, iw] = np.interp(x0, xc, Wavg_c[:, iw], left=Wavg_c[0, iw], right=Wavg_c[-1, iw])
    row_sum = np.sum(Wavg, axis=1, keepdims=True)
    Wavg = np.divide(Wavg, row_sum, out=np.zeros_like(Wavg), where=row_sum > 0)

    P_VW = D0[:, None] * Wavg
    stats = get_distribution_statistics(w, np.sum(P_VW, axis=0))
    global_ratio = stats.sigma / stats.mean if stats.mean else np.nan

    dynamic_range = None
    if range_min_x is not None and range_max_x is not None:
        dynamic_range, _, _ = find_maximum_dynamic_range_from_crossings(w, P_VW, x0, np.asarray(range_min_x), np.asarray(range_max_x), coverage=coverage)

    return ZerothOrderMetrics(D0=D0, x=x0, width_distribution_VW=P_VW, width_statistics=stats, global_distortion_ratio=global_ratio, dynamic_range=dynamic_range)

# =============================================================================
# first_order.py
# =============================================================================

def _weighted_quantile(values, weights, q):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    values, weights = values[valid], weights[valid]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    return float(values[min(np.searchsorted(cdf, q), len(values) - 1)])


def _difference_pmf(w, state_weights, W_current, W_next):
    LW = len(w)
    DW = np.arange(-(LW - 1), LW) * (w[1] - w[0]) if LW > 1 else np.array([0.0])
    P = np.zeros(2 * LW - 1, dtype=float)
    for weight, p0, p1 in zip(state_weights, W_current, W_next):
        if weight <= 0 or np.sum(p0) <= 0 or np.sum(p1) <= 0:
            continue
        joint = weight * p0[:, None] * p1[None, :]
        for k in range(-(LW - 1), LW):
            P[k + LW - 1] += np.sum(np.diag(joint, k=k))
    return DW, P


@dataclass
class FirstOrderMetrics:
    state: object
    D1: np.ndarray
    event_distribution_VDVW: np.ndarray
    event_distribution_VW: np.ndarray
    width_statistics: object
    delta_w_axis_V: np.ndarray
    delta_w_statistics: object
    global_distortion_ratio: float
    local_distortion_ratio: float
    crossing_rate: CrossingModeRateResults
    same_direction_fraction: float
    reversal_fraction: float
    same_direction_rate_Hz: float
    reversal_rate_Hz: float
    dynamic_range: object | None
    drift: object
    overload_reached: bool | None
    max_signal_slope_V_s: float | None

    def summary(self):
        return {
            "average_level_width_V": float(self.width_statistics.mean),
            "sigma_level_width_V": float(self.width_statistics.sigma),
            "global_distortion_ratio": float(self.global_distortion_ratio),
            "local_distortion_ratio": float(self.local_distortion_ratio),
            "average_crossing_rate_Hz": float(self.crossing_rate.mean_rate_Hz),
            "sigma_crossing_rate_Hz": float(self.crossing_rate.sigma_rate_Hz),
            "minimum_crossing_rate_Hz": float(self.crossing_rate.min_rate_Hz),
            "maximum_crossing_rate_Hz": float(self.crossing_rate.max_rate_Hz),
            "same_direction_fraction": 1.0,
            "reversal_fraction": 0.0,
            "drift_per_crossing_V": float(self.drift.mean_error_per_crossing_V),
            "drift_rate_V_s": float(self.drift.drift_rate_V_s),
            "total_drift_V": None if self.drift.total_drift_V is None else float(self.drift.total_drift_V),
        }


@timed("compute_first_order_metrics")
def compute_first_order_metrics(characterization, signal_statistics, range_min_x=None, range_max_x=None, coverage=0.997, reference_width_V=None, signal_duration_s=None, overload_slope_V_s=None):
    """D1(V,V') model assuming V' remains constant until the next crossing.

    D1 is obtained directly by marginalizing the already computed D2 over V''.
    No first-passage propagation and no additional signal/ADC simulation is run.
    Under the constant-slope assumption reversals cannot occur.
    """
    state = prepare_statistical_state(characterization, signal_statistics)
    D1 = np.asarray(state.D1s, dtype=float)
    D1_total = float(np.sum(D1))
    if D1_total <= 0:
        raise RuntimeError("D1 contains no probability mass")
    D1 = D1 / D1_total

    if reference_width_V is None:
        metadata = getattr(characterization, "metadata", None) or {}
        reference_width_V = float(metadata.get("nominal_level_width_V", np.mean(state.w)))

    mean_w = np.sum(state.W_VDV * state.w[None, None, :], axis=2)
    local_flux = np.divide(np.abs(state.DVs)[None, :], mean_w, out=np.zeros_like(mean_w), where=mean_w > 0)
    flux_VDV = D1 * local_flux
    total_rate_Hz = float(np.sum(flux_VDV))

    event_raw = flux_VDV[:, :, None] * state.W_VDV
    event_mass = float(np.sum(event_raw))
    event_P = event_raw / event_mass if event_mass > 0 else np.zeros_like(event_raw)
    event_VW = np.sum(event_raw, axis=1)

    P_w = np.sum(event_raw, axis=(0, 1))
    width_stats = get_distribution_statistics(state.w, P_w)
    global_ratio = width_stats.sigma / width_stats.mean if width_stats.mean else np.nan

    # Consecutive width differences: one full level in the sign of constant V'.
    weights = []
    W0 = []
    W1 = []
    for i in range(state.NVS):
        for j, dv in enumerate(state.DVs):
            q = float(flux_VDV[i, j])
            if q <= 0 or dv == 0:
                continue
            i_next = i + (1 if dv > 0 else -1)
            if i_next < 0 or i_next >= state.NVS:
                continue
            weights.append(q)
            W0.append(state.W_VDV[i, j])
            W1.append(state.W_VDV[i_next, j])
    if weights:
        DW, P_DW = _difference_pmf(state.w, np.asarray(weights), np.asarray(W0), np.asarray(W1))
    else:
        DW = np.array([0.0])
        P_DW = np.array([0.0])
    dw_stats = get_distribution_statistics(DW, P_DW)
    local_ratio = dw_stats.sigma / width_stats.mean if width_stats.mean else np.nan

    # Event interval/rate distribution. W is retained here rather than collapsed.
    dv_grid = np.abs(state.DVs)[None, :, None]
    dt = np.divide(state.w[None, None, :], dv_grid, out=np.full_like(event_P, np.nan), where=dv_grid > 0)
    valid = (event_P > 0) & np.isfinite(dt) & (dt > 0)
    rates = np.divide(1.0, dt[valid])
    p = event_P[valid]
    if np.sum(p) > 0:
        p = p / np.sum(p)
        interval_mean_rate = float(np.sum(p * rates))
        sigma_rate = float(np.sqrt(np.sum(p * (rates - interval_mean_rate) ** 2)))
        mean_dt = float(np.sum(p * dt[valid]))
        sigma_dt = float(np.sqrt(np.sum(p * (dt[valid] - mean_dt) ** 2)))
        expected_n = max(total_rate_Hz * float(signal_duration_s or 0.0), 1.0)
        tail_q = min(0.25, 1.0 / (expected_n + 1.0))
        min_rate = _weighted_quantile(rates, p, tail_q)
        max_rate = _weighted_quantile(rates, p, 1.0 - tail_q)
    else:
        interval_mean_rate = sigma_rate = sigma_dt = min_rate = max_rate = np.nan

    rate_mode = CrossingModeRateResults(
        name="D1_constant_slope",
        mean_rate_Hz=total_rate_Hz,
        interval_mean_rate_Hz=interval_mean_rate,
        sigma_rate_Hz=sigma_rate,
        min_rate_Hz=min_rate,
        max_rate_Hz=max_rate,
        mean_interval_s=(1.0 / total_rate_Hz if total_rate_Hz > 0 else np.nan),
        sigma_interval_s=sigma_dt,
        crossing_time_s=dt[valid],
        crossing_time_PMF=p if np.sum(event_P[valid]) > 0 else np.array([]),
        events_n=(total_rate_Hz * signal_duration_s if signal_duration_s is not None else None),
    )

    dynamic_range = None
    if range_min_x is not None and range_max_x is not None:
        dynamic_range, _, _ = find_maximum_dynamic_range_from_crossings(state.w, event_VW, state.Vs, np.asarray(range_min_x), np.asarray(range_max_x), coverage=coverage)

    # Pass-12 drift reference: average level width before event-rate weighting,
    # weighted only by the D1(V,V') signal occupancy.  The event_P distribution
    # below then supplies the crossing-conditioned actual widths.
    reconstruction_width_V = float(np.sum(D1 * mean_w) / np.sum(D1))
    signs = np.sign(state.DVs)[None, :, None]
    error = signs * (reconstruction_width_V - state.w[None, None, :])
    mean_error = float(np.sum(event_P * error)) if event_mass > 0 else np.nan
    denom = reconstruction_width_V
    drift_rate = mean_error * total_rate_Hz if np.isfinite(mean_error) else np.nan
    total_drift = None if signal_duration_s is None else drift_rate * signal_duration_s
    drift = SimpleNamespace(
        reference_width_V=reconstruction_width_V,
        mean_error_per_crossing_V=mean_error,
        per_crossing_drift_V=mean_error,
        per_crossing_ratio=(mean_error / denom if denom else np.nan),
        drift_rate_V_s=drift_rate,
        total_drift_V=total_drift,
        expected_total_drift_V=total_drift,
    )

    max_slope = float(np.max(np.abs(state.DVs[np.sum(D1, axis=0) > 0]))) if np.any(np.sum(D1, axis=0) > 0) else 0.0
    overload = None if overload_slope_V_s is None else bool(max_slope >= overload_slope_V_s)

    return FirstOrderMetrics(
        state=state,
        D1=D1,
        event_distribution_VDVW=event_P,
        event_distribution_VW=event_VW,
        width_statistics=width_stats,
        delta_w_axis_V=DW,
        delta_w_statistics=dw_stats,
        global_distortion_ratio=global_ratio,
        local_distortion_ratio=local_ratio,
        crossing_rate=rate_mode,
        same_direction_fraction=1.0,
        reversal_fraction=0.0,
        same_direction_rate_Hz=total_rate_Hz,
        reversal_rate_Hz=0.0,
        dynamic_range=dynamic_range,
        drift=drift,
        overload_reached=overload,
        max_signal_slope_V_s=max_slope,
    )

# =============================================================================
# overload.py
# =============================================================================

def overload_reached(D2_data, overload_slope_V_s):
    D2 = np.asarray(D2_data["D2"])
    DV = np.asarray(D2_data["y"])
    occupied_DV = np.any(D2 > 0, axis=(0, 2))
    if not np.any(occupied_DV):
        return False, 0.0
    max_abs_slope = np.max(np.abs(DV[occupied_DV]))
    return max_abs_slope >= overload_slope_V_s, max_abs_slope

# =============================================================================
# metrics.py
# =============================================================================

@dataclass
class StatisticalMetrics:
    state: StatisticalState
    crossing_source: CrossingSourceResults
    crossing_rates: CrossingRateResults
    level_width: LevelWidthResults
    local_distortion: LocalDistortionResults
    distortion: DistortionSummary
    dynamic_range: DynamicRangeResult | None
    drift: StatisticalDriftModes
    overload_reached: bool | None
    max_signal_slope_V_s: float | None

    def summary(self):
        return statistical_summary(self)


def _infer_reference_width(characterization, requested_reference_width_V=None):
    if requested_reference_width_V is not None:
        return float(requested_reference_width_V)
    metadata = getattr(characterization, "metadata", None) or {}
    if metadata.get("nominal_level_width_V") is not None:
        return float(metadata["nominal_level_width_V"])
    x = np.asarray(characterization.m["x"], dtype=float)
    if len(x) > 1:
        return float(np.median(np.diff(x)))
    return float(np.asarray(characterization.m["z"], dtype=float).mean())


@timed("compute_statistical_metrics")
def compute_statistical_metrics(characterization_or_W, signal_or_m, D2_data=None, range_min_x=None, range_max_x=None, coverage=0.997, overload_slope_V_s=None, progress=True, keep_debug=False, reference_width_V=None, signal_duration_s=None, headstart_bins_n=None, reversal_error_bins_n=127, crossing_time_bins_n=128, transition_probability_floor=1e-8, max_state_transitions_n=64):
    """Compute metrics from one common next-crossing transition analysis."""
    state = prepare_statistical_state(characterization_or_W, signal_or_m, D2_data)
    reference_width_V = _infer_reference_width(state.characterization, reference_width_V)

    crossing_source = compute_crossing_source(state)
    local_distortion = compute_local_distortion(
        state,
        crossing_source,
        progress=progress,
        keep_debug=keep_debug,
        headstart_bins_n=headstart_bins_n,
        reversal_error_bins_n=reversal_error_bins_n,
        crossing_time_bins_n=crossing_time_bins_n,
        transition_probability_floor=transition_probability_floor,
        max_state_transitions_n=max_state_transitions_n,
        reference_width_V=reference_width_V,
    )
    crossing_rates = compute_crossing_rates_from_transitions(crossing_source, local_distortion.mode_reversals_reported, local_distortion.mode_reversals_suppressed)
    level_width = compute_level_width_results(state, local_distortion)
    distortion = summarize_distortion(level_width, local_distortion, reference_width_V=reference_width_V)

    dynamic_range = None
    if range_min_x is not None and range_max_x is not None:
        dynamic_range, _, _ = find_maximum_dynamic_range_from_crossings(
            state.w,
            local_distortion.same_width_distribution_VW,
            state.Vs,
            np.asarray(range_min_x),
            np.asarray(range_max_x),
            coverage=coverage,
        )

    if signal_duration_s is None:
        signal_duration_s = (getattr(state.signal_statistics, "metadata", None) or {}).get("duration_s")
    mean_w_VDV = np.sum(state.W_VDV * state.w[None, None, :], axis=2)
    D1_occupancy = np.sum(state.D2s, axis=2)
    occupancy_mass = float(np.sum(D1_occupancy))
    drift_reference_width_V = float(np.sum(D1_occupancy * mean_w_VDV) / occupancy_mass) if occupancy_mass > 0 else float(level_width.crossing_statistics.mean)
    drift = compute_statistical_drift(local_distortion, crossing_rates, drift_reference_width_V, reference_width_V, signal_duration_s=signal_duration_s)

    overload_flag = None
    max_signal_slope = None
    if overload_slope_V_s is not None:
        overload_flag, max_signal_slope = overload_reached(state.signal_statistics, overload_slope_V_s)

    return StatisticalMetrics(
        state=state,
        crossing_source=crossing_source,
        crossing_rates=crossing_rates,
        level_width=level_width,
        local_distortion=local_distortion,
        distortion=distortion,
        dynamic_range=dynamic_range,
        drift=drift,
        overload_reached=overload_flag,
        max_signal_slope_V_s=max_signal_slope,
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


def statistical_summary(metrics):
    reported = metrics.crossing_rates.reversals_reported
    suppressed = metrics.crossing_rates.reversals_suppressed
    summary = {
        "average_crossing_rate_Hz": float(reported.mean_rate_Hz),
        "minimum_crossing_rate_Hz": float(reported.min_rate_Hz),
        "sigma_crossing_rate_Hz": float(reported.sigma_rate_Hz),
        "interval_mean_crossing_rate_Hz": float(reported.interval_mean_rate_Hz),
        "maximum_crossing_rate_Hz": float(reported.max_rate_Hz),
        "mean_crossing_interval_s": float(reported.mean_interval_s),
        "sigma_crossing_interval_s": float(reported.sigma_interval_s),
        "same_direction_fraction": float(metrics.crossing_rates.same_direction_fraction),
        "reversal_fraction": float(metrics.crossing_rates.reversal_fraction),
        "same_direction_rate_Hz": float(metrics.crossing_rates.same_direction_rate_Hz),
        "reversal_rate_Hz": float(metrics.crossing_rates.reversal_rate_Hz),
        "source_rate_approximation_Hz": float(metrics.crossing_source.approximate_intensity_Hz),
        "directional_crossing_probability": float(metrics.local_distortion.directional_crossing_mass),
        "transition_success_probability": float(metrics.local_distortion.successful_transition_mass),
        "average_level_width_occupancy_V": float(metrics.level_width.occupancy_statistics.mean),
        "average_level_width_crossings_V": float(metrics.level_width.crossing_statistics.mean),
        "sigma_level_width_crossings_V": float(metrics.level_width.crossing_statistics.sigma),
        "global_distortion_occupancy_ratio": float(metrics.distortion.global_ratio_occupancy),
        "global_distortion_crossings_ratio": float(metrics.distortion.global_ratio_crossings),
        "local_distortion_sigma_ratio": float(metrics.distortion.local_sigma_ratio),
        "local_distortion_rms_ratio": float(metrics.distortion.local_rms_ratio),
        "reversal_recapture_mean_V": float(metrics.local_distortion.reversal_recapture_statistics.mean),
        "reversal_recapture_sigma_V": float(metrics.local_distortion.reversal_recapture_statistics.sigma),
        "reversal_recapture_sigma_ratio": float(metrics.distortion.reversal_recapture_sigma_ratio),
        "reversal_recapture_rms_ratio": float(metrics.distortion.reversal_recapture_rms_ratio),
        "drift_per_crossing_ratio": float(metrics.drift.reversals_reported.per_crossing_ratio),
        "drift_rate_V_s": float(metrics.drift.reversals_reported.drift_rate_V_s),
        "drift_total_ratio": None if metrics.drift.reversals_reported.total_ratio is None else float(metrics.drift.reversals_reported.total_ratio),
        "drift_total_V": None if metrics.drift.reversals_reported.total_drift_V is None else float(metrics.drift.reversals_reported.total_drift_V),
        "drift_reference_width_V": float(metrics.drift.reversals_reported.reference_width_V),
        "expected_crossings_n": None if metrics.drift.reversals_reported.expected_crossings_n is None else float(metrics.drift.reversals_reported.expected_crossings_n),
        "overload_reached": metrics.overload_reached,
        "max_signal_slope_V_s": metrics.max_signal_slope_V_s,
    }
    summary.update(_mode_summary("reversals_reported", reported))
    summary.update(_mode_summary("reversals_suppressed", suppressed))

    for prefix, drift in (("reversals_reported", metrics.drift.reversals_reported), ("reversals_suppressed", metrics.drift.reversals_suppressed)):
        summary.update({
            f"{prefix}_drift_per_crossing_V": float(drift.mean_error_per_crossing_V),
            f"{prefix}_drift_per_crossing_ratio": float(drift.per_crossing_ratio),
            f"{prefix}_drift_rate_V_s": float(drift.drift_rate_V_s),
            f"{prefix}_expected_total_drift_V": None if drift.total_drift_V is None else float(drift.total_drift_V),
            f"{prefix}_expected_total_ratio": None if drift.total_ratio is None else float(drift.total_ratio),
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
# plotting.py
# =============================================================================

def _histogram_on_centers(values, centers, log=False):
    """Return an empirical PMF on an existing modeled x-grid."""
    values = np.asarray(values, dtype=float)
    centers = np.asarray(centers, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0 or len(centers) == 0:
        return np.zeros_like(centers, dtype=float)
    if len(centers) == 1:
        return np.array([1.0], dtype=float)

    if log:
        positive = centers > 0
        if not np.all(positive):
            raise ValueError("Log histogram centers must be positive")
        edges = np.empty(len(centers) + 1, dtype=float)
        edges[1:-1] = np.sqrt(centers[:-1] * centers[1:])
        edges[0] = centers[0] ** 2 / edges[1]
        edges[-1] = centers[-1] ** 2 / edges[-2]
    else:
        edges = np.empty(len(centers) + 1, dtype=float)
        edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
        edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
        edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])

    counts, _ = np.histogram(values, bins=edges)
    total = float(np.sum(counts))
    return counts.astype(float) / total if total > 0 else np.zeros_like(centers, dtype=float)


def _centers_to_edges(centers):
    centers = np.asarray(centers, dtype=float)
    if len(centers) < 2:
        delta = 0.5 if len(centers) == 0 else max(abs(float(centers[0])) * 0.05, 0.5)
        center = 0.0 if len(centers) == 0 else float(centers[0])
        return np.array([center - delta, center + delta], dtype=float)
    edges = np.empty(len(centers) + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def _same_direction_mean_width(state, local_results):
    model = getattr(local_results, "mode_reversals_reported", None)
    distribution = getattr(model, "same_width_distribution_VW", None) if model is not None else None
    if distribution is None:
        distribution = getattr(local_results, "same_width_distribution_VW", None)
    if distribution is not None:
        P = np.asarray(distribution, dtype=float)
        total = float(np.sum(P))
        if total > 0:
            return float(np.sum(P * state.w[None, :]) / total)
    metadata = getattr(state.characterization, "metadata", None) or {}
    return float(metadata.get("nominal_level_width_V", np.mean(state.w)))


def plot_crossing_source(state, crossing_source, empirical_run=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(state.Vs, state.DVs, crossing_source.cross_source_VDV.T, shading="auto")

    if empirical_run is not None:
        try:
            tx = np.asarray(empirical_run.get_eb_tx_s(), dtype=float)
            vin = np.asarray(empirical_run.get_eb_vin(), dtype=float)
            source = empirical_run.adc.sim_input_signal if hasattr(empirical_run, "adc") else empirical_run.sim_input_signal
            source_t = np.asarray(source.time, dtype=float)
            source_v = np.asarray(source.data, dtype=float)
            if len(source_t) > 2 and len(tx) == len(vin):
                source_dv = np.gradient(source_v, source_t)
                dv_cross = np.interp(tx, source_t, source_dv)
                H, _, _ = np.histogram2d(vin, dv_cross, bins=[_centers_to_edges(state.Vs), _centers_to_edges(state.DVs)])
                if np.sum(H) > 0:
                    H = H / np.sum(H)
                    positive = H[H > 0]
                    if len(positive):
                        levels = np.quantile(positive, [0.50, 0.75, 0.90])
                        levels = np.unique(levels[levels > 0])
                        if len(levels):
                            ax.contour(state.Vs, state.DVs, H.T, levels=levels, linewidths=1.0)
                            ax.plot([], [], label="Empirical golden-truth contours")
        except Exception:
            pass

    ax.set_xlabel("V")
    ax.set_ylabel("V' at source crossing")
    ax.set_title("Estimated source-crossing PMF")
    fig.colorbar(im, ax=ax, label="D2 probability")
    if empirical_run is not None:
        ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax

def plot_level_width_distributions(state, level_width_results, empirical_metrics=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(state.w, level_width_results.occupancy_statistics.pmf, label="D2 occupancy / characterized W")
    ax.plot(state.w, level_width_results.crossing_statistics.pmf, label="D2 same-direction next crossings")
    if empirical_metrics is not None:
        empirical_pmf = _histogram_on_centers(empirical_metrics.widths, state.w)
        ax.step(state.w, empirical_pmf, where="mid", linewidth=1.6, label="Empirical golden truth")
    ax.set_xlabel("Level width w (V)")
    ax.set_ylabel("Probability")
    ax.set_title("Same-direction level-width distribution")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax

def plot_delta_w(state, local_results, empirical_metrics=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    st = local_results.same_direction_statistics
    mean_width = _same_direction_mean_width(state, local_results)
    ax.plot(local_results.DW, st.pmf, label="D2 same-direction Δw")
    if empirical_metrics is not None:
        empirical_pmf = _histogram_on_centers(empirical_metrics.delta_w, local_results.DW)
        ax.step(local_results.DW, empirical_pmf, where="mid", linewidth=1.6, label="Empirical golden truth")
    ax.axvline(st.mean, linestyle="-", label="D2 mean Δw")
    ax.axvline(st.mean_m1, linestyle="--", label="D2 mean Δw ± sigma")
    ax.axvline(st.mean_p1, linestyle="--")
    ax.axvline(-mean_width, linestyle=":", linewidth=1.4, label=f"± mean width = {mean_width*1e3:.1f} mV")
    ax.axvline(+mean_width, linestyle=":", linewidth=1.4)
    ax.set_xlabel(r"$\Delta w = w_{n+1}-w_n$")
    ax.set_ylabel("Probability")
    ax.set_title(f"Same-direction local width variation | mean width = {mean_width*1e3:.2f} mV")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax

def plot_reversal_recapture(local_results, empirical_metrics=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    st = local_results.reversal_recapture_statistics
    P = np.sum(local_results.P_reversal_error_V, axis=0)
    P = P / np.sum(P) if np.sum(P) > 0 else P
    ax.plot(local_results.reversal_error_V, P, label="D2 reversal recapture error")
    if empirical_metrics is not None:
        empirical_pmf = _histogram_on_centers(empirical_metrics.reversal_error_V, local_results.reversal_error_V)
        ax.step(local_results.reversal_error_V, empirical_pmf, where="mid", linewidth=1.6, label="Empirical golden truth")
    ax.axvline(0.0, linestyle=":", label="Ideal = 0 V")
    ax.axvline(st.mean, linestyle="-", label="D2 mean")
    ax.set_xlabel(r"$V_{next\ crossing}-V_{previous\ crossing}$ (V)")
    ax.set_ylabel("Probability")
    ax.set_title("Reversal amplitude recapture")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax

def plot_crossing_time(local_results, empirical_metrics=None):
    fig, ax = plt.subplots(figsize=(8, 4))

    reported = getattr(local_results, "mode_reversals_reported", None)
    suppressed = getattr(local_results, "mode_reversals_suppressed", None)
    if reported is not None:
        ax.plot(reported.crossing_time_s, reported.crossing_time_PMF, label="D2 reversals reported")
        if empirical_metrics is not None and len(empirical_metrics.crossing_rate_reversals_reported.rate_samples_Hz):
            empirical_dt = 1.0 / np.asarray(empirical_metrics.crossing_rate_reversals_reported.rate_samples_Hz, dtype=float)
            empirical_pmf = _histogram_on_centers(empirical_dt, reported.crossing_time_s, log=True)
            ax.step(reported.crossing_time_s, empirical_pmf, where="mid", linewidth=1.5, label="Empirical reported")
    else:
        ax.plot(local_results.crossing_time_s, local_results.crossing_time_PMF, label="D2 reversals reported")
    if suppressed is not None:
        ax.plot(suppressed.crossing_time_s, suppressed.crossing_time_PMF, label="D2 reversals suppressed")
        if empirical_metrics is not None and len(empirical_metrics.crossing_rate_reversals_suppressed.rate_samples_Hz):
            empirical_dt = 1.0 / np.asarray(empirical_metrics.crossing_rate_reversals_suppressed.rate_samples_Hz, dtype=float)
            empirical_pmf = _histogram_on_centers(empirical_dt, suppressed.crossing_time_s, log=True)
            ax.step(suppressed.crossing_time_s, empirical_pmf, where="mid", linewidth=1.5, label="Empirical suppressed")

    ax.set_xscale("log")
    ax.set_xlabel(r"$\Delta t_{reported}$ (s)")
    ax.set_ylabel("Probability")
    ax.set_title("Reported-event interval distributions")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax

def plot_delta_w_vs_amplitude(state, local_results, empirical_run=None):
    positive = local_results.P_DW_same_V[local_results.P_DW_same_V > 0]
    mean_width = _same_direction_mean_width(state, local_results)
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(positive) > 0 and np.min(positive) < np.max(positive):
        im = ax.pcolormesh(state.Vs, local_results.DW, local_results.P_DW_same_V.T, shading="auto", norm=LogNorm(vmin=np.min(positive), vmax=np.max(positive)))
    else:
        im = ax.pcolormesh(state.Vs, local_results.DW, local_results.P_DW_same_V.T, shading="auto")

    if empirical_run is not None:
        from model.empirical import extract_crossing_data
        data = extract_crossing_data(empirical_run)
        interval_width = np.asarray(data["interval_width"], dtype=float)
        event_direction = np.asarray(data["event_direction"], dtype=float)
        vin = np.asarray(data["vin"], dtype=float)
        if len(interval_width) >= 2:
            delta_w_all = np.diff(interval_width)
            same_triplet = (event_direction[:-2] * event_direction[1:-1] > 0) & (event_direction[1:-1] * event_direction[2:] > 0)
            empirical_dw = delta_w_all[same_triplet]
            empirical_v = vin[2:][same_triplet]
            ax.scatter(empirical_v, empirical_dw, s=8, facecolors="none", edgecolors="black", linewidths=0.5, alpha=0.45, label="Empirical golden truth")

    ax.axhline(-mean_width, linestyle=":", linewidth=1.4, label=f"± mean width = {mean_width*1e3:.1f} mV")
    ax.axhline(+mean_width, linestyle=":", linewidth=1.4)
    ax.set_xlabel("V")
    ax.set_ylabel(r"$\Delta w$")
    ax.set_title(rf"Same-direction $\Delta w$ versus amplitude | mean width = {mean_width*1e3:.2f} mV")
    fig.colorbar(im, ax=ax, label="D2 probability mass")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, ax

def plot_dynamic_range(range_min_x, range_max_x, DR_matrix, best, coverage=0.997):
    fig, ax = plt.subplots(figsize=(3.5, 3))
    mesh = ax.pcolormesh(range_max_x, range_min_x, DR_matrix, shading="nearest", cmap="RdYlGn")
    fig.colorbar(mesh, ax=ax, label="Dynamic range (dB)")
    ax.scatter(best.max_x, best.min_x, marker="*", s=250, color="black", zorder=5)
    ax.set_xlabel("Input range high limit")
    ax.set_ylabel("Input range lower limit")
    ax.set_title(f"Dynamic range for {coverage*100:.1f}% coverage")
    ax.invert_xaxis()
    fig.tight_layout()
    return fig, ax


def plot_crossing_probability(state, headstart_bins_n=None):
    """Diagnostic local single-bin Pcross; not used in the final rate metric."""
    probability = np.mean(compute_crossing_probability(state, headstart_bins_n=headstart_bins_n), axis=2)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(state.D2Vs, state.DVs, probability, shading="auto", vmin=0.0, vmax=1.0)
    ax.set_xlabel("V''")
    ax.set_ylabel("V'")
    ax.set_title("Diagnostic: P(cross before leaving one V' bin)")
    fig.colorbar(im, ax=ax, label="P(cross)")
    fig.tight_layout()
    return fig, ax


def plot_model_hierarchy_widths(state, characterization_only_metrics, first_order_metrics=None, second_order_metrics=None, zeroth_order_metrics=None, empirical_metrics=None):
    """Compare available W/D0/D1/D2 width PMFs against empirical golden truth."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(state.w, characterization_only_metrics.width_statistics.pmf, label="W only")
    if zeroth_order_metrics is not None:
        ax.plot(state.w, zeroth_order_metrics.width_statistics.pmf, label="D0 zeroth-order")
    if first_order_metrics is not None:
        ax.plot(state.w, first_order_metrics.width_statistics.pmf, label="D1 first-order")
    if second_order_metrics is not None:
        ax.plot(state.w, second_order_metrics.level_width.crossing_statistics.pmf, label="D2 second-order")
    if empirical_metrics is not None:
        empirical_pmf = _histogram_on_centers(empirical_metrics.widths, state.w)
        ax.step(state.w, empirical_pmf, where="mid", linewidth=1.8, label="Empirical golden truth")
    ax.set_xlabel("Level width w (V)")
    ax.set_ylabel("Probability")
    ax.set_title("Model hierarchy: same-direction level-width distribution")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax

def plot_model_hierarchy_delta_w(first_order_metrics, second_order_metrics, empirical_metrics=None):
    """Compare D1 and D2 same-direction Δw against empirical golden truth."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(first_order_metrics.delta_w_axis_V, first_order_metrics.delta_w_statistics.pmf, label="D1 first-order")
    ax.plot(second_order_metrics.local_distortion.DW, second_order_metrics.local_distortion.same_direction_statistics.pmf, label="D2 second-order")
    if empirical_metrics is not None:
        empirical_pmf = _histogram_on_centers(empirical_metrics.delta_w, second_order_metrics.local_distortion.DW)
        ax.step(second_order_metrics.local_distortion.DW, empirical_pmf, where="mid", linewidth=1.8, label="Empirical golden truth")
    mean_width = second_order_metrics.level_width.crossing_statistics.mean
    ax.axvline(-mean_width, linestyle=":", linewidth=1.4, label=f"± D2 mean width = {mean_width*1e3:.1f} mV")
    ax.axvline(+mean_width, linestyle=":", linewidth=1.4)
    ax.set_xlabel(r"$\Delta w$ (V)")
    ax.set_ylabel("Probability")
    ax.set_title("Model hierarchy: same-direction local width variation")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax

# =============================================================================
# diagnostics.py
# =============================================================================

def conservation_report(local_distortion_results):
    result = local_distortion_results
    return {
        "maximum_absolute_conservation_error": float(np.max(np.abs(result.conservation_error_V))),
        "total_crossing_mass": float(np.sum(result.crossing_mass_V)),
        "total_domain_escape_mass": float(np.sum(result.domain_escape_mass_V)),
        "total_stuck_mass": float(np.sum(result.stuck_mass_V)),
        "total_bin_change_mass": float(np.sum(result.bin_change_mass_V)),
        "total_reversal_mass": float(np.sum(result.reversal_mass_V)),
        "total_invalid_destination_mass": float(np.sum(result.invalid_destination_mass_V)),
    }


def summarize_state(state):
    return {
        "W_shape": tuple(state.W.shape),
        "D2_shape": tuple(state.D2s.shape),
        "V_range": (float(state.Vs[0]), float(state.Vs[-1])),
        "DV_range": (float(state.DVs[0]), float(state.DVs[-1])),
        "D2V_range": (float(state.D2Vs[0]), float(state.D2Vs[-1])),
        "w_range": (float(state.w[0]), float(state.w[-1])),
        "D2_mass": float(np.sum(state.D2s)),
    }
