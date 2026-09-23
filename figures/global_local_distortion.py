#In[]:
# Setup

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from signals.generator import generate_sine
from model.empirical import run_adc
from adc.definitions.global_local_correlated_mismatch_adc import make_adc as make_smooth_adc
from adc.definitions.global_local_random_mismatch_adc import make_adc as make_random_adc
from format import paper


#In[]:
# User controls

RECONSTRUCTION = "spline"  # "spline", "linear", or "zoh"
FIGURE_FILE = ROOT / "figs" / f"global_local_nonuniformity_{RECONSTRUCTION}.pdf"

SINE_FREQUENCY_HZ = 20.0
SIMULATION_FS_HZ = 200_000
SIMULATED_CYCLES_N = 5
EVALUATED_CYCLES_N = 3
EVALUATION_START_CYCLE = 1
ADC_PROGRESS = False

SIGNAL_OFFSET_V = 0.5
LARGE_SIGNAL_AMPLITUDE_V = 0.45
SMALL_SIGNAL_AMPLITUDE_V = 0.025
MAX_ALIGNMENT_SHIFT_S = 0.5 / SINE_FREQUENCY_HZ

SMOOTH_COLOR = paper.DARK_GRAY
RANDOM_COLOR = paper.BLOOD_RED
INPUT_COLOR = paper.BLACK


#In[]:
# Helpers

def get_transfer_metrics(adc):
    widths_V = np.diff(np.asarray(adc.comps.dac.out_vals_V, dtype=float))
    mean_width_V = float(np.mean(widths_V))
    global_distortion = float(np.std(widths_V) / mean_width_V)
    local_sigma = float(np.std(np.diff(widths_V)) / mean_width_V)
    local_rms = float(np.sqrt(np.mean(np.diff(widths_V) ** 2)) / mean_width_V)
    return widths_V, mean_width_V, global_distortion, local_sigma, local_rms


def fit_reconstruction_to_reference(reconstruction, signal, mask):
    metric_time_s = np.asarray(signal.time[mask], dtype=float)
    metric_reference_V = np.asarray(signal.data[mask], dtype=float)

    def evaluate(shift_s, return_fit=False):
        reconstructed_V = np.asarray(reconstruction(metric_time_s + shift_s), dtype=float)
        valid = np.isfinite(reconstructed_V)
        if np.sum(valid) < 0.995 * len(metric_time_s):
            return (np.inf, np.nan, np.nan) if return_fit else np.inf
        reconstructed_valid_V = reconstructed_V[valid]
        reference_valid_V = metric_reference_V[valid]
        A = np.column_stack((reconstructed_valid_V, np.ones_like(reconstructed_valid_V)))
        gain, offset_V = np.linalg.lstsq(A, reference_valid_V, rcond=None)[0]
        if gain <= 0:
            return (np.inf, np.nan, np.nan) if return_fit else np.inf
        fitted_V = gain * reconstructed_valid_V + offset_V
        mse = float(np.mean((fitted_V - reference_valid_V) ** 2))
        return (mse, float(gain), float(offset_V)) if return_fit else mse

    coarse_shifts_s = np.linspace(-MAX_ALIGNMENT_SHIFT_S, MAX_ALIGNMENT_SHIFT_S, 401)
    coarse_errors = np.asarray([evaluate(shift_s) for shift_s in coarse_shifts_s])
    best_index = int(np.argmin(coarse_errors))
    lo_s = coarse_shifts_s[max(0, best_index - 1)]
    hi_s = coarse_shifts_s[min(len(coarse_shifts_s) - 1, best_index + 1)]
    optimum = minimize_scalar(evaluate, bounds=(lo_s, hi_s), method="bounded", options={"xatol": 1e-13})
    shift_s = float(optimum.x)
    mse, gain, offset_V = evaluate(shift_s, return_fit=True)
    aligned_raw_V = np.asarray(reconstruction(np.asarray(signal.time, dtype=float) + shift_s), dtype=float)
    fitted_V = gain * aligned_raw_V + offset_V
    return fitted_V, shift_s, gain, offset_V, mse


def normalized_rmse_db(reference_V, reconstructed_V, amplitude_V, mask):
    valid = mask & np.isfinite(reconstructed_V)
    rmse_V = float(np.sqrt(np.mean((np.asarray(reconstructed_V)[valid] - np.asarray(reference_V)[valid]) ** 2)))
    nrmse = rmse_V / amplitude_V
    return float(20.0 * np.log10(nrmse)), nrmse


def make_signal(amplitude_V, name):
    duration_s = SIMULATED_CYCLES_N / SINE_FREQUENCY_HZ
    signal = generate_sine(frequency_Hz=SINE_FREQUENCY_HZ, sampling_frequency_Hz=SIMULATION_FS_HZ, duration_s=duration_s, amplitude_V=amplitude_V, offset_V=SIGNAL_OFFSET_V, name=name)
    evaluation_start_s = EVALUATION_START_CYCLE / SINE_FREQUENCY_HZ
    evaluation_stop_s = (EVALUATION_START_CYCLE + EVALUATED_CYCLES_N) / SINE_FREQUENCY_HZ
    metric_mask = (signal.time >= evaluation_start_s) & (signal.time <= evaluation_stop_s)
    return signal, metric_mask, evaluation_start_s, evaluation_stop_s


def reconstruct(run, signal, interpolation="spline"):

    adc = run.adc

    event_time_s = np.asarray(run.get_eb_tx_s(), dtype=float)
    event_direction = np.asarray(run.get_eb_dirs(), dtype=int)

    initial_code = int(np.clip(round((signal.data[0] - adc.dp.Vss_V) / adc.dp.lsb_V), 0, adc.dp.lsb_n - 1))
    event_code = initial_code + np.cumsum(event_direction)

    reconstruction_offset_V = float(getattr(adc, "reconstruction_offset_V", 0.0))
    event_level_V = adc.dp.Vss_V + reconstruction_offset_V + event_code * adc.dp.lsb_V

    keep = np.concatenate(([True], np.diff(event_time_s) > 0))
    event_time_s = event_time_s[keep]
    event_level_V = event_level_V[keep]

    if interpolation == "spline":
        reconstruction = CubicSpline(event_time_s, event_level_V, bc_type="natural", extrapolate=False)
    elif interpolation == "linear":
        reconstruction = interp1d(event_time_s, event_level_V, kind="linear", bounds_error=False, fill_value=np.nan)
    elif interpolation in ["zoh", "zero"]:
        reconstruction = interp1d(event_time_s, event_level_V, kind="previous", bounds_error=False, fill_value=np.nan)
    else:
        raise ValueError("interpolation must be 'spline', 'linear', or 'zoh'")

    return reconstruction, event_time_s, event_level_V



def run_case(signal, metric_mask):
    results = {}
    for key, make_adc in (("smooth", make_smooth_adc), ("random", make_random_adc)):
        run = run_adc(make_adc(), signal, fs_Hz=SIMULATION_FS_HZ, progress=ADC_PROGRESS)
        reconstruction, event_time_s, event_level_V = reconstruct(run, signal, interpolation=RECONSTRUCTION)
        fitted_V, shift_s, gain, offset_V, mse = fit_reconstruction_to_reference(reconstruction, signal, metric_mask)
        results[key] = {"run": run, "reconstruction": reconstruction, "fitted_V": fitted_V, "alignment_shift_s": shift_s, "alignment_gain": gain, "alignment_offset_V": offset_V, "fit_mse": mse, "event_time_s": event_time_s, "event_level_V": event_level_V}
    return results


def get_signal_code_span(adc, amplitude_V):
    levels_V = np.asarray(adc.comps.dac.out_vals_V, dtype=float)
    low_V = SIGNAL_OFFSET_V - amplitude_V
    high_V = SIGNAL_OFFSET_V + amplitude_V
    first = max(0, int(np.searchsorted(levels_V, low_V, side="right") - 1))
    last = min(len(levels_V) - 2, int(np.searchsorted(levels_V, high_V, side="left")))
    return first, last


def get_local_distortion_in_signal_span(adc, amplitude_V):
    widths_V = np.diff(np.asarray(adc.comps.dac.out_vals_V, dtype=float))
    mean_width_V = float(np.mean(widths_V))
    first, last = get_signal_code_span(adc, amplitude_V)
    local_widths_V = widths_V[first:last + 1]
    if len(local_widths_V) < 2:
        return 0.0
    return float(np.sqrt(np.mean(np.diff(local_widths_V) ** 2)) / mean_width_V)


#In[]:
# ADC definitions and metrics

adc_smooth = make_smooth_adc()
adc_random = make_random_adc()

widths_smooth_V, mean_width_smooth_V, global_smooth, local_sigma_smooth, local_rms_smooth = get_transfer_metrics(adc_smooth)
widths_random_V, mean_width_random_V, global_random, local_sigma_random, local_rms_random = get_transfer_metrics(adc_random)

assert np.isclose(mean_width_smooth_V, mean_width_random_V)
assert np.isclose(global_smooth, global_random)
assert np.isclose(np.std(adc_smooth.comps.dac.dnl_LSB), np.std(adc_random.comps.dac.dnl_LSB))

local_vcm_smooth = get_local_distortion_in_signal_span(adc_smooth, SMALL_SIGNAL_AMPLITUDE_V)
local_vcm_random = get_local_distortion_in_signal_span(adc_random, SMALL_SIGNAL_AMPLITUDE_V)


#In[]:
# Simulate large and small signals

large_signal, large_mask, large_start_s, large_stop_s = make_signal(LARGE_SIGNAL_AMPLITUDE_V, "Large sine")
large = run_case(large_signal, large_mask)

small_signal, small_mask, small_start_s, small_stop_s = make_signal(SMALL_SIGNAL_AMPLITUDE_V, "5% full-scale sine")
small = run_case(small_signal, small_mask)

for key in ("smooth", "random"):
    large[key]["nrmse_dB"], large[key]["nrmse"] = normalized_rmse_db(large_signal.data, large[key]["fitted_V"], LARGE_SIGNAL_AMPLITUDE_V, large_mask)
    small[key]["nrmse_dB"], small[key]["nrmse"] = normalized_rmse_db(small_signal.data, small[key]["fitted_V"], SMALL_SIGNAL_AMPLITUDE_V, small_mask)

print(f"Reconstruction = {RECONSTRUCTION}")
print(f"ADC A | global distortion = {100 * global_smooth:.2f}% | local distortion = {100 * local_rms_smooth:.2f}% | local @ 5% FS = {100 * local_vcm_smooth:.2f}%")
print(f"ADC B | global distortion = {100 * global_random:.2f}% | local distortion = {100 * local_rms_random:.2f}% | local @ 5% FS = {100 * local_vcm_random:.2f}%")
print(f"Large sine | ADC A NRMSE = {large['smooth']['nrmse_dB']:.2f} dB | ADC B NRMSE = {large['random']['nrmse_dB']:.2f} dB")
print(f"5% FS sine | ADC A NRMSE = {small['smooth']['nrmse_dB']:.2f} dB | ADC B NRMSE = {small['random']['nrmse_dB']:.2f} dB")
for case_name, results in (("Large", large), ("5% FS", small)):
    for key, adc_name in (("smooth", "ADC A"), ("random", "ADC B")):
        print(f"{case_name:8s} | {adc_name} fit: dt = {1e6 * results[key]['alignment_shift_s']:+.2f} us | gain = {results[key]['alignment_gain']:.6f} | offset = {1e3 * results[key]['alignment_offset_V']:+.3f} mV")


#In[]:
# Figure
%matplotlib widget

paper.apply()
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

fig = paper.figure(columns=1, height_in=4)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.75, wspace=0.20)
ax_dnl_smooth = fig.add_subplot(gs[0, 0])
ax_dnl_random = fig.add_subplot(gs[0, 1], sharex=ax_dnl_smooth, sharey=ax_dnl_smooth)
ax_large = fig.add_subplot(gs[1, 0])
ax_small = fig.add_subplot(gs[1, 1])

code_transition = np.arange(len(adc_smooth.comps.dac.dnl_LSB))
for ax, adc, color, title in ((ax_dnl_smooth, adc_smooth, SMOOTH_COLOR, "A: resistor-string gradient"), (ax_dnl_random, adc_random, RANDOM_COLOR, "B: comparator-offset mismatch")):
    small_first, small_last = get_signal_code_span(adc, SMALL_SIGNAL_AMPLITUDE_V)
    ax.axvspan(small_first - 0.5, small_last + 0.5, color=paper.PALE_YELLOW, alpha=0.55, linewidth=0, zorder=0)
    ax.axhline(0.0, color=paper.MID_GRAY, linewidth=0.8, zorder=1)
    ax.bar(code_transition, np.asarray(adc.comps.dac.dnl_LSB), width=0.90, color='black', linewidth=0, zorder=2)
    ax.set_title(title)
    ax.set_xlabel("Code transition")
    local_distortion = local_vcm_smooth if adc is adc_smooth else local_vcm_random
    global_distortion = global_smooth if adc is adc_smooth else global_random
    ax.text(0.02, 0.95, rf"$\epsilon_{{\mathrm{{global}}}}={100 * global_distortion:.1f}\%$;  $\epsilon_{{\mathrm{{local}}}}(V_{{\mathrm{{cm}}}})={100 * local_distortion:.1f}\%$", transform=ax.transAxes, ha="left", va="top", bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=2.0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.35, alpha=0.22)

ax_dnl_smooth.set_ylabel("DNL (LSB)")
ax_dnl_random.tick_params(axis="y", labelleft=False)


def plot_reconstruction_panel(ax, signal, mask, evaluation_start_s, results, title, amplitude_V, show_events=False):
    time_ms = (signal.time[mask] - evaluation_start_s) * 1e3
    reference_mV = (signal.data[mask] - SIGNAL_OFFSET_V) * 1e3
    smooth_mV = (results["smooth"]["fitted_V"][mask] - SIGNAL_OFFSET_V) * 1e3
    mid_mV = (results["random"]["fitted_V"][mask] - SIGNAL_OFFSET_V) * 1e3
    ax.plot(time_ms, reference_mV, color=INPUT_COLOR, linewidth=1.45, label="Input")
    ax.plot(time_ms, smooth_mV, color=SMOOTH_COLOR, linewidth=1.15, label="A")
    ax.plot(time_ms, mid_mV, color=RANDOM_COLOR, linewidth=1.15, label="B")

    if show_events:
        for key, color in (("smooth", SMOOTH_COLOR), ("random", RANDOM_COLOR)):
            fitted_event_time_s = results[key]["event_time_s"] - results[key]["alignment_shift_s"]
            fitted_event_level_V = results[key]["alignment_gain"] * results[key]["event_level_V"] + results[key]["alignment_offset_V"]
            event_mask = (fitted_event_time_s >= signal.time[mask][0]) & (fitted_event_time_s <= signal.time[mask][-1])
            ax.scatter((fitted_event_time_s[event_mask] - evaluation_start_s) * 1e3, (fitted_event_level_V[event_mask] - SIGNAL_OFFSET_V) * 1e3, s=11, facecolors="white", edgecolors=color, linewidths=0.7, zorder=4)
    ax.set_title(title)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(r"$V-0.5$ V (mV)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.35, alpha=0.22)
    ax.text(0.98, 0.95, rf"NRMSE: A $={results['smooth']['nrmse_dB']:.0f}$ dB" + "\n" + rf"B $={results['random']['nrmse_dB']:.0f}$ dB", transform=ax.transAxes, ha="right", va="top", bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=2.0))


plot_reconstruction_panel(ax_large, large_signal, large_mask, large_start_s, large, "-1 dBFS", LARGE_SIGNAL_AMPLITUDE_V, show_events=False)
plot_reconstruction_panel(ax_small, small_signal, small_mask, small_start_s, small, "-26 dBFS", SMALL_SIGNAL_AMPLITUDE_V, show_events=False)

for label, ax in zip(("a", "b", "c", "d"), (ax_dnl_smooth, ax_dnl_random, ax_large, ax_small)):
    ax.text(-0.10, 1.04, label, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")

handles, labels = ax_large.get_legend_handles_labels()
legend = fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01))
paper.style_legend(legend)
fig.subplots_adjust(bottom=0.13, wspace=20)

paper.savefig(fig, FIGURE_FILE)
plt.show()









#In[]:
# Figure
%matplotlib widget

paper.apply()
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

fig = paper.figure(columns=1, height_in=4)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.75, wspace=0.25)
ax_dnl_smooth = fig.add_subplot(gs[0, 0])
ax_dnl_random = fig.add_subplot(gs[0, 1], sharex=ax_dnl_smooth, sharey=ax_dnl_smooth)
ax_large = fig.add_subplot(gs[1, 0])
ax_small = fig.add_subplot(gs[1, 1])

code_transition = np.arange(len(adc_smooth.comps.dac.dnl_LSB))
for ax, adc, color, title in ((ax_dnl_smooth, adc_smooth, SMOOTH_COLOR, "A: spatially correlated"), (ax_dnl_random, adc_random, RANDOM_COLOR, "B: random mismatch")):
    small_first, small_last = get_signal_code_span(adc, SMALL_SIGNAL_AMPLITUDE_V)
    # ax.axvspan(small_first - 0.5, small_last + 0.5, color=paper.PALE_YELLOW, alpha=0.55, linewidth=0, zorder=0)
    ax.axhline(0.0, color=paper.MID_GRAY, linewidth=0.8, zorder=1)
    ax.bar(code_transition, np.asarray(adc.comps.dac.dnl_LSB), width=0.90, color='black', linewidth=0, zorder=2)
    ax.set_title(title)
    ax.set_xlabel("Code transition")
    local_distortion = local_rms_smooth if adc is adc_smooth else local_rms_random
    global_distortion = global_smooth if adc is adc_smooth else global_random
    ax.text(0.02, 0.225, rf"$\epsilon_{{\mathrm{{global}}}}={100 * global_distortion:.0f}\%$;  $\epsilon_{{\mathrm{{local}}}}={100 * local_distortion:.0f}\%$", transform=ax.transAxes, ha="left", va="top", fontsize=12, bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=2.0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.35, alpha=0.22)

ax_dnl_smooth.set_ylabel("DNL (LSB)")
ax_dnl_random.tick_params(axis="y", labelleft=False)


def plot_reconstruction_panel(ax, signal, mask, evaluation_start_s, results, title, amplitude_V, show_events=False):
    time_ms = (signal.time[mask] - evaluation_start_s) * 1e3
    reference_mV = (signal.data[mask] - SIGNAL_OFFSET_V) * 1e3
    smooth_mV = (results["smooth"]["fitted_V"][mask] - SIGNAL_OFFSET_V) * 1e3
    mid_mV = (results["random"]["fitted_V"][mask] - SIGNAL_OFFSET_V) * 1e3
    ax.plot(time_ms, reference_mV, color=INPUT_COLOR, linewidth=6, label="Input", alpha=0.2)
    ax.plot(time_ms, smooth_mV, color=SMOOTH_COLOR, linewidth=3,    alpha=1, label="A")
    ax.plot(time_ms, mid_mV, color=RANDOM_COLOR,    linewidth=1.5,    alpha=1, label="B")

    if show_events:
        for key, color in (("smooth", SMOOTH_COLOR), ("random", RANDOM_COLOR)):
            fitted_event_time_s = results[key]["event_time_s"] - results[key]["alignment_shift_s"]
            fitted_event_level_V = results[key]["alignment_gain"] * results[key]["event_level_V"] + results[key]["alignment_offset_V"]
            event_mask = (fitted_event_time_s >= signal.time[mask][0]) & (fitted_event_time_s <= signal.time[mask][-1])
            ax.scatter((fitted_event_time_s[event_mask] - evaluation_start_s) * 1e3, (fitted_event_level_V[event_mask] - SIGNAL_OFFSET_V) * 1e3, s=11, facecolors="white", edgecolors=color, linewidths=0.7, zorder=4)
    ax.set_title(title,)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(r"V (mV)", labelpad=-2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.35, alpha=0.22)
    ax.text(1, 0.2, rf"NRMSE: A $={-results['smooth']['nrmse_dB']:.0f}$ dB; B $={-results['random']['nrmse_dB']:.0f}$ dB", transform=ax.transAxes, ha="right", va="top", bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=2.0))
    ax.set_ylim(min(reference_mV)*2, max(reference_mV)*1.2)
    ax.set_xlim(60,100)

plot_reconstruction_panel(ax_large, large_signal, large_mask, large_start_s, large, "-1 dBFS", LARGE_SIGNAL_AMPLITUDE_V, show_events=False)
plot_reconstruction_panel(ax_small, small_signal, small_mask, small_start_s, small, "-26 dBFS", SMALL_SIGNAL_AMPLITUDE_V, show_events=False)

for label, ax in zip(("(a)", "(b)", "(c)", "(d)"), (ax_dnl_smooth, ax_dnl_random, ax_large, ax_small)):
    ax.text(-0.10, 1.04, label, transform=ax.transAxes, ha="left", va="bottom")

handles, labels = ax_large.get_legend_handles_labels()
legend = fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07))
paper.style_legend(legend)
fig.subplots_adjust(bottom=0.13)

plt.tight_layout()
paper.savefig(fig, FIGURE_FILE)
plt.show()