#In[]:
# Setup
%reload_ext autoreload
%autoreload 2

from pathlib import Path
import pickle
import signal
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from adc.definitions.slope_range_adc import make_adc
from format import paper
from signals.generator import generate_ramp


#In[]:
# Characterization settings

AMPLITUDE_LOW_V = 0.2
AMPLITUDE_HIGH_V = 0.8
PERIODS_N = 25

MIN_SLOPE_V_S = 2.0
MAX_SLOPE_V_S = 50_000.0
SLOPE_POINTS_N = 50
SLOPES_V_S = np.geomspace(MIN_SLOPE_V_S, MAX_SLOPE_V_S, SLOPE_POINTS_N)

MAX_RUN_TIME_S = 300

SIMULATION_FS_PER_SLOPE = 2000.0

FORCE_RECOMPUTE = True
SHOW_FIGURE = True

CACHE_FILE = ROOT / "outputs" / "slope_range_figure_characterization.pkl"
FIGURE_FILE = ROOT / "figs" / "slope_range.pdf"
CACHE_VERSION = 12


#In[]:
# Simulation resolution

def simulation_fs_Hz(adc, slope_V_s):
    return SIMULATION_FS_PER_SLOPE * float(slope_V_s)


def cache_signature(adc):
    return {
        "version": CACHE_VERSION,
        "amplitude_low_V": AMPLITUDE_LOW_V,
        "amplitude_high_V": AMPLITUDE_HIGH_V,
        "periods_n": PERIODS_N,
        "min_slope_V_s": MIN_SLOPE_V_S,
        "max_slope_V_s": MAX_SLOPE_V_S,
        "slope_points_n": SLOPE_POINTS_N,
        "simulation_fs_per_slope": SIMULATION_FS_PER_SLOPE,
        "adc_name": adc.name,
        "adc_lsb_V": adc.dp.lsb_V,
        "adc_cmp_tau_s": adc.dp.cmp_tau_s,
        "adc_loop_delay_s": adc.dp.loop_delay_s,
        "adc_retention_tau_s": adc.dp.e_l_discharge_tau_s,
        "adc_noise_V": adc.dp.max_noise_V,
    }


#In[]:
# Run one framework ramp characterization

RUN_ATTEMPTS_N = 0
RUN_COMPLETED_N = 0
RUN_SKIPPED_N = 0


class SimulationTimeout(RuntimeError):
    pass


def _simulation_timeout_handler(signum, frame):
    raise SimulationTimeout


def characterize_slope(adc, slope_V_s, point_n, total_points_n):
    global RUN_ATTEMPTS_N, RUN_COMPLETED_N, RUN_SKIPPED_N

    RUN_ATTEMPTS_N += 1
    fs_Hz = simulation_fs_Hz(adc, slope_V_s)

    # periods_n alone controls how many complete ramp repetitions are generated.
    # Do not independently override half_intervals_n.
    ramp = generate_ramp(slope_V_s, amplitude_low_V=AMPLITUDE_LOW_V, amplitude_high_V=AMPLITUDE_HIGH_V, periods_n=PERIODS_N)

    print(f"\n[{point_n:03d}/{total_points_n:03d}] |V'|={slope_V_s:.6g} V/s | fs={fs_Hz/1e3:.1f} kHz | starting", flush=True)

    adc.reset()
    adc.load_input_signal(ramp)
    start_s = time.monotonic()
    previous_handler = signal.signal(signal.SIGALRM, _simulation_timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, MAX_RUN_TIME_S)

    try:
        adc.run(fs_Hz=fs_Hz, progress=True)
    except SimulationTimeout:
        elapsed_s = time.monotonic() - start_s
        RUN_SKIPPED_N += 1
        print(f"\n[{point_n:03d}/{total_points_n:03d}] SKIPPED after {elapsed_s:.1f} s (> {MAX_RUN_TIME_S:.0f} s) | |V'|={slope_V_s:.6g} V/s", flush=True)
        try:
            adc.clean_up()
        except Exception:
            pass
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)

    elapsed_s = time.monotonic() - start_s

    crossing_times_s = np.asarray(adc.eb_txs_s, dtype=float)
    directions = np.asarray(adc.get_eb_dirs(), dtype=int)
    crossing_amplitudes_V = np.interp(crossing_times_s, adc.sim_input_signal.time, adc.sim_input_signal.data)

    adc.clean_up()

    if len(crossing_times_s) < 2:
        RUN_SKIPPED_N += 1
        print(f"[{point_n:03d}/{total_points_n:03d}] SKIPPED: too few crossings", flush=True)
        return None

    # Effective level width is the actual input excursion between consecutive
    # crossings. This remains correct even when a crossing is delayed or missed.
    widths_V = np.abs(float(slope_V_s)) * np.diff(crossing_times_s)
    transition_amplitudes_V = crossing_amplitudes_V[1:]

    # Keep only complete-level transitions. Reversal transitions are treated
    # separately and must not enter the trackable-width characterization.
    same_direction = directions[1:] == directions[:-1]
    widths_V = widths_V[same_direction]
    transition_amplitudes_V = transition_amplitudes_V[same_direction]

    if len(widths_V) == 0:
        RUN_SKIPPED_N += 1
        print(f"[{point_n:03d}/{total_points_n:03d}] SKIPPED: no complete-level transitions", flush=True)
        return None

    RUN_COMPLETED_N += 1
    result = {
        "slope_V_s": float(slope_V_s),
        "simulation_fs_Hz": fs_Hz,
        "widths_V": widths_V,
        "amplitudes_V": transition_amplitudes_V,
        "median_width_V": float(np.median(widths_V)),
        "mean_width_V": float(np.mean(widths_V)),
    }

    print(f"[{point_n:03d}/{total_points_n:03d}] DONE in {elapsed_s:.1f} s | N={len(widths_V):5d} | median w={result['median_width_V'] * 1e3:7.2f} mV | mean w={result['mean_width_V'] * 1e3:7.2f} mV | completed={RUN_COMPLETED_N} skipped={RUN_SKIPPED_N}", flush=True)
    return result


#In[]:
# Characterize all requested slopes

def run_characterization(adc):
    characterization = {}
    total_points_n = len(SLOPES_V_S)

    print("\n=== Slope-range characterization ===", flush=True)
    print(f"Slopes: {MIN_SLOPE_V_S:g} to {MAX_SLOPE_V_S:g} V/s, {SLOPE_POINTS_N} logarithmically spaced points", flush=True)
    print(f"Periods per slope: {PERIODS_N}", flush=True)
    print(f"Simulation frequency: fs = {SIMULATION_FS_PER_SLOPE:g} |V'|", flush=True)
    print(f"Per-run timeout: {MAX_RUN_TIME_S:.0f} s", flush=True)

    total_start_s = time.monotonic()

    for point_n, slope_V_s in enumerate(SLOPES_V_S, start=1):
        result = characterize_slope(adc, float(slope_V_s), point_n, total_points_n)
        characterization[f"{float(slope_V_s):.12g}"] = {"slope_V_s": float(slope_V_s), "skipped": True} if result is None else result

        elapsed_total_s = time.monotonic() - total_start_s
        average_s_per_point = elapsed_total_s / point_n
        remaining_s = average_s_per_point * (total_points_n - point_n)
        print(f"Progress: {point_n}/{total_points_n} ({100.0 * point_n / total_points_n:5.1f}%) | elapsed={elapsed_total_s:.1f} s | estimated remaining={remaining_s:.1f} s", flush=True)

    elapsed_total_s = time.monotonic() - total_start_s
    print(f"\n=== Characterization complete: attempted={RUN_ATTEMPTS_N}, completed={RUN_COMPLETED_N}, skipped={RUN_SKIPPED_N}, total time={elapsed_total_s:.1f} s ===", flush=True)
    return characterization


#In[]:
# Characterize or load cached data

adc = make_adc()
CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
signature = cache_signature(adc)

characterization = None
if CACHE_FILE.exists() and not FORCE_RECOMPUTE:
    with CACHE_FILE.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and payload.get("signature") == signature:
        characterization = payload.get("characterization")

if characterization is None:
    characterization = run_characterization(adc)
    with CACHE_FILE.open("wb") as f:
        pickle.dump({"signature": signature, "characterization": characterization}, f)

print("No tracking-range fit is performed in this version.")


#In[]:
# Prepare measured plot statistics

results = sorted((r for r in characterization.values() if not r.get("skipped", False)), key=lambda x: x["slope_V_s"])
slopes_V_s = np.asarray([r["slope_V_s"] for r in results], dtype=float)
means_V = np.asarray([np.mean(r["widths_V"]) for r in results], dtype=float)
medians_V = np.asarray([np.median(r["widths_V"]) for r in results], dtype=float)
q10_V = np.asarray([np.percentile(r["widths_V"], 10) for r in results], dtype=float)
q90_V = np.asarray([np.percentile(r["widths_V"], 90) for r in results], dtype=float)


#In[]:
# Single-column diagnostic paper figure -- measured data only, no fit
%matplotlib widget

paper.apply()
fig, ax = paper.subplots(columns=1, height_in=3)

ax.fill_between(slopes_V_s, q10_V * 1e3, q90_V * 1e3, color=paper.LIGHT_GRAY, alpha=0.7, linewidth=0, label="10--90% range")
ax.plot(slopes_V_s, means_V * 1e3, "o", color=paper.DARK_GRAY, markersize=2.8, label="Mean measured $w$")
ax.plot(slopes_V_s, medians_V * 1e3, color=paper.BLACK, linewidth=0.8, alpha=0.65, label="Median measured $w$")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(MIN_SLOPE_V_S, MAX_SLOPE_V_S)
ax.set_xlabel(r"Input derivative magnitude $|\dot{V}|$ (V/s)")
ax.set_ylabel(r"Effective level width $w$ (mV)")
ax.grid(True, which="major", linewidth=0.45, alpha=0.25)
ax.grid(True, which="minor", linewidth=0.3, alpha=0.12)
mask = np.logical_and((slopes_V_s > 1e2),(slopes_V_s < 1e3))
mean_w = np.mean( means_V[ mask ] )
ax.axhline(mean_w*1e3, color='red', linestyle='--', zorder=0, label=r"$\bar{w}\pm 10\%$")
ax.axhline(mean_w*1e3*1.1, color='red', linestyle=':', zorder=0)
ax.axhline(mean_w*1e3*0.9, color='red', linestyle=':', zorder=0)
ax.legend(loc="lower right")


fig.tight_layout()
paper.savefig(fig, FIGURE_FILE)
print(f"Saved: {FIGURE_FILE}")

if SHOW_FIGURE:
    plt.show()


#In[]:
# Single-column diagnostic paper figure -- piecewise-linear tracking-range fit
%matplotlib widget

paper.apply()
fig, ax = paper.subplots(columns=1, height_in=3)

# Smooth only the displayed 10--90% envelope
SMOOTH_WINDOW_N = 9

def moving_average_edge(x, window):
    kernel = np.ones(window) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    x_padded = np.pad(x, (pad_left, pad_right), mode="edge")
    return np.convolve(x_padded, kernel, mode="valid")

q10_smooth_V = moving_average_edge(q10_V, SMOOTH_WINDOW_N)
q90_smooth_V = moving_average_edge(q90_V, SMOOTH_WINDOW_N)

# Measured statistics
ax.fill_between(slopes_V_s, q10_smooth_V * 1e3, q90_smooth_V * 1e3, color=paper.LIGHT_GRAY, alpha=0.7, linewidth=0, label="10-90% range")
ax.plot(slopes_V_s, means_V * 1e3, "o", color=paper.DARK_GRAY, markersize=2.8, label="Mean $w$")
ax.plot(slopes_V_s, medians_V * 1e3, color=paper.BLACK, linewidth=0.8, alpha=0.65, label="Median $w$")

# Reference width from the central tracking region
mask = np.logical_and(slopes_V_s > 1e2, slopes_V_s < 1e3)
mean_w = np.mean(means_V[mask])

# Find the best three-region piecewise approximation
x = slopes_V_s
y = means_V

MIN_POINTS_PER_REGION = 5

SS_tot = np.sum((y - np.mean(y)) ** 2)

best_R2 = -np.inf
best_fit = None

for i_low in range(MIN_POINTS_PER_REGION, len(x) - 2 * MIN_POINTS_PER_REGION + 1):
    for i_high in range(i_low + MIN_POINTS_PER_REGION, len(x) - MIN_POINTS_PER_REGION + 1):

        x_low = x[:i_low]
        y_low = y[:i_low]

        x_mid = x[i_low:i_high]
        y_mid = y[i_low:i_high]

        x_high = x[i_high:]
        y_high = y[i_high:]

        # Outer regions: y = ax + b
        a_low, b_low = np.polyfit(x_low, y_low, 1)
        a_high, b_high = np.polyfit(x_high, y_high, 1)

        # Middle region is constrained to the previously determined mean width
        y_hat = np.empty_like(y)
        y_hat[:i_low] = a_low * x_low + b_low
        y_hat[i_low:i_high] = mean_w
        y_hat[i_high:] = a_high * x_high + b_high

        SS_res = np.sum((y - y_hat) ** 2)
        R2 = 1.0 - SS_res / SS_tot

        if R2 > best_R2:
            best_R2 = R2
            best_fit = {
                "i_low": i_low,
                "i_high": i_high,
                "a_low": a_low,
                "b_low": b_low,
                "a_high": a_high,
                "b_high": b_high,
                "y_hat": y_hat.copy(),
            }

# Extract best fit
i_low = best_fit["i_low"]
i_high = best_fit["i_high"]

a_low = best_fit["a_low"]
b_low = best_fit["b_low"]

a_high = best_fit["a_high"]
b_high = best_fit["b_high"]

# Tracking limits are the intersections of the outer fits with the flat region
slope_min_V_s = (mean_w - b_low) / a_low
slope_max_V_s = (mean_w - b_high) / a_high

print(f"Reference width        : {mean_w * 1e3:.3f} mV")
print(f"Best piecewise R²      : {best_R2:.6f}")
print(f"Low-slope fit          : w = {a_low:.6e} |V'| + {b_low:.6e}")
print(f"High-slope fit         : w = {a_high:.6e} |V'| + {b_high:.6e}")
print(f"Lower tracking limit   : {slope_min_V_s:.3f} V/s")
print(f"Upper tracking limit   : {slope_max_V_s:.3f} V/s")
print(f"Low region points      : {i_low}")
print(f"Middle region points   : {i_high - i_low}")
print(f"High region points     : {len(x) - i_high}")

# Plot the best piecewise approximation
x_low_fit = np.geomspace(x[0], slope_min_V_s, 200)
x_mid_fit = np.array([slope_min_V_s, slope_max_V_s])
x_high_fit = np.geomspace(slope_max_V_s, x[-1], 200)

FIT_COLOR = "#C00000"

ax.plot(x_low_fit, (a_low * x_low_fit + b_low) * 1e3, color=FIT_COLOR, linewidth=1.4, label=rf"Piecewise fit")
ax.plot(x_mid_fit, np.full_like(x_mid_fit, mean_w * 1e3), color=FIT_COLOR, linewidth=1.4)
ax.plot(x_high_fit, (a_high * x_high_fit + b_high) * 1e3, color=FIT_COLOR, linewidth=1.4)

# Tracking-range limits
ax.axvline(slope_min_V_s, color=FIT_COLOR, linestyle=":", linewidth=0.9)
ax.axvline(slope_max_V_s, color=FIT_COLOR, linestyle=":", linewidth=0.9)

ax.text(slope_min_V_s, mean_w * 1.8e3, rf"$|\dot{{V}}|_{{\min}}={slope_min_V_s:.1f}$ V/s", rotation=0, va="bottom", ha="left", color=FIT_COLOR)
ax.text(slope_max_V_s, mean_w * 1.8e3, rf"$|\dot{{V}}|_{{\max}}={slope_max_V_s:.0f}$ V/s", rotation=0, va="bottom", ha="left", color=FIT_COLOR)

# Figure formatting
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(MIN_SLOPE_V_S, MAX_SLOPE_V_S)
ax.set_xlabel(r"Input derivative magnitude $|\dot{V}|$ (V/s)")
ax.set_ylabel(r"Effective level width $w$ (mV)")

ax.grid(True, which="major", linewidth=0.45, alpha=0.25)
ax.grid(True, which="minor", linewidth=0.3, alpha=0.12)

ax.legend(loc="lower right")

fig.tight_layout()
paper.savefig(fig, FIGURE_FILE)
print(f"Saved: {FIGURE_FILE}")

if SHOW_FIGURE:
    plt.show()