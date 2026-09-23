#In[]:
# Setup

from pathlib import Path
import pickle
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Some old signal files were pickled before Timeseries moved under tools/.
import tools.timeseries as _timeseries_module
sys.modules.setdefault("timeseries", _timeseries_module)

from tools.timeseries import Timeseries
from signals.loader import load_signal
from signals.generator import generate_sine
from model.empirical import run_adc
from adc.definitions.demo_lc_adc import make_adc
from format.paper import *


#In[]:
# User controls

FIGURE_FILE = ROOT / "figs" / "static2statistical_signal_dependence.pdf"

SIMULATION_FS_HZ = 1_00_000
ADC_PROGRESS = True

SIGNAL_VPP_V = 0.80
SIGNAL_OFFSET_V = 0.50
ECG_BANDWIDTH_HZ = 180.0
SLOW_SINE_AMPLITUDE_FRACTION = 0.50
FR_BITS = 6

# For the top two rows only. The distributions use the complete loaded extract.
# Plot windows are selected manually in the signal dictionary below.

AMPLITUDE_BINS_N = 72
DERIVATIVE_BINS_N = 72
JOINT_AMPLITUDE_BINS_N = 64
JOINT_DERIVATIVE_BINS_N = 64
DERIVATIVE_DISPLAY_PERCENTILE = 99.7




#In[]:
# Load/generate input signals
# This is intentionally kept flat: each signal is visible and editable here.


ecg = load_signal("ecg", start_s=0.0, end_s=2.0, sampling_frequency_Hz=SIMULATION_FS_HZ, amplitude_Vpp=SIGNAL_VPP_V, offset_V=SIGNAL_OFFSET_V, output_name="ECG")

ecg_amplitude_V = 0.5 * np.ptp(ecg.data)
ecg_offset_V = 0.5 * (np.max(ecg.data) + np.min(ecg.data))
ecg_max_slew_V_s = np.max(np.abs(np.gradient(ecg.data, ecg.time)))

sin_fast = generate_sine(frequency_Hz=ECG_BANDWIDTH_HZ, sampling_frequency_Hz=SIMULATION_FS_HZ, duration_s=2.0, amplitude_V=ecg_amplitude_V, offset_V=ecg_offset_V, name="Sine, same amplitude and bandwidth")

sin_slow_amplitude_V = SLOW_SINE_AMPLITUDE_FRACTION * ecg_amplitude_V
sin_slow_frequency_Hz = ecg_max_slew_V_s / (2 * np.pi * sin_slow_amplitude_V)
sin_slow = generate_sine(frequency_Hz=sin_slow_frequency_Hz, sampling_frequency_Hz=SIMULATION_FS_HZ, duration_s=2.0, amplitude_V=sin_slow_amplitude_V, offset_V=ecg_offset_V, name="Sine, matched maximum slew")

# The supplied framework archive currently contains eeg.pkl but no eap.pkl.
# If eap.pkl is added later, this block will use it without any other change.

eap = load_signal("eeg", sampling_frequency_Hz=SIMULATION_FS_HZ*25, amplitude_Vpp=SIGNAL_VPP_V, offset_V=SIGNAL_OFFSET_V, output_name="Neural extract")
EAP_TITLE = "Neural extract"

voice = load_signal("voice", start_s=0.95, end_s=1.95, sampling_frequency_Hz=SIMULATION_FS_HZ, amplitude_Vpp=SIGNAL_VPP_V, offset_V=SIGNAL_OFFSET_V, output_name="Voice")


#In[]:
# Estimate the occupied bandwidth of the non-synthetic signals
# Only used to define the fixed-rate Nyquist reference.

eap_ac = np.asarray(eap.data) - np.mean(eap.data)
eap_spectrum = np.abs(np.fft.rfft(eap_ac * np.hanning(len(eap_ac)))) ** 2
eap_frequency_Hz = np.fft.rfftfreq(len(eap_ac), d=np.median(np.diff(eap.time)))
eap_cumulative = np.cumsum(eap_spectrum[1:])
eap_bandwidth_Hz = eap_frequency_Hz[np.searchsorted(eap_cumulative, 0.995 * eap_cumulative[-1]) + 1]

voice_ac = np.asarray(voice.data) - np.mean(voice.data)
voice_spectrum = np.abs(np.fft.rfft(voice_ac * np.hanning(len(voice_ac)))) ** 2
voice_frequency_Hz = np.fft.rfftfreq(len(voice_ac), d=np.median(np.diff(voice.time)))
voice_cumulative = np.cumsum(voice_spectrum[1:])
voice_bandwidth_Hz = voice_frequency_Hz[np.searchsorted(voice_cumulative, 0.995 * voice_cumulative[-1]) + 1]


#In[]:
# ADC used for every LC simulation and for the FR reference range

adc_reference = make_adc()
adc_reference.verbose = True

VSS_V = adc_reference.dp.Vss_V
VDD_V = adc_reference.dp.Vdd_V
LC_LEVEL_WIDTH_V = adc_reference.dp.lvl_distance_lsbs * adc_reference.dp.lsb_V

fr_step_V = (VDD_V - VSS_V) / (2**FR_BITS - 1)


#In[]:
# Fixed-rate sampling
# fr_* are Timeseries containing the actual uniformly sampled and quantized points.

fr_ecg_fs_Hz = 2 * ECG_BANDWIDTH_HZ
fr_ecg_time = np.arange(ecg.time[0], ecg.time[-1] + 0.25 / fr_ecg_fs_Hz, 1 / fr_ecg_fs_Hz)
fr_ecg_time = fr_ecg_time[fr_ecg_time <= ecg.time[-1]]
fr_ecg_data = np.interp(fr_ecg_time, ecg.time, ecg.data)
fr_ecg_data = VSS_V + np.clip(np.rint((fr_ecg_data - VSS_V) / fr_step_V), 0, 2**FR_BITS - 1) * fr_step_V
fr_ecg = Timeseries("FR ECG", time=fr_ecg_time, data=fr_ecg_data, f_Hz=fr_ecg_fs_Hz)

fr_sin_fast_fs_Hz = 2 * ECG_BANDWIDTH_HZ
fr_sin_fast_time = np.arange(sin_fast.time[0], sin_fast.time[-1] + 0.25 / fr_sin_fast_fs_Hz, 1 / fr_sin_fast_fs_Hz)
fr_sin_fast_time = fr_sin_fast_time[fr_sin_fast_time <= sin_fast.time[-1]]
fr_sin_fast_data = np.interp(fr_sin_fast_time, sin_fast.time, sin_fast.data)
fr_sin_fast_data = VSS_V + np.clip(np.rint((fr_sin_fast_data - VSS_V) / fr_step_V), 0, 2**FR_BITS - 1) * fr_step_V
fr_sin_fast = Timeseries("FR sine fast", time=fr_sin_fast_time, data=fr_sin_fast_data, f_Hz=fr_sin_fast_fs_Hz)

fr_sin_slow_fs_Hz = 2 * ECG_BANDWIDTH_HZ
fr_sin_slow_time = np.arange(sin_slow.time[0], sin_slow.time[-1] + 0.25 / fr_sin_slow_fs_Hz, 1 / fr_sin_slow_fs_Hz)
fr_sin_slow_time = fr_sin_slow_time[fr_sin_slow_time <= sin_slow.time[-1]]
fr_sin_slow_data = np.interp(fr_sin_slow_time, sin_slow.time, sin_slow.data)
fr_sin_slow_data = VSS_V + np.clip(np.rint((fr_sin_slow_data - VSS_V) / fr_step_V), 0, 2**FR_BITS - 1) * fr_step_V
fr_sin_slow = Timeseries("FR sine slow", time=fr_sin_slow_time, data=fr_sin_slow_data, f_Hz=fr_sin_slow_fs_Hz)

fr_eap_fs_Hz = 2 * eap_bandwidth_Hz
fr_eap_time = np.arange(eap.time[0], eap.time[-1] + 0.25 / fr_eap_fs_Hz, 1 / fr_eap_fs_Hz)
fr_eap_time = fr_eap_time[fr_eap_time <= eap.time[-1]]
fr_eap_data = np.interp(fr_eap_time, eap.time, eap.data)
fr_eap_data = VSS_V + np.clip(np.rint((fr_eap_data - VSS_V) / fr_step_V), 0, 2**FR_BITS - 1) * fr_step_V
fr_eap = Timeseries("FR EAP", time=fr_eap_time, data=fr_eap_data, f_Hz=fr_eap_fs_Hz)

fr_voice_fs_Hz = 2 * voice_bandwidth_Hz
fr_voice_time = np.arange(voice.time[0], voice.time[-1] + 0.25 / fr_voice_fs_Hz, 1 / fr_voice_fs_Hz)
fr_voice_time = fr_voice_time[fr_voice_time <= voice.time[-1]]
fr_voice_data = np.interp(fr_voice_time, voice.time, voice.data)
fr_voice_data = VSS_V + np.clip(np.rint((fr_voice_data - VSS_V) / fr_step_V), 0, 2**FR_BITS - 1) * fr_step_V
fr_voice = Timeseries("FR voice", time=fr_voice_time, data=fr_voice_data, f_Hz=fr_voice_fs_Hz)


#In[]:
# Level-crossing simulations
# Each signal is passed through the same actual ADC definition from adc/definitions/rtvcm_adc.py.
%reload_ext autoreload
%autoreload 2

adc_ecg = make_adc()
adc_ecg.verbose = True
run_lc_ecg = run_adc(adc_ecg, ecg, fs_Hz=SIMULATION_FS_HZ, progress=ADC_PROGRESS)
lc_ecg = Timeseries("LC ECG", time=run_lc_ecg.get_eb_tx_s(), data=run_lc_ecg.get_eb_dirs())

#In[]:
# sin fast
# adc_sin_fast = make_adc()
# adc_sin_fast.verbose = True
# run_lc_sin_fast = run_adc(adc_sin_fast, sin_fast, fs_Hz=SIMULATION_FS_HZ, progress=ADC_PROGRESS)
# lc_sin_fast = Timeseries("LC sine fast", time=run_lc_sin_fast.get_eb_tx_s(), data=run_lc_sin_fast.get_eb_dirs())

#In[]:
# sin slow
%reload_ext autoreload
%autoreload 2
adc_sin_slow = make_adc()
adc_sin_slow.verbose = True
run_lc_sin_slow = run_adc(adc_sin_slow, sin_slow, fs_Hz=SIMULATION_FS_HZ, progress=ADC_PROGRESS)
lc_sin_slow = Timeseries("LC sine slow", time=run_lc_sin_slow.get_eb_tx_s(), data=run_lc_sin_slow.get_eb_dirs())

#In[]:
# EAP
%reload_ext autoreload
%autoreload 2
adc_eap = make_adc()
adc_eap.verbose = True
run_lc_eap = run_adc(adc_eap, eap, fs_Hz=SIMULATION_FS_HZ*10, progress=ADC_PROGRESS)
lc_eap = Timeseries("LC EAP", time=run_lc_eap.get_eb_tx_s(), data=run_lc_eap.get_eb_dirs())

#In[]:
#Voice
%reload_ext autoreload
%autoreload 2
adc_voice = make_adc()
adc_voice.verbose = True
run_lc_voice = run_adc(adc_voice, voice, fs_Hz=SIMULATION_FS_HZ, progress=ADC_PROGRESS)
lc_voice = Timeseries("LC voice", time=run_lc_voice.get_eb_tx_s(), data=run_lc_voice.get_eb_dirs())


#In[]:
# LC reconstruction from the events
# No interpolation is used here. Each UP/DOWN event moves the reconstructed
# trace by one nominal LC level. Any mismatch in the simulated ADC therefore
# accumulates naturally as drift against the original input.

rec_lc_ecg_initial = VSS_V + np.clip(np.rint((ecg.data[0] - VSS_V) / LC_LEVEL_WIDTH_V), 0, np.floor((VDD_V - VSS_V) / LC_LEVEL_WIDTH_V)) * LC_LEVEL_WIDTH_V
rec_lc_ecg_time = np.concatenate(([ecg.time[0]], lc_ecg.time, [ecg.time[-1]]))
rec_lc_ecg_levels = rec_lc_ecg_initial + np.cumsum(lc_ecg.data * LC_LEVEL_WIDTH_V)
rec_lc_ecg_data = np.concatenate(([rec_lc_ecg_initial], rec_lc_ecg_levels, [rec_lc_ecg_levels[-1] if len(rec_lc_ecg_levels) else rec_lc_ecg_initial]))
rec_lc_ecg = Timeseries("Reconstructed LC ECG", time=rec_lc_ecg_time, data=rec_lc_ecg_data)

# rec_lc_sin_fast_initial = VSS_V + np.clip(np.rint((sin_fast.data[0] - VSS_V) / LC_LEVEL_WIDTH_V), 0, np.floor((VDD_V - VSS_V) / LC_LEVEL_WIDTH_V)) * LC_LEVEL_WIDTH_V
# rec_lc_sin_fast_time = np.concatenate(([sin_fast.time[0]], lc_sin_fast.time, [sin_fast.time[-1]]))
# rec_lc_sin_fast_levels = rec_lc_sin_fast_initial + np.cumsum(lc_sin_fast.data * LC_LEVEL_WIDTH_V)
# rec_lc_sin_fast_data = np.concatenate(([rec_lc_sin_fast_initial], rec_lc_sin_fast_levels, [rec_lc_sin_fast_levels[-1] if len(rec_lc_sin_fast_levels) else rec_lc_sin_fast_initial]))
# rec_lc_sin_fast = Timeseries("Reconstructed LC sine fast", time=rec_lc_sin_fast_time, data=rec_lc_sin_fast_data)

rec_lc_sin_slow_initial = VSS_V + np.clip(np.rint((sin_slow.data[0] - VSS_V) / LC_LEVEL_WIDTH_V), 0, np.floor((VDD_V - VSS_V) / LC_LEVEL_WIDTH_V)) * LC_LEVEL_WIDTH_V
rec_lc_sin_slow_time = np.concatenate(([sin_slow.time[0]], lc_sin_slow.time, [sin_slow.time[-1]]))
rec_lc_sin_slow_levels = rec_lc_sin_slow_initial + np.cumsum(lc_sin_slow.data * LC_LEVEL_WIDTH_V)
rec_lc_sin_slow_data = np.concatenate(([rec_lc_sin_slow_initial], rec_lc_sin_slow_levels, [rec_lc_sin_slow_levels[-1] if len(rec_lc_sin_slow_levels) else rec_lc_sin_slow_initial]))
rec_lc_sin_slow = Timeseries("Reconstructed LC sine slow", time=rec_lc_sin_slow_time, data=rec_lc_sin_slow_data)

rec_lc_eap_initial = VSS_V + np.clip(np.rint((eap.data[0] - VSS_V) / LC_LEVEL_WIDTH_V), 0, np.floor((VDD_V - VSS_V) / LC_LEVEL_WIDTH_V)) * LC_LEVEL_WIDTH_V
rec_lc_eap_time = np.concatenate(([eap.time[0]], lc_eap.time, [eap.time[-1]]))
rec_lc_eap_levels = rec_lc_eap_initial + np.cumsum(lc_eap.data * LC_LEVEL_WIDTH_V)
rec_lc_eap_data = np.concatenate(([rec_lc_eap_initial], rec_lc_eap_levels, [rec_lc_eap_levels[-1] if len(rec_lc_eap_levels) else rec_lc_eap_initial]))
rec_lc_eap = Timeseries("Reconstructed LC EAP", time=rec_lc_eap_time, data=rec_lc_eap_data)

rec_lc_voice_initial = VSS_V + np.clip(np.rint((voice.data[0] - VSS_V) / LC_LEVEL_WIDTH_V), 0, np.floor((VDD_V - VSS_V) / LC_LEVEL_WIDTH_V)) * LC_LEVEL_WIDTH_V
rec_lc_voice_time = np.concatenate(([voice.time[0]], lc_voice.time, [voice.time[-1]]))
rec_lc_voice_levels = rec_lc_voice_initial + np.cumsum(lc_voice.data * LC_LEVEL_WIDTH_V)
rec_lc_voice_data = np.concatenate(([rec_lc_voice_initial], rec_lc_voice_levels, [rec_lc_voice_levels[-1] if len(rec_lc_voice_levels) else rec_lc_voice_initial]))
rec_lc_voice = Timeseries("Reconstructed LC voice", time=rec_lc_voice_time, data=rec_lc_voice_data)



#In[]:
# test



plt.figure()
plt.step(eap.time, eap.data)
plt.step(rec_lc_eap.time, rec_lc_eap.data, where='post')
arrow_y = 0.5
up_events =         (lc_eap.data > 0)
down_events =       (lc_eap.data < 0)
plt.scatter(   lc_eap.time[up_events], np.full(np.sum(up_events), arrow_y), marker=r"$\uparrow$", s=50, color=BLOOD_RED, linewidths=0, clip_on=False)
plt.scatter(   lc_eap.time[down_events], np.full(np.sum(down_events), arrow_y), marker=r"$\downarrow$", s=50, color=BLACK, linewidths=0, clip_on=False)
plt.xlim(0.0692, 0.0702)
plt.show()


#In[]:
# Average LC sampling rates

lc_ecg_fs_Hz = len(lc_ecg.time) / (ecg.time[-1] - ecg.time[0])
# lc_sin_fast_fs_Hz = len(lc_sin_fast.time) / (sin_fast.time[-1] - sin_fast.time[0])
lc_sin_slow_fs_Hz = len(lc_sin_slow.time) / (sin_slow.time[-1] - sin_slow.time[0])
lc_eap_fs_Hz = len(lc_eap.time) / (eap.time[-1] - eap.time[0])
lc_voice_fs_Hz = len(lc_voice.time) / (voice.time[-1] - voice.time[0])


# Signals shown in the figure
# Reorder/add/remove entries here to change the figure columns.
# plot_window_s manually selects the region shown in the first two rows.
%matplotlib widget
%reload_ext autoreload
%autoreload 2

from format.paper import *
from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import LogLocator

signals = {
    "sin_slow": {
        "title": "Sine: same max slew",
        "signal": sin_slow,
        "fr": fr_sin_slow,
        "lc": lc_sin_slow,
        "rec_lc": rec_lc_sin_slow,
        "fr_rate_Hz": fr_sin_slow_fs_Hz,
        "lc_rate_Hz": lc_sin_slow_fs_Hz,
        "plot_window_s": (0.0, 0.14),
            "annotations": [
                {"row": 0, "text": "fixed-rate samples",    "xytext": (0.57, 0.15), "bbox": dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1)},
                {"row": 1, "text": "level-crossing events", "xytext": (0.6, 0.17)},
                {"row": 1, "text": "asymmetric response   \n↘", "xytext": (0.6, 0.75)},
                {"row": 3, "text": "no slope asymmetry\n↘", "xytext": (0.8, 0.75)},
            ],
    },
    "ecg": {
        "title": "ECG",
        "signal": ecg,
        "fr": fr_ecg,
        "lc": lc_ecg,
        "rec_lc": rec_lc_ecg,
        "fr_rate_Hz": fr_ecg_fs_Hz,
        "lc_rate_Hz": lc_ecg_fs_Hz,
        "plot_window_s": (0.18, 0.32),
        "annotations": [
                {"row": 1, "text": "slope asymmetry\n↓", "xytext": (0.45, 0.75)},
                {"row": 1, "text": "level discharge ↗", "xytext": (0.87, -0.1)},
            ],
    },
    # "sin_fast": {
    #     "title": "Sine: same A, BW",
    #     "signal": sin_fast,
    #     "fr": fr_sin_fast,
    #     "lc": lc_sin_fast,
    #     "rec_lc": rec_lc_sin_fast,
    #     "fr_rate_Hz": fr_sin_fast_fs_Hz,
    #     "lc_rate_Hz": lc_sin_fast_fs_Hz,
    #     "plot_window_s": (0.0, 0.12),
    # },
    "eap": {
        "title": EAP_TITLE,
        "signal": eap,
        "fr": fr_eap,
        "lc": lc_eap,
        "rec_lc": rec_lc_eap,
        "fr_rate_Hz": fr_eap_fs_Hz,
        "lc_rate_Hz": lc_eap_fs_Hz,
        "plot_window_s": (0.068, 0.074),
            "annotations": [
                {"row": 1, "text": "← slope overload", "xytext": (0.7, 0.24)},
            ],
    },
    "voice": {
        "title": "Voice",
        "signal": voice,
        "fr": fr_voice,
        "lc": lc_voice,
        "rec_lc": rec_lc_voice,
        "fr_rate_Hz": fr_voice_fs_Hz,
        "lc_rate_Hz": lc_voice_fs_Hz,
        "plot_window_s": (0.715, 0.73),
        "annotations": [
             {"row": 1, "text": "slope overload\n↘", "xytext": (0.43, 0.65)},
        ],
    },
}


# Pre-compute P(V), P(\dot{V}) and P(V,\dot{V})
# These are cheap and intentionally recomputed every time so plot formatting can
# be iterated without rerunning any ADC simulation.

amplitude_edges = np.linspace(VSS_V, VDD_V, AMPLITUDE_BINS_N + 1)
joint_amplitude_edges = np.linspace(VSS_V, VDD_V, JOINT_AMPLITUDE_BINS_N + 1)

for config in signals.values():
    signal = config["signal"]

    derivative = np.gradient(signal.data, signal.time)
    derivative_limit = max(np.percentile(np.abs(derivative), DERIVATIVE_DISPLAY_PERCENTILE), 1e-12)

    p_v, _ = np.histogram(signal.data, bins=amplitude_edges)
    p_v = p_v / np.sum(p_v)

    derivative_edges = np.linspace(0, derivative_limit, DERIVATIVE_BINS_N + 1)
    p_dv_pos, _ = np.histogram(derivative[derivative > 0], bins=derivative_edges)
    p_dv_neg, _ = np.histogram(-derivative[derivative < 0], bins=derivative_edges)
    p_dv_pos = p_dv_pos / len(derivative)
    p_dv_neg = p_dv_neg / len(derivative)

    joint_derivative_edges = np.linspace(-derivative_limit, derivative_limit, JOINT_DERIVATIVE_BINS_N + 1)
    p_v_dv, _, _ = np.histogram2d(signal.data, derivative, bins=(joint_amplitude_edges, joint_derivative_edges))
    p_v_dv = p_v_dv / np.sum(p_v_dv)

    config["P_V"] = gaussian_filter1d(p_v, sigma=1.5)
    config["P_DV_POS"] = gaussian_filter1d(p_dv_pos, sigma=1.5)
    config["P_DV_NEG"] = gaussian_filter1d(p_dv_neg, sigma=1.5)
    config["P_V_DV"] = p_v_dv
    config["DV_EDGES"] = derivative_edges
    config["JOINT_DV_EDGES"] = joint_derivative_edges



joint_positive = np.concatenate([config["P_V_DV"][config["P_V_DV"] > 0] for config in signals.values()])
joint_vmax = np.max(joint_positive)
joint_vmin = max(np.min(joint_positive), joint_vmax * 1e-4)
joint_norm = LogNorm(vmin=joint_vmin, vmax=joint_vmax)

FIGURE_HEIGHT_IN = 6
fig, axs = subplots(5, len(signals), columns=2, height_in=FIGURE_HEIGHT_IN, squeeze=True, height_ratios=[1.75,1.75,1,1,1])

for ax in axs[3][1:]:
    ax.sharey(axs[3][0])

row_labels = ["Fixed\nrate", "Level\ncrossing", "", "", ""]
row_xlabels = [r"$t$", r"$t$", r"$V$", r"$|\dot{V}|$", r"$V$"]
row_ylabels = [r"$V$", r"$V$", r"$P(V)$", r"$P(\dot{V})$", r"$\dot{V}$"]

for c, config in enumerate(signals.values()):

    signal = config["signal"]
    fr = config["fr"]
    lc = config["lc"]
    rec = config["rec_lc"]
    fr_rate_Hz = config["fr_rate_Hz"]
    lc_rate_Hz = config["lc_rate_Hz"]
    t0, t1 = config["plot_window_s"]

    axs[0][c].set_title(config["title"], pad=2)

    # ------------------------------------------------------------------
    # Fixed-rate sampling
    # ------------------------------------------------------------------
    visible_signal = (signal.time >= t0) & (signal.time <= t1)
    visible_fr = (fr.time >= t0) & (fr.time <= t1)

    axs[0][c].plot(signal.time[visible_signal], signal.data[visible_signal], color=MID_GRAY, linewidth=2.5, alpha=1, zorder=0)

    if np.any(visible_fr):
        y0 = min(np.min(signal.data[visible_signal]), np.min(fr.data[visible_fr])) - 0.05 * np.ptp(signal.data[visible_signal])
        cap_half_width_s = 0.17 / fr_rate_Hz
        axs[0][c].vlines(fr.time[visible_fr], 0, fr.data[visible_fr], color=BLACK, linewidth=0.25)
        axs[0][c].hlines(fr.data[visible_fr], fr.time[visible_fr] - cap_half_width_s, fr.time[visible_fr] + cap_half_width_s, color=BLOOD_RED, linewidth=1.1)

    axs[0][c].text(0.97, 0.93, rf"$f_s={fr_rate_Hz/1e3:.1f}$ kHz" if fr_rate_Hz >= 1e3 else rf"$f_s={fr_rate_Hz:.0f}$ Hz", transform=axs[0][c].transAxes, ha="right", va="top", bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1))
    axs[0][c].set_xlim(t0, t1)

    axs[0][c].set_ylim(-0.1,1.3)

    # ------------------------------------------------------------------
    # Level-crossing sampling
    # ------------------------------------------------------------------
    visible_lc = (lc.time >= t0) & (lc.time <= t1)
    rec_i0 = max(np.searchsorted(rec.time, t0, side="right") - 1, 0)
    rec_i1 = min(np.searchsorted(rec.time, t1, side="right") + 1, len(rec.time))

    axs[1][c].plot(signal.time[visible_signal], signal.data[visible_signal], color=MID_GRAY, linewidth=2.5, alpha=1, zorder=0)

    local_min = np.min(signal.data[visible_signal])
    local_max = np.max(signal.data[visible_signal])
    first_level = -10
    last_level = 1000
    for level_i in range(first_level, last_level + 1):
        level_V = VSS_V + level_i * LC_LEVEL_WIDTH_V
        if VSS_V-1 <= level_V <= 2*VDD_V:
            axs[1][c].axhline(level_V, color=LIGHT_GRAY, linewidth=1, alpha=0.5, zorder=-1)

    offset = ((rec.data[rec_i0]-signal.data[visible_signal][0])//LC_LEVEL_WIDTH_V)*LC_LEVEL_WIDTH_V
    axs[1][c].step(rec.time[rec_i0:rec_i1], (rec.data[rec_i0:rec_i1]-offset), where="post", color=BLOOD_RED, linewidth=1.1, zorder=2)

    yrange = max(local_max - local_min, 1e-6)
    arrow_y = 0
    up_events = visible_lc & (lc.data > 0)
    down_events = visible_lc & (lc.data < 0)
    axs[1][c].scatter(lc.time[up_events], np.full(np.sum(up_events), arrow_y), marker=r"$\uparrow$", s=50, color=BLOOD_RED, linewidths=0, clip_on=False)
    axs[1][c].scatter(lc.time[down_events], np.full(np.sum(down_events), arrow_y), marker=r"$\downarrow$", s=50, color=BLACK, linewidths=0, clip_on=False)

    axs[1][c].text(0.93, 0.93, rf"$\bar{{f_\text{{x}}}}={lc_rate_Hz/1e3:.1f}$ kHz" if lc_rate_Hz >= 1e3 else rf"$\bar{{f_\text{{x}}}}={lc_rate_Hz:.0f}$ Hz", transform=axs[1][c].transAxes, ha="right", va="top", bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1))
    axs[1][c].set_xlim(t0, t1)
    axs[1][c].set_ylim(-0.1, 1.3)

    # ------------------------------------------------------------------
    # P(V)
    # ------------------------------------------------------------------
    amplitude_centers = 0.5 * (amplitude_edges[:-1] + amplitude_edges[1:])
    # axs[2][c].fill_between(amplitude_centers, config["P_V"], step="mid", color=PALE_RED, alpha=0.75, linewidth=0)
    axs[2][c].plot(amplitude_centers, config["P_V"], color=BLOOD_RED, linewidth=2)
    axs[2][c].set_xlim(VSS_V, VDD_V)
    p_v_max = np.max(config["P_V"])
    axs[2][c].set_ylim(-0.08 * p_v_max, 1.12 * p_v_max)

    # ------------------------------------------------------------------
    # P(\dot{V}) -- positive and negative slopes overlaid as |\dot{V}|
    # ------------------------------------------------------------------
    derivative_centers = 0.5 * (config["DV_EDGES"][:-1] + config["DV_EDGES"][1:])
    probability_floor = 0.5 / len(signal.data)
    axs[3][c].plot(derivative_centers, np.maximum(config["P_DV_POS"], probability_floor), color=BLOOD_RED, linestyle=(0, (1, 1)), zorder=10,linewidth=2, label=r"$\dot{V}>0$")
    axs[3][c].plot(derivative_centers, np.maximum(config["P_DV_NEG"], probability_floor), color=MID_GRAY, linestyle=(0, (1, 1)),  zorder=10,linewidth=2, label=r"$\dot{V}<0$")
    axs[3][c].set_yscale("log")
    axs[3][c].margins(y=0.15)
    # axs[3][c].set_xscale("log")
    axs[3][c].set_xlim(config["DV_EDGES"][0], config["DV_EDGES"][-1])

    # ------------------------------------------------------------------
    # P(V,\dot{V})
    # ------------------------------------------------------------------
    joint_mesh = axs[4][c].pcolormesh(joint_amplitude_edges,  config["JOINT_DV_EDGES"], config["P_V_DV"].T, cmap=HEATMAP_CMAP, norm=joint_norm, shading="auto", rasterized=True)
    axs[4][c].axhline(0, color=BLACK, linewidth=0.35, alpha=0.6)
    axs[4][c].set_xlim(VSS_V, VDD_V)
    axs[4][c].set_ylim(min(config["JOINT_DV_EDGES"])*1.3, max(config["JOINT_DV_EDGES"])*1.3)



    for annotation in config.get("annotations", []):

        ax = axs[annotation["row"]][c]

        ax.text(
            annotation["xytext"][0],
            annotation["xytext"][1],
            annotation["text"],
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=11,
            color=BLACK,
            bbox=annotation.get("bbox", None),
        )




# One compact legend is enough for the complete derivative row.
legend = axs[3][3].legend(loc="upper right", borderpad=0.05, handlelength=1.5, handletextpad=0.35, labelspacing=0.18)
style_legend(legend)

# Same compact visual style as the supplied multi-column example.
for r, label in enumerate(row_labels):
    axs[r][0].set_ylabel(row_ylabels[r], rotation=0, ha="right", va="bottom", y=0, labelpad=12)
    if label:
        axs[r][0].text(-0.12, 0.5, label, transform=axs[r][0].transAxes, ha="right", va="center")
    for ax in axs[r]:
        ax.set_xlabel(row_xlabels[r], loc="left", labelpad=3)

for r, row in enumerate(axs):
    for ax in row:
        ax.set_xticks([])
        if r == 3:
            ax.yaxis.set_major_locator(LogLocator(base=10))
            ax.set_axisbelow(True)
            ax.grid(axis="y", which="major", color=VERY_LIGHT_GRAY, linewidth=0.6, alpha=0.8)
        else:
            ax.set_yticks([])
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_facecolor("none")

fig.subplots_adjust(left=0.13, right=0.997, top=0.97, bottom=0.055, wspace=0.04, hspace=0.25)

joint_position = axs[4][0].get_position()
colorbar_ax = fig.add_axes([joint_position.x0 - 0.045, joint_position.y0, 0.008, joint_position.height])
colorbar = fig.colorbar(joint_mesh, cax=colorbar_ax, orientation="vertical")
colorbar.set_label(r"$P(V,\dot{V})$")
colorbar.set_ticks([])
colorbar.ax.yaxis.set_label_position("left")


#In[]
#Save
fig.canvas.draw()

fig.savefig(
    FIGURE_FILE,
    format="pdf",
    facecolor="white",
    transparent=False,
    bbox_inches=None,
)