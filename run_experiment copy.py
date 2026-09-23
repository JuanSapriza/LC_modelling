#In[]:
# Imports

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from model.simulation import CharacterizationParameters, DynamicRangeParameters, EmpiricalSimulationParameters, MetricParameters, RuntimeParameters, SignalStatisticsParameters, SimulationParameters, StatisticalSimulationParameters
from model.simulation import run_experiment as execute_experiment, plot_saved_experiment
from model.storage import RunStore
from signals.loader import load_signal
from signals.generator import generate_sine, generate_sinc


#In[]:
# Shared numerical simulation configuration

SIMULATION = SimulationParameters(
    name="baseline",
    characterization=CharacterizationParameters(
        amplitude_low_V=None,
        amplitude_high_V=None,
        periods_n=10,
        ramp_half_intervals_n=5000,
        slew_range_low_V_s=1,
        slew_range_high_V_s=400,
        slew_range_steps_n=200,
        slew_range_type="linear",
        sampling_frequency_Hz=None,
        width_pmf_bins_n=255,
    ),
    signal_statistics=SignalStatisticsParameters(
        amplitude_bins_n=None,
        amplitude_min_V=None,
        amplitude_max_V=None,
        derivative_bins_n=100,
        derivative_min_V_s=None,
        derivative_max_V_s=None,
        second_derivative_bins_n=100,
        second_derivative_min_V_s2=None,
        second_derivative_max_V_s2=None,
    ),
    statistical=StatisticalSimulationParameters(
        model_order="D1",                  # "W", "D0", "D1" or "D2"
        headstart_bins_n=127,
        reversal_error_bins_n=127,
        crossing_time_bins_n=192,
        transition_probability_floor=1e-6,
        max_state_transitions_n=64,
    ),
    empirical=EmpiricalSimulationParameters(
        sampling_frequency_Hz=5e4,
        sampling_frequency_multiplier=1.0,
        duration_s=3,
    ),
    metrics=MetricParameters(
        dynamic_range=DynamicRangeParameters(
            coverage=0.997,
            min_start_V=0.00,
            min_stop_V=0.40,
            min_step_V=0.05,
            max_start_V=0.60,
            max_stop_V=1.05,
            max_step_V=0.05,
        ),
        drift_reference_width_V=None,
        overload_update_time_s=None,
    ),
    runtime=RuntimeParameters(
        verbosity=2,                        # 0=compact, 1=normal, 2=verbose
        progress=True,
        keep_statistical_debug=False,
        save_statistical_details=False,
        plot_intermediate=True,
    ),
).validate()


#In[]:
# Load or generate signals
#
# Stored signals live in signals/data/<name>.pkl.
# Any change to crop, resampling, amplitude, offset, etc. creates a different
# Timeseries identity and therefore a different cached signal entry.

ECG = load_signal(
    "ecg",
    start_s=0,
    end_s=2,
    sampling_frequency_Hz=500_000,
    offset_V=None,
    amplitude_Vpp=None,
    scale=1.0,
    output_name="ecg",
)

SINE_10HZ = generate_sine(
    frequency_Hz=10,
    sampling_frequency_Hz=1_000,
    duration_s=1,
    amplitude_V=0.35,
    offset_V=0.5,
    name="sine_10hz",
)

SINC = generate_sinc(
    sampling_frequency_Hz=50_000,
    duration_s=5,
    width_s=0.05,
    amplitude_V=0.35,
    offset_V=0.5,
    name="sinc",
)


#In[]:
# Select experiments
#
# ADC names correspond to files in adc/definitions/<name>.py.
# Signals are the Timeseries objects created above.

EXPERIMENTS = [
    # ("test_adc", ECG),
    # ("test_adc", SINE_10HZ),
    ("offinj_adc", SINE_10HZ),
    # ("test_adc", SINC),
    # ("rtvcm_adc", ECG),
]


#In[]:
# Execution controls

CHARACTERIZE_ADC = False
RUN_EMPIRICAL = True
RUN_STATISTICAL = False
RUN_COMPARISON = False
FORCE_RECOMPUTE = False
PLOT_RESULTS = True


#In[]:
# Inspect reusable ADC histories already stored in runs/

store = RunStore(ROOT)
if SIMULATION.runtime.verbosity >= 1:
    store.print_runs()


#In[]:
# Run selected experiments

for ADC_NAME, SIGNAL in EXPERIMENTS:
    run_file, comparison = execute_experiment(
        adc_name=ADC_NAME,
        signal=SIGNAL,
        simulation_parameters=SIMULATION,
        root=ROOT,
        characterize=CHARACTERIZE_ADC,
        empirical=RUN_EMPIRICAL,
        statistical=RUN_STATISTICAL,
        compare=RUN_COMPARISON,
        force=FORCE_RECOMPUTE,
    )

    if SIMULATION.runtime.verbosity >= 1:
        print(f"Saved ADC history: {run_file.name}")

    if PLOT_RESULTS and SIMULATION.runtime.plot_intermediate:
        plot_saved_experiment(ADC_NAME, SIGNAL, SIMULATION, root=ROOT)
