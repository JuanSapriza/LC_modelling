# LC Matrix — Pass 14

## Structure

```text
run_experiment.py
adc/
  adc.py
  definitions/
model/
  characterization.py
  empirical.py
  statistical.py
  simulation.py
  comparison.py
  storage.py
signals/
  generator.py
  loader.py
  data/
tools/
runs/
```

## Signals

Stored signals live only in `signals/data/*.pkl`.

```python
from signals.loader import load_signal

ecg = load_signal(
    "ecg",
    start_s=0,
    end_s=5,
    sampling_frequency_Hz=50_000,
    amplitude_Vpp=0.8,
    offset_V=0.5,
    output_name="ecg_5s",
)
```

`load_signal()` accepts either a directly pickled `Timeseries` or the portable pickle representation used by the bundled ECG, and always returns a new local `Timeseries`.

Synthetic signals are generated on demand:

```python
from signals.generator import generate_sine, generate_sinc

sine = generate_sine(frequency_Hz=10, sampling_frequency_Hz=50_000, duration_s=5, amplitude_V=0.35, offset_V=0.5)
sinc = generate_sinc(sampling_frequency_Hz=50_000, duration_s=5, width_s=0.1, amplitude_V=0.35, offset_V=0.5)
```

`signals/loader.py` also provides `save_signal()` and `list_signals()`.

## Experiments

ADCs are still selected by definition filename in `adc/definitions/`. Signals are actual `Timeseries` objects:

```python
EXPERIMENTS = [
    ("test_adc", ecg),
    ("test_adc", sine),
]
```

The run cache identifies signals from their returned data/time/parameters and loader/generator provenance. Repeating the same signal transformation reuses the cached result; changing crop, sampling rate, amplitude, offset, generator settings, or source pickle creates a new signal entry.

## Model order and verbosity

At the top of `run_experiment.py`:

```python
statistical=StatisticalSimulationParameters(
    model_order="D2",   # "W", "D0", "D1", or "D2"
    ...
)

runtime=RuntimeParameters(
    verbosity=1,        # 0=compact, 1=normal, 2=verbose
    ...
)
```

The selected order is the deepest model evaluated. Lower-order models are also reported when possible:

- `W`: ADC characterization only.
- `D0`: amplitude occupancy `D0(V)`.
- `D1`: amplitude/derivative occupancy `D1(V,V')`, constant derivative between crossings.
- `D2`: full second-order model `D2(V,V',V'')`.

Selecting a lower order avoids computing higher-order signal histograms and transition propagation.

W-only drift uses the global characterized mean width as reconstruction step and averages
`sign(V') * (w_global_mean - w_state_mean)` over characterized `(V,V')` states. It therefore
reports drift per crossing, but not drift per second because W alone contains no signal event rate.
