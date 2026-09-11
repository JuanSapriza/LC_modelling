from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tools.timeseries import Timeseries
from tools.ts_params import TSP_F_HZ
from tools.utils import timed


# ADC CIRCUIT BLOCKS
class Ref_V:
    def __init__(self, nominal_V):
        self.nominal_V  = nominal_V
        self.out_V      = nominal_V
        self.reset()

    def reset(self):
        self.outs_V     = []

    def run(self, t_s):
        self.out_V = self.nominal_V
        self.outs_V.append(self.out_V)
        return self.out_V

class Source_V:
    def __init__(self, time_s, data_V):
        self.time_s = time_s
        self.data_V = data_V
        self.reset()

    def reset(self):
        self.out_V  = 0
        self.outs_V = []

    def interpolate(self, t_s):
        if t_s < self.time_s[0] or t_s > self.time_s[-1]:
            return 0
        return float(np.interp(t_s, self.time_s, self.data_V))

    def run(self, t_s):
        self.out_V = self.interpolate(t_s)
        self.outs_V.append(self.out_V)
        return self.out_V

    def plot_data(self):
        plt.step(self.time_s, self.data_V)

    def plot_output(self, time_s):
        plt.step(time_s, self.outs_V)

class Noise_V:
    """Band-limited input-referred noise with timestep-independent variance.

    Both the thermal and flicker components are updated with exact Ornstein-Uhlenbeck
    transitions. Their stationary variance therefore does not depend on the numerical
    simulation timestep, unlike drawing an independent thermal sample on every call.
    """
    def __init__(self, max_noise_V, thermal_tau_s=1e-6, seed=0):
        self.max_noise_V = max_noise_V
        self.thermal_tau_s = max(float(thermal_tau_s), 1e-15)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.reset(reset_rng=True)

    def reset(self, reset_rng=True):
        if reset_rng:
            self.rng = np.random.default_rng(self.seed)
        self.out_V = 0.0
        self.outs_V = []
        self.noises_V = []
        self.last_t_s = None
        self.thermal_state = self.rng.normal()
        self.flicker_states = np.zeros(8)
        self.flicker_taus_s = np.logspace(-5, 1, len(self.flicker_states))

    @staticmethod
    def _ou_step(state, dt_s, tau_s, white):
        if dt_s <= 0.0:
            return state
        alpha = np.exp(-dt_s / tau_s)
        return alpha * state + np.sqrt(max(0.0, 1.0 - alpha**2)) * white

    def run(self, t_s, x_V):
        if self.last_t_s is None:
            dt_s = 0.0
        else:
            dt_s = max(float(t_s) - float(self.last_t_s), 0.0)
        self.last_t_s = float(t_s)

        if dt_s > 0.0:
            self.thermal_state = self._ou_step(self.thermal_state, dt_s, self.thermal_tau_s, self.rng.normal())
            alpha = np.exp(-dt_s / self.flicker_taus_s)
            white = self.rng.normal(size=len(self.flicker_states))
            self.flicker_states = alpha * self.flicker_states + np.sqrt(np.maximum(0.0, 1.0 - alpha**2)) * white

        flicker = np.mean(self.flicker_states)
        noise_normalized = (self.thermal_state + flicker) / np.sqrt(2.0)
        noise_V = self.max_noise_V * np.clip(noise_normalized, -1.0, 1.0)
        self.out_V = x_V + noise_V
        self.noises_V.append(noise_V)
        self.outs_V.append(self.out_V)
        return self.out_V


class Comparator:
    def __init__(self, supply_p_V=1, supply_n_V=0, offset_V=10e-6, hyst_V=5e-3, gain=100, tau_s = 10e-6):
        self.supply_p_V     = supply_p_V
        self.supply_n_V     = supply_n_V
        self.threshold_V    = (supply_p_V-supply_n_V)/2
        self.offset_V       = offset_V
        self.hyst_V         = hyst_V
        self.gain           = gain
        self.tau_s          = tau_s
        self.reset()

    def reset(self):
        self.branch = 0
        self.out_V  = self.threshold_V
        self.outs_V = []

    def run(self, t_s, in_p, in_n):
        # Add offset to the input difference
        delta_V = in_p + self.offset_V - in_n
        # Add hysteresis depending on the current branch
        delta_V += self.hyst_V if self.branch == 1 else -self.hyst_V
        # Use a sigmoid function to model soft saturation
        out_V_raw = (self.supply_p_V-self.supply_n_V) / (1 + np.exp(-delta_V*self.gain))
        # Hysteresis branch update
        if out_V_raw > 0.99*(self.supply_p_V-self.supply_n_V): self.branch = 1
        if out_V_raw < 0.01*(self.supply_p_V-self.supply_n_V): self.branch = 0
        # Time constraint: simple first-order lag
        if len(self.outs_V) == 0:
            self.out_V = out_V_raw
        else:
              # time constant (adjust as needed)
            dt = t_s - self.prev_t if hasattr(self, 'prev_t') else 0
            alpha = dt / (self.tau_s + dt) if dt > 0 else 1
            self.out_V = (1 - alpha) * self.outs_V[-1] + alpha * out_V_raw

        self.outs_V.append(self.out_V)
        self.prev_t = t_s
        return self.out_V

    def plot_vin_v_vout(self):
        self.reset()
        outs_V = []
        time_s = np.linspace(0,1,1000)
        triangle_V = np.concatenate( (np.linspace(-1,1,500), np.linspace(1, -1, 500)) )
        for t_s, in_p in zip(time_s, triangle_V):
            out_V = self.run(t_s, in_p, 0)
            outs_V.append(out_V)

        plt.step(triangle_V[:500], outs_V[:500],alpha=0.5)
        plt.step(triangle_V[500:], outs_V[500:],alpha=0.5)
        plt.vlines([0],0,1,color='gray', linewidth=1, alpha=0.5)
        plt.hlines([0.5],-1,1,color='gray', linewidth=1, alpha=0.5)
        self.reset()

class Delay:
    def __init__(self, delay_s=0):
        self.delay_s    = delay_s
        self.reset()

    def reset(self):
        self.in_V       = []
        self.time_s     = []
        self.out_V      = 0
        self.outs_V     = []

    def run(self, t_s, in_V):
        self.time_s.append(t_s)
        self.in_V.append(in_V)
        target_t = t_s - self.delay_s

        # If not enough data, output first value
        if not self.time_s or target_t <= self.time_s[0]:
            self.out_V = self.in_V[0]
        else:
            # Find last index with time <= target_t
            for i in range(len(self.time_s)-1, -1, -1):
                if self.time_s[i] <= target_t:
                    idx = i
                    break
            # Interpolate if possible
            if idx == len(self.time_s) - 1:
                self.out_V = self.in_V[idx]
            else:
                t0, t1 = self.time_s[idx], self.time_s[idx+1]
                v0, v1 = self.in_V[idx], self.in_V[idx+1]
                self.out_V = v0 + (v1-v0)*(target_t-t0)/(t1-t0)
        self.outs_V.append(self.out_V)
        return self.out_V

class DAC:
    def __init__(self, Vss, Vdd, nbits):
        self.Vss = Vss
        self.Vdd = Vdd
        self.nbits = nbits
        self.ncodes = 2**(nbits)
        self.lsb_V  = (Vdd-Vss)/(self.ncodes-1)
        self.codes_V  = np.linspace(Vss,Vdd,self.ncodes)
        self.out_vals_V = self.codes_V # Will be overriden if added non-linearity
        self.reset()

    def reset(self):
        self.outs_V = []
        self.out_V  = 0

    def run(self, code):
        code_index = int(np.clip(int(code), 0, self.ncodes - 1))
        self.out_V = float(self.out_vals_V[code_index])
        self.outs_V.append(self.out_V)
        return self.out_V

    def generate_dac_nonlinearity(self, max_dnl_LSB=0.5, seed=0):
        """Generate one reproducible DAC DNL realization.

        ``seed`` is part of the ADC design configuration so numerical/model
        comparisons can reuse the exact same physical DAC realization.
        """
        self.dnl_seed = seed
        rng = np.random.default_rng(seed)

        # Ideal bit weights in LSB
        ideal_weights = 2.0 ** np.arange(self.nbits)

        # Uniform relative error applied independently to every bit
        bit_errors = rng.uniform(-1.0, 1.0, size=self.nbits)
        weights = ideal_weights * (1.0 + bit_errors)

        codes = np.arange(self.ncodes, dtype=np.uint64)
        bit_indices = np.arange(self.nbits, dtype=np.uint64)
        bits = (codes[:, None] >> bit_indices) & 1

        # Initial DAC transfer characteristic
        raw_output_LSB = bits @ weights

        # Compute raw DNL
        raw_steps_LSB = np.diff(raw_output_LSB)
        raw_dnl_LSB = raw_steps_LSB - 1.0

        # Remove average DNL so endpoint gain remains ideal
        raw_dnl_LSB -= np.mean(raw_dnl_LSB)

        # Scale the complete DNL sequence
        peak_dnl = np.max(np.abs(raw_dnl_LSB))

        if peak_dnl == 0:
            self.dnl_LSB = np.zeros(self.ncodes - 1)
        else:
            self.dnl_LSB = raw_dnl_LSB * (max_dnl_LSB / peak_dnl)

        # INL[k] is the accumulated DNL up to code k
        self.inl_LSB = np.concatenate((
            [0.0],
            np.cumsum(self.dnl_LSB)
        ))

        # Actual output value for every code
        ideal_output_LSB = np.arange(self.ncodes, dtype=float)
        output_LSB = ideal_output_LSB + self.inl_LSB

        self.out_vals_V = self.Vss + output_LSB * self.lsb_V

class Counter:
    def __init__(self, bits=32):
        self.bits   = bits
        self.reset()

    def reset(self):
        self.out    = 0
        self.outs   = []

    def run(self, up, dn, posedge):
        if posedge:
            # Increase or decrease the count
            if up and not dn: self.out += 1
            if dn and not up: self.out -= 1
            # Overflow
            self.out %= 2**self.bits
            self.out = int(self.out)
            # Output the value
        self.outs.append(self.out)
        return self.out

class Digitizer:
    def __init__(self, th=0.5, supply_p_V=1, supply_n_V=0):
        self.th = th
        self.supply_p_V = supply_p_V
        self.supply_n_V = supply_n_V

    def run(self, in_V, old):
        if in_V > self.th*(self.supply_p_V-self.supply_n_V): out = self.supply_p_V
        elif in_V < (1-self.th)*(self.supply_p_V-self.supply_n_V): out = self.supply_n_V
        else: out = old
        return out

class SnH:
    def __init__(self):
        self.reset()

    def reset(self):
        self.out_V = 0
        self.outs_V  = []

    def run(self, in_V, sample):
        if sample: self.out_V = in_V
        self.outs_V.append(self.out_V)
        return self.out_V

class EdgeDet:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_x = 0
        self.outs   = []

    def run(self, x, rising = True):
        out = 0
        if rising:
            if x and not self.last_x: out = 1
        else:
            if not x and self.last_x: out = 1
        self.last_x = x
        self.outs.append(out)
        return out

class Clock:
    def __init__(self, f_Hz):
        self.f_Hz   = f_Hz
        self.hT_s    = 1/(2*f_Hz)
        self.reset()

    def reset(self):
        self.out    = 0
        self.outs   = []

    def run(self, t_s):
        out = int(t_s // self.hT_s) % 2
        posedge = ( out == 1 and self.out == 0 )
        negedge = ( out == 0 and self.out == 1 )
        self.out = out
        self.outs.append(out)
        return self.out, posedge, negedge

class Decay:
    def __init__(self, tau_s):
        self.tau_s  = tau_s
        self.reset()

    def reset(self):
        self.t0_s   = 0
        self.v0_V   = 0
        self.out_V  = 0
        self.outs_V = []

    def run(self, t_s, v0_V, refresh):
        if refresh:
            self.v0_V = v0_V
            self.t0_s = t_s
        self.out_V = -self.v0_V*(1 - np.exp(-(t_s-self.t0_s)/self.tau_s))
        # self.out_V = -0.5*(1 - np.exp(-(t_s-self.t0_s)/self.tau_s))
        self.outs_V.append(self.out_V)
        return self.out_V

class Voltage_injection:
    def __init__(self, inj_V_s):
        self.inj_V_s  = inj_V_s
        self.reset()

    def reset(self):
        self.out_V  = 0
        self.outs_V = []

    def run(self, up, dn, inject):
        if inject: self.out_V  += (self.inj_V_s*up) - (self.inj_V_s*dn)
        self.outs_V.append(self.out_V)
        return self.out_V

class Pulse:
    def __init__(self, dt_s):
        self.dt_s       = dt_s
        self.reset()

    def reset(self):
        self.start_s    = 0
        self.out        = 0
        self.outs       = []

    def run( self, t_s, trigger ):
        if self.out == 0:
            if trigger:
                self.out = 1
                self.start_s = t_s
        elif t_s - self.start_s>= self.dt_s:
            self.out = 0
        self.outs.append(self.out)
        return self.out

class LPF:
    def __init__(self, tau_s):
        self.tau_s      = tau_s
        self.reset()

    def reset(self):
        self.last_t_s   = None
        self.out_V      = 0
        self.outs_V     = []

    def run(self, t_s, in_V):
        if self.last_t_s is None:
            self.last_t_s = t_s
            self.out_V = in_V
        else:
            dt = t_s - self.last_t_s
            alpha = dt / (self.tau_s + dt)
            self.out_V += alpha * (in_V - self.out_V)
            self.last_t_s = t_s
        self.outs_V.append(self.out_V)
        return self.out_V

# ADC MODEL
class RES_GEN_TYPE(Enum):
    RETURN_TO_VM        = 1
    DAC_BASED_RES       = 2
    OFFSET_INJ          = 3
    NONE                = 4

@dataclass
class DesignParameters:
    Vdd_V: float    = 1.0
    Vss_V:float     = 0.0
    Vm_V: float     = 0.5
    res_gen_type: RES_GEN_TYPE  = RES_GEN_TYPE.DAC_BASED_RES

    lsb_range_b: int            = 8
    lvl_distance_lsbs: float    = 1
    lvl_offset_V: float         = 0.0

    cmp_offset_V: float         = 0.0
    cmp_tau_s: float            = 0.0
    cmp_hyst_V: float           = 0.0


    inj_pulse_dt_s: float       = 100e-9
    inj_scaling: float          = 1

    loop_delay_s: float         = 1e-6

    lvl_discharge_V_s: float    = 1e-3
    e_l_discharge_tau_s: float  = 10,

    dac_max_dnl: float          = 0
    dac_dnl_seed: int | None     = 0
    max_noise_V: float          = 0
    noise_thermal_tau_s: float | None = None
    noise_seed: int | None      = 0
    emits_reversal_events: bool = True

    @property
    def lsb_n(self) -> int:
        return 2**self.lsb_range_b

    @property
    def cmp_Av_V_V(self)->float:
        return np.sqrt(12)*self.Vm_V/self.lsb_V

    @property
    def FS_V(self) -> float:
        return self.Vdd_V - self.Vss_V

    @property
    def lsb_V(self) -> float:
        # There are 2**B DAC codes spanning both endpoints, hence 2**B-1 intervals.
        return self.FS_V / (self.lsb_n - 1)

    @property
    def inj_slope_V_s(self) -> float:
        return self.inj_scaling*self.lsb_V/self.inj_pulse_dt_s  #(10*lsb_V/(self.inj_pulse_dt_s/xosos.time[1]))

class Components:
    def __init__(self):
        self.Vin        = None
        self.Vlsb_up    = None
        self.Vlsb_dn    = None
        self.cmp_up     = None
        self.cmp_dn     = None
        self.a2d        = None
        self.cnt        = None
        self.dac        = None
        self.snh        = None
        self.pulse      = None
        self.Vinj       = None
        self.RC_lpf     = None
        self.loop_delay = None
        self.rising     = None
        self.e_l        = None
        self.discharge  = None
        self.noise      = None

    def reset(self):
        for component in vars(self).values():
            reset = getattr(component, "reset", None)
            if callable(reset):
                reset()

class LC_ADC:
    def __init__( self, name="lc_unnamed", design_parameters=None, verbose=True ):
        self.name           = name
        self.verbose        = verbose
        self.dp             = design_parameters
        self.input_signal   = None
        self.comps          = Components()
        self.reset()

        # References for the levels up and down
        self.comps.Vlsb_up          = Ref_V(nominal_V=(self.dp.Vm_V + self.dp.lvl_distance_lsbs*self.dp.lsb_V) + self.dp.lvl_offset_V)
        self.comps.Vlsb_dn          = Ref_V(nominal_V=(self.dp.Vm_V - self.dp.lvl_distance_lsbs*self.dp.lsb_V))
        # Up and Down comparators
        self.comps.cmp_up           = Comparator(hyst_V=self.dp.cmp_hyst_V, offset_V=self.dp.cmp_offset_V, tau_s=self.dp.cmp_tau_s, gain=self.dp.cmp_Av_V_V)
        self.comps.cmp_dn           = Comparator(hyst_V=self.dp.cmp_hyst_V, offset_V=self.dp.cmp_offset_V, tau_s=self.dp.cmp_tau_s, gain=self.dp.cmp_Av_V_V)
        # A digitizer to convert the output of the comparators to digital. Can be thought as two cascaded inverters
        self.comps.a2d              = Digitizer(th=self.dp.Vm_V)
        # A counter to keep track of the signal
        self.comps.cnt              = Counter(bits=self.dp.lsb_range_b)
        # A DAC that takes the input of the counter and is used to generate the residue (DAC-based)
        self.comps.dac              = DAC(Vss=self.dp.Vss_V, Vdd=self.dp.Vdd_V, nbits=self.dp.lsb_range_b)
        self.comps.dac.generate_dac_nonlinearity(max_dnl_LSB=self.dp.dac_max_dnl, seed=self.dp.dac_dnl_seed)

        # A Sample-n-hold circuit to sample the input and generate the residue (return to Vm)
        self.comps.snh              = SnH()
        # A pulse generator for the charge injection circuit
        self.comps.pulse            = Pulse(dt_s=self.dp.inj_pulse_dt_s)
        # The voltage of the capacitor holding the injected (inj) current
        self.comps.Vinj             = Voltage_injection(inj_V_s=self.dp.inj_slope_V_s)
        # A LPF to simulate the charge process of the capacitor
        self.comps.RC_lpf           = LPF(tau_s=30e-9)
        # A delay unit (Δt=Dt) to simulate the loop delay (LD)
        # This should still guarantee no overload, so let's do it much smaller than the
        # minimum needed (when added to the rest of the circuit's delay)
        self.comps.loop_delay       = Delay(delay_s=self.dp.loop_delay_s)
        # A clock to control the update of the counter
        # clk_cnt         = Clock(fsim_Hz/4)
        self.comps.rising           = EdgeDet()
        # The error (e) in the level held (l), only applies to (return to Vm and charge injection techniques)
        self.comps.e_l              = Decay(tau_s=self.dp.e_l_discharge_tau_s)
        # The levels discharge (linear discharge on every input signal's sample)
        self.comps.discharge        = Ref_V(nominal_V=self.dp.lvl_discharge_V_s)
        # Add input-referred thermal and flicker noise to the input signal
        noise_tau_s = self.dp.noise_thermal_tau_s
        if noise_tau_s is None:
            noise_tau_s = self.dp.cmp_tau_s if self.dp.cmp_tau_s > 0 else max(self.dp.loop_delay_s, 1e-6)
        self.comps.noise            = Noise_V(max_noise_V=self.dp.max_noise_V, thermal_tau_s=noise_tau_s, seed=self.dp.noise_seed)

        # print(f"lsb up | dn     : {self.dp.Vlsb_up.nominal_V*1e3:1.1f} | {self.dp.Vlsb_dn.nominal_V*1e3:1.1f} mV")
        # print(f"Comparator BW   : {self.dp.f_bw_cmp_Hz/1e3:1.1f} kHz")
        if self.verbose:
            print("----------------------------")
            print("✨ Generated LC design: ", self.name)
            print(f"Comparator tau  : {self.dp.cmp_tau_s*1e6:1.1f} µs")
            print(f"Comparator gain : {self.dp.cmp_Av_V_V:1.1f} V/V")
            print(f"Loop delay      : {self.comps.loop_delay.delay_s*1e6:1.1f} µs")


    def reset(self):
        self.lvl_V          = 0
        self.dac_out_V      = 0
        self.run_time_s     = []
        self.lvls_V         = []
        self.ress_V         = []
        self.tx             = 0
        self.txs            = []
        self.eb_txs_s       = []
        self.up             = 0
        self.dn             = 0
        self.ups            = []
        self.dns            = []
        self.dir            = 0
        self.dirs           = []
        self.eb_dirs        = []
        self.comps.reset()
        if getattr(self, "verbose", True):
            print("----------------------------")
            print("🧼 Reset tracking variables to defaults")


    def load_input_signal(self, input_signal):
        self.input_signal   = input_signal
        self.comps.cnt.out = int(np.clip(round((input_signal.data[0] - self.dp.Vss_V) / self.dp.lsb_V), 0, self.dp.lsb_n - 1))
        if getattr(self, "verbose", True):
            print("----------------------------")
            print(f"🗃️ Loaded: {self.input_signal.name}")
            print(f"Simulation freq : {self.input_signal.params[TSP_F_HZ]/1e6:1.2f} MHz")
            print(f"Set to initial condition: ~{input_signal.data[0]} V, level: {self.comps.cnt.out}")


    def residue_generation(self, t_s, up, dn):
        if t_s == 0:
            hold_V = 0

        input = self.comps.Vin.run(t_s)
        noisy_input = self.comps.noise.run(t_s, input)

        if self.dp.res_gen_type == RES_GEN_TYPE.RETURN_TO_VM:
            # Add loop delay
            # Increase/decrease the count (DAC input), or sample the input
            refresh                 = self.comps.loop_delay.run( t_s, up or dn )
            hold_V                  = self.comps.snh.run(self.comps.noise.out_V, refresh)
            self.lvl_V              = hold_V + self.comps.e_l.run(t_s, hold_V, refresh)

        if self.dp.res_gen_type == RES_GEN_TYPE.DAC_BASED_RES:
            # _, posedge, _   = clk_cnt.run(t_s)
            posedge                 = self.comps.rising.run(up or dn)
            self.comps.cnt.run( up, dn, posedge )
            dac_code                = self.comps.loop_delay.run( t_s, self.comps.cnt.out )
            self.dac_out_V          = self.comps.dac.run(dac_code)
            self.lvl_V              = self.dac_out_V

        if self.dp.res_gen_type == RES_GEN_TYPE.OFFSET_INJ:
            # _, posedge, _   = clk_cnt.run(t_s)
            posedge                 = self.comps.rising.run(up or dn)
            self.comps.cnt.run( up, dn, posedge )
            dac_code                = self.comps.loop_delay.run( t_s, self.comps.cnt.out )
            self.dac_out_V         += self.dp.lsb_V*(up + -1*dn) - self.comps.discharge.run(t_s)
            self.lvl_V              = self.dac_out_V

        res_V                       = self.dp.Vm_V + noisy_input - self.lvl_V

        self.lvls_V.append(self.lvl_V)
        self.ress_V.append(res_V)

        return res_V

    def comparison(self, t_s, res_V):



        # Compare against the reference LSB
        up_V                        = self.comps.cmp_up.run(t_s, res_V, self.comps.Vlsb_up.run(t_s))
        dn_V                        = self.comps.cmp_dn.run(t_s, self.comps.Vlsb_dn.run(t_s), res_V)

        # Analog to digital conversion (single-bit)
        up                          = self.comps.a2d.run(up_V, 0)
        dn                          = self.comps.a2d.run(dn_V, 0)


        if (up or dn):
            self.dir                = dn # Latch the direction change
            if not self.tx:
                self.tx             = True
                self.txs.append(1)
                self.eb_txs_s.append(t_s)
                self.eb_dirs.append(self.dir)
        else:
            self.tx                 = False
            self.txs.append(0)

        self.ups.append(up)
        self.dns.append(dn)
        self.dirs.append(self.dir)

        return up, dn

    @timed("LC_ADC.run")
    def run(self, tf_s = None, fs_Hz=None, progress=True):
        tf_s            = self.input_signal.time[-1] if tf_s == None or tf_s >= self.input_signal.time[-1] else tf_s
        tf_n            = int(tf_s*self.input_signal.params[TSP_F_HZ])
        fs_Hz           = self.input_signal.params[TSP_F_HZ] if fs_Hz == None or fs_Hz < self.input_signal.params[TSP_F_HZ] else fs_Hz
        samples         = int(np.ceil(((tf_s-self.input_signal.time[0]))*fs_Hz))
        run_time_s      = np.linspace(self.input_signal.time[0], tf_s, samples)
        run_data_V      = np.interp(x=run_time_s, xp=self.input_signal.time[:tf_n], fp=self.input_signal.data[:tf_n])

        self.sim_input_signal = Timeseries(f"Input: {self.input_signal}", time=run_time_s, data=run_data_V, f_Hz=fs_Hz)
        self.comps.Vin  = Source_V(self.sim_input_signal.time, self.sim_input_signal.data)

        if getattr(self, "verbose", True):
            print("----------------------------")
            print(f"▶️ Running simulation until {tf_s:1.2f} seconds ({samples} samples @ {fs_Hz/1e6:1.2f}Msps)")

        self.up, self.dn = 0,0
        for t_s in self.sim_input_signal.time:
            res_V                   = self.residue_generation(t_s, self.up, self.dn)
            self.up, self.dn        = self.comparison(t_s, res_V)
            if progress:
                print(f"\r{100*t_s/tf_s:1.2f} %", end='')
        if progress:
            print("")


    def clean_up(self):
        self.lvls_V         = np.array(self.lvls_V)
        self.ress_V         = np.array(self.ress_V)
        self.eb_txs_s       = np.array(self.eb_txs_s)
        self.ups            = np.array(self.ups)
        self.dns            = np.array(self.dns)
        self.dirs           = np.array(self.dirs)
        self.eb_dirs        = np.array(self.eb_dirs)

    def backup(self, output_path, file_suffix):
        with open(output_path+self.name+file_suffix+".pkl",'wb+') as f:
            pickle.dump( self, f)

    def get_eb_tx_s(self):
        return self.eb_txs_s

    def get_eb_dirs(self):
        return np.array([-1 if x else 1 for x in self.eb_dirs])

    def get_fr_tx_s(self):
        return self.txs

    def get_fr_dirs(self):
        return np.array([-1 if d else 1 for d in self.dirs])

    def get_eb_dtx_s(self):
        return np.diff(self.eb_txs_s)

    def get_fr_t_s(self):return self.sim_input_signal.time

    def get_eb_event_count(self):
        return len(self.eb_txs_s)

    def get_eb_vin(self):
        return np.interp(x=self.eb_txs_s, xp=self.sim_input_signal.time, fp=self.sim_input_signal.data)
