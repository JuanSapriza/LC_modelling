import numpy as np
from adc.adc import DesignParameters, LC_ADC, RES_GEN_TYPE
from adc.definitions.global_local_correlated_mismatch_adc import BITS, N_WIDTHS, MID_CODE, TARGET_DNL_STD_LSB

N_LEVELS = 2**BITS
RANDOM_SEED = 211
OFFSET_CLIP_SIGMA = 1.5
BASE_OFFSET_SIGMA = 0.30
MID_OFFSET_BOOST = 0.80
MID_OFFSET_WIDTH_CODES = 35.0

# ADC B: flash/comparator-bank-like threshold mismatch. Each trip point has an
# independent comparator offset. DNL is therefore the difference between two
# neighboring threshold offsets, naturally producing strong code-to-code DNL
# while keeping accumulated INL relatively small.
#
# The offset variance is larger around midscale to emulate a local matching /
# layout hotspot (for example, a bank boundary or locally weaker matching).
# This is an illustrative realization, not a universal property of flash ADCs.
_level_code = np.arange(N_LEVELS, dtype=float)
_mid_envelope = np.exp(-0.5 * ((_level_code - MID_CODE) / MID_OFFSET_WIDTH_CODES) ** 2)
_rng = np.random.default_rng(RANDOM_SEED)
_unit_normal = np.clip(_rng.normal(0.0, 1.0, N_LEVELS), -OFFSET_CLIP_SIGMA, OFFSET_CLIP_SIGMA)
_threshold_offset_raw = (BASE_OFFSET_SIGMA + MID_OFFSET_BOOST * _mid_envelope) * _unit_normal
_raw_dnl = np.diff(_threshold_offset_raw)
_raw_dnl -= np.mean(_raw_dnl)
DNL_LSB = TARGET_DNL_STD_LSB * _raw_dnl / np.std(_raw_dnl)


def make_adc():
    design = DesignParameters(res_gen_type=RES_GEN_TYPE.DAC_BASED_RES, Vdd_V=1.0, Vss_V=0.0, Vm_V=0.5, lsb_range_b=BITS, lvl_distance_lsbs=1, lvl_offset_V=0.0, cmp_offset_V=0.0, cmp_tau_s=0.0, cmp_hyst_V=0.0, loop_delay_s=0.0, lvl_discharge_V_s=0.0, e_l_discharge_tau_s=1.0, dac_max_dnl=0.0, max_noise_V=0.0)
    adc = LC_ADC("global_local_random_mismatch_adc", design_parameters=design, verbose=False)
    widths_LSB = 1.0 + DNL_LSB
    out_vals_V = design.Vss_V + np.concatenate(([0.0], np.cumsum(widths_LSB))) * design.lsb_V
    transfer_shift_V = 0.5 - out_vals_V[MID_CODE]
    adc.comps.dac.dnl_LSB = DNL_LSB.copy()
    adc.comps.dac.inl_LSB = np.concatenate(([0.0], np.cumsum(DNL_LSB)))
    adc.comps.dac.out_vals_V = out_vals_V + transfer_shift_V
    adc.reconstruction_offset_V = transfer_shift_V
    return adc
