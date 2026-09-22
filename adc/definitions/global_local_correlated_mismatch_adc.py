import numpy as np
from adc.adc import DesignParameters, LC_ADC, RES_GEN_TYPE

BITS = 8
N_WIDTHS = 2**BITS - 1
MID_CODE = 2**(BITS - 1)
MID_TRANSITION = MID_CODE - 1
TARGET_DNL_STD_LSB = 0.17

# ADC A: resistor-string / threshold ladder dominated by a slow systematic
# spatial mismatch mode. Nearby elements match well, so DNL changes only very
# slowly with code. Vcm is deliberately placed at the center of a broad DNL
# lobe, giving almost constant level width over a small signal around Vcm.
#
# The broad low-spatial-frequency mode is a compact model of systematic sheet-
# resistance / geometry gradients across a long resistor array; it is not meant
# to imply a literal sinusoidal physical error.
_code = np.arange(N_WIDTHS, dtype=float)
_phase = 2.0 * np.pi * (_code - MID_TRANSITION) / 128.0
_raw = -np.cos(_phase)

# Remove the average error independently below and above midscale. This keeps
# endpoint gain and the midscale threshold aligned without changing the smooth
# local behavior around Vcm.
_raw[:MID_CODE] -= np.mean(_raw[:MID_CODE])
_raw[MID_CODE:] -= np.mean(_raw[MID_CODE:])
DNL_LSB = TARGET_DNL_STD_LSB * _raw / np.std(_raw)


def make_adc():
    design = DesignParameters(res_gen_type=RES_GEN_TYPE.DAC_BASED_RES, Vdd_V=1.0, Vss_V=0.0, Vm_V=0.5, lsb_range_b=BITS, lvl_distance_lsbs=1, lvl_offset_V=0.0, cmp_offset_V=0.0, cmp_tau_s=0.0, cmp_hyst_V=0.0, loop_delay_s=0.0, lvl_discharge_V_s=0.0, e_l_discharge_tau_s=1.0, dac_max_dnl=0.0, max_noise_V=0.0)
    adc = LC_ADC("global_local_correlated_mismatch_adc", design_parameters=design, verbose=False)
    widths_LSB = 1.0 + DNL_LSB
    out_vals_V = design.Vss_V + np.concatenate(([0.0], np.cumsum(widths_LSB))) * design.lsb_V
    transfer_shift_V = 0.5 - out_vals_V[MID_CODE]
    adc.comps.dac.dnl_LSB = DNL_LSB.copy()
    adc.comps.dac.inl_LSB = np.concatenate(([0.0], np.cumsum(DNL_LSB)))
    adc.comps.dac.out_vals_V = out_vals_V + transfer_shift_V
    adc.reconstruction_offset_V = transfer_shift_V
    return adc
