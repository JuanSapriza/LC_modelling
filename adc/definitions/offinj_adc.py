# adc/definitions/offinj_adc.py
from adc.adc import DesignParameters, LC_ADC, RES_GEN_TYPE


def make_adc():
    name = "offinj_adc"
    design = DesignParameters(
        res_gen_type=RES_GEN_TYPE.OFFSET_INJ,
        Vdd_V=1.0,
        Vss_V=0.0,
        Vm_V=0.5,
        lsb_range_b=6,
        lvl_distance_lsbs=1,
        lvl_offset_V=0,
        cmp_offset_V=0,
        cmp_tau_s=0,
        cmp_hyst_V=0.0,
        inj_pulse_dt_s=100e-15,
        inj_scaling=10 / 1e6,
        loop_delay_s=200e-6,
        lvl_discharge_V_s=0,
        e_l_discharge_tau_s=0,
        dac_max_dnl=0.5,
        dac_dnl_seed=0,
        max_noise_V=0,
        noise_seed=0,
        noise_thermal_tau_s=5e-6
    )
    return LC_ADC(name, design_parameters=design, verbose=False)
