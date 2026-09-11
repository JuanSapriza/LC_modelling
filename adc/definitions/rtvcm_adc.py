from adc.adc import DesignParameters, LC_ADC, RES_GEN_TYPE


def make_adc():
    name = "rtvcm_adc"
    design = DesignParameters(
        res_gen_type=RES_GEN_TYPE.RETURN_TO_VM,
        Vdd_V=1.0,
        Vss_V=0.0,
        Vm_V=0.5,
        lsb_range_b=6,
        lvl_distance_lsbs=1,
        lvl_offset_V=-5e-6,
        cmp_offset_V=1e-6,
        cmp_tau_s=5e-6,
        cmp_hyst_V=0.0,
        inj_pulse_dt_s=100e-9,
        inj_scaling=10 / 1e6,
        loop_delay_s=0.5e-6,
        lvl_discharge_V_s=0,
        e_l_discharge_tau_s=1,
        dac_max_dnl=0.5,
        dac_dnl_seed=0,
        max_noise_V=10e-3,
        noise_seed=0,
    )
    return LC_ADC(name, design_parameters=design, verbose=False)
