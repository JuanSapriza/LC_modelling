from adc.adc import DesignParameters, LC_ADC, RES_GEN_TYPE


def make_adc():
    name = "slope_range_adc"
    design = DesignParameters(
        res_gen_type=RES_GEN_TYPE.RETURN_TO_VM,
        Vdd_V=1.0,
        Vss_V=0.0,
        Vm_V=0.5,

        #Irrelevant for this architecture
        lvl_distance_lsbs=1,
        cmp_hyst_V=0.0,
        inj_pulse_dt_s=1,
        lvl_discharge_V_s=1,
        inj_scaling=1,

        lsb_range_b     = 5,
        lvl_offset_V    = 0,
        cmp_offset_V    = 10e-9,
        cmp_tau_s       = 10e-9,
        loop_delay_s    = 200e-9,

        # To affect the lower derivatives
        e_l_discharge_tau_s=100e-3,

        dac_max_dnl=0.0,
        dac_dnl_seed=0,
        max_noise_V=0.5e-3,
        noise_seed=0,
    )
    return LC_ADC(name, design_parameters=design, verbose=False)