# cpp/trex_selector_methods/ad/trex_core/CMakeLists.txt
# ==================================================================================
# Demo executables for the classical T-Rex selector.
# ==================================================================================

add_demo_target(demo_trex_01_single_run
    demo_trex_01_single_run/demo_trex_01_single_run.cpp
    T_REX_SELECTOR_LIBS
)

add_demo_target(demo_trex_02_mc_sim_fixed_support
    demo_trex_02_mc_sim_fixed_support/demo_trex_02_mc_sim_fixed_support.cpp
    T_REX_SELECTOR_LIBS
)

add_demo_target(demo_trex_03_mc_sim_variable_support
    demo_trex_03_mc_sim_variable_support/demo_trex_03_mc_sim_variable_support.cpp
    T_REX_SELECTOR_LIBS
)

add_demo_target(demo_trex_04_mc_sim_lloop_strategies
    demo_trex_04_mc_sim_lloop_strategies/demo_trex_04_mc_sim_lloop_strategies.cpp
    T_REX_SELECTOR_LIBS
)

add_demo_target(demo_trex_05_mc_sim_dummy_distributions
    demo_trex_05_mc_sim_dummy_distributions/demo_trex_05_mc_sim_dummy_distributions.cpp
    T_REX_SELECTOR_LIBS
)

add_demo_target(demo_trex_06_mmap
    demo_trex_06_mmap/demo_trex_06_mmap.cpp
    T_REX_SELECTOR_LIBS
)

add_demo_target(demo_trex_07_mc_sim_mmap
    demo_trex_07_mc_sim_mmap/demo_trex_07_mc_sim_mmap.cpp
    T_REX_SELECTOR_LIBS
)

add_demo_target(demo_trex_08_mc_sim_scalability
    demo_trex_08_mc_sim_scalability/demo_trex_08_mc_sim_scalability.cpp
    T_REX_SELECTOR_LIBS
)

add_demo_target(demo_trex_09_mc_sim_target_fdr
    demo_trex_09_mc_sim_target_fdr/demo_trex_09_mc_sim_target_fdr.cpp
    T_REX_SELECTOR_LIBS
)

add_demo_target(demo_trex_10_mc_sim_dt_ratio
    demo_trex_10_mc_sim_dt_ratio/demo_trex_10_mc_sim_dt_ratio.cpp
    T_REX_SELECTOR_LIBS
)
