# cpp/trex_selector_methods/ad/trex_screening/CMakeLists.txt
# ==================================================================================
# Demo executables for the Screen T-Rex selector.
# ==================================================================================

add_demo_target(demo_trex_scr_01_mc_sim_screen_trex
    demo_trex_scr_01_mc_sim_screen_trex/demo_trex_scr_01_mc_sim_screen_trex.cpp
    T_REX_SELECTOR_LIBS
)
add_demo_target(demo_trex_scr_02_mc_sim_screen_trex_mmap
    demo_trex_scr_02_mc_sim_screen_trex_mmap/demo_trex_scr_02_mc_sim_screen_trex_mmap.cpp
    T_REX_SELECTOR_LIBS
)
add_demo_target(demo_trex_scr_03_mc_sim_correlated
    demo_trex_scr_03_mc_sim_correlated/demo_trex_scr_03_mc_sim_correlated.cpp
    T_REX_SELECTOR_LIBS
)
add_demo_target(demo_trex_scr_04_mc_sim_biobank
    demo_trex_scr_04_mc_sim_biobank/demo_trex_scr_04_mc_sim_biobank.cpp
    T_REX_SELECTOR_LIBS
)
add_demo_target(demo_trex_scr_05_mc_sim_biobank_mmap
    demo_trex_scr_05_mc_sim_biobank_mmap/demo_trex_scr_05_mc_sim_biobank_mmap.cpp
    T_REX_SELECTOR_LIBS
)
add_demo_target(demo_trex_scr_06_mc_sim_solvers
    demo_trex_scr_06_mc_sim_solvers/demo_trex_scr_06_mc_sim_solvers.cpp
    T_REX_SELECTOR_LIBS
)
