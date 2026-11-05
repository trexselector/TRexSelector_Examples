# cpp/trex_selector_methods/ad/trex_spca/CMakeLists.txt
# ==================================================================================
# Demo executables for the Sparse PCA T-Rex selector.
# ==================================================================================

add_demo_target(demo_trex_spca_01_mc_sim
    demo_trex_spca_01_mc_sim/demo_trex_spca_01_mc_sim.cpp
    T_REX_SELECTOR_LIBS
)

add_demo_target(demo_trex_spca_02_mc_sim_pev
    demo_trex_spca_02_mc_sim_pev/demo_trex_spca_02_mc_sim_pev.cpp
    T_REX_SELECTOR_LIBS
)
