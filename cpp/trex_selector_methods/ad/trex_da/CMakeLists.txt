# cpp/trex_selector_methods/ad/trex_da/CMakeLists.txt
# ==================================================================================
# Demo executables for the Dependency-Aware T-Rex selector.
# ==================================================================================

add_demo_target(demo_trex_da_01_mc_sim_ar1
    demo_trex_da_01_mc_sim_ar1/demo_trex_da_01_mc_sim_ar1.cpp
    T_REX_SELECTOR_LIBS
)
add_demo_target(demo_trex_da_02_mc_sim_ar1_blocks_plus_white
    demo_trex_da_02_mc_sim_ar1_blocks_plus_white/demo_trex_da_02_mc_sim_ar1_blocks_plus_white.cpp
    T_REX_SELECTOR_LIBS
)
add_demo_target(demo_trex_da_03_mc_sim_ar1_blocks
    demo_trex_da_03_mc_sim_ar1_blocks/demo_trex_da_03_mc_sim_ar1_blocks.cpp
    T_REX_SELECTOR_LIBS
)
add_demo_target(demo_trex_da_04_mc_sim_ht_blocks
    demo_trex_da_04_mc_sim_ht_blocks/demo_trex_da_04_mc_sim_ht_blocks.cpp
    T_REX_SELECTOR_LIBS
)
add_demo_target(demo_trex_da_05_mc_sim_ht_blocks_plus_white
    demo_trex_da_05_mc_sim_ht_blocks_plus_white/demo_trex_da_05_mc_sim_ht_blocks_plus_white.cpp
    T_REX_SELECTOR_LIBS
)
add_demo_target(demo_trex_da_06_mc_sim_groups
    demo_trex_da_06_mc_sim_groups/demo_trex_da_06_mc_sim_groups.cpp
    T_REX_SELECTOR_LIBS
)
add_demo_target(demo_trex_da_07_mc_sim_nn
    demo_trex_da_07_mc_sim_nn/demo_trex_da_07_mc_sim_nn.cpp
    T_REX_SELECTOR_LIBS
)
add_demo_target(demo_trex_da_08_mc_sim_equi_and_bt
    demo_trex_da_08_mc_sim_equi_and_bt/demo_trex_da_08_mc_sim_equi_and_bt.cpp
    T_REX_SELECTOR_LIBS
)
