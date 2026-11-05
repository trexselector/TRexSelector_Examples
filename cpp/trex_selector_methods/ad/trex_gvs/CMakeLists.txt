# cpp/trex_selector_methods/ad/trex_gvs/CMakeLists.txt
# ==================================================================================
# MC simulation executables for the Grouped Variable Selection T-Rex selector.
# ==================================================================================

add_demo_target(demo_trex_gvs_01_mc_sim_hastie_en_blocks
    demo_trex_gvs_01_mc_sim_hastie_en_blocks/demo_trex_gvs_01_mc_sim_hastie_en_blocks.cpp
    T_REX_SELECTOR_LIBS
)

add_demo_target(demo_trex_gvs_02_mc_sim_scattered_blocks
    demo_trex_gvs_02_mc_sim_scattered_blocks/demo_trex_gvs_02_mc_sim_scattered_blocks.cpp
    T_REX_SELECTOR_LIBS
)

add_demo_target(demo_trex_gvs_03_mc_sim_mixed_blocks
    demo_trex_gvs_03_mc_sim_mixed_blocks/demo_trex_gvs_03_mc_sim_mixed_blocks.cpp
    T_REX_SELECTOR_LIBS
)

add_demo_target(demo_trex_gvs_04_mc_sim_neg_traps
    demo_trex_gvs_04_mc_sim_neg_traps/demo_trex_gvs_04_mc_sim_neg_traps.cpp
    T_REX_SELECTOR_LIBS
)

add_demo_target(demo_trex_gvs_05_mc_sim_t3_blocks
    demo_trex_gvs_05_mc_sim_t3_blocks/demo_trex_gvs_05_mc_sim_t3_blocks.cpp
    T_REX_SELECTOR_LIBS
)

add_demo_target(demo_trex_gvs_06_mc_sim_ar1_blocks
    demo_trex_gvs_06_mc_sim_ar1_blocks/demo_trex_gvs_06_mc_sim_ar1_blocks.cpp
    T_REX_SELECTOR_LIBS
)

add_demo_target(demo_trex_gvs_07_mc_sim_arma_blocks
    demo_trex_gvs_07_mc_sim_arma_blocks/demo_trex_gvs_07_mc_sim_arma_blocks.cpp
    T_REX_SELECTOR_LIBS
)

add_demo_target(demo_trex_gvs_08_mc_sim_block_bench
    demo_trex_gvs_08_mc_sim_block_bench/demo_trex_gvs_08_mc_sim_block_bench.cpp
    T_REX_SELECTOR_LIBS
)


