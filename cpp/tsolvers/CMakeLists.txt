# CMakeLists.txt for tsolvers demos
# ==================================================================================
# CMake configuration for T-Algorithm solver demos.


add_demo_target(demo_ts_01_tlars
    ad/demo_ts_01_tlars/demo_ts_01_tlars.cpp
    T_ALGORITHM_LIBS
)
add_demo_target(demo_ts_02_tlasso
    ad/demo_ts_02_tlasso/demo_ts_02_tlasso.cpp
    T_ALGORITHM_LIBS
)
add_demo_target(demo_ts_03_tstepwise
    ad/demo_ts_03_tstepwise/demo_ts_03_tstepwise.cpp
    T_ALGORITHM_LIBS
)
add_demo_target(demo_ts_04_tenet
    ad/demo_ts_04_tenet/demo_ts_04_tenet.cpp
    T_ALGORITHM_LIBS
)
add_demo_target(demo_ts_05_tstagewise
    ad/demo_ts_05_tstagewise/demo_ts_05_tstagewise.cpp
    T_ALGORITHM_LIBS
)
add_demo_target(demo_ts_06_tomp
    ad/demo_ts_06_tomp/demo_ts_06_tomp.cpp
    T_ALGORITHM_LIBS
)
add_demo_target(demo_ts_07_tgp
    ad/demo_ts_07_tgp/demo_ts_07_tgp.cpp
    T_ALGORITHM_LIBS
)
add_demo_target(demo_ts_08_tacgp
    ad/demo_ts_08_tacgp/demo_ts_08_tacgp.cpp
    T_ALGORITHM_LIBS
)
add_demo_target(demo_ts_09_tmp
    ad/demo_ts_09_tmp/demo_ts_09_tmp.cpp
    T_ALGORITHM_LIBS
)
add_demo_target(demo_ts_10_tncgmp
    ad/demo_ts_10_tncgmp/demo_ts_10_tncgmp.cpp
    T_ALGORITHM_LIBS
)
add_demo_target(demo_ts_11_tools
    ad/demo_ts_11_tools/demo_ts_11_tools.cpp
    T_ALGORITHM_LIBS
)
add_demo_target(demo_ts_12_tafs
    ad/demo_ts_12_tafs/demo_ts_12_tafs.cpp
    T_ALGORITHM_LIBS
)

# Performance demos: all solvers side by side (figures via ts_plt_utils.py)
add_demo_target(demo_ts_perf_01_fdr_tpr_vs_T
    ad/demo_ts_perf_01_fdr_tpr_vs_T/demo_ts_perf_01_fdr_tpr_vs_T.cpp
    T_ALGORITHM_LIBS
)
add_demo_target(demo_ts_perf_02_runtime_vs_dummies
    ad/demo_ts_perf_02_runtime_vs_dummies/demo_ts_perf_02_runtime_vs_dummies.cpp
    T_ALGORITHM_LIBS
)
add_demo_target(demo_ts_perf_03_memory_vs_np
    ad/demo_ts_perf_03_memory_vs_np/demo_ts_perf_03_memory_vs_np.cpp
    T_ALGORITHM_LIBS
)
add_demo_target(demo_ts_perf_04_recovery_vs_dt_ratio
    ad/demo_ts_perf_04_recovery_vs_dt_ratio/demo_ts_perf_04_recovery_vs_dt_ratio.cpp
    T_ALGORITHM_LIBS
)
add_demo_target(demo_ts_perf_05_null_counts_vs_nhg
    ad/demo_ts_perf_05_null_counts_vs_nhg/demo_ts_perf_05_null_counts_vs_nhg.cpp
    T_ALGORITHM_LIBS
)

# Cross-language validation programs moved to the TRexSelector library test suite
# (TRexSelector/cpp/tests/validation/); see that folder's CMakeLists.txt.
