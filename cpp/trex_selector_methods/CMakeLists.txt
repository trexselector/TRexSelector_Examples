# cpp/trex_selector_methods/CMakeLists.txt
# ==================================================================================
# Router: each T-Rex selector variant owns its demo executables.
# ==================================================================================

add_subdirectory(ad/trex_core)
add_subdirectory(ad/trex_screening)
add_subdirectory(ad/trex_da)
add_subdirectory(ad/trex_gvs)
add_subdirectory(ad/trex_spca)

# Cross-language validation programs moved to the TRexSelector library test suite
# (TRexSelector/cpp/tests/validation/); see that folder's CMakeLists.txt.
