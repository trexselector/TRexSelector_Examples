# CMakeLists.txt for ml_methods demos

add_demo_target(demo_mlm_scaler_01
    scaler_methods/demo_mlm_scaler_01/demo_mlm_scaler_01.cpp
    DATA_PREPROCESSING_LIBS
)
target_compile_definitions(demo_mlm_scaler_01 PRIVATE
    "TREX_DEMO_MMAP_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/scaler_methods/demo_mlm_scaler_01/trex_mlm_demo_scaler01_mmap.bin\""
    "TREX_DEMO_SCALER_SER_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/scaler_methods/demo_mlm_scaler_01/trex_mlm_demo_scaler01_ser.bin\""
)

add_demo_target(demo_mlm_hac_01
    hac_clustering/demo_mlm_hac_01/demo_mlm_hac_01.cpp
    DATA_PREPROCESSING_LIBS
)

add_demo_target(demo_mlm_hac_02_mmap
    hac_clustering/demo_mlm_hac_02_mmap/demo_mlm_hac_02_mmap.cpp
    DATA_PREPROCESSING_LIBS
)

target_compile_definitions(demo_mlm_hac_02_mmap PRIVATE
    "TREX_DEMO_MMAP_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/hac_clustering/demo_mlm_hac_02_mmap/trex_mlm_demo_02_hac_mmap.bin\""
)

add_demo_target(demo_mlm_ms_01_ridge_cv_svd
    model_selection/demo_mlm_ms_01_ridge_cv_svd/demo_mlm_ms_01_ridge_cv_svd.cpp
    DATA_PREPROCESSING_LIBS
)
target_compile_definitions(demo_mlm_ms_01_ridge_cv_svd PRIVATE
    "TREX_DEMO_MMAP_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/model_selection/demo_mlm_ms_01_ridge_cv_svd/trex_mlm_demo_ms01_svd_mmap.bin\""
)

add_demo_target(demo_mlm_ms_02_enet_cv_ccd
    model_selection/demo_mlm_ms_02_enet_cv_ccd/demo_mlm_ms_02_enet_cv_ccd.cpp
    DATA_PREPROCESSING_LIBS
)

add_demo_target(demo_mlm_pca_01
    pca/demo_mlm_pca_01/demo_mlm_pca_01.cpp
    DATA_PREPROCESSING_LIBS
)

add_demo_target(demo_mlm_ridge_01
    ridge_regression/demo_mlm_ridge_01/demo_mlm_ridge_01.cpp
    DATA_PREPROCESSING_LIBS
)

add_demo_target(demo_mlm_svd_01
    svd/demo_mlm_svd_01/demo_mlm_svd_01.cpp
    DATA_PREPROCESSING_LIBS
)
