# CMakeLists.txt for memory_mapping demos
# ==================================================================================
# CMake configuration for memory-mapping utility demos.

add_demo_target(demo_memory_mapping
    demo_memory_mapping/demo_memory_mapping.cpp
    DATA_PREPROCESSING_LIBS
)
