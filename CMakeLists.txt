cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(export_eigenvalue LANGUAGES CXX CUDA)

set(SOURCES
    chem_utils.cu
    dydt.cu
    gpu_memory.cu
    jacob.cu
    mass_mole.cu
    mechanism.cu
    rxn_rates.cu
    rxn_rates_pres_mod.cu
    sparse_multiplier.cu
    spec_rates.cu
    jacobs/jacob_0.cu
    jacobs/jacob_1.cu
    jacobs/jacob_2.cu
    jacobs/jacob_3.cu
    jacobs/jacob_4.cu
    jacobs/jacob_5.cu
    jacobs/jacob_6.cu
    export_eigenvalue.cu
)

# add_executable(export_jacobian export_jacobian.cu)
add_executable(export_eigenvalue ${SOURCES})

# --- ヘッダーファイル用の include パス追加 ---
target_include_directories(export_eigenvalue PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/jacobs
    /home/syoukera/github/cuda-samples/Common
)

# --- VTKと数学ライブラリをリンク ---
target_link_libraries(export_eigenvalue PRIVATE
    lapack blas m
)

# C++17 & CUDA標準をそろえる
set_target_properties(export_eigenvalue PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17
    CUDA_STANDARD 17
)

set(CMAKE_CUDA_ARCHITECTURES 89)

target_compile_options(export_eigenvalue PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O2;-lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-fPIC;-Wall;-Wextra>
)
