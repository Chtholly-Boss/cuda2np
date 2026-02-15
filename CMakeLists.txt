cmake_minimum_required(VERSION 3.20)

project(cuda2np
  VERSION 0.1.0
  DESCRIPTION "Convert CUDA operators to Numpy implementations"
  LANGUAGES CXX CUDA
)

# Export compile commands for IDE support (clangd, etc.)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find Python (prefer project venv)
if(EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python")
  set(Python_EXECUTABLE "${CMAKE_SOURCE_DIR}/.venv/bin/python" CACHE FILEPATH "Project venv python" FORCE)
endif()
find_package(Python3 COMPONENTS Interpreter REQUIRED)

find_package(CUDAToolkit REQUIRED)

# Find TVM FFI (required)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m tvm_ffi.config --cmakedir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE tvm_ffi_ROOT
  RESULT_VARIABLE tvm_ffi_RESULT
)
find_package(tvm_ffi CONFIG REQUIRED)

# Helper function to create CUDA library with TVM FFI integration
function(add_cuda2np_library target_name source_files)
    add_library(${target_name} SHARED ${source_files})
    
    set_target_properties(${target_name} PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )
    
    target_include_directories(${target_name} PUBLIC "${CMAKE_SOURCE_DIR}/csrc/include")
    target_link_libraries(${target_name} PRIVATE CUDA::cudart)
    
    tvm_ffi_configure_target(${target_name})
endfunction()

set(TRT_SOURCES
    csrc/trt/clip.cu
)

add_cuda2np_library(cuda2np-trt ${TRT_SOURCES})
