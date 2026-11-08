cmake_minimum_required(VERSION 3.24)
# These two must be set before project() or CMake ignores them.
set(CMAKE_OSX_DEPLOYMENT_TARGET 12.0 CACHE STRING "Minimum macOS version")
set(CMAKE_OSX_ARCHITECTURES arm64 CACHE STRING "Target architecture")
project(OpenFilm VERSION 0.1.0 LANGUAGES C CXX OBJCXX)

# --- Global settings -------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

enable_testing()

# --- Test framework (doctest), downloaded at configure time ----------------
include(FetchContent)
FetchContent_Declare(doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG v2.4.11)
# doctest 2.4.11 declares an old cmake_minimum_required that CMake 4 rejects.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_MakeAvailable(doctest)
unset(CMAKE_POLICY_VERSION_MINIMUM)

# --- Core: plain C++ film maths, no OFX or Apple dependencies ---------------
add_library(openfilm_core STATIC src/core/cpu_renderer.cpp)
target_include_directories(openfilm_core PUBLIC ${CMAKE_SOURCE_DIR}/src)

add_executable(test_core tests/core/test_cpu_renderer.cpp)
target_link_libraries(test_core PRIVATE openfilm_core doctest::doctest_with_main)
add_test(NAME core COMMAND test_core)

# --- Metal: GPU kernels + the host code that launches them ------------------
include(cmake/OpenFilmBundle.cmake)
set(OPENFILM_METALLIB ${CMAKE_BINARY_DIR}/OpenFilm.metallib)
openfilm_add_metallib(openfilm_metallib ${OPENFILM_METALLIB}
  ${CMAKE_SOURCE_DIR}/src/metal/exposure.metal)

add_library(openfilm_metal STATIC src/metal/metal_renderer.mm)
target_link_libraries(openfilm_metal PUBLIC openfilm_core "-framework Metal" "-framework Foundation")
target_compile_options(openfilm_metal PRIVATE -fobjc-arc)

add_executable(test_gpu tests/gpu/test_gpu_vs_cpu.mm)
target_link_libraries(test_gpu PRIVATE openfilm_metal doctest::doctest)
target_compile_options(test_gpu PRIVATE -fobjc-arc)
target_compile_definitions(test_gpu PRIVATE OPENFILM_METALLIB_PATH="${OPENFILM_METALLIB}")
add_dependencies(test_gpu openfilm_metallib)
add_test(NAME gpu COMMAND test_gpu)
