cmake_minimum_required(VERSION 3.24)
project(OpenFilm VERSION 0.1.0 LANGUAGES C CXX OBJCXX)

# --- Global settings -------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OSX_DEPLOYMENT_TARGET 12.0 CACHE STRING "Minimum macOS version")
set(CMAKE_OSX_ARCHITECTURES arm64 CACHE STRING "Target architecture")
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
