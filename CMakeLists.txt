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

# nlohmann/json: reads the film-stock files.
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(nlohmann_json)

# --- Jakob-Hanika table: Rec.2020 RGB -> spectrum coefficients, built once ----
# The optimiser takes a few seconds, so it only reruns when the tool itself changes.
add_executable(rgb2spec_opt tools/rgb2spec/rgb2spec_opt.cpp)
target_compile_options(rgb2spec_opt PRIVATE -O3 -w)  # vendored code: speed matters, warnings don't
set(OPENFILM_COEFF ${CMAKE_BINARY_DIR}/resources/rec2020.coeff)
add_custom_command(
  OUTPUT ${OPENFILM_COEFF}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/resources
  COMMAND rgb2spec_opt 64 ${OPENFILM_COEFF} REC2020
  DEPENDS rgb2spec_opt
  COMMENT "Generating Rec.2020 RGB-to-spectrum table"
  VERBATIM)
add_custom_target(openfilm_coeff DEPENDS ${OPENFILM_COEFF})

# --- Resources: everything the film model reads at run time, staged for the bundle -------------
# Layout matches FilmResources::load(): <resources>/data/spectral, <resources>/data/stocks, rec2020.coeff.
file(GLOB OPENFILM_SPECTRAL_CSV CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/data/spectral/*.csv)
file(GLOB OPENFILM_STOCK_JSON CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/data/stocks/*.json)
set(OPENFILM_RESOURCES_STAMP ${CMAKE_BINARY_DIR}/resources/.stamp)
add_custom_command(
  OUTPUT ${OPENFILM_RESOURCES_STAMP}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/resources/data/spectral ${CMAKE_BINARY_DIR}/resources/data/stocks
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENFILM_SPECTRAL_CSV} ${CMAKE_BINARY_DIR}/resources/data/spectral/
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENFILM_STOCK_JSON} ${CMAKE_BINARY_DIR}/resources/data/stocks/
  COMMAND ${CMAKE_COMMAND} -E touch ${OPENFILM_RESOURCES_STAMP}
  DEPENDS ${OPENFILM_SPECTRAL_CSV} ${OPENFILM_STOCK_JSON} ${OPENFILM_COEFF}
  COMMENT "Staging film resources"
  VERBATIM)
add_custom_target(openfilm_resources DEPENDS ${OPENFILM_RESOURCES_STAMP})
add_dependencies(openfilm_resources openfilm_coeff)  # the stamp rule reads the coeff file, so build it first

# --- Core: plain C++ film maths, no OFX or Apple dependencies ---------------
add_library(openfilm_core STATIC src/core/spectral.cpp src/core/color.cpp src/core/gamut.cpp src/core/rgb2spec.cpp src/core/stock.cpp src/core/densitometry.cpp src/core/negative.cpp src/core/print.cpp src/core/film_model.cpp src/core/lut.cpp src/core/optics.cpp src/core/film_renderer.cpp src/core/blur.cpp src/core/film_spatial.cpp)
target_include_directories(openfilm_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(openfilm_core PUBLIC nlohmann_json::nlohmann_json)

add_executable(test_core tests/core/test_spectral.cpp tests/core/test_color.cpp tests/core/test_gamut.cpp tests/core/test_rgb2spec.cpp tests/core/test_stock.cpp tests/core/test_densitometry.cpp tests/core/test_negative.cpp tests/core/test_print.cpp tests/core/test_film_model.cpp tests/core/test_lut.cpp tests/core/test_optics.cpp tests/core/test_film_renderer.cpp tests/core/test_blur.cpp tests/core/test_film_spatial.cpp)
target_compile_definitions(test_core PRIVATE OPENFILM_SOURCE_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
  OPENFILM_COEFF_PATH="${OPENFILM_COEFF}")
add_dependencies(test_core openfilm_coeff)
target_link_libraries(test_core PRIVATE openfilm_core doctest::doctest_with_main)
add_test(NAME core COMMAND test_core)

# --- Metal: GPU kernels + the host code that launches them ------------------
include(cmake/OpenFilmBundle.cmake)
set(OPENFILM_METALLIB ${CMAKE_BINARY_DIR}/OpenFilm.metallib)
openfilm_add_metallib(openfilm_metallib ${OPENFILM_METALLIB}
  ${CMAKE_SOURCE_DIR}/src/metal/film.metal)

add_library(openfilm_metal STATIC src/metal/metal_renderer.mm)
target_link_libraries(openfilm_metal PUBLIC openfilm_core "-framework Metal" "-framework Foundation")
target_compile_options(openfilm_metal PRIVATE -fobjc-arc)

add_executable(test_gpu tests/gpu/test_gpu_vs_cpu.mm)
target_link_libraries(test_gpu PRIVATE openfilm_metal doctest::doctest)
target_compile_options(test_gpu PRIVATE -fobjc-arc)
target_compile_definitions(test_gpu PRIVATE OPENFILM_METALLIB_PATH="${OPENFILM_METALLIB}"
  OPENFILM_SOURCE_DATA_DIR="${CMAKE_SOURCE_DIR}/data" OPENFILM_COEFF_PATH="${OPENFILM_COEFF}")
add_dependencies(test_gpu openfilm_metallib openfilm_coeff)
add_test(NAME gpu COMMAND test_gpu)

# --- OpenFX SDK: C headers + the official C++ "Support" wrapper library ------
set(OFX_DIR ${CMAKE_SOURCE_DIR}/external/openfx)
file(GLOB OFX_SUPPORT_SRC ${OFX_DIR}/Support/Library/*.cpp)
add_library(ofx_support STATIC ${OFX_SUPPORT_SRC})
target_include_directories(ofx_support
  PUBLIC ${OFX_DIR}/include ${OFX_DIR}/Support/include
  PRIVATE ${OFX_DIR}/Support/Library)
target_compile_options(ofx_support PRIVATE -w)  # third-party code: silence its warnings

# --- The plugin: build/OpenFilm.ofx.bundle ------------------------------------
set(OPENFILM_BUNDLE_DIR ${CMAKE_BINARY_DIR}/OpenFilm.ofx.bundle)
openfilm_add_ofx_bundle(OpenFilm ${OPENFILM_BUNDLE_DIR} ${OPENFILM_METALLIB} openfilm_metallib
  ${OPENFILM_RESOURCES_STAMP} openfilm_resources
  SOURCES src/ofx/openfilm_plugin.cpp
  LIBS ofx_support openfilm_metal)

# --- Plugin bundle tests ------------------------------------------------------
add_executable(test_bundle tests/ofx/test_bundle_loads.cpp)
target_link_libraries(test_bundle PRIVATE openfilm_metal doctest::doctest)
target_include_directories(test_bundle PRIVATE ${OFX_DIR}/include)
target_compile_definitions(test_bundle PRIVATE OPENFILM_BUNDLE_DIR="${OPENFILM_BUNDLE_DIR}")
add_dependencies(test_bundle OpenFilm)
add_test(NAME bundle COMMAND test_bundle)
# Resolve may refuse a bundle whose signature no longer matches its contents.
add_test(NAME bundle_signature COMMAND codesign --verify --deep --strict ${OPENFILM_BUNDLE_DIR})
