# Build helpers for OpenFilm.
#
# Metal shaders are not compiled by the C++ compiler. Apple's `metal` tool turns each .metal file
# into an intermediate .air file, and `metallib` links those into one OpenFilm.metallib that the
# plugin loads at runtime.

# openfilm_add_metallib(<target> <output.metallib> <shader.metal>...)
function(openfilm_add_metallib target out)
  set(airs)
  foreach(src ${ARGN})
    get_filename_component(name ${src} NAME_WE)
    set(air ${CMAKE_BINARY_DIR}/metal/${name}.air)
    add_custom_command(
      OUTPUT ${air}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/metal
      # -fno-fast-math: the kernels must track the CPU reference, and fast-math pow/log10 drift from it.
      COMMAND xcrun -sdk macosx metal -c ${src}
              -fno-fast-math
              -I ${CMAKE_SOURCE_DIR}/src
              -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}
              -o ${air}
      DEPENDS ${src} ${CMAKE_SOURCE_DIR}/src/metal/shader_types.h ${CMAKE_SOURCE_DIR}/src/metal/film_common.h
      COMMENT "Compiling Metal shader ${name}.metal"
      VERBATIM)
    list(APPEND airs ${air})
  endforeach()
  add_custom_command(
    OUTPUT ${out}
    COMMAND xcrun -sdk macosx metallib ${airs} -o ${out}
    DEPENDS ${airs}
    COMMENT "Linking ${out}"
    VERBATIM)
  add_custom_target(${target} ALL DEPENDS ${out})
endfunction()

# openfilm_add_ofx_bundle(<target> <bundle_dir> <metallib> <metallib_target> <res_stamp> <res_target>
#                         SOURCES <src>... LIBS <lib>...)
#
# An OFX plugin on macOS is a folder ("bundle") that Resolve scans in /Library/OFX/Plugins:
#   OpenFilm.ofx.bundle/Contents/Info.plist              bundle description
#   OpenFilm.ofx.bundle/Contents/MacOS/OpenFilm.ofx      the compiled plugin
#   OpenFilm.ofx.bundle/Contents/Resources/OpenFilm.metallib   GPU shaders
#   OpenFilm.ofx.bundle/Contents/Resources/data, rec2020.coeff   film stocks and spectral tables
# Apple Silicon refuses to load unsigned code, so the bundle gets an ad-hoc signature ("-").
function(openfilm_add_ofx_bundle target bundle_dir metallib metallib_target res_stamp res_target)
  cmake_parse_arguments(ARG "" "" "SOURCES;LIBS" ${ARGN})
  add_library(${target} MODULE ${ARG_SOURCES})
  target_link_libraries(${target} PRIVATE ${ARG_LIBS})
  set_target_properties(${target} PROPERTIES
    PREFIX ""
    SUFFIX ".ofx"
    LIBRARY_OUTPUT_DIRECTORY ${bundle_dir}/Contents/MacOS)
  # Only the two OFX entry points are visible to Resolve; everything else stays private.
  target_link_options(${target} PRIVATE
    "LINKER:-exported_symbol,_OfxGetNumberOfPlugins"
    "LINKER:-exported_symbol,_OfxGetPlugin")
  # The plist is generated outside the bundle and copied in just before signing, so editing it
  # (or bumping the project version) can never leave a stale signature behind.
  set(plist ${CMAKE_BINARY_DIR}/Info.plist)
  configure_file(${CMAKE_SOURCE_DIR}/src/ofx/Info.plist.in ${plist} @ONLY)

  # Re-run the post-build steps (copy + sign) whenever the shaders or the plist change.
  set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS ${metallib} ${plist} ${res_stamp})
  add_dependencies(${target} ${metallib_target} ${res_target})

  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${bundle_dir}/Contents/Resources
    COMMAND ${CMAKE_COMMAND} -E copy ${plist} ${bundle_dir}/Contents/Info.plist
    COMMAND ${CMAKE_COMMAND} -E copy ${metallib} ${bundle_dir}/Contents/Resources/
    # Film data goes in before codesign so the signature covers it.
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/resources ${bundle_dir}/Contents/Resources
    COMMAND ${CMAKE_COMMAND} -E remove -f ${bundle_dir}/Contents/Resources/.stamp
    COMMAND codesign --force --deep -s - ${bundle_dir}
    COMMENT "Assembling and signing ${bundle_dir}"
    VERBATIM)
endfunction()
