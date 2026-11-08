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
      COMMAND xcrun -sdk macosx metal -c ${src}
              -I ${CMAKE_SOURCE_DIR}/src
              -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}
              -o ${air}
      DEPENDS ${src} ${CMAKE_SOURCE_DIR}/src/metal/shader_types.h
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
