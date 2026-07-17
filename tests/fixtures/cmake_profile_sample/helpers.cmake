# A tiny helper included by CMakeLists.txt so the trace has include()/function()
# frames beyond the top-level file.
function(add_tagged_executable name)
    add_executable(${name} ${ARGN})
    target_compile_definitions(${name} PRIVATE TAGGED_BUILD=1)
endfunction()
