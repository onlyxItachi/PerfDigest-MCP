cmake_minimum_required(VERSION 3.18)
project(perfdigest_cmake_fixture C)

# Interesting frames for the profile: a find_package probe, an include()d
# helper file, a function() call, and two targets.
find_package(Threads REQUIRED)
include(helpers.cmake)

add_library(fixture_util STATIC util.c)
add_tagged_executable(fixture_app main.c)
target_link_libraries(fixture_app PRIVATE fixture_util Threads::Threads)

foreach(i RANGE 1 3)
    message(STATUS "configuring pass marker ${i}")
endforeach()
