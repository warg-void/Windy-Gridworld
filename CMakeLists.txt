cmake_minimum_required(VERSION 3.10)
project(windyGridworld LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(Matplot++ QUIET)


if(NOT Matplot++_FOUND)
  message(STATUS "Finding Matplot++")
  include(FetchContent)
  FetchContent_Declare(
    matplotplusplus
    GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
    GIT_TAG origin/master
  )
  FetchContent_MakeAvailable(matplotplusplus)

  add_library(Matplot++::matplot ALIAS matplot)
endif()

add_executable(test test.cpp)

target_link_libraries(test PRIVATE Matplot++::matplot)

