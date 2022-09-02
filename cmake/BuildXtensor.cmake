cmake_minimum_required(VERSION 3.14)

include(FetchContent)

FetchContent_Declare(
  xtl
  GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
  GIT_TAG        0.7.4
)

FetchContent_Declare(
  xsimd
  GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
  GIT_TAG        9.0.1
)

FetchContent_Declare(
  xtensor
  GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
  GIT_TAG        0.24.3
)

FetchContent_MakeAvailable(xtl)
FetchContent_MakeAvailable(xsimd)
FetchContent_MakeAvailable(xtensor)
set(xtensor_DIR ${xtensor_BINARY_DIR})
set(xtl_DIR ${xtl_BINARY_DIR})
set(xsimd_DIR ${xsimd_BINARY_DIR})
