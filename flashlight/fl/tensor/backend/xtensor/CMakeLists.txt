cmake_minimum_required(VERSION 3.10)

find_package(xtensor CONFIG)
if(NOT xtensor_FOUND)
  include(${CMAKE_MODULE_PATH}/BuildXtensor.cmake)
  find_package(xtensor CONFIG REQUIRED)
endif()

target_link_libraries(
  flashlight
  PRIVATE
  xtensor
)

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/XtensorBackend.cpp
  ${CMAKE_CURRENT_LIST_DIR}/XtensorTensor.cpp
)
