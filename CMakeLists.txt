cmake_minimum_required(VERSION 3.24)

project(
  PimbaSim
  LANGUAGES CXX)

# cmake settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS True)

# libraries
include(cmake/CPM.cmake)

cpmaddpackage("gh:waneon/ramulator2#42b625986b0b9f8f0635608944f2e66ca58b0c66")

# targets
add_executable(
  PimbaSim
  csrc/main.cpp
  # address mappers
  csrc/addr_mappers/lax_mapper.cpp
  # controllers
  csrc/controllers/lax_controller.cpp
  # traces
  csrc/traces/lax_trace.cpp
  # DRAM specs
  csrc/dram_specs/LAX.cpp)
target_include_directories(PimbaSim PRIVATE csrc/traces csrc/dram_specs)

target_link_libraries(PimbaSim PRIVATE ramulator argparse)

