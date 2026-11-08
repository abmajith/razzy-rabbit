cmake_minimum_required(VERSION 3.16)
project(razzy_sim_cpp CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# MuJoCo: by default reuse the headers + shared lib shipped in the Python wheel (.venv),
# override with -DMUJOCO_DIR=/path/to/mujoco-x.y.z (official release tarball) if preferred.
if(NOT MUJOCO_DIR)
  execute_process(COMMAND ${CMAKE_SOURCE_DIR}/../../.venv/bin/python -c "import mujoco,os;print(os.path.dirname(mujoco.__file__))"
                  OUTPUT_VARIABLE MUJOCO_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
file(GLOB MUJOCO_LIB "${MUJOCO_DIR}/libmujoco.so*" "${MUJOCO_DIR}/lib/libmujoco.so*")
list(GET MUJOCO_LIB 0 MUJOCO_LIB)
message(STATUS "MuJoCo: ${MUJOCO_DIR}  lib: ${MUJOCO_LIB}")

find_package(Threads REQUIRED)
add_executable(batch_rollout batch_rollout.cpp)
target_include_directories(batch_rollout PRIVATE ${MUJOCO_DIR}/include)
target_link_libraries(batch_rollout PRIVATE ${MUJOCO_LIB} Threads::Threads)
get_filename_component(MUJOCO_LIBDIR ${MUJOCO_LIB} DIRECTORY)
set_target_properties(batch_rollout PROPERTIES BUILD_RPATH ${MUJOCO_LIBDIR})
