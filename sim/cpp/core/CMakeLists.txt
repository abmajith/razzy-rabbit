cmake_minimum_required(VERSION 3.18)
project(razzy_core LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# portable by default (CI, other machines); -DRAZZY_NATIVE=ON for -march=native
option(RAZZY_NATIVE "optimise for this CPU" OFF)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
if(RAZZY_NATIVE)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
endif()

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# MuJoCo headers + shared library from the mujoco wheel (same version as at runtime)
execute_process(COMMAND "${Python_EXECUTABLE}" -c "import mujoco,os;print(os.path.dirname(mujoco.__file__))"
                OUTPUT_VARIABLE MUJOCO_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
file(GLOB MUJOCO_LIB "${MUJOCO_DIR}/libmujoco.so*")
list(GET MUJOCO_LIB 0 MUJOCO_LIB)
message(STATUS "MuJoCo: ${MUJOCO_DIR}  lib: ${MUJOCO_LIB}")

find_package(Threads REQUIRED)
file(GLOB CORE_SRC CONFIGURE_DEPENDS csrc/*.cpp)
nanobind_add_module(_core NB_STATIC ${CORE_SRC})
target_include_directories(_core PRIVATE ${MUJOCO_DIR}/include csrc)
target_link_libraries(_core PRIVATE ${MUJOCO_LIB} Threads::Threads)
# at runtime libmujoco comes from the installed mujoco package next to us in site-packages
set_target_properties(_core PROPERTIES INSTALL_RPATH "$ORIGIN/../mujoco" BUILD_WITH_INSTALL_RPATH ON)
install(TARGETS _core LIBRARY DESTINATION razzy_core)
