cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(RWKVCPP)

find_package(CUDAToolkit)
# set cmake cuda architectures
set(CMAKE_CUDA_ARCHITECTURES 61 70 75 80)

# find and add python3
find_package(Python3 COMPONENTS Interpreter Development)

# link python 
include_directories(${Python3_INCLUDE_DIRS})
link_libraries(${Python3_LIBRARIES})

# add libs in ./libtorch/lib
link_directories(libtorch/lib)


find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_executable(RWKVCPP Main.cpp operators.o operators32.o)
target_link_libraries(RWKVCPP "${TORCH_LIBRARIES}")
set_property(TARGET RWKVCPP PROPERTY CXX_STANDARD 14)

# make executable check relative ./lib for DLLs
set_target_properties(RWKVCPP PROPERTIES
                      INSTALL_RPATH "$ORIGIN/lib")


# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET RWKVCPP
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:RWKVCPP>)
endif (MSVC)