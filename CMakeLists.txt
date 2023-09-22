cmake_minimum_required(VERSION 3.26.3 FATAL_ERROR)
project(condensed-sparisty)

find_package(Torch REQUIRED)
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_subdirectory(src/cc/condensed-sparsity bin)

# add_executable(condensed-sparsity ./src/cc/condensed-sparisty/ops/hello_world.cpp)
target_link_libraries(condensed-sparisty "${TORCH_LIBRARIES}")
set_property(TARGET condensed-sparisty PROPERTY CXX_STANDARD 17)

# cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
# cmake --build . --config Release
