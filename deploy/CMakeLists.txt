cmake_minimum_required(VERSION 3.10)
project(ncnn_deploy)

set(CMAKE_CXX_STANDARD 17)

# NCNN path - set via cmake -DNCNN_ROOT=/path/to/ncnn
set(NCNN_ROOT "" CACHE PATH "NCNN installation root")
if(NCNN_ROOT)
    include_directories(${NCNN_ROOT}/include)
    link_directories(${NCNN_ROOT}/lib)
endif()

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# NCNN deploy executable
add_executable(ncnn_deploy ncnn_deploy.cpp)

target_link_libraries(ncnn_deploy
    ${OpenCV_LIBS}
    ncnn
)

# Install
install(TARGETS ncnn_deploy DESTINATION bin)
